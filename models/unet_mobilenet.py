"""Multi-task MobileNetV3-U-Net (GroupNorm everywhere).

Heads:
  1. binary segmentation logits
  2. evidential head -> Dirichlet parameters (single forward pass; NOT MC-Dropout)
  3. terrain classifier branching off the mid-level (stride-8) encoder features
Plus two auxiliary deep-supervision outputs at intermediate decoder stages.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

logger = logging.getLogger(__name__)

# tap = index into ``features`` of the last block at each stride (2,4,8,16,32).
ENCODER_SPECS: Dict[str, Dict[str, Any]] = {
    "mobilenet_v3_large": {"taps": [1, 3, 6, 12, 16], "channels": [16, 24, 40, 112, 960],
                           "weights": "MobileNet_V3_Large_Weights"},
    "mobilenet_v3_small": {"taps": [0, 1, 3, 8, 12], "channels": [16, 16, 24, 48, 576],
                           "weights": "MobileNet_V3_Small_Weights"},
}


def gn_groups(channels: int, max_groups: int = 8) -> int:
    """Largest group count <= ``max_groups`` that divides ``channels``.

    Args:
        channels: Number of channels.
        max_groups: Upper bound.

    Returns:
        Valid GroupNorm group count.
    """
    g = min(max_groups, channels)
    while channels % g:
        g -= 1
    return g


def convert_bn_to_gn(module: nn.Module, max_groups: int = 8) -> nn.Module:
    """Recursively replace every ``BatchNorm2d`` with ``GroupNorm`` (in place).

    Args:
        module: Module to convert.
        max_groups: Max GroupNorm groups.

    Returns:
        The same module (converted).
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(gn_groups(child.num_features, max_groups), child.num_features, eps=1e-5))
        else:
            convert_bn_to_gn(child, max_groups)
    return module


class ConvGNAct(nn.Sequential):
    """Conv2d -> GroupNorm -> ReLU."""

    def __init__(self, cin: int, cout: int, k: int = 3, max_groups: int = 8) -> None:
        """Create the block.

        Args:
            cin: Input channels.
            cout: Output channels.
            k: Kernel size.
            max_groups: Max GroupNorm groups.
        """
        super().__init__(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
                         nn.GroupNorm(gn_groups(cout, max_groups), cout), nn.ReLU(inplace=True))


class DecoderBlock(nn.Module):
    """Upsample -> concat skip -> two ConvGNAct (``conv1``, ``conv2``)."""

    def __init__(self, cin: int, cskip: int, cout: int, max_groups: int = 8) -> None:
        """Create the block.

        Args:
            cin: Channels of the (lower-resolution) input.
            cskip: Channels of the skip connection.
            cout: Output channels.
            max_groups: Max GroupNorm groups.
        """
        super().__init__()
        self.conv1 = ConvGNAct(cin + cskip, cout, 3, max_groups)
        self.conv2 = ConvGNAct(cout, cout, 3, max_groups)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: Low-res features.
            skip: Encoder skip features.

        Returns:
            Decoded features at the skip's resolution.
        """
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv2(self.conv1(torch.cat([x, skip], dim=1)))


class MobileNetV3Encoder(nn.Module):
    """torchvision MobileNetV3 trunk returning features at strides 2/4/8/16/32."""

    def __init__(self, name: str, pretrained: bool, max_groups: int = 8) -> None:
        """Create the encoder.

        Args:
            name: ``mobilenet_v3_large`` or ``mobilenet_v3_small``.
            pretrained: Try to load ImageNet weights (falls back to random init on failure).
            max_groups: Max GroupNorm groups used when replacing BatchNorm.
        """
        super().__init__()
        spec = ENCODER_SPECS[name]
        weights = None
        if pretrained:
            try:
                weights = getattr(torchvision.models, spec["weights"]).IMAGENET1K_V1
            except Exception as e:  # pragma: no cover
                logger.warning("No pretrained weights enum for %s: %s", name, e)
        try:
            net = getattr(torchvision.models, name)(weights=weights)
        except Exception as e:
            logger.warning("Could not load ImageNet weights for %s (%s). Using RANDOM init.", name, e)
            net = getattr(torchvision.models, name)(weights=None)
        # BatchNorm -> GroupNorm (conv weights keep their pretrained values; norm stats are re-learned).
        self.features = convert_bn_to_gn(net.features, max_groups)
        self.taps: List[int] = spec["taps"]
        self.channels: List[int] = spec["channels"]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward.

        Args:
            x: ``(B,3,H,W)`` normalised image.

        Returns:
            Features ``[s2, s4, s8, s16, s32]``.
        """
        outs: List[torch.Tensor] = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.taps:
                outs.append(x)
        return outs


class SegmentationDecoder(nn.Module):
    """Separate decoder branch for multi-task segmentation (used for night decoder)."""

    def __init__(self, c_enc: Sequence[int], decoder_channels: Sequence[int] = (128, 64, 40, 24),
                 gn_max_groups: int = 8, seg_classes: int = 2, evidence_clip: float = 1000.0) -> None:
        super().__init__()
        c2, c4, c8, c16, c32 = c_enc
        d16, d8, d4, d2 = decoder_channels
        self.evidence_clip = evidence_clip
        self.dec16 = DecoderBlock(c32, c16, d16, gn_max_groups)
        self.dec8 = DecoderBlock(d16, c8, d8, gn_max_groups)
        self.dec4 = DecoderBlock(d8, c4, d4, gn_max_groups)
        self.dec2 = DecoderBlock(d4, c2, d2, gn_max_groups)
        self.final = ConvGNAct(d2, d2, 3, gn_max_groups)
        self.seg_head = nn.Conv2d(d2, seg_classes, 1)
        self.evid_conv = ConvGNAct(d2, d2, 3, gn_max_groups)
        self.evid_head = nn.Conv2d(d2, seg_classes, 1)
        self.aux_heads = nn.ModuleList([nn.Conv2d(d8, seg_classes, 1), nn.Conv2d(d4, seg_classes, 1)])

    def forward(self, features: Sequence[torch.Tensor], size: Sequence[int]) -> Dict[str, Any]:
        f2, f4, f8, f16, f32 = features
        d16 = self.dec16(f32, f16)
        d8 = self.dec8(d16, f8)
        d4 = self.dec4(d8, f4)
        d2 = self.dec2(d4, f2)
        feat = self.final(F.interpolate(d2, size=size, mode="bilinear", align_corners=False))
        seg = self.seg_head(feat)
        evidence = F.softplus(self.evid_head(self.evid_conv(feat))).clamp(max=self.evidence_clip)
        return {
            "seg_logits": seg,
            "alpha": evidence + 1.0,
            "aux_logits": [self.aux_heads[0](d8), self.aux_heads[1](d4)],
        }


class MultiTaskUNet(nn.Module):
    """Multi-task U-Net with MobileNetV3 encoder.

    ``forward`` returns a dict:
      * ``seg_logits``  (B, seg_classes, H, W)
      * ``alpha``       (B, seg_classes, H, W) Dirichlet parameters (>= 1)
      * ``terrain_logits`` (B, terrain_classes)
      * ``aux_logits``  list of 2 tensors at stride 8 and stride 4
    """

    def __init__(self, encoder: str = "mobilenet_v3_large", pretrained: bool = True,
                 decoder_channels: Sequence[int] = (128, 64, 40, 24), gn_max_groups: int = 8,
                 seg_classes: int = 2, terrain_classes: int = 4, terrain_hidden: int = 64,
                 terrain_dropout: float = 0.2, evidence_clip: float = 1000.0) -> None:
        """Build the network.

        Args:
            encoder: Encoder name.
            pretrained: Load ImageNet weights if possible.
            decoder_channels: Output channels of the four decoder blocks (s16, s8, s4, s2).
            gn_max_groups: Max GroupNorm groups.
            seg_classes: Number of segmentation / Dirichlet classes.
            terrain_classes: Number of terrain classes.
            terrain_hidden: Hidden channels of the terrain head.
            terrain_dropout: Dropout of the terrain head.
            evidence_clip: Upper clamp for the evidence (numerical safety).
        """
        super().__init__()
        self.seg_classes, self.evidence_clip = seg_classes, evidence_clip
        self.decoder_channels = list(decoder_channels)
        self.gn_max_groups = gn_max_groups
        self.encoder = MobileNetV3Encoder(encoder, pretrained, gn_max_groups)
        c2, c4, c8, c16, c32 = self.encoder.channels
        d16, d8, d4, d2 = decoder_channels
        self.dec16 = DecoderBlock(c32, c16, d16, gn_max_groups)
        self.dec8 = DecoderBlock(d16, c8, d8, gn_max_groups)
        self.dec4 = DecoderBlock(d8, c4, d4, gn_max_groups)
        self.dec2 = DecoderBlock(d4, c2, d2, gn_max_groups)
        self.final = ConvGNAct(d2, d2, 3, gn_max_groups)
        self.seg_head = nn.Conv2d(d2, seg_classes, 1)
        self.evid_conv = ConvGNAct(d2, d2, 3, gn_max_groups)
        self.evid_head = nn.Conv2d(d2, seg_classes, 1)
        self.aux_heads = nn.ModuleList([nn.Conv2d(d8, seg_classes, 1), nn.Conv2d(d4, seg_classes, 1)])
        self.terrain_conv = ConvGNAct(c8, terrain_hidden, 3, gn_max_groups)  # mid-level (stride 8) branch
        self.terrain_drop = nn.Dropout(terrain_dropout)
        self.terrain_fc = nn.Linear(terrain_hidden, terrain_classes)
        self.night_decoder: Optional[SegmentationDecoder] = None
        self.mode: str = "day"

    def set_mode(self, mode: str) -> None:
        """Set active decoder mode ('day' or 'night')."""
        if mode not in ("day", "night"):
            raise ValueError(f"Unknown mode '{mode}', expected 'day' or 'night'")
        self.mode = mode

    def add_night_decoder(self) -> SegmentationDecoder:
        """Add a separate night decoder branch, warm-started from day decoder weights."""
        if self.night_decoder is None:
            self.night_decoder = SegmentationDecoder(
                self.encoder.channels, self.decoder_channels, self.gn_max_groups,
                self.seg_classes, self.evidence_clip
            )
            # Warm start from existing day decoder weights
            self.night_decoder.dec16.load_state_dict(self.dec16.state_dict())
            self.night_decoder.dec8.load_state_dict(self.dec8.state_dict())
            self.night_decoder.dec4.load_state_dict(self.dec4.state_dict())
            self.night_decoder.dec2.load_state_dict(self.dec2.state_dict())
            self.night_decoder.final.load_state_dict(self.final.state_dict())
            self.night_decoder.seg_head.load_state_dict(self.seg_head.state_dict())
            self.night_decoder.evid_conv.load_state_dict(self.evid_conv.state_dict())
            self.night_decoder.evid_head.load_state_dict(self.evid_head.state_dict())
            self.night_decoder.aux_heads.load_state_dict(self.aux_heads.state_dict())
        return self.night_decoder

    def forward(self, x: torch.Tensor, mode: Optional[str] = None) -> Dict[str, Any]:
        """Forward pass.

        Args:
            x: ``(B,3,H,W)`` normalised images.
            mode: Optional override ('day' or 'night'). Defaults to self.mode.

        Returns:
            Output dict (see class docstring).
        """
        size = x.shape[-2:]
        f2, f4, f8, f16, f32 = self.encoder(x)
        
        target_mode = (mode or self.mode).lower()
        if target_mode == "night" and self.night_decoder is not None:
            dec_out = self.night_decoder([f2, f4, f8, f16, f32], size)
            seg = dec_out["seg_logits"]
            alpha = dec_out["alpha"]
            aux = dec_out["aux_logits"]
        else:
            d16 = self.dec16(f32, f16)
            d8 = self.dec8(d16, f8)
            d4 = self.dec4(d8, f4)
            d2 = self.dec2(d4, f2)
            feat = self.final(F.interpolate(d2, size=size, mode="bilinear", align_corners=False))
            seg = self.seg_head(feat)
            evidence = F.softplus(self.evid_head(self.evid_conv(feat))).clamp(max=self.evidence_clip)
            alpha = evidence + 1.0
            aux = [self.aux_heads[0](d8), self.aux_heads[1](d4)]

        t = self.terrain_conv(f8).mean(dim=(2, 3))
        return {
            "seg_logits": seg,
            "alpha": alpha,
            "terrain_logits": self.terrain_fc(self.terrain_drop(t)),
            "aux_logits": aux,
        }


def build_model(model_cfg: Dict[str, Any], evidence_clip: float = 1000.0) -> MultiTaskUNet:
    """Build a model from a ``model``/``student`` config section.

    Args:
        model_cfg: Config section.
        evidence_clip: Evidence clamp.

    Returns:
        :class:`MultiTaskUNet`.
    """
    return MultiTaskUNet(model_cfg["encoder"], model_cfg.get("pretrained", False), model_cfg["decoder_channels"],
                         model_cfg.get("gn_max_groups", 8), model_cfg["seg_classes"], model_cfg["terrain_classes"],
                         model_cfg.get("terrain_hidden", 64), model_cfg.get("terrain_dropout", 0.2), evidence_clip)


def evidential_uncertainty(alpha: torch.Tensor) -> torch.Tensor:
    """Per-pixel vacuity ``u = K / S`` (S = sum of Dirichlet params); in (0, 1].

    Args:
        alpha: ``(B,K,H,W)`` Dirichlet parameters.

    Returns:
        ``(B,H,W)`` uncertainty.
    """
    return alpha.shape[1] / alpha.sum(dim=1)


def evidential_probs(alpha: torch.Tensor) -> torch.Tensor:
    """Expected class probabilities of the Dirichlet, ``alpha / S``.

    Args:
        alpha: ``(B,K,H,W)``.

    Returns:
        ``(B,K,H,W)`` probabilities.
    """
    return alpha / alpha.sum(dim=1, keepdim=True)


def count_params(model: nn.Module) -> int:
    """Number of parameters.

    Args:
        model: Module.

    Returns:
        Parameter count.
    """
    return sum(p.numel() for p in model.parameters())
