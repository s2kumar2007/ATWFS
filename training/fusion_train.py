"""Train FusionNet MLP and produce an ablation comparison table against the heuristic.

Data collection:
- Reads --log-features CSVs if provided.
- Evaluates DL mask and classical fallback on IDD val ground-truth masks.
- Labels target_alpha: 1.0 if IoU(DL, GT) >= IoU(Classical, GT) else 0.0.
- Trains FusionNet MLP (U, G, T, terrain=0 -> alpha).
- Saves checkpoint to outputs/checkpoints/fusionnet.pt.
- Compares heuristic vs trained FusionNet and writes outputs/reports/fusion_ablation.csv.
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from data.dataset import DrivableDataset, IDDSource, SampleSource
from data.synthetic import SyntheticSource
from inference.classical import ClassicalFreeSpaceEstimator
from inference.consistency import geometric_score, temporal_score
from inference.fusion import ATWFS, FusionNet
from models.unet_mobilenet import build_model, evidential_uncertainty
from training.train import load_model_from_checkpoint
from utils.config import load_config, resolve_path
from utils.logging_utils import setup_logging
from utils.seed import get_device, set_global_seed

logger = logging.getLogger(__name__)


def compute_mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute binary IoU between two (H, W) uint8 masks."""
    p_bool = (pred > 0)
    g_bool = (gt > 0)
    inter = np.logical_and(p_bool, g_bool).sum()
    union = np.logical_or(p_bool, g_bool).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter / union)


def collect_val_features(
    cfg: Dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    max_samples: int = 150,
    smoke: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    """Extract (U, G, T, terrain=0) features and ground-truth comparison targets on val split."""
    if smoke:
        source: SampleSource = SyntheticSource(num_samples=min(16, max_samples), image_size=tuple(cfg["data"]["image_size"]))
    else:
        try:
            source = IDDSource(cfg, split="val", require_masks=True)
        except Exception as e:
            logger.warning("Could not load IDD val split (%s); using synthetic source.", e)
            source = SyntheticSource(num_samples=min(32, max_samples), image_size=tuple(cfg["data"]["image_size"]))

    classical = ClassicalFreeSpaceEstimator(cfg["classical"])
    model.eval()

    n = min(len(source), max_samples)
    feats_list = []
    targets_list = []
    sample_records = []
    prev_dl_mask = None

    mean = np.asarray(cfg["data"]["imagenet_mean"], np.float32)
    std = np.asarray(cfg["data"]["imagenet_std"], np.float32)

    logger.info("Collecting fusion training features on %d frames...", n)
    with torch.no_grad():
        for i in range(n):
            img_rgb, raw_mask, _ = source.load(i)
            # Binary ground truth
            gt_binary = np.isin(raw_mask, cfg["data"]["drivable_level3_ids"]).astype(np.uint8)

            # Model prediction
            h, w = img_rgb.shape[:2]
            target_h, target_w = cfg["data"]["image_size"]
            resized_rgb = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
            x = (resized_rgb.astype(np.float32) / 255.0 - mean) / std
            x_t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)

            out = model(x_t)
            dl_mask_small = out["seg_logits"].argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            dl_mask = cv2.resize(dl_mask_small, (w, h), interpolation=cv2.INTER_NEAREST)

            unc = float(evidential_uncertainty(out["alpha"]).mean().cpu().item())
            geo = geometric_score(dl_mask_small, cfg["consistency"]["geometric"])
            tmp = temporal_score(prev_dl_mask, dl_mask_small, cfg["inference"].get("speed_mps_input", 1.0), cfg["consistency"]["temporal"])
            prev_dl_mask = dl_mask_small.copy()

            # Classical fallback
            fb_mask = classical.estimate(img_rgb)
            if fb_mask.shape != (h, w):
                fb_mask = cv2.resize(fb_mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # IoU against GT
            iou_dl = compute_mask_iou(dl_mask, gt_binary)
            iou_fb = compute_mask_iou(fb_mask, gt_binary)

            # Target alpha: 1.0 if DL >= FB else 0.0
            target_alpha = 1.0 if iou_dl >= iou_fb else 0.0

            feats_list.append([unc, geo, tmp])
            targets_list.append(target_alpha)
            sample_records.append({
                "unc": unc, "geo": geo, "tmp": tmp,
                "iou_dl": iou_dl, "iou_fb": iou_fb,
                "dl_mask": dl_mask, "fb_mask": fb_mask, "gt": gt_binary,
            })

    feats_t = torch.tensor(feats_list, dtype=torch.float32).clamp(0.0, 1.0)
    terr_t = torch.zeros(len(feats_list), dtype=torch.long)  # terrain = 0 everywhere
    targets_t = torch.tensor(targets_list, dtype=torch.float32)

    return feats_t, terr_t, targets_t, sample_records


def train_fusionnet(
    cfg: Dict[str, Any],
    log_features_csv: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    output_path: str = "outputs/checkpoints/fusionnet.pt",
    report_path: str = "outputs/reports/fusion_ablation.csv",
    epochs: int = 100,
    lr: float = 1e-2,
    max_samples: int = 150,
    smoke: bool = False,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Train FusionNet MLP and produce ablation report."""
    set_global_seed(cfg["seed"])
    dev = device or get_device()

    # 1. Load base segmentation model
    ck = Path(checkpoint_path) if checkpoint_path else resolve_path(cfg["inference"]["checkpoint"])
    if ck.exists():
        model, _ = load_model_from_checkpoint(str(ck), dev)
    else:
        model = build_model({**cfg["model"], "pretrained": False}).to(dev)

    # 2. Collect features and targets from val set
    feats, terr, targets, val_records = collect_val_features(
        cfg, model, dev, max_samples=max_samples, smoke=smoke
    )

    # 3. Add external log features if provided
    if log_features_csv and Path(log_features_csv).exists():
        logger.info("Reading additional features from log CSV: %s", log_features_csv)
        with open(log_features_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            extra_feats, extra_terr, extra_targets = [], [], []
            for r in reader:
                try:
                    u = float(r["U"])
                    g = float(r["G"])
                    t = float(r["T"])
                    a = float(r.get("alpha", 1.0))
                    extra_feats.append([u, g, t])
                    extra_terr.append(0)
                    extra_targets.append(a)
                except (ValueError, KeyError):
                    continue
            if extra_feats:
                feats = torch.cat([feats, torch.tensor(extra_feats, dtype=torch.float32)], dim=0)
                terr = torch.cat([terr, torch.tensor(extra_terr, dtype=torch.long)], dim=0)
                targets = torch.cat([targets, torch.tensor(extra_targets, dtype=torch.float32)], dim=0)

    # Add physical anchor points to preserve trust boundaries and monotonicity
    anchor_feats = torch.tensor([
        [0.05, 0.95, 0.95],
        [0.02, 0.98, 0.98],
        [0.0, 1.0, 1.0],
        [0.95, 0.20, 0.20],
        [0.98, 0.10, 0.10],
        [1.0, 0.0, 0.0],
    ], dtype=torch.float32)
    anchor_terr = torch.tensor([0, 0, 0, 3, 3, 3], dtype=torch.long)
    anchor_targets = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32)

    feats = torch.cat([feats, anchor_feats], dim=0)
    terr = torch.cat([terr, anchor_terr], dim=0)
    targets = torch.cat([targets, anchor_targets], dim=0)

    # 4. Train FusionNet MLP (warm-started from heuristic)
    net = FusionNet(cfg["fusion"], init_heuristic=True).to(dev)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)

    feats_dev = feats.to(dev)
    terr_dev = terr.to(dev)
    targets_dev = targets.to(dev)

    train_epochs = 5 if smoke else epochs
    net.train()
    logger.info("Training FusionNet MLP for %d epochs...", train_epochs)
    for ep in range(train_epochs):
        optimizer.zero_grad()
        pred = net(feats_dev, terr_dev)
        loss = F.binary_cross_entropy(pred.clamp(1e-6, 1.0 - 1e-6), targets_dev)
        loss.backward()
        optimizer.step()
        if (ep + 1) % max(1, train_epochs // 5) == 0 or ep == train_epochs - 1:
            logger.info("FusionNet epoch %d/%d - BCE loss: %.4f", ep + 1, train_epochs, float(loss.item()))

    # 5. Save trained weights
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), str(out_p))
    logger.info("Saved trained FusionNet checkpoint to %s", out_p)

    # Also save to outputs/checkpoints/fusion_net.pt for legacy compatibility
    legacy_p = out_p.parent / "fusion_net.pt"
    torch.save(net.state_dict(), str(legacy_p))

    # 6. Ablation comparison: Heuristic vs Trained FusionNet
    net.eval()
    heuristic_net = FusionNet(cfg["fusion"], init_heuristic=True).to(dev).eval()

    fb_threshold = float(cfg["fusion"].get("fallback_threshold", 0.5))
    mask_threshold = float(cfg["fusion"].get("mask_threshold", 0.5))

    def evaluate_fusion_variant(eval_net: FusionNet) -> Tuple[float, float, float]:
        ious, fallbacks, alphas = [], [], []
        with torch.no_grad():
            for r in val_records:
                f_in = torch.tensor([[r["unc"], r["geo"], r["tmp"]]], dtype=torch.float32, device=dev)
                t_in = torch.tensor([0], dtype=torch.long, device=dev)
                alpha = float(eval_net(f_in, t_in)[0].item())
                alphas.append(alpha)
                fallbacks.append(1.0 if alpha < fb_threshold else 0.0)

                # Fuse masks
                soft = alpha * r["dl_mask"].astype(np.float32) + (1.0 - alpha) * r["fb_mask"].astype(np.float32)
                fused_mask = (soft >= mask_threshold).astype(np.uint8)
                ious.append(compute_mask_iou(fused_mask, r["gt"]))

        return float(np.mean(ious)), float(np.mean(fallbacks)), float(np.mean(alphas))

    miou_h, fb_rate_h, mean_a_h = evaluate_fusion_variant(heuristic_net)
    miou_t, fb_rate_t, mean_a_t = evaluate_fusion_variant(net)

    logger.info("=== Fusion Ablation Results ===")
    logger.info("Heuristic:         mIoU=%.4f, Fallback Rate=%.1f%%, Mean Alpha=%.3f", miou_h, fb_rate_h * 100.0, mean_a_h)
    logger.info("Trained FusionNet: mIoU=%.4f, Fallback Rate=%.1f%%, Mean Alpha=%.3f", miou_t, fb_rate_t * 100.0, mean_a_t)

    # 7. Write ablation table
    rep_p = Path(report_path)
    rep_p.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_p, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "mean_iou", "fallback_rate", "mean_alpha"])
        writer.writerow(["heuristic", f"{miou_h:.4f}", f"{fb_rate_h:.4f}", f"{mean_a_h:.4f}"])
        writer.writerow(["trained_fusionnet", f"{miou_t:.4f}", f"{fb_rate_t:.4f}", f"{mean_a_t:.4f}"])

    logger.info("Ablation report saved to %s", rep_p)
    return {
        "checkpoint": str(out_p),
        "report": str(rep_p),
        "heuristic": {"mean_iou": miou_h, "fallback_rate": fb_rate_h, "mean_alpha": mean_a_h},
        "trained": {"mean_iou": miou_t, "fallback_rate": fb_rate_t, "mean_alpha": mean_a_t},
    }


def train_fusion(cfg: Dict[str, Any], csv_path: Optional[str] = None, epochs: int = 200, lr: float = 1e-2,
                 out_path: Optional[str] = None) -> Optional[str]:
    """Backward compatibility function for stage 9 smoke test."""
    p = Path(csv_path) if csv_path else resolve_path(cfg["fusion"].get("train_csv", "outputs/fusion_dummy.csv"))
    if not p.exists():
        return None
    with open(p, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    feats = torch.tensor([[float(r["uncertainty"]), float(r["geometric"]), float(r["temporal"])] for r in rows])
    terr = torch.tensor([int(r["terrain"]) for r in rows])
    y = torch.tensor([float(r["target_alpha"]) for r in rows])
    dev = get_device()
    net = FusionNet(cfg["fusion"], init_heuristic=False).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    feats, terr, y = feats.to(dev), terr.to(dev), y.to(dev)
    for ep in range(epochs):
        opt.zero_grad()
        loss = F.binary_cross_entropy(net(feats, terr).clamp(1e-6, 1 - 1e-6), y)
        loss.backward()
        opt.step()
    out = Path(out_path) if out_path else resolve_path(cfg["paths"]["checkpoint_dir"]) / "fusion_net.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out)
    return str(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train FusionNet MLP and run ablation study.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--log-features", default=None)
    ap.add_argument("--checkpoint", default="outputs/checkpoints/student_best.pt")
    ap.add_argument("--output", default="outputs/checkpoints/fusionnet.pt")
    ap.add_argument("--report", default="outputs/reports/fusion_ablation.csv")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--max-samples", type=int, default=150)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    setup_logging()
    cfg = load_config(a.config)
    train_fusionnet(
        cfg=cfg,
        log_features_csv=a.log_features,
        checkpoint_path=a.checkpoint,
        output_path=a.output,
        report_path=a.report,
        epochs=a.epochs,
        lr=a.lr,
        max_samples=a.max_samples,
        smoke=a.smoke,
    )


if __name__ == "__main__":
    main()
