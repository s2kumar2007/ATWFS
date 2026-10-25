"""Fine-tune a dedicated night decoder on low-light / night images.

Uses:
1. Synthetic darkening from data/augment.py applied to IDD dataset.
2. Optional real night frames if --night-dir is provided.
3. Freezes encoder and day decoder; optimizes only the night decoder branch.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data.augment import synthetic_darken
from data.dataset import DrivableDataset, IDDSource, SampleSource
from data.synthetic import SyntheticSource
from models.unet_mobilenet import MultiTaskUNet, build_model
from training.losses import MultiTaskLoss
from training.train import load_model_from_checkpoint
from utils.config import load_config, resolve_path
from utils.logging_utils import setup_logging
from utils.seed import get_device, set_global_seed

logger = logging.getLogger(__name__)


class NightSampleSource:
    """Wraps an underlying SampleSource and applies synthetic darkening to images."""

    def __init__(self, base_source: SampleSource, real_night_images: Optional[List[Path]] = None) -> None:
        self.base_source = base_source
        self.real_night_images = real_night_images or []

    def __len__(self) -> int:
        return len(self.base_source) + len(self.real_night_images)

    def load(self, i: int) -> Tuple[np.ndarray, np.ndarray, int]:
        if i < len(self.base_source):
            img, mask, terrain = self.base_source.load(i)
            # Apply synthetic darkening
            factor = float(np.random.uniform(0.15, 0.40))
            gamma = float(np.random.uniform(1.4, 2.2))
            dark_img = synthetic_darken(img, factor=factor, gamma=gamma, noise_std=4.0)
            return dark_img, mask, terrain
        else:
            # Real night image (unlabeled or self-masked)
            real_idx = i - len(self.base_source)
            p = self.real_night_images[real_idx]
            bgr = cv2.imread(str(p))
            if bgr is None:
                img = np.zeros((256, 256, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            return img, mask, -1


def finetune_night(
    cfg: Dict[str, Any],
    checkpoint_path: Optional[str] = None,
    output_path: str = "outputs/checkpoints/student_night.pt",
    night_dir: Optional[str] = None,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 1e-4,
    smoke: bool = False,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Fine-tune the night decoder."""
    set_global_seed(cfg["seed"])
    dev = device or get_device()

    # 1. Load or build model
    ck = Path(checkpoint_path) if checkpoint_path else resolve_path(cfg["inference"]["checkpoint"])
    if ck.exists():
        logger.info("Loading base model from %s", ck)
        model, _ = load_model_from_checkpoint(str(ck), dev)
    else:
        logger.info("Base checkpoint not found at %s; creating randomly initialized model", ck)
        model = build_model({**cfg["model"], "pretrained": False})

    model = model.to(dev)

    # 2. Add night decoder and freeze everything else
    if hasattr(model, "add_night_decoder"):
        night_dec = model.add_night_decoder()
    else:
        raise AttributeError("Model does not support add_night_decoder()")

    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad = False
    # Unfreeze only the night decoder branch
    for p in night_dec.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logger.info("Trainable parameters (night decoder): %d | Frozen parameters: %d", trainable, frozen)

    # 3. Check for real night images
    real_night_files: List[Path] = []
    if night_dir:
        nd = Path(night_dir)
        if nd.exists() and nd.is_dir():
            real_night_files = sorted(list(nd.glob("*.png")) + list(nd.glob("*.jpg")) + list(nd.glob("*.mp4")))
        if not real_night_files:
            print("NEEDS USER DATA: Real night frames folder (--night-dir) is empty or not found; using synthetic darkening on IDD.")
    else:
        print("NEEDS USER DATA: Real night frames folder (--night-dir) not provided; fine-tuning night decoder on synthetically darkened IDD images.")

    # 4. Prepare dataset
    if smoke:
        base_source: SampleSource = SyntheticSource(num_samples=16, image_size=tuple(cfg["data"]["image_size"]))
        epochs = 1
    else:
        try:
            base_source = IDDSource(cfg, split="train", require_masks=True)
        except Exception as e:
            logger.warning("Could not load IDD train split (%s); falling back to synthetic source.", e)
            base_source = SyntheticSource(num_samples=32, image_size=tuple(cfg["data"]["image_size"]))

    night_source = NightSampleSource(base_source, real_night_files)
    dataset = DrivableDataset(night_source, cfg, train=True, seed=cfg["seed"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 5. Training loop
    optimizer = torch.optim.AdamW(night_dec.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = MultiTaskLoss(cfg["loss"]).to(dev)

    model.train()
    history = []
    logger.info("Starting night decoder fine-tuning for %d epochs...", epochs)

    for epoch in range(epochs):
        loss_fn.set_epoch(epoch)
        epoch_losses = []
        for batch in loader:
            imgs = batch["image"].to(dev)
            optimizer.zero_grad()
            # Forward in night mode
            out = model(imgs, mode="night")
            total_loss, info = loss_fn(out, {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch.items()})
            total_loss.backward()
            optimizer.step()
            epoch_losses.append(float(total_loss.detach().item()))

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        logger.info("Epoch %d/%d - Night decoder loss: %.4f", epoch + 1, epochs, mean_loss)
        history.append({"epoch": epoch + 1, "loss": mean_loss})

    # 6. Save checkpoint
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "epoch": epochs,
        "night_decoder": night_dec.state_dict(),
        "config": cfg,
    }, str(out_p))
    logger.info("Saved night-capable model to %s", out_p)

    return {"output": str(out_p), "history": history, "trainable_params": trainable}


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune dedicated night decoder.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", default="outputs/checkpoints/student_best.pt")
    ap.add_argument("--output", default="outputs/checkpoints/student_night.pt")
    ap.add_argument("--night-dir", default=None, help="Optional directory with real night frames")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    setup_logging()
    cfg = load_config(a.config)
    finetune_night(
        cfg=cfg,
        checkpoint_path=a.checkpoint,
        output_path=a.output,
        night_dir=a.night_dir,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        smoke=a.smoke,
    )


if __name__ == "__main__":
    main()
