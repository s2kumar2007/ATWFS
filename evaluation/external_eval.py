"""External and cross-dataset evaluation (BDD100K zero-shot validation).

Computes mIoU, mean evidential uncertainty, and ECE on external datasets.
BDD100K drivable area classes:
  0: direct drivable (ego vehicle lane)
  1: alternative drivable (other lanes)
  2: background (non-drivable)
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

from evaluation.evaluate import official_confusion, official_ious
from models.unet_mobilenet import evidential_probs, evidential_uncertainty
from training.metrics import ECEAccumulator
from training.train import load_model_from_checkpoint
from utils.config import load_config
from utils.logging_utils import setup_logging
from utils.seed import get_device

logger = logging.getLogger(__name__)


def discover_bdd100k_pairs(root: Path) -> List[Tuple[Path, Path]]:
    """Discover (image, mask) pairs in standard or flat BDD100K directory layouts."""
    root = Path(root)
    if not root.exists():
        return []

    img_candidates = [
        root / "images" / "100k" / "val",
        root / "images" / "val",
        root / "images",
        root / "val" / "images",
        root / "val",
        root,
    ]
    mask_candidates = [
        root / "labels" / "drivable" / "masks" / "val",
        root / "labels" / "drivable" / "val",
        root / "labels" / "masks" / "val",
        root / "labels" / "val",
        root / "masks" / "val",
        root / "masks",
        root / "labels",
    ]

    img_dir = None
    for cand in img_candidates:
        if cand.is_dir() and (any(cand.glob("*.jpg")) or any(cand.glob("*.png")) or any(cand.glob("*.jpeg"))):
            img_dir = cand
            break

    if img_dir is None:
        return []

    mask_dir = None
    for cand in mask_candidates:
        if cand.is_dir() and (any(cand.glob("*.png")) or any(cand.glob("*.jpg"))):
            mask_dir = cand
            break

    all_imgs = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg")))
    pairs: List[Tuple[Path, Path]] = []

    for img in all_imgs:
        # Don't treat mask files as images if img_dir == mask_dir
        if "_mask" in img.stem or "_drivable" in img.stem:
            continue

        stem = img.stem
        candidates = []
        if mask_dir:
            candidates.extend([
                mask_dir / f"{stem}.png",
                mask_dir / f"{stem}_drivable_id.png",
                mask_dir / f"{stem}_drivable_color.png",
                mask_dir / f"{stem}_mask.png",
                mask_dir / f"{stem}.jpg",
            ])
        candidates.extend([
            img.parent / f"{stem}_mask.png",
            img.parent / f"{stem}_drivable_id.png",
            root / "masks" / f"{stem}.png",
        ])

        for cand in candidates:
            if cand.is_file():
                pairs.append((img, cand))
                break

    return pairs


class BDD100KDataset(Dataset):
    """BDD100K drivable-area dataset with class mapping.

    Classes: 0=direct, 1=alternative, 2=background.
    Mapping:
      - direct_only=False (default): classes 0 and 1 -> 1 (drivable), class 2 -> 0 (background)
      - direct_only=True: class 0 -> 1 (drivable), classes 1 and 2 -> 0 (background)
      - 255: ignore
    """

    def __init__(
        self,
        root: str | Path | None = None,
        cfg: Optional[Dict[str, Any]] = None,
        direct_only: bool = False,
        images_dir: Optional[str | Path] = None,
        masks_dir: Optional[str | Path] = None,
    ) -> None:
        self.root = Path(root) if root else None
        self.cfg = cfg or {}
        self.direct_only = direct_only

        data_cfg = self.cfg.get("data", {})
        w, h = data_cfg.get("image_size", (512, 288))
        mean = data_cfg.get("imagenet_mean", [0.485, 0.456, 0.406])
        std = data_cfg.get("imagenet_std", [0.229, 0.224, 0.225])

        self.target_size = (w, h)
        self.transform = T.Compose([
            T.Resize((h, w)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

        if images_dir and masks_dir:
            img_p, mask_p = Path(images_dir), Path(masks_dir)
            imgs = sorted(list(img_p.glob("*.jpg")) + list(img_p.glob("*.png")) + list(img_p.glob("*.jpeg")))
            self.pairs = []
            for img in imgs:
                m = mask_p / f"{img.stem}.png"
                if not m.exists():
                    m = mask_p / f"{img.stem}_drivable_id.png"
                if m.exists():
                    self.pairs.append((img, m))
        elif self.root:
            self.pairs = discover_bdd100k_pairs(self.root)
        else:
            self.pairs = []

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, mask_path = self.pairs[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        w, h = self.target_size
        mask = mask.resize((w, h), Image.NEAREST)
        raw_mask = np.array(mask, dtype=np.int64)

        if self.direct_only:
            target = np.where(raw_mask == 0, 1, np.where(raw_mask == 255, 255, 0))
        else:
            target = np.where((raw_mask == 0) | (raw_mask == 1), 1, np.where(raw_mask == 255, 255, 0))

        img_t = self.transform(img)
        mask_t = torch.from_numpy(target).long()
        return {"image": img_t, "mask": mask_t, "path": str(img_path)}


class ExternalDataset(Dataset):
    """Generic external image/mask dataset for custom validation folders."""

    def __init__(self, images_dir: str | Path, masks_dir: str | Path, cfg: Dict[str, Any]) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.cfg = cfg

        self.images = sorted(list(self.images_dir.glob("*.png")) + list(self.images_dir.glob("*.jpg")))
        w, h = cfg["data"]["image_size"]
        self.transform = T.Compose([
            T.Resize((h, w)),
            T.ToTensor(),
            T.Normalize(mean=cfg["data"]["imagenet_mean"], std=cfg["data"]["imagenet_std"]),
        ])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.images[idx]
        mask_path = self.masks_dir / (img_path.stem + ".png")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        w, h = self.cfg["data"]["image_size"]
        mask = mask.resize((w, h), Image.NEAREST)

        img_t = self.transform(img)
        mask_t = torch.from_numpy(np.array(mask)).long()

        return {"image": img_t, "mask": mask_t, "path": str(img_path)}


def run_cross_dataset_eval(
    bdd100k_dir: str | Path | None,
    checkpoint: str | Path,
    config_path: str = "config.yaml",
    output_csv: str = "outputs/reports/cross_dataset.csv",
    direct_only: bool = False,
    device: Optional[torch.device] = None,
) -> Optional[Dict[str, Any]]:
    """Run zero-shot evaluation on BDD100K drivable area validation set."""
    if not bdd100k_dir or not Path(bdd100k_dir).exists():
        print("NEEDS USER DATA: BDD100K dataset path (--bdd100k-dir)")
        return None

    cfg = load_config(config_path) if isinstance(config_path, str) else config_path
    if device is None:
        device = get_device()

    ds = BDD100KDataset(bdd100k_dir, cfg=cfg, direct_only=direct_only)
    if len(ds) == 0:
        print("NEEDS USER DATA: BDD100K dataset path (--bdd100k-dir)")
        return None

    model, _ = load_model_from_checkpoint(str(checkpoint), device)
    model.eval()

    loader = DataLoader(ds, batch_size=cfg.get("train", {}).get("batch_size", 4), shuffle=False, num_workers=0)

    mat = np.zeros((3, 3), np.int64)
    ece_accum = ECEAccumulator(bins=cfg.get("train", {}).get("ece_bins", 15))
    uncertainties: List[float] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["mask"].to(device)

            out = model(images)
            logits = out["seg_logits"]
            preds = logits.argmax(dim=1).cpu().numpy()
            gts = targets.cpu().numpy()

            mat = official_confusion(gts, preds, 2, mat)

            alpha = out["alpha"].float()
            probs = evidential_probs(alpha).cpu()
            u = evidential_uncertainty(alpha).cpu()

            valid_targets = torch.where(targets == 255, 0, targets).cpu()
            ece_accum.update(probs, valid_targets)
            uncertainties.append(float(u.mean().item()))

    ious = official_ious(mat)
    iou_nondrivable = float(ious[0])
    iou_drivable = float(ious[1])
    miou = float(np.nanmean(ious))
    mean_uncertainty = float(np.mean(uncertainties)) if uncertainties else 0.0
    ece_val = float(ece_accum.compute())

    results = {
        "dataset": "BDD100K",
        "checkpoint": str(checkpoint),
        "num_samples": len(ds),
        "miou": miou,
        "iou_drivable": iou_drivable,
        "iou_nondrivable": iou_nondrivable,
        "mean_uncertainty": mean_uncertainty,
        "ece": ece_val,
    }

    out_p = Path(output_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results.keys()))
        writer.writeheader()
        writer.writerow(results)

    logger.info(
        "BDD100K zero-shot eval: mIoU=%.4f, drivable=%.4f, uncertainty=%.4f, ECE=%.4f",
        miou, iou_drivable, mean_uncertainty, ece_val,
    )
    print(
        f"BDD100K zero-shot eval completed: mIoU={miou:.4f}, drivable_iou={iou_drivable:.4f}, "
        f"mean_U={mean_uncertainty:.4f}, ECE={ece_val:.4f} -> {out_p}"
    )
    return results


def run_external_eval() -> None:
    """CLI runner for external evaluation and BDD100K cross-dataset testing."""
    ap = argparse.ArgumentParser(description="External and cross-dataset evaluation.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--images", default=None)
    ap.add_argument("--masks", default=None)
    ap.add_argument("--checkpoint", default="outputs/checkpoints/student_best.pt")
    ap.add_argument("--output-csv", default=None)
    ap.add_argument("--bdd100k-dir", default=None)
    ap.add_argument("--direct-only", action="store_true", help="Map only direct drivable to drivable class")
    a = ap.parse_args()

    setup_logging()

    if a.bdd100k_dir is not None:
        out_csv = a.output_csv or "outputs/reports/cross_dataset.csv"
        res = run_cross_dataset_eval(
            bdd100k_dir=a.bdd100k_dir,
            checkpoint=a.checkpoint,
            config_path=a.config,
            output_csv=out_csv,
            direct_only=a.direct_only,
        )
        if res is None:
            sys.exit(0)
    elif a.images is not None and a.masks is not None:
        cfg = load_config(a.config)
        device = get_device()
        model, _ = load_model_from_checkpoint(a.checkpoint, device)
        model.eval()

        ds = ExternalDataset(a.images, a.masks, cfg)
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

        total_inter, total_union = 0, 0
        total_inter_bg, total_union_bg = 0, 0
        ece_bins = cfg.get("train", {}).get("ece_bins", 15)
        conf_sum = np.zeros(ece_bins)
        acc_sum = np.zeros(ece_bins)
        bin_counts = np.zeros(ece_bins)
        results = []

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                targets = batch["mask"].to(device)
                out = model(images)
                logits = out["seg_logits"]
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                max_conf, _ = probs.max(dim=1)

                for i in range(images.size(0)):
                    p, t = preds[i], targets[i]
                    inter = ((p == 1) & (t == 1)).sum().item()
                    union = ((p == 1) | (t == 1)).sum().item()
                    iou = inter / max(union, 1e-6)
                    results.append({"path": batch["path"][i], "iou": iou})
                    total_inter += inter
                    total_union += union

                    inter_bg = ((p == 0) & (t == 0)).sum().item()
                    union_bg = ((p == 0) | (t == 0)).sum().item()
                    total_inter_bg += inter_bg
                    total_union_bg += union_bg

                    conf = max_conf[i].cpu().numpy()
                    acc = (p == t).float().cpu().numpy()
                    bin_idx = np.clip(np.floor(conf * ece_bins).astype(int), 0, ece_bins - 1)
                    for b, c, a_ in zip(bin_idx.flatten(), conf.flatten(), acc.flatten()):
                        bin_counts[b] += 1
                        conf_sum[b] += c
                        acc_sum[b] += a_

        drivable_iou = total_inter / max(total_union, 1e-6)
        bg_iou = total_inter_bg / max(total_union_bg, 1e-6)
        miou = (drivable_iou + bg_iou) / 2.0

        valid_bins = bin_counts > 0
        bin_accs = acc_sum[valid_bins] / bin_counts[valid_bins]
        bin_confs = conf_sum[valid_bins] / bin_counts[valid_bins]
        ece = np.sum(np.abs(bin_accs - bin_confs) * (bin_counts[valid_bins] / bin_counts.sum()))

        logger.info(f"Drivable IoU: {drivable_iou:.4f}, mIoU: {miou:.4f}, ECE: {ece:.4f}")
        results.sort(key=lambda x: x["iou"])
        out_path = Path(a.output_csv or "outputs/reports/external_eval.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "iou"])
            writer.writeheader()
            writer.writerows(results)
    else:
        print("NEEDS USER DATA: BDD100K dataset path (--bdd100k-dir)")
        sys.exit(0)


if __name__ == "__main__":
    run_external_eval()
