"""Evaluate the full held-out test split and produce a comparison table.

Usage
-----
  python -m evaluation.test_split_eval \\
      --data-src /path/to/IDD_RESIZED \\
      --checkpoint outputs/checkpoints/teacher_v2_best.pt \\
      --student-checkpoint outputs/checkpoints/student_v2_best.pt \\
      --onnx outputs/checkpoints/student_v2.onnx \\
      --baseline-checkpoint outputs/checkpoints/student_best.pt \\
      --splits-dir data/splits \\
      --output outputs/reports/final_eval_v2.json

All images are loaded at native resolution, then **stretched** (not letterboxed)
to 288x512, matching the training preprocessing for the 512x512 squashed dataset.

The test split is never seen during training or early-stopping.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.unet_mobilenet import evidential_probs, evidential_uncertainty
from training.metrics import ECEAccumulator
from training.train import load_model_from_checkpoint
from utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Drivable-fallback is level-3 id 0 (road) and 1 (drivable other) in the IDD taxonomy.
# After build_dataset_from_splits.py: drivable -> 0, non-drivable -> 3.
DRIVABLE_LABEL = 0
IGNORE_LABEL = 255


def preprocess(bgr: np.ndarray, input_h: int = 288, input_w: int = 512) -> np.ndarray:
    """Stretch (not letterbox) BGR -> CHW float32 normalised tensor.

    The Kaggle IDD_RESIZED images are 512x512 squashed from 16:9 frames.
    Inference (live.py) also stretches native frames.  Both must agree.
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    x = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return x.transpose(2, 0, 1)  # CHW


def preprocess_mask(mask: np.ndarray, input_h: int = 288, input_w: int = 512) -> np.ndarray:
    """Stretch mask to training size, preserving ignore label."""
    return cv2.resize(mask, (input_w, input_h), interpolation=cv2.INTER_NEAREST)


def iou_from_confusion(conf: np.ndarray) -> np.ndarray:
    """IoU per class from a 2x2 confusion matrix (labels 0..1, ignore=2+)."""
    ious = []
    for c in range(conf.shape[0] - 1):  # last row/col is catch-all
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        denom = tp + fp + fn
        ious.append(float(tp) / max(1e-8, float(denom)))
    return np.array(ious)


class TestDataset(Dataset):
    """Load images + masks from the split txt file and the raw archive."""

    def __init__(
        self,
        file_list: list[str],
        img_dir: Path,
        mask_dir: Path,
        input_h: int = 288,
        input_w: int = 512,
    ) -> None:
        self.files = file_list
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.input_h = input_h
        self.input_w = input_w

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        fname = self.files[idx]
        img_id = fname.replace("Image_", "").replace(".png", "")
        ip = self.img_dir / fname
        mp = self.mask_dir / f"Mask_{img_id}.png"

        bgr = cv2.imread(str(ip))
        if bgr is None:
            raise FileNotFoundError(f"Missing image: {ip}")
        chw = preprocess(bgr, self.input_h, self.input_w)

        raw_mask = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
        if raw_mask is None:
            raise FileNotFoundError(f"Missing mask: {mp}")
        # Binary collapse matching DrivableDataset: drivable (==1) -> 1, rest -> 0, keep 255 as ignore
        binary = np.where(raw_mask == 255, 255, np.where(raw_mask == 1, 1, 0)).astype(np.uint8)
        binary_resized = preprocess_mask(binary, self.input_h, self.input_w)

        return {
            "image": torch.from_numpy(chw).float(),
            "mask": torch.from_numpy(binary_resized.astype(np.int64)),
            "fname": fname,
        }


def update_confusion(conf: np.ndarray, pred: np.ndarray, gt: np.ndarray, n_cls: int = 2) -> np.ndarray:
    valid = gt != IGNORE_LABEL
    p = pred[valid].clip(0, n_cls - 1).astype(np.int64)
    g = gt[valid].clip(0, n_cls - 1).astype(np.int64)
    np.add.at(conf, (g, p), 1)
    return conf


@torch.no_grad()
def eval_pytorch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    ece_bins: int = 15,
) -> Dict[str, float]:
    model.eval()
    conf = np.zeros((3, 3), dtype=np.int64)  # 2 real classes + overflow
    dirt_conf = np.zeros((3, 3), dtype=np.int64)
    ece = ECEAccumulator(ece_bins)
    uncertainties: list[float] = []
    dirt_n = 0

    for batch in loader:
        imgs = batch["image"].to(device)
        masks_np = batch["mask"].numpy()
        out = model(imgs)
        preds = out["seg_logits"].argmax(1).cpu().numpy()
        alphas = out["alpha"].cpu().float()
        probs = evidential_probs(alphas)
        uncs = evidential_uncertainty(alphas).mean(dim=(1, 2)).numpy()
        uncertainties.extend(uncs.tolist())

        for i in range(len(imgs)):
            mask = masks_np[i]
            pred = preds[i]
            conf = update_confusion(conf, pred, mask)
            gt_t = torch.from_numpy(np.where(mask == IGNORE_LABEL, 0, mask)).long().unsqueeze(0)
            ece.update(probs[i : i + 1], gt_t)
            valid = mask != IGNORE_LABEL
            if valid.any():
                driv_frac = float((mask[valid] == 1).mean())
                if 0.05 < driv_frac < 0.50:  # dirt-heavy: some but not all drivable
                    dirt_conf = update_confusion(dirt_conf, pred, mask)
                    dirt_n += 1

    ious = iou_from_confusion(conf)
    dirt_ious = iou_from_confusion(dirt_conf)
    return {
        "mIoU": float(np.mean(ious)),
        "iou_drivable": float(ious[1]),
        "iou_nondrivable": float(ious[0]),
        "dirt_drivable_iou": float(dirt_ious[1]) if dirt_n > 0 else float("nan"),
        "ECE": float(ece.compute()),
        "mean_uncertainty": float(np.mean(uncertainties)) if uncertainties else float("nan"),
        "dirt_samples": dirt_n,
        "total_samples": int(conf[:-1, :-1].sum()),
    }


def eval_onnx(
    sess: Any,
    loader: DataLoader,
    ece_bins: int = 15,
) -> Dict[str, float]:
    conf = np.zeros((3, 3), dtype=np.int64)
    dirt_conf = np.zeros((3, 3), dtype=np.int64)
    ece = ECEAccumulator(ece_bins)
    uncertainties: list[float] = []
    dirt_n = 0

    for batch in loader:
        imgs_np = batch["image"].numpy()
        masks_np = batch["mask"].numpy()
        seg_logits_np, alpha_np = sess.run(None, {"image": imgs_np})
        preds = np.argmax(seg_logits_np, axis=1)
        alphas = torch.from_numpy(alpha_np).float()
        probs = evidential_probs(alphas)
        uncs = evidential_uncertainty(alphas).mean(dim=(1, 2)).numpy()
        uncertainties.extend(uncs.tolist())

        for i in range(len(imgs_np)):
            mask = masks_np[i]
            pred = preds[i]
            conf = update_confusion(conf, pred, mask)
            gt_t = torch.from_numpy(np.where(mask == IGNORE_LABEL, 0, mask)).long().unsqueeze(0)
            ece.update(probs[i : i + 1], gt_t)
            valid = mask != IGNORE_LABEL
            if valid.any():
                driv_frac = float((mask[valid] == 1).mean())
                if 0.05 < driv_frac < 0.50:
                    dirt_conf = update_confusion(dirt_conf, pred, mask)
                    dirt_n += 1

    ious = iou_from_confusion(conf)
    dirt_ious = iou_from_confusion(dirt_conf)
    return {
        "mIoU": float(np.mean(ious)),
        "iou_drivable": float(ious[1]),
        "iou_nondrivable": float(ious[0]),
        "dirt_drivable_iou": float(dirt_ious[1]) if dirt_n > 0 else float("nan"),
        "ECE": float(ece.compute()),
        "mean_uncertainty": float(np.mean(uncertainties)) if uncertainties else float("nan"),
        "dirt_samples": dirt_n,
        "total_samples": int(conf[:-1, :-1].sum()),
    }


def print_table(results: Dict[str, Any]) -> None:
    header = f"{'Model':<28} {'mIoU':>6} {'DriveIoU':>9} {'NonDrvIoU':>10} {'DirtIoU':>8} {'ECE':>6} {'Unc':>6}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        if not isinstance(m, dict) or "mIoU" not in m:
            continue
        print(
            f"{name:<28} {m['mIoU']:>6.4f} {m['iou_drivable']:>9.4f} "
            f"{m['iou_nondrivable']:>10.4f} {m.get('dirt_drivable_iou', float('nan')):>8.4f} "
            f"{m['ECE']:>6.4f} {m['mean_uncertainty']:>6.4f}"
        )
    print("=" * len(header))
    print(f"Target: drivable IoU >= 0.85 on full held-out test split")
    if "numerical_check" in results:
        nc = results["numerical_check"]
        print(f"\nONNX vs PyTorch Numerical Check (50 images):")
        print(f"  Max abs diff (seg_logits): {nc['max_abs_diff_logits']:.4e}")
        print(f"  Max abs diff (alpha):      {nc['max_abs_diff_alpha']:.4e}")
        print(f"  Mean Mask IoU:             {nc['mean_mask_iou']:.6f}  (target >= 0.99)")
        print(f"  Min Mask IoU:              {nc['min_mask_iou']:.6f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate on the held-out test split at 288x512.")
    ap.add_argument("--data-src", default=None,
                    help="IDD_RESIZED root (contains image_archive/ and mask_archive/). If omitted, uses data/idd_converted.")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Limit number of test samples (default: None, evaluates full test split)")
    ap.add_argument("--splits-dir", default="data/splits",
                    help="Dir with test.txt (default: data/splits)")
    ap.add_argument("--checkpoint", default=None,
                    help="Teacher checkpoint (.pt)")
    ap.add_argument("--student-checkpoint", default=None,
                    help="Student checkpoint (.pt)")
    ap.add_argument("--onnx", default=None,
                    help="Student ONNX path (.onnx)")
    ap.add_argument("--baseline-checkpoint", default=None,
                    help="Baseline student checkpoint for comparison table")
    ap.add_argument("--output", default="outputs/reports/final_eval.json",
                    help="Output JSON path")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--input-h", type=int, default=288)
    ap.add_argument("--input-w", type=int, default=512)
    a = ap.parse_args()

    setup_logging()
    splits_dir = Path(a.splits_dir)
    test_file = splits_dir / "test.txt"
    if not test_file.exists():
        sys.exit(f"ERROR: {test_file} not found. Run scripts/build_dataset_from_splits.py first.")

    test_files = [ln.strip() for ln in test_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if a.max_samples and a.max_samples > 0:
        test_files = test_files[: a.max_samples]
    logger.info("Test split: evaluating %d images", len(test_files))

    if a.data_src:
        data_src = Path(a.data_src)
        img_dir = data_src / "image_archive"
        mask_dir = data_src / "mask_archive"
        if not img_dir.is_dir() or not mask_dir.is_dir():
            sys.exit(f"ERROR: {data_src} must contain image_archive/ and mask_archive/.")
        ds: Dataset = TestDataset(test_files, img_dir, mask_dir, a.input_h, a.input_w)
    else:
        # Fallback to data/idd_converted using name_mapping.json
        mapping_file = splits_dir / "name_mapping.json"
        conv_root = Path("data/idd_converted")
        if mapping_file.is_file() and (conv_root / "leftImg8bit" / "test").is_dir():
            with open(mapping_file, "r", encoding="utf-8") as f:
                name_map = json.load(f)

            class ConvertedTestDataset(Dataset):
                def __init__(self, fnames: list[str], root: Path, in_h: int, in_w: int):
                    self.fnames = fnames
                    self.root = root
                    self.in_h, self.in_w = in_h, in_w

                def __len__(self):
                    return len(self.fnames)

                def __getitem__(self, idx):
                    fn = self.fnames[idx]
                    rel = name_map[fn]
                    ip = self.root / "leftImg8bit" / "test" / rel
                    mp = self.root / "gtFine" / "test" / rel.replace("_leftImg8bit.png", "_gtFine_labellevel3Ids.png")
                    bgr = cv2.imread(str(ip))
                    if bgr is None:
                        raise FileNotFoundError(f"Missing {ip}")
                    chw = preprocess(bgr, self.in_h, self.in_w)
                    raw_mask = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
                    if raw_mask.ndim == 3:
                        raw_mask = raw_mask[:, :, 0]
                    # In converted dataset: 0 is road (drivable), 3 is non-drivable
                    # map: 0 -> 1 (drivable), 3 -> 0 (non-drivable)
                    binary = np.where(raw_mask == 255, 255, np.where(raw_mask == 0, 1, 0)).astype(np.uint8)
                    binary_resized = preprocess_mask(binary, self.in_h, self.in_w)
                    return {
                        "image": torch.from_numpy(chw).float(),
                        "mask": torch.from_numpy(binary_resized.astype(np.int64)),
                        "fname": fn,
                    }

            ds = ConvertedTestDataset(test_files, conv_root, a.input_h, a.input_w)
        else:
            sys.exit("ERROR: --data-src must be provided or data/idd_converted must be built. "
                     "Run scripts/build_dataset_from_splits.py --src <path_to_IDD_RESIZED> first.")

    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                        num_workers=a.num_workers, pin_memory=torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Evaluating on device: %s  (%d images)", device, len(test_files))

    results: Dict[str, Dict[str, float]] = {}

    if a.baseline_checkpoint and Path(a.baseline_checkpoint).exists():
        logger.info("Evaluating baseline student: %s", a.baseline_checkpoint)
        m, _ = load_model_from_checkpoint(a.baseline_checkpoint, device)
        results["student_best (baseline)"] = eval_pytorch(m, loader, device)
        del m

    if a.checkpoint and Path(a.checkpoint).exists():
        logger.info("Evaluating teacher: %s", a.checkpoint)
        m, _ = load_model_from_checkpoint(a.checkpoint, device)
        results["teacher_v2"] = eval_pytorch(m, loader, device)
        del m

    student_pt_model = None
    if a.student_checkpoint and Path(a.student_checkpoint).exists():
        logger.info("Evaluating student: %s", a.student_checkpoint)
        student_pt_model, _ = load_model_from_checkpoint(a.student_checkpoint, device)
        results["student_v2"] = eval_pytorch(student_pt_model, loader, device)

    if a.onnx and Path(a.onnx).exists():
        try:
            import onnxruntime as ort  # type: ignore
            logger.info("Evaluating ONNX student: %s", a.onnx)
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 6
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if torch.cuda.is_available()
                else ["CPUExecutionProvider"]
            )
            sess = ort.InferenceSession(a.onnx, opts, providers=providers)
            results["student_v2_onnx"] = eval_onnx(sess, loader)

            # Numerical equivalence check between PyTorch and ONNX if PyTorch student is available
            eval_pt = student_pt_model
            if eval_pt is None and a.baseline_checkpoint and Path(a.baseline_checkpoint).exists():
                eval_pt, _ = load_model_from_checkpoint(a.baseline_checkpoint, device)
            if eval_pt is not None:
                logger.info("Checking PyTorch vs ONNX numerical equivalence on first 50 test samples...")
                max_diff_l = 0.0
                max_diff_a = 0.0
                mask_ious = []
                eval_pt.eval()
                with torch.no_grad():
                    n_chk = 0
                    for b in loader:
                        imgs_t = b["image"].to(device)
                        py_out = eval_pt(imgs_t)
                        py_l = py_out["seg_logits"].cpu().numpy()
                        py_a = py_out["alpha"].cpu().numpy()
                        py_m = np.argmax(py_l, axis=1)

                        ox_l, ox_a = sess.run(None, {"image": b["image"].numpy()})
                        ox_m = np.argmax(ox_l, axis=1)

                        max_diff_l = max(max_diff_l, float(np.max(np.abs(py_l - ox_l))))
                        max_diff_a = max(max_diff_a, float(np.max(np.abs(py_a - ox_a))))

                        for j in range(len(imgs_t)):
                            inter = np.logical_and(py_m[j] == 1, ox_m[j] == 1).sum()
                            union = np.logical_or(py_m[j] == 1, ox_m[j] == 1).sum()
                            iou = 1.0 if union == 0 else float(inter / union)
                            mask_ious.append(iou)
                            n_chk += 1
                            if n_chk >= 50:
                                break
                        if n_chk >= 50:
                            break
                mean_iou = float(np.mean(mask_ious))
                min_iou = float(np.min(mask_ious))
                logger.info("ONNX vs PyTorch (50 images): max abs diff logits=%.4e, alpha=%.4e, mean Mask IoU=%.6f, min Mask IoU=%.6f",
                            max_diff_l, max_diff_a, mean_iou, min_iou)
                results["numerical_check"] = {
                    "max_abs_diff_logits": max_diff_l,
                    "max_abs_diff_alpha": max_diff_a,
                    "mean_mask_iou": mean_iou,
                    "min_mask_iou": min_iou,
                }
        except ImportError:
            logger.warning("onnxruntime not installed; skipping ONNX evaluation.")

    if not results:
        sys.exit("No valid checkpoints provided. Pass at least one of --checkpoint / --student-checkpoint / --onnx.")

    print_table(results)

    out_path = Path(a.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "split": "held-out-test",
        "resolution": f"{a.input_h}x{a.input_w}",
        "preprocessing": "stretch (512x512 squashed Kaggle -> 288x512 stretch, no letterbox)",
        "total_test_images": len(test_files),
        "results": results,
    }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved evaluation report to %s", out_path)


if __name__ == "__main__":
    main()
