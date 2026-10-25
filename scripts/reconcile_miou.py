"""Evaluate student at 256x256 and 288x512 on:
(a) Custom val split (real IDD resized archive with binary collapse)
(b) IDD val split (using AutoNUE official evaluation protocol with ignore penalty)
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from utils.config import load_config
from training.train import load_model_from_checkpoint
from evaluation.evaluate import official_confusion, official_ious
from data.dataset import build_dataloader, build_dataset

import argparse

ap = argparse.ArgumentParser(description="Evaluate student at 256x256 and 288x512")
ap.add_argument("--checkpoint", default="outputs/checkpoints/student_best.pt")
ap.add_argument("--data-src", default=None, help="Root of IDD_RESIZED (contains image_archive and mask_archive)")
args = ap.parse_args()

device = torch.device("cpu")
cfg = load_config()

# Load model
model, ckpt = load_model_from_checkpoint(args.checkpoint, device)
model.eval()
print(f"Loaded {args.checkpoint}")

# (b) IDD Val split (synthetic IDD benchmark used in evaluation/evaluate.py)
print("\n--- (b) IDD Val Split (AutoNUE protocol in evaluate.py) ---")
for size in [(256, 256), (288, 512)]:
    cfg_b = load_config()
    cfg_b["data"]["image_size"] = list(size)
    loader = build_dataloader(build_dataset(cfg_b, "val", True), cfg_b, False)
    mat = np.zeros((3, 3), np.int64)
    with torch.no_grad():
        for b in loader:
            out = model(b["image"].to(device))
            pred = out["seg_logits"].argmax(1).cpu().numpy()
            gt = np.where(b["valid"].numpy() > 0, b["mask"].numpy(), 255)
            mat = official_confusion(gt, pred, 2, mat)
    ious = official_ious(mat)
    miou = float(np.nanmean(ious))
    print(f"IDD Val (synthetic) @ {size[0]}x{size[1]}: mIoU = {miou:.4f} (drivable: {ious[1]:.4f}, non-drivable: {ious[0]:.4f})")

# (a) Custom Val split (from archive 2 real images)
if args.data_src and Path(args.data_src).is_dir():
    print(f"\n--- (a) Custom Val Split (Real IDD: {args.data_src}) ---")
    img_dir = Path(args.data_src) / "image_archive"
    mask_dir = Path(args.data_src) / "mask_archive"

    all_imgs = sorted(list(img_dir.glob("*.png")))
    # Use validation partition: last 981 images
    val_imgs = all_imgs[-981:] if len(all_imgs) >= 981 else all_imgs[:200]
    print(f"Evaluating {len(val_imgs)} validation images from {args.data_src}...")

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

for size in [(256, 256), (288, 512)]:
    h_in, w_in = size
    tp_c = [0, 0]
    fp_c = [0, 0]
    fn_c = [0, 0]
    with torch.no_grad():
        for ip in val_imgs[:200]:
            stem = ip.name.replace("Image_", "Mask_")
            mp = mask_dir / stem
            if not mp.is_file():
                continue
            bgr = cv2.imread(str(ip))
            gt_bin = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
            if gt_bin.ndim == 3:
                gt_bin = gt_bin[:, :, 0]

            gt_bin = (gt_bin > 0).astype(np.uint8)

            # Resize input
            bgr_resized = cv2.resize(bgr, (w_in, h_in), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(bgr_resized, cv2.COLOR_BGR2RGB)
            x = (rgb.astype(np.float32) / 255.0 - mean) / std
            x_t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)

            out = model(x_t)
            pred = out["seg_logits"].argmax(1)[0].cpu().numpy()

            # Resize pred to GT size (512x512)
            pred_full = cv2.resize(pred.astype(np.uint8), (gt_bin.shape[1], gt_bin.shape[0]), interpolation=cv2.INTER_NEAREST)

            for c in range(2):
                p_c = (pred_full == c)
                g_c = (gt_bin == c)
                tp = int(np.logical_and(p_c, g_c).sum())
                fp = int(np.logical_and(p_c, ~g_c).sum())
                fn = int(np.logical_and(~p_c, g_c).sum())
                tp_c[c] += tp
                fp_c[c] += fp
                fn_c[c] += fn

    ious = [tp_c[c] / max(1, tp_c[c] + fp_c[c] + fn_c[c]) for c in range(2)]
    miou = float(np.mean(ious))
    print(f"Custom Val (archive 2) @ {size[0]}x{size[1]}: mIoU = {miou:.4f} (drivable: {ious[1]:.4f}, non-drivable: {ious[0]:.4f})")
