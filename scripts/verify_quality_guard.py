"""Verify quality guards on 50 evenly spaced frames and extract 8 annotated check frames."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.detector import build_detector
from inference.live import LivePipeline, annotate
from training.train import load_model_from_checkpoint
from utils.config import load_config, resolve_path
from utils.seed import get_device, set_global_seed


import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Verify quality guards on video.")
    parser.add_argument("--video", type=str, default="dashcam_raw.mp4", help="Path to input video")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = args.video
    if not Path(video_path).exists():
        sys.exit(f"ERROR: Video file not found: {video_path}. Provide via --video <path>.")

    cfg = load_config("config.yaml")
    set_global_seed(cfg["seed"])
    device = get_device()
    inf = cfg["inference"]
    model, _ = load_model_from_checkpoint(str(resolve_path(inf["checkpoint"])), device)

    # Use null detector for comparing raw segmentation masks against before_masks_50
    pipeline_seg = LivePipeline(cfg, model, device, build_detector({"backend": "null"}, cfg["paths"]["output_dir"]))

    before_npz_path = Path("outputs/inference/before_masks_50.npz")
    if not before_npz_path.exists():
        print(f"ERROR: {before_npz_path} not found.")
        return

    before_masks = np.load(before_npz_path)
    cap = cv2.VideoCapture(video_path)

    ious = []
    indices = sorted([int(k) for k in before_masks.files])

    for f_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_idx))
        ok, frame = cap.read()
        if not ok:
            continue
        if frame.shape[1] > 1280:
            scale = 1280.0 / frame.shape[1]
            frame = cv2.resize(frame, (1280, int(round(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)

        res = pipeline_seg.process_frame(frame, frame_idx=int(f_idx))
        cur_mask = res.mask.astype(np.uint8)
        prev_mask = before_masks[str(f_idx)].astype(np.uint8)

        # Compute IoU between masks
        inter = np.logical_and(cur_mask > 0, prev_mask > 0).sum()
        union = np.logical_or(cur_mask > 0, prev_mask > 0).sum()
        iou = (inter / max(1.0, float(union))) if union > 0 else 1.0
        ious.append(float(iou))

    mean_iou = float(np.mean(ious))
    min_iou = float(np.min(ious))
    print(f"Quality Guard 50-Frame Mask IoU: mean={mean_iou:.4f}, min={min_iou:.4f} (Target: >= 0.9700)")

    # Save 8 evenly spaced annotated check frames with full detector
    check_dir = Path("outputs/inference/check_frames")
    check_dir.mkdir(parents=True, exist_ok=True)

    pipeline_full = LivePipeline(cfg, model, device, build_detector(inf["detector"], cfg["paths"]["output_dir"], device=device))
    check_indices = np.linspace(0, max(0, len(indices) - 1), 8, dtype=int)
    for i, idx_in_list in enumerate(check_indices):
        real_idx = indices[idx_in_list]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(real_idx))
        ok, frame = cap.read()
        if not ok:
            continue
        if frame.shape[1] > 1280:
            scale = 1280.0 / frame.shape[1]
            frame = cv2.resize(frame, (1280, int(round(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)

        res = pipeline_full.process_frame(frame, frame_idx=int(real_idx))
        ann = annotate(frame, res, frame_idx=int(real_idx), total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cfg=cfg)
        out_img_p = check_dir / f"frame_{i + 1:02d}_idx{real_idx}.jpg"
        cv2.imwrite(str(out_img_p), ann)
        print(f"Saved check frame {i + 1}/8: {out_img_p}")

    cap.release()

    guard_res = {
        "mean_mask_iou": mean_iou,
        "min_mask_iou": min_iou,
        "passed": bool(mean_iou >= 0.97),
    }
    Path("outputs/reports/quality_guard.json").write_text(json.dumps(guard_res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
