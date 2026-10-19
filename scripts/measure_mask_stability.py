"""Measure mask stability on WhatsApp Video:
- Consecutive-frame IoU (N=1) over 300 frames (mean and min)
- N=2 reused mask vs N=1 mask over 300 frames (mean and min)
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from utils.config import load_config
from training.train import load_model_from_checkpoint
import onnxruntime
import torch

import argparse

parser = argparse.ArgumentParser(description="Measure mask stability on video.")
parser.add_argument("--video", type=str, default="dashcam_raw.mp4", help="Path to input video")
args = parser.parse_args()

video_path = args.video
if not Path(video_path).exists():
    sys.exit(f"ERROR: Video file not found: {video_path}. Provide via --video <path>.")
cap = cv2.VideoCapture(video_path)

cfg = load_config()
h_in, w_in = 288, 512
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

opts = onnxruntime.SessionOptions()
opts.intra_op_num_threads = 10
sess = onnxruntime.InferenceSession("outputs/checkpoints/student_best.onnx", opts, providers=["CPUExecutionProvider"])

n_frames = 300
masks_n1 = []

print(f"Running inference on {n_frames} frames to extract N=1 masks...")
for idx in range(n_frames):
    ret, frame = cap.read()
    if not ret:
        break
    fh, fw = frame.shape[:2]
    # Resize to 1280 max width if larger
    if fw > 1280:
        frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
    
    small = cv2.resize(frame, (w_in, h_in), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    x_norm = (rgb.astype(np.float32) / 255.0 - mean) / std
    x_np = np.ascontiguousarray(x_norm.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32))

    logits, _ = sess.run(None, {"image": x_np})
    m = np.argmax(logits, axis=1)[0].astype(np.uint8)
    masks_n1.append(m)

cap.release()
print(f"Extracted {len(masks_n1)} masks.")

# 1. Consecutive-frame IoU (N=1)
consec_ious = []
for t in range(1, len(masks_n1)):
    m1 = masks_n1[t - 1]
    m2 = masks_n1[t]
    inter = np.logical_and(m1 == 1, m2 == 1).sum()
    union = np.logical_or(m1 == 1, m2 == 1).sum()
    iou = float(inter / union) if union > 0 else 1.0
    consec_ious.append(iou)

mean_consec = float(np.mean(consec_ious))
min_consec = float(np.min(consec_ious))

# 2. Reused mask (N=2) vs N=1
# For N=2, on odd frames (1, 3, 5...), reused mask = mask from t-1 (0, 2, 4...)
n2_ious = []
for t in range(len(masks_n1)):
    m_n1 = masks_n1[t]
    m_n2 = masks_n1[t if (t % 2 == 0) else (t - 1)]
    inter = np.logical_and(m_n1 == 1, m_n2 == 1).sum()
    union = np.logical_or(m_n1 == 1, m_n2 == 1).sum()
    iou = float(inter / union) if union > 0 else 1.0
    n2_ious.append(iou)

mean_n2 = float(np.mean(n2_ious))
min_n2 = float(np.min(n2_ious))

print(f"\n=== MASK STABILITY METRICS ({len(masks_n1)} frames) ===")
print(f"Consecutive masks (N=1): Mean IoU = {mean_consec:.4f}, Min IoU = {min_consec:.4f}")
print(f"N=2 reused vs N=1:       Mean IoU = {mean_n2:.4f}, Min IoU = {min_n2:.4f}")

# Check with 3-frame temporal smoothing
smoothed_masks = []
for t in range(len(masks_n1)):
    window = masks_n1[max(0, t - 2):t + 1]
    smoothed = (np.sum(window, axis=0) > (len(window) // 2)).astype(np.uint8)
    smoothed_masks.append(smoothed)

consec_smooth_ious = []
for t in range(1, len(smoothed_masks)):
    m1 = smoothed_masks[t - 1]
    m2 = smoothed_masks[t]
    inter = np.logical_and(m1 == 1, m2 == 1).sum()
    union = np.logical_or(m1 == 1, m2 == 1).sum()
    iou = float(inter / union) if union > 0 else 1.0
    consec_smooth_ious.append(iou)

mean_smooth = float(np.mean(consec_smooth_ious))
min_smooth = float(np.min(consec_smooth_ious))
print(f"Consecutive smoothed:    Mean IoU = {mean_smooth:.4f}, Min IoU = {min_smooth:.4f}")
