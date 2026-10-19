import collections
import json
import logging
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch
import onnxruntime as ort

REPO_DIR = Path(r"c:\Users\K\OneDrive\Documents\idd_drivable\idd_drivable")
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from utils.config import load_config, resolve_path
from inference.live import LivePipeline, annotate, _open_writer
from inference.detector import build_detector
from inference.hood_calibration import calibrate_or_load_hood_polygon

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("video_check_300")

def main():
    print("=== Step 7: VIDEO CHECK (300 FRAMES) ===")
    video_path = r"C:\Users\K\Downloads\WhatsApp Video 2026-10-12 at 6.18.52 AM.mp4"
    out_dir = REPO_DIR / "outputs" / "inference" / "check_frames_v4"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cfg = load_config(str(REPO_DIR / "config.yaml"))
    inf = cfg["inference"]
    
    # Camera settings
    cfg["camera"]["height_m"] = 1.3
    cfg["camera"]["hfov_deg"] = 75.0
    cfg["camera"]["vfov_deg"] = 45.0
    
    # Calibrate hood
    hood_poly = calibrate_or_load_hood_polygon(video_path)
    cfg["inference"]["hood_polygon"] = hood_poly
    
    # CPU profile
    prof_cfg = inf.get("profiles", {}).get("cpu", {})
    inf["detector"]["weights"] = str(REPO_DIR / prof_cfg.get("detector_weights", "yolov8n.pt"))
    inf["detector"]["imgsz"] = prof_cfg.get("detector_imgsz", 480)
    inf["detector"]["every_n"] = prof_cfg.get("detector_every_n", 4)
    inf["detector"]["crop_every_n"] = 0
    inf["detector"]["crop_upscale"] = 1.0
    inf["fallback_alpha_gate"] = prof_cfg.get("fallback_alpha_gate", 0.5)
    inf["ipm_every_n"] = prof_cfg.get("ipm_every_n", 10)
    inf["seg_every"] = prof_cfg.get("seg_every", 1)
    inf["input_size"] = [288, 512]
    
    # ONNX student session
    onnx_path = REPO_DIR / "outputs" / "checkpoints" / "student_v2.onnx"
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    onnx_session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
    
    detector = build_detector(inf["detector"], str(REPO_DIR / "outputs"), live_camera=False, device=torch.device("cpu"))
    pipeline = LivePipeline(cfg, None, torch.device("cpu"), detector, profile_name="cpu", onnx_session=onnx_session)
    
    # Target reference frames (1-based: 1, 101, 201, 237, 259, 285)
    target_frames_1based = {1, 101, 201, 237, 259, 285}
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")
        
    trust_vals = []
    fallback_count = 0
    consecutive_mask_ious = []
    prev_mask = None
    
    stage_timings = collections.defaultdict(list)
    t_start = time.perf_counter()
    
    frame_idx = 0
    max_frames = 300
    
    print(f"Running inference on {max_frames} frames with student_v2.onnx...")
    while frame_idx < max_frames:
        t_r0 = time.perf_counter()
        ret, frame_bgr = cap.read()
        t_read_ms = (time.perf_counter() - t_r0) * 1000.0
        if not ret:
            break
            
        stage_timings["read"].append(t_read_ms)
        
        # Pipeline process frame
        res = pipeline.process_frame(frame_bgr, frame_idx=frame_idx, dt=0.033, speed=1.0)
        
        # Trust & Fallback
        trust = float(res.command.trust)
        trust_vals.append(trust)
        is_fallback = (res.active.upper() == "FALLBACK" or trust < pipeline.fallback_alpha_gate)
        if is_fallback:
            fallback_count += 1
            
        # Mask IoU between consecutive frames
        curr_mask = (res.mask > 0).astype(np.uint8)
        if prev_mask is not None:
            inter = np.logical_and(curr_mask == 1, prev_mask == 1).sum()
            union = np.logical_or(curr_mask == 1, prev_mask == 1).sum()
            iou = float(inter / union) if union > 0 else 1.0
            consecutive_mask_ious.append(iou)
        prev_mask = curr_mask
        
        # Annotate
        t_a0 = time.perf_counter()
        ann = annotate(frame_bgr, res, frame_idx=frame_idx, total_frames=max_frames, fps=15.0, cfg=cfg)
        t_ann_ms = (time.perf_counter() - t_a0) * 1000.0
        stage_timings["annotate"].append(t_ann_ms)
        
        # Save reference frames (1-based: 1, 101, 201, 237, 259, 285 -> frame_idx + 1)
        f_num = frame_idx + 1
        if f_num in target_frames_1based:
            out_img_name = f"frame_{f_num:03d}_idx{f_num}.jpg"
            cv2.imwrite(str(out_dir / out_img_name), ann)
            print(f"Saved reference frame {f_num} to {out_dir / out_img_name}")
            
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  Processed {frame_idx}/{max_frames} frames...")
            
    cap.release()
    total_elapsed = time.perf_counter() - t_start
    total_fps = frame_idx / total_elapsed
    
    # Collect pipeline internal stage timings
    pipeline_stages = pipeline.stage_samples
    
    stage_table = {}
    for stage_name, samples in pipeline_stages.items():
        if samples:
            stage_table[stage_name] = float(np.mean(samples))
    stage_table["read"] = float(np.mean(stage_timings["read"])) if stage_timings["read"] else 0.0
    stage_table["annotate"] = float(np.mean(stage_timings["annotate"])) if stage_timings["annotate"] else 0.0
    stage_table["total_fps"] = total_fps
    
    trust_min = float(np.min(trust_vals))
    trust_mean = float(np.mean(trust_vals))
    trust_max = float(np.max(trust_vals))
    fallback_pct = (fallback_count / max(1, frame_idx)) * 100.0
    
    mean_consec_iou = float(np.mean(consecutive_mask_ious)) if consecutive_mask_ious else 1.0
    min_consec_iou = float(np.min(consecutive_mask_ious)) if consecutive_mask_ious else 1.0
    
    summary = {
        "frames_processed": frame_idx,
        "total_fps": total_fps,
        "trust": {
            "min": trust_min,
            "mean": trust_mean,
            "max": trust_max
        },
        "fallback_pct": fallback_pct,
        "consecutive_mask_iou": {
            "mean": mean_consec_iou,
            "min": min_consec_iou
        },
        "stage_latency_ms": stage_table
    }
    
    with open(out_dir / "video_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print("\n" + "=" * 60)
    print("=== VIDEO INFERENCE BENCHMARK RESULTS (300 FRAMES) ===")
    print("=" * 60)
    print(f"Total FPS:               {total_fps:.2f} FPS")
    print(f"Trust (min / mean / max): {trust_min:.3f} / {trust_mean:.3f} / {trust_max:.3f}")
    print(f"Fallback %:              {fallback_pct:.2f}%")
    print(f"Consecutive Mask IoU:    mean={mean_consec_iou:.4f}, min={min_consec_iou:.4f}")
    print("\nStage Latency Breakdown (ms):")
    for k, v in stage_table.items():
        if k != "total_fps":
            print(f"  {k:15s}: {v:6.2f} ms")
    print("=" * 60)
    
if __name__ == "__main__":
    main()
