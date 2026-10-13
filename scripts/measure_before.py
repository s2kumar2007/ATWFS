"""Capture BEFORE baseline metrics, per-stage timings, steering saturation, and 50 evenly-spaced masks."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.decision import compute_command, subtract_obstacles
from inference.detector import build_detector
from inference.live import LivePipeline, _open_writer, annotate
from training.train import load_model_from_checkpoint
from utils.config import load_config, resolve_path
from utils.seed import get_device, set_global_seed


def main() -> None:
    video_path = r"C:\Users\K\Downloads\WhatsApp Video 2026-10-12 at 6.18.52 AM.mp4"
    if not Path(video_path).exists():
        video_path = "dashcam_raw.mp4"

    cfg = load_config("config.yaml")
    set_global_seed(cfg["seed"])
    device = get_device()
    inf = cfg["inference"]
    ckpt = resolve_path(inf["checkpoint"])
    model, _ = load_model_from_checkpoint(str(ckpt), device)
    detector = build_detector(inf["detector"], cfg["paths"]["output_dir"], live_camera=False)
    pipeline = LivePipeline(cfg, model, device, detector)

    cap = cv2.VideoCapture(video_path)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # 1. Measure 60 sequential frames for per-stage before timing & stats
    num_measure = 60
    stage_times = {k: [] for k in ["read", "preprocess", "model", "classical", "YOLO", "IPM", "postprocess", "annotate", "write"]}
    frame_times = []
    angles = []
    trusts = []
    fallbacks = 0
    alpha_below_thr = 0
    obs_flags = 0
    det_counts = []
    steer_before_rows = []

    out_dir = Path("outputs/inference")
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = None

    for idx in range(num_measure):
        t_frame0 = time.perf_counter()
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] > 1280:
            scale = 1280.0 / frame.shape[1]
            frame = cv2.resize(frame, (1280, int(round(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        t_read = (time.perf_counter() - t0) * 1000.0

        # Snapshot cumulative timings before process_frame
        prev_t = dict(pipeline.timings)
        res = pipeline.process_frame(frame, frame_idx=idx)
        cur_t = dict(pipeline.timings)

        t_prep = (cur_t.get("preprocess", 0.0) - prev_t.get("preprocess", 0.0)) * 1000.0
        t_mod = (cur_t.get("model", 0.0) - prev_t.get("model", 0.0)) * 1000.0
        t_cf = (cur_t.get("classical_fusion", 0.0) - prev_t.get("classical_fusion", 0.0)) * 1000.0
        t_post = (cur_t.get("postprocess", 0.0) - prev_t.get("postprocess", 0.0)) * 1000.0
        t_yolo = (cur_t.get("yolo", 0.0) - prev_t.get("yolo", 0.0)) * 1000.0
        t_ipm = (cur_t.get("ipm", 0.0) - prev_t.get("ipm", 0.0)) * 1000.0

        t0 = time.perf_counter()
        ann = annotate(frame, res, skipped=False, fps=0.0, frame_idx=idx, total_frames=num_measure, cfg=cfg)
        t_ann = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        if writer is None:
            writer, _ = _open_writer(out_dir / "fps_before_sample.mp4", fps_video, (ann.shape[1], ann.shape[0]), inf["video_fourcc"])
        writer.write(ann)
        t_write = (time.perf_counter() - t0) * 1000.0

        t_total = (time.perf_counter() - t_frame0) * 1000.0
        frame_times.append(t_total)

        stage_times["read"].append(t_read)
        stage_times["preprocess"].append(t_prep)
        stage_times["model"].append(t_mod)
        stage_times["classical"].append(t_cf)
        stage_times["postprocess"].append(t_post)
        stage_times["YOLO"].append(t_yolo)
        stage_times["IPM"].append(t_ipm)
        stage_times["annotate"].append(t_ann)
        stage_times["write"].append(t_write)

        angles.append(res.command.angle)
        trusts.append(res.command.trust)
        if res.active == "FALLBACK":
            fallbacks += 1
        if res.command.trust < cfg["fusion"]["fallback_threshold"]:
            alpha_below_thr += 1
        if res.command.obstacle_flag:
            obs_flags += 1
        det_counts.append(len(res.detections))

        # Record why old steering saturated
        steer_before_rows.append({
            "frame": idx,
            "angle_deg": res.command.angle,
            "lateral_offset_m": res.command.offset_m,
            "road_width_m": res.command.width_m,
            "centroid_x": res.command.centroid_x,
        })

    if writer is not None:
        writer.release()

    # 2. Extract 50 evenly-spaced masks across the full video for Step 7 IoU guard
    even_indices = np.linspace(0, max(0, total_video_frames - 1), 50, dtype=int)
    before_masks = {}
    # Reset pipeline temporal history for clean mask extraction
    pipeline_masks = LivePipeline(cfg, model, device, build_detector({"backend": "null"}, cfg["paths"]["output_dir"]))
    for f_idx in even_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_idx))
        ok, frame = cap.read()
        if not ok:
            continue
        if frame.shape[1] > 1280:
            scale = 1280.0 / frame.shape[1]
            frame = cv2.resize(frame, (1280, int(round(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        # Compute raw segmentation mask before obstacle subtraction
        r = pipeline_masks.process_frame(frame, frame_idx=int(f_idx))
        before_masks[str(int(f_idx))] = r.mask.astype(np.uint8)
        angles.append(r.command.angle)

    cap.release()
    np.savez_compressed(out_dir / "before_masks_50.npz", **before_masks)

    stage_summary = {}
    for k, vals in stage_times.items():
        arr = np.array(vals, dtype=np.float64)
        stage_summary[k] = {
            "mean_ms": float(np.mean(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "total_s": float(np.sum(arr) / 1000.0),
        }

    ft_arr = np.array(frame_times, dtype=np.float64)
    fps_arr = 1000.0 / np.maximum(ft_arr, 1e-6)
    angles_arr = np.array(angles[:num_measure], dtype=np.float64)
    sat_pct = float(np.mean(np.abs(angles_arr) >= 29.5) * 100.0)

    summary = {
        "frames_measured": num_measure,
        "total_video_frames": total_video_frames,
        "mean_frame_ms": float(np.mean(ft_arr)),
        "p95_frame_ms": float(np.percentile(ft_arr, 95)),
        "avg_fps": float(1000.0 / np.mean(ft_arr)),
        "p95_fps": float(np.percentile(fps_arr, 5)),
        "stages": stage_summary,
        "fallback_pct": float(fallbacks / num_measure * 100.0),
        "alpha_below_thr_pct": float(alpha_below_thr / num_measure * 100.0),
        "trust_min": float(np.min(trusts)),
        "trust_mean": float(np.mean(trusts)),
        "trust_max": float(np.max(trusts)),
        "obstacle_flag_pct": float(obs_flags / num_measure * 100.0),
        "mean_detections": float(np.mean(det_counts)),
        "steer_saturation_pct_before": sat_pct,
        "steer_before_sample": steer_before_rows[:15],
    }

    (out_dir / "before_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
