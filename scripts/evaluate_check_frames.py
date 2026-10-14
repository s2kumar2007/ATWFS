"""Evaluate and export the 8 reference check frames and measure pipeline metrics."""
import collections
import json
import logging
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch

from utils.config import load_config, resolve_path
from inference.live import LivePipeline, annotate, _open_writer
from inference.detector import build_detector
from inference.hood_calibration import calibrate_or_load_hood_polygon

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval_check_frames")

def main():
    video_path = r"C:\Users\K\Downloads\WhatsApp Video 2026-10-12 at 6.18.52 AM.mp4"
    out_dir = Path("outputs/inference/check_frames_after")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    inf = cfg["inference"]

    # Dashboard phone assumptions
    cfg["camera"]["height_m"] = 1.3
    cfg["camera"]["hfov_deg"] = 75.0
    cfg["camera"]["vfov_deg"] = 45.0

    # Load or calibrate hood
    hood_poly = calibrate_or_load_hood_polygon(video_path)
    cfg["inference"]["hood_polygon"] = hood_poly

    # CPU profile settings
    prof_cfg = inf.get("profiles", {}).get("cpu", {})
    inf["detector"]["weights"] = prof_cfg.get("detector_weights", "yolov8n.pt")
    inf["detector"]["imgsz"] = prof_cfg.get("detector_imgsz", 480)
    inf["detector"]["every_n"] = prof_cfg.get("detector_every_n", 4)
    inf["detector"]["crop_every_n"] = 0
    inf["detector"]["crop_upscale"] = 1.0
    inf["fallback_alpha_gate"] = prof_cfg.get("fallback_alpha_gate", 0.5)
    inf["ipm_every_n"] = prof_cfg.get("ipm_every_n", 10)
    inf["seg_every"] = prof_cfg.get("seg_every", 1)
    if "input_size" in prof_cfg:
        inf["input_size"] = prof_cfg["input_size"]

    # ONNX model
    from scripts.export_onnx import export_to_onnx
    import onnxruntime
    ck_path = resolve_path(inf["checkpoint"])
    onnx_path = ck_path.with_suffix(".onnx")
    if not onnx_path.is_file() and ck_path.is_file():
        onnx_path = export_to_onnx(str(ck_path), str(onnx_path), tuple(inf.get("input_size", [288, 512])))

    opts = onnxruntime.SessionOptions()
    opts.intra_op_num_threads = 4
    onnx_session = onnxruntime.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])

    detector = build_detector(inf["detector"], cfg["paths"]["output_dir"], live_camera=False, device=torch.device("cpu"))
    pipeline = LivePipeline(cfg, None, torch.device("cpu"), detector, profile_name="cpu", onnx_session=onnx_session)

    target_indices = {
        0: "frame_01_idx0.jpg",
        242: "frame_02_idx242.jpg",
        484: "frame_03_idx484.jpg",
        726: "frame_04_idx726.jpg",
        968: "frame_05_idx968.jpg",
        1210: "frame_06_idx1210.jpg",
        1452: "frame_07_idx1452.jpg",
        1695: "frame_08_idx1695.jpg",
    }

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frame_idx = 0
    state = {}
    results_table = []

    horizon_success_count = 0
    horizon_total_count = 0

    fps_q = collections.deque(maxlen=30)
    fps_q.append(fps)
    t_last = time.perf_counter()

    max_target = max(target_indices.keys())

    print(f"Starting evaluation up to index {max_target} (total {total_frames} frames)...")

    while frame_idx <= max_target:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        t0 = time.perf_counter()
        res = pipeline.process_frame(frame, frame_idx=frame_idx, dt=1.0 / fps)
        t_now = time.perf_counter()
        fps_q.append(1.0 / max(1e-4, t_now - t_last))
        t_last = t_now
        rolling_fps = float(sum(fps_q) / len(fps_q))

        if pipeline.ipm.horizon_est.total_frames > horizon_total_count:
            horizon_total_count = pipeline.ipm.horizon_est.total_frames
            horizon_success_count = pipeline.ipm.horizon_est.success_count

        if frame_idx in target_indices:
            # Measure stats for check frame
            ann_frame = annotate(
                frame,
                res,
                skipped=False,
                frame_idx=frame_idx + 1,
                total_frames=total_frames,
                fps=rolling_fps,
                state=state,
                cfg=cfg,
            )
            out_file = out_dir / target_indices[frame_idx]
            cv2.imwrite(str(out_file), ann_frame)

            h, w = frame.shape[:2]
            # Bottom 15% mask analysis
            bottom_15_y = int(h * 0.85)
            bottom_15_area = (h - bottom_15_y) * w

            # Blue drivable mask in bottom 15%
            mask_bottom = res.mask[bottom_15_y:, :]
            blue_cov_px = int(np.count_nonzero(mask_bottom))
            blue_cov_pct = (blue_cov_px / float(bottom_15_area)) * 100.0
            covers_near = "yes" if blue_cov_pct > 1.0 else "no"

            # Ghost boxes check
            # Any box with conf < 0.25 or unupdated or out-of-frame or empty edge box?
            ghost_boxes = "no"
            for d in res.detections:
                age = frame_idx - pipeline.detector.last_real_detection_frame.get(d.tracker_id, -999)
                if age > 3 or d.conf < 0.25 or (d.x1 <= 0 and d.x2 < 30 and d.height > 200):
                    ghost_boxes = "yes"

            obs_flag_str = "T" if res.command.obstacle_flag else "F"
            speed_val = res.command.speed
            if res.command.width_m is not None and res.command.width_m >= 2.5 and getattr(res.ipm, "valid", False):
                width_str = f"{res.command.width_m:.2f} m"
            else:
                width_str = "n/a"

            item = {
                "idx": frame_idx,
                "file": target_indices[frame_idx],
                "covers_near": covers_near,
                "bottom_15_pct": round(blue_cov_pct, 1),
                "ghost_boxes": ghost_boxes,
                "obstacle_flag": obs_flag_str,
                "speed": round(speed_val, 2),
                "road_width": width_str,
                "num_dets": len(res.detections),
                "classes": [d.name for d in res.detections],
            }
            results_table.append(item)
            print(f"Frame {frame_idx:4d}: Near Cover={covers_near} ({blue_cov_pct:4.1f}%), Ghost={ghost_boxes}, Obs={obs_flag_str}, Speed={speed_val:.2f} m/s, Width={width_str}, Dets={len(res.detections)} {item['classes']}")

        frame_idx += 1

    cap.release()
    pipeline.detector.close()

    hz_rate = (horizon_success_count / max(1, horizon_total_count)) * 100.0
    print(f"\nHorizon estimation: {horizon_success_count}/{horizon_total_count} ({hz_rate:.1f}% raw success rate)")

    summary_json = out_dir / "check_frames_metrics.json"
    summary_json.write_text(json.dumps(results_table, indent=2), encoding="utf-8")
    print(f"Metrics saved to {summary_json}")

if __name__ == "__main__":
    main()
