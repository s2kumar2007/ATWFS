"""Live / offline video inference with a threaded producer-consumer pipeline.
"""
from __future__ import annotations

import os
import argparse
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import collections
import csv

import cv2
import numpy as np
import torch

from inference.classical import ClassicalFreeSpaceEstimator
from inference.consistency import geometric_score, temporal_score
from inference.decision import Command, compute_command, subtract_obstacles
from inference.detector import Detection, ObstacleDetector, build_detector
from inference.fusion import ATWFS
from models.unet_mobilenet import build_model, evidential_uncertainty
from training.train import load_model_from_checkpoint
from utils.config import load_config, resolve_path
from utils.logging_utils import setup_logging
from utils.seed import set_global_seed, get_device

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    mask: np.ndarray
    command: Command
    active: str
    uncertainty: float
    geometric: float
    temporal: float
    terrain: int
    detections: List[Detection]


class LivePipeline:
    def __init__(self, cfg: Dict[str, Any], model: torch.nn.Module, device: torch.device,
                 detector: ObstacleDetector, fusion: Optional[ATWFS] = None) -> None:
        self.cfg = cfg
        self.device = device
        self.model = model.to(device).eval()
        if torch.cuda.is_available():
            self.model = self.model.half() # fp16
            
        self.detector = detector
        self.size = tuple(cfg["inference"]["input_size"])  # (H, W) = (288, 512)
        self.mean = np.asarray(cfg["data"]["imagenet_mean"], np.float32)
        self.std = np.asarray(cfg["data"]["imagenet_std"], np.float32)
        self.classical = ClassicalFreeSpaceEstimator(cfg["classical"])
        self.fusion = fusion or ATWFS(cfg["fusion"])
        self.prev_dl_mask: Optional[np.ndarray] = None
        self._logged_unc_fallback = False
        
        # Timing stats
        self.timings = collections.defaultdict(float)

    @torch.inference_mode()
    def process_frame(self, frame_bgr: np.ndarray, speed_mps: Optional[float] = None) -> FrameResult:
        cfg = self.cfg
        speed = cfg["inference"]["speed_mps_input"] if speed_mps is None else speed_mps
        fh, fw = frame_bgr.shape[:2]
        mh, mw = self.size
        
        # Letterbox (pad, don't stretch)
        scale = min(mw / fw, mh / fh)
        nw, nh = int(fw * scale), int(fh * scale)
        small = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        padded = np.zeros((mh, mw, 3), dtype=np.uint8)
        
        pad_top = (mh - nh) // 2
        pad_left = (mw - nw) // 2
        padded[pad_top:pad_top + nh, pad_left:pad_left + nw] = small
        
        # ROI / Hood
        roi_pts = np.array(cfg["inference"].get("roi_polygon", [[0.0, 1.0], [0.2, 0.6], [0.8, 0.6], [1.0, 1.0]]))
        roi_px = np.zeros_like(roi_pts, dtype=np.int32)
        roi_px[:, 0] = roi_pts[:, 0] * mw
        roi_px[:, 1] = roi_pts[:, 1] * mh
        cv2.fillPoly(padded, [roi_px], (0, 0, 0))
        
        t0 = time.time()
        x = (cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - self.mean) / self.std
        x_tensor = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        if torch.cuda.is_available():
            x_tensor = x_tensor.half()
        self.timings["preprocess"] += time.time() - t0
        
        t0 = time.time()
        out = self.model(x_tensor)
        dl_mask_padded = out["seg_logits"].argmax(1)[0].byte().cpu().numpy()
        unc = float(evidential_uncertainty(out["alpha"]).mean().item())
        
        if unc < 0.01:
            if not getattr(self, "_logged_unc_fallback", False):
                logger.info("Evidential uncertainty is ~0, falling back to normalized entropy.")
                self._logged_unc_fallback = True
            probs = torch.softmax(out["seg_logits"], dim=1)
            unc = float((-(probs * torch.log(probs + 1e-6)).sum(dim=1) / np.log(probs.shape[1])).mean().cpu().item())
            
        if cfg["inference"].get("use_terrain", False):
            terrain = int(out["terrain_logits"].argmax(1)[0].item())
        else:
            terrain = 0
        self.timings["model"] += time.time() - t0
        
        # Un-letterbox mask
        dl_mask = np.zeros((nh, nw), dtype=np.uint8)
        dl_mask = dl_mask_padded[pad_top:pad_top + nh, pad_left:pad_left + nw]

        t0 = time.time()
        geo = geometric_score(dl_mask, cfg["consistency"]["geometric"])
        tmp = temporal_score(self.prev_dl_mask, dl_mask, speed, cfg["consistency"]["temporal"])
        self.prev_dl_mask = dl_mask.copy()
        
        # Classical fallback only if DL might fail
        alpha = self.fusion.alpha(unc, geo, tmp, terrain)
        if alpha < 0.8:
            fb_mask_padded = self.classical.estimate(padded)
            fb_mask = fb_mask_padded[pad_top:pad_top + nh, pad_left:pad_left + nw]
        else:
            fb_mask = np.zeros_like(dl_mask)
            
        fused = self.fusion.fuse(dl_mask, fb_mask, unc, geo, tmp, terrain)
        mask = fused.mask
        self.timings["classical_fusion"] += time.time() - t0
        
        t0 = time.time()
        # Post-process at model resolution
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Largest connected component nearest to bottom-center of ROI
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            # target point: bottom center of frame
            target_pt = np.array([nw / 2.0, nh])
            best_label = 1
            min_dist = float('inf')
            
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < 100:
                    continue
                dist = np.linalg.norm(centroids[i] - target_pt)
                if dist < min_dist:
                    min_dist = dist
                    best_label = i
            if min_dist < float('inf'):
                mask = (labels == best_label).astype(np.uint8)
            else:
                mask = np.zeros_like(mask)
        self.timings["postprocess"] += time.time() - t0

        t0 = time.time()
        dets = self.detector.detect(frame_bgr)
        self.timings["yolo"] += time.time() - t0

        # Upscale mask for decision & drawing
        mask = cv2.resize(mask, (fw, fh), interpolation=cv2.INTER_NEAREST)
        raw_mask = mask.copy()
        
        # Obstacle subtraction
        mask = subtract_obstacles(mask, dets, cfg["inference"]["decision"]["obstacle_dilate_px"])
        cmd = compute_command(mask, dets, fused.alpha, terrain, cfg["inference"]["decision"], raw_mask=raw_mask)
        
        return FrameResult(mask, cmd, fused.active, unc, geo, tmp, terrain, dets)


def annotate(frame_bgr: np.ndarray, res: FrameResult, skipped: bool = False,
             frame_idx: int = 0, total_frames: Optional[int] = None, fps: float = 0.0,
             state: Optional[Dict[str, Any]] = None, cfg: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if state is None: state = {}
    if cfg is None: cfg = {}
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    
    s = max(w / 1280.0, 0.5)
    thickness = max(1, int(round(2 * s)))
    font_scale = min(0.7 * s, 0.55)
    
    overlay_cfg = cfg.get("inference", {}).get("overlay", {})
    ema_x = overlay_cfg.get("ema_x", 0.35)
    show_mask = overlay_cfg.get("show_mask", False)

    # Draw mask
    overlay = out.copy()
    overlay[res.mask > 0] = [0, 200, 0]
    cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)
    
    contours, _ = cv2.findContours(res.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 0), thickness, cv2.LINE_AA)
    
    if show_mask:
        cv2.drawContours(out, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)

    # Center path
    band = cfg.get("inference", {}).get("decision", {}).get("lookahead_rows", [0.6, 0.9])
    y_start, y_end = int(band[0] * h), int(band[1] * h)
    
    if res.mask[y_start:y_end].any():
        c_pts = []
        for y in range(y_start, y_end, max(1, (y_end - y_start) // 10)):
            row = res.mask[y]
            nz = np.nonzero(row)[0]
            if len(nz) > 0:
                cx = np.mean(nz)
                c_pts.append((cx, y))
        
        if len(c_pts) >= 2:
            pts = np.array(c_pts, dtype=np.float32)
            if "ema_pts" not in state or state["ema_pts"].shape != pts.shape:
                state["ema_pts"] = pts
            else:
                state["ema_pts"] = ema_x * pts + (1 - ema_x) * state["ema_pts"]
            
            p = state["ema_pts"].astype(np.int32)
            cv2.polylines(out, [p], False, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.polylines(out, [p], False, (0, 255, 255), thickness, cv2.LINE_AA)
            cv2.arrowedLine(out, tuple(p[-2]), tuple(p[-1]), (0, 0, 0), thickness + 2, cv2.LINE_AA, tipLength=0.2)
            cv2.arrowedLine(out, tuple(p[-2]), tuple(p[-1]), (0, 255, 255), thickness, cv2.LINE_AA, tipLength=0.2)
    else:
        if "ema_pts" in state:
            del state["ema_pts"]

    # Detections
    for d in res.detections:
        if d.y2 - d.y1 < 0.03 * h:
            cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), (100, 100, 100), 1)
            continue
            
        color = (0, 165, 255) if d.approach_rate > 1.05 else (0, 0, 255)
        cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), color, thickness)
        
        label = f"{d.name} {d.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        lbl_y = max(th + 5, d.y1 - 5)
        cv2.rectangle(out, (d.x1, lbl_y - th - 3), (d.x1 + tw, lbl_y + 3), (0, 0, 0), -1)
        cv2.putText(out, label, (d.x1, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    frames_str = f"{frame_idx}/{total_frames}" if total_frames else f"{frame_idx}"
    ang = res.command.angle
    curve = "STRAIGHT" if abs(ang) < 5 else ("RIGHT SHARP" if ang > 15 else ("RIGHT" if ang > 5 else ("LEFT SHARP" if ang < -15 else "LEFT")))
    
    lines = [
        f"ATWFS DRIVABLE REGION | FRAME: {frames_str} | {fps:.1f} FPS",
        f"ESTIMATOR: {res.active}   TRUST: {res.command.trust:.2f}",
        f"STEER: {ang:+.1f} deg {curve}   SPEED: {res.command.speed:.2f} m/s",
        f"OBSTACLE: {'YES' if res.command.obstacle_flag else 'NO'}",
        f"ROAD WIDTH: {res.command.width_frac * 100.0:.1f}%",
        "LEGEND: Green=Corridor, Yellow=Path"
    ]
    
    y = int(25 * s)
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(out, (int(15*s) - 2, y - th - 3), (int(15*s) + tw + 2, y + 3), (0, 0, 0), -1)
        cv2.putText(out, line, (int(15*s), y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += int(30 * s)

    return out

class FrameGrabber(threading.Thread):
    def __init__(self, source: Union[int, str], out_q: "queue.Queue", stop: threading.Event, live: bool,
                 max_frames: Optional[int] = None, max_width: Optional[int] = None) -> None:
        super().__init__()
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open {source}")
        self.out_q = out_q
        self.stop = stop
        self.live = live
        self.max_frames = max_frames
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not live else None
        self.max_width = max_width

    def run(self) -> None:
        count = 0
        while not self.stop.is_set():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if self.max_width and frame.shape[1] > self.max_width:
                scale = self.max_width / frame.shape[1]
                frame = cv2.resize(frame, (self.max_width, int(frame.shape[0] * scale)))

            if self.live:
                if self.out_q.full():
                    try:
                        self.out_q.get_nowait()
                    except queue.Empty:
                        pass
                self.out_q.put(frame, timeout=1.0)
            else:
                self.out_q.put(frame)
            count += 1
            if self.max_frames and count >= self.max_frames:
                break
        self.out_q.put(None)
        self.cap.release()

class InferenceWorker(threading.Thread):
    def __init__(self, pipeline: LivePipeline, in_q: "queue.Queue", out_q: "queue.Queue",
                 stop: threading.Event, skip: int, errors: List[BaseException], total_frames: Optional[int]) -> None:
        super().__init__()
        self.p, self.in_q, self.out_q, self.stop, self.skip = pipeline, in_q, out_q, stop, max(1, skip)
        self.errors = errors
        self.annot_state: Dict[str, Any] = {}
        self.total_frames = total_frames

    def run(self) -> None:
        idx, last_frame, last_res, last_annot = 0, None, None, None
        fps_q = collections.deque(maxlen=30)
        t_last = time.time()
        
        try:
            while not self.stop.is_set():
                t_read0 = time.time()
                frame = self.in_q.get()
                self.p.timings["read"] += time.time() - t_read0
                
                if frame is None:
                    break
                
                t_start = time.time()
                is_skip = (idx % self.skip != 0)
                if not is_skip or last_res is None:
                    res = self.p.process_frame(frame)
                    last_res = res
                else:
                    res = last_res
                
                t_annot0 = time.time()
                fps = sum(fps_q) / len(fps_q) if fps_q else 0.0
                annotated = annotate(
                    frame, res, skipped=is_skip,
                    frame_idx=idx + 1, total_frames=self.total_frames,
                    fps=fps, state=self.annot_state, cfg=self.p.cfg
                )
                self.p.timings["annotate"] += time.time() - t_annot0
                
                self._put((idx, annotated, res.command.packet, res))
                idx += 1
                
                t_now = time.time()
                fps_q.append(1.0 / max(1e-4, t_now - t_last))
                t_last = t_now
        except BaseException as e:
            logger.exception("Inference thread failed")
            self.errors.append(e)
            self.stop.set()
        finally:
            while True:
                try:
                    self.out_q.put(None, timeout=0.05)
                    break
                except queue.Full:
                    try:
                        self.out_q.get_nowait()
                    except queue.Empty:
                        pass
                        
    def _put(self, item: Any) -> None:
        while not self.stop.is_set():
            try:
                self.out_q.put(item, timeout=0.1)
                break
            except queue.Full:
                pass


def _open_writer(path: Path, fps: float, size_wh: Tuple[int, int], fourcc: str) -> Tuple[cv2.VideoWriter, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size_wh)
    if wr.isOpened():
        return wr, path
    alt = path.with_suffix(".avi")
    logger.warning("Codec '%s' unavailable, falling back to MJPG %s", fourcc, alt)
    wr = cv2.VideoWriter(str(alt), cv2.VideoWriter_fourcc(*"MJPG"), fps, size_wh)
    if not wr.isOpened():
        raise RuntimeError("Could not open any video writer")
    return wr, alt


def run_video(cfg: Dict[str, Any], source: Union[int, str], output_path: str, model: Optional[torch.nn.Module] = None,
              checkpoint: Optional[str] = None, detector: Optional[ObstacleDetector] = None,
              max_frames: Optional[int] = None, display: Optional[bool] = None, max_width: Optional[int] = 1280,
              log_features: Optional[str] = None, speed_mps: Optional[float] = None) -> Dict[str, Any]:
    set_global_seed(cfg["seed"])
    device = get_device()
    inf = cfg["inference"]
    
    if speed_mps is not None:
        cfg["inference"]["speed_mps_input"] = speed_mps
        
    if model is None:
        ck = Path(checkpoint) if checkpoint else resolve_path(inf["checkpoint"])
        if ck.exists():
            model, _ = load_model_from_checkpoint(str(ck), device)
        else:
            model = build_model({**cfg["model"], "pretrained": False})
            
    detector = detector or build_detector(inf["detector"], cfg["paths"]["output_dir"])
    pipeline = LivePipeline(cfg, model, device, detector)
    
    live = isinstance(source, int) or str(source).lower().startswith(("rtsp://", "http://", "https://"))
    stop, errors = threading.Event(), []
    frame_q, result_q = queue.Queue(inf["queue_size"]), queue.Queue(inf["queue_size"])
    grabber = FrameGrabber(source, frame_q, stop, live, max_frames, max_width)
    worker = InferenceWorker(pipeline, frame_q, result_q, stop, inf["frame_skip"], errors, grabber.total_frames)
    
    grabber.start(); worker.start()
    display = inf["display"] if display is None else display
    writer, out_path, packets = None, Path(output_path), []
    
    csv_f = None
    csv_writer = None
    if log_features:
        os.makedirs(os.path.dirname(log_features) or ".", exist_ok=True)
        csv_f = open(log_features, "w", newline="")
        csv_writer = csv.writer(csv_f)
        csv_writer.writerow(["U", "G", "T", "alpha", "active", "speed", "obstacle_flag"])
    
    t_start_total = time.time()
    frames_processed = 0
    
    try:
        while True:
            item = None
            try:
                item = result_q.get(timeout=0.1)
            except queue.Empty:
                if stop.is_set() and not worker.is_alive():
                    break
                continue
            if item is None:
                break
            idx, frame, pkt, res = item
            
            t_w0 = time.time()
            if writer is None:
                writer, out_path = _open_writer(out_path, grabber.fps, (frame.shape[1], frame.shape[0]), inf["video_fourcc"])
            writer.write(frame)
            pipeline.timings["write"] += time.time() - t_w0
            
            packets.append(pkt)
            print(f"[{pkt[0]:.2f}, {pkt[1]:.3f}, {pkt[2]:.3f}, {pkt[3]}]", flush=True)
            if csv_writer and res is not None:
                csv_writer.writerow([res.uncertainty, res.geometric, res.temporal, getattr(res.command, "trust", 0.0), res.active, res.command.speed, res.command.obstacle_flag])
    except KeyboardInterrupt:
        stop.set()
    finally:
        stop.set()
        grabber.join(timeout=5); worker.join(timeout=5)
        if writer is not None:
            writer.release()
        if hasattr(detector, 'stop'):
            detector.stop()
        if csv_f:
            csv_f.close()
            
    if errors:
        raise RuntimeError(f"Inference thread crashed: {errors[0]!r}") from errors[0]

    # FFmpeg re-encoding
    if out_path.exists() and writer is not None:
        import shutil, subprocess
        ffmpeg_exe = shutil.which("ffmpeg")
        if not ffmpeg_exe:
            try:
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                pass
        if ffmpeg_exe:
            temp_out = out_path.with_suffix(".tmp.mp4")
            cmd = [ffmpeg_exe, "-y", "-i", str(out_path), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "ultrafast", "-movflags", "+faststart", str(temp_out)]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                os.replace(str(temp_out), str(out_path))
            except Exception:
                if temp_out.exists(): temp_out.unlink()

    total_time = time.time() - t_start_total
    logger.info("Average FPS: %.2f", len(packets) / total_time if total_time > 0 else 0)
    logger.info("Per-stage timings (seconds total):")
    for k, v in pipeline.timings.items():
        logger.info(f"  {k}: {v:.2f}")

    return {"output_path": str(out_path), "frames": len(packets), "packets": packets}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", default="./outputs/inference/annotated.mp4")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--frame-skip", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--display", action="store_true")
    ap.add_argument("--max-width", type=int, default=1280)
    ap.add_argument("--log-features", default=None)
    ap.add_argument("--speed-mps", type=float, default=None)
    a = ap.parse_args()
    
    setup_logging()
    cfg = load_config(a.config)
    if a.frame_skip:
        cfg["inference"]["frame_skip"] = a.frame_skip
    src = int(a.source) if a.source.isdigit() else a.source
    run_video(cfg, src, str(resolve_path(a.output)), checkpoint=a.checkpoint, max_frames=a.max_frames, display=a.display or None, max_width=a.max_width, log_features=a.log_features, speed_mps=a.speed_mps)

if __name__ == "__main__":
    main()
