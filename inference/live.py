import os
import argparse
import logging
import queue
import threading
import time
from dataclasses import dataclass
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
from inference.ipm import IMUReader, InversePerspectiveMapping, IPMResult
from inference.serial_out import SerialSender
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
    ipm: Optional[IPMResult] = None
    mode: str = "day"


class LivePipeline:
    def __init__(self, cfg: Dict[str, Any], model: torch.nn.Module, device: torch.device,
                 detector: ObstacleDetector, fusion: Optional[ATWFS] = None) -> None:
        self.cfg = cfg
        self.device = device
        self.model = model.to(device).eval()
        if self.device.type == "cuda":
            self.model = self.model.half()
            
        self.detector = detector
        self.size = tuple(cfg["inference"]["input_size"])  # (H, W) = (288, 512)
        self.mean = np.asarray(cfg["data"]["imagenet_mean"], np.float32)
        self.std = np.asarray(cfg["data"]["imagenet_std"], np.float32)
        self.classical = ClassicalFreeSpaceEstimator(cfg["classical"])
        self.fusion = fusion or ATWFS(cfg["fusion"])
        self.prev_dl_mask: Optional[np.ndarray] = None
        self._logged_unc_fallback = False
        
        self.mask_history = collections.deque(maxlen=5)
        
        # Timing stats
        self.timings = collections.defaultdict(float)
        
        # Hood polygon setup
        self.hood_poly_norm = np.array(cfg["inference"].get("hood_polygon", [[0.0, 1.0], [0.1, 0.75], [0.9, 0.75], [1.0, 1.0]]))
        self.hood_poly_model = None
        self.ipm = InversePerspectiveMapping(cfg)
        
        # Night mode state & hysteresis debounce
        night_cfg = cfg["inference"].get("night_mode", {})
        self.night_mode_enabled = bool(night_cfg.get("enabled", True))
        self.night_thresh_low = float(night_cfg.get("threshold_low", 45.0))
        self.night_thresh_high = float(night_cfg.get("threshold_high", 60.0))
        self.night_debounce_frames = int(night_cfg.get("debounce_frames", 10))
        self.current_mode = "day"
        self.night_debounce_counter = 0
        if hasattr(self.model, "add_night_decoder"):
            self.model.add_night_decoder()

    @torch.inference_mode()
    def process_frame(self, frame_bgr: np.ndarray, speed_mps: Optional[float] = None, frame_idx: int = 0,
                      imu_sample: Optional[Dict[str, float]] = None, dt: float = 0.04) -> FrameResult:
        cfg = self.cfg
        speed = cfg["inference"]["speed_mps_input"] if speed_mps is None else speed_mps
        fh, fw = frame_bgr.shape[:2]
        mh, mw = self.size
        
        # Letterbox
        scale = min(mw / fw, mh / fh)
        nw, nh = int(fw * scale), int(fh * scale)
        small = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        padded = np.zeros((mh, mw, 3), dtype=np.uint8)
        
        pad_top = (mh - nh) // 2
        pad_left = (mw - nw) // 2
        padded[pad_top:pad_top + nh, pad_left:pad_left + nw] = small
        
        # Hood masking on model input
        if self.hood_poly_model is None:
            self.hood_poly_model = np.zeros_like(self.hood_poly_norm, dtype=np.int32)
            self.hood_poly_model[:, 0] = self.hood_poly_norm[:, 0] * mw
            self.hood_poly_model[:, 1] = self.hood_poly_norm[:, 1] * mh
            
        # Sample patch just above hood for mean road color
        # Get hood apex
        apex_y = np.min(self.hood_poly_model[:, 1])
        patch_y1, patch_y2 = max(0, int(apex_y - 20)), int(apex_y)
        patch_x1, patch_x2 = int(mw/2 - 20), int(mw/2 + 20)
        
        road_color = (128, 128, 128)
        if patch_y2 > patch_y1:
            patch = padded[patch_y1:patch_y2, patch_x1:patch_x2]
            if patch.size > 0:
                road_color = patch.mean(axis=(0, 1)).astype(int).tolist()
                
        cv2.fillPoly(padded, [self.hood_poly_model], road_color)
        
        # Low-light / night mode detection via road region mean luminance
        if self.night_mode_enabled:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            if self.prev_dl_mask is not None and np.count_nonzero(self.prev_dl_mask) > 50:
                m_resized = cv2.resize(self.prev_dl_mask, (fw, fh), interpolation=cv2.INTER_NEAREST)
                road_luma = float(np.mean(gray[m_resized > 0]))
            else:
                road_luma = float(np.mean(gray[int(fh * 0.50):int(fh * 0.70), int(fw * 0.30):int(fw * 0.70)]))
            
            # Hysteresis + 10-frame debounce
            if self.current_mode == "day":
                if road_luma < self.night_thresh_low:
                    self.night_debounce_counter += 1
                    if self.night_debounce_counter >= self.night_debounce_frames:
                        self.current_mode = "night"
                        self.night_debounce_counter = 0
                else:
                    self.night_debounce_counter = 0
            else:
                if road_luma > self.night_thresh_high:
                    self.night_debounce_counter += 1
                    if self.night_debounce_counter >= self.night_debounce_frames:
                        self.current_mode = "day"
                        self.night_debounce_counter = 0
                else:
                    self.night_debounce_counter = 0
            
            if hasattr(self.model, "set_mode"):
                self.model.set_mode(self.current_mode)

        t0 = time.time()
        x = (cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - self.mean) / self.std
        x_tensor = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        if self.device.type == "cuda":
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
        
        dl_mask = dl_mask_padded[pad_top:pad_top + nh, pad_left:pad_left + nw]

        t0 = time.time()
        
        # Ego seed check: sample just above hood
        geo = geometric_score(dl_mask, cfg["consistency"]["geometric"], apex_y=int(apex_y - pad_top))
        

        tmp = temporal_score(self.prev_dl_mask, dl_mask, speed, cfg["consistency"]["temporal"])
        self.prev_dl_mask = dl_mask.copy()
        
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
        # Post-process
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Mask out hood poly on OUTPUT mask at model resolution
        hp_model_adj = self.hood_poly_model.copy()
        hp_model_adj[:, 0] -= pad_left
        hp_model_adj[:, 1] -= pad_top
        cv2.fillPoly(mask, [hp_model_adj], 0)
        
        # Largest connected component nearest bottom center
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            target_pt = np.array([nw / 2.0, nh])
            best_label = 1
            min_dist = float('inf')
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < 100: continue
                dist = np.linalg.norm(centroids[i] - target_pt)
                if dist < min_dist:
                    min_dist = dist
                    best_label = i
            if min_dist < float('inf'):
                mask = (labels == best_label).astype(np.uint8)
            else:
                mask = np.zeros_like(mask)
                
        # Temporal smoothing (majority vote)
        self.mask_history.append(mask)
        if len(self.mask_history) > 2:
            sum_mask = np.sum(self.mask_history, axis=0)
            mask = (sum_mask > len(self.mask_history) // 2).astype(np.uint8)
            
        self.timings["postprocess"] += time.time() - t0

        t0 = time.time()
        dets = self.detector.detect(frame_bgr, frame_idx)
        
        # Ignore detections inside hood
        hp_fw_fh = np.zeros_like(self.hood_poly_norm, dtype=np.int32)
        hp_fw_fh[:, 0] = self.hood_poly_norm[:, 0] * fw
        hp_fw_fh[:, 1] = self.hood_poly_norm[:, 1] * fh
        
        valid_dets = []
        for d in dets:
            if cv2.pointPolygonTest(hp_fw_fh, (d.center_x, d.center_y), False) < 0:
                valid_dets.append(d)
        dets = valid_dets
        self.timings["yolo"] += time.time() - t0

        mask = cv2.resize(mask, (fw, fh), interpolation=cv2.INTER_NEAREST)
        raw_mask = mask.copy()
        
        # Self-calibrating IPM (bird's-eye view)
        t_ipm0 = time.time()
        ipm_res = self.ipm.process_frame(frame_bgr, mask, dt=dt, imu_sample=imu_sample)
        self.timings["ipm"] += time.time() - t_ipm0

        # Create hood map for decision logic (don't compute path in hood)
        # Decision.py logic already takes topmost mask, which is 0 in hood area.
        mask = subtract_obstacles(mask, dets, cfg["inference"]["decision"]["obstacle_dilate_px"])
        cmd = compute_command(
            mask, dets, fused.alpha, terrain, cfg["inference"]["decision"],
            raw_mask=raw_mask,
            road_width_m=ipm_res.road_width_m,
            lateral_offset_m=ipm_res.lateral_offset_m
        )
        
        return FrameResult(mask, cmd, fused.active, unc, geo, tmp, terrain, dets, ipm=ipm_res, mode=self.current_mode)

def is_overlap(r1, r2):
    return not (r1[0] >= r2[2] or r1[2] <= r2[0] or r1[1] >= r2[3] or r1[3] <= r2[1])

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
    
    colors = cfg.get("inference", {}).get("colors", {})
    def get_color(name, default):
        c = colors.get(name, default)
        return (c[2], c[1], c[0]) # RGB to BGR
        
    c_drivable = get_color("drivable", [30, 144, 255])
    c_hood = get_color("hood", [40, 40, 40])
    c_veh_def = get_color("vehicle_default", [0, 255, 255])
    c_veh_app = get_color("vehicle_approach", [255, 165, 0])
    c_veh_obs = get_color("vehicle_obstacle", [255, 0, 0])
    c_person = get_color("person", [255, 0, 255])
    c_animal = get_color("animal", [128, 0, 128])
    c_tiny = get_color("tiny", [128, 128, 128])

    # Draw drivable mask
    overlay = out.copy()
    overlay[res.mask > 0] = c_drivable
    cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)
    
    contours, _ = cv2.findContours(res.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 255, 255), thickness, cv2.LINE_AA)
    
    # Draw hood
    hood_poly = np.array(cfg["inference"].get("hood_polygon", [[0.0, 1.0], [0.1, 0.75], [0.9, 0.75], [1.0, 1.0]]))
    hp_px = np.zeros_like(hood_poly, dtype=np.int32)
    hp_px[:, 0] = hood_poly[:, 0] * w
    hp_px[:, 1] = hood_poly[:, 1] * h
    overlay_hood = out.copy()
    cv2.fillPoly(overlay_hood, [hp_px], c_hood)
    cv2.addWeighted(overlay_hood, 0.5, out, 0.5, 0, out)

    # Center path
    band = cfg.get("inference", {}).get("decision", {}).get("lookahead_rows", [0.55, 0.9])
    y_start, y_end = int(band[0] * h), int(band[1] * h)
    
    # Find topmost hood y at center to restrict path
    apex_y = int(np.min(hp_px[:, 1]))
    if y_end > apex_y:
        y_end = apex_y
        
    if y_end > y_start and res.mask[y_start:y_end].any():
        c_pts = []
        for y in range(y_start, y_end, max(1, (y_end - y_start) // 10)):
            row = res.mask[y]
            nz = np.nonzero(row)[0]
            if len(nz) > 0:
                cx = np.mean(nz)
                c_pts.append((cx, y))
        
        if len(c_pts) >= 2:
            pts = np.array(c_pts, dtype=np.float32)
            ema_x = cfg.get("inference", {}).get("overlay", {}).get("ema_x", 0.35)
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
        if "ema_pts" in state: del state["ema_pts"]

    placed_rects = []
    
    # Draw Detections
    for d in res.detections:
        if d.height < 0.03 * h:
            cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), c_tiny, 1)
            continue
            
        color = c_veh_def
        if d.kind == "person": color = c_person
        elif d.kind == "animal": color = c_animal
        else:
            if res.command.obstacle_flag and res.command.obstacle_proximity > 0 and d.approach_rate > 1.0: 
                # simplification: if it's the obstacle causing the flag
                color = c_veh_obs
            elif d.approach_rate > 1.05 or abs(d.center_x - w/2) < cfg.get("inference", {}).get("decision", {}).get("corridor_half_width", 0.2) * w:
                color = c_veh_app
                
        cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), color, thickness)
        
        label = f"{d.name} {d.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Label placement without overlap
        lbl_x, lbl_y = d.x1, d.y1 - 5
        rect = [lbl_x, lbl_y - th - 3, lbl_x + tw, lbl_y + 3]
        
        # HUD area (top left 350x250 approx)
        hud_rect = [0, 0, int(350 * s), int(250 * s)]
        
        attempts = 0
        while (any(is_overlap(rect, pr) for pr in placed_rects) or is_overlap(rect, hud_rect)) and attempts < 10:
            attempts += 1
            # Move up
            rect[1] -= th + 5
            rect[3] -= th + 5
            lbl_y -= th + 5
            if rect[1] < 0:
                # Move below
                lbl_y = d.y2 + th + 5
                rect = [lbl_x, lbl_y - th - 3, lbl_x + tw, lbl_y + 3]
                if any(is_overlap(rect, pr) for pr in placed_rects) or is_overlap(rect, hud_rect):
                    # Move right
                    shift_dx = max(30, int(tw + 10))
                    rect[0] += shift_dx
                    rect[2] += shift_dx
                    lbl_x += shift_dx
        
        placed_rects.append(rect)
        cv2.rectangle(out, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 0), -1)
        cv2.putText(out, label, (lbl_x, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    frames_str = f"{frame_idx}/{total_frames}" if total_frames else f"{frame_idx}"
    ang = res.command.angle
    curve = "STRAIGHT" if abs(ang) < 5 else ("RIGHT SHARP" if ang > 15 else ("RIGHT" if ang > 5 else ("LEFT SHARP" if ang < -15 else "LEFT")))
    
    width_text = f"ROAD WIDTH: {res.command.width_m:.2f} m" if res.command.width_m is not None else f"ROAD WIDTH: {res.command.width_frac * 100.0:.1f}%"
    
    lines = [
        f"ATWFS DRIVABLE REGION | FRAME: {frames_str} | {fps:.1f} FPS",
        f"ESTIMATOR: {res.active}   TRUST: {res.command.trust:.2f}   MODE: {res.mode.upper()}",
        f"STEER: {ang:+.1f} deg {curve}   SPEED: {res.command.speed:.2f} m/s",
        f"OBSTACLE: {'YES' if res.command.obstacle_flag else 'NO'}",
        width_text,
    ]
    if res.ipm and res.ipm.valid:
        lines.append(f"IPM: {res.ipm.method.upper()} (P:{res.ipm.pitch_deg:+.1f} deg R:{res.ipm.roll_deg:+.1f} deg)")
    lines.append("LEGEND: Blu=Drivable Yel=Path Cy=Veh Org=App/In-corr Red=Obs")
    
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
            if count == 0:
                print(f"First-frame resolution: {frame.shape[1]}x{frame.shape[0]}")
            
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
                 stop: threading.Event, skip: int, errors: List[BaseException], total_frames: Optional[int],
                 imu_reader: Optional[IMUReader] = None) -> None:
        super().__init__()
        self.p, self.in_q, self.out_q, self.stop, self.skip = pipeline, in_q, out_q, stop, max(1, skip)
        self.errors = errors
        self.annot_state: Dict[str, Any] = {}
        self.total_frames = total_frames
        self.imu_reader = imu_reader

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
                    imu_sample = self.imu_reader.get_sample(idx) if self.imu_reader else None
                    res = self.p.process_frame(frame, frame_idx=idx, imu_sample=imu_sample)
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
                
                pkt_tuple = (res.command.angle, res.command.speed, res.command.trust, int(res.command.obstacle_flag))
                self._put((idx, annotated, pkt_tuple, res))
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
              log_features: Optional[str] = None, speed_mps: Optional[float] = None,
              force: bool = False, imu_csv: Optional[str] = None,
              serial_port: Optional[str] = None, serial_baud: int = 115200,
              serial_dry_run: bool = False) -> Dict[str, Any]:
    
    # Check safeguard on BASENAME only
    if isinstance(source, str) and not force:
        base_lower = Path(source).name.lower()
        if "result" in base_lower or "annotated" in base_lower:
            err_msg = f"ABORT: Source video basename '{Path(source).name}' contains 'result' or 'annotated'. ALWAYS test on the ORIGINAL RAW dashcam video. Use --force to override."
            logger.error(err_msg)
            return {"error": err_msg, "frames": 0, "packets": []}
              
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
            
    live = isinstance(source, int) or str(source).lower().startswith(("rtsp://", "http://", "https://"))
    detector = detector or build_detector(inf["detector"], cfg["paths"]["output_dir"], live_camera=live)
    pipeline = LivePipeline(cfg, model, device, detector)
    
    stop, errors = threading.Event(), []
    frame_q, result_q = queue.Queue(inf["queue_size"]), queue.Queue(inf["queue_size"])
    grabber = FrameGrabber(source, frame_q, stop, live, max_frames, max_width)
    imu_reader = IMUReader(imu_csv) if imu_csv else None
    worker = InferenceWorker(pipeline, frame_q, result_q, stop, inf["frame_skip"], errors, grabber.total_frames, imu_reader=imu_reader)
    
    grabber.start(); worker.start()
    display = inf["display"] if display is None else display
    writer, out_path, packets = None, Path(output_path), []
    
    serial_sender = SerialSender(port=serial_port, baud=serial_baud, dry_run=serial_dry_run) if (serial_port or serial_dry_run) else None
    
    csv_f = None
    csv_writer = None
    if log_features:
        import os
        os.makedirs(os.path.dirname(log_features) or ".", exist_ok=True)
        csv_f = open(log_features, "w", newline="")
        csv_writer = csv.writer(csv_f)
        csv_writer.writerow(["U", "G", "T", "alpha", "active", "speed", "obstacle_flag"])
    
    t_start_total = time.time()
    fallback_count = 0
    obstacle_count = 0
    det_count = 0
    trust_list = []
    
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
            
            if res.active == "Classical": fallback_count += 1
            if res.command.obstacle_flag: obstacle_count += 1
            det_count += len(res.detections)
            trust_list.append(res.command.trust)
            
            t_w0 = time.time()
            if writer is None:
                writer, out_path = _open_writer(out_path, grabber.fps, (frame.shape[1], frame.shape[0]), inf["video_fourcc"])
            writer.write(frame)
            pipeline.timings["write"] += time.time() - t_w0
            
            pkt_str = f"[{pkt[0]:+.2f}, {pkt[1]:.3f}, {pkt[2]:.3f}, {pkt[3]}]"
            print(pkt_str, flush=True)
            if serial_sender is not None:
                serial_sender.send_packet(pkt)
            packets.append(pkt)
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
        if serial_sender is not None:
            serial_sender.close()
            
    if errors:
        raise RuntimeError(f"Inference thread crashed: {errors[0]!r}") from errors[0]

    # FFmpeg re-encoding
    if out_path.exists() and writer is not None:
        import shutil, subprocess, os
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
        
    logger.info(f"Trust (min/mean/max): {min(trust_list):.3f} / {sum(trust_list)/len(trust_list):.3f} / {max(trust_list):.3f}" if trust_list else "Trust: N/A")
    logger.info(f"FALLBACK frames: {fallback_count/max(1, len(packets))*100:.1f}%")
    logger.info(f"Obstacle Flag frames: {obstacle_count/max(1, len(packets))*100:.1f}%")
    logger.info(f"Mean Detections/frame: {det_count/max(1, len(packets)):.2f}")

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
    ap.add_argument("--force", action="store_true", help="Override safeguard check on filename")
    ap.add_argument("--imu-csv", default=None, help="Path to IMU CSV log file")
    ap.add_argument("--serial-port", default=None, help="Arduino serial port (e.g. COM3)")
    ap.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    ap.add_argument("--dry-run", action="store_true", help="Run serial output in dry-run simulation mode")
    a = ap.parse_args()
    
    setup_logging()
    cfg = load_config(a.config)
    if a.frame_skip:
        cfg["inference"]["frame_skip"] = a.frame_skip
    src = int(a.source) if a.source.isdigit() else a.source
    run_video(cfg, src, str(resolve_path(a.output)), checkpoint=a.checkpoint, max_frames=a.max_frames, display=a.display or None, max_width=a.max_width, log_features=a.log_features, speed_mps=a.speed_mps, force=a.force, imu_csv=a.imu_csv, serial_port=a.serial_port, serial_baud=a.baud, serial_dry_run=a.dry_run)

if __name__ == "__main__":
    main()
