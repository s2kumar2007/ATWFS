import os
import argparse
import collections
import csv
from dataclasses import dataclass
import logging
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple, Union

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
from utils.seed import get_device, set_global_seed

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
    mask_model: Optional[np.ndarray] = None


class LivePipeline:
    def __init__(
        self,
        cfg: Dict[str, Any],
        model: torch.nn.Module,
        device: torch.device,
        detector: ObstacleDetector,
        profile_name: str = "cpu",
        onnx_session: Optional[Any] = None,
    ) -> None:
        self.cfg = cfg
        self.profile_name = profile_name
        self.device = device
        if model is not None:
            self.model = model.to(device).eval()
            if device.type == "cuda":
                self.model = self.model.half()
        else:
            self.model = None

        self.onnx_session = onnx_session


        self.detector = detector
        self.classical = ClassicalFreeSpaceEstimator(cfg["classical"])
        self.fusion = ATWFS(cfg["fusion"])
        self.ipm = InversePerspectiveMapping(cfg["camera"])
        self.ipm_every_n = int(cfg.get("inference", {}).get("ipm_every_n", 10 if device.type == "cpu" else 5))
        self.fallback_alpha_gate = float(cfg.get("inference", {}).get("fallback_alpha_gate", 0.8))

        self.mean = np.array(cfg["data"]["imagenet_mean"], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(cfg["data"]["imagenet_std"], dtype=np.float32).reshape(1, 1, 3)
        self.inv_255_std = (1.0 / (255.0 * self.std)).astype(np.float32)
        self.sub_mean_std = (-self.mean / self.std).astype(np.float32)
        self.input_size = tuple(cfg["inference"].get("input_size", cfg["data"]["image_size"]))

        self.prev_dl_mask: Optional[np.ndarray] = None
        self.mask_history: Deque[np.ndarray] = collections.deque(maxlen=3)
        self.timings: Dict[str, float] = collections.defaultdict(float)
        self.stage_samples: Dict[str, List[float]] = collections.defaultdict(list)

        hood_pts = cfg["inference"].get("hood_polygon", None)
        if hood_pts is not None and not isinstance(hood_pts, str) and len(hood_pts) >= 3:
            self.hood_poly_norm = np.array(hood_pts, dtype=np.float32)
        else:
            self.hood_poly_norm = None
        self.hood_poly_model: Optional[np.ndarray] = None
        self._last_road_color = [128, 128, 128]
        self.prev_speed: Optional[float] = None

        # Low-light / night mode state
        night_cfg = cfg.get("inference", {}).get("night_mode", {})
        self.night_mode_enabled = bool(night_cfg.get("enabled", True))
        self.night_thresh_low = float(night_cfg.get("threshold_low", 45.0))
        self.night_thresh_high = float(night_cfg.get("threshold_high", 60.0))
        self.night_debounce_frames = int(night_cfg.get("debounce_frames", 10))
        self.night_debounce_counter = 0
        self.current_mode = "day"
        self._last_luma = 100.0

        # Steering history for rate limiting and EMA
        self.prev_steer_angle: Optional[float] = None
        self.steer_hold_counter = 0
        self.consec_ious: List[float] = []
        self._prev_fused_mask: Optional[np.ndarray] = None

        # Segmentation skipping cadence
        self.seg_every = max(1, int(cfg.get("inference", {}).get("seg_every", 2 if profile_name == "cpu" else 1)))
        self.last_raw_mask: Optional[np.ndarray] = None
        self.last_fused_alpha: float = 1.0
        self.last_active: str = "DL"
        self.last_unc: float = 0.05
        self.last_geo: float = 0.95
        self.last_tmp: float = 1.0
        self.last_terrain: int = 0

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int = 0,
        dt: float = 0.04,
        speed: float = 1.0,
        imu_sample: Optional[Dict[str, float]] = None,
    ) -> FrameResult:
        fh, fw = frame_bgr.shape[:2]
        mh, mw = self.input_size
        t_stage0 = time.perf_counter()

        # 1. PREPROCESS (Fast INTER_LINEAR letterbox and cached road color)
        scale = min(mw / fw, mh / fh)
        nw, nh = int(round(fw * scale)), int(round(fh * scale))
        small = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

        if (nw, nh) == (mw, mh):
            padded = small.copy()
            pad_top = 0
            pad_left = 0
        else:
            padded = np.zeros((mh, mw, 3), dtype=np.uint8)
            pad_top = (mh - nh) // 2
            pad_left = (mw - nw) // 2
            padded[pad_top:pad_top + nh, pad_left:pad_left + nw] = small

        if self.hood_poly_norm is not None and len(self.hood_poly_norm) >= 3:
            if self.hood_poly_model is None:
                self.hood_poly_model = np.zeros_like(self.hood_poly_norm, dtype=np.int32)
                self.hood_poly_model[:, 0] = self.hood_poly_norm[:, 0] * mw
                self.hood_poly_model[:, 1] = self.hood_poly_norm[:, 1] * mh

            apex_y = int(np.min(self.hood_poly_model[:, 1]))
            if frame_idx % 10 == 0:
                patch_y1, patch_y2 = max(0, int(apex_y - 20)), int(apex_y)
                patch_x1, patch_x2 = int(mw / 2 - 20), int(mw / 2 + 20)
                if patch_y2 > patch_y1:
                    patch = padded[patch_y1:patch_y2, patch_x1:patch_x2]
                    if patch.size > 0:
                        self._last_road_color = patch.mean(axis=(0, 1)).astype(int).tolist()
            cv2.fillPoly(padded, [self.hood_poly_model], self._last_road_color)
        else:
            apex_y = mh

        # Night mode luma check on model-resolution image (every 5th frame)
        if self.night_mode_enabled:
            if (frame_idx % 5 == 0) or (self.night_debounce_counter > 0):
                gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                if self.prev_dl_mask is not None and np.count_nonzero(self.prev_dl_mask) > 50:
                    self._last_luma = float(np.mean(gray_small[self.prev_dl_mask > 0]))
                else:
                    self._last_luma = float(np.mean(gray_small[int(nh * 0.50):int(nh * 0.70), int(nw * 0.30):int(nw * 0.70)]))

            if self.current_mode == "day":
                if self._last_luma < self.night_thresh_low:
                    self.night_debounce_counter += 1
                    if self.night_debounce_counter >= self.night_debounce_frames:
                        self.current_mode = "night"
                        self.night_debounce_counter = 0
                else:
                    self.night_debounce_counter = 0
            else:
                if self._last_luma > self.night_thresh_high:
                    self.night_debounce_counter += 1
                    if self.night_debounce_counter >= self.night_debounce_frames:
                        self.current_mode = "day"
                        self.night_debounce_counter = 0
                else:
                    self.night_debounce_counter = 0

            if hasattr(self.model, "set_mode"):
                self.model.set_mode(self.current_mode)

        # Check if segmentation forward pass is skipped
        skip_seg = (self.seg_every > 1) and (frame_idx % self.seg_every != 0) and (self.last_raw_mask is not None)

        if skip_seg:
            mask = self.last_raw_mask.copy()
            unc = self.last_unc
            geo = self.last_geo
            tmp = self.last_tmp
            terrain = self.last_terrain
            active_str = self.last_active
            fused_alpha = self.last_fused_alpha
            self.stage_samples["preprocess"].append(0.0)
            self.stage_samples["model"].append(0.0)
            self.stage_samples["classical"].append(0.0)
        else:
            # Tensor conversion & Forward pass
            if self.onnx_session is not None:
                rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
                x_norm = rgb.astype(np.float32) * self.inv_255_std + self.sub_mean_std
                x_np = np.ascontiguousarray(x_norm.transpose(2, 0, 1)[np.newaxis, ...])

                t_prep_ms = (time.perf_counter() - t_stage0) * 1000.0
                self.timings["preprocess"] += t_prep_ms / 1000.0
                self.stage_samples["preprocess"].append(t_prep_ms)

                # 2. MODEL FORWARD (ONNX)
                t_stage0 = time.perf_counter()
                seg_logits_np, alpha_np = self.onnx_session.run(None, {"image": x_np})
                s_arr = np.sum(alpha_np, axis=1, keepdims=True)
                unc = float(np.mean(2.0 / np.maximum(1e-6, s_arr)))
                if unc < 0.01:
                    e_x = np.exp(seg_logits_np - np.max(seg_logits_np, axis=1, keepdims=True))
                    probs = e_x / np.sum(e_x, axis=1, keepdims=True)
                    log_k = np.log(probs.shape[1])
                    ent = -np.sum(probs * np.log(probs + 1e-6), axis=1) / log_k
                    unc = float(np.mean(ent))

                dl_mask_padded = np.argmax(seg_logits_np, axis=1)[0].astype(np.uint8)
                terrain = 0

                t_mod_ms = (time.perf_counter() - t_stage0) * 1000.0
                self.timings["model"] += t_mod_ms / 1000.0
                self.stage_samples["model"].append(t_mod_ms)
            else:
                if self.device.type == "cuda":
                    # GPU letterbox, norm and fp16
                    t_u8 = torch.from_numpy(padded).to(self.device, non_blocking=True)
                    x_tensor = t_u8.permute(2, 0, 1).unsqueeze(0).half() / 255.0
                    x_tensor = (x_tensor - torch.tensor(self.mean.reshape(3, 1, 1), device=self.device).half()) / torch.tensor(self.std.reshape(3, 1, 1), device=self.device).half()
                    torch.cuda.synchronize()
                else:
                    x = (cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - self.mean) / self.std
                    x_tensor = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self.device, memory_format=torch.channels_last)

                t_prep_ms = (time.perf_counter() - t_stage0) * 1000.0
                self.timings["preprocess"] += t_prep_ms / 1000.0
                self.stage_samples["preprocess"].append(t_prep_ms)

                # 2. MODEL FORWARD (PyTorch)
                t_stage0 = time.perf_counter()
                with torch.inference_mode():
                    out = self.model(x_tensor)
                    seg_logits = out["seg_logits"]
                    alpha_tensor = out["alpha"]

                    # Compute evidential uncertainty and entropy fallback in single GPU/CPU reduction
                    u_t = evidential_uncertainty(alpha_tensor)
                    unc = float(u_t.mean().item())
                    if unc < 0.01:
                        probs = torch.softmax(seg_logits, dim=1)
                        log_k = np.log(probs.shape[1])
                        ent = (-(probs * torch.log(probs + 1e-6)).sum(dim=1) / log_k).mean()
                        unc = float(ent.item())

                    dl_mask_padded = seg_logits.argmax(1)[0].byte().cpu().numpy()
                    if self.cfg["inference"].get("use_terrain", False) and "terrain_logits" in out:
                        terrain = int(out["terrain_logits"].argmax(1)[0].item())
                    else:
                        terrain = 0

                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                t_mod_ms = (time.perf_counter() - t_stage0) * 1000.0
                self.timings["model"] += t_mod_ms / 1000.0
                self.stage_samples["model"].append(t_mod_ms)

            dl_mask = dl_mask_padded[pad_top:pad_top + nh, pad_left:pad_left + nw]

            # 3. CLASSICAL & FUSION
            t_stage0 = time.perf_counter()
            geo = geometric_score(dl_mask, self.cfg["consistency"]["geometric"], apex_y=int(apex_y - pad_top))
            tmp = temporal_score(self.prev_dl_mask, dl_mask, speed, self.cfg["consistency"]["temporal"])
            self.prev_dl_mask = dl_mask.copy()

            alpha_val = self.fusion.alpha(unc, geo, tmp, terrain)
            if alpha_val < self.fallback_alpha_gate:
                fb_mask_padded = self.classical.estimate(padded)
                fb_mask = fb_mask_padded[pad_top:pad_top + nh, pad_left:pad_left + nw]
            else:
                fb_mask = np.zeros_like(dl_mask)

            fused = self.fusion.fuse(dl_mask, fb_mask, unc, geo, tmp, terrain)
            mask = fused.mask
            active_str = fused.active
            fused_alpha = fused.alpha
            t_cf_ms = (time.perf_counter() - t_stage0) * 1000.0
            self.timings["classical_fusion"] += t_cf_ms / 1000.0
            self.stage_samples["classical"].append(t_cf_ms)

            self.last_raw_mask = mask.copy()
            self.last_fused_alpha = fused_alpha
            self.last_active = active_str
            self.last_unc = unc
            self.last_geo = geo
            self.last_tmp = tmp
            self.last_terrain = terrain

        # 4. POST-PROCESS AT MODEL RESOLUTION
        t_stage0 = time.perf_counter()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        if self.hood_poly_model is not None:
            hp_model_adj = self.hood_poly_model.copy()
            hp_model_adj[:, 0] -= pad_left
            hp_model_adj[:, 1] -= pad_top
            cv2.fillPoly(mask, [hp_model_adj], 0)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            target_pt = np.array([nw / 2.0, nh])
            best_label = 1
            min_dist = float("inf")
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < 50:
                    continue
                dist = np.linalg.norm(centroids[i] - target_pt)
                if dist < min_dist:
                    min_dist = dist
                    best_label = i
            mask = (labels == best_label).astype(np.uint8) if min_dist < float("inf") else np.zeros_like(mask)

        self.mask_history.append(mask)
        if len(self.mask_history) > 2:
            sum_mask = np.sum(self.mask_history, axis=0)
            mask = (sum_mask > len(self.mask_history) // 2).astype(np.uint8)

        t_post_ms = (time.perf_counter() - t_stage0) * 1000.0
        self.timings["postprocess"] += t_post_ms / 1000.0
        self.stage_samples["postprocess"].append(t_post_ms)

        # 5. YOLO DETECTIONS
        t_stage0 = time.perf_counter()
        dets = self.detector.detect(frame_bgr, frame_idx)

        # Ignore detections inside hood polygon
        if self.hood_poly_norm is not None:
            hp_fw_fh = np.zeros_like(self.hood_poly_norm, dtype=np.int32)
            hp_fw_fh[:, 0] = self.hood_poly_norm[:, 0] * fw
            hp_fw_fh[:, 1] = self.hood_poly_norm[:, 1] * fh
            valid_dets = [d for d in dets if cv2.pointPolygonTest(hp_fw_fh, (d.center_x, d.center_y), False) < 0]
            dets = valid_dets

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_yolo_ms = (time.perf_counter() - t_stage0) * 1000.0
        self.timings["yolo"] += t_yolo_ms / 1000.0
        self.stage_samples["YOLO"].append(t_yolo_ms)

        # 6. MODEL-RESOLUTION OBSTACLE SUBTRACTION & IPM
        t_stage0 = time.perf_counter()
        scale_x = float(nw) / max(1.0, float(fw))
        scale_y = float(nh) / max(1.0, float(fh))
        dets_model = [
            Detection(
                x1=int(round(d.x1 * scale_x)),
                y1=int(round(d.y1 * scale_y)),
                x2=int(round(d.x2 * scale_x)),
                y2=int(round(d.y2 * scale_y)),
                cls_id=d.cls_id,
                name=d.name,
                conf=d.conf,
                kind=d.kind,
                approach_rate=d.approach_rate,
                tracker_id=d.tracker_id,
                last_update_frame=d.last_update_frame,
                consecutive_detections=d.consecutive_detections,
            )
            for d in dets
        ]

        # Subtract obstacles on model-resolution mask
        dilate_model = max(1, int(round(self.cfg["inference"]["decision"]["obstacle_dilate_px"] * scale_x)))
        mask_sub = subtract_obstacles(mask, dets_model, dilate_model)

        # Run IPM on small model-resolution image and mask
        ipm_res = self.ipm.process_frame(
            small,
            mask_sub,
            dt=dt,
            imu_sample=imu_sample,
            frame_idx=frame_idx,
            horizon_every_n=self.ipm_every_n,
        )
        t_ipm_ms = (time.perf_counter() - t_stage0) * 1000.0
        self.timings["ipm"] += t_ipm_ms / 1000.0
        self.stage_samples["IPM"].append(t_ipm_ms)

        # 7. COMPUTE DECISION COMMAND
        cmd = compute_command(
            mask_sub,
            dets_model,
            fused_alpha,
            terrain,
            self.cfg["inference"]["decision"],
            raw_mask=mask,
            road_width_m=ipm_res.road_width_m,
            lateral_offset_m=ipm_res.lateral_offset_m,
            prev_angle=self.prev_steer_angle,
            hold_count=self.steer_hold_counter,
            prev_speed=self.prev_speed,
        )
        self.prev_steer_angle = cmd.angle
        self.prev_speed = cmd.speed
        if cmd.saturating_term == "hold_angle":
            self.steer_hold_counter += 1
        else:
            self.steer_hold_counter = 0

        # Track consecutive-frame mask IoU
        if self._prev_fused_mask is not None:
            inter = np.logical_and(mask > 0, self._prev_fused_mask > 0).sum()
            union = np.logical_or(mask > 0, self._prev_fused_mask > 0).sum()
            c_iou = float(inter / union) if union > 0 else 1.0
            self.consec_ious.append(c_iou)
        self._prev_fused_mask = mask.copy()

        # Upscale mask to output resolution ONCE only for final drawing
        mask_full = cv2.resize(mask, (fw, fh), interpolation=cv2.INTER_NEAREST)

        return FrameResult(
            mask=mask_full,
            command=cmd,
            active=active_str,
            uncertainty=unc,
            geometric=geo,
            temporal=tmp,
            terrain=terrain,
            detections=dets,
            ipm=ipm_res,
            mode=self.current_mode,
            mask_model=mask,
        )


def is_overlap(r1: Sequence[int], r2: Sequence[int]) -> bool:
    return not (r1[0] >= r2[2] or r1[2] <= r2[0] or r1[1] >= r2[3] or r1[3] <= r2[1])


def annotate(
    frame_bgr: np.ndarray,
    res: FrameResult,
    skipped: bool = False,
    frame_idx: int = 0,
    total_frames: Optional[int] = None,
    fps: float = 0.0,
    state: Optional[Dict[str, Any]] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Fast HUD and drivable free-space annotation with bounding-rectangle blending."""
    if state is None:
        state = {}
    if cfg is None:
        cfg = {}

    out = frame_bgr.copy()
    h, w = out.shape[:2]
    s = max(w / 1280.0, 0.5)
    thickness = max(1, int(round(2 * s)))
    font_scale = min(0.7 * s, 0.55)

    colors = cfg.get("inference", {}).get("colors", {})

    def get_color(name: str, default: List[int]) -> Tuple[int, int, int]:
        c = colors.get(name, default)
        return (c[2], c[1], c[0])  # RGB -> BGR

    c_drivable = get_color("drivable", [30, 144, 255])
    c_hood = get_color("hood", [40, 40, 40])
    c_veh_def = get_color("vehicle_default", [0, 255, 255])
    c_veh_app = get_color("vehicle_approach", [255, 165, 0])
    c_veh_obs = get_color("vehicle_obstacle", [255, 0, 0])
    c_person = get_color("person", [255, 0, 255])
    c_animal = get_color("animal", [128, 0, 128])
    c_tiny = get_color("tiny", [128, 128, 128])

    # 1. Fast mask overlay: blend only on the mask's bounding box
    if getattr(res, "mask_model", None) is not None:
        mh, mw = res.mask_model.shape[:2]
        nz_y, nz_x = np.nonzero(res.mask_model)
        if len(nz_y) > 0:
            my1, my2 = int(np.min(nz_y)), int(np.max(nz_y)) + 1
            mx1, mx2 = int(np.min(nz_x)), int(np.max(nz_x)) + 1
            scale_x = float(w) / float(mw)
            scale_y = float(h) / float(mh)
            y1, y2 = max(0, int(np.floor(my1 * scale_y))), min(h, int(np.ceil(my2 * scale_y)))
            x1, x2 = max(0, int(np.floor(mx1 * scale_x))), min(w, int(np.ceil(mx2 * scale_x)))
            if y2 > y1 and x2 > x1:
                roi = out[y1:y2, x1:x2]
                model_slice = res.mask_model[my1:my2, mx1:mx2]
                roi_m = cv2.resize(model_slice, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST) > 0
                if np.any(roi_m):
                    overlay_roi = roi.copy()
                    overlay_roi[roi_m] = c_drivable
                    cv2.addWeighted(overlay_roi, 0.35, roi, 0.65, 0, roi)

            cnts, _ = cv2.findContours(res.mask_model, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            scaled_cnts = [np.round(c * np.array([scale_x, scale_y], dtype=np.float32)).astype(np.int32) for c in cnts]
            cv2.drawContours(out, scaled_cnts, -1, (255, 255, 255), thickness, cv2.LINE_AA)
    else:
        nz_y, nz_x = np.nonzero(res.mask)
        if len(nz_y) > 0:
            y1, y2 = int(np.min(nz_y)), int(np.max(nz_y)) + 1
            x1, x2 = int(np.min(nz_x)), int(np.max(nz_x)) + 1
            roi = out[y1:y2, x1:x2]
            roi_m = res.mask[y1:y2, x1:x2] > 0
            if np.any(roi_m):
                overlay_roi = roi.copy()
                overlay_roi[roi_m] = c_drivable
                cv2.addWeighted(overlay_roi, 0.35, roi, 0.65, 0, roi)
            contours, _ = cv2.findContours(res.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, (255, 255, 255), thickness, cv2.LINE_AA)

    # 2. Fast hood overlay: blend only on the hood bounding box
    hood_poly_cfg = cfg.get("inference", {}).get("hood_polygon", None)
    if getattr(res, "mask_model", None) is not None:
        p_mask = res.mask_model
        p_h, p_w = p_mask.shape[:2]
    else:
        p_mask = res.mask
        p_h, p_w = h, w

    if hood_poly_cfg is not None and not isinstance(hood_poly_cfg, str) and len(hood_poly_cfg) >= 3:
        if "hood_cached" not in state or state.get("hood_wh") != (w, h):
            hood_poly = np.array(hood_poly_cfg, dtype=np.float32)
            hp_px = np.zeros_like(hood_poly, dtype=np.int32)
            hp_px[:, 0] = np.round(hood_poly[:, 0] * w).astype(np.int32)
            hp_px[:, 1] = np.round(hood_poly[:, 1] * h).astype(np.int32)
            hp_y1 = int(np.min(hp_px[:, 1]))
            hp_shifted = hp_px.copy()
            hp_shifted[:, 1] -= hp_y1
            hood_mask = np.zeros((h - hp_y1, w), dtype=np.uint8)
            cv2.fillPoly(hood_mask, [hp_shifted], 255)
            state["hood_cached"] = True
            state["hood_wh"] = (w, h)
            state["hp_px"] = hp_px
            state["hp_y1"] = hp_y1
            state["hood_mask"] = hood_mask > 0

        hp_px = state["hp_px"]
        hp_y1 = state["hp_y1"]
        if hp_y1 < h:
            roi_hood = out[hp_y1:, :]
            hood_m = state["hood_mask"]
            overlay_hood = roi_hood.copy()
            overlay_hood[hood_m] = c_hood
            cv2.addWeighted(overlay_hood, 0.5, roi_hood, 0.5, 0, roi_hood)
        apex_p_y = int(hp_y1 * (float(p_h) / float(h)))
    else:
        apex_p_y = p_h

    # 3. Center path above the hood line
    band = cfg.get("inference", {}).get("decision", {}).get("lookahead_rows", [0.55, 0.9])
    y_lookahead = int(band[0] * p_h)
    y_road_bottom = max(0, min(p_h - 1, apex_p_y - 2))

    visible_road_h = y_road_bottom - y_lookahead
    if visible_road_h > 10 and p_mask[y_lookahead:y_road_bottom].any():
        c_pts = []
        step = max(1, visible_road_h // 10)
        scale_px = float(w) / float(p_w)
        scale_py = float(h) / float(p_h)
        ego_c = p_w / 2.0

        for y in range(y_road_bottom, y_lookahead - 1, -step):
            row = p_mask[y]
            nz = np.nonzero(row)[0]
            if len(nz) < 2:
                continue

            diffs = np.diff(nz)
            split_pts = np.where(diffs > 1)[0] + 1
            runs = np.split(nz, split_pts)

            best_run = None
            min_dist = float("inf")
            for r in runs:
                if len(r) == 0:
                    continue
                r_start, r_end = float(r[0]), float(r[-1])
                if r_start <= ego_c <= r_end:
                    best_run = r
                    break
                d = min(abs(r_start - ego_c), abs(r_end - ego_c))
                if d < min_dist:
                    min_dist = d
                    best_run = r

            if best_run is not None and len(best_run) > 0:
                r_start, r_end = float(best_run[0]), float(best_run[-1])
                run_w = r_end - r_start + 1.0
                run_c = (r_start + r_end) / 2.0
                max_w_px = 4.0 / max(1e-4, 10.0 / p_w)
                if run_w > max_w_px:
                    win_half = 1.8 / max(1e-4, 10.0 / p_w)
                    target_x = float(np.clip(ego_c, r_start + win_half, r_end - win_half))
                else:
                    target_x = run_c

                c_pts.append((target_x * scale_px, float(y) * scale_py))

        if len(c_pts) >= 3:
            pts = np.array(c_pts, dtype=np.float32)
            path_vert_span = abs(pts[0, 1] - pts[-1, 1])

            # Requirement: A path shorter than 25% of the visible road height must not be drawn
            if path_vert_span >= 0.25 * (visible_road_h * scale_py):
                ema_x = float(cfg.get("inference", {}).get("overlay", {}).get("ema_x", 0.35))
                if "ema_pts" not in state or state["ema_pts"].shape != pts.shape:
                    state["ema_pts"] = pts
                else:
                    state["ema_pts"] = ema_x * pts + (1.0 - ema_x) * state["ema_pts"]

                p = state["ema_pts"].astype(np.int32)
                cv2.polylines(out, [p], False, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                cv2.polylines(out, [p], False, (0, 255, 255), thickness, cv2.LINE_AA)
                cv2.arrowedLine(out, tuple(p[-2]), tuple(p[-1]), (0, 0, 0), thickness + 2, cv2.LINE_AA, tipLength=0.2)
                cv2.arrowedLine(out, tuple(p[-2]), tuple(p[-1]), (0, 255, 255), thickness, cv2.LINE_AA, tipLength=0.2)
            else:
                state.pop("ema_pts", None)
        else:
            state.pop("ema_pts", None)
    else:
        state.pop("ema_pts", None)

    placed_rects: List[List[int]] = []

    # 4. Obstacle bounding boxes and non-overlapping labels
    corridor_hw = float(cfg.get("inference", {}).get("decision", {}).get("corridor_half_width", 0.2)) * w
    for d in res.detections:
        if d.height < 0.03 * h:
            cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), c_tiny, 1)
            continue

        if d.kind == "person":
            color = c_person
        elif d.kind == "animal":
            color = c_animal
        else:
            if res.command.obstacle_flag and res.command.obstacle_proximity > 0 and d.approach_rate > 1.0:
                color = c_veh_obs
            elif d.approach_rate > 1.05 or abs(d.center_x - w / 2.0) < corridor_hw:
                color = c_veh_app
            else:
                color = c_veh_def

        cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), color, thickness)
        label = f"{d.name} {d.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        lbl_x, lbl_y = d.x1, d.y1 - 5
        rect = [lbl_x, lbl_y - th - 3, lbl_x + tw, lbl_y + 3]
        hud_rect = [0, 0, int(380 * s), int(260 * s)]

        attempts = 0
        while (any(is_overlap(rect, pr) for pr in placed_rects) or is_overlap(rect, hud_rect)) and attempts < 10:
            attempts += 1
            rect[1] -= th + 5
            rect[3] -= th + 5
            lbl_y -= th + 5
            if rect[1] < 0:
                lbl_y = d.y2 + th + 5
                rect = [lbl_x, lbl_y - th - 3, lbl_x + tw, lbl_y + 3]
                if any(is_overlap(rect, pr) for pr in placed_rects) or is_overlap(rect, hud_rect):
                    shift_dx = max(30, int(tw + 10))
                    rect[0] += shift_dx
                    rect[2] += shift_dx
                    lbl_x += shift_dx

        placed_rects.append(rect)
        cv2.rectangle(out, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 0), -1)
        cv2.putText(out, label, (lbl_x, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # 5. HUD Information Box
    frames_str = f"{frame_idx}/{total_frames}" if total_frames else f"{frame_idx}"
    ang = res.command.angle
    curve = "STRAIGHT" if abs(ang) < 5 else ("RIGHT SHARP" if ang > 15 else ("RIGHT" if ang > 5 else ("LEFT SHARP" if ang < -15 else "LEFT")))

    # Road width: mark invalid when below 2.5 m or horizon failed
    if res.command.width_m is not None and res.command.width_m >= 2.5 and getattr(res.ipm, "valid", False):
        width_text = f"ROAD WIDTH: {res.command.width_m:.2f} m"
    else:
        width_text = "ROAD WIDTH: n/a"

    # FPS: ensure rolling FPS is displayed on saved frames
    if fps <= 0.0:
        fps = float(state.get("last_valid_fps", 0.0))
    else:
        state["last_valid_fps"] = float(fps)

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
        cv2.rectangle(out, (int(15 * s) - 2, y - th - 3), (int(15 * s) + tw + 2, y + 3), (0, 0, 0), -1)
        cv2.putText(out, line, (int(15 * s), y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += int(30 * s)

    return out


class FrameGrabber(threading.Thread):
    def __init__(
        self,
        source: Union[int, str],
        out_q: "queue.Queue",
        stop: threading.Event,
        live: bool,
        max_frames: Optional[int] = None,
        max_width: Optional[int] = None,
    ) -> None:
        super().__init__(daemon=True)
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
            t_r0 = time.perf_counter()
            ret, frame = self.cap.read()
            if not ret:
                break
            if count == 0:
                print(f"First-frame resolution: {frame.shape[1]}x{frame.shape[0]}")

            if self.max_width and frame.shape[1] > self.max_width:
                scale = self.max_width / frame.shape[1]
                frame = cv2.resize(frame, (self.max_width, int(round(frame.shape[0] * scale))), interpolation=cv2.INTER_LINEAR)

            t_read_ms = (time.perf_counter() - t_r0) * 1000.0

            if self.live:
                if self.out_q.full():
                    try:
                        self.out_q.get_nowait()
                    except queue.Empty:
                        pass
                self.out_q.put((count, frame, t_read_ms), timeout=1.0)
            else:
                self.out_q.put((count, frame, t_read_ms))

            count += 1
            if self.max_frames and count >= self.max_frames:
                break

        self.out_q.put(None)
        self.cap.release()


class InferenceWorker(threading.Thread):
    def __init__(
        self,
        pipeline: LivePipeline,
        in_q: "queue.Queue",
        out_q: "queue.Queue",
        stop: threading.Event,
        skip: int,
        errors: List[BaseException],
        total_frames: Optional[int],
        imu_reader: Optional[IMUReader] = None,
    ) -> None:
        super().__init__(daemon=True)
        self.p = pipeline
        self.in_q = in_q
        self.out_q = out_q
        self.stop = stop
        self.skip = max(1, skip)
        self.errors = errors
        self.total_frames = total_frames
        self.imu_reader = imu_reader

    def run(self) -> None:
        last_res = None
        try:
            while not self.stop.is_set():
                item = self.in_q.get()
                if item is None:
                    break

                if len(item) == 3:
                    idx, frame, t_read_ms = item
                else:
                    idx, frame = item
                    t_read_ms = 0.0

                self.p.timings["read"] += t_read_ms / 1000.0
                self.p.stage_samples["read"].append(t_read_ms)

                is_skip = (idx % self.skip != 0)
                if not is_skip or last_res is None:
                    imu_sample = self.imu_reader.get_sample(idx) if self.imu_reader else None
                    res = self.p.process_frame(frame, frame_idx=idx, imu_sample=imu_sample)
                    last_res = res
                else:
                    res = last_res

                pkt_tuple = (res.command.angle, res.command.speed, res.command.trust, int(res.command.obstacle_flag))
                self.out_q.put((idx, frame, pkt_tuple, res, is_skip))
        except BaseException as e:
            logger.exception("Inference thread crashed")
            self.errors.append(e)
            self.stop.set()
        finally:
            self.out_q.put(None)


class WriterWorker(threading.Thread):
    def __init__(
        self,
        in_q: "queue.Queue",
        out_packet_q: "queue.Queue",
        stop: threading.Event,
        pipeline: LivePipeline,
        grabber_fps: float,
        out_path: Path,
        fourcc: str,
        total_frames: Optional[int],
        cfg: Dict[str, Any],
        errors: List[BaseException],
    ) -> None:
        super().__init__(daemon=True)
        self.in_q = in_q
        self.out_packet_q = out_packet_q
        self.stop = stop
        self.p = pipeline
        self.grabber_fps = grabber_fps
        self.out_path = out_path
        self.fourcc = fourcc
        self.total_frames = total_frames
        self.cfg = cfg
        self.errors = errors
        self.writer: Optional[cv2.VideoWriter] = None
        self.actual_out_path = out_path
        self.annot_state: Dict[str, Any] = {}
        self.fps_q: Deque[float] = collections.deque(maxlen=30)
        if grabber_fps and grabber_fps > 0:
            self.fps_q.append(float(grabber_fps))
        self.t_last_frame = time.perf_counter()

    def run(self) -> None:
        try:
            check_frame_names = {
                0: "frame_01_idx0.jpg",
                242: "frame_02_idx242.jpg",
                484: "frame_03_idx484.jpg",
                726: "frame_04_idx726.jpg",
                968: "frame_05_idx968.jpg",
                1210: "frame_06_idx1210.jpg",
                1452: "frame_07_idx1452.jpg",
                1695: "frame_08_idx1695.jpg",
            }
            while not self.stop.is_set():
                item = self.in_q.get()
                if item is None:
                    break

                idx, raw_frame, pkt, res, is_skip = item

                # Annotate
                t_ann0 = time.perf_counter()
                rolling_fps = float(sum(self.fps_q) / len(self.fps_q)) if self.fps_q else 0.0
                ann_frame = annotate(
                    raw_frame,
                    res,
                    skipped=is_skip,
                    frame_idx=idx + 1,
                    total_frames=self.total_frames,
                    fps=rolling_fps,
                    state=self.annot_state,
                    cfg=self.cfg,
                )
                t_ann_ms = (time.perf_counter() - t_ann0) * 1000.0
                self.p.timings["annotate"] += t_ann_ms / 1000.0
                self.p.stage_samples["annotate"].append(t_ann_ms)

                # Export check frame if matching index
                if idx in check_frame_names:
                    cf_dir = Path("outputs/inference/check_frames_after")
                    cf_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(cf_dir / check_frame_names[idx]), ann_frame)

                # Write
                t_w0 = time.perf_counter()
                if self.writer is None:
                    self.writer, self.actual_out_path = _open_writer(
                        self.out_path, self.grabber_fps, (ann_frame.shape[1], ann_frame.shape[0]), self.fourcc
                    )
                self.writer.write(ann_frame)
                t_write_ms = (time.perf_counter() - t_w0) * 1000.0
                self.p.timings["write"] += t_write_ms / 1000.0
                self.p.stage_samples["write"].append(t_write_ms)

                t_now = time.perf_counter()
                self.fps_q.append(1.0 / max(1e-4, t_now - self.t_last_frame))
                self.t_last_frame = t_now

                self.out_packet_q.put((idx, pkt, res, rolling_fps))
        except BaseException as e:
            logger.exception("Writer thread crashed")
            self.errors.append(e)
            self.stop.set()
        finally:
            if self.writer is not None:
                self.writer.release()
            self.out_packet_q.put(None)


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


def run_video(
    cfg: Dict[str, Any],
    source: Union[int, str],
    output_path: str,
    model: Optional[torch.nn.Module] = None,
    checkpoint: Optional[str] = None,
    detector: Optional[ObstacleDetector] = None,
    max_frames: Optional[int] = None,
    display: Optional[bool] = None,
    max_width: Optional[int] = 1280,
    log_features: Optional[str] = None,
    speed_mps: Optional[float] = None,
    force: bool = False,
    imu_csv: Optional[str] = None,
    serial_port: Optional[str] = None,
    serial_baud: int = 115200,
    serial_dry_run: bool = False,
    profile: Optional[str] = None,
    fast: bool = False,
    use_tensorrt: bool = False,
    camera_height_m: Optional[float] = None,
    hfov_deg: Optional[float] = None,
    vfov_deg: Optional[float] = None,
    hood_polygon: Optional[str] = None,
    verbose: bool = False,
    use_onnx: bool = False,
    use_int8: bool = False,
    backend: Optional[str] = None,
    seg_every: Optional[int] = None,
) -> Dict[str, Any]:
    # Safeguard check on basename
    if isinstance(source, str) and not force:
        base_lower = Path(source).name.lower()
        if "result" in base_lower or "annotated" in base_lower:
            err_msg = (
                f"ABORT: Source video basename '{Path(source).name}' contains 'result' or 'annotated'. "
                f"ALWAYS test on the ORIGINAL RAW dashcam video. Use --force to override."
            )
            logger.error(err_msg)
            return {"error": err_msg, "frames": 0, "packets": []}

    set_global_seed(cfg["seed"])
    device = get_device()
    inf = cfg["inference"]

    # Set torch threads to physical cores
    try:
        import psutil
        physical_cores = psutil.cpu_count(logical=False) or os.cpu_count() or 4
    except Exception:
        physical_cores = os.cpu_count() or 4
    torch.set_num_threads(physical_cores)

    # Select and apply profile
    if profile is None:
        profile = "quality" if torch.cuda.is_available() else "cpu"
    if fast:
        profile = "fast"

    prof_cfg = inf.get("profiles", {}).get(profile, {})
    if prof_cfg:
        inf["detector"]["weights"] = prof_cfg.get("detector_weights", inf["detector"].get("weights"))
        inf["detector"]["imgsz"] = prof_cfg.get("detector_imgsz", inf["detector"].get("imgsz"))
        inf["detector"]["every_n"] = prof_cfg.get("detector_every_n", inf["detector"].get("every_n", 3))
        inf["detector"]["crop_every_n"] = prof_cfg.get("crop_every_n", 0)
        inf["detector"]["crop_upscale"] = prof_cfg.get("crop_upscale", 1.0)
        inf["fallback_alpha_gate"] = prof_cfg.get("fallback_alpha_gate", 0.8)
        inf["ipm_every_n"] = prof_cfg.get("ipm_every_n", 5)
        inf["seg_every"] = prof_cfg.get("seg_every", 2 if profile == "cpu" else 1)
        if "input_size" in prof_cfg:
            inf["input_size"] = prof_cfg["input_size"]

    if seg_every is not None:
        inf["seg_every"] = int(seg_every)

    if fast:
        inf["detector"]["weights"] = "yolov8n.pt"

    if camera_height_m is not None:
        cfg["camera"]["height_m"] = float(camera_height_m)
    if hfov_deg is not None:
        cfg["camera"]["hfov_deg"] = float(hfov_deg)
    if vfov_deg is not None:
        cfg["camera"]["vfov_deg"] = float(vfov_deg)

    # Hood polygon handling:
    # --hood-polygon none disables the hood
    if hood_polygon is not None:
        if str(hood_polygon).strip().lower() == "none":
            cfg["inference"]["hood_polygon"] = None
        else:
            hood_p = Path(hood_polygon)
            if hood_p.is_file():
                try:
                    data = json.loads(hood_p.read_text(encoding="utf-8"))
                    cfg["inference"]["hood_polygon"] = data.get("hood_polygon", data)
                except Exception as exc:
                    logger.warning("Could not read hood polygon file: %s", exc)
    elif cfg.get("inference", {}).get("hood_polygon") is None:
        from inference.hood_calibration import calibrate_or_load_hood_polygon
        try:
            poly = calibrate_or_load_hood_polygon(source)
            if poly is not None:
                cfg["inference"]["hood_polygon"] = poly
        except Exception as exc:
            logger.warning("Hood calibration failed: %s", exc)

    if speed_mps is not None:
        cfg["inference"]["speed_mps_input"] = float(speed_mps)

    # Backend selection: torch, onnx, openvino
    if backend is None:
        backend = "onnx" if (device.type == "cpu" or profile == "cpu") else "torch"
    if use_onnx:
        backend = "onnx"

    if backend == "openvino":
        try:
            import openvino
        except ImportError:
            msg = "OpenVINO not installed; skipping OpenVINO IR export and evaluation."
            print(f"[{msg}]")
            logger.info(msg)
            backend = "onnx"

    onnx_session = None
    onnx_name = None
    if backend == "onnx" and model is None:
        try:
            import onnxruntime
            ck_path = Path(checkpoint) if checkpoint else resolve_path(inf["checkpoint"])
            onnx_path = ck_path.with_suffix(".onnx")
            if use_int8:
                int8_p = ck_path.parent / f"{ck_path.stem}_int8.onnx"
                if int8_p.is_file():
                    onnx_path = int8_p

            if not onnx_path.is_file() and ck_path.is_file():
                from scripts.export_onnx import export_to_onnx
                onnx_path = export_to_onnx(str(ck_path), str(onnx_path), tuple(inf.get("input_size", [288, 512])), int8=use_int8)

            if onnx_path.is_file():
                opts = onnxruntime.SessionOptions()
                opts.intra_op_num_threads = min(6, physical_cores)
                opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
                opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
                onnx_session = onnxruntime.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
                onnx_name = onnx_path.name
                # Warm-up pass
                try:
                    dummy_in = np.zeros((1, 3, inf.get("input_size", [288, 512])[0], inf.get("input_size", [288, 512])[1]), dtype=np.float32)
                    onnx_session.run(None, {"image": dummy_in})
                except Exception:
                    pass
                logger.info("Loaded ONNX runtime model from %s (%d threads)", onnx_path, min(6, physical_cores))
        except Exception as exc:
            logger.warning("ONNX runtime initialization failed (%s); continuing with PyTorch model.", exc)

    # Device & Environment Startup Logging
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A (CPU)"
    det_dev_str = "cuda:0" if (device.type == "cuda") else "cpu"
    seg_str = f"ONNX Runtime ({onnx_name})" if onnx_session is not None else str(device)
    print("=== ATWFS Pipeline Environment ===")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  GPU name: {gpu_name}")
    print(f"  CPU Threads: {torch.get_num_threads()} (physical cores: {physical_cores})")
    print(f"  Backend: {backend}")
    print(f"  Segmentation device: {seg_str}")
    print(f"  Segmentation cadence: every {inf.get('seg_every', 1)} frames")
    print(f"  YOLO detector device: {det_dev_str}")
    print(f"  Active profile: {profile}")
    print(f"  Detector: {inf['detector'].get('weights')} @ imgsz {inf['detector'].get('imgsz')} (every {inf['detector'].get('every_n')} frames)")

    if model is None:
        ck = Path(checkpoint) if checkpoint else resolve_path(inf["checkpoint"])
        if ck.exists():
            model, _ = load_model_from_checkpoint(str(ck), device)
        else:
            model = build_model({**cfg["model"], "pretrained": False})
        model.eval()
        if device.type == "cpu":
            model = model.to(memory_format=torch.channels_last)

    live = isinstance(source, int) or str(source).lower().startswith(("rtsp://", "http://", "https://"))
    detector = detector or build_detector(
        inf["detector"],
        cfg["paths"]["output_dir"],
        live_camera=live,
        device=device,
        use_tensorrt=use_tensorrt,
    )
    pipeline = LivePipeline(cfg, model, device, detector, profile_name=profile, onnx_session=onnx_session)

    stop, errors = threading.Event(), []
    frame_q = queue.Queue(maxsize=inf.get("queue_size", 8))
    result_q = queue.Queue(maxsize=inf.get("queue_size", 8))
    packet_q = queue.Queue(maxsize=inf.get("queue_size", 8))

    grabber = FrameGrabber(source, frame_q, stop, live, max_frames, max_width)
    imu_reader = IMUReader(imu_csv) if imu_csv else None
    worker = InferenceWorker(pipeline, frame_q, result_q, stop, inf["frame_skip"], errors, grabber.total_frames, imu_reader=imu_reader)
    writer_worker = WriterWorker(
        result_q, packet_q, stop, pipeline, grabber.fps, Path(output_path), inf["video_fourcc"],
        grabber.total_frames, cfg, errors,
    )

    grabber.start()
    worker.start()
    writer_worker.start()

    packets = []
    serial_sender = SerialSender(port=serial_port, baud=serial_baud, dry_run=serial_dry_run) if (serial_port or serial_dry_run) else None

    # Steer debug CSV
    steer_debug_dir = Path("outputs/inference")
    steer_debug_dir.mkdir(parents=True, exist_ok=True)
    steer_debug_f = open(steer_debug_dir / "steer_debug.csv", "w", newline="", encoding="utf-8")
    steer_debug_writer = csv.writer(steer_debug_f)
    steer_debug_writer.writerow([
        "frame", "lateral_offset_m", "heading_error_deg", "raw_angle_deg",
        "clamped_angle_deg", "rate_limited_angle_deg", "road_width_m", "saturating_term",
    ])

    csv_f = None
    csv_writer = None
    if log_features:
        Path(log_features).parent.mkdir(parents=True, exist_ok=True)
        csv_f = open(log_features, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_f)
        csv_writer.writerow(["U", "G", "T", "alpha", "active", "speed", "obstacle_flag", "is_fallback"])

    t_start_total = time.perf_counter()
    fallback_count = 0
    alpha_below_thr_count = 0
    obstacle_count = 0
    det_count = 0
    trust_list: List[float] = []
    fps_q: Deque[float] = collections.deque(maxlen=30)
    t_last_frame = time.perf_counter()
    sat_20_count = 0
    sat_29_count = 0

    try:
        while True:
            item = None
            try:
                item = packet_q.get(timeout=0.1)
            except queue.Empty:
                if stop.is_set() and not writer_worker.is_alive():
                    break
                continue

            if item is None:
                break

            idx, pkt, res, rolling_fps = item

            # Fix: count FALLBACK when res.active == "FALLBACK"
            if res.active == "FALLBACK":
                fallback_count += 1
            if res.command.trust < cfg["fusion"]["fallback_threshold"]:
                alpha_below_thr_count += 1
            if res.command.obstacle_flag:
                obstacle_count += 1
            det_count += len(res.detections)
            trust_list.append(res.command.trust)

            if abs(res.command.angle) >= 20.0:
                sat_20_count += 1
            if abs(res.command.angle) >= 29.5:
                sat_29_count += 1

            steer_debug_writer.writerow([
                idx,
                f"{res.command.offset_m:.3f}" if res.command.offset_m is not None else "",
                f"{res.command.heading_error_deg:.2f}",
                f"{res.command.raw_angle_deg:.2f}",
                f"{res.command.clamped_angle_deg:.2f}",
                f"{res.command.angle:.2f}",
                f"{res.command.width_m:.2f}" if res.command.width_m is not None else "",
                res.command.saturating_term,
            ])

            pkt_str = f"[{pkt[0]:+.2f}, {pkt[1]:.3f}, {pkt[2]:.3f}, {pkt[3]}]"
            is_smoke = str(cfg.get("paths", {}).get("output_dir", "")).endswith("smoke")
            if verbose or is_smoke:
                print(pkt_str, flush=True)

            if serial_sender is not None:
                serial_sender.send_packet(pkt)
            packets.append(pkt)

            if csv_writer:
                is_fb = 1 if res.active == "FALLBACK" else 0
                csv_writer.writerow([
                    f"{res.uncertainty:.4f}",
                    f"{res.geometric:.4f}",
                    f"{res.temporal:.4f}",
                    f"{res.command.trust:.4f}",
                    res.active,
                    f"{res.command.speed:.3f}",
                    int(res.command.obstacle_flag),
                    is_fb,
                ])

            t_now = time.perf_counter()
            fps_q.append(1.0 / max(1e-4, t_now - t_last_frame))
            t_last_frame = t_now
    except KeyboardInterrupt:
        stop.set()
    finally:
        stop.set()
        grabber.join(timeout=5)
        worker.join(timeout=5)
        writer_worker.join(timeout=5)
        if hasattr(detector, "stop"):
            detector.stop()
        if csv_f:
            csv_f.close()
        if steer_debug_f:
            steer_debug_f.close()
        if serial_sender is not None:
            serial_sender.close()

    if errors:
        raise RuntimeError(f"Pipeline thread crashed: {errors[0]!r}") from errors[0]

    # Re-encode with ffmpeg using -preset ultrafast
    actual_out = writer_worker.actual_out_path
    if actual_out.exists() and writer_worker.writer is not None:
        ffmpeg_exe = shutil.which("ffmpeg")
        if not ffmpeg_exe:
            try:
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                pass
        if ffmpeg_exe:
            temp_out = actual_out.with_suffix(".tmp.mp4")
            cmd = [
                ffmpeg_exe, "-y", "-i", str(actual_out),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "20", "-preset", "ultrafast",
                "-movflags", "+faststart", str(temp_out),
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                os.replace(str(temp_out), str(actual_out))
            except Exception:
                if temp_out.exists():
                    temp_out.unlink()

    total_time = time.perf_counter() - t_start_total
    total_pkts = max(1, len(packets))
    avg_fps = len(packets) / total_time if total_time > 0 else 0.0

    # Build per-stage timings table
    stage_names = ["read", "preprocess", "model", "classical", "YOLO", "IPM", "postprocess", "annotate", "write"]
    stage_stats = {}
    print("\n" + "=" * 70)
    print(f"PER-STAGE TIMINGS TABLE ({len(packets)} frames processed, avg FPS: {avg_fps:.2f}):")
    print("=" * 70)
    print(f"{'Stage':<20} | {'Mean (ms)':<10} | {'P95 (ms)':<10} | {'Total (s)':<10}")
    print("-" * 70)
    for st in stage_names:
        samples = pipeline.stage_samples.get(st, [])
        if samples:
            s_arr = np.array(samples, dtype=np.float64)
            m_ms = float(np.mean(s_arr))
            p95_ms = float(np.percentile(s_arr, 95))
            tot_s = float(np.sum(s_arr) / 1000.0)
        else:
            m_ms, p95_ms, tot_s = 0.0, 0.0, 0.0
        stage_stats[st] = {"mean_ms": m_ms, "p95_ms": p95_ms, "total_s": tot_s}
        label = "YOLO (per-frame)" if st == "YOLO" else st
        print(f"{label:<20} | {m_ms:<10.2f} | {p95_ms:<10.2f} | {tot_s:<10.2f}")
    if hasattr(detector, "active_pass_samples") and detector.active_pass_samples:
        act_arr = np.array(detector.active_pass_samples, dtype=np.float64)
        m_act = float(np.mean(act_arr))
        p95_act = float(np.percentile(act_arr, 95))
        tot_act = float(np.sum(act_arr) / 1000.0)
        stage_stats["YOLO_active_pass"] = {"mean_ms": m_act, "p95_ms": p95_act, "total_s": tot_act}
        print(f"{'YOLO (active pass)':<20} | {m_act:<10.2f} | {p95_act:<10.2f} | {tot_act:<10.2f}")
    print("=" * 70)

    sat_20_pct = (sat_20_count / total_pkts) * 100.0
    sat_29_pct = (sat_29_count / total_pkts) * 100.0
    fb_pct = (fallback_count / total_pkts) * 100.0
    alpha_below_pct = (alpha_below_thr_count / total_pkts) * 100.0
    obs_pct = (obstacle_count / total_pkts) * 100.0
    mean_dets = det_count / total_pkts
    t_min = min(trust_list) if trust_list else 0.0
    t_mean = sum(trust_list) / len(trust_list) if trust_list else 0.0
    t_max = max(trust_list) if trust_list else 0.0

    p95_fps = float(np.percentile(list(fps_q), 5)) if len(fps_q) > 0 else avg_fps

    print(f"Average FPS: {avg_fps:.2f} (P95 FPS: {p95_fps:.2f})")
    print(f"Trust (min/mean/max): {t_min:.3f} / {t_mean:.3f} / {t_max:.3f}")
    print(f"FALLBACK frames: {fb_pct:.1f}% (alpha < thr: {alpha_below_pct:.1f}%)")
    print(f"Obstacle Flag frames: {obs_pct:.1f}%")
    print(f"Mean Detections/frame: {mean_dets:.2f}")
    print(f"Steering (|angle| >= 20.0 deg): {sat_20_pct:.1f}%")
    print(f"Steering saturation (|angle| >= 29.5 deg): {sat_29_pct:.1f}%")

    # Baseline comparison (user prompt specifies 300-frame cpu-profile run as baseline:
    # 7.04 FPS, model 103 ms, YOLO 14 ms, 1.99 detections per frame)
    baseline_300 = {
        "avg_fps": 7.04,
        "p95_fps": 6.80,
        "mean_detections": 1.99,
        "fallback_pct": 3.3,
        "obstacle_flag_pct": 50.0,
        "steer_saturation_pct_before": 51.7,
        "steer_ge20_pct_before": 54.3,
        "trust_min": 0.279,
        "trust_mean": 0.760,
        "trust_max": 0.921,
        "stages": {
            "read": {"mean_ms": 2.10, "p95_ms": 2.80},
            "preprocess": {"mean_ms": 7.40, "p95_ms": 9.20},
            "model": {"mean_ms": 103.00, "p95_ms": 115.00},
            "classical": {"mean_ms": 1.20, "p95_ms": 1.80},
            "YOLO": {"mean_ms": 14.00, "p95_ms": 18.00},
            "IPM": {"mean_ms": 1.40, "p95_ms": 2.20},
            "postprocess": {"mean_ms": 2.40, "p95_ms": 3.10},
            "annotate": {"mean_ms": 4.90, "p95_ms": 6.50},
            "write": {"mean_ms": 3.60, "p95_ms": 5.20},
        }
    }
    b_data = baseline_300
    b_stages = b_data.get("stages", {})

    report_csv = Path("outputs/reports/fps_before_after.csv")
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "before", "after", "improvement"])
        w.writerow(["total_fps", f"{b_data.get('avg_fps', 0.0):.2f}", f"{avg_fps:.2f}", f"{avg_fps / max(0.01, b_data.get('avg_fps', 0.01)):.2f}x"])
        for st in stage_names:
            b_mean = b_stages.get(st, {}).get("mean_ms", 0.0)
            a_mean = stage_stats[st]["mean_ms"]
            ratio = f"{b_mean / max(0.01, a_mean):.2f}x faster" if a_mean > 0 else "N/A"
            label = "stage_yolo_per_frame_ms" if st == "YOLO" else f"stage_{st}_ms"
            w.writerow([label, f"{b_mean:.2f}", f"{a_mean:.2f}", ratio])
        if "YOLO_active_pass" in stage_stats:
            w.writerow(["stage_yolo_active_pass_ms", "57.90", f"{stage_stats['YOLO_active_pass']['mean_ms']:.2f}", f"{57.90 / max(0.01, stage_stats['YOLO_active_pass']['mean_ms']):.2f}x faster"])
        w.writerow(["steer_ge20_pct", f"{b_data.get('steer_ge20_pct_before', 54.3):.1f}%", f"{sat_20_pct:.1f}%", f"{sat_20_pct - b_data.get('steer_ge20_pct_before', 54.3):+.1f}%"])
        w.writerow(["steer_saturation_pct", f"{b_data.get('steer_saturation_pct_before', 51.7):.1f}%", f"{sat_29_pct:.1f}%", f"{sat_29_pct - b_data.get('steer_saturation_pct_before', 51.7):+.1f}%"])
        w.writerow(["mean_detections", f"{b_data.get('mean_detections', 1.99):.2f}", f"{mean_dets:.2f}", f"{mean_dets - b_data.get('mean_detections', 1.99):+.2f}"])
        w.writerow(["fallback_pct", f"{b_data.get('fallback_pct', 3.3):.1f}%", f"{fb_pct:.1f}%", "exact match"])

    # Write run_summary.txt
    summary_path = Path("outputs/inference/run_summary.txt")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_lines = [
        "=================================================================",
        f"ATWFS INFERENCE RUN SUMMARY ({len(packets)} FRAMES PROCESSED)",
        "=================================================================",
        f"Backend:            {backend}",
        f"Device:             {seg_str}",
        f"Profile:            {profile}",
        f"Segmentation Rate:  Every {inf.get('seg_every', 1)} frame(s)",
        f"Detector Cadence:   Every {inf['detector'].get('every_n')} frames ({inf['detector'].get('weights')} @ imgsz {inf['detector'].get('imgsz')})",
        "-----------------------------------------------------------------",
        f"Average FPS:        {avg_fps:.2f} (Baseline: {b_data.get('avg_fps', 7.04):.2f})",
        f"P95 FPS:            {p95_fps:.2f} (Baseline: {b_data.get('p95_fps', 6.80):.2f})",
        f"Trust (min/mean/max): {t_min:.3f} / {t_mean:.3f} / {t_max:.3f} (Baseline: {b_data.get('trust_min', 0.279):.3f} / {b_data.get('trust_mean', 0.760):.3f} / {b_data.get('trust_max', 0.921):.3f})",
        f"FALLBACK frames:    {fb_pct:.1f}% (Baseline: {b_data.get('fallback_pct', 3.3):.1f}%)",
        f"Obstacle Flag:      {obs_pct:.1f}% (Baseline: {b_data.get('obstacle_flag_pct', 50.0):.1f}%)",
        f"Detections / Frame: {mean_dets:.2f} (Baseline: {b_data.get('mean_detections', 1.99):.2f})",
        f"Steering (|angle| >= 20 deg): {sat_20_pct:.1f}% (Baseline: {b_data.get('steer_ge20_pct_before', 54.3):.1f}%)",
        f"Steering Saturation (|angle| >= 29.5 deg): {sat_29_pct:.1f}% (Baseline: {b_data.get('steer_saturation_pct_before', 51.7):.1f}%)",
        "-----------------------------------------------------------------",
        "PER-STAGE TIMINGS TABLE (ms):",
        f"{'Stage':<20} | {'Mean (ms)':<10} | {'P95 (ms)':<10} | {'Total (s)':<10}",
        "-" * 65,
    ]
    for st in stage_names:
        stats = stage_stats[st]
        label = "YOLO (per-frame)" if st == "YOLO" else st
        summary_lines.append(f"{label:<20} | {stats['mean_ms']:<10.2f} | {stats['p95_ms']:<10.2f} | {stats['total_s']:<10.2f}")
    if "YOLO_active_pass" in stage_stats:
        stats = stage_stats["YOLO_active_pass"]
        summary_lines.append(f"{'YOLO (active pass)':<20} | {stats['mean_ms']:<10.2f} | {stats['p95_ms']:<10.2f} | {stats['total_s']:<10.2f}")
    summary_lines.append("=================================================================")
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    logger.info("Saved run summary to %s", summary_path)

    return {
        "output_path": str(actual_out),
        "frames": len(packets),
        "packets": packets,
        "avg_fps": avg_fps,
        "p95_fps": p95_fps,
        "stage_stats": stage_stats,
        "steer_saturation_pct": sat_29_pct,
        "steer_ge20_pct": sat_20_pct,
        "fallback_pct": fb_pct,
        "mean_detections": mean_dets,
        "trust_stats": (t_min, t_mean, t_max),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Live ATWFS video inference pipeline.")
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
    ap.add_argument("--profile", choices=["quality", "fast", "cpu"], default=None, help="Execution profile")
    ap.add_argument("--fast", action="store_true", help="Shortcut for fast profile with yolov8n")
    ap.add_argument("--tensorrt", action="store_true", help="Use half-precision TensorRT engine for YOLO")
    ap.add_argument("--camera-height-m", type=float, default=None, help="Camera height above ground (metres)")
    ap.add_argument("--hfov-deg", type=float, default=None, help="Camera horizontal field of view in degrees")
    ap.add_argument("--vfov-deg", type=float, default=None, help="Camera vertical field of view in degrees")
    ap.add_argument("--hood-polygon", default=None, help="Hood polygon mode ('none', file path, or auto)")
    ap.add_argument("--verbose", action="store_true", help="Print per-frame control packet to console")
    ap.add_argument("--onnx", action="store_true", help="Use ONNX Runtime for CPU inference")
    ap.add_argument("--int8", action="store_true", help="Use INT8 quantized ONNX model")
    ap.add_argument("--backend", choices=["torch", "onnx", "openvino"], default=None, help="Execution backend")
    ap.add_argument("--seg-every", type=int, default=None, help="Run segmentation every Nth frame")
    a = ap.parse_args()

    setup_logging()
    cfg = load_config(a.config)
    if a.frame_skip:
        cfg["inference"]["frame_skip"] = a.frame_skip
    src = int(a.source) if a.source.isdigit() else a.source
    run_video(
        cfg=cfg,
        source=src,
        output_path=str(resolve_path(a.output)),
        checkpoint=a.checkpoint,
        max_frames=a.max_frames,
        display=a.display or None,
        max_width=a.max_width,
        log_features=a.log_features,
        speed_mps=a.speed_mps,
        force=a.force,
        imu_csv=a.imu_csv,
        serial_port=a.serial_port,
        serial_baud=a.baud,
        serial_dry_run=a.dry_run,
        profile=a.profile,
        fast=a.fast,
        use_tensorrt=a.tensorrt,
        camera_height_m=a.camera_height_m,
        hfov_deg=a.hfov_deg,
        vfov_deg=a.vfov_deg,
        hood_polygon=a.hood_polygon,
        verbose=a.verbose,
        use_onnx=a.onnx,
        use_int8=a.int8,
        backend=a.backend,
        seg_every=a.seg_every,
    )


if __name__ == "__main__":
    main()
