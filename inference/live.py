import json
import sys
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
from models.unet_mobilenet import evidential_uncertainty
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
    prob_model: Optional[np.ndarray] = None
    processing_ms: float = 0.0


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
        if isinstance(device, str):
            device = torch.device(device)
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
        self.prev_prob_crop: Optional[np.ndarray] = None
        self._last_road_color = [128, 128, 128]
        self.prev_speed: Optional[float] = None

        # Low-light / night mode state
        night_cfg = cfg.get("inference", {}).get("night_mode", {})
        self.night_mode_enabled = bool(night_cfg.get("enabled", True))
        self.night_thresh_low = float(night_cfg.get("threshold_low", 35.0))
        self.night_thresh_high = float(night_cfg.get("threshold_high", 45.0))
        self.night_debounce_frames = int(night_cfg.get("debounce_frames", 15))
        self.night_debounce_counter = 0
        self.current_mode = "day"
        self._last_mean_luma = 100.0
        self._last_p90_luma = 100.0
        self.mode_switch_count = 0

        # Steering history for rate limiting and EMA
        self.prev_steer_angle: Optional[float] = None
        self.steer_hold_counter = 0
        self.prev_waypoints: Optional[np.ndarray] = None
        self.consec_ious: List[float] = []
        self._prev_fused_mask: Optional[np.ndarray] = None

        # Estimator hysteresis state (DL â†” FALLBACK)
        inf_cfg = cfg.get("inference", {})
        self.current_estimator: str = "DL"
        self.estimator_dwell: int = 0
        self.fallback_consec_count: int = 0
        self.min_dwell: int = int(inf_cfg.get("estimator_min_dwell", 10))
        self.fallback_consec_threshold: int = int(inf_cfg.get("fallback_consec_threshold", 5))
        self.estimator_switch_count: int = 0

        # Classical estimator skip/cache parameters
        self.classical_skip_alpha: float = float(inf_cfg.get("classical_skip_alpha", 0.8))
        self.classical_cache_frames: int = int(inf_cfg.get("classical_cache_frames", 2))

        # Segmentation skipping cadence
        self.seg_every = max(1, int(inf_cfg.get("seg_every", 2 if profile_name == "cpu" else 1)))
        self.last_raw_mask: Optional[np.ndarray] = None
        self.last_prob_smooth: Optional[np.ndarray] = None
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
        if (not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3
                or frame_bgr.shape[2] != 3 or min(frame_bgr.shape[:2]) == 0
                or frame_bgr.dtype != np.uint8):
            raise ValueError("Frame must be a nonempty uint8 BGR image with 3 channels")
        if not np.isfinite(dt) or dt <= 0 or dt > 1.0:
            raise ValueError("Frame dt must be finite and in (0, 1] seconds")
        t_stage0_frame = time.perf_counter()
        fh, fw = frame_bgr.shape[:2]
        mh, mw = self.input_size
        t_stage0 = time.perf_counter()

        # 1. PREPROCESS (Fast INTER_LINEAR letterbox and cached road color)
        scale = min(mw / fw, mh / fh)
        nw, nh = int(round(fw * scale)), int(round(fh * scale))
        if not self.cfg["data"].get("letterbox", False):
            nw, nh = mw, mh
        nw, nh = max(1, nw), max(1, nh)
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

        # Whole-frame brightness check for night mode (mean < 35 and p90 < 65, debounce 15 frames)
        if self.night_mode_enabled:
            if (frame_idx % 5 == 0) or (self.night_debounce_counter > 0):
                gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                self._last_mean_luma = float(np.mean(gray_small))
                self._last_p90_luma = float(np.percentile(gray_small, 90))

            if self.current_mode == "day":
                # Switch only if whole frame is dark
                if self._last_mean_luma < self.night_thresh_low and self._last_p90_luma < 65.0:
                    self.night_debounce_counter += 1
                    if self.night_debounce_counter >= self.night_debounce_frames:
                        self.current_mode = "night"
                        self.mode_switch_count += 1
                        self.night_debounce_counter = 0
                else:
                    self.night_debounce_counter = 0
            else:
                # Hysteresis to switch back to day
                if self._last_mean_luma >= self.night_thresh_high or self._last_p90_luma >= 80.0:
                    self.night_debounce_counter += 1
                    if self.night_debounce_counter >= self.night_debounce_frames:
                        self.current_mode = "day"
                        self.mode_switch_count += 1
                        self.night_debounce_counter = 0
                else:
                    self.night_debounce_counter = 0

            if hasattr(self.model, "set_mode"):
                self.model.set_mode(self.current_mode)

        # End preprocess timing
        t_prep_ms = (time.perf_counter() - t_stage0) * 1000.0
        self.timings["preprocess"] += t_prep_ms / 1000.0
        self.stage_samples["preprocess"].append(t_prep_ms)

        # 1b. YOLO DETECTIONS (runs early so detections suppress classical fallback mask)
        t_yolo0 = time.perf_counter()
        dets = self.detector.detect(frame_bgr, frame_idx)
        if self.hood_poly_norm is not None:
            hp_fw_fh = np.zeros_like(self.hood_poly_norm, dtype=np.int32)
            hp_fw_fh[:, 0] = self.hood_poly_norm[:, 0] * fw
            hp_fw_fh[:, 1] = self.hood_poly_norm[:, 1] * fh
            valid_dets = [d for d in dets if cv2.pointPolygonTest(hp_fw_fh, (d.center_x, d.center_y), False) < 0]
            dets = valid_dets
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_yolo_ms = (time.perf_counter() - t_yolo0) * 1000.0
        self.timings["yolo"] += t_yolo_ms / 1000.0
        self.stage_samples["YOLO"].append(t_yolo_ms)

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
        dets_padded = [
            Detection(
                x1=d.x1 + pad_left,
                y1=d.y1 + pad_top,
                x2=d.x2 + pad_left,
                y2=d.y2 + pad_top,
                cls_id=d.cls_id,
                name=d.name,
                conf=d.conf,
                kind=d.kind,
                approach_rate=d.approach_rate,
                tracker_id=d.tracker_id,
                last_update_frame=d.last_update_frame,
                consecutive_detections=d.consecutive_detections,
            )
            for d in dets_model
        ]

        # Check if segmentation forward pass is skipped
        skip_seg = (self.seg_every > 1) and (frame_idx % self.seg_every != 0) and (self.last_raw_mask is not None)

        if skip_seg:
            mask = self.last_raw_mask.copy()
            prob_smooth = self.last_prob_smooth.copy() if self.last_prob_smooth is not None else mask.astype(np.float32)
            unc = self.last_unc
            geo = self.last_geo
            tmp = self.last_tmp
            terrain = self.last_terrain
            active_str = self.last_active
            fused_alpha = self.last_fused_alpha
            self.stage_samples["model"].append(0.0)
            self.stage_samples["classical"].append(0.0)
        else:
            # Tensor conversion & Forward pass
            if self.onnx_session is not None:
                rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
                x_norm = rgb.astype(np.float32) * self.inv_255_std + self.sub_mean_std
                x_np = np.ascontiguousarray(x_norm.transpose(2, 0, 1)[np.newaxis, ...])
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

                logit_diff = seg_logits_np[0, 1] - seg_logits_np[0, 0]
                prob_road_padded = 1.0 / (1.0 + np.exp(-logit_diff))
                dl_mask_padded = (prob_road_padded >= 0.50).astype(np.uint8)
                terrain = 0

                t_mod_ms = (time.perf_counter() - t_stage0) * 1000.0
                self.timings["model"] += t_mod_ms / 1000.0
                self.stage_samples["model"].append(t_mod_ms)
            elif self.model is not None:
                if self.device.type == "cuda":
                    # GPU letterbox, norm and fp16
                    t_u8 = torch.from_numpy(padded).to(self.device, non_blocking=True)
                    x_tensor = t_u8.permute(2, 0, 1).unsqueeze(0).half() / 255.0
                    x_tensor = (x_tensor - torch.tensor(self.mean.reshape(3, 1, 1), device=self.device).half()) / torch.tensor(self.std.reshape(3, 1, 1), device=self.device).half()
                    torch.cuda.synchronize()
                else:
                    x = (cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - self.mean) / self.std
                    x_tensor = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self.device, memory_format=torch.channels_last)

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

                    logit_diff = (seg_logits[0, 1] - seg_logits[0, 0]).cpu().numpy()
                    prob_road_padded = 1.0 / (1.0 + np.exp(-logit_diff))
                    dl_mask_padded = (prob_road_padded >= 0.50).astype(np.uint8)
                    if self.cfg["inference"].get("use_terrain", False) and "terrain_logits" in out:
                        terrain = int(out["terrain_logits"].argmax(1)[0].item())
                    else:
                        terrain = 0

                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                t_mod_ms = (time.perf_counter() - t_stage0) * 1000.0
                self.timings["model"] += t_mod_ms / 1000.0
                self.stage_samples["model"].append(t_mod_ms)
            else:
                prob_road_padded = np.zeros((mh, mw), dtype=np.float32)
                dl_mask_padded = np.zeros((mh, mw), dtype=np.uint8)
                unc = 0.95
                terrain = 0

            # Crop letterbox padding and apply continuous temporal probability EMA
            prob_crop = prob_road_padded[pad_top:pad_top + nh, pad_left:pad_left + nw]
            if self.prev_prob_crop is None or self.prev_prob_crop.shape != prob_crop.shape:
                self.prev_prob_crop = prob_crop.copy()
            else:
                prob_crop = 0.40 * prob_crop + 0.60 * self.prev_prob_crop
                self.prev_prob_crop = prob_crop.copy()

            # Soft spatial Gaussian filtering and vertical morphological closing
            prob_crop = prob_crop.astype(np.float32)
            prob_smooth = cv2.GaussianBlur(prob_crop, (5, 5), 1.0)
            k_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 21))
            dl_mask = (prob_smooth >= 0.45).astype(np.uint8)
            dl_mask = cv2.morphologyEx(dl_mask, cv2.MORPH_CLOSE, k_vert)

            # 3. CLASSICAL & FUSION
            t_stage0 = time.perf_counter()
            geo = geometric_score(dl_mask, self.cfg["consistency"]["geometric"], apex_y=int(apex_y - pad_top))
            tmp = temporal_score(self.prev_dl_mask, dl_mask, speed, self.cfg["consistency"]["temporal"])
            self.prev_dl_mask = dl_mask.copy()

            alpha_val = self.fusion.alpha(unc, geo, tmp, terrain)
            fb_mask_padded, _ = self.classical.estimate_with_cache(
                padded,
                alpha=alpha_val,
                skip_alpha=self.classical_skip_alpha,
                cache_frames=self.classical_cache_frames,
                horizon_y=int(0.25 * mh),
                dets=dets_padded,
            )
            fb_mask = fb_mask_padded[pad_top:pad_top + nh, pad_left:pad_left + nw]

            fused = self.fusion.fuse(dl_mask, fb_mask, unc, geo, tmp, terrain)
            mask = fused.mask
            active_str = fused.active
            fused_alpha = fused.alpha
            t_cf_ms = (time.perf_counter() - t_stage0) * 1000.0
            self.timings["classical_fusion"] += t_cf_ms / 1000.0
            self.stage_samples["classical"].append(t_cf_ms)

            self.last_raw_mask = mask.copy()
            self.last_prob_smooth = prob_smooth.copy()
            self.last_fused_alpha = fused_alpha
            self.last_active = active_str
            self.last_unc = unc
            self.last_geo = geo
            self.last_tmp = tmp
            self.last_terrain = terrain

        # 4. POST-PROCESS AT MODEL RESOLUTION
        t_stage0 = time.perf_counter()
        k_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 21))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_vert)

        if self.hood_poly_model is not None:
            hp_model_adj = self.hood_poly_model.copy()
            hp_model_adj[:, 0] -= pad_left
            hp_model_adj[:, 1] -= pad_top
            cv2.fillPoly(mask, [hp_model_adj], 0)

        # Mask reflective strip at bottom center of hood ONLY if hood is present
        if self.hood_poly_model is not None:
            r_y1 = int(round(0.91 * nh))
            r_x1 = int(round(0.38 * nw))
            r_x2 = int(round(0.65 * nw))
            mask[r_y1:, r_x1:r_x2] = 0

        # Smart component retention: keep all significant road components in the forward corridor
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            kept_mask = np.zeros_like(mask)
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < 80:
                    continue
                cx, cy = centroids[i][0], centroids[i][1]
                # Retain if significant area (>800 px) or inside the forward driving corridor
                if area > 800 or (cy > 0.35 * nh and 0.15 * nw < cx < 0.85 * nw):
                    kept_mask[labels == i] = 1
            if np.count_nonzero(kept_mask) < 200:
                largest_i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                kept_mask = (labels == largest_i).astype(np.uint8)
            mask = kept_mask

        t_post_ms = (time.perf_counter() - t_stage0) * 1000.0
        self.timings["postprocess"] += t_post_ms / 1000.0
        self.stage_samples["postprocess"].append(t_post_ms)

        # 5. MODEL-RESOLUTION OBSTACLE SUBTRACTION & IPM
        t_stage0 = time.perf_counter()
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

        # 6. COMPUTE DECISION COMMAND
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
            ipm_valid=ipm_res.valid,
            prev_waypoints=self.prev_waypoints,
        )
        self.prev_steer_angle = cmd.angle
        self.prev_speed = cmd.speed
        self.prev_waypoints = getattr(cmd, "path_waypoints", None)
        if cmd.saturating_term in ("hold_angle", "decay_angle"):
            self.steer_hold_counter += 1
        else:
            self.steer_hold_counter = 0

        # 7. ESTIMATOR HYSTERESIS:
        # Require 5 consecutive frames below threshold before switching to FALLBACK.
        # Require 5 consecutive frames above threshold before switching back to DL.
        # Enforce minimum 10-frame dwell once in a mode.
        fallback_threshold = float(self.cfg["fusion"].get("fallback_threshold", 0.5))
        trust_is_low = (cmd.trust < fallback_threshold)

        if self.current_estimator == "DL":
            if trust_is_low:
                self.fallback_consec_count += 1
            else:
                self.fallback_consec_count = 0
            if self.fallback_consec_count >= self.fallback_consec_threshold and self.estimator_dwell >= self.min_dwell:
                self.current_estimator = "FALLBACK"
                self.estimator_dwell = 0
                self.estimator_switch_count += 1
                self.fallback_consec_count = 0
            else:
                self.estimator_dwell += 1
        else:  # current_estimator == "FALLBACK"
            if not trust_is_low:
                self.fallback_consec_count += 1
            else:
                self.fallback_consec_count = 0
            if self.fallback_consec_count >= self.fallback_consec_threshold and self.estimator_dwell >= self.min_dwell:
                self.current_estimator = "DL"
                self.estimator_dwell = 0
                self.estimator_switch_count += 1
                self.fallback_consec_count = 0
            else:
                self.estimator_dwell += 1

        # Display override:
        # If in FALLBACK mode via hysteresis OR trust-capped (<= 0.15), show "FALLBACK"
        if cmd.trust < fallback_threshold:
            active_str = "FALLBACK"
        else:
            active_str = "DL"

        # Track consecutive-frame mask IoU
        if self._prev_fused_mask is not None:
            inter = np.logical_and(mask > 0, self._prev_fused_mask > 0).sum()
            union = np.logical_or(mask > 0, self._prev_fused_mask > 0).sum()
            c_iou = float(inter / union) if union > 0 else 1.0
            self.consec_ious.append(c_iou)
        self._prev_fused_mask = mask.copy()

        # Upscale mask to output resolution ONCE only for final drawing
        mask_full = cv2.resize(mask, (fw, fh), interpolation=cv2.INTER_NEAREST)

        # Masked probability map for high-res smooth vector rendering
        clean_prob = prob_smooth * mask.astype(np.float32)

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
            prob_model=clean_prob,
            processing_ms=(time.perf_counter() - t_stage0_frame) * 1000.0,
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

    # 1. High-resolution antialiased butter-smooth mask overlay and vector contour
    if getattr(res, "prob_model", None) is not None and np.any(res.prob_model > 0.05):
        p_slice = res.prob_model
        mh, mw = p_slice.shape[:2]

        # Fast model-scale Continuous Temporal EMA
        if "disp_prob" not in state or state["disp_prob"].shape != (mh, mw):
            state["disp_prob"] = p_slice.copy()
        else:
            state["disp_prob"] = 0.25 * p_slice + 0.75 * state["disp_prob"]

        smooth_field = cv2.GaussianBlur(state["disp_prob"], (5, 5), 1.2)
        # Dual-threshold hysteresis at model scale: confident seeds (>= 0.45) & candidate road (>= 0.30)
        seed_mask = smooth_field >= 0.45
        cand_mask = (smooth_field >= 0.30).astype(np.uint8)
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 11))
        cand_mask = cv2.morphologyEx(cand_mask, cv2.MORPH_CLOSE, k_close)

        # Eliminate disconnected spurious floating blobs on buildings/trees at model scale
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cand_mask, connectivity=8)
        clean_mask = np.zeros_like(cand_mask)
        min_road_area = int(0.004 * mw * mh)
        ego_bottom_y = int(0.60 * mh)
        for lbl in range(1, num_labels):
            if stats[lbl, cv2.CC_STAT_AREA] < min_road_area:
                continue
            lbl_m = labels == lbl
            bottom_reach = stats[lbl, cv2.CC_STAT_TOP] + stats[lbl, cv2.CC_STAT_HEIGHT]
            if bottom_reach >= ego_bottom_y or np.any(seed_mask[lbl_m]):
                clean_mask[lbl_m] = 1

        # Upscale clean binary mask to full frame resolution
        high_mask = cv2.resize(clean_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        if np.any(high_mask):
            overlay = out.copy()
            overlay[high_mask > 0] = c_drivable
            cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)

            cnts, _ = cv2.findContours(high_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
            for c in cnts:
                if cv2.contourArea(c) > 1000:
                    eps = 0.0015 * cv2.arcLength(c, True)
                    smooth_c = cv2.approxPolyDP(c, eps, True)
                    cv2.drawContours(out, [smooth_c], -1, (255, 255, 255), thickness, cv2.LINE_AA)
    elif getattr(res, "mask_model", None) is not None:
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
                roi_m = cv2.resize(model_slice, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR) > 0.5
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

    # 3. Center path above the hood line using polynomial x(y) = a*y^2 + b*y + c
    band = cfg.get("inference", {}).get("decision", {}).get("lookahead_rows", [0.55, 0.9])
    y_lookahead = int(band[0] * p_h)
    y_road_bottom = max(0, min(p_h - 1, apex_p_y - 2))
    visible_road_h = y_road_bottom - y_lookahead
    scale_px = float(w) / float(p_w)
    scale_py = float(h) / float(p_h)

    path_wp = getattr(res.command, "path_waypoints", None)
    path_y = getattr(res.command, "path_y_evals", None)
    path_poly = getattr(res.command, "path_poly", None)
    path_y_range = getattr(res.command, "path_y_range", None)

    if path_wp is not None and path_y is not None and len(path_wp) >= 4:
        raw_pts = np.column_stack([path_wp * scale_px, path_y * scale_py]).astype(np.float32)
        x0, y0 = float(w / 2.0), float(h - 2)
        x1, y1 = float(raw_pts[-1, 0]), float(raw_pts[-1, 1])
        steer_deg = float(getattr(res.command, "angle", 0.0))
        steer_rad = np.radians(steer_deg)

        # Hermite spline from ego bumper (x0, y0) to lookahead horizon (x1, y1)
        num_pts = 25
        t = np.linspace(0.0, 1.0, num_pts, dtype=np.float32)
        y_dense = y0 + (y1 - y0) * t
        s_t = 3.0 * (t ** 2) - 2.0 * (t ** 3)

        # Road heading velocity at lookahead horizon (aligns path derivative with steering/road heading)
        v1 = -(y1 - y0) * float(np.tan(steer_rad))
        v1 = np.clip(v1, -0.8 * w, 0.8 * w)
        x_dense = (1.0 - s_t) * x0 + s_t * x1 + (t ** 3 - t ** 2) * v1
        dense_pts = np.column_stack([x_dense, y_dense]).astype(np.float32)

        alpha_smooth = float(np.clip(0.18 + 0.22 * (min(abs(steer_deg), 15.0) / 15.0), 0.18, 0.40))
        if "disp_pts" not in state or state["disp_pts"].shape != dense_pts.shape:
            state["disp_pts"] = dense_pts
        else:
            state["disp_pts"] = alpha_smooth * dense_pts + (1.0 - alpha_smooth) * state["disp_pts"]
        state["path_hold_count"] = 0
    elif state.get("disp_pts") is not None and state.get("path_hold_count", 0) < 5:
        state["path_hold_count"] = state.get("path_hold_count", 0) + 1
    elif path_poly is not None and path_y_range is not None and (path_y_range[1] - path_y_range[0]) > 10:
        y_top, y_bot = path_y_range
        a_p, b_p, c_p = path_poly
        x0, y0 = float(w / 2.0), float(h - 2)
        y1 = float(y_top * scale_py)
        x1 = float((a_p * (y_top ** 2) + b_p * y_top + c_p) * scale_px)
        steer_deg = float(getattr(res.command, "angle", 0.0))
        steer_rad = np.radians(steer_deg)

        num_pts = 25
        t = np.linspace(0.0, 1.0, num_pts, dtype=np.float32)
        y_dense = y0 + (y1 - y0) * t
        s_t = 3.0 * (t ** 2) - 2.0 * (t ** 3)
        v1 = -(y1 - y0) * float(np.tan(steer_rad))
        v1 = np.clip(v1, -0.8 * w, 0.8 * w)
        x_dense = (1.0 - s_t) * x0 + s_t * x1 + (t ** 3 - t ** 2) * v1
        dense_pts = np.column_stack([x_dense, y_dense]).astype(np.float32)

        alpha_smooth = float(np.clip(0.18 + 0.22 * (min(abs(steer_deg), 15.0) / 15.0), 0.18, 0.40))
        if "disp_pts" not in state or state["disp_pts"].shape != dense_pts.shape:
            state["disp_pts"] = dense_pts
        else:
            state["disp_pts"] = alpha_smooth * dense_pts + (1.0 - alpha_smooth) * state["disp_pts"]
        state["path_hold_count"] = 0
    else:
        state.pop("disp_pts", None)

    if "disp_pts" in state and state["disp_pts"] is not None and len(state["disp_pts"]) >= 3:
        pts = state["disp_pts"].astype(np.float32)

        # 1. Compute exact forward tangent vector at the trajectory tip
        vec = pts[-1] - pts[-2]
        norm = float(np.hypot(vec[0], vec[1]))
        if norm > 1e-3:
            u = vec / norm
        else:
            steer_rad = np.radians(float(getattr(res.command, "angle", 0.0)))
            u = np.array([np.sin(steer_rad), -np.cos(steer_rad)], dtype=np.float32)

        n = np.array([-u[1], u[0]], dtype=np.float32)

        scale = max(0.8, min(2.0, w / 848.0))
        arrow_len = 16.0 * scale
        arrow_width = 10.0 * scale
        notch_depth = 5.0 * scale

        # Arrowhead apex is EXACTLY the last point of the trajectory
        p_tip = pts[-1]
        p_right = p_tip - u * arrow_len + n * arrow_width
        p_notch = p_tip - u * (arrow_len - notch_depth)
        p_left = p_tip - u * arrow_len - n * arrow_width

        # 2. Draw the trajectory line seamlessly up to p_notch
        # Filter out any points closer to p_tip than p_notch to prevent back-tracking zigzags
        d_notch = arrow_len - notch_depth
        valid_mask = [float(np.dot(p_tip - pt, u)) > d_notch for pt in pts]
        valid_pts = pts[valid_mask]
        if len(valid_pts) > 0:
            line_pts = np.vstack([valid_pts, p_notch.reshape(1, 2)])
        else:
            line_pts = np.vstack([pts[0:1], p_notch.reshape(1, 2)])
        p_int = np.round(line_pts).astype(np.int32)

        cv2.polylines(out, [p_int], False, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.polylines(out, [p_int], False, (0, 255, 255), thickness + 1, cv2.LINE_AA)

        # 3. Draw the aerodynamic arrowhead
        poly = np.round(np.array([p_tip, p_right, p_notch, p_left])).astype(np.int32)
        cv2.drawContours(out, [poly], -1, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.fillPoly(out, [poly], (0, 255, 255), cv2.LINE_AA)
        cv2.polylines(out, [poly], True, (0, 0, 0), thickness, cv2.LINE_AA)

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
            is_threat = (d.y2 / float(h) >= 0.55) or getattr(d, "sustained_growth", False)
            if res.command.obstacle_flag and is_threat and (abs(d.center_x - w / 2.0) < corridor_hw):
                color = c_veh_obs
            elif getattr(d, "sustained_growth", False):
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

    state["label_rects"] = placed_rects
    state["hud_lines"] = lines
    return out


def _queue_put(q: "queue.Queue", item: Any, stop: threading.Event) -> bool:
    while not stop.is_set():
        try:
            q.put(item, timeout=0.1)
            return True
        except queue.Full:
            continue
    return False


class FrameGrabber(threading.Thread):
    def __init__(
        self,
        source: Union[int, str],
        out_q: "queue.Queue",
        stop: threading.Event,
        live: bool,
        max_frames: Optional[int] = None,
        max_width: Optional[int] = None,
        errors: Optional[List[BaseException]] = None,
    ) -> None:
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.cap.release()
            raise ValueError("Cannot open input video or camera")
        self.out_q = out_q
        self.stop = stop
        self.live = live
        self.max_frames = max_frames
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(self.fps) or self.fps <= 0:
            self.fps = 25.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not live else None
        self.max_width = max_width
        self.errors = errors if errors is not None else []

    def run(self) -> None:
        count = 0
        shape = None
        try:
            while not self.stop.is_set():
                t_r0 = time.perf_counter()
                ret, frame = self.cap.read()
                if not ret:
                    if count == 0 or self.live or (self.total_frames and count < self.total_frames):
                        raise ValueError("Input ended before expected frames or camera disconnected")
                    break
                if frame is None or frame.size == 0:
                    raise ValueError("Empty or corrupt input frame")
                if shape is not None and shape != frame.shape:
                    raise ValueError("Input frame size changed mid-stream")
                shape = frame.shape
                if self.max_width and frame.shape[1] > self.max_width:
                    scale = self.max_width / frame.shape[1]
                    frame = cv2.resize(frame, (self.max_width, max(1, int(round(frame.shape[0] * scale)))))
                t_read_ms = (time.perf_counter() - t_r0) * 1000.0
                if self.live and self.out_q.full():
                    try:
                        self.out_q.get_nowait()
                    except queue.Empty:
                        pass
                if not _queue_put(self.out_q, (count, frame, t_read_ms), self.stop):
                    break
                count += 1
                if self.max_frames and count >= self.max_frames:
                    break
        except BaseException as exc:
            self.errors.append(exc)
            logger.error("Capture failed: %s", exc)
            self.stop.set()
        finally:
            self.cap.release()
            _queue_put(self.out_q, None, self.stop)


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
                try:
                    item = self.in_q.get(timeout=0.1)
                except queue.Empty:
                    continue
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
                _queue_put(self.out_q, (idx, frame, pkt_tuple, res, is_skip), self.stop)
        except BaseException as e:
            logger.error("Inference thread failed: %s", e)
            self.errors.append(e)
            self.stop.set()
        finally:
            _queue_put(self.out_q, None, self.stop)


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
        out_dir: Optional[Path] = None,
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
        self.out_dir = out_dir if out_dir is not None else out_path.parent
        self.writer: Optional[cv2.VideoWriter] = None
        self.actual_out_path = out_path
        self.annot_state: Dict[str, Any] = {}
        self.fps_q: Deque[float] = collections.deque(maxlen=30)
        if grabber_fps and grabber_fps > 0:
            self.fps_q.append(float(grabber_fps))
        self.t_last_frame = time.perf_counter()

    def run(self) -> None:
        telemetry = None
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            telemetry = open(self.out_dir / "telemetry.jsonl", "w", encoding="utf-8")
            check_frame_indices = {
                1: "frame_01_idx1.jpg",
                26: "frame_02_idx26.jpg",
                51: "frame_03_idx51.jpg",
                76: "frame_04_idx76.jpg",
                101: "frame_05_idx101.jpg",
                126: "frame_06_idx126.jpg",
                151: "frame_07_idx151.jpg",
                176: "frame_08_idx176.jpg",
                201: "frame_09_idx201.jpg",
                226: "frame_10_idx226.jpg",
                237: "frame_11_idx237.jpg",
                244: "frame_12_idx244.jpg",
                259: "frame_13_idx259.jpg",
                266: "frame_14_idx266.jpg",
                276: "frame_15_idx276.jpg",
                285: "frame_16_idx285.jpg",
                299: "frame_17_idx300.jpg",
                300: "frame_17_idx300.jpg",
            }
            while not self.stop.is_set():
                try:
                    item = self.in_q.get(timeout=0.1)
                except queue.Empty:
                    continue
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

                # Export reference check frames to check_frames_v4
                if idx in check_frame_indices:
                    cf_dir = self.out_dir / "check_frames_v4"
                    cf_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(cf_dir / check_frame_indices[idx]), ann_frame)

                # Write
                t_w0 = time.perf_counter()
                if self.writer is None:
                    self.writer, self.actual_out_path = _open_writer(
                        self.out_path, self.grabber_fps, (ann_frame.shape[1], ann_frame.shape[0]), self.fourcc
                    )
                self.writer.write(ann_frame)
                record = {
                    "frame": idx, "angle": res.command.angle, "speed": res.command.speed,
                    "trust": res.command.trust, "obstacle": bool(res.command.obstacle_flag),
                    "active": res.active, "mode": res.mode, "processing_ms": res.processing_ms,
                    "fps": rolling_fps, "skipped": is_skip,
                    "boxes": [[d.x1,d.y1,d.x2,d.y2,d.cls_id] for d in res.detections],
                    "labels": self.annot_state.get("label_rects", []),
                    "hud": self.annot_state.get("hud_lines", []),
                    "path": self.annot_state.get("disp_pts", np.empty((0,2))).tolist(),
                }
                telemetry.write(json.dumps(record, allow_nan=False) + "\n")
                if idx % 10 == 0:
                    mask_dir = self.out_dir / "masks"
                    mask_dir.mkdir(exist_ok=True)
                    cv2.imwrite(str(mask_dir / f"{idx:06d}.png"), res.mask * 255)
                t_write_ms = (time.perf_counter() - t_w0) * 1000.0
                self.p.timings["write"] += t_write_ms / 1000.0
                self.p.stage_samples["write"].append(t_write_ms)

                t_now = time.perf_counter()
                self.fps_q.append(1.0 / max(1e-4, t_now - self.t_last_frame))
                self.t_last_frame = t_now

                _queue_put(self.out_packet_q, (idx, pkt, res, rolling_fps), self.stop)
        except BaseException as e:
            logger.error("Writer thread failed: %s", e)
            self.errors.append(e)
            self.stop.set()
        finally:
            if telemetry is not None:
                telemetry.close()
            if self.writer is not None:
                self.writer.release()
            _queue_put(self.out_packet_q, None, self.stop)


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
    out_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    if out_dir is not None:
        out_dir_path = Path(out_dir)
    else:
        out_dir_path = Path(output_path).parent
    if out_dir is not None:
        root = out_dir_path.resolve()
        for candidate in (output_path, log_features):
            if candidate is not None and not Path(candidate).resolve().is_relative_to(root):
                raise ValueError("All output files must be under --out-dir")
    if isinstance(source, str) and not source.lower().startswith(("rtsp://", "http://", "https://")):
        if not Path(source).is_file() and "%" not in source:
            raise FileNotFoundError("Input video not found")
    if checkpoint is not None and not Path(checkpoint).is_file():
        raise FileNotFoundError("Requested PyTorch checkpoint missing")
    out_dir_path.mkdir(parents=True, exist_ok=True)

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
    device = torch.device("cpu") if profile == "cpu" else get_device()
    inf = cfg["inference"]

    # Set torch threads to physical cores
    try:
        import psutil
        physical_cores = psutil.cpu_count(logical=False) or os.cpu_count() or 4
        try:
            p = psutil.Process()
            p.nice(psutil.HIGH_PRIORITY_CLASS)
        except Exception:
            pass
    except Exception:
        physical_cores = os.cpu_count() or 4
    torch.set_num_threads(physical_cores)
    try:
        cv2.setNumThreads(min(4, physical_cores))
    except Exception:
        pass

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
            poly = calibrate_or_load_hood_polygon(source, profile_dir=out_dir_path / "camera_profiles")
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
            # Support explicit --onnx-path override (injected as _onnx_path_override)
            _onnx_override = inf.get("_onnx_path_override")
            if _onnx_override:
                onnx_path = Path(_onnx_override)
            else:
                onnx_path = ck_path.with_suffix(".onnx")
            if use_int8:
                int8_p = ck_path.parent / f"{ck_path.stem}_int8.onnx"
                if int8_p.is_file():
                    onnx_path = int8_p

            if _onnx_override and not onnx_path.is_file():
                raise FileNotFoundError("Requested ONNX weights missing")
            if not onnx_path.is_file() and ck_path.is_file():
                from scripts.export_onnx import export_to_onnx
                onnx_path = export_to_onnx(str(ck_path), str(onnx_path), tuple(inf.get("input_size", [288, 512])), int8=use_int8)

            if onnx_path.is_file():
                opts = onnxruntime.SessionOptions()
                opts.intra_op_num_threads = min(6, physical_cores)
                opts.inter_op_num_threads = 1
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
            raise ValueError(f"Cannot load ONNX model: {exc}") from exc

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

    if model is None and onnx_session is None:
        ck = Path(checkpoint) if checkpoint else resolve_path(inf["checkpoint"])
        if ck.exists():
            model, _ = load_model_from_checkpoint(str(ck), device)
        else:
            raise FileNotFoundError("Segmentation weights missing; supply --model-path or --onnx-path")
        model.eval()
        if device.type == "cpu":
            model = model.to(memory_format=torch.channels_last)

    live = isinstance(source, int) or str(source).lower().startswith(("rtsp://", "http://", "https://"))
    detector = detector or build_detector(
        inf["detector"],
        str(out_dir_path),
        live_camera=live,
        device=device,
        use_tensorrt=use_tensorrt,
    )
    pipeline = LivePipeline(cfg, model, device, detector, profile_name=profile, onnx_session=onnx_session)

    stop, errors = threading.Event(), []
    frame_q = queue.Queue(maxsize=inf.get("queue_size", 8))
    result_q = queue.Queue(maxsize=inf.get("queue_size", 8))
    packet_q = queue.Queue(maxsize=inf.get("queue_size", 8))

    grabber = FrameGrabber(source, frame_q, stop, live, max_frames, max_width, errors)
    imu_reader = IMUReader(imu_csv) if imu_csv else None
    worker = InferenceWorker(pipeline, frame_q, result_q, stop, inf["frame_skip"], errors, grabber.total_frames, imu_reader=imu_reader)
    writer_worker = WriterWorker(
        result_q, packet_q, stop, pipeline, grabber.fps, Path(output_path), inf["video_fourcc"],
        grabber.total_frames, cfg, errors, out_dir=out_dir_path,
    )

    grabber.start()
    worker.start()
    writer_worker.start()

    packets = []
    serial_sender = SerialSender(port=serial_port, baud=serial_baud, dry_run=serial_dry_run) if (serial_port or serial_dry_run) else None

    # Steer debug CSV
    steer_debug_dir = out_dir_path
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
    sat_15_count = 0
    sat_20_count = 0
    sat_29_count = 0
    speed_zero_count = 0
    steer_zero_count = 0
    sign_agree_count = 0
    zero_det_count = 0
    zero_det_frames: List[int] = []
    obs_frames: List[int] = []

    try:
        while True:
            item = None
            try:
                item = packet_q.get(timeout=0.1)
            except queue.Empty:
                if errors or (stop.is_set() and not writer_worker.is_alive()):
                    break
                continue

            if item is None:
                break

            if errors:
                if serial_sender is not None:
                    serial_sender.emergency_stop("pipeline worker failed")
                break
            idx, pkt, res, rolling_fps = item

            # Fix: count FALLBACK when res.active == "FALLBACK"
            if res.active == "FALLBACK":
                fallback_count += 1
            if res.command.trust < cfg["fusion"]["fallback_threshold"]:
                alpha_below_thr_count += 1
            if res.command.obstacle_flag:
                obstacle_count += 1
                obs_frames.append(idx)
            det_count += len(res.detections)
            if len(res.detections) == 0:
                zero_det_count += 1
                zero_det_frames.append(idx)
            trust_list.append(res.command.trust)

            if abs(res.command.angle) < 1e-3:
                steer_zero_count += 1
            if res.command.speed < 0.05:
                speed_zero_count += 1
            dx = getattr(res.command, "lateral_target_dx", 0.0)
            ang = res.command.angle
            if (dx > 0 and ang > 0) or (dx < 0 and ang < 0) or (abs(dx) <= 0.5 and abs(ang) <= 0.5) or (abs(dx) < 1e-3 and abs(ang) < 1e-3):
                sign_agree_count += 1

            if abs(res.command.angle) >= 15.0:
                sat_15_count += 1
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

    sat_15_pct = (sat_15_count / total_pkts) * 100.0
    sat_20_pct = (sat_20_count / total_pkts) * 100.0
    sat_29_pct = (sat_29_count / total_pkts) * 100.0
    speed_zero_pct = (speed_zero_count / total_pkts) * 100.0
    fb_pct = (fallback_count / total_pkts) * 100.0
    alpha_below_pct = (alpha_below_thr_count / total_pkts) * 100.0
    obs_pct = (obstacle_count / total_pkts) * 100.0
    mean_dets = det_count / total_pkts
    t_min = min(trust_list) if trust_list else 0.0
    t_mean = sum(trust_list) / len(trust_list) if trust_list else 0.0
    t_max = max(trust_list) if trust_list else 0.0

    p95_fps = float(np.percentile(list(fps_q), 5)) if len(fps_q) > 0 else avg_fps

    steer_zero_pct = (steer_zero_count / total_pkts) * 100.0
    sign_agree_pct = (sign_agree_count / total_pkts) * 100.0
    zero_det_pct = (zero_det_count / total_pkts) * 100.0
    mode_switches = getattr(pipeline, "mode_switch_count", 0)
    est_switches = getattr(pipeline, "estimator_switch_count", 0)
    model_gt_150 = sum(1 for m in pipeline.stage_samples.get("model", []) if m > 150.0)

    print(f"Average FPS: {avg_fps:.2f} (P95 FPS: {p95_fps:.2f})")
    print(f"Trust (min/mean/max): {t_min:.3f} / {t_mean:.3f} / {t_max:.3f}")
    print(f"FALLBACK frames: {fb_pct:.1f}% (alpha < thr: {alpha_below_pct:.1f}%)")
    print(f"Estimator Switches (DL<->FALLBACK): {est_switches}")
    print(f"Obstacle Flag frames: {obs_pct:.1f}% ({len(obs_frames)} frames: {obs_frames[:15]}...)")
    print(f"Mean Detections/frame: {mean_dets:.2f}")
    print(f"Frames with 0 detections: {zero_det_pct:.1f}% ({zero_det_count}/{total_pkts} frames: {zero_det_frames[:15]}...)")
    print(f"Steering == 0.0 deg: {steer_zero_pct:.1f}% ({steer_zero_count}/{total_pkts} frames)")
    print(f"Steering (|angle| >= 15.0 deg): {sat_15_pct:.1f}%")
    print(f"Steering (|angle| >= 20.0 deg): {sat_20_pct:.1f}%")
    print(f"Steering saturation (|angle| >= 29.5 deg): {sat_29_pct:.1f}%")
    print(f"Speed == 0.0 m/s frames: {speed_zero_pct:.1f}% ({speed_zero_count}/{total_pkts} frames)")
    print(f"Frames with model time > 150ms: {model_gt_150}")
    print(f"Path/Steer sign agreement: {sign_agree_pct:.1f}%")
    print(f"Mode switches (day/night): {mode_switches} (Current: {pipeline.current_mode})")

    report_csv = out_dir_path / "timings.csv"
    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stage", "mean_ms", "p95_ms", "total_s"])
        for stage, stats in stage_stats.items():
            w.writerow([stage, stats["mean_ms"], stats["p95_ms"], stats["total_s"]])

    # Write run_summary.txt to out_dir_path
    summary_path = out_dir_path / "run_summary.txt"
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
        f"Average FPS:        {avg_fps:.2f}",
        f"P95 FPS:            {p95_fps:.2f}",
        f"Trust (min/mean/max): {t_min:.3f} / {t_mean:.3f} / {t_max:.3f}",
        f"FALLBACK frames:    {fb_pct:.1f}%",
        f"Estimator Switches: {est_switches}",
        f"Obstacle Flag:      {obs_pct:.1f}%",
        f"Detections / Frame: {mean_dets:.2f}",
        f"Frames 0 Detections: {zero_det_pct:.1f}% ({zero_det_count}/{total_pkts})",
        f"Steering == 0.0 deg: {steer_zero_pct:.1f}% ({steer_zero_count}/{total_pkts})",
        f"Steering (|angle| >= 15 deg): {sat_15_pct:.1f}%",
        f"Steering (|angle| >= 20 deg): {sat_20_pct:.1f}%",
        f"Steering Saturation (|angle| >= 29.5 deg): {sat_29_pct:.1f}%",
        f"Speed == 0.0 m/s:   {speed_zero_pct:.1f}% ({speed_zero_count}/{total_pkts})",
        f"Frames Model >150ms:{model_gt_150}",
        f"Path/Steer Sign Agreement: {sign_agree_pct:.1f}%",
        f"Mode Switches:      {mode_switches} (Current: {pipeline.current_mode})",
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
    ap.add_argument("--out-dir", default=None, help="Directory for output reports and frames")
    ap.add_argument("--model-path", default=None, dest="model_path",
                    help="Explicit path to a PyTorch .pt checkpoint (overrides --checkpoint and config)")
    ap.add_argument("--onnx-path", default=None, dest="onnx_path",
                    help="Explicit path to a .onnx model (forces ONNX backend; no renaming needed)")
    a = ap.parse_args()

    setup_logging()
    cfg = load_config(a.config)
    if a.frame_skip:
        cfg["inference"]["frame_skip"] = a.frame_skip
    src = int(a.source) if a.source.isdigit() else a.source

    # --model-path overrides --checkpoint; --onnx-path forces onnx backend
    checkpoint_arg = a.model_path or a.checkpoint
    backend_arg = a.backend
    use_onnx_arg = a.onnx
    if a.onnx_path:
        # Override ONNX discovery: inject path into config so run_video picks it up
        cfg.setdefault("inference", {})["_onnx_path_override"] = a.onnx_path
        use_onnx_arg = True
        if backend_arg is None:
            backend_arg = "onnx"

    result = run_video(
        cfg=cfg,
        source=src,
        output_path=str(Path(a.out_dir) / Path(a.output).name) if a.out_dir else str(resolve_path(a.output)),
        checkpoint=checkpoint_arg,
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
        use_onnx=use_onnx_arg,
        use_int8=a.int8,
        backend=backend_arg,
        seg_every=a.seg_every,
        out_dir=a.out_dir,
    )
    if result.get("error"):
        raise ValueError(result["error"])


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, MemoryError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

