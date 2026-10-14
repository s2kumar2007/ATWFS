"""Steering and speed control law with ego-lane extraction, pure pursuit, and rate limiting.

Features:
- Ego-lane contiguous run selection with +/-1.8m window on wide multi-lane roads (>4.0m)
- Pure pursuit steering angle: ``atan(2 * wheelbase * sin(alpha) / lookahead)``
- Exponential moving average (EMA) smoothing and per-frame angular rate limiting
- Obstacle footprint subtraction and proximity/approach-rate speed decay
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from inference.detector import Detection


@dataclass
class Command:
    angle: float
    speed: float
    trust: float
    obstacle_flag: bool
    width_frac: float
    centroid_x: float
    obstacle_proximity: float
    width_m: Optional[float] = None
    offset_m: Optional[float] = None
    heading_error_deg: float = 0.0
    raw_angle_deg: float = 0.0
    clamped_angle_deg: float = 0.0
    saturating_term: str = "none"

    @property
    def packet(self) -> str:
        return f"[{self.angle:+.2f}, {self.speed:.3f}, {self.trust:.3f}, {int(self.obstacle_flag)}]"


def subtract_obstacles(mask: np.ndarray, dets: List[Detection], dilate_px: int) -> np.ndarray:
    """Zero out obstacle bounding boxes (dilated by ``dilate_px``) from the drivable mask."""
    if not dets:
        return mask
    h, w = mask.shape
    out = mask.copy()
    for d in dets:
        if d.height < 0.04 * h:
            continue
        x1 = max(0, d.x1 - dilate_px)
        x2 = min(w, d.x2 + dilate_px)
        y1 = max(0, d.y1)
        y2 = min(h, d.y2 + dilate_px)
        out[y1:y2, x1:x2] = 0
    return out


def extract_ego_lane_target_px(
    band_mask: np.ndarray,
    w: int,
    bev_width_m: float = 10.0,
    max_lane_width_m: float = 4.0,
    ego_window_half_m: float = 1.8,
) -> Tuple[float, float, float]:
    """Find the ego-lane target x-coordinate (px), run width (m), and lateral offset (m) in image/BEV band.

    Chooses the contiguous run of drivable columns containing the vehicle centre line (w/2)
    or nearest to it. If the run is wider than ``max_lane_width_m`` (4.0 m), uses a window
    of ``+/- ego_window_half_m`` (1.8 m) around the point in that run nearest ``w/2``.
    """
    ego_center_px = w / 2.0
    col_active = (band_mask > 0).any(axis=0)
    nz_cols = np.where(col_active)[0]
    if len(nz_cols) == 0:
        return ego_center_px, 0.0, 0.0

    px_to_m = bev_width_m / max(1.0, float(w))

    diffs = np.diff(nz_cols)
    split_pts = np.where(diffs > 1)[0] + 1
    runs = np.split(nz_cols, split_pts)

    best_run = None
    min_dist = float("inf")
    for r in runs:
        if len(r) == 0:
            continue
        r_start, r_end = float(r[0]), float(r[-1])
        if r_start <= ego_center_px <= r_end:
            best_run = r
            break
        d = min(abs(r_start - ego_center_px), abs(r_end - ego_center_px))
        if d < min_dist:
            min_dist = d
            best_run = r

    if best_run is None or len(best_run) == 0:
        return ego_center_px, 0.0, 0.0

    r_start, r_end = float(best_run[0]), float(best_run[-1])
    run_width_px = r_end - r_start + 1.0
    run_width_m = run_width_px * px_to_m
    run_center_px = (r_start + r_end) / 2.0

    if run_width_m > max_lane_width_m:
        window_half_px = ego_window_half_m / max(1e-6, px_to_m)
        min_allowed = r_start + window_half_px
        max_allowed = r_end - window_half_px
        if min_allowed < max_allowed:
            target_px = float(np.clip(ego_center_px, min_allowed, max_allowed))
        else:
            target_px = run_center_px
    else:
        target_px = run_center_px

    offset_m = (target_px - ego_center_px) * px_to_m
    return target_px, run_width_m, offset_m


def pure_pursuit_angle_deg(
    lateral_offset_m: float,
    lookahead_m: float = 6.0,
    wheelbase_m: float = 2.5,
) -> Tuple[float, float]:
    """Compute (heading_error_deg, pure_pursuit_steer_deg) from lateral offset and lookahead."""
    ld = max(0.5, float(lookahead_m))
    wb = max(0.2, float(wheelbase_m))
    alpha = math.atan2(float(lateral_offset_m), ld)
    heading_err_deg = math.degrees(alpha)
    steer_rad = math.atan((2.0 * wb * math.sin(alpha)) / ld)
    steer_deg = math.degrees(steer_rad)
    return heading_err_deg, steer_deg


def compute_command(
    mask: np.ndarray,
    dets: List[Detection],
    trust: float,
    terrain: int,
    cfg: Dict[str, Any],
    raw_mask: Optional[np.ndarray] = None,
    road_width_m: Optional[float] = None,
    lateral_offset_m: Optional[float] = None,
    prev_angle: Optional[float] = None,
    hold_count: int = 0,
    prev_speed: Optional[float] = None,
) -> Command:
    h, w = mask.shape

    nz_y, nz_x = np.nonzero(mask)
    if len(nz_y) == 0:
        if prev_angle is not None and hold_count < 5:
            held = float(prev_angle)
            return Command(
                angle=held, speed=0.0, trust=trust, obstacle_flag=False, width_frac=0.0,
                centroid_x=w / 2.0, obstacle_proximity=0.0, width_m=road_width_m, offset_m=lateral_offset_m,
                heading_error_deg=0.0, raw_angle_deg=held, clamped_angle_deg=held, saturating_term="hold_angle",
            )
        return Command(0.0, 0.0, trust, False, 0.0, w / 2.0, 0.0, width_m=road_width_m, offset_m=lateral_offset_m)

    bottom_y = int(np.max(nz_y))

    band = cfg["lookahead_rows"]
    y_start = int(band[0] * h)
    y_end = int(band[1] * h)
    if y_end > bottom_y:
        y_end = bottom_y
    if y_start >= y_end:
        y_start = max(0, y_end - max(4, int(0.1 * h)))

    band_mask = mask[y_start:y_end]
    cx, img_run_width_m, img_offset_m = extract_ego_lane_target_px(
        band_mask,
        w,
        bev_width_m=float(cfg.get("bev_width_m", 10.0)),
        max_lane_width_m=float(cfg.get("max_ego_lane_width_m", 4.0)),
        ego_window_half_m=float(cfg.get("ego_window_half_m", 1.8)),
    )

    # If ego-lane corridor in lookahead band is missing, hold angle for up to 5 frames
    if img_run_width_m <= 0.0 and (road_width_m is None or road_width_m < 1.0):
        if prev_angle is not None and hold_count < 5:
            held = float(prev_angle)
            return Command(
                angle=held, speed=0.0, trust=trust, obstacle_flag=False, width_frac=0.0,
                centroid_x=cx, obstacle_proximity=0.0, width_m=road_width_m, offset_m=lateral_offset_m,
                heading_error_deg=0.0, raw_angle_deg=held, clamped_angle_deg=held, saturating_term="hold_angle",
            )

    # Width fraction at row just above hood
    row_idx = max(0, bottom_y - 5)
    row_nz = np.nonzero(mask[row_idx])[0]
    if len(row_nz) > 0:
        width_px = row_nz[-1] - row_nz[0]
        width_frac = float(np.clip(width_px / w, 0.0, 1.0))
    else:
        width_frac = 0.0

    # Pure pursuit steering angle
    lookahead_m = float(cfg.get("lookahead_m", cfg.get("bev_lookahead_m", 6.0)))
    wheelbase_m = float(cfg.get("wheelbase_m", 2.5))
    max_steer = float(cfg.get("max_steer_deg", 30.0))

    # Only use IPM lateral_offset_m if road_width_m is valid (>= 2.5m)
    use_ipm_offset = (
        lateral_offset_m is not None
        and (road_width_m is not None and road_width_m >= 2.5)
    )
    eff_offset_m = float(lateral_offset_m) if use_ipm_offset else float(img_offset_m)
    eff_width_m = road_width_m if (road_width_m is not None and road_width_m >= 2.5) else img_run_width_m

    heading_err_deg, raw_angle_deg = pure_pursuit_angle_deg(eff_offset_m, lookahead_m, wheelbase_m)
    clamped_angle_deg = max(-max_steer, min(max_steer, raw_angle_deg))

    # EMA smoothing and per-frame angular rate limiter
    saturating_term = "none"
    if abs(raw_angle_deg) >= max_steer - 0.5:
        saturating_term = "lateral_offset"

    if prev_angle is not None:
        ema_alpha = float(cfg.get("steer_ema_alpha", 0.4))
        max_rate = float(cfg.get("max_steer_rate_deg", 5.0))
        ema_angle = ema_alpha * clamped_angle_deg + (1.0 - ema_alpha) * float(prev_angle)
        delta = ema_angle - float(prev_angle)
        if abs(delta) > max_rate:
            saturating_term = "rate_limit"
            delta = max(-max_rate, min(max_rate, delta))
        angle_deg = max(-max_steer, min(max_steer, float(prev_angle) + delta))
    else:
        angle_deg = clamped_angle_deg

    # Obstacle check: confirmed detection (>= 2 consecutive real detections), inside ego corridor, and close enough
    corridor = float(cfg.get("corridor_half_width", 0.2)) * w
    obs_flag = False
    prox = 0.0

    for d in dets:
        in_corridor = abs(d.center_x - w / 2.0) < corridor
        rel_y = d.y2 / h
        if in_corridor and rel_y > prox:
            prox = rel_y

        is_confirmed = (d.tracker_id is None) or (getattr(d, "consecutive_detections", 1) >= 2)
        is_close_or_growing = (rel_y >= 0.55) or (d.approach_rate > 1.0)
        if is_confirmed and in_corridor and is_close_or_growing:
            obs_flag = True

    # Speed calculation
    v_max = float(cfg.get("v_max", 1.5))
    has_slow_floor = ("slow_speed_floor" in cfg) and (cfg["slow_speed_floor"] is not None)
    slow_floor = float(cfg.get("slow_speed_floor", 0.0) or 0.0)

    speed = v_max
    if trust < float(cfg.get("trust_min", 0.3)):
        speed = slow_floor if has_slow_floor else 0.0

    speed *= cfg.get("terrain_speed_factor", [1.0, 0.7, 0.5, 0.3])[terrain]
    speed *= max(0.5, trust)

    # Width handicap
    if eff_width_m is not None and eff_width_m >= 2.5:
        ref_w = float(cfg.get("road_width_ref_m", 3.5))
        if eff_width_m < ref_w:
            speed *= max(0.3, eff_width_m / ref_w)
    elif width_frac < cfg.get("width_ref", 0.35):
        speed *= max(0.3, width_frac / max(1e-4, cfg.get("width_ref", 0.35)))

    # Obstacle speed drop
    if obs_flag:
        prox_stop = float(cfg.get("prox_stop", 0.85))
        prox_slow = float(cfg.get("prox_slow", 0.55))
        factor = float(np.clip((prox_stop - prox) / max(0.01, prox_stop - prox_slow), 0.0, 1.0))
        speed *= factor

        confirmed_growing = [
            d.approach_rate for d in dets
            if abs(d.center_x - w / 2.0) < corridor and (
                (d.tracker_id is None) or (getattr(d, "consecutive_detections", 1) >= 2)
            )
        ]
        if confirmed_growing:
            max_app = max(confirmed_growing)
            if max_app > 1.05:
                speed *= max(0.1, 1.0 - (max_app - 1.0) * 5.0)

    # If no obstacle, speed floor ensures car doesn't stall on valid road
    if not obs_flag and has_slow_floor and trust >= float(cfg.get("trust_min", 0.3)) and speed < slow_floor:
        speed = slow_floor

    # Ramp speed instead of stopping instantly
    if prev_speed is not None:
        max_ramp = float(cfg.get("max_speed_ramp", 0.15))
        delta_v = speed - float(prev_speed)
        delta_v = max(-max_ramp, min(max_ramp, delta_v))
        speed = max(0.0, float(prev_speed) + delta_v)

    if speed < 0.05:
        speed = 0.0

    return Command(
        angle=angle_deg,
        speed=speed,
        trust=trust,
        obstacle_flag=obs_flag,
        width_frac=width_frac,
        centroid_x=cx,
        obstacle_proximity=prox,
        width_m=eff_width_m,
        offset_m=eff_offset_m,
        heading_error_deg=heading_err_deg,
        raw_angle_deg=raw_angle_deg,
        clamped_angle_deg=clamped_angle_deg,
        saturating_term=saturating_term,
    )
