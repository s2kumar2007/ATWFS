"""Steering and speed control law with ego-lane extraction, pure pursuit, and rate limiting.

Features:
- Ego-lane contiguous run selection with +/-1.8m window on wide multi-lane roads (>4.0m)
- Pure pursuit steering angle: ``atan(2 * wheelbase * sin(alpha) / lookahead)``
- Fixed metric lookahead (default 4.0 m) evaluated at the corresponding image row
- Exponential moving average (EMA) smoothing and per-frame angular rate limiting
- Obstacle footprint subtraction and proximity/approach-rate speed decay
- Low-visibility: hold angle 5 frames then decay, crawl rather than stop
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
    path_poly: Optional[Tuple[float, float, float]] = None
    path_y_range: Optional[Tuple[int, int]] = None
    lateral_target_dx: float = 0.0  # pixels, positive = target is right of ego centre
    held_angle: float = 0.0         # angle held/decayed during low-visibility
    path_waypoints: Optional[np.ndarray] = None
    path_y_evals: Optional[np.ndarray] = None

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
    lookahead_m: float = 4.0,
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


def _monotonic_path_pts(c_pts: List[Tuple[float, float]], ego_center_px: float) -> List[Tuple[float, float]]:
    """Return c_pts filtered so the x-values are monotonic (no sudden reversals > 30 px).

    Iterates from the bottom (high y) upward, dropping any point that would create a hook
    (reversal of more than 30 px relative to the previous kept point).
    """
    if len(c_pts) < 2:
        return c_pts
    kept = [c_pts[0]]
    for pt in c_pts[1:]:
        prev = kept[-1]
        # Direction from bottom → current
        dx_new = pt[1] - prev[1]
        if len(kept) >= 2:
            dx_old = kept[-1][1] - kept[-2][1]
            # Drop if the direction reverses by more than 30 px
            if dx_old * dx_new < 0 and abs(dx_new) > 30.0:
                continue
        kept.append(pt)
    return kept


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
    ipm_valid: bool = False,
    prev_waypoints: Optional[np.ndarray] = None,
) -> Command:
    if (not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.size == 0
            or not np.isfinite(mask).all() or not math.isfinite(float(trust))
            or not 0 <= float(trust) <= 1 or terrain not in range(4)):
        return Command(0.0, 0.0, 0.0, True, 0.0, 0.0, 0.0)
    h, w = mask.shape
    ego_center_px = w / 2.0

    if trust < float(cfg.get("trust_min", 0.3)) and cfg.get("slow_speed_floor") is not None:
        return _low_vis_command(trust, dets, cfg, prev_angle, hold_count, prev_speed, h, w)
    nz_y, nz_x = np.nonzero(mask)
    if len(nz_y) == 0:
        # No drivable pixels at all — hold/decay angle, crawl
        return _low_vis_command(
            trust=min(trust, 0.15), dets=dets, cfg=cfg,
            prev_angle=prev_angle, hold_count=hold_count,
            prev_speed=prev_speed, h=h, w=w,
            road_width_m=road_width_m, lateral_offset_m=lateral_offset_m,
        )

    max_road_y = int(np.max(nz_y))
    band = cfg["lookahead_rows"]
    y_start = int(band[0] * h)
    # Stop bottom_y at band[1]*h to avoid hood-split distortion
    bottom_y = min(int(band[1] * h), max_road_y)
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

    # Check if DL mask covers < 10% of visible road (low-visibility branch)
    y_eval_start = int(0.45 * h)
    y_eval_end = int(0.75 * h)
    eval_area = max(1, (y_eval_end - y_eval_start) * int(0.6 * w))
    eval_roi = mask[y_eval_start:y_eval_end, int(0.2 * w):int(0.8 * w)]
    upper_frac = float(np.count_nonzero(eval_roi)) / float(eval_area)
    if upper_frac < 0.05:
        return _low_vis_command(
            trust=min(trust, 0.15), dets=dets, cfg=cfg,
            prev_angle=prev_angle, hold_count=hold_count,
            prev_speed=prev_speed, h=h, w=w,
            road_width_m=road_width_m, lateral_offset_m=lateral_offset_m,
            cx=cx,
        )

    # If ego-lane corridor in lookahead band is missing, use low-vis branch (up to 5 frames)
    if img_run_width_m <= 0.0 and (road_width_m is None or road_width_m < 1.0):
        if prev_angle is not None and hold_count < 5:
            return _low_vis_command(
                trust=trust, dets=dets, cfg=cfg,
                prev_angle=prev_angle, hold_count=hold_count,
                prev_speed=prev_speed, h=h, w=w,
                road_width_m=road_width_m, lateral_offset_m=lateral_offset_m,
                cx=cx,
            )

    px_to_m = float(cfg.get("bev_width_m", 10.0)) / max(1.0, float(w))
    lane_width_m = float(cfg.get("max_ego_lane_width_m", 3.5))
    lane_width_px = lane_width_m / max(1e-6, px_to_m)
    lane_half_px = lane_width_px / 2.0

    # 1. Extract per-row ego-lane waypoints across visible road band
    n_pts = 16
    y_evals = np.linspace(bottom_y, y_start, n_pts).astype(int)
    curr_x = ego_center_px
    raw_waypoints = []

    for y in y_evals:
        row = mask[y]
        nz = np.where(row > 0)[0]
        if len(nz) < 2:
            raw_waypoints.append(curr_x)
            continue
        diffs = np.diff(nz)
        splits = np.where(diffs > 1)[0] + 1
        runs = np.split(nz, splits)

        best_run = None
        min_dist = float("inf")
        for r in runs:
            if len(r) == 0:
                continue
            r_start, r_end = float(r[0]), float(r[-1])
            if r_start <= curr_x <= r_end:
                best_run = r
                break
            d = min(abs(r_start - curr_x), abs(r_end - curr_x))
            if d < min_dist:
                min_dist = d
                best_run = r

        if best_run is not None and len(best_run) > 0:
            r_start, r_end = float(best_run[0]), float(best_run[-1])
            r_w = r_end - r_start
            r_center = (r_start + r_end) / 2.0
            if r_w > 1.2 * lane_width_px:
                # Smooth continuous transition based on road width
                w_blend = min(1.0, max(0.0, (r_w - 1.2 * lane_width_px) / max(1.0, 0.8 * lane_width_px)))
                ego_target = float(np.clip(curr_x, r_start + lane_half_px, r_end - lane_half_px))
                target_x = (1.0 - w_blend) * r_center + w_blend * ego_target
            else:
                target_x = r_center
            curr_x = 0.65 * curr_x + 0.35 * target_x
            raw_waypoints.append(curr_x)
        else:
            raw_waypoints.append(curr_x)

    raw_wp = np.array(raw_waypoints, dtype=np.float32)

    # 2. Temporal exponential moving average on waypoints across frames
    if prev_waypoints is not None and len(prev_waypoints) == len(raw_wp):
        wp_alpha = float(cfg.get("trajectory_ema_alpha", 0.35))
        smooth_wp = wp_alpha * raw_wp + (1.0 - wp_alpha) * np.array(prev_waypoints, dtype=np.float32)
    else:
        smooth_wp = raw_wp

    # 3. Fit polynomial for smooth trajectory evaluation, legacy consumers, and obstacle checking
    poly_tuple: Optional[Tuple[float, float, float]] = None
    try:
        poly_deg = 2 if len(smooth_wp) >= 4 else 1
        coeffs = np.polyfit(y_evals, smooth_wp, poly_deg)
        if poly_deg == 1:
            poly_tuple = (0.0, float(coeffs[0]), float(coeffs[1]))
        else:
            poly_tuple = (float(coeffs[0]), float(coeffs[1]), float(coeffs[2]))
    except Exception:
        poly_tuple = None

    # Evaluate smooth parabolic trajectory from polynomial fit
    if poly_tuple is not None:
        a_p, b_p, c_p = poly_tuple
        traj_wp = a_p * (y_evals ** 2) + b_p * y_evals + c_p
        traj_wp[0] = 0.6 * traj_wp[0] + 0.4 * ego_center_px
    else:
        traj_wp = smooth_wp

    lookahead_m = float(cfg.get("lookahead_m", 4.0))
    wheelbase_m = float(cfg.get("wheelbase_m", 2.5))
    max_steer = float(cfg.get("max_steer_deg", 30.0))

    # Lookahead target evaluated from smooth trajectory waypoints
    delta_x = float(traj_wp[-1] - ego_center_px)

    # Width fraction at row just above hood
    row_idx = max(0, bottom_y - 5)
    row_nz = np.nonzero(mask[row_idx])[0]
    if len(row_nz) > 0:
        width_px = row_nz[-1] - row_nz[0]
        width_frac = float(np.clip(width_px / w, 0.0, 1.0))
    else:
        width_frac = 0.0

    eff_offset_m = delta_x * px_to_m
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

    # Enforce sign agreement between drawn path lateral target and steering angle
    if abs(delta_x) > 0.5:
        if delta_x > 0 and angle_deg < 0:
            angle_deg = min(max_steer, max(0.5, clamped_angle_deg * 0.5))
        elif delta_x < -0.5 and angle_deg > 0:
            angle_deg = max(-max_steer, min(-0.5, clamped_angle_deg * 0.5))

    # Safety rate limit applies after every display/sign correction too.
    if prev_angle is not None:
        max_rate = float(cfg.get("max_steer_rate_deg", 5.0))
        angle_deg = float(np.clip(angle_deg, prev_angle - max_rate, prev_angle + max_rate))

    # Obstacle check
    obs_flag, prox = _check_obstacles(dets, h, poly_tuple, ego_center_px, cx, y_start)

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

    # Steering-angle speed scaling: high angle → slow down
    angle_factor_min = float(cfg.get("speed_angle_factor_min", 0.4))
    angle_denom = float(cfg.get("speed_angle_denom", 45.0))
    speed *= max(angle_factor_min, 1.0 - abs(angle_deg) / angle_denom)

    # Obstacle speed drop
    if obs_flag:
        prox_stop = float(cfg.get("prox_stop", 0.85))
        prox_slow = float(cfg.get("prox_slow", 0.55))
        factor = float(np.clip((prox_stop - prox) / max(0.01, prox_stop - prox_slow), 0.0, 1.0))
        speed *= factor

        confirmed_growing = [
            d.approach_rate for d in dets
            if (getattr(d, "sustained_growth", False) or d.approach_rate > 1.05) and (d.y2 / float(h) >= 0.40)
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
        path_poly=poly_tuple,
        path_y_range=(y_start, bottom_y),
        lateral_target_dx=delta_x,
        held_angle=0.0,
        path_waypoints=traj_wp,
        path_y_evals=y_evals,
    )


def _check_obstacles(
    dets: List[Detection],
    h: int,
    poly_tuple: Optional[Tuple[float, float, float]],
    ego_center_px: float,
    cx: float,
    y_start: int,
) -> Tuple[bool, float]:
    """Return (obstacle_flag, proximity) from detections vs ego corridor."""
    obs_flag = False
    prox = 0.0
    for d in dets:
        rel_y = d.y2 / float(h)
        if rel_y < 0.40:
            continue
        y_eval = min(h - 1, max(0, d.y2))
        if poly_tuple is not None:
            a_p, b_p, c_p = poly_tuple
            x_center_path = a_p * (y_eval ** 2) + b_p * y_eval + c_p
        else:
            x_center_path = ego_center_px + (cx - ego_center_px) * ((h - y_eval) / max(1.0, h - y_start))

        w_corr = max(0.12 * (d.x2 - d.x1 + d.width) / max(1, d.width) * 100,
                     (0.08 + 0.14 * (y_eval / float(h))) * (ego_center_px * 2))
        # simpler: use corridor half-width in pixels
        w_corr = max(0.12 * ego_center_px * 2, (0.08 + 0.14 * (y_eval / float(h))) * ego_center_px * 2)
        corr_left = x_center_path - w_corr
        corr_right = x_center_path + w_corr

        in_corridor = (d.x2 > corr_left) and (d.x1 < corr_right)
        if in_corridor and rel_y > prox:
            prox = rel_y

        is_confirmed = (d.tracker_id is None) or (getattr(d, "consecutive_detections", 1) >= 2) or rel_y >= 0.85
        has_sustained_growth = getattr(d, "sustained_growth", False) or (d.approach_rate > 1.05 and d.tracker_id is None)
        is_threat = (rel_y >= 0.55) or has_sustained_growth

        if is_confirmed and in_corridor and is_threat:
            obs_flag = True
    return obs_flag, prox


def _low_vis_command(
    trust: float,
    dets: List[Detection],
    cfg: Dict[str, Any],
    prev_angle: Optional[float],
    hold_count: int,
    prev_speed: Optional[float],
    h: int,
    w: int,
    road_width_m: Optional[float] = None,
    lateral_offset_m: Optional[float] = None,
    cx: Optional[float] = None,
) -> Command:
    """Low-visibility fallback: hold angle 5 frames then decay to 0, crawl instead of stop."""
    ego_center_px = w / 2.0
    if cx is None:
        cx = ego_center_px

    hold_frames = int(cfg.get("low_vis_hold_frames", 5))
    decay_frames = int(cfg.get("low_vis_decay_frames", 10))
    crawl_floor = float(cfg.get("crawl_speed_floor", cfg.get("slow_speed_floor", 0.20)))
    max_steer = float(cfg.get("max_steer_deg", 30.0))

    if prev_angle is not None:
        if hold_count < hold_frames:
            # Hold last valid angle
            held = float(prev_angle)
            saturating_term = "hold_angle"
        else:
            # Decay towards 0 over decay_frames frames
            decay_steps = hold_count - hold_frames + 1
            decay_factor = max(0.0, 1.0 - decay_steps / max(1, decay_frames))
            held = float(prev_angle) * decay_factor
            saturating_term = "decay_angle"
        angle_deg = max(-max_steer, min(max_steer, held))
    else:
        held = 0.0
        angle_deg = 0.0
        saturating_term = "hold_angle"

    # Still compute obstacle check for safety
    obs_flag = False
    if dets:
        # Quick corridor check without poly
        for d in dets:
            rel_y = d.y2 / float(h)
            if rel_y >= 0.55 and abs(d.center_x - ego_center_px) < 0.25 * w:
                obs_flag = True
                break

    # Crawl speed (not zero), unless obstacle confirmed close
    speed = crawl_floor
    if obs_flag:
        speed = 0.0
    elif prev_speed is not None:
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
        width_frac=0.0,
        centroid_x=cx,
        obstacle_proximity=0.0,
        width_m=road_width_m,
        offset_m=lateral_offset_m,
        heading_error_deg=0.0,
        raw_angle_deg=angle_deg,
        clamped_angle_deg=angle_deg,
        saturating_term=saturating_term,
        path_poly=None,
        path_y_range=None,
        lateral_target_dx=0.0,
        held_angle=angle_deg,
        path_waypoints=None,
        path_y_evals=None,
    )
