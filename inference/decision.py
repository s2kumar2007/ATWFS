"""Steering + speed decision logic and obstacle subtraction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

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

    @property
    def packet(self) -> Tuple[float, float, float, int]:
        return (self.angle, self.speed, self.trust, int(self.obstacle_flag))


def subtract_obstacles(mask: np.ndarray, dets: Sequence[Detection], dilate_px: int) -> np.ndarray:
    out = mask.copy()
    h, w = out.shape
    for d in dets:
        box_h = d.y2 - d.y1
        if box_h < 0.04 * h:
            continue
        
        # B1: remove shadow, just subtract the box expanded by dilate_px
        out[max(0, d.y1 - dilate_px):min(h, d.y2 + dilate_px + 1), max(0, d.x1 - dilate_px):min(w, d.x2 + dilate_px + 1)] = 0
    return out


def drivable_geometry(mask: np.ndarray, band: Sequence[float]) -> Tuple[float, float]:
    h, w = mask.shape
    sub = mask[int(band[0] * h):max(int(band[0] * h) + 1, int(band[1] * h))] > 0
    if not sub.any():
        return w / 2.0, 0.0
    cols = np.nonzero(sub)[1]
    return float(cols.mean()), float(np.median(sub.sum(axis=1)) / w)


def assess_obstacles(dets: Sequence[Detection], frame_shape: Tuple[int, int], centroid_x: float,
                     cfg: Dict[str, Any]) -> Tuple[bool, float, float]:
    h, w = frame_shape
    lo, hi = centroid_x - cfg["corridor_half_width"] * w, centroid_x + cfg["corridor_half_width"] * w
    prox = 0.0
    max_approach = 0.0
    
    for d in dets:
        # check overlap
        if d.x2 >= lo and d.x1 <= hi:
            p = min(1.0, d.y2 / float(h))
            prox = max(prox, p)
            max_approach = max(max_approach, d.approach_rate)
            
    if prox == 0.0:
        return False, 0.0, 1.0
        
    # flag is true if prox >= prox_slow or approach rate is high enough
    flag = prox >= cfg["prox_slow"] or max_approach > 1.05
    
    factor = float(np.clip((cfg["prox_stop"] - prox) / max(1e-6, cfg["prox_stop"] - cfg["prox_slow"]), 0.0, 1.0))
    if max_approach > 1.02:
        factor *= float(np.clip(1.2 - max_approach, 0.1, 1.0))
        
    return flag, prox, factor


def compute_command(final_mask: np.ndarray, dets: Sequence[Detection], trust: float, terrain: int,
                    cfg: Dict[str, Any], raw_mask: np.ndarray = None) -> Command:
    h, w = final_mask.shape
    
    # Steering from final mask
    cx, width_frac = drivable_geometry(final_mask, cfg["lookahead_rows"])
    angle = float(cfg["max_steer_deg"] * np.clip((cx - w / 2.0) / (w / 2.0), -1.0, 1.0)) if width_frac > 0 else 0.0
    
    # Obstacle check from ego corridor (using pre-subtraction mask if provided)
    mask_for_obs = raw_mask if raw_mask is not None else final_mask
    obs_cx, _ = drivable_geometry(mask_for_obs, cfg["lookahead_rows"])
    # If no drivable space, assume center
    if _ == 0: obs_cx = w / 2.0
        
    flag, prox, obs_f = assess_obstacles(dets, (h, w), obs_cx, cfg)
    
    trust_f = float(np.clip((trust - cfg["trust_min"]) / max(1e-6, 1.0 - cfg["trust_min"]), 0.0, 1.0))
    width_f = float(np.clip(width_frac / cfg["width_ref"], 0.0, 1.0))
    terr_f = float(cfg["terrain_speed_factor"][int(np.clip(terrain, 0, len(cfg["terrain_speed_factor"]) - 1))])
    
    speed = float(cfg["v_max"] * trust_f * width_f * obs_f * terr_f)
    return Command(angle, speed, float(trust), bool(flag), width_frac, cx, prox)
