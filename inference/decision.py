import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
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
    
    @property
    def packet(self) -> str:
        return f"[{self.angle:+.2f}, {self.speed:.3f}, {self.trust:.3f}, {int(self.obstacle_flag)}]"

def subtract_obstacles(mask: np.ndarray, dets: List[Detection], dilate_px: int) -> np.ndarray:
    if not dets:
        return mask
    h, w = mask.shape
    out = mask.copy()
    for d in dets:
        # Ignore small or far detections
        if d.height < 0.04 * h:
            continue
        # Subtract obstacle footprint
        x1 = max(0, d.x1 - dilate_px)
        x2 = min(w, d.x2 + dilate_px)
        y1 = max(0, d.y1)
        y2 = min(h, d.y2 + dilate_px)
        out[y1:y2, x1:x2] = 0
    return out

def compute_command(mask: np.ndarray, dets: List[Detection], trust: float, terrain: int,
                    cfg: Dict[str, Any], raw_mask: Optional[np.ndarray] = None,
                    road_width_m: Optional[float] = None, lateral_offset_m: Optional[float] = None) -> Command:
    h, w = mask.shape
    
    # 1. Hood apex calculation
    # In live.py the mask is already 0 in the hood area.
    # To find the hood apex (topmost y of the hood line at center), we check the bottom rows.
    # Or simply: just find the bottom-most non-zero row. But wait, the path must be ABOVE the hood line.
    # Since the mask has 0s in the hood polygon, `y_end` should be constrained.
    # We find the bottom-most non-zero row in `mask`.
    nz_y, nz_x = np.nonzero(mask)
    if len(nz_y) == 0:
        return Command(0.0, 0.0, trust, False, 0.0, w / 2, 0.0, width_m=road_width_m, offset_m=lateral_offset_m)
        
    bottom_y = np.max(nz_y)
    
    band = cfg["lookahead_rows"]
    y_start = int(band[0] * h)
    y_end = int(band[1] * h)
    
    # Constrain y_end to be above the hood line
    if y_end > bottom_y:
        y_end = bottom_y

    # Calculate centroid in band
    band_mask = mask[y_start:y_end]
    nz_band_y, nz_band_x = np.nonzero(band_mask)
    if len(nz_band_x) > 0:
        cx = np.mean(nz_band_x)
    else:
        # Fallback to the whole mask
        if len(nz_x) > 0:
            cx = np.mean(nz_x)
        else:
            cx = w / 2

    # Width at row just above hood
    row_idx = max(0, bottom_y - 5)
    row_nz = np.nonzero(mask[row_idx])[0]
    if len(row_nz) > 0:
        width_px = row_nz[-1] - row_nz[0]
        # Express road width as % of visible road (w)
        width_frac = float(np.clip(width_px / w, 0.0, 1.0))
    else:
        width_frac = 0.0

    # Steer angle
    if lateral_offset_m is not None:
        lookahead_m = float(cfg.get("bev_lookahead_m", 6.0))
        angle_rad = math.atan2(lateral_offset_m, lookahead_m)
        angle_deg = math.degrees(angle_rad)
        angle_deg = max(-cfg["max_steer_deg"], min(cfg["max_steer_deg"], angle_deg))
    else:
        dx = cx - (w / 2)
        dy = h - y_start
        angle_rad = math.atan2(dx, dy)
        angle_deg = math.degrees(angle_rad)
        angle_deg = max(-cfg["max_steer_deg"], min(cfg["max_steer_deg"], angle_deg))

    # Obstacle Check
    corridor = cfg["corridor_half_width"] * w
    obs_flag = False
    prox = 0.0
    
    for d in dets:
        # Check if in corridor around ego center (w/2)
        if abs(d.center_x - w/2) < corridor:
            rel_y = d.y2 / h
            if rel_y > prox:
                prox = rel_y
            if rel_y > cfg["prox_slow"] or d.approach_rate > 1.05:
                obs_flag = True

    # Speed calculation
    speed = cfg["v_max"]
    if trust < cfg["trust_min"]:
        speed = 0.0
    
    speed *= cfg["terrain_speed_factor"][terrain]
    speed *= max(0.5, trust)
    
    # Width handicap
    if road_width_m is not None:
        ref_w = float(cfg.get("road_width_ref_m", 3.5))
        if road_width_m < ref_w:
            speed *= max(0.3, road_width_m / ref_w)
    elif width_frac < cfg["width_ref"]:
        speed *= max(0.3, width_frac / cfg["width_ref"])

    # Obstacle speed drop
    if obs_flag:
        prox_stop = cfg["prox_stop"]
        prox_slow = cfg["prox_slow"]
        factor = np.clip((prox_stop - prox) / max(0.01, prox_stop - prox_slow), 0.0, 1.0)
        speed *= factor
        
        # Further slash speed if approaching rapidly
        max_app = max([d.approach_rate for d in dets if abs(d.center_x - w/2) < corridor] + [1.0])
        if max_app > 1.05:
            # Example: rate 1.1 -> drop by 50%
            speed *= max(0.1, 1.0 - (max_app - 1.0)*5.0)

    if speed < 0.1:
        speed = 0.0

    return Command(angle_deg, speed, trust, obs_flag, width_frac, cx, prox, width_m=road_width_m, offset_m=lateral_offset_m)
