"""Classical (non-neural) free-space estimator: colour-space seeding + edge barriers (OpenCV).

Idea: assume the patch directly in front of the vehicle (a strip just above the hood
bottom-centre) is drivable. Learn its colour statistics in CIE-Lab, mark pixels with
similar colour as candidate free space, cut candidates by strong Canny edges
(obstacle/road boundaries), then keep the connected region(s) that touch the seed.
This is the FALLBACK used when the learned model is untrusted (see :mod:`inference.fusion`).

v4 improvements:
- Seed from a narrow horizontal strip just ABOVE the hood/bottom-centre (not a large trapezoid).
- Flood-fill connected road-coloured region only.
- Hard clip any output pixels above horizon_y.
- Remove regions overlapping detected person/vehicle boxes.
- Temporal smoothing: blend with the previous mask at 0.4 weight.
- Skip entirely when alpha >= classical_skip_alpha; cache for classical_cache_frames.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class ClassicalFreeSpaceEstimator:
    """Colour + edge heuristic free-space estimator (v4)."""

    _SIGMA_MIN = np.array([8.0, 4.0, 4.0], np.float32)  # floors for (L, a, b) std

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self._prev_mask: Optional[np.ndarray] = None
        self._cached_mask: Optional[np.ndarray] = None
        self._cache_age: int = 0

    def seed_mask(self, h: int, w: int) -> np.ndarray:
        """Narrow horizontal seed strip just ABOVE the hood bottom-centre.

        Samples from the centre strip of the bottom ~8% of the image (above hood),
        which is reliably road-surface on Indian dashcam footage.
        """
        # Use a narrow strip at [0.86*h, 0.94*h] × [0.25*w, 0.75*w]
        top = int(0.86 * h)
        bottom = int(0.94 * h)
        left = int(0.25 * w)
        right = int(0.75 * w)
        m = np.zeros((h, w), np.uint8)
        m[top:bottom, left:right] = 1
        return m

    def estimate(
        self,
        bgr: np.ndarray,
        horizon_y: Optional[int] = None,
        dets: Optional[List[Any]] = None,
    ) -> np.ndarray:
        """Estimate the free-space mask.

        Args:
            bgr: ``(H, W, 3)`` uint8 BGR frame.
            horizon_y: Row index of horizon. Pixels above this are zeroed.
            dets: Optional list of Detection objects; mask is zeroed under person/vehicle boxes.

        Returns:
            ``(H, W)`` uint8 binary mask (1 = free/drivable).
        """
        c = self.cfg
        h, w = bgr.shape[:2]
        blur = cv2.GaussianBlur(bgr, (5, 5), 0)
        lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)
        seed = self.seed_mask(h, w).astype(bool)
        pix = lab[seed]
        if pix.shape[0] == 0:
            return np.zeros((h, w), np.uint8)
        mu, sd = pix.mean(0), np.maximum(pix.std(0), self._SIGMA_MIN)
        dist = np.sqrt((((lab - mu) / sd) ** 2).mean(axis=2))
        free = (dist < c.get("color_k", 2.5))
        edges = cv2.Canny(cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY), *c.get("canny", [60, 160]))
        edge_dilate = int(c.get("edge_dilate", 2))
        if edge_dilate > 0:
            edges = cv2.dilate(edges, np.ones((2 * edge_dilate + 1,) * 2, np.uint8))
        free &= (edges == 0)

        # Hard clip above horizon (with a 5px margin buffer below)
        eff_horizon = horizon_y if horizon_y is not None else int(c.get("horizon_frac", 0.25) * h)
        eff_horizon = max(0, eff_horizon - 5)  # 5 px buffer so road near horizon isn't clipped
        free[:eff_horizon] = False

        # Road perspective ROI trapezoid
        roi_pts = np.array([
            [int(0.15 * w), eff_horizon],
            [int(0.85 * w), eff_horizon],
            [w, h],
            [0, h],
        ], dtype=np.int32)
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(roi_mask, [roi_pts], 1)
        free &= (roi_mask > 0)

        k = np.ones((c.get("morph_kernel", 7),) * 2, np.uint8)
        free_u8 = cv2.morphologyEx(free.astype(np.uint8), cv2.MORPH_CLOSE, k)
        free_u8 = cv2.morphologyEx(free_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lbl = cv2.connectedComponents(free_u8, connectivity=8)
        if n <= 1:
            return np.zeros((h, w), np.uint8)

        # Keep only components that overlap the seed
        overlap = np.bincount(lbl[seed].ravel(), minlength=n)
        overlap[0] = 0
        min_seed_overlap = max(1, int(0.05 * seed.sum()))
        keep = np.flatnonzero(overlap >= min_seed_overlap)
        if keep.size == 0:
            return np.zeros((h, w), np.uint8)
        out = np.isin(lbl, keep).astype(np.uint8)

        # Fill interior holes
        inv = (1 - out).astype(np.uint8)
        ff = inv.copy()
        cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 2)
        holes = (ff == 1)
        out[holes] = 1

        # Hard clip above horizon again (after hole-filling)
        out[:eff_horizon] = 0

        # Zero out regions under person/vehicle detection boxes
        if dets:
            for d in dets:
                if getattr(d, "kind", "") in ("person", "vehicle", "animal") or getattr(d, "cls_id", -1) == 0:
                    bx1 = max(0, int(d.x1))
                    bx2 = min(w, int(d.x2))
                    by1 = max(eff_horizon, int(d.y1))
                    by2 = min(h, int(d.y2))
                    if bx2 > bx1 and by2 > by1:
                        out[by1:by2, bx1:bx2] = 0

        # Temporal smoothing: blend with previous mask
        if self._prev_mask is not None and self._prev_mask.shape == out.shape:
            blend_w = float(c.get("temporal_blend_w", 0.35))
            blended = (1.0 - blend_w) * out.astype(np.float32) + blend_w * self._prev_mask.astype(np.float32)
            out = (blended > 0.5).astype(np.uint8)
        self._prev_mask = out.copy()

        return out

    def estimate_with_cache(
        self,
        bgr: np.ndarray,
        alpha: float,
        skip_alpha: float = 0.8,
        cache_frames: int = 2,
        horizon_y: Optional[int] = None,
        dets: Optional[List[Any]] = None,
    ) -> Tuple[np.ndarray, bool]:
        """Run estimate with caching.

        Returns (mask, ran_full_estimate).
        - If alpha >= skip_alpha: return zeros (DL is trusted enough).
        - If cache_age < cache_frames and we have a cached mask: return cached mask.
        - Otherwise: run full estimate, cache the result.
        """
        h, w = bgr.shape[:2]

        if alpha >= skip_alpha:
            # DL trusted — return zeros (classical not needed)
            self._cache_age = 0
            self._cached_mask = np.zeros((h, w), np.uint8)
            return self._cached_mask, False

        # Use cached result if fresh enough
        if self._cached_mask is not None and self._cache_age < cache_frames and self._cached_mask.shape == (h, w):
            self._cache_age += 1
            return self._cached_mask, False

        # Run full estimate
        mask = self.estimate(bgr, horizon_y=horizon_y, dets=dets)
        self._cached_mask = mask
        self._cache_age = 1
        return mask, True
