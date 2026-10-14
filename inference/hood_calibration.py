"""Per-source automatic camera hood calibration.

Computes the per-pixel temporal standard deviation over video frames.
The static vehicle hood exhibits low temporal variance and touches the bottom
edge of the frame. The connected low-variance region touching the bottom edge
is extracted and converted to a normalized polygon saved in outputs/camera_profiles/.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def calibrate_hood_from_frames(
    frames: List[np.ndarray],
    std_threshold: Optional[float] = None,
) -> Optional[List[List[float]]]:
    """Compute hood polygon from a sequence of video frames.

    Args:
        frames: List of BGR or grayscale frames (at least 10 frames recommended).
        std_threshold: Optional fixed variance threshold. If None, automatically
            estimated from the bottom region of the temporal standard deviation.

    Returns:
        List of normalized [x, y] coordinates in [0.0, 1.0], or None if no static hood.
    """
    if len(frames) < 5:
        logger.warning("Too few frames (%d) for reliable hood calibration.", len(frames))
        return None

    # Convert to grayscale if BGR
    gray_frames = []
    for f in frames:
        if len(f.shape) == 3:
            gray_frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        else:
            gray_frames.append(f)

    h, w = gray_frames[0].shape
    stack = np.stack(gray_frames, axis=0).astype(np.float32)
    std_map = np.std(stack, axis=0)

    # Automatically estimate low-variance threshold if not provided
    if std_threshold is None:
        bottom_region = std_map[int(h * 0.75):, :]
        p20 = float(np.percentile(bottom_region, 20))
        mean_b = float(np.mean(bottom_region))
        # Static hood has low std relative to road texture/motion
        std_threshold = min(22.0, max(8.0, min(p20 + 4.0, mean_b * 0.6)))

    bin_mask = (std_map < std_threshold).astype(np.uint8)

    # Morphological close to join adjacent low-variance patches
    k_size = max(5, int(min(w, h) * 0.015))
    if k_size % 2 == 0:
        k_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)

    # Find connected components touching bottom edge
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask)
    hood_mask = np.zeros((h, w), dtype=np.uint8)
    total_area = h * w
    has_component = False

    for lbl in range(1, num):
        # Must touch bottom rows
        touches_bottom = np.any(labels[max(0, h - 4):, :] == lbl)
        if touches_bottom:
            area = stats[lbl, cv2.CC_STAT_AREA]
            # Must occupy at least 0.4% of total frame area
            if area >= 0.004 * total_area:
                hood_mask[labels == lbl] = 255
                has_component = True

    if not has_component:
        logger.info("No static hood component detected touching the bottom edge.")
        return None

    # Morphological dilation/closing to consolidate multiple hood lobes
    hood_mask = cv2.morphologyEx(hood_mask, cv2.MORPH_CLOSE, kernel)

    # Find external contours
    cnts, _ = cv2.findContours(hood_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    # Merge all qualifying contours or take convex hull of all hood pixels
    nz_y, nz_x = np.nonzero(hood_mask)
    if len(nz_y) < 50:
        return None

    # Check that hood is actually at the bottom (top of hood should not exceed top half of frame)
    min_y = np.min(nz_y)
    if min_y < 0.35 * h:
        # If static region goes higher than y=0.35, it's not a car hood
        hood_mask[:int(0.35 * h), :] = 0
        nz_y, nz_x = np.nonzero(hood_mask)
        if len(nz_y) < 50:
            return None

    pts_2d = np.column_stack([nz_x, nz_y])
    hull = cv2.convexHull(pts_2d)

    # Approximate polygon to keep representation compact
    epsilon = 0.012 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    norm_pts: List[List[float]] = []
    for pt in approx:
        px, py = float(pt[0][0]), float(pt[0][1])
        nx = round(float(np.clip(px / w, 0.0, 1.0)), 4)
        ny = round(float(np.clip(py / h, 0.0, 1.0)), 4)
        norm_pts.append([nx, ny])

    return norm_pts


def calibrate_or_load_hood_polygon(
    source_path: Union[str, Path, int],
    profile_dir: Union[str, Path] = "outputs/camera_profiles",
    force_recalibrate: bool = False,
    max_frames: int = 150,
) -> Optional[List[List[float]]]:
    """Retrieve existing hood polygon from profile or calibrate from video frames."""
    if isinstance(source_path, int):
        profile_stem = f"camera_{source_path}"
    else:
        profile_stem = Path(str(source_path)).stem

    p_dir = Path(profile_dir)
    p_dir.mkdir(parents=True, exist_ok=True)
    profile_file = p_dir / f"{profile_stem}.json"

    if profile_file.is_file() and not force_recalibrate:
        try:
            data = json.loads(profile_file.read_text(encoding="utf-8"))
            poly = data.get("hood_polygon")
            if poly is not None:
                logger.info("Loaded cached camera hood profile from %s", profile_file)
                return poly
        except Exception as exc:
            logger.warning("Error reading %s (%s); recalibrating.", profile_file, exc)

    # Calibrate from video source
    if isinstance(source_path, int) or Path(str(source_path)).exists():
        cap = cv2.VideoCapture(source_path)
        frames: List[np.ndarray] = []
        try:
            for _ in range(max_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
        finally:
            cap.release()

        if len(frames) >= 25:
            norm_pts = calibrate_hood_from_frames(frames)
            profile_data = {
                "source": str(source_path),
                "resolution": [int(frames[0].shape[1]), int(frames[0].shape[0])],
                "frames_sampled": len(frames),
                "hood_polygon": norm_pts,
            }
            profile_file.write_text(json.dumps(profile_data, indent=2), encoding="utf-8")
            logger.info("Saved new camera hood profile to %s", profile_file)
            return norm_pts

    return None
