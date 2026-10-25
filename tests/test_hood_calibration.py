import numpy as np
import pytest
from inference.hood_calibration import calibrate_hood_from_frames, calibrate_or_load_hood_polygon


def test_synthetic_hood_calibration():
    # 30 synthetic frames (100x100)
    # Background has random noise changing every frame
    # Bottom 15% (y >= 85) has a static hood pattern
    np.random.seed(42)
    frames = []
    for _ in range(30):
        frame = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
        # Static hood region in bottom 15%
        frame[85:, 40:] = 30  # Constant dark gray hood
        frames.append(frame)

    poly = calibrate_hood_from_frames(frames, std_threshold=10.0)
    assert poly is not None, "Failed to calibrate hood on synthetic video"
    assert len(poly) >= 3, "Polygon must have at least 3 points"

    # All polygon y-coordinates should be at the bottom (>= 0.80)
    poly_arr = np.array(poly)
    assert np.all(poly_arr[:, 1] >= 0.80), f"Hood y coordinates should be >= 0.80, got {poly_arr[:, 1]}"
    # Polygon x-coordinates should cover the right side (>= 0.35)
    assert np.all(poly_arr[:, 0] >= 0.35), f"Hood x coordinates should be >= 0.35, got {poly_arr[:, 0]}"


def test_no_hood_when_scene_is_all_dynamic():
    np.random.seed(42)
    frames = [np.random.randint(0, 255, (80, 80), dtype=np.uint8) for _ in range(20)]
    poly = calibrate_hood_from_frames(frames, std_threshold=5.0)
    assert poly is None, "Should not detect a hood when the entire scene is dynamic"
