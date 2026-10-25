"""Unit tests for Self-Calibrating Inverse Perspective Mapping (IPM) - Phase 1."""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from inference.ipm import (
    ComplementaryFilter,
    HorizonEstimator,
    IMUReader,
    InversePerspectiveMapping,
    IPMResult,
)


@pytest.fixture
def base_cfg():
    return {
        "camera": {
            "height_m": 1.3,
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
            "hfov_deg": 75.0,
            "vfov_deg": 45.0,
            "bev_width_m": 10.0,
            "bev_height_m": 25.0,
            "bev_min_dist_m": 1.5,
            "bev_resolution": 0.05,
            "filter_alpha": 0.98,
        }
    }


def test_complementary_filter():
    filt = ComplementaryFilter(alpha=0.98, nominal_pitch_deg=0.0, nominal_roll_deg=0.0)
    # Stationary level sensor: gravity down (az = 9.81 m/s^2), no rotation
    pitch, roll = filt.update(ax=0.0, ay=0.0, az=9.81, gx=0.0, gy=0.0, gz=0.0, dt=0.04)
    assert abs(math.degrees(pitch)) < 1.0
    assert abs(math.degrees(roll)) < 1.0

    # Sensor tilted pitch up by ~10 degrees (ax negative)
    # sin(10 deg) ~ 0.1736 * 9.81 ~ 1.70
    pitch_tilted, _ = filt.update(ax=-1.70, ay=0.0, az=9.66, gx=0.0, gy=0.0, gz=0.0, dt=0.04)
    # Should move toward positive pitch
    assert pitch_tilted > 0.0


def test_imu_reader():
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
        tmp.write("timestamp,ax,ay,az,gx,gy,gz\n")
        tmp.write("0.0,0.0,0.0,9.81,0.0,0.0,0.0\n")
        tmp.write("0.04,-0.1,0.0,9.80,0.01,0.0,0.0\n")
        tmp_path = tmp.name

    try:
        reader = IMUReader(tmp_path)
        sample0 = reader.get_sample(0)
        assert sample0 is not None
        assert sample0["az"] == 9.81
        sample1 = reader.get_sample(1)
        assert sample1 is not None
        assert sample1["ax"] == -0.1
        # Beyond length clamped to last
        sample_late = reader.get_sample(10)
        assert sample_late["ax"] == -0.1
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_pitch_changes_homography(base_cfg):
    ipm = InversePerspectiveMapping(base_cfg)
    img_w, img_h = 1280, 720

    H_flat, _ = ipm.compute_homography(pitch_rad=0.0, roll_rad=0.0, img_w=img_w, img_h=img_h)
    H_pitched, _ = ipm.compute_homography(
        pitch_rad=math.radians(5.0), roll_rad=0.0, img_w=img_w, img_h=img_h
    )

    assert H_flat.shape == (3, 3)
    assert H_pitched.shape == (3, 3)
    # Changing camera pitch must alter the perspective warp
    assert not np.allclose(H_flat, H_pitched, atol=1e-3)


def test_roll_changes_homography(base_cfg):
    ipm = InversePerspectiveMapping(base_cfg)
    img_w, img_h = 1280, 720

    H_flat, _ = ipm.compute_homography(pitch_rad=0.0, roll_rad=0.0, img_w=img_w, img_h=img_h)
    H_rolled, _ = ipm.compute_homography(
        pitch_rad=0.0, roll_rad=math.radians(3.0), img_w=img_w, img_h=img_h
    )

    assert not np.allclose(H_flat, H_rolled, atol=1e-3)


def test_road_width_recovery_within_10_percent(base_cfg):
    """Synthetic test: A known ground road width (4.0m) must be recovered within 10% in BEV."""
    ipm = InversePerspectiveMapping(base_cfg)
    img_w, img_h = 1280, 720

    # In BEV coordinates, a 4.0m wide road spans -2.0m to +2.0m laterally.
    # Longitudinal distance is 6.0m ahead.
    bev_w_px = ipm.bev_w_px
    bev_h_px = ipm.bev_h_px
    res = ipm.bev_resolution

    # Synthetic ground mask in BEV space: 4.0m wide
    bev_mask_ground = np.zeros((bev_h_px, bev_w_px), dtype=np.uint8)
    cx_bev = bev_w_px // 2
    half_w_px = int(round(2.0 / res))  # 2.0 m / 0.05 m/px = 40 px

    # Fill road area across all rows
    bev_mask_ground[:, max(0, cx_bev - half_w_px):min(bev_w_px, cx_bev + half_w_px)] = 1

    H_img2bev, H_bev2img = ipm.compute_homography(0.0, 0.0, img_w, img_h)

    # Warp from BEV to camera image
    img_mask = cv2.warpPerspective(bev_mask_ground, H_bev2img, (img_w, img_h), flags=cv2.INTER_NEAREST)

    # Now run IPM forward pipeline: warp image mask back to BEV
    recovered_bev = ipm.warp_mask_to_bev(img_mask, H_img2bev)

    # Extract road metrics at 6.0m ahead
    recovered_width_m, offset_m = ipm.extract_road_metrics(recovered_bev, distance_m=6.0)

    assert recovered_width_m is not None, "Failed to extract road width from BEV mask"
    assert abs(recovered_width_m - 4.0) / 4.0 <= 0.10, (
        f"Recovered width {recovered_width_m:.2f}m is not within 10% of true 4.0m"
    )
    if offset_m is not None:
        assert abs(offset_m) < 0.35, f"Offset {offset_m:.2f}m too far from center"


def test_ipm_fallback_hierarchy(base_cfg):
    ipm = InversePerspectiveMapping(base_cfg)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    mask = np.zeros((720, 1280), dtype=np.uint8)

    # 1. IMU provided -> method should be imu_fused
    imu_sample = {"ax": 0.0, "ay": 0.0, "az": 9.81, "gx": 0.0, "gy": 0.0, "gz": 0.0}
    res_imu = ipm.process_frame(frame, mask, imu_sample=imu_sample)
    assert res_imu.method == "imu_fused"

    # 2. No IMU and blank image -> horizon estimator returns None -> falls back to nominal
    res_nominal = ipm.process_frame(frame, mask, imu_sample=None)
    assert res_nominal.method == "nominal"
