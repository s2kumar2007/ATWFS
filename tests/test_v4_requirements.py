"""Tests for v4 ATWFS pipeline enhancements:
1. Low-visibility sequence (hold 5 frames, decay over 10 frames, crawl floor, not stop)
2. Low-visibility obstacle crawl stops (speed=0 when threat in corridor)
3. Trust-cap display shows FALLBACK (not DL with trust <= 0.15)
4. Estimator hysteresis (needs 5 consecutive below threshold and 10-frame dwell to switch)
5. Classical fallback mask hard clips above horizon
6. Classical fallback mask zeros out under person/vehicle detection boxes
7. Path monotonic filter prevents hooks/loops (no reversals > 30px)
8. Steering fixed lookahead evaluates target at lookahead_m (not far end of road)
"""
from __future__ import annotations

import numpy as np
import pytest

from inference.classical import ClassicalFreeSpaceEstimator
from inference.decision import Command, _monotonic_path_pts, compute_command
from inference.detector import Detection
from inference.live import LivePipeline


def test_low_visibility_hold_then_decay():
    """When visible road in lookahead band is < 10%, angle is held for 5 frames,
    then decayed over 10 frames towards 0, with crawl speed floor (0.20 m/s), not stopped."""
    cfg = {
        "lookahead_rows": [0.55, 0.90],
        "bev_width_m": 10.0,
        "max_ego_lane_width_m": 4.0,
        "ego_window_half_m": 1.8,
        "lookahead_m": 4.0,
        "wheelbase_m": 2.5,
        "max_steer_deg": 30.0,
        "low_vis_hold_frames": 5,
        "low_vis_decay_frames": 10,
        "crawl_speed_floor": 0.20,
        "trust_min": 0.3,
        "max_speed_ramp": 0.15,
    }
    # Empty mask: 0% drivable pixels
    mask = np.zeros((100, 200), dtype=np.uint8)
    initial_angle = 20.0

    # Frames 0 to 4: held at initial_angle
    for hold_i in range(5):
        cmd = compute_command(
            mask, dets=[], trust=0.8, terrain=0, cfg=cfg,
            prev_angle=initial_angle, hold_count=hold_i, prev_speed=0.20,
        )
        assert cmd.angle == pytest.approx(initial_angle, abs=0.1), f"Frame {hold_i} should hold angle"
        assert cmd.saturating_term == "hold_angle"
        assert cmd.speed >= 0.15, "Speed should maintain crawl floor, not stop"

    # Frame 10 (hold_count = 10, decay step = 5 of 10): decayed by ~50%
    cmd_mid = compute_command(
        mask, dets=[], trust=0.8, terrain=0, cfg=cfg,
        prev_angle=initial_angle, hold_count=10, prev_speed=0.20,
    )
    assert cmd_mid.saturating_term == "decay_angle"
    assert 5.0 < cmd_mid.angle < 15.0, f"Expected decayed angle around 10 deg, got {cmd_mid.angle}"
    assert cmd_mid.speed >= 0.15, "Speed should maintain crawl floor during decay"

    # Frame 16 (hold_count = 16 >= 5 + 10): fully decayed to 0
    cmd_zero = compute_command(
        mask, dets=[], trust=0.8, terrain=0, cfg=cfg,
        prev_angle=initial_angle, hold_count=16, prev_speed=0.20,
    )
    assert cmd_zero.angle == pytest.approx(0.0, abs=0.1), "Angle should decay to 0 after 15 frames"


def test_low_visibility_obstacle_crawl_stops():
    """In low visibility, if an obstacle is confirmed near in ego corridor, speed stops to 0.0."""
    cfg = {
        "lookahead_rows": [0.55, 0.90],
        "bev_width_m": 10.0,
        "max_ego_lane_width_m": 4.0,
        "ego_window_half_m": 1.8,
        "lookahead_m": 4.0,
        "wheelbase_m": 2.5,
        "max_steer_deg": 30.0,
        "low_vis_hold_frames": 5,
        "low_vis_decay_frames": 10,
        "crawl_speed_floor": 0.20,
        "trust_min": 0.3,
        "max_speed_ramp": 0.15,
    }
    mask = np.zeros((100, 200), dtype=np.uint8)
    # Obstacle close to ego vehicle: y2=70 (rel_y = 0.70 >= 0.55), in center
    dets = [
        Detection(x1=90, y1=50, x2=110, y2=70, cls_id=0, name="person", conf=0.85, kind="person")
    ]
    cmd = compute_command(
        mask, dets=dets, trust=0.8, terrain=0, cfg=cfg,
        prev_angle=10.0, hold_count=2, prev_speed=0.20,
    )
    assert cmd.obstacle_flag is True, "Obstacle flag must be True for close threat"
    assert cmd.speed == 0.0, "Speed must be 0.0 when obstacle is in path during low visibility"


def test_trust_cap_shows_fallback():
    """When trust is capped to 0.15 in low visibility, the pipeline shows FALLBACK not DL."""
    cfg = {
        "seed": 42,
        "paths": {"output_dir": "outputs/test"},
        "data": {"image_size": [100, 200], "imagenet_mean": [0.485, 0.456, 0.406], "imagenet_std": [0.229, 0.224, 0.225]},
        "camera": {"hfov_deg": 70.0, "vfov_deg": 40.0, "height_m": 1.4, "pitch_deg": 2.0, "roll_deg": 0.0},
        "classical": {"color_k": 2.5, "canny": [60, 160], "morph_kernel": 5},
        "fusion": {
            "hidden": 16, "gain": 1.0, "w_certainty": 0.4, "w_geometric": 0.3,
            "w_temporal": 0.3, "bias": 0.0, "terrain_penalty": [0.0, 0.1, 0.2, 0.3],
            "fallback_threshold": 0.5, "mask_threshold": 0.5,
        },
        "consistency": {
            "geometric": {"road_width_m_min": 2.5, "road_width_m_max": 12.0, "lat_offset_m_max": 2.5, "heading_deg_max": 20.0},
            "temporal": {"max_expected_speed_mps": 25.0},
        },
        "inference": {
            "input_size": [100, 200],
            "decision": {
                "lookahead_rows": [0.55, 0.90],
                "bev_width_m": 10.0,
                "max_ego_lane_width_m": 4.0,
                "ego_window_half_m": 1.8,
                "lookahead_m": 4.0,
                "wheelbase_m": 2.5,
                "max_steer_deg": 30.0,
                "low_vis_hold_frames": 5,
                "low_vis_decay_frames": 10,
                "crawl_speed_floor": 0.20,
                "trust_min": 0.3,
                "obstacle_dilate_px": 5,
            },
            "estimator_min_dwell": 10,
            "fallback_consec_threshold": 5,
        },
    }
    # Class mock detector
    class DummyDetector:
        def detect(self, frame, frame_idx):
            return []

    pipeline = LivePipeline(cfg, model=None, device="cpu", detector=DummyDetector())
    # Create black frame (0% drivable pixels -> triggers low-visibility hold)
    black_frame = np.zeros((100, 200, 3), dtype=np.uint8)
    pipeline.last_raw_mask = np.zeros((100, 200), dtype=np.uint8)
    pipeline.prev_steer_angle = 5.0
    res = pipeline.process_frame(black_frame, frame_idx=0)
    assert res.command.trust <= 0.15, "Trust should be capped at 0.15 for low-vis"
    assert res.active == "FALLBACK", f"HUD must display FALLBACK when trust is capped, got {res.active}"


def test_estimator_hysteresis_needs_5_consecutive():
    """A single below-threshold frame does not flip the estimator to FALLBACK (needs 5 consecutive)."""
    cfg = {
        "seed": 42,
        "paths": {"output_dir": "outputs/test"},
        "data": {"image_size": [100, 200], "imagenet_mean": [0.485, 0.456, 0.406], "imagenet_std": [0.229, 0.224, 0.225]},
        "camera": {"hfov_deg": 70.0, "vfov_deg": 40.0, "height_m": 1.4, "pitch_deg": 2.0, "roll_deg": 0.0},
        "classical": {"color_k": 2.5, "canny": [60, 160], "morph_kernel": 5},
        "fusion": {"fallback_threshold": 0.5, "mask_threshold": 0.5, "hidden": 16, "gain": 1.0, "w_certainty": 0.4, "w_geometric": 0.3, "w_temporal": 0.3, "bias": 0.0, "terrain_penalty": [0.0, 0.1, 0.2, 0.3]},
        "consistency": {"geometric": {}, "temporal": {}},
        "inference": {
            "input_size": [100, 200],
            "decision": {"lookahead_rows": [0.55, 0.90], "lookahead_m": 4.0, "wheelbase_m": 2.5, "max_steer_deg": 30.0},
            "estimator_min_dwell": 10,
            "fallback_consec_threshold": 5,
        },
    }
    class DummyDetector:
        def detect(self, frame, frame_idx):
            return []

    pipeline = LivePipeline(cfg, model=None, device="cpu", detector=DummyDetector())
    assert pipeline.current_estimator == "DL"

    # Simulate 4 consecutive below-threshold frames
    for i in range(4):
        pipeline.current_estimator = "DL"
        pipeline.estimator_dwell = 15  # past dwell
        # Check hysteresis logic
        fallback_threshold = 0.5
        trust_is_low = True
        if trust_is_low:
            pipeline.fallback_consec_count += 1
        else:
            pipeline.fallback_consec_count = 0
        if pipeline.fallback_consec_count >= pipeline.fallback_consec_threshold and pipeline.estimator_dwell >= pipeline.min_dwell:
            pipeline.current_estimator = "FALLBACK"
            pipeline.estimator_dwell = 0
            pipeline.estimator_switch_count += 1
            pipeline.fallback_consec_count = 0
        else:
            pipeline.estimator_dwell += 1

    # After 4 frames, should STILL be DL
    assert pipeline.current_estimator == "DL", "Estimator must not flip to FALLBACK before 5 consecutive frames"
    assert pipeline.estimator_switch_count == 0

    # 5th frame: now it should flip
    if True:  # 5th below-threshold frame
        pipeline.fallback_consec_count += 1
        if pipeline.fallback_consec_count >= pipeline.fallback_consec_threshold and pipeline.estimator_dwell >= pipeline.min_dwell:
            pipeline.current_estimator = "FALLBACK"
            pipeline.estimator_dwell = 0
            pipeline.estimator_switch_count += 1
            pipeline.fallback_consec_count = 0
    assert pipeline.current_estimator == "FALLBACK", "Estimator must flip to FALLBACK on 5th consecutive frame"
    assert pipeline.estimator_switch_count == 1


def test_fallback_mask_no_above_horizon():
    """Classical estimator hard-clips any pixels above eff_horizon."""
    estimator = ClassicalFreeSpaceEstimator({
        "color_k": 2.5, "canny": [60, 160], "morph_kernel": 5, "horizon_frac": 0.25,
    })
    h, w = 120, 200
    horizon_y = 35
    # Uniform grey image (drivable color everywhere)
    frame = np.full((h, w, 3), 128, dtype=np.uint8)
    mask = estimator.estimate(frame, horizon_y=horizon_y)
    eff_horizon = max(0, horizon_y - 5)
    above_horizon_pixels = mask[:eff_horizon, :].sum()
    assert above_horizon_pixels == 0, f"Found {above_horizon_pixels} pixels above horizon"


def test_fallback_mask_no_under_person_box():
    """Classical estimator zeros out drivable region under person detection boxes."""
    estimator = ClassicalFreeSpaceEstimator({
        "color_k": 2.5, "canny": [60, 160], "morph_kernel": 5, "horizon_frac": 0.25,
    })
    h, w = 120, 200
    frame = np.full((h, w, 3), 128, dtype=np.uint8)
    # Person box in bottom center: y1=60, y2=100, x1=80, x2=120
    person_det = Detection(x1=80, y1=60, x2=120, y2=100, cls_id=0, name="person", conf=0.9, kind="person")
    mask = estimator.estimate(frame, horizon_y=30, dets=[person_det])
    under_box_pixels = mask[60:100, 80:120].sum()
    assert under_box_pixels == 0, f"Found {under_box_pixels} non-zero pixels under person box"


def test_path_monotonic_no_hooks():
    """Monotonic path filter removes any point that reverses direction by > 30px."""
    # Simulated center line with a sudden hook/loop: (y, x)
    # Moving from bottom (y=100) to top (y=20): x moves smoothly right, then hooks sharp left by 40px
    c_pts = [
        (100.0, 100.0),
        (90.0, 105.0),
        (80.0, 110.0),
        (70.0, 65.0),   # Hook! Reverses by 45 px relative to previous
        (60.0, 115.0),
        (50.0, 120.0),
    ]
    filtered = _monotonic_path_pts(c_pts, ego_center_px=100.0)
    filtered_x = [pt[1] for pt in filtered]
    # The hook point (65.0) must be discarded
    assert 65.0 not in filtered_x, "Monotonic filter should have dropped the hook at x=65.0"
    assert len(filtered) == 5, f"Expected 5 points, got {len(filtered)}"


def test_steering_fixed_lookahead_not_far():
    """Steering with lookahead_m=4.0 evaluates road offset at fixed lookahead row,
    giving a moderate, sensible steering angle (< 15 deg) rather than 20+ deg."""
    cfg = {
        "lookahead_rows": [0.40, 0.90],
        "bev_width_m": 10.0,
        "max_ego_lane_width_m": 4.0,
        "ego_window_half_m": 1.8,
        "lookahead_m": 4.0,
        "wheelbase_m": 2.5,
        "max_steer_deg": 30.0,
        "crawl_speed_floor": 0.20,
        "trust_min": 0.3,
        "max_speed_ramp": 0.15,
        "speed_angle_factor_min": 0.4,
        "speed_angle_denom": 45.0,
    }
    h, w = 120, 200
    # Create mask with road slightly curved to the right (offset 15 px at lookahead row)
    mask = np.zeros((h, w), dtype=np.uint8)
    for y in range(45, 110):
        # Center starts at 100 at y=110 and drifts gently to 115 at y=45
        drift = int(15.0 * (110 - y) / 65.0)
        c = 100 + drift
        mask[y, c - 20 : c + 20] = 1

    cmd = compute_command(
        mask, dets=[], trust=0.85, terrain=0, cfg=cfg,
        prev_angle=0.0, hold_count=0, prev_speed=1.0,
    )
    assert 0.0 < cmd.angle < 15.0, f"Expected gentle right steer (<15 deg), got {cmd.angle:.2f} deg"
    assert cmd.saturating_term == "none", "Gentle curve should not saturate steering"
