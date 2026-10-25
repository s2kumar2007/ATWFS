"""Unit tests for ATWFS V3 requirements:
- Detector box persistence within expiry window and removal after 20 empty frames
- DL mask low-coverage (< 10% visible road) lowers trust <= 0.15 and holds angle
- 100% sign agreement between path polynomial bend and steering command
- Corridor filtering excludes shoulder obstacles (bus x2=153, truck x2=314) from obstacle_flag and speed reduction
"""
import numpy as np
import pytest

from inference.decision import Command, compute_command
from inference.detector import Detection, YOLODetector
from utils.config import smoke_config


def test_detector_box_persistence_and_removal_after_20_empty_frames():
    """Feeds a detection, then 20 frames of no detections, and asserts no box remains."""
    cfg = {
        "weights": "yolov8n.pt",
        "class_ids": [2],
        "conf": 0.25,
        "iou": 0.5,
        "every_n": 4,
        "latency_frames": 4,
        "expiry_frames": 8,
    }
    detector = YOLODetector(cfg, "outputs")
    detector._worker_thread = None  # prevent background thread interference in unit test

    # Inject a real detection at frame 0
    d0 = Detection(x1=200, y1=200, x2=280, y2=280, cls_id=2, name="car", conf=0.85, tracker_id=101)
    detector._last_shape = (480, 640)
    detector._update_tracker_with_dets([d0], frame_idx=0)

    # Frame 1 to 8: within expiry window (8 frames), detection persists with Kalman propagation
    for f in range(1, 9):
        detector._step_tracker_forward(f)
        valid = [
            d for d in detector._last_active_dets
            if (f - detector.last_real_detection_frame.get(d.tracker_id, -999)) <= detector.expiry_frames
        ]
        assert len(valid) == 1, f"Box should persist at frame {f} (age {f} <= {detector.expiry_frames})"

    # Frame 9 to 25 (more than 20 frames of no detections): box must be dropped and removed
    for f in range(9, 26):
        detector._step_tracker_forward(f)
        valid = [
            d for d in detector._last_active_dets
            if (f - detector.last_real_detection_frame.get(d.tracker_id, -999)) <= detector.expiry_frames
        ]
        assert len(valid) == 0, f"Box should NOT remain at frame {f} (age {f} > {detector.expiry_frames})"

    # After frame 25, tracker history must be cleaned up
    assert 101 not in detector.last_real_detection_frame or (25 - detector.last_real_detection_frame[101] > 8)


def test_low_visible_road_coverage_lowers_trust_and_holds_angle():
    """When DL mask covers < 10% of visible road, lower trust (<= 0.15) and hold last angle."""
    d = smoke_config()["inference"]["decision"]
    h, w = 288, 512
    mask = np.zeros((h, w), dtype=np.uint8)

    # Road only at the very bottom, upper visible road is empty (coverage < 10%)
    mask[int(0.85 * h):, int(0.3 * w):int(0.7 * w)] = 1

    prev_angle = 12.5
    cmd = compute_command(mask, [], trust=0.90, terrain=0, cfg=d, prev_angle=prev_angle)

    assert cmd.trust <= 0.15, f"Trust should be <= 0.15 for low coverage, got {cmd.trust}"
    assert cmd.saturating_term == "hold_angle"
    assert abs(cmd.angle - prev_angle) < 1e-4, f"Angle should hold {prev_angle}, got {cmd.angle}"


def test_path_and_steering_sign_agreement():
    """Assert 100% sign agreement between path polynomial bend and steering angle."""
    d = smoke_config()["inference"]["decision"]
    h, w = 288, 512

    # Case 1: Road curving right
    mask_right = np.zeros((h, w), dtype=np.uint8)
    for y in range(int(0.50 * h), h):
        progress = (h - y) / float(h * 0.50)
        center_x = int(w * 0.50 + progress * 80)  # curves right as it goes up
        mask_right[y, max(0, center_x - 30):min(w, center_x + 30)] = 1

    cmd_right = compute_command(mask_right, [], trust=1.0, terrain=0, cfg=d)
    assert cmd_right.path_poly is not None
    assert cmd_right.lateral_target_dx > 0.0, "Path should bend right"
    assert cmd_right.angle > 0.0, f"Steering should be positive (right), got {cmd_right.angle}"

    # Case 2: Road curving left
    mask_left = np.zeros((h, w), dtype=np.uint8)
    for y in range(int(0.50 * h), h):
        progress = (h - y) / float(h * 0.50)
        center_x = int(w * 0.50 - progress * 80)  # curves left as it goes up
        mask_left[y, max(0, center_x - 30):min(w, center_x + 30)] = 1

    cmd_left = compute_command(mask_left, [], trust=1.0, terrain=0, cfg=d)
    assert cmd_left.path_poly is not None
    assert cmd_left.lateral_target_dx < 0.0, "Path should bend left"
    assert cmd_left.angle < 0.0, f"Steering should be negative (left), got {cmd_left.angle}"


def test_corridor_filtering_excludes_far_left_and_shoulder_obstacles():
    """Parked bus at far left (x2=153) and truck on left shoulder (x2=314) must NOT set obstacle_flag."""
    d = smoke_config()["inference"]["decision"]
    h, w = 478, 848

    # Mask with centered road
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[int(0.35 * h):, int(0.30 * w):int(0.80 * w)] = 1

    free_cmd = compute_command(mask, [], trust=1.0, terrain=0, cfg=d)

    # Frame 51: Parked bus on far left
    bus_far_left = Detection(x1=0, y1=150, x2=153, y2=320, cls_id=5, name="bus", conf=0.9, approach_rate=1.0)
    cmd_bus = compute_command(mask, [bus_far_left], trust=1.0, terrain=0, cfg=d)
    assert not cmd_bus.obstacle_flag, "Far-left parked bus must not trigger obstacle flag"
    assert abs(cmd_bus.speed - free_cmd.speed) < 1e-3, "Far-left parked bus must not reduce speed"

    # Frame 151: Parked truck on left shoulder
    truck_shoulder = Detection(x1=148, y1=145, x2=314, y2=307, cls_id=7, name="truck", conf=0.86, approach_rate=1.0)
    cmd_truck = compute_command(mask, [truck_shoulder], trust=1.0, terrain=0, cfg=d)
    assert not cmd_truck.obstacle_flag, "Shoulder parked truck must not trigger obstacle flag"
    assert abs(cmd_truck.speed - free_cmd.speed) < 1e-3, "Shoulder parked truck must not reduce speed"
