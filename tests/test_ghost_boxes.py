import numpy as np
import pytest
from inference.detector import Detection, YOLODetector


def test_ghost_boxes_dropped_and_purged():
    cfg = {
        "weights": "yolov8n.pt",
        "every_n": 1,
        "conf": 0.25,
        "iou": 0.5,
        "class_ids": [0, 1, 2, 3, 5, 7],
    }
    detector = YOLODetector(cfg, output_dir="outputs", device=None)
    # Mock active pass worker to prevent async background inference during unit test
    detector.stop()

    # Feed a real detection at frame 0
    det0 = [Detection(100, 100, 200, 200, 2, "vehicle", 0.9, "vehicle", 1.0, None)]
    detector._update_tracker_with_dets(det0, frame_idx=0)

    # Frame 0: detection should be present
    res0 = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8), frame_idx=0)
    assert len(res0) == 1, "Expected 1 detection at frame 0"
    track_id = res0[0].tracker_id
    assert track_id is not None

    # Frame 1: propagated box (age = 1 <= 3) should still be returned
    detector._step_tracker_forward(frame_idx=1)
    res1 = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8), frame_idx=1)
    assert len(res1) == 1, "Expected propagated box at frame 1"

    # Frame 2: propagated box (age = 2 <= 3) should still be returned
    detector._step_tracker_forward(frame_idx=2)
    res2 = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8), frame_idx=2)
    assert len(res2) == 1, "Expected propagated box at frame 2"

    # Frame 3: propagated box (age = 3 <= 3) should still be returned
    detector._step_tracker_forward(frame_idx=3)
    res3 = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8), frame_idx=3)
    assert len(res3) == 1, "Expected propagated box at frame 3"

    # Frame 4: propagated box older than 3 frames (age = 4 > 3) must be DROPPED
    detector._step_tracker_forward(frame_idx=4)
    res4 = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8), frame_idx=4)
    assert len(res4) == 0, f"Expected 0 boxes at frame 4 (dropped stale), got {len(res4)}"

    # Feed 20 frames of no detections
    for f in range(5, 25):
        detector._update_tracker_with_dets([], frame_idx=f)
        detector._step_tracker_forward(frame_idx=f)
        res = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8), frame_idx=f)
        assert len(res) == 0, f"Ghost box remained at frame {f}"

    # Track must be completely purged (age > 8)
    assert track_id not in detector.last_real_detection_frame
    assert track_id not in detector.track_histories
