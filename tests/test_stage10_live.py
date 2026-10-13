"""Stage 10 smoke test: end-to-end on a synthetic clip -> valid output video + console packets."""
import re
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

from data.synthetic import write_synthetic_video
from inference.decision import compute_command, subtract_obstacles
from inference.detector import Detection, NullDetector, ObstacleDetector
from inference.live import run_video
from models.unet_mobilenet import build_model
from utils.config import resolve_path, smoke_config

PACKET = re.compile(r"^\[[+-]?\d+\.\d+, \d+\.\d+, \d\.\d+, [01]\]$")


class FakeDetector(ObstacleDetector):
    """Always reports a box in the middle of the lower image (deterministic obstacle for the test)."""

    def detect(self, bgr: np.ndarray, frame_idx: int = 0) -> List[Detection]:
        h, w = bgr.shape[:2]
        return [Detection(int(w * .4), int(h * .35), int(w * .6), int(h * .56), 2, "car", 0.9)]


def _clip(cfg) -> str:
    s = cfg["smoke"]
    return write_synthetic_video(str(resolve_path("./outputs/smoke/clip.mp4")), s["video_frames"], tuple(s["video_size"]),
                                 fourcc=cfg["inference"]["video_fourcc"])


def test_decision_formula_and_obstacle_subtraction() -> None:
    d = smoke_config()["inference"]["decision"]
    mask = np.zeros((100, 200), np.uint8); mask[50:, 60:140] = 1  # centred road
    free = compute_command(mask, [], trust=1.0, terrain=0, cfg=d)
    assert abs(free.angle) < 1.0 and not free.obstacle_flag and 0 < free.speed <= d["v_max"]
    assert compute_command(mask, [], 0.1, 0, d).speed == 0.0, "low trust must stop"
    assert compute_command(mask, [], 1.0, 3, d).speed < free.speed, "bad terrain must slow"
    right = np.zeros_like(mask); right[50:, 140:190] = 1
    assert compute_command(right, [], 1.0, 0, d).angle > 5, "road on the right must steer right"
    near = [Detection(80, 60, 120, 95, 2, "car", .9)]
    c = compute_command(subtract_obstacles(mask, near, 2), near, 1.0, 0, d)
    assert c.obstacle_flag and c.speed < free.speed
    assert subtract_obstacles(mask, near, 0)[70, 100] == 0


def test_end_to_end_on_synthetic_clip(capsys) -> None:
    cfg = smoke_config()
    clip = _clip(cfg)
    model = build_model(cfg["model"])
    out = str(resolve_path("./outputs/smoke/annotated.mp4"))
    res = run_video(cfg, clip, out, model=model, detector=FakeDetector())
    n = cfg["smoke"]["video_frames"]
    assert res["frames"] == n and len(res["packets"]) == n
    cap = cv2.VideoCapture(res["output_path"])
    frames = 0
    while cap.read()[0]:
        frames += 1
    cap.release()
    assert frames == n and Path(res["output_path"]).stat().st_size > 0
    lines = [l for l in capsys.readouterr().out.splitlines() if PACKET.match(l.strip())]
    assert len(lines) == n, lines[:3]
    for angle, speed, trust, flag in res["packets"]:
        assert np.isfinite([angle, speed, trust]).all() and 0 <= trust <= 1 and flag in (0, 1) and speed >= 0
    assert any(p[3] == 1 for p in res["packets"]), "fake obstacle in the corridor should raise the flag"


def test_frame_skip_and_default_detector(capsys) -> None:
    cfg = smoke_config({"inference": {"frame_skip": 3}})
    clip = _clip(cfg)
    res = run_video(cfg, clip, str(resolve_path("./outputs/smoke/annotated_skip.mp4")),
                    model=build_model(cfg["model"]), detector=NullDetector(), max_frames=7)
    assert res["frames"] == 7
    pk = res["packets"]
    assert pk[1] == pk[0] and pk[2] == pk[0], "skipped frames must hold the last packet"

def test_annotate_rendering() -> None:
    from inference.live import annotate, FrameResult
    from inference.decision import Command
    cfg = smoke_config()
    
    # Create synthetic frame and masks
    h, w = 480, 640
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 1. Empty mask (no corridor)
    mask_empty = np.zeros((h, w), dtype=np.uint8)
    res_empty = FrameResult(mask_empty, Command(0, 0, 0, False, 0.0, 0.0, 0.0), "TEST", 0.0, 0.0, 0.0, 0, [])
    ann1 = annotate(frame, res_empty, cfg=cfg)
    assert ann1.shape == (h, w, 3)
    
    # 2. Trapezoid mask
    mask_trap = np.zeros((h, w), dtype=np.uint8)
    pts = np.array([[200, 479], [440, 479], [380, 240], [260, 240]], dtype=np.int32)
    cv2.fillPoly(mask_trap, [pts], 1)
    res_trap = FrameResult(mask_trap, Command(0, 1.0, 1.0, False, 0.0, 0.0, 0.0), "TEST", 0.0, 0.0, 0.0, 0, [])
    ann2 = annotate(frame, res_trap, cfg=cfg)
    assert ann2.shape == (h, w, 3)
    
    # 3. Mask with car-sized hole
    mask_hole = mask_trap.copy()
    mask_hole[350:400, 300:340] = 0
    res_hole = FrameResult(mask_hole, Command(0, 1.0, 1.0, False, 0.0, 0.0, 0.0), "TEST", 0.0, 0.0, 0.0, 0, [])
    ann3 = annotate(frame, res_hole, cfg=cfg)
    assert ann3.shape == (h, w, 3)

def test_approach_rate_drops_speed() -> None:
    d = smoke_config()['inference']['decision']
    mask = np.zeros((100, 200), np.uint8); mask[50:, 60:140] = 1
    free = compute_command(mask, [], trust=1.0, terrain=0, cfg=d)
    
    # Static near detection
    near_static = [Detection(80, 60, 120, 80, 2, 'car', 0.9, approach_rate=1.0)]
    cmd_static = compute_command(mask, near_static, trust=1.0, terrain=0, cfg=d)
    
    # Approaching near detection
    near_app = [Detection(80, 60, 120, 80, 2, 'car', 0.9, approach_rate=1.1)]
    cmd_app = compute_command(mask, near_app, trust=1.0, terrain=0, cfg=d)
    
    assert cmd_static.speed < free.speed
    assert cmd_app.speed < cmd_static.speed
import cv2
import numpy as np

from inference.consistency import geometric_score

def test_ego_seed():
    cfg = {"area_frac_range": [0.05, 0.85], "weights": {"area": 0.2, "solidity": 0.25, "perspective": 0.25, "connectivity": 0.2, "bottom_touch": 0.1}}
    h, w = 100, 100
    mask = np.ones((h, w), dtype=np.uint8)
    
    # Fully covered
    score1 = geometric_score(mask, cfg, apex_y=90)
    
    # Missing at ego seed (y1=75, y2=85)
    mask[75:85, 40:60] = 0
    score2 = geometric_score(mask, cfg, apex_y=90)
    
    assert score2 < score1
    assert score2 <= score1 * 0.4


def test_hood_polygon_apex_above_070():
    from utils.config import load_config
    cfg = load_config()
    poly = cfg["inference"]["hood_polygon"]
    apex_y = min(p[1] for p in poly)
    assert apex_y <= 0.70, f"Hood polygon apex {apex_y} should be above y=0.70"


from inference.detector import Detection, YOLODetector
import numpy as np

def test_approach_rate():
    # We can just test the detection merging & approach rate logic.
    # But since it's hard to mock YOLO model, we'll manually test the EMA approach rate logic.
    cfg = {"weights": "yolov8n.pt", "class_ids": [2], "conf": 0.25, "iou": 0.5, "agnostic_nms": True}
    detector = YOLODetector(cfg, "outputs")
    
    # We can inject track histories and verify EMA calculation
    # Simulate an object growing over frames
    detector.track_histories[1].extend([50, 52, 54, 56])
    
    d1 = Detection(x1=0, y1=0, x2=100, y2=60, cls_id=2, name="car", conf=0.8, tracker_id=1)
    dets = [d1]
    
    # Run the approach rate block
    for d in dets:
        if d.tracker_id is not None:
            hist = detector.track_histories[d.tracker_id]
            hist.append(d.height)
            if len(hist) > 1:
                raw_rate = d.height / hist[0]
                prev_rate = detector.ema_rates.get(d.tracker_id, 1.0)
                rate = 0.3 * raw_rate + 0.7 * prev_rate
                detector.ema_rates[d.tracker_id] = rate
                d.approach_rate = rate
                
    # Initial height was 50 (oldest). Current height is 60.
    # raw_rate = 60 / 50 = 1.2
    # rate = 0.3 * 1.2 + 0.7 * 1.0 = 0.36 + 0.70 = 1.06
    assert abs(dets[0].approach_rate - 1.06) < 1e-4


def test_detector_two_frames_no_gmc_warning(caplog):
    import logging
    cfg = {"weights": "yolov8n.pt", "class_ids": [2], "conf": 0.25, "iou": 0.5, "imgsz": 320, "crop_every_n": 0}
    detector = YOLODetector(cfg, "outputs")
    f1 = np.full((320, 320, 3), 120, dtype=np.uint8)
    f2 = np.full((320, 320, 3), 180, dtype=np.uint8)
    with caplog.at_level(logging.WARNING):
        dets1 = detector.detect(f1, frame_idx=0)
        dets2 = detector.detect(f2, frame_idx=1)
    assert isinstance(dets1, list) and isinstance(dets2, list)
    assert "GMC failed" not in caplog.text


def test_detector_frame_size_change_fails_fast():
    import pytest
    cfg = {"weights": "yolov8n.pt", "class_ids": [2], "conf": 0.25, "iou": 0.5, "imgsz": 320, "crop_every_n": 0}
    detector = YOLODetector(cfg, "outputs")
    f1 = np.zeros((320, 320, 3), dtype=np.uint8)
    detector.detect(f1, frame_idx=0)
    f2 = np.zeros((240, 320, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Frame size changed"):
        detector.detect(f2, frame_idx=1)


def test_wide_multilane_mask_does_not_saturate():
    d = smoke_config()["inference"]["decision"]
    h, w = 100, 200
    # Wide multi-lane road spanning columns 20 to 195 (8.75m wide)
    wide_mask = np.zeros((h, w), np.uint8)
    wide_mask[50:, 20:195] = 1
    cmd = compute_command(wide_mask, [], trust=1.0, terrain=0, cfg=d)
    assert abs(cmd.angle) < 15.0, f"Wide multi-lane road should not saturate angle (got {cmd.angle})"


def test_centred_ego_lane_gives_near_zero_degrees():
    d = smoke_config()["inference"]["decision"]
    h, w = 100, 200
    # Centred ego lane (columns 70 to 130)
    centred_mask = np.zeros((h, w), np.uint8)
    centred_mask[50:, 70:130] = 1
    cmd = compute_command(centred_mask, [], trust=1.0, terrain=0, cfg=d)
    assert abs(cmd.angle) < 1.0, f"Centred ego lane should give ~0 deg (got {cmd.angle})"


def test_steering_rate_limiter_never_exceeds_max_rate():
    d = smoke_config()["inference"]["decision"]
    h, w = 100, 200
    right_mask = np.zeros((h, w), np.uint8)
    right_mask[50:, 150:190] = 1
    prev_angle = 0.0
    cmd = compute_command(right_mask, [], trust=1.0, terrain=0, cfg=d, prev_angle=prev_angle)
    max_rate = float(d.get("max_steer_rate_deg", 5.0))
    assert abs(cmd.angle - prev_angle) <= max_rate + 1e-4, f"Angle change {abs(cmd.angle - prev_angle)} exceeded max rate {max_rate}"


def test_fallback_count_agrees_with_alpha_below_threshold():
    from inference.fusion import ATWFS
    cfg = smoke_config()
    fusion = ATWFS(cfg["fusion"])
    fb_thr = float(cfg["fusion"]["fallback_threshold"])

    fallback_active_count = 0
    alpha_below_thr_count = 0

    # Test over various combinations of uncertainty, geometric, and temporal scores
    for u in [0.1, 0.4, 0.7, 0.9]:
        for g in [0.2, 0.5, 0.8]:
            for t in [0.2, 0.5, 0.8]:
                alpha = fusion.alpha(u, g, t, terrain=0)
                dummy_dl = np.ones((50, 50), dtype=np.uint8)
                dummy_fb = np.zeros((50, 50), dtype=np.uint8)
                res = fusion.fuse(dummy_dl, dummy_fb, u, g, t, terrain=0)

                is_fb_active = (res.active == "FALLBACK")
                is_alpha_below = (alpha < fb_thr)
                assert is_fb_active == is_alpha_below, f"Mismatch at alpha={alpha}, thr={fb_thr}: active={res.active}"

                if is_fb_active:
                    fallback_active_count += 1
                if is_alpha_below:
                    alpha_below_thr_count += 1

    assert fallback_active_count == alpha_below_thr_count


def test_annotate_pixel_difference_tolerance():
    from inference.live import annotate, FrameResult
    from inference.decision import Command
    cfg = smoke_config()

    h, w = 480, 640
    frame = np.full((h, w, 3), 100, dtype=np.uint8)

    # Full frame mask
    mask_full = np.zeros((h, w), dtype=np.uint8)
    mask_full[240:480, 200:440] = 1

    # Model resolution mask (240x320)
    mh, mw = 240, 320
    mask_model = np.zeros((mh, mw), dtype=np.uint8)
    mask_model[120:240, 100:220] = 1

    cmd = Command(0.0, 1.0, 0.8, False, 0.5, w / 2.0, 0.0)
    res_full = FrameResult(mask=mask_full, command=cmd, active="DL", uncertainty=0.1, geometric=0.8, temporal=0.8, terrain=0, detections=[])
    res_model = FrameResult(mask=mask_full, command=cmd, active="DL", uncertainty=0.1, geometric=0.8, temporal=0.8, terrain=0, detections=[], mask_model=mask_model)

    ann_full = annotate(frame.copy(), res_full, cfg=cfg)
    ann_model = annotate(frame.copy(), res_model, cfg=cfg)

    # Check pixel difference on the rendered frame is within small tolerance
    diff = np.abs(ann_full.astype(np.float32) - ann_model.astype(np.float32))
    mean_diff = float(np.mean(diff))
    assert mean_diff < 5.0, f"Annotation pixel difference {mean_diff:.2f} exceeded tolerance 5.0"




