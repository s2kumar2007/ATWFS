"""Shared constructor regression: the video entry point uses the same FrameResult."""
import copy
from dataclasses import fields
import cv2
import numpy as np
import torch
from inference.detector import NullDetector
from inference.live import LivePipeline, build_live_pipeline, run_video
from models.unet_mobilenet import build_model
from utils.config import smoke_config


def test_builder_matches_video_frame(tmp_path, monkeypatch):
    cfg = smoke_config()
    cfg['inference']['hood_polygon'] = 'none'
    cfg['inference']['frame_skip'] = 1
    torch.manual_seed(7)
    model = build_model(cfg['model'])
    direct = build_live_pipeline(copy.deepcopy(cfg), model=copy.deepcopy(model),
                                 detector=NullDetector(), backend='torch', device='cpu')
    assert direct.hood_poly_norm is None
    # Read the same encoded image through both paths, avoiding codec differences.
    image = np.full((96, 160, 3), 125, dtype=np.uint8)
    image[48:] = 85
    source = tmp_path / 'source.avi'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 25.0, (160, 96))
    assert writer.isOpened()
    writer.write(image)
    writer.release()
    cap = cv2.VideoCapture(str(source))
    ok, decoded = cap.read()
    cap.release()
    assert ok
    expected = direct.process_frame(decoded.copy(), 0)
    captured = []
    original = LivePipeline.process_frame
    def capture(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        captured.append(result)
        return result
    monkeypatch.setattr(LivePipeline, 'process_frame', capture)
    run_video(copy.deepcopy(cfg), str(source), str(tmp_path / 'video.mp4'),
              model=copy.deepcopy(model), detector=NullDetector(), backend='torch',
              display=False, max_frames=1, hood_polygon='none')
    assert len(captured) == 1
    actual = captured[0]
    for field in fields(expected):
        a, b = getattr(expected, field.name), getattr(actual, field.name)
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif field.name == 'command':
            for f in fields(a):
                av, bv = getattr(a, f.name), getattr(b, f.name)
                if isinstance(av, np.ndarray):
                    np.testing.assert_array_equal(av, bv)
                else:
                    assert av == bv, f.name
        elif field.name != 'ipm':
            assert a == b, field.name
