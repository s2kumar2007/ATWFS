"""Unit tests for Phase 2: Night / Low-Light Mode."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from data.augment import synthetic_darken
from models.unet_mobilenet import MultiTaskUNet, build_model
from training.finetune_night import finetune_night
from utils.config import load_config


@pytest.fixture
def base_cfg():
    return load_config("config.yaml")


def test_synthetic_darken():
    bright = np.full((100, 100, 3), 180, dtype=np.uint8)
    darkened = synthetic_darken(bright, factor=0.25, gamma=1.8, noise_std=0.0)
    assert darkened.shape == bright.shape
    # Darkened luminance must be substantially lower than original
    assert darkened.mean() < bright.mean() * 0.40


def test_model_night_decoder_isolation(base_cfg):
    model = build_model({**base_cfg["model"], "pretrained": False})
    x = torch.randn(1, 3, 128, 128)

    # Initial day output
    out_day1 = model(x, mode="day")
    
    # Add night decoder
    night_dec = model.add_night_decoder()
    assert night_dec is not None
    assert model.night_decoder is not None

    # Before modifying night weights, day and night outputs match closely (warm-started)
    out_night_init = model(x, mode="night")
    assert torch.allclose(out_day1["seg_logits"], out_night_init["seg_logits"], atol=1e-5)

    # Perturb night decoder weights
    with torch.no_grad():
        night_dec.seg_head.weight.add_(1.0)

    # Day output must be COMPLETELY unaffected
    out_day2 = model(x, mode="day")
    assert torch.allclose(out_day1["seg_logits"], out_day2["seg_logits"], atol=1e-7)

    # Night output must reflect the change
    out_night2 = model(x, mode="night")
    assert not torch.allclose(out_day1["seg_logits"], out_night2["seg_logits"], atol=1e-3)


def test_live_pipeline_brightness_hysteresis(base_cfg):
    from inference.detector import ObstacleDetector
    from inference.live import LivePipeline

    class DummyDetector(ObstacleDetector):
        def detect(self, frame, frame_idx=0):
            return []

    model = build_model({**base_cfg["model"], "pretrained": False})
    detector = DummyDetector()
    pipeline = LivePipeline(base_cfg, model, torch.device("cpu"), detector)

    # Frame dimensions
    fh, fw = 288, 512
    bright_frame = np.full((fh, fw, 3), 160, dtype=np.uint8)
    dark_frame = np.full((fh, fw, 3), 20, dtype=np.uint8)

    # Start in DAY mode
    assert pipeline.current_mode == "day"

    # Send 5 dark frames: debounce not yet met (needs 10)
    for idx in range(5):
        res = pipeline.process_frame(dark_frame, frame_idx=idx)
        assert res.mode == "day"

    # Send 5 more dark frames (reaching 10 total)
    for idx in range(5, 10):
        res = pipeline.process_frame(dark_frame, frame_idx=idx)

    # Frame 10 must switch to NIGHT
    assert pipeline.current_mode == "night"
    assert res.mode == "night"

    # Send 5 bright frames: debounce not yet met
    for idx in range(10, 15):
        res = pipeline.process_frame(bright_frame, frame_idx=idx)
        assert res.mode == "night"

    # Send 5 more bright frames: switches back to DAY
    for idx in range(15, 20):
        res = pipeline.process_frame(bright_frame, frame_idx=idx)

    assert pipeline.current_mode == "day"
    assert res.mode == "day"


def test_finetune_night_smoke(base_cfg):
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_ckpt = Path(tmp_dir) / "student_night.pt"
        res = finetune_night(
            cfg=base_cfg,
            checkpoint_path=None,
            output_path=str(out_ckpt),
            night_dir=None,
            epochs=1,
            batch_size=4,
            smoke=True,
            device=torch.device("cpu"),
        )
        assert Path(res["output"]).exists()
        assert res["trainable_params"] > 0
        assert len(res["history"]) == 1
