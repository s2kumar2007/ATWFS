"""Unit tests for Phase 3: Trained FusionNet."""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from inference.fusion import ATWFS, FusionNet
from training.fusion_train import train_fusionnet
from utils.config import load_config


@pytest.fixture
def base_cfg():
    return load_config("config.yaml")


def test_train_fusionnet_and_ablation_report(base_cfg):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        ckpt_out = tmp_p / "fusionnet.pt"
        report_out = tmp_p / "fusion_ablation.csv"

        res = train_fusionnet(
            cfg=base_cfg,
            checkpoint_path=None,
            output_path=str(ckpt_out),
            report_path=str(report_out),
            epochs=5,
            max_samples=16,
            smoke=True,
            device=torch.device("cpu"),
        )

        assert ckpt_out.exists(), "Trained FusionNet checkpoint was not created"
        assert report_out.exists(), "Fusion ablation report was not created"

        # Check report format
        with open(report_out, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            assert len(reader) == 2
            methods = [r["method"] for r in reader]
            assert "heuristic" in methods
            assert "trained_fusionnet" in methods
            for r in reader:
                assert 0.0 <= float(r["mean_iou"]) <= 1.0
                assert 0.0 <= float(r["fallback_rate"]) <= 1.0
                assert 0.0 <= float(r["mean_alpha"]) <= 1.0


def test_atwfs_loads_trained_checkpoint(base_cfg):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_ckpt = Path(tmp_dir) / "fusionnet.pt"
        # Create a trained FusionNet with custom weights
        net = FusionNet(base_cfg["fusion"], init_heuristic=False)
        with torch.no_grad():
            net.fc2.bias.fill_(5.0)  # High bias -> alpha close to 1.0
        torch.save(net.state_dict(), str(tmp_ckpt))

        # Pass custom checkpoint in config
        custom_cfg = {**base_cfg["fusion"], "checkpoint_path": str(tmp_ckpt)}
        atwfs = ATWFS(custom_cfg)
        
        # Test that alpha reflects the loaded weights
        alpha = atwfs.alpha(0.5, 0.5, 0.5, 0)
        assert alpha > 0.95
