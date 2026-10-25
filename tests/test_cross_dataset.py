"""Tests for BDD100K cross-dataset evaluation."""
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
import pytest

from evaluation.external_eval import BDD100KDataset, run_cross_dataset_eval
from utils.config import smoke_config


def create_synthetic_bdd_dataset(tmp_path: Path, num_images: int = 5) -> Path:
    """Create a minimal synthetic BDD100K directory structure."""
    images_dir = tmp_path / "images" / "100k" / "val"
    masks_dir = tmp_path / "labels" / "drivable" / "masks" / "val"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    w, h = 64, 36
    for i in range(num_images):
        # Create image
        img_arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        img = Image.fromarray(img_arr)
        img.save(images_dir / f"test_bdd_{i:04d}.jpg")

        # Create mask: 0=direct drivable (centre), 1=alternative drivable (right), 2=background (rest)
        mask_arr = np.full((h, w), 2, dtype=np.uint8)
        mask_arr[15:, 10:35] = 0  # direct
        mask_arr[15:, 35:55] = 1  # alternative
        mask = Image.fromarray(mask_arr)
        mask.save(masks_dir / f"test_bdd_{i:04d}.png")

    return tmp_path


def test_bdd100k_dataset_loader(tmp_path: Path) -> None:
    bdd_dir = create_synthetic_bdd_dataset(tmp_path, num_images=5)
    cfg = smoke_config()
    ds = BDD100KDataset(bdd_dir, cfg=cfg, direct_only=False)
    assert len(ds) == 5

    sample = ds[0]
    assert "image" in sample and "mask" in sample and "path" in sample
    # Classes 0 and 1 should both be mapped to 1
    mask_np = sample["mask"].numpy()
    assert (mask_np == 1).sum() > 0
    assert (mask_np == 0).sum() > 0
    assert set(np.unique(mask_np)).issubset({0, 1})

    # Test direct_only
    ds_direct = BDD100KDataset(bdd_dir, cfg=cfg, direct_only=True)
    sample_direct = ds_direct[0]
    mask_direct_np = sample_direct["mask"].numpy()
    # Direct only should have fewer drivable pixels than all-lanes
    assert (mask_direct_np == 1).sum() < (mask_np == 1).sum()


def test_cross_dataset_eval_synthetic(tmp_path: Path) -> None:
    bdd_dir = create_synthetic_bdd_dataset(tmp_path, num_images=5)
    out_csv = tmp_path / "cross_dataset.csv"

    # Use existing student_best checkpoint if available, else smoke checkpoint
    checkpoint = "outputs/checkpoints/student_best.pt"
    if not Path(checkpoint).exists():
        pytest.skip("outputs/checkpoints/student_best.pt not present")

    cfg = smoke_config()
    res = run_cross_dataset_eval(
        bdd100k_dir=bdd_dir,
        checkpoint=checkpoint,
        config_path=cfg,
        output_csv=str(out_csv),
        direct_only=False,
    )
    assert res is not None
    assert res["dataset"] == "BDD100K"
    assert res["num_samples"] == 5
    assert 0.0 <= res["miou"] <= 1.0
    assert 0.0 <= res["iou_drivable"] <= 1.0
    assert 0.0 <= res["mean_uncertainty"] <= 1.0
    assert 0.0 <= res["ece"] <= 1.0
    assert out_csv.exists()


def test_cross_dataset_needs_user_data_on_missing(capsys: pytest.CaptureFixture[str]) -> None:
    res = run_cross_dataset_eval(
        bdd100k_dir="non_existent_folder_xyz",
        checkpoint="outputs/checkpoints/student_best.pt",
    )
    assert res is None
    captured = capsys.readouterr()
    assert "NEEDS USER DATA: BDD100K dataset path (--bdd100k-dir)" in captured.out


def test_cross_dataset_cli_missing_exits_cleanly() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "evaluation.external_eval", "--bdd100k-dir", "non_existent_dir_123"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "NEEDS USER DATA: BDD100K dataset path (--bdd100k-dir)" in proc.stdout
