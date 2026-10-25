"""Unit tests for scene-disjoint train/val/test splits."""
import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = REPO_ROOT / "data" / "splits"


def test_splits_files_exist_and_non_empty():
    """Verify that train.txt, val.txt, and test.txt exist and contain frames."""
    train_file = SPLITS_DIR / "train.txt"
    val_file = SPLITS_DIR / "val.txt"
    test_file = SPLITS_DIR / "test.txt"

    assert train_file.exists(), "train.txt missing in data/splits/"
    assert val_file.exists(), "val.txt missing in data/splits/"
    assert test_file.exists(), "test.txt missing in data/splits/"

    train_lines = [line.strip() for line in train_file.read_text().splitlines() if line.strip()]
    val_lines = [line.strip() for line in val_file.read_text().splitlines() if line.strip()]
    test_lines = [line.strip() for line in test_file.read_text().splitlines() if line.strip()]

    assert len(train_lines) > 0, "train.txt is empty"
    assert len(val_lines) > 0, "val.txt is empty"
    assert len(test_lines) > 0, "test.txt is empty"

    # All frames partition the 6993 images exactly
    total_frames = len(train_lines) + len(val_lines) + len(test_lines)
    assert total_frames == 6993, f"Expected 6993 total frames across splits, got {total_frames}"


def test_zero_image_overlap_across_splits():
    """Assert zero image overlap between train, val, and test splits."""
    train_lines = set(line.strip() for line in (SPLITS_DIR / "train.txt").read_text().splitlines() if line.strip())
    val_lines = set(line.strip() for line in (SPLITS_DIR / "val.txt").read_text().splitlines() if line.strip())
    test_lines = set(line.strip() for line in (SPLITS_DIR / "test.txt").read_text().splitlines() if line.strip())

    train_val_overlap = train_lines.intersection(val_lines)
    train_test_overlap = train_lines.intersection(test_lines)
    val_test_overlap = val_lines.intersection(test_lines)

    assert len(train_val_overlap) == 0, f"Found {len(train_val_overlap)} images overlapping between train and val"
    assert len(train_test_overlap) == 0, f"Found {len(train_test_overlap)} images overlapping between train and test"
    assert len(val_test_overlap) == 0, f"Found {len(val_test_overlap)} images overlapping between val and test"


def test_zero_scene_overlap_across_splits():
    """Assert that each drive sequence/scene appears in strictly one split."""
    scene_info_file = SPLITS_DIR / "scene_info.json"
    assert scene_info_file.exists(), "scene_info.json missing in data/splits/"

    with open(scene_info_file, "r") as f:
        meta = json.load(f)

    train_scenes = set(meta["train_scenes"])
    val_scenes = set(meta["val_scenes"])
    test_scenes = set(meta["test_scenes"])

    assert len(train_scenes.intersection(val_scenes)) == 0, "Scene overlap between train and val"
    assert len(train_scenes.intersection(test_scenes)) == 0, "Scene overlap between train and test"
    assert len(val_scenes.intersection(test_scenes)) == 0, "Scene overlap between val and test"

    # Verify that every frame in each scene belongs strictly to that split's list
    train_lines = set(line.strip() for line in (SPLITS_DIR / "train.txt").read_text().splitlines() if line.strip())
    val_lines = set(line.strip() for line in (SPLITS_DIR / "val.txt").read_text().splitlines() if line.strip())
    test_lines = set(line.strip() for line in (SPLITS_DIR / "test.txt").read_text().splitlines() if line.strip())

    for scene in meta["scenes"]:
        sid = scene["scene_id"]
        expected_split = "train" if sid in train_scenes else ("val" if sid in val_scenes else "test")
        split_set = train_lines if expected_split == "train" else (val_lines if expected_split == "val" else test_lines)

        for frame_idx in scene["frames"]:
            fname = f"Image_{frame_idx}.png"
            assert fname in split_set, f"Frame {fname} from scene {sid} expected in {expected_split} split but missing"
