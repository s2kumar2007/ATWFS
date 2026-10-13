"""Tests for Phase 6 reproducibility package (download_weights, scripts, MODEL_CARD, README)."""
from pathlib import Path
import pytest

from scripts.download_weights import download_weights


def test_download_weights_needs_user_data_when_url_empty(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    res = download_weights(
        base_url="",
        checkpoint_dir=tmp_path,
        files=["missing_teacher.pt", "missing_student.pt"],
        force=True,
    )
    assert res["status"] == "needs_user_data"
    out = capsys.readouterr().out
    assert "NEEDS USER DATA: Set paths.weights_url in config.yaml" in out


def test_download_weights_skips_when_checkpoints_exist(tmp_path: Path) -> None:
    (tmp_path / "teacher_best.pt").write_bytes(b"dummy_teacher")
    (tmp_path / "student_best.pt").write_bytes(b"dummy_student")

    res = download_weights(
        base_url="",
        checkpoint_dir=tmp_path,
        files=["teacher_best.pt", "student_best.pt"],
        force=False,
    )
    assert res["status"] == "exists"
    assert len(res["files"]) == 2


def test_reproducibility_files_and_docs_exist() -> None:
    assert Path("scripts/download_weights.py").is_file()
    assert Path("scripts/reproduce.ps1").is_file()
    assert Path("scripts/reproduce.sh").is_file()
    assert Path("MODEL_CARD.md").is_file()
    assert Path("README.md").is_file()

    readme_text = Path("README.md").read_text(encoding="utf-8")
    assert "out of scope" in readme_text.lower()
    assert "fusion_train.py (STUB)" not in readme_text

    model_card = Path("MODEL_CARD.md").read_text(encoding="utf-8")
    assert "mobilenet_v3_large" in model_card
    assert "mobilenet_v3_small" in model_card
    assert "not trained (out of scope)" in model_card
