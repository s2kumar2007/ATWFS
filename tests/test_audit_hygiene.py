"""Repository hygiene regression checks."""
from pathlib import Path
import subprocess


def test_generated_artifacts_are_not_tracked():
    root = Path(__file__).resolve().parents[1]
    files = subprocess.check_output(['git', 'ls-files'], cwd=root, text=True).splitlines()
    forbidden = ('outputs/', 'check_frames/', 'weights/')
    assert not [p for p in files if p.startswith(forbidden) or p.endswith(('.onnx', '.pt', '.mp4', '.part'))]
