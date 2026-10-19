"""Forwarding stub for scripts.export_onnx --verify.

This script has been integrated into scripts.export_onnx and evaluation.test_split_eval.
Run:
    python scripts/export_onnx.py --checkpoint outputs/checkpoints/student_best.pt --output outputs/checkpoints/student_v2.onnx --verify
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_onnx import main

if __name__ == "__main__":
    main()
