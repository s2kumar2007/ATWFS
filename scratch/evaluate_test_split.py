"""Forwarding stub for evaluation.test_split_eval.

This script has been moved to evaluation.test_split_eval.
Run:
    python -m evaluation.test_split_eval [options]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.test_split_eval import main

if __name__ == "__main__":
    main()
