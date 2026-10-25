#!/usr/bin/env bash
set -euo pipefail

CONFIG="config.yaml"
SYN_FLAG=""
CPU_FLAG=""

for arg in "$@"; do
  case $arg in
    --synthetic)
      SYN_FLAG="--synthetic"
      CPU_FLAG="--cpu-fallback"
      ;;
    --config=*)
      CONFIG="${arg#*=}"
      ;;
  esac
done

echo "=== ATWFS Reproducibility Pipeline ==="

echo "[1/7] Checking / Downloading Weights..."
python scripts/download_weights.py --config "$CONFIG"

echo "[2/7] Stage 1-4: Training Teacher..."
python -m training.train --config "$CONFIG" $SYN_FLAG $CPU_FLAG

echo "[3/7] Stage 5: Distilling Student..."
python -m training.distill --config "$CONFIG" $SYN_FLAG

echo "[4/7] Stage 6: Compression & Benchmark..."
python -m evaluation.benchmark --config "$CONFIG" $SYN_FLAG

echo "[5/7] Stage 9: Training FusionNet & Ablation Report..."
python -m training.fusion_train --config "$CONFIG"

echo "[6/7] Stage 11: IDD Protocol Evaluation..."
CKPT="outputs/checkpoints/student_best.pt"
if [ ! -f "$CKPT" ]; then
  CKPT="outputs/checkpoints/teacher_best.pt"
fi
python -m evaluation.evaluate --config "$CONFIG" --checkpoint "$CKPT" $SYN_FLAG

echo "[7/7] Running Smoke Tests..."
python run_all_smoke_tests.py

echo "=== Reproducibility Pipeline Complete ==="
