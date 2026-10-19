#!/usr/bin/env bash
# One-command GPU training pipeline (Linux / WSL / Colab / Kaggle)
#
# Usage:
#   bash scripts/train_gpu.sh --data-src /path/to/IDD_RESIZED [--resume]
#
# Options:
#   --data-src PATH     IDD_RESIZED root (required)
#   --resume            Resume from latest checkpoint
#   --config FILE       Config file (default: config.yaml)
#   --teacher-tag TAG   Tag for teacher checkpoint (default: teacher_v2)
#   --student-tag TAG   Tag for student checkpoint (default: student_v2)
#   --batch-size N      Override batch size (default: from config, 16)
#   --num-workers N     DataLoader workers (default: 4)
set -euo pipefail

# ---------- Defaults ---------------------------------------------------------
DATA_SRC=""
RESUME=0
CONFIG="config.yaml"
TEACHER_TAG="teacher_v2"
STUDENT_TAG="student_v2"
BATCH_SIZE=0
NUM_WORKERS=4

# ---------- Parse args -------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-src)    DATA_SRC="$2"; shift 2 ;;
    --resume)      RESUME=1; shift ;;
    --config)      CONFIG="$2"; shift 2 ;;
    --teacher-tag) TEACHER_TAG="$2"; shift 2 ;;
    --student-tag) STUDENT_TAG="$2"; shift 2 ;;
    --batch-size)  BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$DATA_SRC" ]]; then
  echo "ERROR: --data-src is required.  Usage: bash scripts/train_gpu.sh --data-src /path/to/IDD_RESIZED"
  exit 1
fi

# ---------- Logging ----------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p outputs/logs
TS="$(date +%Y%m%d_%H%M%S)"
LOG="outputs/logs/train_${TS}.txt"
log() { echo "$(date +%H:%M:%S)  $*" | tee -a "$LOG"; }

log "=== ATWFS GPU Training Pipeline ==="
log "Repo:    $REPO_ROOT"
log "Data:    $DATA_SRC"
log "Log:     $LOG"

# ---------- CUDA check -------------------------------------------------------
log "Checking CUDA..."
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null | while read L; do log "GPU: $L"; done || log "WARNING: nvidia-smi not found."
CUDA_OK=$(python -c "import torch; print(int(torch.cuda.is_available()))")
if [[ "$CUDA_OK" != "1" ]]; then
  log "ERROR: torch.cuda.is_available() is False. Activate a CUDA environment first."
  exit 1
fi
GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
GPU_MEM=$(python -c "import torch; print(round(torch.cuda.get_device_properties(0).total_memory/1e9,1),'GB')")
log "GPU: $GPU_NAME  ($GPU_MEM total)"

# ---------- Build dataset ----------------------------------------------------
log "Step 1/6: Building IDD dataset from scene-disjoint splits..."
python scripts/build_dataset_from_splits.py --src "$DATA_SRC" --dst data/idd_converted 2>&1 | tee -a "$LOG"

# ---------- Smoke run --------------------------------------------------------
log "Step 2/6: Smoke run (1 epoch, synthetic data)..."
T0=$(date +%s)
python -m training.train --config "$CONFIG" --synthetic --epochs 1 --tag smoke_v2 2>&1 | tee -a "$LOG"
T1=$(date +%s)
SMOKE_SEC=$((T1 - T0))
EST_MIN=$(( SMOKE_SEC * 50 / 60 ))
log "Smoke: ${SMOKE_SEC}s/epoch.  Estimated total (50ep): ${EST_MIN}min"

# ---------- Teacher training -------------------------------------------------
TEACHER_CKPT="outputs/checkpoints/${TEACHER_TAG}_best.pt"
log "Step 3/6: Training teacher ($TEACHER_TAG)..."
if [[ -f "$TEACHER_CKPT" && "$RESUME" -eq 0 ]]; then
  log "  $TEACHER_CKPT exists and --resume not set. Skipping."
else
  RESUME_ARG=""
  [[ "$RESUME" -eq 1 && -f "$TEACHER_CKPT" ]] && RESUME_ARG="--resume $TEACHER_CKPT"
  CURRENT_BS=${BATCH_SIZE}
  [[ $CURRENT_BS -eq 0 ]] && CURRENT_BS=16
  TRAIN_OK=0
  while [[ $CURRENT_BS -ge 2 && $TRAIN_OK -eq 0 ]]; do
    python -m training.train --config "$CONFIG" --tag "$TEACHER_TAG" \
      --num-workers "$NUM_WORKERS" $RESUME_ARG 2>&1 | tee -a "$LOG" && TRAIN_OK=1 || {
      EC=$?
      if [[ $EC -eq 2 ]]; then
        CURRENT_BS=$((CURRENT_BS / 2))
        log "OOM detected. Retrying with batch_size=$CURRENT_BS..."
        RESUME_ARG="$RESUME_ARG --batch-size-override $CURRENT_BS"
      else
        log "ERROR: Teacher training exited with code $EC."; exit 1
      fi
    }
  done
  [[ $TRAIN_OK -eq 0 ]] && { log "ERROR: Teacher training failed."; exit 1; }
fi
log "  Teacher checkpoint: $TEACHER_CKPT"

# ---------- Student distillation ---------------------------------------------
STUDENT_CKPT="outputs/checkpoints/${STUDENT_TAG}_best.pt"
log "Step 4/6: Distilling student ($STUDENT_TAG)..."
if [[ -f "$STUDENT_CKPT" && "$RESUME" -eq 0 ]]; then
  log "  $STUDENT_CKPT exists and --resume not set. Skipping."
else
  RESUME_ARG=""
  [[ "$RESUME" -eq 1 && -f "$STUDENT_CKPT" ]] && RESUME_ARG="--resume $STUDENT_CKPT"
  python -m training.distill --config "$CONFIG" --teacher "$TEACHER_CKPT" \
    --tag "$STUDENT_TAG" --num-workers "$NUM_WORKERS" $RESUME_ARG 2>&1 | tee -a "$LOG"
fi
log "  Student checkpoint: $STUDENT_CKPT"

# ---------- ONNX export ------------------------------------------------------
STUDENT_ONNX="outputs/checkpoints/${STUDENT_TAG}.onnx"
log "Step 5/6: ONNX export + verification..."
if [[ ! -f "$STUDENT_ONNX" ]]; then
  python scripts/export_onnx.py --checkpoint "$STUDENT_CKPT" --output "$STUDENT_ONNX" 2>&1 | tee -a "$LOG"
fi
python scripts/export_onnx.py --checkpoint "$STUDENT_CKPT" --output "$STUDENT_ONNX" \
  --verify --data-src "$DATA_SRC" --verify-n 50 2>&1 | tee -a "$LOG"

# ---------- Full test evaluation ---------------------------------------------
log "Step 6/6: Full held-out test-split evaluation..."
python -m evaluation.test_split_eval \
  --data-src "$DATA_SRC" \
  --checkpoint "$TEACHER_CKPT" \
  --student-checkpoint "$STUDENT_CKPT" \
  --onnx "$STUDENT_ONNX" \
  --baseline-checkpoint "outputs/checkpoints/student_best.pt" \
  --output "outputs/reports/final_eval_v2.json" \
  2>&1 | tee -a "$LOG"

log "=== Pipeline complete.  Log: $LOG ==="
