# Vision-Based Road Drivable-Region Estimation for Lane-Less Roads (ATWFS)

Adaptive Trust-Weighted Free-Space (ATWFS) estimation pipeline in PyTorch (IDD 20k Part I):
- **Backbone & Heads:** MobileNetV3-Large (Teacher) and MobileNetV3-Small (Distilled Student) U-Net with binary drivable segmentation, single-pass **evidential (Dirichlet)** uncertainty, optional **night-mode decoder** with road-luminance hysteresis, and deep supervision (`GroupNorm` throughout).
- **Self-Calibrating IPM:** Online vanishing-point horizon tracker + optional IMU complementary filter (`inference/ipm.py`) projecting the drivable mask into a bird's-eye metric corridor (road width and lateral offset in metres).
- **Trained Trust Fusion (`ATWFS`):** MLP (`FusionNet`) trained on per-frame validation features $(U, G, T, \text{terrain}=0)$ to blend deep-learning probabilities with a classical HSV/Lab + Canny + FloodFill fallback (`training/fusion_train.py`, `inference/fusion.py`).
- **Obstacle Subtraction & Hardware Link:** Synchronous/threaded YOLOv8 obstacle tracking (`inference/detector.py`), dome-shaped ego-hood masking (`inference/calibrate_roi.py`), and serial packet output (`inference/serial_out.py`) with Arduino firmware safety clamping (`arduino/safety_clamp/safety_clamp.ino`).

> **Scope Note (Explicitly Out of Scope):**
> 1. **Terrain classification head training** is out of scope (`inference.use_terrain: false` and `terrain = 0` everywhere; `terrain_acc` is reported as `"not trained (out of scope)"`).
> 2. **Own-dataset / CVAT / SAM auto-labelling tools** are out of scope.

---

## 1. Quick Start & Setup

1. Install CPU dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Optional for NVIDIA GPU)* Install CUDA wheels:
   ```bash
   pip install -r requirements-cuda.txt
   ```
2. Download or verify model weights:
   ```bash
   python scripts/download_weights.py
   ```
3. If converting the Kaggle IDD archive (`image_archive/` and `mask_archive/`):
   ```bash
   python convert_idd.py --images image_archive --masks mask_archive --out ./data/idd_converted
   ```
4. Run all 11 smoke test stages and the full pytest suite:
   ```bash
   python run_all_smoke_tests.py
   pytest
   ```

---

## 2. Repository Structure

```text
config.yaml                  All paths, camera intrinsics, hood polygon, and hyperparameters
MODEL_CARD.md                Architecture details, measured metrics, latency, and safety limits
run_all_smoke_tests.py       Runs all 11 stage smoke tests in order
scripts/
  download_weights.py        Downloads checkpoints from paths.weights_url in config.yaml
  reproduce.ps1              End-to-end reproduction script (Windows PowerShell)
  reproduce.sh               End-to-end reproduction script (Linux/macOS Bash)
arduino/
  safety_clamp/
    safety_clamp.ino         Arduino firmware enforcing turn-radius, rate, and trust clamps
data/
  labels.py                  IDD level-3 hierarchy and binary drivable collapse
  dataset.py                 IDD & synthetic dataset loaders
  augment.py                 Standard + rare-condition + synthetic night darkening augmentations
  synthetic.py               Procedural driving scenes and synthetic video generator
models/
  unet_mobilenet.py          MultiTaskUNet (day + night decoders, evidential head, GroupNorm)
  compression.py             FX static INT8 PTQ, structured L1 decoder pruning, FPS/size helpers
training/
  losses.py                  Focal + Dice + Evidential KL + Kendall multi-task loss
  metrics.py                 Confusion matrix and pixel ECE accumulators
  train.py                   Teacher training pipeline
  distill.py                 Knowledge distillation (Teacher -> Student)
  finetune_night.py          Night decoder fine-tuning on darkened or real night images
  fusion_train.py            Trains FusionNet on validation (U, G, T) features + ablation CSV
inference/
  calibrate_roi.py           Interactive/automated dome hood polygon calibration
  classical.py               Classical HSV/Lab + Canny + FloodFill road fallback
  consistency.py             Geometric (G) and warper-based temporal (T) consistency scores
  ipm.py                     Self-calibrating Inverse Perspective Mapping + IMU complementary filter
  fusion.py                  ATWFS adaptive trust fusion (loads outputs/checkpoints/fusionnet.pt)
  detector.py                YOLOv8 obstacle detector + IoU tracker with track expiry
  decision.py                Metric/pixel steering & speed control law + obstacle subtraction
  serial_out.py              PySerial / dry-run transmitter for [angle, speed, trust, obs]
  live.py                    Real-time video inference, HUD overlay, and H.264 output writer
evaluation/
  evaluate.py                AutoNUE-style official mIoU and ECE evaluation
  benchmark.py               Teacher vs Student vs INT8 PTQ vs Pruned benchmark
  external_eval.py           BDD100K zero-shot cross-dataset evaluation
  slide_metrics.py           Hardware & latency profiler (GMACs, GFLOPs, ms/frame, RAM)
  failure_taxonomy.py        Automated failure tagging and worst-frame extractor
  gradcam.py                 Grad-CAM explainability visualiser
tests/                       Comprehensive unit and stage smoke tests
```

---

## 3. Common Commands

| Task | Command |
|---|---|
| Run all smoke tests (11 stages) | `python run_all_smoke_tests.py` |
| Run full unit test suite | `pytest` |
| Calibrate dome hood polygon | `python -m inference.calibrate_roi --source dashcam_raw.mp4 --preset dome` |
| Live / raw video inference | `python -m inference.live --source dashcam_raw.mp4 --checkpoint outputs/checkpoints/student_best.pt` |
| Video inference + IMU + Serial dry-run | `python -m inference.live --source dashcam_raw.mp4 --imu-csv imu.csv --dry-run` |
| Train teacher on IDD | `python -m training.train` |
| Distil student | `python -m training.distill` |
| Fine-tune night decoder | `python -m training.finetune_night --checkpoint outputs/checkpoints/student_best.pt` |
| Train FusionNet & ablation CSV | `python -m training.fusion_train --checkpoint outputs/checkpoints/student_best.pt` |
| Evaluate checkpoint (IDD protocol) | `python -m evaluation.evaluate --checkpoint outputs/checkpoints/student_best.pt` |
| Full held-out test split evaluation | `python -m evaluation.test_split_eval --baseline-checkpoint outputs/checkpoints/student_best.pt --onnx outputs/checkpoints/student_v2.onnx` |
| Compression benchmark | `python -m evaluation.benchmark` |
| Cross-dataset BDD100K evaluation | `python -m evaluation.external_eval --bdd100k-dir /path/to/bdd100k --checkpoint outputs/checkpoints/student_best.pt` |
| End-to-end reproduce (PowerShell) | `powershell -ExecutionPolicy Bypass -File scripts/reproduce.ps1 -Synthetic` |

---

## 4. GPU Training Pipeline (Scene-Disjoint Splits)

To run training on another Windows laptop with an NVIDIA GPU or Linux/Colab/Kaggle machine:

### A. One-Command Automated Pipeline
The automated scripts perform: CUDA check -> 1-epoch smoke run (time estimation) -> teacher training (`teacher_v2`) -> student distillation (`student_v2`) -> ONNX export + 50-image verification (mask IoU $\ge$ 0.99) -> full test-split evaluation:

**Windows PowerShell:**
```powershell
.\scripts\train_gpu.ps1 -DataSrc "D:\path\to\IDD_RESIZED" [-Resume]
```

**Linux / Colab / Kaggle (Bash):**
```bash
bash scripts/train_gpu.sh --data-src /path/to/IDD_RESIZED [--resume]
```

### B. Manual Step-by-Step Training Commands

1. **Build Dataset from Scene-Disjoint Splits (4,878 train / 1,063 val / 1,052 test):**
   ```bash
   python scripts/build_dataset_from_splits.py --src /path/to/IDD_RESIZED --dst data/idd_converted --drivable-value 1
   ```

2. **Train Teacher Model (`teacher_v2`):**
   ```bash
   python -m training.train --config config.yaml --tag teacher_v2
   ```

3. **Distil Student Model (`student_v2`):**
   ```bash
   python -m training.distill --config config.yaml --teacher outputs/checkpoints/teacher_v2_best.pt --tag student_v2
   ```

4. **Export ONNX & Verify Numerical Equivalence (50 images):**
   ```bash
   python scripts/export_onnx.py --checkpoint outputs/checkpoints/student_v2_best.pt --output outputs/checkpoints/student_v2.onnx --verify --verify-n 50
   ```

5. **Evaluate Full Held-Out Test Split (288x512):**
   ```bash
   python -m evaluation.test_split_eval \
       --checkpoint outputs/checkpoints/teacher_v2_best.pt \
       --student-checkpoint outputs/checkpoints/student_v2_best.pt \
       --onnx outputs/checkpoints/student_v2.onnx \
       --baseline-checkpoint outputs/checkpoints/student_best.pt \
       --output outputs/reports/final_eval.json
   ```

6. **Run Video Inference with New Model:**
   ```bash
   python -m inference.live --source dashcam_raw.mp4 --onnx-path outputs/checkpoints/student_v2.onnx
   # Or with PyTorch checkpoint:
   python -m inference.live --source dashcam_raw.mp4 --model-path outputs/checkpoints/student_v2_best.pt
   ```

> **Important Safeguard:** `inference/live.py` inspects the input video basename and refuses to run on already-annotated files (filenames containing `"result"` or `"annotated"`) unless `--force` is explicitly passed. Always test inference on raw dashcam video (`dashcam_raw.mp4`).

## Audit and safe reproduction

The audit branch is `real`; simulation work is separate. See [requirements compliance](docs/compliance.md), [bug ledger](docs/audit/bugs.md), [video measurements](docs/audit/video_report.md), [evaluation evidence](docs/audit/evaluation.md) and [third-party inventory](docs/THIRD_PARTY.md).

Existing checkpoints must be supplied independently. `paths.weights_url` is currently empty: the download command explains the missing release URL and does not guarantee a download. ONNX/PyTorch/serial/audit dependencies are declared in requirements.txt, but clean installation remains unverified in the audited environment. Use one OpenCV distribution; GUI display requires replacing the headless package. The tracker also needs the lap dependency supplied by its upstream tracker environment.

Set `ATWFS_TEST_VIDEO` to an input video locally, then run (PowerShell):

```powershell
python -m inference.live --source "$env:ATWFS_TEST_VIDEO" --profile cpu --onnx-path outputs/checkpoints/student_v2.onnx --max-frames 300 --out-dir outputs/audit/task_a --log-features outputs/audit/task_a/features.csv
python scripts/verify_output_video.py --run-dir outputs/audit/task_a --out-dir outputs/audit/task_a/verification
python scripts/check_metric_claims.py
```

Remove `--max-frames` for the full clip. The verifier compares pipeline boxes to an independently run YOLOv8s at a lower confidence threshold; this is not labelled detection accuracy. Reports, masks, contact sheets, models and videos remain untracked. A clean checkout needs the underlying reports regenerated before metric-claim verification can pass.

Do not connect actuators based on video or synthetic tests alone. The host has a latched software stop and final command clamps; the firmware and mandatory physical emergency stop still require hardware tests. The standalone serial demo accepts test packets only in dry-run mode. No stop-sign, traffic-light, roundabout, parking or overtaking controller is claimed.
