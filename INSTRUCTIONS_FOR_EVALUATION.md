# ATWFS Project Evaluation & Execution Guide

**Project:** Vision-Based Road Drivable Region Estimation for Lane-Less Roads with Adaptive Trust-Weighted Fusion Segmentation (ATWFS)  
**Course:** BACSE324 – Embedded Systems Architecture  
**Authors:** Sunil Kumar (25BCE0388), Varshith R S (25BCE0351), Suyash Gupta (25BCE0371)

---

## 🚀 Quick Start (Under 3 Minutes)

This package contains the complete, self-contained codebase, pre-trained model weights, and sample test video. Follow the steps below to run the project on a fresh machine.

### Step 1: Open Terminal & Navigate to Project Root
```bash
cd ATWFS
```

### Step 2: Create & Activate a Python Virtual Environment
*(Recommended: Python 3.10 or 3.11)*

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```
*(Optional for NVIDIA GPU acceleration):*
```bash
pip install -r requirements-cuda.txt
```

---

## 🎬 Running the Project

### 1. Run Real-Time Drivable Region Estimation (Sample Video)
To execute the live perception pipeline with HUD telemetry, dynamic obstacle avoidance, and steering guidance on the included sample road video:
```bash
python -m inference.live --source data/sample_laneless.mp4 --fast
```
* **Output:** Displays the live in-cab HUD with Dodger Blue drivable free-space mask, yellow Hermite trajectory spline, YOLOv8 obstacle bounding boxes, and rolling FPS telemetry.

### 2. Run on Your Own Video or Webcam
```bash
# Custom video file
python -m inference.live --source path/to/your_video.mp4 --fast

# Live USB Webcam (Device 0)
python -m inference.live --source 0 --fast
```

### 3. Run Automated Unit & Safety Test Suite
To verify the complete test suite across all architecture, loss, IPM, fusion, and kinematic safety components:
```bash
pytest
```
* **Expected Result:** `159 passed` (100% test pass rate).

### 4. Run Stage-by-Stage Smoke Tests
To run all 11 architectural verification stages sequentially:
```bash
python run_all_smoke_tests.py
```

---

## 📁 Repository Structure & Key Assets

```text
ATWFS/
├── INSTRUCTIONS_FOR_EVALUATION.md   <-- This quick-start evaluation guide
├── README.md                        <-- Full technical documentation & theory
├── config.yaml                      <-- System configuration, camera intrinsics & thresholds
├── requirements.txt                 <-- Base Python dependencies (CPU/PyTorch/ONNX/OpenCV)
├── requirements-cuda.txt            <-- GPU-accelerated PyTorch dependencies
├── pytest.ini                       <-- Pytest configuration
├── run_all_smoke_tests.py           <-- End-to-end 11-stage pipeline verification
│
├── data/
│   └── sample_laneless.mp4          <-- Ready-to-run sample unstructured road video
│
├── outputs/
│   ├── checkpoints/                 <-- Pre-trained deep models
│   │   ├── student_v2_best.onnx     <-- Primary ONNX INT8/FP16 production student model
│   │   ├── student_v2_best.pt       <-- PyTorch MobileNetV3-Small student checkpoint
│   │   └── teacher_v2_best.pt       <-- PyTorch MobileNetV3-Large teacher checkpoint
│   ├── weights/
│   │   └── yolov8n.pt               <-- Lightweight dynamic obstacle detector weights
│   └── camera_profiles/             <-- Video camera calibration JSONs (pitch/roll homography)
│
├── inference/                       <-- Real-time inference, IPM, fusion, and HUD annotation
│   ├── live.py                      <-- Main live video pipeline entry point
│   ├── fusion.py                    <-- ATWFS trust arbitration & logistic blending
│   ├── detector.py                  <-- Thread-decoupled asynchronous YOLOv8 obstacle gating
│   └── ipm.py                       <-- Self-calibrating Inverse Perspective Mapping
│
├── models/                          <-- MobileNetV3 U-Net, Evidential, and Terrain heads
├── training/                        <-- Multi-task distillation loss & training routines
├── evaluation/                      <-- Metric evaluation & comparative slide benchmarks
└── tests/                           <-- Complete 159-test verification suite
```

---

## 📊 Presentation & Deliverables
- **PowerPoint Presentation:** `CV Based Drivable Region Estimation Final.pptx` (Includes embedded output demonstration video).
- **PDF Report:** `CV_Based_Drivable_Region_Estimation_Final.pdf`.
