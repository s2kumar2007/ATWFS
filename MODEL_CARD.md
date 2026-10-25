# Model Card: ATWFS Drivable-Region & Evidential Uncertainty Network

## 1. Model Details
- **Architecture:** Multi-Task MobileNetV3 U-Net (`MultiTaskUNet`) with `GroupNorm` throughout (replacing all `BatchNorm2d` layers for small-batch stability).
- **Teacher Backbone:** `mobilenet_v3_large` (`decoder_channels: [128, 64, 40, 24]`).
- **Student Backbone:** `mobilenet_v3_small` (`decoder_channels: [64, 32, 24, 16]`), trained via knowledge distillation from the Teacher.
- **Output Heads:**
  1. **Binary Drivable Segmentation (`seg_logits`):** 2 classes (`0 = non-drivable`, `1 = drivable`).
  2. **Single-Pass Evidential Head (`alpha`):** Dirichlet concentration parameters $\alpha = \text{softplus}(\text{clip}(z)) + 1$, providing closed-form epistemic vacuity $u = K / \sum_{k} \alpha_k \in (0, 1]$ without MC-Dropout sampling.
  3. **Night-Mode Decoder Branch (`night_decoder`, optional):** Separate low-light segmentation + evidential decoder sharing the frozen MobileNetV3 encoder, activated via road-luminance hysteresis ($Y < 45$ low threshold, $Y > 60$ high threshold, 10-frame debounce).
  4. **Terrain Head (`terrain_logits`):** Architectural branch off stride-8 encoder features (**OUT OF SCOPE / NOT TRAINED**: `inference.use_terrain: false` and `terrain = 0` everywhere).
- **Trust Fusion (`ATWFS` / `FusionNet`):** Combines deep-learning probability with a classical HSV/Lab + Canny + FloodFill fallback mask using an adaptive weight $\alpha_{\text{trust}} \in [0, 1]$ predicted from $(U, G, T, \text{terrain}=0)$.

---

## 2. Intended Use & Scope
- **Primary Use:** Real-time drivable free-space segmentation, metric corridor estimation (via self-calibrating Inverse Perspective Mapping), and safety-clamped steering/speed generation on unstructured, lane-less Indian roads.
- **Target Domain:** Dashcam video feeds (daytime, dusk/night with night decoder, unstructured urban/rural roads).
- **Out-of-Scope Items:**
  - **Terrain classification training:** Explicitly out of scope; terrain head is disabled at runtime (`terrain_acc` is reported as `"not trained (out of scope)"`).
  - **Custom dataset / CVAT / SAM auto-labelling tools:** Explicitly out of scope.

---

## 3. Training & Evaluation Data
- **Primary Dataset:** Indian Driving Dataset (IDD 20k Part I), converted to binary drivable (`level3Ids` $\{0, 1\} \to 1$, all other valid IDs $\to 0$, `255` $\to$ ignore bucket).
- **Cross-Dataset Zero-Shot Validation:** BDD100K drivable-area validation split (`evaluation/external_eval.py --bdd100k-dir <path>`).

---

## 4. Measured Performance & Computational Efficiency

All measurements below were collected on this machine (`Intel64 Family 6 Model 186 Stepping 2`, 10 physical / 16 logical cores, 15.7 GB RAM, PyTorch `2.5.1+cu121`, CPU execution) via `evaluation/evaluate.py`, `evaluation/benchmark.py`, and `evaluation/slide_metrics.py`.

### 4.1 Segmentation & Calibration Metrics (IDD Validation Split, $256 \times 256$)
| Variant | Method | Params (M) | State Size (MB) | CPU FPS ($256\times 256$) | mIoU | Drivable IoU | Non-Drivable IoU | Pixel Acc | ECE | Terrain Acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **Teacher (`mobilenet_v3_large`)** | FP32 | 4.587 | 18.428 | 29.72 | 0.9469 | 0.9220 | 0.9717 | 0.9788 | 0.0020 | `not trained (out of scope)` |
| **Student (`mobilenet_v3_small`)** | FP32 Distilled | 1.394 | 5.646 | 49.13 | 0.9415 | 0.9142 | — | 0.9766 | 0.0014 | `not trained (out of scope)` |
| **INT8 Static PTQ** | FX Graph PTQ (`onednn`) | — | 5.199 | 11.96 | 0.9339 | 0.9031 | — | 0.9734 | 0.0051 | `not trained (out of scope)` |
| **Structured Pruned** | Decoder L1 Filter Prune (30%) | 4.117 | 16.551 | 32.32 | 0.9282 | 0.8955 | — | 0.9707 | 0.0071 | `not trained (out of scope)` |

### 4.2 High-Resolution Complexity & End-to-End Pipeline Latency ($288 \times 512$ Model Input, $1280\text{ px}$ Video)
Measured by `evaluation/slide_metrics.py`:

| Metric | Teacher (`teacher_best.pt`) | Student (`student_best.pt`) | Reduction / Speedup |
|---|---:|---:|---:|
| **Parameters (Millions)** | 4.587 M | 1.394 M | **3.29x fewer** |
| **Full Checkpoint File Size** | 52.74 MB | 16.17 MB | **3.26x smaller** |
| **GMACs ($288 \times 512$)** | 4.531 GMACs | 1.685 GMACs | **2.69x fewer** |
| **GFLOPs ($288 \times 512$)** | 9.062 GFLOPs | 3.369 GFLOPs | **2.69x fewer** |
| **Model-Only CPU Latency (mean / median / p95)** | 258.87 / 223.30 / 488.79 ms | 132.31 / 129.28 / 195.69 ms | **1.96x faster** |
| **Model-Only CPU Throughput** | 3.86 FPS | 7.56 FPS | **1.96x higher** |
| **End-to-End CPU Latency (mean / median / p95)** | 316.06 / 236.29 / 613.82 ms | 158.04 / 149.06 / 211.22 ms | **2.00x faster** |
| **Peak RAM (End-to-End)** | 1069.8 MB | 893.4 MB | **-176.4 MB** |

---

## 5. Safety & Limitations
- **Classical Fallback:** When evidential uncertainty $U$ is high or geometric/temporal consistency $(G, T)$ drops, `ATWFS` smoothly shifts weight toward the classical vision estimator.
- **Hardware Safety Clamp (`arduino/safety_clamp/safety_clamp.ino`):** Enforces emergency stop on `obstacle_flag == 1`, low-trust creep speed when `trust < 0.40`, speed-dependent minimum turn radius ($R_{\min} = v^2 / a_{\text{lat,max}}$), maximum steering rate of $5^\circ/\text{cycle}$, and a 500 ms serial watchdog timeout.
