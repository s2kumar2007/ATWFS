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

## 4. Measurement status

Historical performance tables have been removed because their checkpoint, preprocessing and run provenance were not verified in this audit. Current measured results, their exact commands and limitations are in `docs/audit/video_report.md` and `docs/audit/evaluation.md`. Model-only throughput is distinct from complete video throughput. Synthetic tests are not real-road validation.

## 5. Safety and limitations

This research prototype is not validated for road actuation. The serial sender applies steering/speed/trust bounds, an independent watchdog and a software E-stop latch. Closing the sender transmits a stop. The firmware converts speed from metres per second into normalized throttle and rejects nonfinite packets. Firmware and physical actuator behavior still require hardware validation. A physical emergency stop is mandatory and has not been tested here.

Low-trust fallback can crawl while the steering holds briefly and decays; it cannot establish that the surface is safe. Monocular image proximity is not measured metric obstacle range. Reference YOLO agreement is not human-labelled detection accuracy. The optional night decoder and fusion checkpoint need domain-specific validation. BDD100K, track behaviors and GPU throughput are not claimed as passed without a corresponding run.
