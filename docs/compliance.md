# Requirements and competition compliance

Status applies to the evidence described, not to physical road deployment. Competition items are mapped to the user-supplied UAMC checklist; this is not an independent certification of competition eligibility.

| ID | Requirement | Status | File and exact command / proof | Limitation |
|---|---|---|---|---|
| R1 | Lightweight multi-task MobileNetV3 U-Net with evidential head | PASS | models/unet_mobilenet.py; `python -m pytest -q tests/test_stage2_model.py` | Terrain training excluded; no new model training in audit |
| R2 | Calibrated uncertainty / ECE reported | PASS | evaluation/test_split_eval.py; `python -m evaluation.test_split_eval --onnx outputs/checkpoints/student_v2.onnx --output outputs/reports/audit_full_test.json --batch-size 4 --num-workers 0` | ECE measurement is not proof of calibration under every shift |
| R3 | Self-calibrating IPM, horizon and optional IMU | PASS | inference/ipm.py; `python -m pytest -q tests/test_ipm.py` | Software/synthetic only; physical calibration NOT RUN |
| R4 | U/G/T trust fusion, heuristic and trained FusionNet, classical fallback | PASS | inference/fusion.py; `python -m pytest -q tests/test_stage9_fusion.py tests/test_fusion_train.py` | Real video loads existing fusionnet.pt; no real-world ablation rerun |
| R5 | In-path obstacle override | PASS | inference/decision.py and serial_out.py; `python -m pytest -q tests/test_audit_safety.py` | Synthetic first-detection braking; physical reaction time NOT RUN |
| R6 | Night/low-light mode | PASS | inference/live.py; `python -m pytest -q tests/test_night_mode.py` | Real night-video accuracy NOT RUN |
| R7 | Coupled angle(deg), speed(m/s), trust, obstacle packet | PASS | inference/decision.py, serial_out.py; `python -m pytest -q tests/test_serial_out.py tests/test_audit_safety.py` | Firmware/hardware not executed |
| R8 | Distillation, INT8 and pruning: size/speed/mIoU comparison | NOT RUN | evaluation/benchmark.py; synthetic stage test: `python -m pytest -q tests/test_stage6_compression.py` | Real comparison would include pruning fine-tuning; no deployed weights changed; historical table removed |
| R9 | Held-out split, taxonomy, BDD100K, Grad-CAM | NOT RUN | `python -m pytest -q tests/test_splits_disjoint.py tests/test_stage11_eval.py tests/test_cross_dataset.py` | Held-out ONNX evaluation completed; taxonomy/Grad-CAM synthetic checks pass; BDD100K data unavailable |
| R10 | Reproducible code, weights, scripts, README, model card | FAIL | scripts/download_weights.py; `python scripts/download_weights.py` | Empty weights URL, missing v2 PyTorch pair and blocked clean installation |
| R11 | Serial link and Arduino safety clamp | NOT RUN | arduino/safety_clamp/safety_clamp.ino; `python -m pytest -q tests/test_serial_out.py tests/test_audit_safety.py` | Host software tested; Arduino compile, serial hardware and physical clamp NOT RUN |
| R12 | Supervisor target GPU throughput and measured CPU result | NOT RUN | inference/live.py; exact Task A/B commands in audit/video_report.md | CPU measured; GPU target remains unverified |
| U1 | Working E-stop | NOT RUN | inference/serial_out.py; `python -m pytest -q tests/test_audit_safety.py::test_latched_estop_watchdog_and_shutdown` | Software latch tested; mandatory physical E-stop not covered |
| U2 | No teleoperation in timed autonomous runs | PASS | inference/live.py and serial_out.py source review; serial demo rejects hardware packets | Software control path only; no timed competition run performed |
| U3 | Round-one lane following and obstacle stop on self-built track | NOT RUN | Task A/B are raw-road video processing, not a self-built track demo | Needs a track video and vehicle/track execution |
| U4 | Later-round stop sign, pedestrian crossing, traffic light, roundabout, parking, overtaking | FAIL | Detector class filters and decision.py source review | Pedestrian detection/obstacle stop exists; full crossing protocol, signs/lights, roundabout, parking and overtaking behavior do not exist |
| U5 | Original design and third-party attribution | NOT RUN | docs/THIRD_PARTY.md | Inventory supplied; original vehicle authorship/eligibility not established by software tests |
| O1 | Terrain-head training | OUT OF SCOPE | config.yaml inference.use_terrain=false; MODEL_CARD.md | Deliberately not trained |
| O2 | Own-dataset / CVAT / SAM tooling | OUT OF SCOPE | README.md scope note | Not part of this audit |

Runtime failures and remaining perception/visual findings are in audit/bugs.md and audit/video_report.md. A PASS for software unit checks is not an actuator or competition PASS.
