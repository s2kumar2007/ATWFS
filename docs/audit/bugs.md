# Audit bug ledger

Baseline: 03d2dfbc8da7b27be6569494a7df7713dbeeb47b; 232 tracked files. Branch real from main; origin/new already contained. No sim merge.

| ID | Status | File:line | Symptom | Root cause | Fix | Regression test | Commit |
|---|---|---|---|---|---|---|---|
| B001 | fixed | .gitignore:1 | Generated media, inference outputs, camera profiles and partial weights tracked | Incomplete ignore rules and already indexed artifacts | Removed generated artifacts from index; ignore generated trees and models | tests/test_audit_hygiene.py | phase 0 commit |

# Execution limitations

BLOCKED: git pull: fatal: unable to access remote: getaddrinfo() thread failed to start. Remote freshness unverified.

| B002 | fixed | requirements.txt:1 | CPU ONNX command and audit tools absent from installation manifest | onnxruntime, onnx, pyflakes and pyserial omitted | Added runtime/audit dependencies | tests/test_audit_contracts.py::test_runtime_dependencies | phase 1 |
| B003 | fixed | inference/live.py:166 | Non-16:9 inference geometry differs from training | Unconditional letterbox despite data.letterbox false | Honor data.letterbox | test_actual_training_and_inference_preprocessing | phase 2 |
| B004 | fixed | inference/live.py:1371 | Missing model silently runs random segmentation; bad ONNX silently selects another model | Permissive fallback and random build_model | Reject missing/corrupt requested models; no random fallback | test_missing_explicit_onnx_does_not_use_other_weights | phase 2 |
| B005 | fixed | inference/live.py:943 | Negative/NaN FPS accepted by writer | Truthiness used instead of positive finite validation | Require positive finite FPS else 25 FPS timebase | test_negative_capture_fps | phase 2 |
| B006 | fixed | inference/live.py:1729 | Report overwrites shared outputs outside --out-dir | Hardcoded report destination and fabricated baseline constants | Write measured timings only under run directory; remove hardcoded baseline | test_output_containment | phase 2 |
| B007 | fixed | inference/decision.py:388 | Sign enforcement can bypass steering rate limiter | Post-limiter angle reassignment | Apply limiter after sign correction | test_final_steering_rate_limit_after_sign_correction | phase 2 |
| B008 | fixed | arduino/safety_clamp/safety_clamp.ino:82 | Speed m/s interpreted as fraction of maximum throttle | Protocol unit mismatch | Convert m/s to normalized throttle once; ACK m/s | test_firmware_units_and_timeout | phase 2 |
| B009 | fixed | inference/serial_out.py:70 | Unclamped arbitrary packets reach serial; no latched stop/watchdog/shutdown stop | Sender only formats values | Final serial clamps, latched stop, independent watchdog, shutdown stop; dry-run-only packet demo | test_latched_estop_watchdog_and_shutdown | phase 2 |
| B010 | fixed | inference/live.py:165 | NaN, grayscale, empty frames and invalid dt fail deep in processing | Missing boundary validation | Validate BGR uint8, dt and finite decision inputs | test_invalid_frame_is_rejected / test_invalid_dt_is_rejected | phase 2 |
| B011 | fixed | inference/live.py:960 | Capture errors swallowed; blocking queue writes hang shutdown | No grabber exception channel; unbounded queue waits | Cancellable queues, capture error channel and bounded waits | test_shutdown_cancels_full_queue / test_capture_error_is_visible | phase 2 |
| B012 | fixed | inference/live.py:1870 | User errors show traceback or report success | CLI lacks exception and error-result handling | CLI catches user/runtime errors and exits 2 | test_cli_missing_input | phase 2 |
| B013 | fixed | inference/live.py:215 | Documented night thresholds unused | Hardcoded thresholds replace configuration | Read configured low/high thresholds | tests/test_night_mode.py | phase 2 |

| B014 | fixed | inference/live.py:499 | Low-visibility decay resets hold counter, repeatedly holding steering | Counter only increments for hold_angle | Continue counter through decay phase | test_low_visibility_counter_reaches_decay | phase 2 |
| B015 | fixed | inference/live.py:542 | FALLBACK HUD disagrees with trust threshold | Hysteresis overrides active display | Display FALLBACK exactly when trust below threshold | test_hud_trust_contract | phase 2 |
| B016 | fixed | inference/detector.py:217 | Detector worker error leaves stale detections and continued motion | Exceptions only logged | Propagate background detector errors to inference | test_detector_thread_failure | phase 2 |
| B017 | fixed | utils/config.py:53 | Invalid config produces deep failures | No config boundary validation | Validate config at load boundary | test_config_validation | phase 2 |
| B018 | fixed | inference/live.py:1207 | --out-dir permits output escape; hood cache writes elsewhere | Output destinations not validated | Validate output containment; run-local profiles and detector exports | test_output_containment | phase 2 |
| B019 | fixed | arduino/safety_clamp/safety_clamp.ino:139 | Invalid/NaN serial input and 500 ms timeout unsafe | Permissive toFloat parsing and no finite checks | Strict finite firmware parser and 300 ms watchdog | test_firmware_units_and_timeout | phase 2 |
