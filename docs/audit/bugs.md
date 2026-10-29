# Audit bug ledger

Baseline: 03d2dfbc8da7b27be6569494a7df7713dbeeb47b; 232 tracked files. Branch real from main; origin/new already contained. No sim merge.

| ID | Status | File:line | Symptom | Root cause | Fix | Regression test | Commit |
|---|---|---|---|---|---|---|---|
| B001 | fixed | .gitignore:1 | Generated media, inference outputs, camera profiles and partial weights tracked | Incomplete ignore rules and already indexed artifacts | Removed generated artifacts from index; ignore generated trees and models | tests/test_audit_hygiene.py | phase 0 commit |

# Execution limitations

BLOCKED: git pull: fatal: unable to access remote: getaddrinfo() thread failed to start. Remote freshness unverified.

| B002 | fixed | requirements.txt:1 | CPU ONNX command and audit tools absent from installation manifest | onnxruntime, onnx, pyflakes and pyserial omitted | Added runtime/audit dependencies | tests/test_audit_contracts.py::test_runtime_dependencies | phase 1 |
| B003 | found | inference/live.py:166 | Non-16:9 inference geometry differs from training | Unconditional letterbox despite data.letterbox false | Pending | Pending | Pending |
| B004 | found | inference/live.py:1371 | Missing model silently runs random segmentation; bad ONNX silently selects another model | Permissive fallback and random build_model | Pending | Pending | Pending |
| B005 | found | inference/live.py:943 | Negative/NaN FPS accepted by writer | Truthiness used instead of positive finite validation | Pending | Pending | Pending |
| B006 | found | inference/live.py:1729 | Report overwrites shared outputs outside --out-dir | Hardcoded report destination and fabricated baseline constants | Pending | Pending | Pending |
| B007 | found | inference/decision.py:388 | Sign enforcement can bypass steering rate limiter | Post-limiter angle reassignment | Pending | Pending | Pending |
| B008 | found | arduino/safety_clamp/safety_clamp.ino:82 | Speed m/s interpreted as fraction of maximum throttle | Protocol unit mismatch | Pending | Pending | Pending |
| B009 | found | inference/serial_out.py:70 | Unclamped arbitrary packets reach serial; no latched stop/watchdog/shutdown stop | Sender only formats values | Pending | Pending | Pending |
| B010 | found | inference/live.py:165 | NaN, grayscale, empty frames and invalid dt fail deep in processing | Missing boundary validation | Pending | Pending | Pending |
| B011 | found | inference/live.py:960 | Capture errors swallowed; blocking queue writes hang shutdown | No grabber exception channel; unbounded queue waits | Pending | Pending | Pending |
| B012 | found | inference/live.py:1870 | User errors show traceback or report success | CLI lacks exception and error-result handling | Pending | Pending | Pending |
| B013 | found | inference/live.py:215 | Documented night thresholds unused | Hardcoded thresholds replace configuration | Pending | Pending | Pending |

