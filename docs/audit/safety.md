# Safety and resilience audit

The autonomous call path is capture -> LivePipeline -> compute_command -> SerialSender clamps -> firmware parser/clamp -> actuator writes. The standalone serial demo refuses hardware test packets and accepts --dry-run only. This is a source-level path review; arbitrary external programs can still open a serial device, so this is not an operating-system access-control guarantee.

The sender rejects nonfinite values by latching a stop, bounds angle/speed/trust, limits angular steps, applies immediate obstacle speed zero, checks stale commands before send, and monitors freshness on a separate thread. The watchdog guard triggers at 250 ms with a polling interval of 5 ms, reserving scheduling margin within the required 300 ms budget. Desktop scheduling still prevents a hard real-time guarantee. The earlier 300 ms trigger with a coarse polling interval exceeded the budget in a measured test and was corrected. The Arduino timeout is 300 ms. Hardware timing and physical braking remain NOT RUN.

The software E-stop remains latched until sender restart. Closing the sender emits a stop; lost serial writes raise an error and rely on the firmware timeout if a stop cannot be transmitted. A physical E-stop remains mandatory and is outside this software verification.

Synthetic regressions cover steering reversals, first-detection imminent obstacles, shoulder vehicles, finite inputs, low-trust hold/decay, invalid dt, empty/grayscale frames, worker failures, detector staleness, capture memory errors, camera size changes, full-queue cancellation, missing serial ports, write failures and shutdown stop. These tests do not establish road safety. Existing tracker tests cover expiry and ID histories; generalized adversarial ID-switch behavior remains NOT RUN.

The detector uses a bounded background-work queue, fixed detection cadence and skips submitting new work while busy. No new adaptive performance threshold is claimed. Detector errors now reach the inference failure path. Output video/report/feature paths are contained within the selected run directory. Tests redirect generated files into isolated test workspaces and mock detector model inference for unit-level tracker tests; real YOLO is exercised separately in video runs.

BLOCKED: hardware serial/Arduino compilation, physical brake actuation, physical emergency stop, hard real-time scheduling and physical sensor-disconnect validation require equipment not available here. CUDA provider recovery and actual process-level GPU out-of-memory recovery remain NOT RUN. Injected Python MemoryError tests the software stop path only.

Phase 4 gate: pytest PASS (exit 0, 181.0 s), smoke PASS (exit 0, 328.6 s), pyflakes BLOCKED (module missing). Logs: outputs/audit/phase_4/. Physical E-stop and Arduino hardware NOT RUN.
