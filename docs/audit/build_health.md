# Phase 1 build health

Fresh environment created with `python -m venv .venv-audit`.
`pip install -r requirements.txt --retries 0 --timeout 10` failed: No matching distribution found for torch>=2.0.0 (index returned no versions).
BLOCKED: package index unavailable; no pinned-version existence claims made. requirements-cuda.txt selects CUDA 12.1 wheels and remains optional.
Installed host Python 3.11.9 has torch, cv2, pytest, onnxruntime, onnx, yaml and ultralytics. pyflakes is absent; installation also failed with no available versions.
NOT RUN: clean-environment README execution, pending dependency installation. Existing environment used for remaining checks.
