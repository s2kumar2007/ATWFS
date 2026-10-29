# Phase 1 build health

Fresh environment created with `python -m venv .venv-audit`.
`pip install -r requirements.txt --retries 0 --timeout 10` failed: No matching distribution found for torch>=2.0.0 (index returned no versions).
BLOCKED: package index unavailable; no pinned-version existence claims made. requirements-cuda.txt selects CUDA 12.1 wheels and remains optional.
Installed host Python 3.11.9 has torch, cv2, pytest, onnxruntime, onnx, yaml and ultralytics. pyflakes is absent; installation also failed with no available versions.
NOT RUN: clean-environment README execution, pending dependency installation. Existing environment used for remaining checks.

## PyPI release existence (web verification)

The shell package index remains unavailable. As a separate check, official PyPI release pages and their distribution filenames were retrieved on the audit date for every pinned CPU dependency:

| Package | Release page |
|---|---|
| numpy | https://pypi.org/project/numpy/2.4.4/ and https://pypi.org/project/numpy/2.2.6/ |
| scipy | https://pypi.org/project/scipy/1.17.1/ and https://pypi.org/project/scipy/1.15.3/ |
| opencv-python-headless | https://pypi.org/project/opencv-python-headless/4.13.0.92/ |
| pillow | https://pypi.org/project/pillow/12.1.1/ |
| PyYAML | https://pypi.org/project/PyYAML/6.0.3/ |
| matplotlib | https://pypi.org/project/matplotlib/3.10.8/ |
| pytest | https://pypi.org/project/pytest/9.1.1/ |
| ultralytics | https://pypi.org/project/ultralytics/8.4.173/ |

These releases exist; installation and mutual compatibility in a fresh environment remain NOT RUN. The host environment differs from requirements: numpy 2.4.6, OpenCV 5.0.0.93, Pillow 12.3.0, pytest 9.0.2 and Ultralytics 8.4.134. All video/evaluation results belong to that host environment, not the pinned clean environment. Torch 2.5.1+cu121 and torchvision 0.20.1+cu121 are installed locally; CUDA wheel pins are from the PyTorch index, not PyPI.

## Branch bookkeeping

The initial setup commit landed on local main after HEAD changed back from real during setup. The identical commit was fast-forwarded onto real, and the main reference was restored to its original baseline once verified to contain only that audit commit. No user commits were removed and no history was pushed to main. All subsequent audit commits are on real.

Phase 5 gate: see outputs/audit/phase_5/results.json.
