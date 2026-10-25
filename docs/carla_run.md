# Windows CARLA / ATWFS run guide

## Install and prepare

Use a dedicated CARLA server: each seeded run reloads the requested town. No other client may tick the world. Owned actors are destroyed and synchronous/fixed-step settings restored on exit, errors and Ctrl+C. A forcibly killed process or crashed server requires restarting CARLA.

1. Install NVIDIA drivers and check `nvidia-smi`. Start with Low quality for about 6 GB VRAM; actual VRAM consumption has not been measured here.
2. Download and unpack a Windows CARLA 0.9.x release. UE5 has not been verified. Choose a Python version supported by that release's Windows wheel.
3. From the repository root, create/activate a virtual environment and install requirements and the CARLA wheel matching the server:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install "<CARLA-folder>/PythonAPI/carla/dist/<matching-Windows-wheel>.whl"
```

Install a CUDA PyTorch build compatible with the NVIDIA driver. For GPU ONNX execution install compatible `onnxruntime-gpu`; avoid conflicting CPU/GPU runtime packages. PyTorch CUDA availability does not establish the ONNX provider; its actual providers are logged.

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

4. Obtain the existing exported student at `outputs/checkpoints/student_v2.onnx` and YOLO weights at `yolov8n.pt`, or supply paths. No training, weight modification, automatic download or export occurs in simulation. Missing/broken requested models stop the run. `scripts/download_weights.py` still supports configured release checkpoints; no verified public model URL is configured in this repo. Get actual project weights from the model owner.

## Start and smoke test

Start `CarlaUE4.exe -quality-level=Low -windowed -ResX=848 -ResY=478`, then:

```powershell
python -m sim.run_carla --smoke --onnx-path outputs/checkpoints/student_v2.onnx
```

Or let the launcher start CARLA and wait for its port:

```powershell
.\scripts\run_carla.ps1 -Smoke -CarlaExe "<CARLA-folder>/CarlaUE4.exe" -Town Town01 -Quality low
```

Smoke runs 200 ticks (10 simulation seconds), verifies matching RGB/semantic frames, delayed commands and measured movement. Missing CUDA is a warning. Camera: 848x478, hfov 75 / implied vfov 46.78 degrees, windshield x=1.2 m/z=1.3 m, pitch zero, hood masking disabled. Inspect the recorded view for unexpected hood/mesh intrusion on another vehicle; actual rendering still needs CARLA validation.

## Full experiment

```powershell
python -m sim.run_carla --scenario all --runs 10 --seed 0 --onnx-path outputs/checkpoints/student_v2.onnx
.\scripts\run_carla.ps1 -Scenarios all -Runs 10 -Town Town01 -Quality low -OnnxPath outputs/checkpoints/student_v2.onnx
```

The launcher always runs smoke first. Options: `--host localhost --port 2000`, `--model-path <existing-checkpoint>`, `--detector-weights <existing-file>`, `--v-max-mps 6`, `--seconds 20`, `--latency-mode real` (default) or `--latency-mode fixed:150`. Fixed mode changes actuation delay only; actual FPS/latency remain measured. `--no-video` is diagnostic only. Exit 0 means execution without collisions; exit 2 means completed with collisions; exit 1 means execution failure; Ctrl+C returns 130. A zero exit alone does not establish scenario coverage or functional success.

## Outputs and CPU mock validation

Send the complete printed timestamped folder under `outputs/sim_carla/` (or the launcher's folder). It contains:

- `summary.csv`: aggregate rows per scenario. PASS means zero recorded collisions in all requested runs; triggered and functional-success counts are separate.
- `metrics.json`: metadata, per-run measurements, seeds and outcomes. Undefined/unobserved values are null, never fabricated zeros.
- `run_log.txt`: interfaces/providers, each tick's speed/position/latency, actor/trigger events, every override and clear exceptions. Launcher writes an additional transcript.
- One MP4 per scenario concatenates all runs, showing camera, drivable overlay, detections and HUD (speed, steer, trust, controls, supervisor). Video 20 FPS is simulation time, not inference throughput.

```powershell
python -m sim.run_carla --smoke --mock
python -m sim.run_carla --scenario all --runs 3 --seed 0 --mock
python -m pytest
python run_all_smoke_tests.py --stop
```

Default mock uses synthetic image segmentation and pixel-color detector test doubles plus the real LivePipeline preprocessing, classical fallback, fusion, IPM and decision. It needs no CARLA/GPU/network/trained weights. Its front ends are explicitly labelled and do not measure trained-model accuracy. Supply `--onnx-path`/`--model-path` or `--mock-perception real` to use trained front ends in the mock world. The mock has bicycle dynamics, walkers/vehicles, perspective rendering/occlusion, semantic images, collisions and seeded scenario parameters; its physics is not validated. Real processing delay varies even with the same seed; use fixed latency mode for deterministic actuation delays.

Scenarios: lane following with Traffic Manager vehicles; a walker crossing beside a parked vehicle; sudden lead braking then restart; blocked lane cleared by a scripted move after a randomized hold; scripted steering cut-in. Ego is never on autopilot. Parameters, triggers, path-clear, stop and resume outcomes are logged. A straight start of about 70 m is selected. Town-specific placement/traffic behavior: **not run: needs CARLA**.

## Metric definitions

Ground truth is collected after control selection for metrics only; scenario scripts use actor truth for orchestration. Supervisor input is pipeline outputs plus ego speed and calibration, never obstacle ground truth. Traffic light/sign state is a separately labelled rule exception.

Minimum distance approximates planar bounding-box surface clearance. Ground-truth TTC is longitudinal surface gap divided by positive closing speed for actors in the ego collision corridor. Lane RMS is nearest lane-centre waypoint distance (can jump lanes). False-stop rate is eligible ticks below 0.1 m/s after 2 seconds, excluding nearby truth blockers and simulator rule stops; legitimate perception caution can count as a false stop.

Every tenth frame compares final ATWFS mask against Road + RoadLine semantics. CARLA 0.9.14+ changed tags: Road=1, RoadLine=24, Pedestrian=12 normalize to canonical Road=7, RoadLine=6, Pedestrian=4. Raw red-channel tags are used. Poles/lights are not road.

Reaction time starts at first semantic pedestrian visibility and ends at applied brake >0.1. Already braking gives reaction=0 with an explicit flag. Actual stopping distance is displacement from visibility to first speed <0.1 m/s. Required distance is `v*measured_reaction_time+v^2/(2*configured_deceleration)`. Default deceleration 4 m/s^2 is an assumption, not measured brake performance. Unobserved visibility/braking/stopping gives null. Required-minus-actual distance is a comparison residual, not pedestrian clearance.

## Limits and stand-ins

The software supervisor estimates range/TTC from detection footpoints and a flat-ground camera, limits steering rate/curvature speed, slows/straightens on low trust and supports latched E-stop (`latch_estop`, explicit `reset`). Camera pitch errors, missed detections and tracking can invalidate TTC. Hardware Arduino behavior is not validated here.

CARLA light state and stop-sign actors are explicit stand-ins for missing ATWFS light/sign detectors, applied separately and logged as rule overrides. No traffic-sign/light perception is claimed. Existing fusion uses an available trained checkpoint or heuristic fallback; startup logs identify the choice.

An existing local report, `outputs/reports/final_eval.json`, records a **held-out test mIoU of 0.9376195228 for student_v2_onnx**, on **1,052 test images at 288x512**. This is a historical supplied measurement, not rerun by this task. That report uses stretch preprocessing; LivePipeline uses letterboxing. The report has no model-file hash, so its identity with a supplied ONNX file is not independently verified. The older model card's 0.9415 is a separate IDD validation number, not this held-out result. The simulation metadata leaves its own held-out-test measurement null because this experiment does not rerun that evaluation. Synthetic mock IoU/recall is not trained-model performance; there is a real-photo/CARLA domain gap.

Windows/NVIDIA server startup, mesh visibility, actual CARLA physics/traffic/collision timing, live semantic mapping, GPU execution and trained-model scenario safety: **not run: needs CARLA**. Simulation is not proof on a real car. Collision-free PASS without triggers/stops/resumption is insufficient evidence; inspect coverage flags and videos.

## Installation references

- [CARLA release downloads](https://github.com/carla-simulator/carla/releases)
- [CARLA quick-start guide](https://carla.readthedocs.io/en/latest/start_quickstart/)
- [CARLA Python API](https://carla.readthedocs.io/en/latest/python_api/)
- [Official semantic tag definitions](https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera)
- [CUDA PyTorch installer](https://pytorch.org/get-started/locally/)

If PowerShell blocks virtual-environment activation or the launcher, use an allowed process-local execution policy for that terminal (for example `Set-ExecutionPolicy -Scope Process Bypass`), then repeat the command. No system-wide policy change is required.
