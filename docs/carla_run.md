# CARLA run guide

1. Install CARLA for Windows and its PythonAPI matching the server release. Install this repository's `requirements.txt` in a Python environment compatible with that CARLA release.
2. Start the CARLA server with Town01 and low quality, or pass `-CarlaExe` to the launcher.
3. From the repository root run `python -m sim.run_carla --smoke --mock` to check the CPU mock. For CARLA use `python -m sim.run_carla --smoke --host localhost --port 2000`.
4. For scenarios, provide the student ONNX path and run the scenario command documented by `python -m sim.run_carla --help`.
5. Send the timestamped folder under `outputs/sim_carla/` with its logs and metrics.

Limitations: the CARLA image domain differs from real road photos. Simulation is not proof of safety on a real car. Report held-out mIoU from the model card/evaluation artifact; it has not been measured by this bridge. Traffic-light/stop-sign state is a CARLA stand-in because ATWFS has no corresponding detector. Semantic segmentation is ground truth for mask comparison only. Current mock and bridge code does not yet provide validated CARLA scenario metrics or recordings.
