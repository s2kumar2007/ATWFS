# Negative controls without decision changes

Run the matched seeded study from the repository root:

```powershell
python -m scripts.run_sim_negative_controls --baseline outputs/sim_carla/20261025_221245_546590
```

Default: one run per condition/scenario, seed 0, 20 simulation seconds. This is a falsification experiment, not a statistical safety estimate or a full factorial. Use `--runs N` for repetitions. Cases within the study share scenario parameters/seeds. The earlier 15-run report is a separate historical experiment; exact per-run parameters are retained in JSON/CSV.

Ablations: 0/100/200/400 ms fixed command delay; nominal 30%/60% missed boxes at 100 ms; a 3 s RGB-delivery dropout at the trigger; supervisor OFF; combined 400 ms + 60% misses + supervisor OFF, with and without the blackout. Real computation time is measured separately from injected delay. Detection omissions are independent seeded Bernoulli samples; inspect observed miss rates, not just the nominal probabilities. Segmentation still sees the object even when its box is omitted.

If the combined worst case does not collide or produce target clearance <0.5 m / target TTC <0.5 s, the runner increases the speed cap and shortens the centre-to-centre trigger distance, retaining every attempted setting. It does not edit ATWFS decision logic, supervisor thresholds, segmentation, physics or model weights. The default hardening sequence is (9 m/s, 8 m), (12, 6), (15, 4.5), (18, 4.25). These are deliberately adverse stress settings, not the nominal 6 m/s scenario.

CLI equivalents for individual runs:

```powershell
python -m sim.run_carla --mock --scenario pedestrian_dash --latency-mode fixed:400 --miss-detections 0.6 --disable-supervisor
python -m sim.run_carla --mock --scenario lead_brake --latency-mode fixed:100 --sensor-dropout-seconds 3
python -m sim.run_carla --mock --scenario pedestrian_dash --latency-mode fixed:400 --miss-detections 0.6 --disable-supervisor --v-max-mps 9 --trigger-distance-m 8
```

RGB delivery to perception is suppressed for 60 simulation ticks at the trigger, while the plant and metric sensors continue. Pending commands may mature; the existing policy then holds the last available percept/command. No watchdog was added. A video can still show the simulated sensor view with stale perception overlays; the blackout is labelled. Supervisor OFF bypasses SafetySupervisor only; the unchanged ATWFS decision limits and speed PID remain. Ground-truth traffic-rule stand-ins are separate; the two mock scenarios have no such signals.

Outputs are report.md, negative_controls.csv/json, full and original historical CSV copies, run_log.txt, and videos for combined-worst/hardened cases. The original historical folder is never overwritten. Expected collisions are evidence that the negative control failed; the batch command succeeds only after observing control failure and at least one collision. The individual simulation command still reports collisions as a nonzero exit.

## Definitions and provenance

All negative-control scenario results, and all 15 earlier scenario results, use synthetic segmentation/pixel-detector test doubles with the real downstream ATWFS pipeline. They are not trained-model perception results. The separate trained student_v2 ONNX + YOLO CPU smoke has its own metadata and timings; no trained-model negative-control results are claimed. CARLA/GPU: not run: needs CARLA.

The table reports target (pedestrian or lead vehicle) surface clearance and GT TTC, alongside total collisions and target-collision counts in CSV. Before the replay audit, TTC uses the legacy longitudinal approximation; after a verified audit, it uses contact-consistent swept world-axis boxes, including lateral motion. A near-miss threshold is a reporting definition, not a claimed real-world safety boundary. Mock collisions are bounding-box overlap events and do not simulate impact response. Post-contact trajectories/stopping distances are not crash mechanics.

Pedestrian reaction is measured from first visibility; lead reaction is measured from scripted brake onset. Required distance uses speed at that reference: v*t_react + v^2/(2*a), with assumed a=4 m/s^2. Actual stopping distance is displacement to the first speed <0.1 m/s. Missing braking/stopping is right-censored: > / >= in the report denotes a measured lower bound, never an invented stop. If braking was already active at the reference, reaction=0 does not prove a prompt reaction.

False stops are events/ticks/rate in the existing eligible periods. These hazard-only scenarios do not establish false-alarm specificity. Initial command-wait ticks are excluded from new supervisor-override counts; earlier legacy counts included them. Earlier lead-braking reaction/stopping fields were not collected and remain unavailable, rather than backfilled.

The historical latency audit parses all 6,000 original logged inference samples and checks their means against stored per-run means. Scenario/global p95 is a pooled percentile, not an average of per-run percentiles. The original summary's FPS column averaged per-run FPS; the expanded CSV also provides pooled frame-based timings. Both original and full scalar summaries are retained.


## Replay and regenerate a completed experiment

```powershell
python -m scripts.run_sim_negative_controls --baseline outputs/sim_carla/20261025_221245_546590 --audit-existing outputs/sim_carla/negative_20261025_224339
```

This replays recorded actuator commands without rerunning perception. It rejects mismatched frame counts, trajectories, speeds or collision counts. Audited world-axis box clearance and swept-box TTC match the mock collision geometry; the original ego-frame approximations remain in CSV/JSON. Reaction and stopping are censored at first target contact or run end. A required-distance lower bound uses the measured reaction lower bound. The experiment report labels both raw and audited quantities; no decision or safety algorithm is altered.

## Verified results for this change

See [the full measured report](negative_control_results.md) for all 21 negative controls, the complete original and expanded historical CSV, and pooled latency. Final validation: 135 pytest tests passed; all 11 smoke stages passed; the replay matched every recorded pose/speed and all five target collisions; five MP4s decoded with 400 frames each. The unchanged decision and supervisor sources match the experiment hashes and their prior committed contents (allowing Git line-ending normalization).

All scenario numbers use synthetic perception. The separately labelled historical trained-model CPU smoke is not a trained-model negative-control experiment. CARLA/GPU: not run: needs CARLA.
