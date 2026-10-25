# Mock negative controls and earlier-run audit

**Perception provenance:** all earlier 15 scenario runs and all negative controls use SYNTHETIC segmentation / pixel detections with real ATWFS preprocessing, fusion, IPM and decision. They are not real-model perception accuracy or real CARLA results.
The separately identified trained student_v2 ONNX + YOLO smoke ran in the mock world on CPU; its numbers are not mixed with these scenario tables. Actual CARLA/GPU: not run: needs CARLA.

Seed 0; 1 run(s) per condition/scenario. Matched ablations, not a full factorial or a statistical safety estimate. Fixed latency is applied to commands; computation time is measured separately.
Box omissions are seeded independent Bernoulli drops before the pipeline consumes detections; segmentation still sees the images. Nominal and measured miss rates are both in CSV/JSON. During the 3-second RGB-delivery blackout (at trigger), the plant/GT sensors continue and the last available commands/percepts are held. No new watchdog or safety behavior was added.
Supervisor OFF bypasses only SafetySupervisor; existing ATWFS decision limits and the unchanged PID remain active. Thus supervisor OFF does not mean blind full throttle.
Near-miss threshold for this report: target surface clearance <0.5 m OR target GT TTC <0.5 s. Collisions are recorded by the mock collision sensor. The hardening sequence, if needed, increases v-max and shortens the existing centre-to-centre trigger distance; every attempted level is retained.

## Conditions

| Condition | Delay ms | Nominal misses | RGB dropout s | Supervisor |
|---|---:|---:|---:|---|
| L0 | 0 | 0% | 0 | ON |
| L100 | 100 | 0% | 0 | ON |
| L200 | 200 | 0% | 0 | ON |
| L400 | 400 | 0% | 0 | ON |
| M30 | 100 | 30% | 0 | ON |
| M60 | 100 | 60% | 0 | ON |
| D3 | 100 | 0% | 3 | ON |
| OFF | 100 | 0% | 0 | OFF |
| WORST | 400 | 60% | 0 | OFF |
| WORST_D3 | 400 | 60% | 3 | OFF |

## Measured negative-control results

For audit_verified rows, distance/TTC use the replay-verified mock collision geometry; stopping/reaction are censored at first target contact or run end. Original raw metrics remain in CSV/JSON. See the geometry and censoring definitions below.

| Scenario | Condition | Run | Collisions | Target min distance m | Target min TTC s | Stop / required m | Reaction s | Overrides ticks | False stops events/ticks (rate) |
|---|---|---:|---:|---:|---:|---|---:|---:|---|
| pedestrian_dash | WORST | 1 | 0 | 1.31 | NA | not stopped; >=82.69 / >=86.27 | >14.95 | 0 | 0/0 (0.00%) |
| pedestrian_dash | WORST_H1 | 1 | 1 | 0.00 | 0.00 | not stopped; >=8.94 / >=17.04 | >1.10 | 0 | 0/0 (0.00%) |
| lead_brake | WORST | 1 | 1 | 0.00 | 0.00 | not stopped; >=10.23 / 11.72 | 1.65 | 0 | 0/0 (0.00%) |
| pedestrian_dash | L0 | 1 | 0 | 1.37 | NA | 6.38 / 7.93 | 0.75 | 32 | 1/13 (3.88%) |
| lead_brake | L0 | 1 | 0 | 5.64 | 1.72 | 4.26 / 5.79 | 0.35 | 172 | 1/16 (14.29%) |
| pedestrian_dash | L100 | 1 | 0 | 1.37 | NA | 6.93 / 8.49 | 0.85 | 32 | 1/15 (4.48%) |
| lead_brake | L100 | 1 | 0 | 4.92 | 1.38 | 5.06 / 6.53 | 0.50 | 172 | 1/16 (15.38%) |
| pedestrian_dash | L200 | 1 | 0 | 1.36 | NA | 7.48 / 9.04 | 0.95 | 32 | 1/15 (4.48%) |
| lead_brake | L200 | 1 | 0 | 4.54 | 1.26 | 5.36 / 6.80 | 0.55 | 173 | 1/20 (17.39%) |
| pedestrian_dash | L400 | 1 | 0 | 1.45 | NA | 7.24 / 8.76 | 0.90 | 36 | 1/19 (5.67%) |
| lead_brake | L400 | 1 | 0 | 3.26 | 1.00 | 6.65 / 8.02 | 0.80 | 174 | 1/5 (4.42%) |
| pedestrian_dash | M30 | 1 | 0 | 1.39 | NA | 6.93 / 8.47 | 0.85 | 32 | 1/15 (4.46%) |
| lead_brake | M30 | 1 | 0 | 4.71 | 1.30 | 5.29 / 6.74 | 0.55 | 169 | 1/16 (15.53%) |
| pedestrian_dash | M60 | 1 | 0 | 1.41 | NA | 7.75 / 9.31 | 1.00 | 27 | 1/10 (2.98%) |
| lead_brake | M60 | 1 | 0 | 4.02 | 1.14 | 5.87 / 7.38 | 0.65 | 171 | 1/11 (11.58%) |
| pedestrian_dash | D3 | 1 | 0 | 1.16 | NA | not stopped; >=85.06 / >=87.89 | >15.25 | 0 | 0/0 (0.00%) |
| lead_brake | D3 | 1 | 1 | 0.00 | 0.00 | not stopped; >=10.02 / >=13.68 | >1.80 | 3 | 0/0 (0.00%) |
| pedestrian_dash | OFF | 1 | 0 | 1.34 | NA | not stopped; >=84.65 / >=87.89 | >15.25 | 0 | 0/0 (0.00%) |
| lead_brake | OFF | 1 | 1 | 0.00 | 0.00 | not stopped; >=9.95 / 10.98 | 1.30 | 0 | 0/0 (0.00%) |
| pedestrian_dash | WORST_D3 | 1 | 0 | 1.16 | NA | not stopped; >=83.03 / >=86.27 | >14.95 | 0 | 0/0 (0.00%) |
| lead_brake | WORST_D3 | 1 | 1 | 0.00 | 0.00 | not stopped; >=10.24 / >=13.26 | >1.95 | 0 | 0/0 (0.00%) |

## Findings and measured fault exposure

The baseline pedestrian geometry can clear before the vehicle arrives even with no braking; segmentation also continues to see objects when their detection boxes are omitted. A collision-free easy case therefore does not demonstrate a correct reaction. WORST_H1 deliberately raises the speed cap to 9 m/s and reduces the trigger distance to 8 m, keeping the same seed and algorithms. This harder result is not a matched comparison against the nominal 6 m/s case.
Lead braking exposes the existing hold-last-perception limitation: a 3-second RGB blackout can produce a collision even with the supervisor enabled. These are measured failures, not a corrected safety policy. One run per cell does not estimate a population failure probability.

| Scenario | Condition | Speed cap m/s | Trigger distance m | Nominal box loss | Observed box loss | Target contacts |
|---|---|---:|---:|---:|---:|---:|
| pedestrian_dash | WORST | 6 | 18.43 | 60.0% | 59.26% | 0 |
| pedestrian_dash | WORST_H1 | 9 | 8 | 60.0% | 57.86% | 1 |
| lead_brake | WORST | 6 | 12.97 | 60.0% | 54.29% | 1 |
| pedestrian_dash | L0 | 6 | 18.43 | 0.0% | 0.00% | 0 |
| lead_brake | L0 | 6 | 12.97 | 0.0% | 0.00% | 0 |
| pedestrian_dash | L100 | 6 | 18.43 | 0.0% | 0.00% | 0 |
| lead_brake | L100 | 6 | 12.97 | 0.0% | 0.00% | 0 |
| pedestrian_dash | L200 | 6 | 18.43 | 0.0% | 0.00% | 0 |
| lead_brake | L200 | 6 | 12.97 | 0.0% | 0.00% | 0 |
| pedestrian_dash | L400 | 6 | 18.43 | 0.0% | 0.00% | 0 |
| lead_brake | L400 | 6 | 12.97 | 0.0% | 0.00% | 0 |
| pedestrian_dash | M30 | 6 | 18.43 | 30.0% | 29.35% | 0 |
| lead_brake | M30 | 6 | 12.97 | 30.0% | 29.75% | 0 |
| pedestrian_dash | M60 | 6 | 18.43 | 60.0% | 59.09% | 0 |
| lead_brake | M60 | 6 | 12.97 | 60.0% | 58.00% | 0 |
| pedestrian_dash | D3 | 6 | 18.43 | 0.0% | 0.00% | 0 |
| lead_brake | D3 | 6 | 12.97 | 0.0% | 0.00% | 1 |
| pedestrian_dash | OFF | 6 | 18.43 | 0.0% | 0.00% | 0 |
| lead_brake | OFF | 6 | 12.97 | 0.0% | 0.00% | 1 |
| pedestrian_dash | WORST_D3 | 6 | 18.43 | 60.0% | 60.20% | 0 |
| lead_brake | WORST_D3 | 6 | 12.97 | 60.0% | 54.77% | 1 |

Stop is displacement to first speed <0.1 m/s. Required = v at reference * measured reaction + v^2/(2*a), with configured a=4 m/s^2 (an assumption; mock maximum braking is 6 m/s^2). Reaction reference is first visibility for a pedestrian and brake onset for a lead vehicle. NA means unobserved; > / >= marks right-censored values, not invented stops. False stops use 50 ms ticks and the existing eligible-period definition. Overrides exclude initial waiting for a delayed command.

For hardened cases, consult the v_max_mps, trigger_distance_m, start_distance_m and reference speed columns in negative_controls.csv. Harder rows must not be compared as the same environment as the original easy rows.

## Earlier 0-collision run: original summary.csv, every original column

```csv
scenario,runs,collisions,status,triggered_runs,functional_success_runs,distance_m,pipeline_fps,latency_mean_ms,lane_deviation_rms_m,drivable_iou_mean,reaction_time_s,supervisor_override_ticks
lane_follow,3,0,PASS,3,3,101.16565407191251,10.084393810212964,99.41147175033014,0.07594679097128829,0.9925133600789571,,61
pedestrian_dash,3,0,PASS,3,3,87.07826125433321,14.35619456619493,73.35367175015563,0.23437768485216579,0.991950535203611,0.43333333333333296,143
lead_brake,3,0,PASS,3,3,53.20713734966204,19.04158848171416,53.583898166495295,0.6353420582857675,0.9908236795317592,,492
blocked_lane,3,0,PASS,3,3,65.2434326375332,19.82139262262959,51.9842786671264,0.2993106531489575,0.9920797859769378,,356
cut_in,3,0,PASS,3,3,91.02527264604437,22.199436235891657,45.04741316678216,0.44698386966274534,0.9916654490428084,,100
```

The original summary omitted distance/TTC/stopping/false-stop columns; earlier_summary_full.csv expands all stored scalar metrics. Missing earlier lead-braking reaction/stopping measurements remain NA. Legacy earlier override counts included initial command-wait ticks; the negative-control counts above exclude those waits. No earlier outputs were overwritten.

## Earlier pipeline latency (pooled original raw logged samples)

| Scenario | Frames | Mean ms | P95 ms |
|---|---:|---:|---:|
| lane_follow | 1200 | 99.411472 | 137.319470 |
| pedestrian_dash | 1200 | 73.353672 | 89.522075 |
| lead_brake | 1200 | 53.583898 | 89.007100 |
| blocked_lane | 1200 | 51.984279 | 81.643310 |
| cut_in | 1200 | 45.047413 | 50.068325 |
| ALL | 6000 | 64.676147 | 105.632900 |

P95 was recomputed from all original frame latencies, not averaged from per-run percentiles. These synthetic-front-end FPS/latencies do not describe the trained model.


## Earlier expanded CSV: all persisted scalar metrics

```csv
scenario,runs,collisions,status,distance_m,frames,commands,fps,latency_ms,min_distance_walkers_m,min_distance_vehicles_m,min_ttc_s,lane_deviation_rms_m,drivable_iou_mean,iou_samples,reaction_time_s,pedestrian_first_visible_s,braking_at_first_visibility_runs,stopping_distance_m,required_stopping_distance_m,configured_deceleration_mps2,stopping_margin_m,false_stop_rate,false_stop_eligible_ticks,supervisor_override_ticks,pipeline_fps,latency_mean_ms,latency_p95_ms,triggered_runs,trigger_time_s,path_clear_time_s,stopped_before_hazard_runs,resumed_runs,functional_success_runs,pass_no_collision_runs,latency_percentile_source,false_stop_ticks,stopping_distance_samples,reaction_time_samples,supervisor_override_reasons
lane_follow,3,0,PASS,101.16565407191251,1200,1182,6.597143991233222,99.41147175033014,,15.47898884013075,9.817281976316066,0.07594679097128829,0.9925133600789571,120,,,0,,,4.0,,0.0,1080,61,10.059201240994394,99.41147175033014,137.31946999614593,3,0.0,,0,0,3,3,pooled raw inference-frame latencies,0,0,0,"{""pipeline TTC emergency brake"": 53, ""waiting for delayed command"": 8}"
pedestrian_dash,3,0,PASS,87.07826125433321,1200,1195,9.28817299460666,73.35367175015563,1.543294286391988,1.0388448236292875,2.0735835744538478,0.23437768485216579,0.991950535203611,120,0.43333333333333296,4.550000000000001,0,4.649976489160618,6.162668382814057,4.0,1.5126918936534384,0.04604604604604605,999,143,13.632582747950561,73.35367175015563,89.52207499823999,3,4.466666666666668,6.916666666666667,3,3,3,3,pooled raw inference-frame latencies,46,3,3,"{""pedestrian emergency hold until clear"": 137, ""pipeline TTC emergency brake"": 9, ""steering rate limit"": 4, ""waiting for delayed command"": 6}"
lead_brake,3,0,PASS,53.20713734966204,1200,1196,12.524348501003416,53.583898166495295,,5.166896628967313,1.424174842086831,0.6353420582857675,0.9908236795317592,120,,,0,,,4.0,,0.11027568922305764,399,492,18.662322716664082,53.583898166495295,89.00710002199048,3,8.416666666666666,13.616666666666667,3,3,3,3,pooled raw inference-frame latencies,44,0,0,"{""pipeline TTC emergency brake"": 27, ""vehicle emergency hold until path opens"": 477, ""waiting for delayed command"": 4}"
blocked_lane,3,0,PASS,65.2434326375332,1200,1197,12.935793010852022,51.9842786671264,,2.7659172611718574,1.6819018466535856,0.2993106531489575,0.9920797859769378,120,,,0,,,4.0,,0.04689655172413793,725,356,19.236585091491822,51.9842786671264,81.64330999570666,3,5.116666666666666,10.066666666666666,3,3,3,3,pooled raw inference-frame latencies,34,0,0,"{""pipeline TTC emergency brake"": 35, ""vehicle emergency hold until path opens"": 324, ""waiting for delayed command"": 4}"
cut_in,3,0,PASS,91.02527264604437,1200,1197,14.509172244934499,45.04741316678216,,4.806372505056526,2.4662416460421617,0.44698386966274534,0.9916654490428084,120,,,0,,,4.0,,0.0,900,100,22.19883295623726,45.04741316678216,50.068325006577645,3,5.2,8.15,0,3,3,3,pooled raw inference-frame latencies,0,0,0,"{""pipeline TTC emergency brake"": 2, ""vehicle emergency hold until path opens"": 97, ""waiting for delayed command"": 3}"
```

The original FPS values average per-run rates; the expanded file uses pooled inference-frame timing. Synthetic segmentation still sees obstacles when boxes are missed. Easy pedestrian geometry may clear before ego arrives; collision-free passage therefore does not prove a braking response. The mock records overlaps but has no impact-response dynamics, so post-contact trajectories are not crash-mechanics measurements.
These two hazard scenarios are not a false-alarm specificity dataset; a zero false-stop count is not a specificity claim.
Geometry caveat: raw sampled clearance/TTC use an ego-frame rectangular approximation, whereas mock contacts use world-axis boxes. Both raw values remain in CSV/JSON. When audit_verified is true, the table uses independently replayed world-axis box clearance and swept-box TTC, matching the mock contact geometry. TTC assumes constant relative velocity, including lateral motion; NA means no projected overlap. The replay must reproduce every recorded pose/speed and both total/target contact counts, or reporting fails. It executes no perception or decision code.
Audited reaction/stopping values are censored at the first target contact (or run end). No post-impact stopping response is presented as valid. A lower bound is measured displacement at censoring, not a predicted stopping distance. Raw uncensored fields remain for traceability.

Replay audit: 21 runs verified; maximum position error 0 m; maximum speed error 0 m/s. Total collisions 5; target collisions 5.

## Separately measured historical trained-model smoke

Source: outputs\sim_carla\verification_real_pipeline_mock\metrics.json; trained student_v2 ONNX + YOLO, CPUExecutionProvider, mock world. This was a smoke test, not a trained-model negative-control experiment.
Frames 200; collisions 0; pipeline FPS 2.184495; mean latency 457.771719 ms; p95 652.425805 ms.