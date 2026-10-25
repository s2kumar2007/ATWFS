"""Matched mock ablations and an adaptive hard negative control; no decision changes.
Run from the repository root: python -m scripts.run_sim_negative_controls --baseline <old-run-folder>
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from sim.run_carla import build_pipeline, reset_pipeline, Tee
from sim.carla_bridge import CarlaBridge
from sim.carla_scenarios import Scenario, make_scenarios
from sim.carla_metrics import Recorder
from sim.reporting import historical_records, full_summary, write_csv
from sim.audit_controls import audit_mock_controls

CONDITIONS=[
    ('L0',0,0.,0.,True),('L100',100,0.,0.,True),('L200',200,0.,0.,True),('L400',400,0.,0.,True),
    ('M30',100,.3,0.,True),('M60',100,.6,0.,True),('D3',100,0.,3.,True),
    ('OFF',100,0.,0.,False),('WORST',400,.6,0.,False),('WORST_D3',400,.6,3.,False)]
HARDER=[(9.,8.),(12.,6.),(15.,4.5),(18.,4.25)]

def control_failure(r):
    return r['collisions']>0 or (r.get('min_distance_target_m') is not None and r['min_distance_target_m']<.5) or (r.get('min_target_ttc_s') is not None and r['min_target_ttc_s']<.5)

def display(value,digits=2):return 'NA' if value is None else f'{value:.{digits}f}'

def table_row(r):
    if r.get('audit_verified'):
        # Keep legacy raw fields in CSV/JSON; show contact-consistent audited fields.
        r=dict(r, min_distance_target_m=r['audited_min_target_distance_m'],
               min_target_ttc_s=r['audited_min_target_ttc_s'],
               reaction_time_s=r.get('report_reaction_time_s'),
               reaction_censored=r.get('report_reaction_lower_bound_s') is not None,
               reaction_lower_bound_s=r.get('report_reaction_lower_bound_s'),
               stopping_distance_m=r.get('report_stopping_distance_m'),
               observed_distance_since_hazard_m=r.get('report_stopping_lower_bound_m'),
               required_stopping_distance_m=r.get('report_required_stopping_m'),
               required_stopping_lower_bound_m=r.get('report_required_stopping_lower_bound_m'))
    reaction=display(r.get('reaction_time_s'))
    if r.get('reaction_censored'):reaction='>'+display(r.get('reaction_lower_bound_s'))
    stop=display(r.get('stopping_distance_m'))
    if r.get('stopping_distance_m') is None and r.get('observed_distance_since_hazard_m') is not None:stop='not stopped; >='+display(r['observed_distance_since_hazard_m'])
    required=display(r.get('required_stopping_distance_m'))
    if r.get('required_stopping_distance_m') is None and r.get('required_stopping_lower_bound_m') is not None:required='>='+display(r['required_stopping_lower_bound_m'])
    return '| '+' | '.join([r['scenario'],r['condition'],str(r['run']),str(r['collisions']),display(r.get('min_distance_target_m')),display(r.get('min_target_ttc_s')),stop+' / '+required,reaction,str(r['supervisor_override_ticks']),str(r.get('false_stop_events',0))+'/'+str(r.get('false_stop_ticks',0))+' ('+display(None if r.get('false_stop_rate') is None else r['false_stop_rate']*100)+'%)'])+' |'

def save(out,records,meta,old,old_records):
    flat=[]
    for r in records:
        row={k:v for k,v in r.items() if k not in ('latency_samples_ms','parameters','supervisor_override_reasons')}
        row['parameters']=json.dumps(r['parameters'],sort_keys=True)
        row['supervisor_override_reasons']=json.dumps(r['supervisor_override_reasons'],sort_keys=True)
        flat.append(row)
    write_csv(out/'negative_controls.csv',flat)
    (out/'negative_controls.json').write_text(json.dumps(dict(metadata=meta,runs=records),indent=2,allow_nan=False),encoding='utf-8')
    old_rows=full_summary(old_records,3);write_csv(out/'earlier_summary_full.csv',old_rows)
    original=(old/'summary.csv').read_text();(out/'earlier_summary_original.csv').write_text(original,encoding='utf-8')
    text=['# Mock negative controls and earlier-run audit','',
          '**Perception provenance:** all earlier 15 scenario runs and all negative controls use SYNTHETIC segmentation / pixel detections with real ATWFS preprocessing, fusion, IPM and decision. They are not real-model perception accuracy or real CARLA results.',
          'The separately identified trained student_v2 ONNX + YOLO smoke ran in the mock world on CPU; its numbers are not mixed with these scenario tables. Actual CARLA/GPU: not run: needs CARLA.','',
          f"Seed {meta['seed']}; {meta['runs_per_condition']} run(s) per condition/scenario. Matched ablations, not a full factorial or a statistical safety estimate. Fixed latency is applied to commands; computation time is measured separately.",
          'Box omissions are seeded independent Bernoulli drops before the pipeline consumes detections; segmentation still sees the images. Nominal and measured miss rates are both in CSV/JSON. During the 3-second RGB-delivery blackout (at trigger), the plant/GT sensors continue and the last available commands/percepts are held. No new watchdog or safety behavior was added.',
          'Supervisor OFF bypasses only SafetySupervisor; existing ATWFS decision limits and the unchanged PID remain active. Thus supervisor OFF does not mean blind full throttle.',
          'Near-miss threshold for this report: target surface clearance <0.5 m OR target GT TTC <0.5 s. Collisions are recorded by the mock collision sensor. The hardening sequence, if needed, increases v-max and shortens the existing centre-to-centre trigger distance; every attempted level is retained.',
          '','## Conditions','', '| Condition | Delay ms | Nominal misses | RGB dropout s | Supervisor |','|---|---:|---:|---:|---|']
    for name,delay,miss,drop,on in CONDITIONS:text.append(f'| {name} | {delay} | {miss:.0%} | {drop:g} | {"ON" if on else "OFF"} |')
    text+=['','## Measured negative-control results','',
           'For audit_verified rows, distance/TTC use the replay-verified mock collision geometry; stopping/reaction are censored at first target contact or run end. Original raw metrics remain in CSV/JSON. See the geometry and censoring definitions below.', '',
           '| Scenario | Condition | Run | Collisions | Target min distance m | Target min TTC s | Stop / required m | Reaction s | Overrides ticks | False stops events/ticks (rate) |',
           '|---|---|---:|---:|---:|---:|---|---:|---:|---|']
    text += [table_row(r) for r in records]
    text+=['', '## Findings and measured fault exposure', '',
           'The baseline pedestrian geometry can clear before the vehicle arrives even with no braking; segmentation also continues to see objects when their detection boxes are omitted. A collision-free easy case therefore does not demonstrate a correct reaction. WORST_H1 deliberately raises the speed cap to 9 m/s and reduces the trigger distance to 8 m, keeping the same seed and algorithms. This harder result is not a matched comparison against the nominal 6 m/s case.',
           'Lead braking exposes the existing hold-last-perception limitation: a 3-second RGB blackout can produce a collision even with the supervisor enabled. These are measured failures, not a corrected safety policy. One run per cell does not estimate a population failure probability.', '',
           '| Scenario | Condition | Speed cap m/s | Trigger distance m | Nominal box loss | Observed box loss | Target contacts |',
           '|---|---|---:|---:|---:|---:|---:|']
    for r in records:
        text.append(f"| {r['scenario']} | {r['condition']} | {r['v_max_mps']:g} | {r['trigger_distance_m']:g} | {r['nominal_miss_rate']:.1%} | {display(None if r.get('observed_miss_rate') is None else 100*r['observed_miss_rate'])}% | {r['target_collisions']} |")
    text+=['','Stop is displacement to first speed <0.1 m/s. Required = v at reference * measured reaction + v^2/(2*a), with configured a=4 m/s^2 (an assumption; mock maximum braking is 6 m/s^2). Reaction reference is first visibility for a pedestrian and brake onset for a lead vehicle. NA means unobserved; > / >= marks right-censored values, not invented stops. False stops use 50 ms ticks and the existing eligible-period definition. Overrides exclude initial waiting for a delayed command.','',
           'For hardened cases, consult the v_max_mps, trigger_distance_m, start_distance_m and reference speed columns in negative_controls.csv. Harder rows must not be compared as the same environment as the original easy rows.','',
           '## Earlier 0-collision run: original summary.csv, every original column','', '```csv',original.rstrip(),'```','',
           'The original summary omitted distance/TTC/stopping/false-stop columns; earlier_summary_full.csv expands all stored scalar metrics. Missing earlier lead-braking reaction/stopping measurements remain NA. Legacy earlier override counts included initial command-wait ticks; the negative-control counts above exclude those waits. No earlier outputs were overwritten.','',
           '## Earlier pipeline latency (pooled original raw logged samples)','', '| Scenario | Frames | Mean ms | P95 ms |','|---|---:|---:|---:|']
    samples=[]
    for row in old_rows:
        group=[v for r in old_records if r['scenario']==row['scenario'] for v in r['latency_samples_ms']];samples.extend(group)
        text.append(f"| {row['scenario']} | {len(group)} | {np.mean(group):.6f} | {np.percentile(group,95):.6f} |")
    text.append(f'| ALL | {len(samples)} | {np.mean(samples):.6f} | {np.percentile(samples,95):.6f} |')
    text+=['','P95 was recomputed from all original frame latencies, not averaged from per-run percentiles. These synthetic-front-end FPS/latencies do not describe the trained model.','']
    text+=['','## Earlier expanded CSV: all persisted scalar metrics','', '```csv', (out/'earlier_summary_full.csv').read_text().rstrip(), '```','',
           'The original FPS values average per-run rates; the expanded file uses pooled inference-frame timing. Synthetic segmentation still sees obstacles when boxes are missed. Easy pedestrian geometry may clear before ego arrives; collision-free passage therefore does not prove a braking response. The mock records overlaps but has no impact-response dynamics, so post-contact trajectories are not crash-mechanics measurements.',
           'These two hazard scenarios are not a false-alarm specificity dataset; a zero false-stop count is not a specificity claim.',
           'Geometry caveat: raw sampled clearance/TTC use an ego-frame rectangular approximation, whereas mock contacts use world-axis boxes. Both raw values remain in CSV/JSON. When audit_verified is true, the table uses independently replayed world-axis box clearance and swept-box TTC, matching the mock contact geometry. TTC assumes constant relative velocity, including lateral motion; NA means no projected overlap. The replay must reproduce every recorded pose/speed and both total/target contact counts, or reporting fails. It executes no perception or decision code.',
           'Audited reaction/stopping values are censored at the first target contact (or run end). No post-impact stopping response is presented as valid. A lower bound is measured displacement at censoring, not a predicted stopping distance. Raw uncensored fields remain for traceability.']
    if records and all(r.get('audit_verified') for r in records):
        text+=['',f"Replay audit: {len(records)} runs verified; maximum position error {max(r['audit_max_position_error_m'] for r in records):.12g} m; maximum speed error {max(r['audit_max_speed_error_mps'] for r in records):.12g} m/s. Total collisions {sum(r['collisions'] for r in records)}; target collisions {sum(r['target_collisions'] for r in records)}."]
    real=meta.get('historical_real_model_smoke')
    if real:
        text+=['','## Separately measured historical trained-model smoke','',
               'Source: '+real['source']+'; trained student_v2 ONNX + YOLO, CPUExecutionProvider, mock world. This was a smoke test, not a trained-model negative-control experiment.',
               f"Frames {real['frames']}; collisions {real['collisions']}; pipeline FPS {real['pipeline_fps']:.6f}; mean latency {real['latency_mean_ms']:.6f} ms; p95 {real['latency_p95_ms']:.6f} ms."]
    (out/'report.md').write_text('\n'.join(text),encoding='utf-8')

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output-dir',type=Path);p.add_argument('--seed',type=int,default=0);p.add_argument('--runs',type=int,default=1)
    p.add_argument('--real-model-smoke',type=Path);p.add_argument('--seconds',type=float,default=20.);p.add_argument('--no-video',action='store_true')
    p.add_argument('--audit-existing',type=Path,help='Replay existing controls and regenerate the report without rerunning perception.')
    args=p.parse_args(argv)
    if args.runs<1 or args.seconds<=0:p.error('runs and seconds must be positive')
    old,old_records=historical_records(args.baseline)
    if args.audit_existing:
        out=args.audit_existing
        saved=json.loads((out/'negative_controls.json').read_text(encoding='utf-8'))
        records=audit_mock_controls(saved['runs'],out/'run_log.txt')
        meta=saved['metadata'];meta['control_replay_audited']=True
        save(out,records,meta,args.baseline,old_records)
        print(f'Audited {len(records)} recorded runs. Report: {out}/report.md')
        return 0
    out=args.output_dir or Path('outputs/sim_carla')/('negative_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True,exist_ok=True);records=[]
    protected=['inference/decision.py','sim/supervisor.py']
    hashes={path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in protected}
    meta=dict(seed=args.seed,runs_per_condition=args.runs,seconds=args.seconds,synthetic_perception=True,
              near_miss_distance_m=.5,near_miss_ttc_s=.5,baseline_folder=str(args.baseline),protected_code_sha256=hashes,plan=CONDITIONS,hardening_levels=HARDER)
    if args.real_model_smoke:
        model_report=json.loads((args.real_model_smoke/'metrics.json').read_text())
        if not model_report['metadata'].get('trained_perception'):raise ValueError('The supplied real-model reference is not labelled trained perception.')
        ref=model_report['runs'][0]
        meta['historical_real_model_smoke']=dict(source=str(args.real_model_smoke/'metrics.json'),**{key:ref[key] for key in ('frames','collisions','pipeline_fps','latency_mean_ms','latency_p95_ms')})
    pipe=None;console=sys.stdout;err=sys.stderr
    with (out/'run_log.txt').open('w',encoding='utf-8') as logfile:
        sys.stdout=Tee(console,logfile);sys.stderr=Tee(err,logfile)
        handler=logging.StreamHandler(logfile);root=logging.getLogger();old_level=root.level;root.setLevel(logging.INFO);root.addHandler(handler)
        try:
            pipe,source=build_pipeline(SimpleNamespace(mock=True,mock_perception='synthetic',onnx_path=None,model_path=None,detector_weights='yolov8n.pt'))
            meta.update(source);meta['fusion_checkpoint']=pipe.fusion.checkpoint_path
            params=make_scenarios(args.seed,args.runs)
            def run_cell(name,condition,j,level=0,speed=6.,trigger=None):
                tag,delay,miss,drop,on=condition;case=copy.deepcopy(params[name][j])
                if trigger is not None:case['trigger_distance']=trigger
                reset_pipeline(pipe);scene=Scenario(name,case)
                label=tag if not level else tag+f'_H{level}'
                recorder=Recorder(out/f'{name}_{label}_{j+1}.mp4') if not args.no_video and tag.startswith('WORST') else None
                try:
                    with CarlaBridge(mock=True,pipeline=pipe,latency_mode=f'fixed:{delay}',v_max=speed,seed=case['seed'],
                                     missed_detection_rate=miss,sensor_dropout_seconds=drop,supervisor_enabled=on) as bridge:
                        r=bridge.run(args.seconds,scene,recorder,j+1,f'MOCK SYNTHETIC {label} supervisor {"ON" if on else "OFF"}')
                finally:
                    if recorder:recorder.close()
                r.update(scenario=name,condition=label,run=j+1,parameters=case,delay_ms=delay,nominal_miss_rate=miss,
                         sensor_dropout_seconds=drop,supervisor_enabled=on,v_max_mps=speed,trigger_distance_m=case['trigger_distance'],
                         start_distance_m=case['start_distance_m'],hardening_level=level)
                r['control_failure']=control_failure(r);records.append(r);save(out,records,meta,args.baseline,old_records)
                print(f"DONE {name} {label}: collisions={r['collisions']} target_min_m={r['min_distance_target_m']} TTC={r['min_target_ttc_s']} failure={r['control_failure']}",flush=True)
                return r
            # Run WORST first: falsification is checked before spending time on the whole sweep.
            order=[CONDITIONS[8]]+[c for c in CONDITIONS if c[0]!='WORST']
            for condition in order:
                for name in ('pedestrian_dash','lead_brake'):
                    for j in range(args.runs):
                        r=run_cell(name,condition,j)
                        if condition[0]=='WORST' and not control_failure(r):
                            for level,(speed,trigger) in enumerate(HARDER,1):
                                r=run_cell(name,condition,j,level,speed,trigger)
                                if control_failure(r):break
                            if not control_failure(r):raise RuntimeError(f'Could not falsify worst control for {name}; inspect all retained hardening attempts, do not claim sensitivity.')
            if not any(r['collisions'] for r in records):raise RuntimeError('No collisions observed in the controls; investigate and harden further before claiming falsification.')
            for path,digest in hashes.items():
                if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise RuntimeError('Protected decision/supervisor source changed during experiment.')
            handler.flush();logfile.flush()
            audit_mock_controls(records,out/'run_log.txt');meta['control_replay_audited']=True
            save(out,records,meta,args.baseline,old_records)
            print(f'Complete: {len(records)} measured runs. Report: {out}/report.md');return 0
        except BaseException as e:
            meta['error']=str(e);save(out,records,meta,args.baseline,old_records);logging.exception('Negative-control runner failed');raise
        finally:
            if pipe:pipe.detector.stop()
            root.removeHandler(handler);root.setLevel(old_level);sys.stdout=console;sys.stderr=err

if __name__=='__main__':raise SystemExit(main())
