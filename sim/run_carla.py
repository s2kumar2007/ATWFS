"""One-command ATWFS/CARLA experiment runner; portable synthetic mock mode included."""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import logging
import math
import sys
from dataclasses import fields
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import torch
from inference.live import build_live_pipeline, LivePipeline, FrameResult
from inference.decision import Command
from inference.detector import Detection, NullDetector
from sim.carla_bridge import CarlaBridge, latency_ticks
from sim.carla_scenarios import SCENARIOS, Scenario, make_scenarios
from sim.carla_metrics import Recorder
from utils.config import load_config, resolve_path

log=logging.getLogger(__name__)

class Tee:
    def __init__(self,console,file):self.console=console;self.file=file
    def write(self,text):self.console.write(text);self.file.write(text);self.file.flush()
    def flush(self):self.console.flush();self.file.flush()
    def isatty(self):return False

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def interfaces():
    print('LivePipeline.process_frame(frame_bgr:uint8 BGR, frame_idx:int, dt:seconds, speed:m/s, imu_sample:optional dict) -> FrameResult')
    print('FrameResult fields: '+', '.join(f.name for f in fields(FrameResult)))
    print('Command fields: '+', '.join(f.name for f in fields(Command)))
    print('Detection fields: '+', '.join(f.name for f in fields(Detection)))
    print('Command.angle: degrees, positive right; Command.speed: scaled-car m/s; target_mps=clip(Command.speed/pipeline_vmax,0,1)*v_max_mps.')
    print('Detection boxes: full-frame pixels, conf/trust: 0..1. Ground truth never feeds perception or safety clamps.')

def reset_pipeline(pipe):
    detector=pipe.detector
    if hasattr(detector,'prev'):detector.prev.clear()
    if hasattr(detector,'tracker'):
        detector.tracker.reset()
        for name in ('track_histories','ema_rates','last_seen','last_real_detection_frame','consecutive_detection_count'):
            getattr(detector,name).clear()
        detector._last_active_dets=[];detector._last_shape=None;detector._last_tracker_frame=None;detector._next_track_id=10000
    # Reset temporal buffers and fusion state; underlying model/session/detector weights are reused.
    LivePipeline.__init__(pipe,copy.deepcopy(pipe.cfg),pipe.model,pipe.device,detector,
                          profile_name='simulation',onnx_session=pipe.onnx_session,fusion_override=pipe.fusion)

def build_pipeline(args):
    cfg=load_config();cfg['inference']['hood_polygon']=None
    cfg['camera'].update(height_m=1.3,hfov_deg=75.,vfov_deg=math.degrees(2*math.atan(478/848*math.tan(math.radians(37.5)))))
    cfg['inference']['detector']['synchronous']=True
    cfg['inference']['profiles']['simulation']=dict(detector_weights=args.detector_weights,detector_imgsz=480,
        detector_every_n=1,ipm_every_n=1,seg_every=1,input_size=[288,512])
    synthetic=args.mock and args.mock_perception=='synthetic' and args.onnx_path is None and args.model_path is None
    if synthetic:
        from sim.mock_perception import SyntheticSegmentation, PixelDetector
        cfg['inference']['profiles']['simulation']['input_size']=[128,224]
        cfg['inference']['night_mode']['enabled']=False
        pipe=build_live_pipeline(cfg,backend='torch',device='cpu',profile='simulation',
                                 model=SyntheticSegmentation(),detector=PixelDetector(),hood_polygon='none')
        torch.set_num_threads(2);cv2.setNumThreads(2)
        return pipe,dict(perception='SYNTHETIC pixel test doubles + real ATWFS fusion/decision',trained_perception=False)
    path=Path(args.onnx_path or args.model_path or 'outputs/checkpoints/student_v2.onnx')
    if not path.is_file():raise FileNotFoundError(f'Model missing: {path}. Supply --onnx-path or --model-path. No random model or automatic weight download is allowed in simulation.')
    detector=Path(args.detector_weights)
    if not detector.is_file():raise FileNotFoundError(f'Detector weights missing: {detector}. Supply --detector-weights pointing to existing YOLO weights.')
    cfg['inference']['profiles']['simulation']['detector_weights']=str(detector)
    is_onnx=path.suffix.lower()=='.onnx'
    device='cuda' if torch.cuda.is_available() else 'cpu'
    pipe=build_live_pipeline(cfg,onnx_path=str(path) if is_onnx else None,model_path=None if is_onnx else str(path),
                             backend='onnx' if is_onnx else 'torch',device=device,profile='simulation',hood_polygon='none',source='simulation')
    if isinstance(pipe.detector,NullDetector):raise RuntimeError('Real perception requires YOLO; detector initialized as NullDetector.')
    return pipe,dict(perception='trained segmentation + YOLO + real ATWFS fusion/decision',trained_perception=True,
                     model_path=str(path),model_sha256=sha256(path),detector_path=str(detector),detector_sha256=sha256(detector),
                     providers=pipe.onnx_session.get_providers() if pipe.onnx_session else [str(pipe.device)])

def write_results(out,records,meta,expected):
    rows=[]
    for name in dict.fromkeys(r['scenario'] for r in records):
        group=[r for r in records if r['scenario']==name];collisions=sum(r['collisions'] for r in group)
        def mean(key):
            values=[r[key] for r in group if r.get(key) is not None]
            return sum(values)/len(values) if values else None
        rows.append(dict(scenario=name,runs=len(group),collisions=collisions,
            status='PASS' if len(group)==expected and collisions==0 else 'FAIL',
            triggered_runs=sum(bool(r.get('triggered')) for r in group),functional_success_runs=sum(bool(r.get('functional_success')) for r in group),
            distance_m=mean('distance_m'),pipeline_fps=mean('pipeline_fps'),latency_mean_ms=mean('latency_mean_ms'),
            lane_deviation_rms_m=mean('lane_deviation_rms_m'),drivable_iou_mean=mean('drivable_iou_mean'),
            reaction_time_s=mean('reaction_time_s'),supervisor_override_ticks=sum(r.get('supervisor_override_ticks',0) for r in group)))
    with open(out/'summary.csv','w',newline='',encoding='utf-8') as f:
        if rows:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'metrics.json').write_text(json.dumps(dict(metadata=meta,summary=rows,runs=records),indent=2,allow_nan=False),encoding='utf-8')
    return rows

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',default='localhost');p.add_argument('--port',type=int,default=2000);p.add_argument('--town',default='Town01')
    p.add_argument('--mock',action='store_true');p.add_argument('--mock-perception',choices=['synthetic','real'],default='synthetic')
    p.add_argument('--smoke',action='store_true');p.add_argument('--scenario',choices=['all',*SCENARIOS],default='lane_follow')
    p.add_argument('--runs',type=int,default=1);p.add_argument('--seed',type=int,default=0);p.add_argument('--seconds',type=float,default=20.)
    p.add_argument('--onnx-path');p.add_argument('--model-path');p.add_argument('--detector-weights',default='yolov8n.pt')
    p.add_argument('--latency-mode',default='real');p.add_argument('--v-max-mps',type=float,default=6.);p.add_argument('--decel-mps2',type=float,default=4.)
    p.add_argument('--no-video',action='store_true',help='Skip video only for quick diagnostics; standard experiments record MP4s.')
    p.add_argument('--output-dir',type=Path);p.add_argument('--interfaces',action='store_true')
    args=p.parse_args(argv)
    if args.interfaces:interfaces();return 0
    if args.runs<1 or not math.isfinite(args.seconds) or args.seconds<=0 or not math.isfinite(args.v_max_mps) or args.v_max_mps<=0 or not math.isfinite(args.decel_mps2) or args.decel_mps2<=0:p.error('runs, seconds, v-max-mps and decel-mps2 must be positive.')
    try:latency_ticks(args.latency_mode,0.)
    except ValueError as e:p.error(str(e))
    out=args.output_dir or Path('outputs/sim_carla')/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=True)
    pipe=None;records=[];meta=dict(torch_version=torch.__version__,cuda_available=torch.cuda.is_available(),mock=args.mock,seed=args.seed,dt_s=.05,latency_mode=args.latency_mode,
        v_max_mps=args.v_max_mps,carla_validation='not run: needs CARLA' if args.mock else 'executing CARLA',
        fusion='existing ATWFS loader selects available trained fusion checkpoint or heuristic fallback; selection is logged',traffic_rule_source='CARLA light state / stop actor stand-ins',
        ground_truth_usage='scenario orchestration and evaluation only',held_out_test_miou=None,
        historical_model_card_student_validation_miou=.9415,seconds_per_run=10. if args.smoke else args.seconds)
    original_out,original_err=sys.stdout,sys.stderr
    with open(out/'run_log.txt','w',encoding='utf-8') as logfile:
        sys.stdout=Tee(original_out,logfile);sys.stderr=Tee(original_err,logfile)
        handler=logging.StreamHandler(logfile);handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
        root=logging.getLogger();old_level=root.level;root.setLevel(logging.INFO);root.addHandler(handler)
        try:
            print(f'Output folder: {out}');interfaces()
            if not torch.cuda.is_available():print('WARNING: CUDA unavailable; CPU inference enabled. This is not a GPU performance measurement.')
            if not args.mock:
                try:import carla
                except ImportError as e:raise RuntimeError('CARLA Python package missing. Install the wheel matching the server; see docs/carla_run.md.') from e
            pipe,source=build_pipeline(args);meta.update(source);meta['fusion_checkpoint']=pipe.fusion.checkpoint_path;meta['fusion_sha256']=sha256(pipe.fusion.checkpoint_path) if pipe.fusion.checkpoint_path else None;print('Perception: '+source['perception'])
            if not args.mock:print('Dedicated CARLA server required: each seeded run reloads the requested town. Traffic rules use simulator-state stand-ins.')
            names=['smoke'] if args.smoke else list(SCENARIOS) if args.scenario=='all' else [args.scenario]
            params=make_scenarios(args.seed,args.runs);n=1 if args.smoke else args.runs
            for name in names:
                recorder=None
                try:
                    if not args.no_video:recorder=Recorder(out/(name+'.mp4'))
                    for j in range(n):
                        reset_pipeline(pipe)
                        scenario=None if args.smoke else Scenario(name,params[name][j])
                        seed=args.seed if scenario is None else scenario.params['seed']
                        with CarlaBridge(args.mock,args.host,args.port,pipeline=pipe,latency_mode=args.latency_mode,
                                         v_max=args.v_max_mps,log_dir=out,town=args.town,seed=seed,reload_world=not args.mock,decel=args.decel_mps2) as bridge:
                            result=bridge.run(meta['seconds_per_run'],scenario,recorder,j+1,('MOCK - ' if args.mock else 'CARLA - ')+source['perception'])
                        result.update(scenario=name,run=j+1,seed=seed,parameters=None if scenario is None else scenario.params);records.append(result)
                        write_results(out,records,meta,n)
                        if args.smoke and (result['frames']!=200 or result['commands']==0 or result['distance_m']<=.01):
                            raise RuntimeError(f'Smoke failed: need 200 camera frames, applied commands and movement; measured {result}')
                finally:
                    if recorder:recorder.close()
            rows=write_results(out,records,meta,n)
            print('\nSCENARIO             RUNS COLLISIONS STATUS TRIGGERED FUNCTIONAL PIPELINE_FPS LATENCY_MS')
            for r in rows:print(f"{r['scenario']:<20} {r['runs']:>4} {r['collisions']:>10} {r['status']:>6} {r['triggered_runs']:>9} {r['functional_success_runs']:>10} {r['pipeline_fps'] or 0:>12.2f} {r['latency_mean_ms'] or 0:>10.2f}")
            print('Reports and videos: '+str(out));return 2 if any(r['collisions'] for r in rows) else 0
        except KeyboardInterrupt:
            meta['error']='Interrupted by user';write_results(out,records,meta,1 if args.smoke else args.runs)
            print('Interrupted; CARLA context cleanup restored settings and destroyed owned actors.');return 130
        except Exception as e:
            meta['error']=str(e);write_results(out,records,meta,1 if args.smoke else args.runs)
            log.exception('CARLA experiment failed');print(f'ERROR: {e}. See {out}/run_log.txt');return 1
        finally:
            if pipe is not None:pipe.detector.stop()
            root.removeHandler(handler);root.setLevel(old_level);sys.stdout=original_out;sys.stderr=original_err

if __name__=='__main__':raise SystemExit(main())
