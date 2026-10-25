"""Run CARLA bridge; use --mock for CPU-only integration."""
import argparse, logging, json
from datetime import datetime
from pathlib import Path
import torch
from sim.carla_bridge import CarlaBridge

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--host',default='localhost');p.add_argument('--port',type=int,default=2000);p.add_argument('--mock',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--seconds',type=float,default=10);p.add_argument('--onnx-path');p.add_argument('--latency-mode',default='real');p.add_argument('--v-max-mps',type=float,default=6.)
 a=p.parse_args(argv);out=Path('outputs/sim_carla')/datetime.now().strftime('%Y%m%d_%H%M%S');out.mkdir(parents=True,exist_ok=True)
 logging.basicConfig(filename=out/'run_log.txt',level=logging.INFO)
 if not torch.cuda.is_available(): print('WARNING: CUDA is unavailable; using CPU inference.')
 try:
  from utils.config import load_config
  cfg=load_config(); cfg['inference']['hood_polygon']=None
  from inference.live import build_live_pipeline
  pipe=build_live_pipeline(cfg,onnx_path=a.onnx_path,backend='onnx' if a.onnx_path else None,device='cuda' if torch.cuda.is_available() else 'cpu',hood_polygon='none')
  with CarlaBridge(a.mock,a.host,a.port,pipeline=pipe,latency_mode=a.latency_mode,v_max=a.v_max_mps,log_dir=out) as bridge: metrics=bridge.run(a.seconds)
  print(metrics)
  logging.info("Measured mock=%s metrics=%s",a.mock,metrics)
  (out/'metrics.json').write_text(json.dumps(dict(mock=a.mock,**metrics),indent=2),encoding='utf-8')
  if a.smoke and (metrics['frames']<int(a.seconds/.05) or metrics['commands']==0 or metrics['distance_m']<=0.01): raise RuntimeError('Smoke test failed: camera frames, applied commands, or vehicle movement missing.')
  return 0
 except KeyboardInterrupt: print('Interrupted; restoring CARLA state and destroying actors.'); return 130
 except Exception as e: logging.exception('CARLA run failed'); print(f'CARLA run failed: {e}'); return 1
if __name__=='__main__':raise SystemExit(main())
