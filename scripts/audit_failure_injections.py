"""Run isolated CLI failure injections without writing outside the audit directory."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import cv2
import numpy as np


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--onnx',required=True)
    ap.add_argument('--out-dir',required=True)
    a=ap.parse_args()
    root=Path(a.out_dir).resolve()
    root.mkdir(parents=True,exist_ok=True)
    empty=root/'empty.mp4';empty.write_bytes(b'')
    corrupt=root/'corrupt.mp4';corrupt.write_bytes(b'invalid video')
    badmodel=root/'corrupt.onnx';badmodel.write_bytes(b'invalid model')
    clip=root/'input.avi'
    wr=cv2.VideoWriter(str(clip),cv2.VideoWriter_fourcc(*'MJPG'),25,(160,90))
    for i in range(3):wr.write(np.full((90,160,3),i*30,np.uint8))
    wr.release()
    for i in range(3):cv2.imwrite(str(root/f'frame_{i:03d}.png'),np.full((80,120),50+i*20,np.uint8))
    large=root/'large.avi'
    wr=cv2.VideoWriter(str(large),cv2.VideoWriter_fourcc(*'MJPG'),25,(3840,2160))
    wr.write(np.zeros((2160,3840,3),np.uint8));wr.release()
    cases=[('missing_video',root/'absent.mp4',a.onnx,2),('empty_video',empty,a.onnx,2),
           ('corrupt_video',corrupt,a.onnx,2),('missing_model',clip,root/'absent.onnx',2),
           ('corrupt_model',clip,badmodel,2),('png_grayscale_non169',root/'frame_%03d.png',a.onnx,0),
           ('large_frame',large,a.onnx,0)]
    rows=[]
    for name,source,model,expected in cases:
        run=root/name
        cmd=[sys.executable,'-m','inference.live','--source',str(source),'--profile','cpu','--onnx-path',str(model),'--out-dir',str(run),'--hood-polygon','none']
        try:
            proc=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=120)
            output=proc.stdout+'\n'+proc.stderr
            error='\n'.join(line for line in output.splitlines() if 'ERROR:' in line or 'failed:' in line)
            # Reports must never embed machine-specific paths.
            error=error.replace(str(root),'<audit-dir>').replace(str(Path(a.onnx).resolve()),'<onnx-model>')
            error=error.replace(os.environ.get('USERPROFILE','<none>'),'<user>')
            rows.append({'case':name,'exit_code':proc.returncode,'expected':expected,'passed':proc.returncode==expected and 'Traceback (most recent call last)' not in output,'error':error,'stack_trace':'none' if 'Traceback (most recent call last)' not in output else 'FAIL: traceback emitted'})
        except subprocess.TimeoutExpired:
            rows.append({'case':name,'passed':False,'error':'BLOCKED: subprocess timeout after 120 seconds'})
        print(rows[-1],flush=True)
    (root/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    return 0 if all(r['passed'] for r in rows) else 1


if __name__=='__main__':
    sys.exit(main())
