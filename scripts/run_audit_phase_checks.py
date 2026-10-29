"""Run the three required audit checks and preserve exact exit codes per phase."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--phase',type=int,required=True)
    a=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    out=root/'outputs'/'audit'/f'phase_{a.phase}'
    out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUTF8='1')
    rows=[]
    commands=[('pyflakes',[sys.executable,'-m','pyflakes','.']),('pytest',[sys.executable,'-m','pytest','-q']),('smoke',[sys.executable,'run_all_smoke_tests.py'])]
    for name,command in commands:
        start=time.perf_counter()
        try:
            with (out/f'{name}.log').open('w',encoding='utf-8') as log:
                proc=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
            text=(out/f'{name}.log').read_text(encoding='utf-8')
            status='PASS' if proc.returncode==0 else 'BLOCKED' if name=='pyflakes' and 'No module named pyflakes' in text else 'FAIL'
            row={'phase':a.phase,'check':name,'exit_code':proc.returncode,'status':status,'seconds':time.perf_counter()-start}
        except subprocess.TimeoutExpired:
            row={'phase':a.phase,'check':name,'status':'BLOCKED','reason':'900 second timeout'}
        rows.append(row)
        print(json.dumps(row),flush=True)
    (out/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    return int(any(row['status']=='FAIL' for row in rows))


if __name__=='__main__':sys.exit(main())
