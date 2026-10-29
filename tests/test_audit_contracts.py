from pathlib import Path


def test_runtime_dependencies():
    text = Path('requirements.txt').read_text()
    for name in ('onnxruntime', 'onnx', 'pyserial', 'pyflakes'):
        assert name in text.splitlines()

import copy
import numpy as np
import pytest
import torch
from utils.config import load_config, validate_config
from inference.live import LivePipeline, FrameGrabber
from inference.detector import NullDetector
from data.dataset import DrivableDataset


class CaptureONNX:
    def run(self, names, feeds):
        self.x = feeds['image'].copy()
        _, _, h, w = self.x.shape
        return [np.zeros((1, 2, h, w), np.float32), np.ones((1, 2, h, w), np.float32)*2]


@pytest.mark.parametrize('shape', [(72,128), (100,100), (90,160), (80,120)])
@pytest.mark.parametrize('letterbox', [False, True])
def test_actual_training_and_inference_preprocessing(shape, letterbox):
    cfg = load_config()
    cfg['data']['letterbox'] = letterbox
    cfg['model']['pretrained'] = False
    cfg['inference']['hood_polygon'] = None
    cfg['inference']['night_mode']['enabled'] = False
    rng = np.random.default_rng(3)
    rgb = rng.integers(0,256,(*shape,3),dtype=np.uint8)
    class Source:
        def load(self, i):
            return rgb, np.zeros(shape,np.uint8), -1
        def __len__(self):
            return 1
    expected = DrivableDataset(Source(), cfg)[0]['image'].numpy()
    session = CaptureONNX()
    pipe = LivePipeline(cfg, None, torch.device('cpu'), NullDetector(), onnx_session=session)
    pipe.process_frame(rgb[:,:,::-1].copy())
    np.testing.assert_allclose(session.x[0],expected,atol=2e-6)


def test_config_validation():
    cfg = load_config()
    validate_config(cfg)
    for section,key,value in [('inference','queue_size',0), ('fusion','fallback_threshold',float('nan')), ('data','image_size',[0,512])]:
        bad=copy.deepcopy(cfg)
        bad[section][key]=value
        with pytest.raises(ValueError):
            validate_config(bad)


@pytest.mark.parametrize('fps', [-1, 0, float('nan')])
def test_negative_capture_fps(monkeypatch, fps):
    import threading
    import queue
    class Capture:
        def isOpened(self): return True
        def get(self, prop): return fps if prop == 5 else 0
        def release(self): pass
    monkeypatch.setattr('inference.live._open_capture',lambda _:Capture())
    grabber=FrameGrabber('sequence_%03d.png',queue.Queue(),threading.Event(),False)
    assert grabber.fps == 25


def test_missing_explicit_onnx_does_not_use_other_weights(tmp_path):
    from inference.live import run_video
    cfg=load_config()
    cfg['inference']['_onnx_path_override']=str(tmp_path/'missing.onnx')
    (tmp_path/'input.mp4').write_bytes(b'placeholder')
    with pytest.raises(ValueError, match='Requested ONNX weights missing'):
        run_video(cfg,str(tmp_path/'input.mp4'),str(tmp_path/'out.mp4'),profile='cpu',hood_polygon='none',detector=NullDetector())


@pytest.mark.parametrize('frame', [None, np.zeros((0,0,3),np.uint8), np.zeros((20,20),np.uint8), np.full((20,20,3),np.nan)])
def test_invalid_frame_is_rejected(frame):
    pipe=LivePipeline(load_config(),None,torch.device('cpu'),NullDetector(),onnx_session=CaptureONNX())
    with pytest.raises(ValueError,match='Frame must'):
        pipe.process_frame(frame)


@pytest.mark.parametrize('dt', [0,-1,float('nan'),2])
def test_invalid_dt_is_rejected(dt):
    pipe=LivePipeline(load_config(),None,torch.device('cpu'),NullDetector(),onnx_session=CaptureONNX())
    with pytest.raises(ValueError,match='Frame dt'):
        pipe.process_frame(np.zeros((20,20,3),np.uint8),dt=dt)


def test_documented_readme_module_flags_exist():
    import ast
    import re
    readme=Path('README.md').read_text(encoding='utf-8')
    for line in readme.splitlines():
        match=re.search(r'python -m ([\w.]+)([^`]*)(?:`|$)',line)
        if not match: continue
        module,tail=match.groups()
        path=Path(*module.split('.')).with_suffix('.py')
        if not path.is_file(): continue
        tree=ast.parse(path.read_text(encoding='utf-8'))
        flags={arg.value for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument' for arg in node.args if isinstance(arg,ast.Constant) and isinstance(arg.value,str) and arg.value.startswith('--')}
        assert set(re.findall(r'--[a-z][a-z0-9-]*',tail)) <= flags,(module,tail)


def test_corrupt_torch_model_has_clear_error(tmp_path):
    from inference.live import run_video
    source=tmp_path/'input.mp4';source.write_bytes(b'placeholder')
    model=tmp_path/'corrupt.pt';model.write_bytes(b'bad checkpoint')
    with pytest.raises(ValueError,match='Cannot load PyTorch model'):
        run_video(load_config(),str(source),str(tmp_path/'out.mp4'),checkpoint=str(model),backend='torch',profile='cpu',hood_polygon='none',detector=NullDetector())


def test_invalid_hood_polygon_rejected(tmp_path):
    from inference.live import run_video
    source=tmp_path/'input.mp4';source.write_bytes(b'placeholder')
    polygon=tmp_path/'hood.json';polygon.write_text('[[0,0],[1,0],[2,1]]')
    with pytest.raises(ValueError,match='normalized finite'):
        run_video(load_config(),str(source),str(tmp_path/'out.mp4'),hood_polygon=str(polygon),detector=NullDetector())


def test_auto_export_is_under_run_directory(tmp_path,monkeypatch):
    from inference.live import run_video
    cfg=load_config()
    source=tmp_path/'input.mp4';source.write_bytes(b'placeholder')
    ck=tmp_path/'model.pt';ck.write_bytes(b'placeholder')
    out=tmp_path/'run';captured={}
    def export(checkpoint,dest,*args,**kwargs):
        captured['dest']=Path(dest)
        raise RuntimeError('stop after export-path probe')
    monkeypatch.setattr('scripts.export_onnx.export_to_onnx',export)
    with pytest.raises(ValueError,match='export-path probe'):
        run_video(cfg,str(source),str(out/'annotated.mp4'),checkpoint=str(ck),backend='onnx',profile='cpu',out_dir=out,hood_polygon='none',detector=NullDetector())
    assert captured['dest'].resolve().is_relative_to(out.resolve())


def test_cli_missing_input():
    import subprocess
    import sys
    result=subprocess.run([sys.executable,'-m','inference.live','--source','missing-audit-input.mp4'],capture_output=True,text=True)
    assert result.returncode==2
    assert 'Input video not found' in result.stderr
    assert 'Traceback' not in result.stderr
