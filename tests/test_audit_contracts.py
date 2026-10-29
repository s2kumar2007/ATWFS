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
    monkeypatch.setattr('inference.live.cv2.VideoCapture',lambda _:Capture())
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
