"""Safety regressions with synthetic inputs and in-memory serial only."""
import queue
import threading
import numpy as np
import pytest
from inference.decision import compute_command
from inference.detector import Detection
from inference.serial_out import SerialSender
from inference.live import _queue_put
from utils.config import load_config
from tests.test_serial_out import MockSerialPort


def test_latched_estop_watchdog_and_shutdown():
    port=MockSerialPort()
    sender=SerialSender(serial_conn=port)
    sender.send(30,1,1,False)
    sender.check_watchdog(sender._last_send+0.301)
    assert sender.send(-30,1,1,False) == '[+0.00, 0.000, 0.000, 1]\n'
    sender.close()
    assert port.written[-1] == b'[+0.00, 0.000, 0.000, 1]\n'
    assert port.closed


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite_packet_latches_stop(value):
    with SerialSender(dry_run=True) as sender:
        assert sender.send(value,1,1,0) == '[+0.00, 0.000, 0.000, 1]\n'
        assert sender.send(1,1,1,0) == '[+0.00, 0.000, 0.000, 1]\n'


def test_final_steering_rate_limit_after_sign_correction():
    cfg=load_config()['inference']['decision']
    mask=np.zeros((100,200),np.uint8)
    mask[40:,150:195]=1
    previous=-30.0
    for _ in range(15):
        cmd=compute_command(mask,[],1,0,cfg,prev_angle=previous)
        assert abs(cmd.angle-previous) <= cfg['max_steer_rate_deg']+1e-6
        previous=cmd.angle


def test_invalid_decision_stops_and_obstacle_brakes():
    cfg=load_config()['inference']['decision']
    mask=np.ones((100,200),np.uint8)
    assert compute_command(mask,[],float('nan'),0,cfg).speed == 0
    near=[Detection(80,50,120,99,0,'person',.9)]
    cmd=compute_command(mask,near,1,0,cfg,prev_speed=1)
    assert cmd.obstacle_flag and cmd.speed < 1
    # Serial safety override is immediate, regardless of pipeline speed ramp.
    with SerialSender(dry_run=True) as sender:
        assert '0.000' in sender.send(cmd.angle,cmd.speed,cmd.trust,cmd.obstacle_flag)
    parked=[Detection(0,40,15,90,2,'car',.9)]
    assert not compute_command(mask,parked,1,0,cfg).obstacle_flag


def test_shutdown_cancels_full_queue():
    q=queue.Queue(maxsize=1)
    q.put('full')
    stop=threading.Event()
    stop.set()
    assert not _queue_put(q,None,stop)


def test_firmware_units_and_timeout():
    from pathlib import Path
    text=Path('arduino/safety_clamp/safety_clamp.ino').read_text()
    assert 'reqSpeed / MAX_SPEED_MPS' in text
    assert 'WATCHDOG_MS = 300UL' in text
    assert '!isfinite(values[i])' in text


def test_output_containment(tmp_path):
    from inference.live import run_video
    cfg=load_config()
    with pytest.raises(ValueError,match='under --out-dir'):
        run_video(cfg,'absent.mp4',str(tmp_path/'escape.mp4'),out_dir=tmp_path/'run')


def test_capture_error_is_visible(monkeypatch):
    from inference.live import FrameGrabber
    class Capture:
        def isOpened(self): return True
        def get(self, prop): return 25 if prop == 5 else 10
        def read(self): raise MemoryError('injected memory pressure')
        def release(self): self.released=True
    monkeypatch.setattr('inference.live.cv2.VideoCapture',lambda _:Capture())
    errors=[]
    stop=threading.Event()
    grabber=FrameGrabber('fake',queue.Queue(),stop,False,errors=errors)
    grabber.run()
    assert stop.is_set() and isinstance(errors[0],MemoryError)
    assert grabber.cap.released


def test_detector_thread_failure():
    from inference.detector import ThreadedDetector, ObstacleDetector
    class BadDetector(ObstacleDetector):
        def detect(self, frame, frame_idx=0): raise RuntimeError('injected detector failure')
    detector=ThreadedDetector(BadDetector(),1)
    detector.detect(np.zeros((20,20,3),np.uint8))
    detector.thread.join(timeout=2)
    try:
        with pytest.raises(RuntimeError,match='Detector thread failed'):
            detector.detect(np.zeros((20,20,3),np.uint8))
    finally:
        detector.stop()


def test_low_visibility_counter_reaches_decay():
    from inference.live import LivePipeline
    from tests.test_audit_contracts import CaptureONNX
    import torch
    cfg=load_config()
    pipe=LivePipeline(cfg,None,torch.device('cpu'),__import__('inference.detector',fromlist=['NullDetector']).NullDetector(),onnx_session=CaptureONNX())
    pipe.prev_steer_angle=20
    counts=[]
    for i in range(20):
        pipe.process_frame(np.zeros((72,128,3),np.uint8),frame_idx=i)
        counts.append(pipe.steer_hold_counter)
    assert max(counts)>5
    assert abs(pipe.prev_steer_angle)<1


def test_low_trust_nonempty_mask_decays():
    cfg=load_config()['inference']['decision']
    previous=20
    for count in range(20):
        cmd=compute_command(np.ones((100,200),np.uint8),[],.1,0,cfg,prev_angle=previous,hold_count=count)
        previous=cmd.angle
    assert abs(previous)<.1
    assert cmd.speed <= cfg['crawl_speed_floor']


def test_hud_trust_contract():
    from inference.live import LivePipeline
    from inference.detector import NullDetector
    from tests.test_audit_contracts import CaptureONNX
    import torch
    cfg=load_config()
    pipe=LivePipeline(cfg,None,torch.device('cpu'),NullDetector(),onnx_session=CaptureONNX())
    for i in range(12):
        result=pipe.process_frame(np.zeros((72,128,3),np.uint8),frame_idx=i)
        assert (result.active=='FALLBACK') == (result.command.trust<cfg['fusion']['fallback_threshold'])


def test_grayscale_png_sequence_is_bgr(tmp_path):
    import cv2
    from inference.live import FrameGrabber
    for i in range(3):
        assert cv2.imwrite(str(tmp_path/f'frame_{i:03d}.png'),np.full((20,30),100,np.uint8))
    q=queue.Queue()
    errors=[]
    grabber=FrameGrabber(str(tmp_path/'frame_%03d.png'),q,threading.Event(),False,errors=errors)
    grabber.run()
    assert not errors
    assert q.qsize()==4
    assert q.get()[1].shape==(20,30,3)


def test_camera_size_change_stops_capture(monkeypatch):
    from inference.live import FrameGrabber
    class Capture:
        def __init__(self): self.n=0
        def isOpened(self): return True
        def get(self,prop): return 25 if prop==5 else 2
        def read(self):
            self.n+=1
            return True,np.zeros((20+self.n,30,3),np.uint8)
        def release(self): pass
    monkeypatch.setattr('inference.live.cv2.VideoCapture',lambda _:Capture())
    errors=[];stop=threading.Event()
    grabber=FrameGrabber('fake',queue.Queue(),stop,False,errors=errors)
    grabber.run()
    assert stop.is_set() and 'size changed' in str(errors[0])


def test_missing_serial_and_write_failure(monkeypatch):
    import sys
    import types
    def fail(**kwargs): raise OSError('injected missing port')
    monkeypatch.setitem(sys.modules,'serial',types.SimpleNamespace(Serial=fail))
    with pytest.raises(OSError,match='Requested serial port unavailable'):
        SerialSender(port='missing-test-port')
    class BadPort(MockSerialPort):
        def write(self,data): raise OSError('injected write failure')
    sender=SerialSender(serial_conn=BadPort())
    try:
        with pytest.raises(OSError,match='Serial write failed'):
            sender.send(0,1,1,0)
        assert sender._estop
    finally:
        sender.close()


def test_threaded_detector_stale_result_expires():
    from inference.detector import ThreadedDetector, NullDetector
    detector=ThreadedDetector(NullDetector(),3)
    try:
        with detector.lock:
            detector.latest=[Detection(1,1,10,10,2,'car',.9)]
            detector.latest_frame=0
        assert detector.detect(np.zeros((20,20,3),np.uint8),frame_idx=8)==[]
    finally:detector.stop()


def test_imminent_new_track_brakes_without_confirmation():
    cfg=load_config()['inference']['decision']
    det=Detection(80,50,120,99,0,'person',.9,tracker_id=99,consecutive_detections=1)
    cmd=compute_command(np.ones((100,200),np.uint8),[det],1,0,cfg)
    assert cmd.obstacle_flag
    assert cmd.speed==0


@pytest.mark.parametrize('bad',[0.,-1.,float('nan'),float('inf')])
def test_invalid_dirichlet_concentrations_are_rejected(bad):
    import torch
    from models.unet_mobilenet import evidential_uncertainty,evidential_probs
    from training.losses import evidential_loss
    alpha=torch.full((1,2,3,3),bad)
    for function in (evidential_uncertainty,evidential_probs):
        with pytest.raises(ValueError,match='Dirichlet concentrations'):
            function(alpha)
    with pytest.raises(ValueError,match='Dirichlet concentrations'):
        evidential_loss(alpha,torch.zeros((1,3,3),dtype=torch.long),.5)


def test_worker_forwards_source_dt_and_speed():
    from types import SimpleNamespace
    from inference.live import InferenceWorker
    from inference.decision import Command
    class Pipeline:
        timings={'read':0}
        stage_samples={'read':[]}
        def process_frame(self,frame,**kwargs):
            self.kwargs=kwargs
            return SimpleNamespace(command=Command(0,0,1,False,0,0,0))
    p=Pipeline();inq=queue.Queue();outq=queue.Queue()
    inq.put((0,np.zeros((20,30,3),np.uint8),0));inq.put(None)
    worker=InferenceWorker(p,inq,outq,threading.Event(),1,[],1,fps=20,speed_mps=2.5)
    worker.run()
    assert p.kwargs['dt']==.05 and p.kwargs['speed']==2.5


def test_estop_stops_speed_without_steering_jump():
    import ast
    with SerialSender(dry_run=True) as sender:
        for _ in range(6):sender.send(30,1,1,0)
        previous=sender._angle
        packet=ast.literal_eval(sender.emergency_stop('test'))
        assert packet[1]==0 and abs(packet[0]-previous)<=5


def test_pipeline_starts_with_ramp_state():
    from inference.live import LivePipeline
    from inference.detector import NullDetector
    from tests.test_audit_contracts import CaptureONNX
    import torch
    pipe=LivePipeline(load_config(),None,torch.device('cpu'),NullDetector(),onnx_session=CaptureONNX())
    assert pipe.prev_speed==0 and pipe.prev_steer_angle==0


def test_watchdog_has_margin_before_three_hundred_ms_budget():
    with SerialSender(dry_run=True) as sender:
        sender.send(0,1,1,0)
        sender.check_watchdog(sender._last_send+.251)
        assert sender._estop and ', 0.000,' in sender.history[-1]
