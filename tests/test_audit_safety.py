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
