"""Artifact and CLI integration checks using the real downstream ATWFS pipeline."""
import json
import cv2
import numpy as np
from sim.carla_metrics import Recorder
from sim.run_carla import main
from tests.test_carla_safety_scenarios import frame

def test_recorder_writes_readable_mp4(tmp_path):
    path=tmp_path/'test.mp4';recorder=Recorder(path)
    try:
        recorder.write(np.zeros((478,848,3),np.uint8),frame(),1,0.,0.,dict(steer=0.,throttle=0.,brake=1.),['E-stop'],'MOCK')
    finally:recorder.close()
    cap=cv2.VideoCapture(str(path));ok,image=cap.read();cap.release()
    assert ok and image.shape==(478,848,3)

def test_all_scenario_cli_saves_metrics_without_assets(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(['--mock','--scenario','all','--runs','1','--seconds','.2','--latency-mode','fixed:0','--no-video','--output-dir',str(tmp_path)])==0
    data=json.loads((tmp_path/'metrics.json').read_text())
    assert len(data['runs'])==5 and all(r['frames']==4 for r in data['runs'])
    assert data['metadata']['trained_perception'] is False
    assert all(r['iou_samples']==1 for r in data['runs'])
    assert len((tmp_path/'summary.csv').read_text().splitlines())==6
    assert 'LivePipeline.process_frame' in (tmp_path/'run_log.txt').read_text()

def test_missing_requested_model_is_clear_and_fatal(tmp_path,capsys):
    rc=main(['--mock','--onnx-path',str(tmp_path/'missing.onnx'),'--output-dir',str(tmp_path)])
    assert rc==1
    assert 'Model missing' in capsys.readouterr().out
    assert 'error' in json.loads((tmp_path/'metrics.json').read_text())['metadata']
