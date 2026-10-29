"""Isolate test writes and use a fake model for tracker unit tests.

The real-detector integration runs are separate audit commands, not unit tests.
"""
from pathlib import Path
import uuid
from types import SimpleNamespace
import pytest


def pytest_configure(config):
    if config.option.basetemp is None:
        # Keep managed-Windows atomic writes inside the approved workspace.
        parent = Path(__file__).resolve().parents[1] / "outputs" / "audit" / "t"
        parent.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(parent / uuid.uuid4().hex[:8])


@pytest.fixture(scope='session',autouse=True)
def isolated_test_root(tmp_path_factory):
    import utils.config as config
    previous=config.PROJECT_ROOT
    config.PROJECT_ROOT=tmp_path_factory.mktemp('atwfs-test-workspace')
    yield
    config.PROJECT_ROOT=previous


@pytest.fixture(autouse=True)
def offline_tracker_model(monkeypatch,tmp_path):
    from inference.detector import YOLODetector
    instances=[]
    original_init=YOLODetector.__init__
    class Model:
        names={0:'person',2:'car'}
        def predict(self,*args,**kwargs):return [SimpleNamespace(boxes=None)]
    monkeypatch.setattr(YOLODetector,'_load_yolo_model',lambda *args,**kwargs:Model())
    def init(self,cfg,output_dir,*args,**kwargs):
        original_init(self,cfg,str(tmp_path/'detector'),*args,**kwargs)
        instances.append((self,self._worker_thread))
    monkeypatch.setattr(YOLODetector,'__init__',init)
    yield
    for detector,thread in instances:
        detector._stop_event.set()
        thread.join(timeout=2)
