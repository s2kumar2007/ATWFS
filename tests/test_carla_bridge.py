from types import SimpleNamespace
import pytest
from utils.config import load_config
from sim.carla_bridge import CarlaBridge, command_to_control

class Pipeline:
    def __init__(self, fail=False): self.cfg=load_config(); self.fail=fail
    def process_frame(self,*args):
        if self.fail: raise RuntimeError('injected inference failure')
        return SimpleNamespace(command=SimpleNamespace(angle=0.,speed=1.5))

def test_scaled_speed_reaches_vehicle_cap():
    c,_=command_to_control(SimpleNamespace(angle=0.,speed=1.5),speed=5.,vmax=6.)
    assert c['throttle']>0 and c['brake']==0

def test_delayed_mock_moves_and_restores_settings():
    bridge=CarlaBridge(mock=True,pipeline=Pipeline(),latency_mode='fixed:100')
    with bridge:
        result=bridge.run(.5)
        assert result['commands']==8
        assert result['distance_m']>0
        assert bridge.vehicle_wheelbase==2.7
    assert all(not a.alive for a in bridge.actors)
    assert not bridge.world.get_settings().synchronous_mode
    assert bridge.world.get_settings().fixed_delta_seconds is None

def test_cleanup_after_inference_error():
    bridge=CarlaBridge(mock=True,pipeline=Pipeline(True))
    with pytest.raises(RuntimeError,match='injected'):
        with bridge: bridge.run(.5)
    assert all(not a.alive for a in bridge.actors)
    assert not bridge.world.get_settings().synchronous_mode
