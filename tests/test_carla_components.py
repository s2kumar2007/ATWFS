import numpy as np
from types import SimpleNamespace
from sim.carla_bridge import command_to_control
from sim.supervisor import SafetySupervisor
from sim.carla_scenarios import make_scenarios,triggered

def test_control_clips_angle():
 c,_=command_to_control(SimpleNamespace(angle=100.,speed=4.),max_steer_deg=30.,vmax=6.)
 assert c['steer']==1.
def test_supervisor_latches_and_ttc():
 s=SafetySupervisor();s.latch_estop();assert s.apply(SimpleNamespace(angle=0.,speed=4.,trust=1.),ttc=2)[1]==0.;s.reset();assert s.ttc(5,2)==2.5
def test_scenario_seed_reproducible():
 assert make_scenarios(10,3)==make_scenarios(10,3); assert triggered(2,3)
