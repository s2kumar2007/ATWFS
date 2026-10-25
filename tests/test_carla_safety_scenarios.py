"""Safety and scenario evidence tests; no CARLA package, GPU, training or downloads."""
from types import SimpleNamespace
import numpy as np
import pytest
from inference.decision import Command
from inference.detector import Detection
from sim.supervisor import SafetySupervisor
from sim.carla_scenarios import Scenario, SCENARIOS, make_scenarios
from sim.carla_bridge import CarlaBridge, latency_ticks, wheel_geometry
from sim.carla_metrics import Metrics, iou, stopping_distance
from sim.mock_carla import Transform, Vector
from tests.test_carla_bridge import Pipeline

def frame(detections=(),trust=1.):
    return SimpleNamespace(command=Command(0.,1.5,trust,False,1.,424.,0.),
                           detections=list(detections),mask=np.ones((478,848),np.uint8))

def test_ttc_pedestrian_emergency_uses_only_frame_outputs():
    result=frame([Detection(400,280,450,465,0,'person',.9,kind='person')])
    s=SafetySupervisor();c=s.step(result,6.,.05,2.7,1.5,6.)
    assert c.estimated_ttc_s<1.5 and c.emergency and c.command.speed==0
    s.latch_estop()
    for _ in range(3):assert s.step(frame(),6.,.05,2.7,1.5,6.).emergency
    s.reset();assert not s.step(frame(),0.,.05,2.7,1.5,6.).emergency

def test_steering_rate_curvature_and_low_trust():
    s=SafetySupervisor();r=frame();r.command.angle=30
    c=s.step(r,4.,.05,2.7,1.5,6.)
    assert c.command.angle<=2.25 and c.command.speed<1.5
    low=s.step(frame(trust=.01),4.,.05,2.7,1.5,6.)
    assert low.command.speed<=.15/6*1.5 and 'low trust slow-and-straight' in low.reasons

def test_latency_mode_validation_and_rounding():
    assert latency_ticks('real',.051)==2
    assert latency_ticks('fixed:100',999)==2
    assert latency_ticks('fixed:0',1)==0
    for mode in ('fixed:-1','fixed:nan','fixed:inf','anything'):
        with pytest.raises(ValueError):latency_ticks(mode,0)

@pytest.mark.parametrize('name',SCENARIOS)
def test_seeded_scenario_triggers_and_owns_all_actors(name):
    params=make_scenarios(8,1)[name][0]
    with CarlaBridge(mock=True,pipeline=Pipeline()) as b:
        b.setup();s=Scenario(name,params);s.start(b)
        assert set(a.id for a in s.actors)<=set(a.id for a in b.actors)
        if s.actor:
            b.vehicle.set_transform(Transform(Vector(s.actor.get_location().x-params['trigger_distance']+.1,0,.5)))
        s.update(1.,2.)
        assert s.trigger_time is not None
        if name=='pedestrian_dash':assert s.actor.walk_control.speed==params['speed']
        if name=='lead_brake':assert s.actor.control.brake==1.
    assert all(not a.is_alive for a in b.actors)

def test_mock_semantics_and_collision_events():
    with CarlaBridge(mock=True,pipeline=Pipeline()) as b:
        b.setup();lib=b.world.get_blueprint_library()
        b.spawn(lib.filter('walker.pedestrian.*')[0],Transform(Vector(1.,0,.5)))
        idx=b.world.tick();sem=b.sensor_frame(b.semantic,idx)
        assert np.any(sem==7)
        assert len(b.collisions)==1
        b.world.tick();assert len(b.collisions)==1

def test_measured_reaction_and_stopping_metrics():
    m=Metrics(4.)
    for t,speed,brake,x in [(2.,4.,0.,0.),(2.3,3.,.2,1.),(3.,0.,1.,4.)]:
        truth=dict(min_walker=2.,min_vehicle=float('inf'),ttc=1.,lane=.5,
                   blocking=True,person_visible=True,position=(x,0.,0.))
        m.update(t,truth,speed,brake,.05,[],np.ones((2,2)),np.full((2,2),7))
    r=m.summary()
    assert r['reaction_time_s']==pytest.approx(.3)
    assert r['required_stopping_distance_m']==pytest.approx(3.2)
    assert r['stopping_distance_m']==4.
    assert r['lane_deviation_rms_m']==.5 and r['drivable_iou_mean']==1.
    assert r['min_distance_vehicles_m'] is None

def test_iou_and_stopping_formula():
    assert iou(np.array([[1,0]]),np.array([[1,1]]))==.5
    assert stopping_distance(6.,.5,4.)==7.5


def test_pedestrian_starts_occluded_without_overlapping_parked_vehicle():
    with CarlaBridge(mock=True,pipeline=Pipeline()) as b:
        b.setup();s=Scenario('pedestrian_dash',make_scenarios(0,1)['pedestrian_dash'][0]);s.start(b)
        b.world.render()
        assert s.actor.id not in b.world.last_visible_ids
        assert abs(s.actor.get_location().x-s.parked.get_location().x)>s.actor.bounding_box.extent.x+s.parked.bounding_box.extent.x


def test_pedestrian_brake_is_held_while_person_stays_visible():
    result=frame([Detection(400,280,450,465,0,'person',.9,kind='person')])
    s=SafetySupervisor();assert s.step(result,6.,.05,2.7,1.5,6.).emergency
    # Dropping ego speed raises TTC; a visible pedestrian must not release the brake.
    assert s.step(result,0.,.05,2.7,1.5,6.).emergency
    for _ in range(12):last=s.step(frame(),0.,.05,2.7,1.5,6.)
    assert not last.emergency


def test_modern_semantic_tags_normalize_without_confusing_poles_and_lights():
    with CarlaBridge(mock=True,pipeline=Pipeline()) as b:
        b.modern_semantics=True
        tags=np.zeros((1,5,4),np.uint8);tags[0,:,2]=[1,24,12,6,7]
        b._frame(SimpleNamespace(raw_data=tags.tobytes(),height=1,width=5,frame=1),True)
        _,canonical=b.semantic.get_nowait()
        np.testing.assert_array_equal(canonical,[[7,6,4,0,0]])


def test_vehicle_brake_holds_as_ego_stops_until_blocker_clears():
    result=frame([Detection(390,280,455,465,2,'car',.9,kind='vehicle')])
    s=SafetySupervisor();assert s.step(result,6.,.05,2.7,1.5,6.).emergency
    assert s.step(result,0.,.05,2.7,1.5,6.).emergency
    for _ in range(12):last=s.step(frame(),0.,.05,2.7,1.5,6.)
    assert not last.emergency


def test_crossed_pedestrian_on_far_side_does_not_deadlock_brake():
    centre=frame([Detection(400,280,450,465,0,'person',.9,kind='person',tracker_id=1)])
    far_side=frame([Detection(40,280,90,465,0,'person',.9,kind='person',tracker_id=1)])
    s=SafetySupervisor();assert s.step(centre,6.,.05,2.7,1.5,6.).emergency
    for _ in range(12):last=s.step(far_side,0.,.05,2.7,1.5,6.)
    assert not last.emergency
