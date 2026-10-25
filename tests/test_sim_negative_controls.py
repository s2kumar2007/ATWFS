"""Negative-control instrumentation must not modify the decision algorithm."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
from sim.faults import MissedDetections, in_dropout
from sim.carla_bridge import CarlaBridge
from sim.carla_metrics import Metrics
from sim.reporting import full_summary
from tests.test_carla_bridge import Pipeline

class Detector:
    def detect(self,*args):return list(range(1000))
    def stop(self):pass

def test_misses_are_seeded_and_fraction_is_measured():
    a=MissedDetections(Detector(),.6,3);b=MissedDetections(Detector(),.6,3)
    assert a.detect(None)==b.detect(None)
    assert .55<a.stats()['observed_miss_rate']<.65
    clean=MissedDetections(Detector(),0,0);assert len(clean.detect(None))==1000
    assert clean.stats()['detector_boxes_missed']==0

def test_three_second_dropout_is_exactly_sixty_ticks():
    for start in (.1,3.95,5.85):assert sum(in_dropout(i*.05,start,3.) for i in range(300))==60
    assert not in_dropout(10,None,3)

def test_supervisor_can_be_disabled_without_touching_pipeline():
    with CarlaBridge(mock=True,pipeline=Pipeline(),supervisor_enabled=False,latency_mode='fixed:0') as b:
        def forbidden(*args):raise AssertionError('Supervisor was not disabled')
        b.supervisor.step=forbidden
        result=b.run(.2)
        assert result['distance_m']>0 and result['supervisor_override_ticks']==0

def test_dropout_keeps_plant_running_but_does_not_invent_inference_samples():
    class Scenario:
        actor=None;name='lead_brake';trigger_time=.1
        def start(self,b):pass
        def update(self,*args):pass
        def outcome(self,*args):return {}
    with CarlaBridge(mock=True,pipeline=Pipeline(),sensor_dropout_seconds=3.,latency_mode='fixed:0') as b:
        r=b.run(3.5,Scenario())
        assert r['frames']==70 and r['processed_frames']==10 and r['sensor_dropout_ticks']==60
        assert len(r['latency_samples_ms'])==10 and r['distance_m']>0

def test_lead_brake_reaction_and_censoring():
    m=Metrics(4.)
    for t,brake,x in [(2.,0.,0.),(2.3,.2,1.),(3.,1.,4.)]:
        truth=dict(min_walker=float('inf'),min_vehicle=3.,ttc=1.,lane=0.,blocking=True,person_visible=False,lead_braking_event=True,position=(x,0,0))
        m.update(t,truth,0. if t==3. else 4.,brake,.1,[])
    r=m.summary();assert r['reaction_reference']=='lead_vehicle_brake_onset'
    assert r['reaction_time_s']==pytest.approx(.3) and r['required_stopping_distance_m']==pytest.approx(3.2)
    assert r['pedestrian_first_visible_s'] is None

def test_full_summary_uses_pooled_p95_not_mean_of_percentiles():
    base=dict(scenario='pedestrian_dash',collisions=0,frames=2,false_stop_eligible_ticks=2,false_stop_rate=.5,
              supervisor_override_reasons={},latency_samples_ms=[1.,2.])
    second=dict(base,latency_samples_ms=[10.,100.])
    row=full_summary([base,second],2)[0]
    assert row['latency_p95_ms']==pytest.approx(np.percentile([1,2,10,100],95))
    assert row['false_stop_ticks']==2 and row['false_stop_rate']==.5


@pytest.mark.parametrize('args,expected', [
    ((10.,0.,-2.,0.,2.,1.),4.),
    ((10.,5.,-2.,0.,2.,1.),float('inf')),
    ((0.,5.,0.,-2.,2.,1.),2.),
    ((0.,0.,0.,0.,2.,1.),0.),
    ((10.,0.,2.,0.,2.,1.),float('inf')),
])
def test_contact_ttc_handles_lateral_misses_and_existing_overlap(args,expected):
    from sim.audit_controls import swept_aabb_ttc
    assert swept_aabb_ttc(*args)==expected


def test_table_uses_contact_censored_audit_and_preserves_raw_values():
    from scripts.run_sim_negative_controls import table_row
    r=dict(scenario='lead_brake',condition='WORST',run=1,collisions=1,
           min_distance_target_m=2.,min_target_ttc_s=3.,reaction_time_s=4.,
           stopping_distance_m=50.,required_stopping_distance_m=60.,
           supervisor_override_ticks=0,audit_verified=True,
           audited_min_target_distance_m=0.,audited_min_target_ttc_s=0.,
           report_reaction_time_s=None,report_reaction_lower_bound_s=.5,
           report_stopping_distance_m=None,report_stopping_lower_bound_m=2.,
           report_required_stopping_m=None,report_required_stopping_lower_bound_m=3.)
    before=dict(r);row=table_row(r)
    assert '| 0.00 | 0.00 | not stopped; >=2.00 / >=3.00 | >0.50 |' in row
    assert r==before


def test_replay_validates_trajectory_and_rejects_corrupted_log(tmp_path,caplog):
    import copy
    import logging
    from sim.audit_controls import audit_mock_controls
    from sim.carla_scenarios import Scenario,make_scenarios
    caplog.set_level(logging.INFO,logger='sim.carla_bridge')
    params=make_scenarios(0,1)['lead_brake'][0]
    with CarlaBridge(mock=True,pipeline=Pipeline(),seed=params['seed'],latency_mode='fixed:0') as b:
        r=b.run(.5,Scenario('lead_brake',params))
    r.update(scenario='lead_brake',parameters=params)
    logfile=tmp_path/'controls.txt';logfile.write_text(caplog.text,encoding='utf-8')
    audited=audit_mock_controls([copy.deepcopy(r)],logfile)[0]
    assert audited['audit_verified'] and audited['audit_max_position_error_m']==0.
    log=logfile.read_text();logfile.write_text(log.replace("'speed_mps': 0.0", "'speed_mps': 100.0",1))
    with pytest.raises(ValueError,match='trajectory mismatch'):
        audit_mock_controls([copy.deepcopy(r)],logfile)
