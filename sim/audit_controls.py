"""Read-only kinematic replay of recorded controls for contact-consistent mock metrics.
No perception or decision code executes in this audit. Every pose, speed and contact
count must match the original run, or the audit fails rather than inventing values.
"""
from __future__ import annotations
import ast
import contextlib
import io
import math
from types import SimpleNamespace
from sim.carla_bridge import CarlaBridge, DT
from sim.carla_scenarios import Scenario
from sim.carla_metrics import finite, stopping_distance
from utils.config import load_config

def swept_aabb_ttc(dx,dy,vx,vy,hx,hy):
    enter=-math.inf;leave=math.inf
    for position,velocity,extent in ((dx,vx,hx),(dy,vy,hy)):
        if abs(velocity)<1e-12:
            if abs(position)>extent:return math.inf
        else:
            a=(-extent-position)/velocity;b=(extent-position)/velocity
            enter=max(enter,min(a,b));leave=min(leave,max(a,b))
    start=max(0.,enter)
    return start if leave>=start else math.inf

def read_controls(logfile):
    groups=[]
    for line in logfile.read_text(encoding='utf-8').splitlines():
        pos=line.find('tick {')
        if pos<0:continue
        row=ast.literal_eval(line[pos+5:])
        if row['tick']==0:groups.append([])
        if not all(k in row for k in ('brake','throttle','angle_deg')):
            raise ValueError('Control audit requires logged actuator commands, unavailable in this historical file.')
        groups[-1].append(row)
    return groups

def audit_mock_controls(records,logfile):
    groups=read_controls(logfile)
    if len(groups)!=len(records):raise ValueError('Audit run count mismatch.')
    for r,samples in zip(records,groups):
        if len(samples)!=r['frames']:raise ValueError('Audit frame count mismatch.')
        pipe=SimpleNamespace(cfg=load_config());scene=Scenario(r['scenario'],r['parameters'])
        min_distance=math.inf;min_ttc=math.inf;max_error=0.;max_speed_error=0.;first_contact=None
        with CarlaBridge(mock=True,pipeline=pipe,seed=r['parameters']['seed']) as b:
            with contextlib.redirect_stdout(io.StringIO()):b.setup();scene.start(b)
            for i,row in enumerate(samples):
                if row['tick']!=i:raise ValueError('Nonconsecutive audit ticks.')
                vel=b.vehicle.get_velocity();speed=math.sqrt(vel.x**2+vel.y**2+vel.z**2)
                scene.update(i*DT,speed)
                idx=b.world.tick();b.sensor_frame(b.frames,idx);b.sensor_frame(b.semantic,idx)
                loc=b.vehicle.get_location();vel=b.vehicle.get_velocity();speed=math.sqrt(vel.x**2+vel.y**2+vel.z**2)
                max_error=max(max_error,math.dist((loc.x,loc.y,loc.z),row['position']))
                max_speed_error=max(max_speed_error,abs(speed-row['speed_mps']))
                a=scene.actor;aloc=a.get_location();av=a.get_velocity();e=b.vehicle.bounding_box.extent;o=a.bounding_box.extent
                dx=aloc.x-loc.x;dy=aloc.y-loc.y;hx=e.x+o.x;hy=e.y+o.y
                min_distance=min(min_distance,math.hypot(max(0.,abs(dx)-hx),max(0.,abs(dy)-hy)))
                min_ttc=min(min_ttc,swept_aabb_ttc(dx,dy,av.x-vel.x,av.y-vel.y,hx,hy))
                if first_contact is None and any(event.other_actor.id==a.id for event in b.collisions):first_contact=i*DT
                b.vehicle.apply_control(b.carla.VehicleControl(steer=row['angle_deg']/b.max_steer,throttle=row['throttle'],brake=row['brake']))
            target_contacts=sum(event.other_actor.id==scene.actor.id for event in b.collisions)
            if len(b.collisions)!=r['collisions'] or target_contacts!=r['target_collisions']:
                raise ValueError('Audit contact count differs from original.')
        if max_error>1e-8 or max_speed_error>1e-8:raise ValueError(f'Audit trajectory mismatch: position {max_error}, speed {max_speed_error}.')
        r.update(audit_verified=True,audit_max_position_error_m=max_error,audit_max_speed_error_mps=max_speed_error,
                 audited_min_target_distance_m=finite(min_distance),audited_min_target_ttc_s=finite(min_ttc),
                 first_target_contact_s=first_contact)
        reference=r.get('hazard_reference_time_s')
        if reference is None:continue
        start=round(reference/DT);end=round(first_contact/DT) if first_contact is not None else len(samples)-1
        if end<start:raise ValueError('Contact predates the reaction reference; cannot apply standard event metric.')
        original_brake=next((sample['tick']*DT for sample in samples[start:] if sample['brake']>.1),None)
        expected_reaction=None if original_brake is None else original_brake-reference
        if (r['reaction_time_s'] is None)!=(expected_reaction is None) or (expected_reaction is not None and abs(expected_reaction-r['reaction_time_s'])>1e-8):
            raise ValueError('Logged brake reaction disagrees with stored metric.')
        interval=samples[start:end+1]
        brake=next((sample for sample in interval if sample['brake']>.1),None)
        stopped=next((sample for sample in interval if sample['speed_mps']<.1),None)
        reaction=None if brake is None else max(0.,brake['tick']*DT-reference)
        bound=max(0.,end*DT-reference)
        actual=None if stopped is None else math.dist(stopped['position'],samples[start]['position'])
        lower=math.dist(samples[end]['position'],samples[start]['position']) if stopped is None else None
        r.update(report_reaction_time_s=reaction,report_reaction_lower_bound_s=bound if brake is None else None,
                 report_stopping_distance_m=actual,report_stopping_lower_bound_m=lower,
                 report_required_stopping_m=None if reaction is None else stopping_distance(r['hazard_reference_speed_mps'],reaction,r['configured_deceleration_mps2']),
                 report_required_stopping_lower_bound_m=stopping_distance(r['hazard_reference_speed_mps'],bound,r['configured_deceleration_mps2']) if reaction is None else None,
                 report_censoring_reason='first target contact' if first_contact is not None else 'end of run',
                 report_brake_before_contact=brake is not None)
    return records
