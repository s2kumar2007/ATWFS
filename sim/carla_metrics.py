"""Measured simulation metrics; simulator truth must never enter the supervisor."""
from __future__ import annotations
import math
import cv2
import numpy as np
from sim.carla_scenarios import relative

def iou(mask,ground_truth):
    a=np.asarray(mask)>0;b=np.asarray(ground_truth)>0
    if a.shape!=b.shape:raise ValueError(f'IoU shape mismatch: {a.shape} vs {b.shape}')
    u=np.logical_or(a,b).sum()
    return float(np.logical_and(a,b).sum()/u) if u else 1.

def stopping_distance(v,t_react=.5,decel=4.):
    return max(0.,v)*max(0.,t_react)+max(0.,v)**2/(2*max(.01,decel))

def ttc(distance,closing_speed):return math.inf if closing_speed<=0 else max(0.,distance/closing_speed)

def finite(v):return float(v) if math.isfinite(v) else None

def truth_snapshot(bridge,scenario,semantic):
    ego=bridge.vehicle;pose=ego.get_transform();loc=pose.location;ev=ego.get_velocity();f=pose.get_forward_vector()
    minimum={'walker':math.inf,'vehicle':math.inf};min_ttc=math.inf;blocking=False;target_distance=math.inf;target_ttc=math.inf
    actors=bridge.world.get_actors();e=ego.bounding_box.extent
    for a in actors:
        if a.id==ego.id or not a.is_alive:continue
        kind='walker' if a.type_id.startswith('walker.') else 'vehicle' if a.type_id.startswith('vehicle.') else None
        if kind is None:continue
        along,side=relative(pose,a.get_location());ext=a.bounding_box.extent
        gap_x=max(0.,abs(along)-e.x-ext.x);gap_y=max(0.,abs(side)-e.y-ext.y)
        clearance=math.hypot(gap_x,gap_y)
        minimum[kind]=min(minimum[kind],clearance)
        is_target=scenario is not None and scenario.actor is not None and a.id==scenario.actor.id
        if is_target:target_distance=min(target_distance,clearance)
        if along>0 and abs(side)<e.y+ext.y+.25:
            av=a.get_velocity();closing=(ev.x-av.x)*f.x+(ev.y-av.y)*f.y
            actor_ttc=ttc(max(0.,along-e.x-ext.x),closing)
            min_ttc=min(min_ttc,actor_ttc)
            if is_target:target_ttc=min(target_ttc,actor_ttc)
            if along-e.x-ext.x<15:blocking=True
    wp=bridge.world.get_map().get_waypoint(loc,project_to_road=True)
    lane=None if wp is None else math.hypot(loc.x-wp.transform.location.x,loc.y-wp.transform.location.y)
    person_visible=bool(np.any(semantic==4)) if semantic is not None else False
    if bridge.mock and scenario is not None and scenario.name=='pedestrian_dash':
        person_visible=scenario.actor.id in bridge.world.last_visible_ids
    return dict(min_walker=minimum['walker'],min_vehicle=minimum['vehicle'],ttc=min_ttc,lane=lane,
                blocking=blocking,person_visible=person_visible,position=(loc.x,loc.y,loc.z),target_distance=target_distance,target_ttc=target_ttc,lead_braking_event=scenario is not None and scenario.name=='lead_brake' and scenario.trigger_time is not None)

class Metrics:
    def __init__(self,decel=4.):
        self.decel=decel;self.min_walker=math.inf;self.min_vehicle=math.inf;self.min_ttc=math.inf
        self.lanes=[];self.ious=[];self.latencies=[];self.first_visible=None;self.first_brake=None
        self.visible_position=None;self.visible_speed=None;self.stop_distance=None
        self.false_stops=0;self.eligible_ticks=0;self.override_ticks=0;self.override_reasons={};self.braking_at_visibility=False;self.reference_kind=None;self.last_time=None;self.observed_distance=None
        self.min_target_distance=math.inf;self.min_target_ttc=math.inf;self.false_stop_events=0;self.was_false_stop=False
    def update(self,t,truth,speed,brake,latency,reasons,mask=None,semantic=None,rule_stop=False):
        self.min_walker=min(self.min_walker,truth['min_walker']);self.min_vehicle=min(self.min_vehicle,truth['min_vehicle']);self.min_ttc=min(self.min_ttc,truth['ttc'])
        if truth['lane'] is not None:self.lanes.append(truth['lane'])
        if latency is not None:self.latencies.append(latency)
        self.last_time=t
        self.min_target_distance=min(self.min_target_distance,truth.get('target_distance',math.inf))
        self.min_target_ttc=min(self.min_target_ttc,truth.get('target_ttc',math.inf))
        if mask is not None and semantic is not None:self.ious.append(iou(mask,np.isin(semantic,[6,7])))
        if (truth['person_visible'] or truth.get('lead_braking_event',False)) and self.first_visible is None:
            self.reference_kind='pedestrian_first_visible' if truth['person_visible'] else 'lead_vehicle_brake_onset'
            self.first_visible=t;self.visible_position=truth['position'];self.visible_speed=speed;self.braking_at_visibility=brake>.1
        if self.first_visible is not None:
            self.observed_distance=math.dist(truth['position'],self.visible_position)
            if brake>.1 and self.first_brake is None:self.first_brake=t
            if speed<.1 and self.stop_distance is None:self.stop_distance=math.dist(truth['position'],self.visible_position)
        false_stop=t>=2 and not truth['blocking'] and not rule_stop and speed<.1
        if false_stop and not self.was_false_stop:self.false_stop_events+=1
        self.was_false_stop=false_stop
        if t>=2 and not truth['blocking'] and not rule_stop:
            self.eligible_ticks+=1
            if speed<.1:self.false_stops+=1
        if reasons:self.override_ticks+=1
        for reason in reasons:self.override_reasons[reason]=self.override_reasons.get(reason,0)+1
    def summary(self):
        reaction=None if self.first_visible is None or self.first_brake is None else max(0.,self.first_brake-self.first_visible)
        required=None if reaction is None else stopping_distance(self.visible_speed,reaction,self.decel)
        censored_reaction=None if reaction is not None or self.first_visible is None else max(0.,self.last_time-self.first_visible)
        n=len(self.latencies);total=sum(self.latencies)
        return dict(min_distance_target_m=finite(self.min_target_distance),min_target_ttc_s=finite(self.min_target_ttc),reaction_reference=self.reference_kind,hazard_reference_time_s=self.first_visible,hazard_reference_speed_mps=self.visible_speed,observed_distance_since_hazard_m=self.observed_distance,reaction_censored=censored_reaction is not None,reaction_lower_bound_s=censored_reaction,required_stopping_lower_bound_m=None if censored_reaction is None else stopping_distance(self.visible_speed,censored_reaction,self.decel),min_distance_walkers_m=finite(self.min_walker),min_distance_vehicles_m=finite(self.min_vehicle),min_ttc_s=finite(self.min_ttc),
            lane_deviation_rms_m=float(np.sqrt(np.mean(np.square(self.lanes)))) if self.lanes else None,
            drivable_iou_mean=float(np.mean(self.ious)) if self.ious else None,iou_samples=len(self.ious),
            reaction_time_s=reaction,pedestrian_first_visible_s=self.first_visible if self.reference_kind=='pedestrian_first_visible' else None,braking_at_first_visibility=self.braking_at_visibility,
            stopping_distance_m=self.stop_distance,required_stopping_distance_m=required,configured_deceleration_mps2=self.decel,
            stopping_margin_m=None if required is None or self.stop_distance is None else required-self.stop_distance,
            false_stop_ticks=self.false_stops,false_stop_events=self.false_stop_events,false_stop_rate=self.false_stops/self.eligible_ticks if self.eligible_ticks else None,false_stop_eligible_ticks=self.eligible_ticks,
            supervisor_override_ticks=self.override_ticks,supervisor_override_reasons=self.override_reasons,
            pipeline_fps=n/total if total else None,latency_mean_ms=total*1000/n if n else None,
            latency_p95_ms=float(np.percentile(self.latencies,95)*1000) if n else None,latency_samples_ms=[v*1000 for v in self.latencies])

class Recorder:
    def __init__(self,path):
        self.path=path;self.frames=0
        self.writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),20.,(848,478))
        if not self.writer.isOpened():raise RuntimeError(f'MP4 encoder unavailable: {path}. Install an OpenCV build with MP4 support.')
    def write(self,frame,result,run,t,speed,control,reasons,label):
        out=frame.copy();mask=result.mask>0
        out[mask]=(out[mask].astype(np.float32)*.65+np.array([190,210,30])*.35).astype(np.uint8)
        for d in result.detections:
            cv2.rectangle(out,(d.x1,d.y1),(d.x2,d.y2),(0,0,255) if d.kind=='person' else (255,220,0),2)
            cv2.putText(out,f'{d.name} {d.conf:.2f}',(d.x1,max(12,d.y1-3)),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
        lines=[label,f'run {run} | t {t:.2f}s | speed {speed:.2f} m/s | steer {control["steer"]:+.2f}',
               f'trust {result.command.trust:.3f} | throttle {control["throttle"]:.2f} brake {control["brake"]:.2f}',
               'supervisor: '+(', '.join(reasons) if reasons else 'clear')]
        cv2.rectangle(out,(0,0),(848,92),(0,0,0),-1)
        for i,line in enumerate(lines):cv2.putText(out,line[:115],(8,19+i*22),cv2.FONT_HERSHEY_SIMPLEX,.44,(255,255,255),1)
        self.writer.write(out);self.frames+=1
    def close(self):self.writer.release()
