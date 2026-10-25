"""Software safety clamp. Obstacle evidence comes only from FrameResult pixels/outputs.
Ground-plane distances and TTC are estimates, not simulator truth.
"""
from __future__ import annotations
import copy
import math
from dataclasses import dataclass

@dataclass
class Supervised:
    command: object
    emergency: bool
    reasons: list
    estimated_ttc_s: float

class SafetySupervisor:
    def __init__(self,ttc_threshold=1.5,low_trust=.25,speed_limit=6.,steer_rate_deg_s=45.,lateral_accel=2.):
        self.ttc_threshold=ttc_threshold;self.low_trust=low_trust;self.speed_limit=speed_limit
        self.steer_rate=steer_rate_deg_s;self.lateral_accel=lateral_accel
        self.estop=False;self.last_angle=0.;self.overrides=[];self.previous={};self.clock=0.;self.pedestrian_hold=False;self.last_person_seen=-math.inf;self.vehicle_hold=False;self.last_near_vehicle=-math.inf;self.people_seen_in_lane=set()
    @staticmethod
    def ttc(distance_m,closing_speed_mps):
        return math.inf if closing_speed_mps<=1e-6 else max(0.,distance_m/closing_speed_mps)
    def latch_estop(self):self.estop=True
    def reset(self):self.estop=False;self.pedestrian_hold=False;self.last_person_seen=-math.inf;self.vehicle_hold=False;self.last_near_vehicle=-math.inf;self.people_seen_in_lane=set()
    def apply(self,command,dt=.05,ttc=math.inf,curvature=0.):
        reason=[];angle=float(command.angle);speed=float(command.speed)
        if self.estop:speed=0.;reason.append('latched E-stop')
        if ttc<self.ttc_threshold:speed=0.;reason.append('pipeline TTC emergency brake')
        if command.trust<self.low_trust:speed=min(speed,.15);angle=0.;reason.append('low trust slow-and-straight')
        lim=min(self.speed_limit,math.sqrt(self.lateral_accel/max(1e-6,abs(curvature))))
        if speed>lim:speed=lim;reason.append('curvature speed limit')
        cap=self.steer_rate*dt;clamped=max(self.last_angle-cap,min(self.last_angle+cap,angle))
        if clamped!=angle:reason.append('steering rate limit')
        self.last_angle=clamped
        self.overrides.extend(reason)
        return clamped,speed,reason
    def estimate_ttc(self,result,speed_mps,dt):
        self.clock+=dt;h,w=result.mask.shape[:2];focal=w/(2*math.tan(math.radians(37.5)))
        best=math.inf;current={}
        for i,d in enumerate(result.detections):
            foot=float(d.y2)-h/2
            if foot<=2 or d.conf<.2:continue
            distance=max(0.,1.3*focal/foot-2.0)
            lateral=(d.center_x-w/2)*max(.1,distance+2.)/focal
            half_width=max(.2,(d.x2-d.x1)*max(.1,distance+2.)/(2*focal))
            if abs(lateral)>(3.5 if d.kind=='person' else 1.1+half_width):continue
            key=(d.kind,d.tracker_id if d.tracker_id is not None else i)
            if d.kind=='person':
                if abs(lateral)<1.2:self.people_seen_in_lane.add(key)
                if key in self.people_seen_in_lane and abs(lateral)>1.8:continue
                self.last_person_seen=self.clock
            elif distance<12.:self.last_near_vehicle=self.clock
            old=self.previous.get(key);closing=speed_mps
            if old:
                elapsed=self.clock-old[1]
                if elapsed>0:closing=max(0.,(old[0]-distance)/elapsed)
                # Conservative for crossing people, even without image expansion.
                if d.kind=='person':closing=max(closing,speed_mps)
            current[key]=(distance,self.clock)
            observed_ttc=self.ttc(distance,closing)
            if d.kind=='person' and observed_ttc<self.ttc_threshold:self.pedestrian_hold=True
            elif d.kind!='person' and observed_ttc<self.ttc_threshold:self.vehicle_hold=True
            best=min(best,observed_ttc)
        if self.clock-self.last_person_seen>.5:self.pedestrian_hold=False
        if self.clock-self.last_near_vehicle>.5:self.vehicle_hold=False
        self.previous=current
        return best
    def step(self,result,speed_mps,dt,wheelbase_m,pipeline_vmax,vmax_mps):
        ttc=self.estimate_ttc(result,speed_mps,dt)
        c=copy.copy(result.command)
        # Clamp in physical units, then map speed back to the pipeline's scaled units.
        c.speed=max(0.,min(1.,c.speed/max(1e-6,pipeline_vmax)))*vmax_mps
        curvature=math.tan(math.radians(c.angle))/max(.1,wheelbase_m)
        angle,speed,reasons=self.apply(c,dt,ttc,curvature)
        if result.mask.size and not result.mask[int(result.mask.shape[0]*.7):].any():
            speed=0.;reasons.append('no drivable pixels in near field');self.overrides.append(reasons[-1])
        if self.pedestrian_hold:
            speed=0.;reasons.append('pedestrian emergency hold until clear');self.overrides.append(reasons[-1])
        if self.vehicle_hold:
            speed=0.;reasons.append('vehicle emergency hold until path opens');self.overrides.append(reasons[-1])
        c.angle=angle;c.speed=speed/max(1e-6,vmax_mps)*pipeline_vmax
        return Supervised(c,self.estop or ttc<self.ttc_threshold or self.pedestrian_hold or self.vehicle_hold,reasons,ttc)
