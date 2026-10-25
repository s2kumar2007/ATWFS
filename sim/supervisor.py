"""Pipeline-output-only safety clamp."""
from __future__ import annotations
import math

class SafetySupervisor:
 def __init__(self,ttc_threshold=1.5,low_trust=.25,speed_limit=6.,steer_rate_deg_s=45.):
  self.ttc_threshold=ttc_threshold;self.low_trust=low_trust;self.speed_limit=speed_limit;self.steer_rate=steer_rate_deg_s;self.estop=False;self.last_angle=0.;self.overrides=[]
 @staticmethod
 def ttc(distance_m,closing_speed_mps):return float('inf') if closing_speed_mps<=1e-6 else max(0.,distance_m/closing_speed_mps)
 def latch_estop(self):self.estop=True
 def reset(self):self.estop=False
 def apply(self,command,dt=.05,ttc=float('inf'),curvature=0.):
  reason=[];angle=float(command.angle);speed=float(command.speed)
  if self.estop:speed=0.;reason.append('latched E-stop')
  if ttc<self.ttc_threshold:speed=0.;reason.append('pipeline TTC emergency brake')
  if command.trust<self.low_trust:speed=min(speed,.5);angle=0.;reason.append('low trust slow-and-straight')
  lim=self.speed_limit/(1+abs(curvature)*5)
  if speed>lim:speed=lim;reason.append('curvature speed limit')
  cap=self.steer_rate*dt;clamped=max(self.last_angle-cap,min(self.last_angle+cap,angle))
  if clamped!=angle:reason.append('steering rate limit')
  self.last_angle=clamped
  if reason:self.overrides.extend(reason)
  return clamped,speed,reason
