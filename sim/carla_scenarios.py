"""Seeded traffic scenarios. Ground truth here controls actors and evaluates outcomes only."""
from __future__ import annotations
import math
import random

SCENARIOS=('lane_follow','pedestrian_dash','lead_brake','blocked_lane','cut_in')

def make_scenarios(seed=0,runs=1):
    r=random.Random(seed)
    return {name:[dict(start=0.,trigger_distance=round(r.uniform(12,19),2),speed=round(r.uniform(1.2,3.0),2),seed=r.randrange(2**31),clear_after=r.uniform(4,6),start_distance_m=r.uniform(28,34),traffic_distance_m=r.uniform(45,55)) for _ in range(runs)] for name in SCENARIOS}

def triggered(distance,trigger_distance):return distance<=trigger_distance

def offset(api, pose, forward=0., right=0., z=0.):
    f=pose.get_forward_vector();r=pose.get_right_vector();p=pose.location
    return api.Transform(api.Location(x=p.x+f.x*forward+r.x*right,y=p.y+f.y*forward+r.y*right,z=p.z+z),
                         api.Rotation(yaw=pose.rotation.yaw))

def relative(pose, location):
    f=pose.get_forward_vector();r=pose.get_right_vector();dx=location.x-pose.location.x;dy=location.y-pose.location.y
    return dx*f.x+dy*f.y,dx*r.x+dy*r.y

class Scenario:
    def __init__(self,name,params):
        if name not in SCENARIOS:raise ValueError(f'Unknown scenario: {name}')
        self.name=name;self.params=params;self.trigger_time=None;self.clear_time=None
        self.actor=None;self.actors=[];self.background=[];self.parked=None;self.stopped_before_hazard=False;self.resumed=False
    def spawn(self,bridge,kind,pose,blueprint=None):
        lib=bridge.world.get_blueprint_library();choices=lib.filter(blueprint or ('walker.pedestrian.*' if kind=='walker' else 'vehicle.*'))
        if not choices:raise RuntimeError(f'No {kind} blueprint in CARLA world.')
        bp=choices[0]
        if bp.has_attribute('is_invincible'):bp.set_attribute('is_invincible','false')
        a=bridge.spawn(bp,pose);self.actors.append(a);return a
    def start(self,bridge):
        self.bridge=bridge;self.api=bridge.carla;self.origin=bridge.vehicle.get_transform()
        p=self.origin;distance=self.params.get("start_distance_m",31.)
        if self.name=='lane_follow':
            for d in (self.params.get("traffic_distance_m",45.),70.):
                a=self.spawn(bridge,'vehicle',offset(self.api,p,d));a.set_autopilot(True,bridge.tm.get_port());self.background.append(a)
        elif self.name=='pedestrian_dash':
            self.parked=self.spawn(bridge,'vehicle',offset(self.api,p,distance,2.5),'vehicle.mercedes.sprinter');self.parked.apply_control(self.api.VehicleControl(brake=1.,hand_brake=True))
            self.actor=self.spawn(bridge,'walker',offset(self.api,p,distance+self.parked.bounding_box.extent.x+.6,2.5));self.actor.apply_control(self.api.WalkerControl(speed=0.))
        elif self.name=='lead_brake':
            self.actor=self.spawn(bridge,'vehicle',offset(self.api,p,distance-5.));self.actor.set_autopilot(True,bridge.tm.get_port())
        elif self.name=='blocked_lane':
            self.actor=self.spawn(bridge,'vehicle',offset(self.api,p,distance));self.actor.apply_control(self.api.VehicleControl(brake=1.,hand_brake=True))
        else:
            self.actor=self.spawn(bridge,'vehicle',offset(self.api,p,distance,3.5));self.actor.apply_control(self.api.VehicleControl(throttle=.25))
        if self.actor is None:self.trigger_time=0.
        bridge.log_event('scenario_start',name=self.name,parameters=self.params)
    def update(self,t,speed):
        if self.actor is None:return
        ego=self.bridge.vehicle;along,lateral=relative(ego.get_transform(),self.actor.get_location())
        if self.trigger_time is None and triggered(along,self.params['trigger_distance']):
            self.trigger_time=t;self.bridge.log_event('scenario_trigger',scenario=self.name,time=t,distance=along)
            if self.name=='pedestrian_dash':
                r=self.origin.get_right_vector()
                self.actor.apply_control(self.api.WalkerControl(direction=self.api.Vector3D(x=-r.x,y=-r.y),speed=self.params['speed']))
            elif self.name=='lead_brake':
                self.actor.set_autopilot(False,self.bridge.tm.get_port());self.actor.apply_control(self.api.VehicleControl(brake=1.))
            elif self.name=='cut_in':self.actor.apply_control(self.api.VehicleControl(throttle=.3,steer=-.3))
        if self.trigger_time is None:return
        _,lane_side=relative(self.origin,self.actor.get_location())
        hazard_ahead=along>0 and abs(lateral)<1.5
        if along>0 and self.clear_time is None and speed<.1:self.stopped_before_hazard=True
        if self.name=='pedestrian_dash' and lane_side< -3.:
            self.actor.apply_control(self.api.WalkerControl(speed=0.));self._clear(t)
        elif self.clear_time is None and self.name in ('lead_brake','blocked_lane') and t-self.trigger_time>self.params['clear_after']:
            if self.name=='blocked_lane':
                # Scripted clearing after the test stop, explicitly logged.
                self.actor.set_transform(offset(self.api,self.actor.get_transform(),right=5.))
            else:self.actor.set_autopilot(True,self.bridge.tm.get_port())
            self._clear(t)
        elif self.clear_time is None and self.name=='cut_in' and lane_side<.2:
            pose=self.actor.get_transform();pose.rotation.yaw=self.origin.rotation.yaw
            self.actor.set_transform(pose);self.actor.apply_control(self.api.VehicleControl(throttle=.4));self._clear(t)
        if self.clear_time is not None and t>self.clear_time+.5 and speed>1.:
            location=ego.get_location();wp=self.bridge.world.get_map().get_waypoint(location)
            if wp is not None and location.distance(wp.transform.location)<1.:self.resumed=True
    def _clear(self,t):
        if self.clear_time is None:
            self.clear_time=t;self.bridge.log_event('scenario_path_clear',time=t,scenario=self.name)
    def outcome(self,collisions):
        triggered_ok=self.trigger_time is not None
        functional=(triggered_ok and (self.name=='lane_follow' or self.clear_time is not None))
        if self.name in ('pedestrian_dash','lead_brake','blocked_lane'):functional= functional and self.stopped_before_hazard and self.resumed
        return dict(triggered=triggered_ok,trigger_time_s=self.trigger_time,path_clear_time_s=self.clear_time,
                    stopped_before_hazard=self.stopped_before_hazard,resumed=self.resumed,
                    functional_success=bool(functional and collisions==0),pass_no_collision=collisions==0)

class TrafficRuleStandIn:
    """Explicit exception: simulator signal/sign state; never claimed as perception."""
    def __init__(self):self.stop_start={};self.completed=set()
    def reason(self,bridge,t,speed):
        v=bridge.vehicle
        if v.is_at_traffic_light() and str(v.get_traffic_light_state()).lower().endswith(('red','yellow')):
            return 'CARLA traffic-light state stand-in'
        for a in bridge.world.get_actors():
            if not a.type_id.startswith('traffic.stop') or a.id in self.completed:continue
            along,side=relative(v.get_transform(),a.get_location())
            if 0<along<9 and abs(side)<5:
                if speed<.1:self.stop_start.setdefault(a.id,t)
                if a.id in self.stop_start and t-self.stop_start[a.id]>=2.:
                    self.completed.add(a.id);continue
                return 'CARLA stop-sign actor stand-in (2 s stop)'
        return None
