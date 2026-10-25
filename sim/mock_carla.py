"""Deterministic CARLA-shaped CPU world. Rendering is synthetic, never real-car evidence."""
from __future__ import annotations
import copy
import math
from dataclasses import dataclass, field
from types import SimpleNamespace
import cv2
import numpy as np

@dataclass
class Vector:
    x: float = 0.
    y: float = 0.
    z: float = 0.
    def distance(self, other): return math.dist((self.x,self.y,self.z),(other.x,other.y,other.z))
    def __add__(self, other): return Vector(self.x+other.x,self.y+other.y,self.z+other.z)
    def __mul__(self, value): return Vector(self.x*value,self.y*value,self.z*value)

@dataclass
class Rotation:
    pitch: float = 0.
    yaw: float = 0.
    roll: float = 0.

@dataclass
class Transform:
    location: Vector = field(default_factory=Vector)
    rotation: Rotation = field(default_factory=Rotation)
    def get_forward_vector(self):
        a=math.radians(self.rotation.yaw); return Vector(math.cos(a),math.sin(a),0)
    def get_right_vector(self):
        a=math.radians(self.rotation.yaw); return Vector(-math.sin(a),math.cos(a),0)

@dataclass
class MockControl:
    steer: float=0.
    throttle: float=0.
    brake: float=0.
    hand_brake: bool=False

@dataclass
class WalkerControl:
    direction: Vector=field(default_factory=Vector)
    speed: float=0.

@dataclass
class MockSettings:
    synchronous_mode: bool=False
    fixed_delta_seconds: float|None=None

class Blueprint:
    def __init__(self, name): self.id=name; self.attrs={}
    def set_attribute(self,k,v): self.attrs[k]=v
    def has_attribute(self,k): return False

class Library:
    def find(self,n): return Blueprint(n)
    def filter(self,n): return [Blueprint(n.replace('*','mock'))]

class Actor:
    def __init__(self,world,blueprint,transform,parent=None):
        self.world=world; self.id=len(world.actors)+1; self.type_id=blueprint.id
        self.pose=copy.deepcopy(transform); self.parent=parent; self.callback=None
        self.alive=True; self.is_alive=True; self.speed=0.; self.control=MockControl()
        self.autopilot=False; self.walk_control=WalkerControl(); self.physics=True
        self.bounding_box=SimpleNamespace(extent=Vector(.3,.3,.9) if 'walker' in self.type_id else Vector(2.,.9,1.3 if 'sprinter' in self.type_id else .8))
        self.hidden=False
    def destroy(self): self.alive=False; self.is_alive=False; return True
    def listen(self,callback): self.callback=callback
    def stop(self): self.callback=None
    def get_location(self): return copy.deepcopy(self.pose.location)
    def get_transform(self): return copy.deepcopy(self.pose)
    def set_transform(self,t): self.pose=copy.deepcopy(t)
    def get_velocity(self):
        if 'walker' in self.type_id:return self.walk_control.direction*self.walk_control.speed
        return self.pose.get_forward_vector()*self.speed
    def get_physics_control(self):
        return SimpleNamespace(wheels=[SimpleNamespace(max_steer_angle=35. if i<2 else 0.,position=Vector(x,y,0)) for i,(x,y) in enumerate(((135,-75),(135,75),(-135,-75),(-135,75)))])
    def apply_control(self,c):
        if isinstance(c,WalkerControl): self.walk_control=c
        else:self.control=c
    def set_autopilot(self,value,*args): self.autopilot=value
    def set_simulate_physics(self,value): self.physics=value
    def is_at_traffic_light(self): return False
    def get_traffic_light_state(self): return 'Green'

class MockMap:
    name='Town01'
    def get_spawn_points(self): return [Transform(Vector(i*35,0,.5)) for i in range(20)]
    def get_waypoint(self,loc,**kw):
        return Waypoint(Transform(Vector(loc.x,0,loc.z)))

class Waypoint:
    def __init__(self,t): self.transform=t; self.lane_width=3.5; self.is_junction=False
    def next(self,distance): return [Waypoint(Transform(Vector(self.transform.location.x+distance,0,.5)))]

class TrafficManager:
    def __init__(self): self.sync=False
    def set_synchronous_mode(self,v): self.sync=v
    def set_random_device_seed(self,v): pass
    def get_port(self): return 8000

class MockWorld:
    def __init__(self):
        self.settings=MockSettings(); self.actors=[]; self.frame=0; self.ego=None
        self.collision_pairs=set(); self.last_visible_ids=set(); self.last_bboxes={}
    def get_settings(self): return copy.copy(self.settings)
    def apply_settings(self,s): self.settings=copy.copy(s)
    def get_blueprint_library(self): return Library()
    def get_map(self): return MockMap()
    def get_actors(self): return [a for a in self.actors if a.alive]
    def spawn_actor(self,b,t,attach_to=None):
        a=Actor(self,b,t,attach_to); self.actors.append(a)
        if self.ego is None and 'vehicle' in b.id:self.ego=a
        return a
    def try_spawn_actor(self,b,t): return self.spawn_actor(b,t)
    def render(self):
        h,w=478,848; focal=w/(2*math.tan(math.radians(37.5)))
        rgb=np.zeros((h,w,4),np.uint8);rgb[:]=(150,190,225,255);rgb[h//2:]=(60,125,60,255)
        tags=np.zeros((h,w,4),np.uint8); tags[:,:,3]=255
        ego=self.ego
        yaw=math.radians(ego.pose.rotation.yaw)
        boundaries=[]
        for lateral in (-3.5,3.5):
            points=[]
            for dx in np.geomspace(.5,150.,100):
                dy=lateral-ego.pose.location.y
                depth=dx*math.cos(yaw)+dy*math.sin(yaw)
                side=-dx*math.sin(yaw)+dy*math.cos(yaw)
                if depth>.2:
                    points.append((int(w/2+focal*side/depth),int(h/2+focal*1.3/depth)))
            boundaries.append(points)
        if all(boundaries):
            polygon=np.array(boundaries[0]+list(reversed(boundaries[1])),np.int32)
            cv2.fillPoly(rgb,[polygon],(105,105,105,255))
            cv2.fillPoly(tags,[polygon],(0,0,7,255))
            for points in boundaries:
                cv2.polylines(rgb,[np.array(points,np.int32)],False,(230,230,230,255),3)
                cv2.polylines(tags,[np.array(points,np.int32)],False,(0,0,6,255),3)
        objects=[]
        for a in self.actors:
            if not a.alive or a is ego or a.parent or a.hidden:continue
            if 'vehicle' not in a.type_id and 'walker' not in a.type_id:continue
            dx=a.pose.location.x-ego.pose.location.x-1.2;dy=a.pose.location.y-ego.pose.location.y
            yaw=math.radians(ego.pose.rotation.yaw);d=dx*math.cos(yaw)+dy*math.sin(yaw);side=-dx*math.sin(yaw)+dy*math.cos(yaw)
            if d<=.4:continue
            person='walker' in a.type_id;height=1.75 if person else 2*a.bounding_box.extent.z;width=.55 if person else 1.8
            x=int(w/2+focal*side/d);bottom=int(h/2+focal*1.3/d)
            box=(int(x-focal*width/(2*d)),int(bottom-focal*height/d),int(x+focal*width/(2*d)),bottom)
            objects.append((d,a,box,person))
        self.last_visible_ids=set();self.last_bboxes={}
        occupied=np.zeros((h,w),bool)
        for d,a,box,person in sorted(objects,key=lambda x:x[0]):
            x1,y1,x2,y2=box;x1=max(0,x1);y1=max(0,y1);x2=min(w,x2);y2=min(h,y2)
            if x2<=x1 or y2<=y1:continue
            if np.mean(occupied[y1:y2,x1:x2])>.9:continue
            self.last_visible_ids.add(a.id);self.last_bboxes[a.id]=(x1,y1,x2,y2)
            occupied[y1:y2,x1:x2]=True
        for d,a,box,person in sorted(objects,key=lambda x:-x[0]):
            x1,y1,x2,y2=box
            cv2.rectangle(rgb,(x1,y1),(x2,y2),(30,30,220,255) if person else (190,70,20,255),-1)
            cv2.rectangle(tags,(x1,y1),(x2,y2),(0,0,4 if person else 10,255),-1)
        return rgb,tags
    def tick(self):
        dt=self.settings.fixed_delta_seconds or .05;self.frame+=1
        for a in self.actors:
            if not a.alive or not a.physics:continue
            if 'walker' in a.type_id:a.pose.location=a.pose.location+a.get_velocity()*dt
            elif 'vehicle' in a.type_id:
                c=a.control
                if a.autopilot:c=MockControl(throttle=.5 if a.speed<4 else 0.)
                a.speed=max(0.,a.speed+(c.throttle*2.5-c.brake*6.-.08*a.speed)*dt)
                a.pose.rotation.yaw+=math.degrees(a.speed/2.7*math.tan(math.radians(c.steer*35)))*dt
                a.pose.location=a.pose.location+a.pose.get_forward_vector()*a.speed*dt
        rgb,tags=self.render()
        for a in self.actors:
            if not a.alive or not a.callback:continue
            if 'camera' in a.type_id:
                arr=tags if 'semantic' in a.type_id else rgb
                a.callback(SimpleNamespace(raw_data=arr.tobytes(),width=848,height=478,frame=self.frame))
        ego=self.ego
        for a in self.actors:
            if not a.alive or a is ego or a.parent or a.hidden:continue
            if 'walker' not in a.type_id and 'vehicle' not in a.type_id:continue
            e=ego.bounding_box.extent;o=a.bounding_box.extent
            if abs(a.pose.location.x-ego.pose.location.x)<e.x+o.x and abs(a.pose.location.y-ego.pose.location.y)<e.y+o.y:
                key=(ego.id,a.id)
                if key not in self.collision_pairs:
                    self.collision_pairs.add(key)
                    for s in self.actors:
                        if s.alive and s.callback and 'collision' in s.type_id:s.callback(SimpleNamespace(frame=self.frame,other_actor=a))
        return self.frame

class MockClient:
    def __init__(self,*a): self.world=MockWorld();self.tm=TrafficManager()
    def set_timeout(self,v):pass
    def get_world(self):return self.world
    def load_world(self,town):self.world=MockWorld();return self.world
    def get_server_version(self):return 'mock-1'
    def get_client_version(self):return 'mock-1'
    def get_trafficmanager(self,port=8000):return self.tm

class MockCarla:
    Client=MockClient;VehicleControl=MockControl;Location=Vector;Vector3D=Vector
    Rotation=Rotation;Transform=Transform;WalkerControl=WalkerControl
    TrafficLightState=SimpleNamespace(Red='Red',Yellow='Yellow',Green='Green')
