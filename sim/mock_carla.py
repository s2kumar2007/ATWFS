"""Tiny deterministic CARLA-shaped API for CPU-only closed-loop tests."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class MockControl:
    steer: float=0.; throttle: float=0.; brake: float=0.

@dataclass
class MockSettings:
    synchronous_mode: bool=False
    fixed_delta_seconds: float|None=None

class MockActor:
    _ids=0
    def __init__(self, world, kind="actor"):
        type(self)._ids+=1; self.id=type(self)._ids; self.world=world; self.kind=kind; self.alive=True
    def destroy(self): self.alive=False; return True
    def listen(self, callback): self.callback=callback
    def stop(self): pass
    def get_velocity(self): return type("V",(),dict(x=self.world.speed,y=0.,z=0.))()
    def get_location(self): return type("L",(),dict(x=self.world.position,y=0.,z=0.))()
    def get_transform(self): return type("T",(),dict(location=self.get_location(),rotation=type("R",(),dict(yaw=0.))()))()
    def get_physics_control(self): return type("P",(),dict(wheels=[type("W",(),dict(max_steer_angle=35.,position=type("P",(),dict(x=x))()))() for x in (-135,-135,135,135)]))()
    def apply_control(self,c): self.world.control=c
    def set_autopilot(self,*a): pass
    def set_transform(self,*a): pass

class MockBlueprint:
    def __init__(self,id): self.id=id; self.attrs={}
    def set_attribute(self,k,v): self.attrs[k]=v

class MockBlueprintLibrary:
    def find(self,n): return MockBlueprint(n)
    def filter(self,n): return [MockBlueprint(n)]

class MockMap:
    name="Town01"
    def get_spawn_points(self): return [object()]

class MockWorld:
    def __init__(self): self.settings=MockSettings(); self.actors=[]; self.speed=0.; self.position=0.; self.control=MockControl(); self.frame=0
    def get_settings(self): return self.settings
    def apply_settings(self,s): self.settings=s
    def get_blueprint_library(self): return MockBlueprintLibrary()
    def get_map(self): return MockMap()
    def spawn_actor(self,b,t,attach_to=None):
        a=MockActor(self,b.id)
        if "camera.rgb" in b.id: a.kind="rgb"
        elif "segmentation" in b.id: a.kind="semantic"
        self.actors.append(a); return a
    def try_spawn_actor(self,b,t): return self.spawn_actor(b,t)
    def tick(self):
        dt=self.settings.fixed_delta_seconds or .05; c=self.control
        self.speed=max(0.,self.speed+(c.throttle*2.0-c.brake*5.0)*dt); self.position+=self.speed*dt; self.frame+=1
        for a in self.actors:
            if a.alive and a.kind in ("rgb","semantic"): a.callback(type("Image",(),dict(raw_data=np.full((478,848,4),120,np.uint8).tobytes(),width=848,height=478,frame=self.frame))())
        return self.frame

class MockClient:
    def __init__(self,*a): self.world=MockWorld()
    def get_world(self): return self.world
    def get_server_version(self): return "mock-1"
    def get_client_version(self): return "mock-1"

class MockCarla:
    Client=MockClient
    VehicleControl=MockControl
    Location=lambda self,**kw: type("Location",(),kw)()
    Rotation=lambda self,**kw: type("Rotation",(),kw)()
    Transform=lambda self,*a: type("Transform",(),dict(location=a[0] if a else None))()
    ColorConverter=type("ColorConverter",(),dict(CityScapesPalette=0))
    @staticmethod
    def libcarla_version(): return "mock-1"
