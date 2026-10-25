"""CARLA synchronous bridge. Sensor truth is isolated to scenario orchestration/metrics."""
from __future__ import annotations
import copy
import heapq
import logging
import math
import queue
import re
import time
import numpy as np
from sim.supervisor import SafetySupervisor
from sim.carla_scenarios import TrafficRuleStandIn
from sim.carla_metrics import Metrics, truth_snapshot

DT=.05
log=logging.getLogger(__name__)

def command_to_control(command,max_steer_deg=35.,vmax=6.,speed=0.,integral=0.,dt=DT,wheelbase_m=2.7,pipeline_vmax=1.5):
    if max_steer_deg<=0 or pipeline_vmax<=0:raise ValueError('Positive steering and pipeline speed caps are required.')
    target=float(np.clip(command.speed/pipeline_vmax,0,1))*vmax
    error=target-speed;integral=float(np.clip(integral+error*dt,-10,10));effort=.45*error+.03*integral
    return dict(steer=float(np.clip(command.angle/max_steer_deg,-1,1)),throttle=float(np.clip(effort,0,1)),
                brake=max(.35 if target<.05 else 0.,float(np.clip(-effort,0,1)))),integral

class SpeedPID:
    def __init__(self):self.integral=0.;self.previous=None
    def update(self,target,speed):
        error=target-speed;derivative=0. if self.previous is None else (error-self.previous)/DT
        self.previous=error
        if abs(error)<2:self.integral=float(np.clip(self.integral+error*DT,-5,5))
        effort=.55*error+.08*self.integral+.015*derivative
        return float(np.clip(effort,0,1)),max(.35 if target<.05 else 0.,float(np.clip(-effort,0,1)))

def latency_ticks(mode,latency,dt=DT):
    if mode=='real':delay=latency
    else:
        try:
            prefix,value=mode.split(':');delay=float(value)/1000
            if prefix!='fixed' or not math.isfinite(delay) or delay<0:raise ValueError
        except (ValueError,TypeError):raise ValueError('Latency mode must be real or fixed:<nonnegative milliseconds>.')
    return math.ceil(delay/dt)

def wheel_geometry(wheels):
    if len(wheels)<4:raise RuntimeError('Use a four-wheel CARLA vehicle; wheel physics are incomplete.')
    front=np.mean([[w.position.x,w.position.y,w.position.z] for w in wheels[:2]],axis=0)
    rear=np.mean([[w.position.x,w.position.y,w.position.z] for w in wheels[2:4]],axis=0)
    raw=float(np.linalg.norm(front-rear))
    candidates=[(raw,'m'),(raw/100.,'cm')]
    plausible=[(value,unit) for value,unit in candidates if 1.5<=value<=6.]
    if len(plausible)!=1:raise RuntimeError(f'Cannot resolve wheelbase units from physics axle separation {raw}; choose a standard four-wheel car.')
    wheelbase,unit=plausible[0];steer=max(float(w.max_steer_angle) for w in wheels[:2])
    if steer<=0:raise RuntimeError('Vehicle front wheels report no steering angle.')
    return wheelbase,steer,unit

class CarlaBridge:
    def __init__(self,mock=False,host='localhost',port=2000,pipeline=None,latency_mode='real',v_max=6.,log_dir=None,town='Town01',seed=0,reload_world=False,decel=4.,enable_rules=True):
        latency_ticks(latency_mode,0.)
        self.mock=mock;self.pipeline=pipeline;self.latency_mode=latency_mode;self.v_max=v_max
        if mock:
            from sim.mock_carla import MockCarla
            self.carla=MockCarla
        else:
            try:import carla
            except ImportError as e:raise RuntimeError('CARLA Python API is missing. Install the wheel matching the CARLA server release; see docs/carla_run.md.') from e
            self.carla=carla
        self.client=self.carla.Client(host,int(port));self.client.set_timeout(15.)
        try:
            client_version=self.client.get_client_version();server_version=self.client.get_server_version()
            if client_version!=server_version:raise RuntimeError(f'CARLA version mismatch: client {client_version}, server {server_version}. Install matching releases.')
            self.world=self.client.get_world()
            if reload_world or not self.world.get_map().name.endswith(town):self.world=self.client.load_world(town)
        except RuntimeError:raise
        except Exception as e:raise RuntimeError(f'Cannot connect to CARLA at {host}:{port}: {e}. Start the server and check firewall/port.') from e
        match=re.match(r'(\d+)\.(\d+)\.(\d+)',server_version)
        self.modern_semantics=bool(match and tuple(map(int,match.groups())) >= (0,9,14))
        if not mock and not match:raise RuntimeError(f'Unrecognized CARLA version {server_version}; cannot choose semantic tag mapping safely.')
        self.original=self.world.get_settings();self.actors=[];self.tm=None;self.tm_enabled=False;self.closed=False
        self.frames=queue.Queue();self.semantic=queue.Queue();self.collisions=[];self.applied=0
        self.pipeline_vmax=1.5;self.latencies=[];self.seed=seed;self.decel=decel;self.enable_rules=enable_rules
        self.supervisor=SafetySupervisor(speed_limit=v_max);self.rules=TrafficRuleStandIn()
        self.log_event('connection',client=client_version,server=server_version,mock=mock,town=town)
    def log_event(self,event,**fields):log.info('%s %s',event,fields)
    def spawn(self,bp,transform,attach_to=None):
        try:a=self.world.spawn_actor(bp,transform,attach_to=attach_to)
        except Exception as e:raise RuntimeError(f'Could not spawn {bp.id} at {transform}: {e}') from e
        self.actors.append(a);return a
    def setup(self):
        s=self.world.get_settings();s.synchronous_mode=True;s.fixed_delta_seconds=DT;self.world.apply_settings(s)
        self.tm=self.client.get_trafficmanager(8000);self.tm.set_synchronous_mode(True);self.tm_enabled=True;self.tm.set_random_device_seed(self.seed)
        if hasattr(self.world,'set_pedestrians_seed'):self.world.set_pedestrians_seed(self.seed)
        lib=self.world.get_blueprint_library();choices=lib.filter('vehicle.tesla.model3') or lib.filter('vehicle.*')
        if not choices:raise RuntimeError('No vehicle blueprints available.')
        spawns=self.world.get_map().get_spawn_points()
        if not spawns:raise RuntimeError('Map has no vehicle spawn points.')
        # Prefer a straight segment long enough for the encounter; no GT route enters perception.
        spawn=None
        for p in spawns:
            wp=self.world.get_map().get_waypoint(p.location);ahead=wp.next(70.) if wp else []
            if ahead and not wp.is_junction and abs((ahead[0].transform.rotation.yaw-p.rotation.yaw+180)%360-180)<10:
                spawn=p;break
        if spawn is None:raise RuntimeError('Map has no suitable 70 m straight start; use Town01 or choose a suitable map.')
        self.vehicle=self.spawn(choices[0],spawn);self.vehicle.apply_control(self.carla.VehicleControl(brake=1.))
        self.vehicle_wheelbase,self.max_steer,unit=wheel_geometry(self.vehicle.get_physics_control().wheels)
        if self.pipeline is None:raise RuntimeError('Build the ATWFS pipeline once and pass it to CarlaBridge.')
        cfg=self.pipeline.cfg;cfg['inference']['decision']['wheelbase_m']=self.vehicle_wheelbase
        self.pipeline_vmax=float(cfg['inference']['decision']['v_max'])
        cfg['camera'].update(height_m=1.3,hfov_deg=75.,vfov_deg=math.degrees(2*math.atan(478/848*math.tan(math.radians(37.5)))))
        self.log_event('vehicle_physics',wheelbase_m=self.vehicle_wheelbase,wheel_position_units=unit,max_steer_deg=self.max_steer)
        pose=self.carla.Transform(self.carla.Location(x=1.2,z=1.3),self.carla.Rotation(pitch=0))
        for name,sem in (('sensor.camera.rgb',False),('sensor.camera.semantic_segmentation',True)):
            bp=lib.find(name)
            for k,v in (('image_size_x','848'),('image_size_y','478'),('fov','75'),('sensor_tick',str(DT))):bp.set_attribute(k,v)
            camera=self.spawn(bp,pose,self.vehicle);camera.listen(lambda im,sem=sem:self._frame(im,sem))
        collision=self.spawn(lib.find('sensor.other.collision'),self.carla.Transform(),self.vehicle)
        collision.listen(lambda event:self.collisions.append(event))
        self.log_event('camera',resolution='848x478',hfov=75.,vfov=cfg['camera']['vfov_deg'],height_m=1.3,pitch=0.,hood_polygon=None)
        print('Camera 848x478 at 1.3 m, pitch 0, hfov 75 / vfov %.2f; hood masking off.'%cfg['camera']['vfov_deg'])
    def _frame(self,im,semantic):
        raw=np.frombuffer(im.raw_data,dtype=np.uint8).reshape(im.height,im.width,4)
        if semantic:
            tags=raw[:,:,2].copy()
            if self.modern_semantics:
                canonical=np.zeros_like(tags)
                canonical[tags==1]=7;canonical[tags==24]=6;canonical[tags==12]=4
                tags=canonical
            self.semantic.put((im.frame,tags))
        else:self.frames.put((im.frame,raw[:,:,:3].copy()))
    @staticmethod
    def sensor_frame(inbox,expected):
        deadline=time.monotonic()+10.
        while True:
            try:idx,data=inbox.get(timeout=max(.001,deadline-time.monotonic()))
            except queue.Empty:raise RuntimeError(f'Camera timeout waiting for synchronous frame {expected}. Check server/sensors.')
            if idx==expected:return data
            if idx>expected:raise RuntimeError(f'Camera synchronization failed: expected {expected}, received {idx}. Another client may be ticking the world.')
    def run(self,seconds=10,scenario=None,recorder=None,run_id=1,label='ATWFS'):
        if not math.isfinite(seconds) or seconds<=0:raise ValueError('Simulation duration must be positive.')
        self.setup()
        if scenario is not None:scenario.start(self)
        start=self.vehicle.get_location();distance=0.;last_location=start;steps=int(round(seconds/DT))
        pending=[];latest=-1;active=None;pid=SpeedPID();metrics=Metrics(self.decel)
        control=dict(steer=0.,throttle=0.,brake=1.);t0=time.perf_counter()
        for i in range(steps):
            t=i*DT;vel=self.vehicle.get_velocity();speed=math.sqrt(vel.x**2+vel.y**2+vel.z**2)
            if scenario is not None:scenario.update(t,speed)
            idx=self.world.tick();frame=self.sensor_frame(self.frames,idx);semantic=self.sensor_frame(self.semantic,idx)
            vel=self.vehicle.get_velocity();speed=math.sqrt(vel.x**2+vel.y**2+vel.z**2)
            before=time.perf_counter();result=self.pipeline.process_frame(frame,i,DT,speed,None)
            if self.pipeline.device.type=='cuda':
                import torch
                torch.cuda.synchronize(self.pipeline.device)
            latency=time.perf_counter()-before;self.latencies.append(latency)
            heapq.heappush(pending,(i+latency_ticks(self.latency_mode,latency),i,result))
            while pending and pending[0][0]<=i:
                _,seq,item=heapq.heappop(pending)
                if seq>latest:active=item;latest=seq;self.applied+=1
            reasons=['waiting for delayed command'] if active is None else []
            if active is not None:
                supervised=self.supervisor.step(active,speed,DT,self.vehicle_wheelbase,self.pipeline_vmax,self.v_max)
                control,_=command_to_control(supervised.command,self.max_steer,self.v_max,speed,pipeline_vmax=self.pipeline_vmax)
                target=float(np.clip(supervised.command.speed/self.pipeline_vmax,0,1))*self.v_max
                control['throttle'],control['brake']=pid.update(target,speed)
                reasons=supervised.reasons
                if supervised.emergency:control.update(throttle=0.,brake=1.)
            rule=self.rules.reason(self,t,speed) if self.enable_rules else None
            if rule:control.update(throttle=0.,brake=1.);reasons=reasons+[rule]
            self.vehicle.apply_control(self.carla.VehicleControl(**control))
            if reasons:self.log_event('supervisor_override',tick=i,reasons=reasons)
            # Truth is collected only AFTER the control decision, and never passed to the supervisor.
            truth=truth_snapshot(self,scenario,semantic)
            metrics.update(t,truth,speed,control['brake'],latency,reasons,result.mask if i%10==0 else None,semantic if i%10==0 else None,bool(rule))
            loc=self.vehicle.get_location();distance+=loc.distance(last_location);last_location=loc
            self.log_event('tick',tick=i,speed_mps=speed,position=(loc.x,loc.y,loc.z),angle_deg=control['steer']*self.max_steer,trust=result.command.trust,latency_ms=latency*1000)
            if recorder is not None:recorder.write(frame,result,run_id,t,speed,control,reasons,label)
            if i%100==0:print(f'run={run_id} tick={i}/{steps} speed={speed:.2f} m/s collisions={len(self.collisions)}')
        # Final tick collision callbacks can lag; synchronous sensor callbacks have already arrived.
        summary=dict(distance_m=distance,collisions=len(self.collisions),frames=steps,commands=self.applied,
                     fps=steps/max(1e-6,time.perf_counter()-t0),latency_ms=float(np.mean(self.latencies)*1000),**metrics.summary())
        if scenario is not None:summary.update(scenario.outcome(len(self.collisions)))
        return summary
    def close(self):
        if self.closed:return
        self.closed=True;errors=[]
        for actor in reversed(self.actors):
            try:
                if actor.type_id.startswith('sensor.'):actor.stop()
                if actor.is_alive:actor.destroy()
            except Exception as e:errors.append(str(e));log.exception('Actor cleanup failed')
        try:
            if self.tm_enabled:self.tm.set_synchronous_mode(False)
        finally:self.world.apply_settings(self.original)
        if errors:raise RuntimeError('CARLA cleanup failed: '+'; '.join(errors))
    def __enter__(self):return self
    def __exit__(self,kind,value,tb):
        try:self.close()
        except Exception:
            if kind is None:raise
            log.exception('Cleanup also failed while handling original error')
