"""Closed-loop CARLA bridge with deterministic mock support."""
from __future__ import annotations
import argparse, math, time, logging, queue, heapq, copy
from pathlib import Path
from datetime import datetime
import cv2, numpy as np, torch
from utils.config import load_config
from inference.live import build_live_pipeline

DT=.05
log=logging.getLogger("sim.carla")

def command_to_control(command,max_steer_deg=35.,vmax=6.,speed=0.,integral=0.,dt=DT,wheelbase_m=2.7,pipeline_vmax=1.5):
    """CARLA normalized steer; angle degrees, speed m/s; bicycle geometry caps angle."""
    lim=max(1e-3,float(max_steer_deg)); steer=float(np.clip(float(command.angle)/lim,-1,1))
    # retain the bicycle model geometry in the physical wheel-angle conversion
    _curvature=math.tan(math.radians(steer*lim))/max(.1,wheelbase_m)
    target=float(np.clip(command.speed/max(1e-6,pipeline_vmax),0,1))*vmax
    err=target-speed; throttle=float(np.clip(.45*err+integral*.01,0,1)); brake=float(np.clip(-.6*err,0,1))
    if target<.05: throttle=0.; brake=max(brake,.35)
    return dict(steer=steer,throttle=throttle,brake=brake),integral+err*dt

class CarlaBridge:
    def __init__(self, mock=False, host="localhost", port=2000, pipeline=None, latency_mode="real", v_max=6., log_dir=None):
        if latency_mode != 'real':
            try:
                if not latency_mode.startswith('fixed:') or float(latency_mode.split(':',1)[1]) < 0:
                    raise ValueError
            except (ValueError, IndexError):
                raise ValueError("Latency mode must be real or fixed:<nonnegative milliseconds>.")
        if mock:
            from sim.mock_carla import MockCarla
            self.carla=MockCarla(); self.client=self.carla.Client(host,port)
        else:
            try: import carla
            except ImportError as e: raise RuntimeError("CARLA Python API unavailable. Install the CARLA matching-version PythonAPI and start the server.") from e
            self.carla=carla; self.client=carla.Client(host,int(port)); self.client.set_timeout(10.)
        cv=self.client.get_client_version(); sv=self.client.get_server_version()
        if cv!=sv: raise RuntimeError(f"CARLA client/server version mismatch: client {cv}, server {sv}. Install matching CARLA PythonAPI/server versions.")
        self.world=self.client.get_world(); self.original=copy.copy(self.world.get_settings()); self.actors=[]; self.pipeline=pipeline
        self.latency_mode=latency_mode; self.v_max=v_max; self.log_dir=Path(log_dir or "outputs/sim_carla")
        self.frames=queue.Queue(maxsize=1); self.commands=[]; self.latencies=[]; self.applied=0; self.collisions=[]; self.semantic_frames={}; self.pipeline_vmax=1.5
    def spawn(self,bp,transform,attach_to=None):
        a=self.world.spawn_actor(bp,transform,attach_to=attach_to); self.actors.append(a); return a
    def setup(self):
        s=copy.copy(self.world.get_settings()); s.synchronous_mode=True; s.fixed_delta_seconds=DT; self.world.apply_settings(s)
        lib=self.world.get_blueprint_library(); bp=lib.filter("vehicle.*")[0]
        sp=self.world.get_map().get_spawn_points()[0]; self.vehicle=self.spawn(bp,sp)
        wheels=self.vehicle.get_physics_control().wheels
        xs=[w.position.x for w in wheels]
        self.vehicle_wheelbase=(max(xs)-min(xs))/100.0
        if self.vehicle_wheelbase<=0: raise RuntimeError("Vehicle physics has invalid wheel positions; cannot determine wheelbase.")
        self.max_steer=max(w.max_steer_angle for w in wheels)
        if self.pipeline is not None:
            self.pipeline.cfg['inference']['decision']['wheelbase_m']=self.vehicle_wheelbase
            self.pipeline_vmax=float(self.pipeline.cfg['inference']['decision']['v_max'])
        transform=self.carla.Transform(self.carla.Location(x=1.2,z=1.3),self.carla.Rotation(pitch=0))
        for name in ("sensor.camera.rgb","sensor.camera.semantic_segmentation"):
            cam=lib.find(name); cam.set_attribute("image_size_x","848");cam.set_attribute("image_size_y","478");cam.set_attribute("fov","75")
            a=self.spawn(cam,transform,attach_to=self.vehicle)
            if "rgb" in name:a.listen(lambda im:self._frame(im,False))
            else:a.listen(lambda im:self._frame(im,True))
        col=lib.find("sensor.other.collision"); ca=self.spawn(col,self.carla.Transform(self.carla.Location(x=0,z=1)),attach_to=self.vehicle); ca.listen(self.collisions.append)
        print("Camera: 848x478, FOV 75; implied vertical FOV 45.1 deg. config.camera hfov=75/vfov=45-47; hood polygon disabled.")
    def _frame(self,im,semantic):
        raw=np.frombuffer(im.raw_data,dtype=np.uint8).reshape(im.height,im.width,4)
        if semantic:
            self.semantic_frames[im.frame]=np.isin(raw[:,:,2], [6,7])
            for key in list(self.semantic_frames):
                if key < im.frame-5: del self.semantic_frames[key]
            return
        frame=raw[:,:,:3].copy()
        if not semantic:
            try:self.frames.put_nowait((im.frame,frame))
            except queue.Full:
                try:self.frames.get_nowait()
                except queue.Empty:pass
                self.frames.put_nowait((im.frame,frame))
    def run(self,seconds=10):
        self.setup(); start_position=self.vehicle.get_location().x; cfg=load_config(); cfg['inference']['hood_polygon']=None
        if self.pipeline is None:self.pipeline=build_live_pipeline(cfg,backend="onnx" if cfg.get('_onnx') else None,device="cuda" if torch.cuda.is_available() else "cpu",hood_polygon="none")
        steps=int(seconds/DT); pending=[]; prev=None; integral=0.; t0=time.perf_counter()
        for i in range(steps):
            self.world.tick()
            try:idx,frame=self.frames.get(timeout=2)
            except queue.Empty:raise RuntimeError("RGB camera produced no frames; check CARLA sensors/server.")
            vel=self.vehicle.get_velocity(); speed=math.sqrt(vel.x**2+vel.y**2+vel.z**2)
            t=time.perf_counter(); result=self.pipeline.process_frame(frame,i,DT,speed,None); latency=time.perf_counter()-t; self.latencies.append(latency)
            delay=math.ceil((float(self.latency_mode.split(':')[1])/1000 if self.latency_mode.startswith('fixed:') else latency)/DT)
            heapq.heappush(pending,(i+delay,i,result.command))
            while pending and pending[0][0]<=i:
                c=heapq.heappop(pending)[2]; ctrl,integral=command_to_control(c,self.max_steer,self.v_max,speed,integral,DT,self.vehicle_wheelbase,self.pipeline_vmax)
                self.vehicle.apply_control(self.carla.VehicleControl(**ctrl)); self.applied+=1
            log.info("tick=%s speed_mps=%s position=%s",i,speed,self.vehicle.get_location().x)
            if i%20==0: print(f"tick={i} speed_mps={speed:.2f} position={self.vehicle.get_location().x:.2f}")
        return dict(distance_m=abs(self.vehicle.get_location().x-start_position),collisions=len(self.collisions),frames=steps,commands=self.applied,fps=steps/max(1e-6,time.perf_counter()-t0),pipeline_fps=steps/max(1e-6,sum(self.latencies)),latency_ms=float(np.mean(self.latencies)*1000))
    def close(self):
        for a in reversed(self.actors):
            try:a.stop()
            except Exception:pass
            try:a.destroy()
            except Exception:pass
        self.world.apply_settings(self.original)
    def __enter__(self):return self
    def __exit__(self,*e):self.close()
