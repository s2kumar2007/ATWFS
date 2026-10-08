"""Obstacle detection for obstacle subtraction.

Default: pretrained COCO YOLOv8n via the ``ultralytics`` package, filtered to person/vehicle classes.
"""
from __future__ import annotations

import logging
import threading
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Optional

import numpy as np
import cv2

from utils.config import resolve_path

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """One detected obstacle (pixel coordinates in the input frame)."""
    x1: int
    y1: int
    x2: int
    y2: int
    cls_id: int
    name: str
    conf: float
    kind: str = "vehicle"
    approach_rate: float = 0.0
    tracker_id: int = -1


class ObstacleDetector:
    def detect(self, bgr: np.ndarray) -> List[Detection]:
        raise NotImplementedError
    def stop(self):
        pass


class NullDetector(ObstacleDetector):
    def detect(self, bgr: np.ndarray) -> List[Detection]:
        return []


class YoloV8nDetector(ObstacleDetector):
    def __init__(self, weights: str, conf: float, class_ids: Sequence[int], imgsz: int = 640) -> None:
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf
        self.class_ids = list(class_ids)
        self.names: Dict[int, str] = dict(self.model.names)
        self.imgsz = imgsz
        
        # classes: 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck, 15=cat, 16=dog, 17=horse, 18=sheep, 19=cow
        self.vehicles = {1, 2, 3, 5, 7}
        self.persons = {0}

    def detect(self, bgr: np.ndarray) -> List[Detection]:
        # Run at downscaled imgsz
        res = self.model.track(bgr, conf=self.conf, classes=self.class_ids, persist=True, verbose=False, tracker="botsort.yaml", imgsz=self.imgsz)[0]
        out: List[Detection] = []
        if res.boxes is None or len(res.boxes) == 0:
            return out
        for box in res.boxes:
            c = int(box.cls.item())
            p = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Ignore hood bottom
            if y2 >= bgr.shape[0] - 10 and y1 > bgr.shape[0] * 0.4:
                continue
            
            tid = int(box.id.item()) if box.id is not None else -1
            base_name = self.names.get(c, str(c))
            
            kind = "vehicle" if c in self.vehicles else ("person" if c in self.persons else "animal")
            out.append(Detection(x1, y1, x2, y2, c, base_name, p, kind=kind, tracker_id=tid))
        return out


class ThreadedDetector(ObstacleDetector):
    """Runs YOLO in a background thread every N frames."""
    def __init__(self, base: ObstacleDetector, every_n: int = 3):
        self.base = base
        self.every_n = every_n
        self.frame_count = 0
        self.latest_dets: List[Detection] = []
        self.lock = threading.Lock()
        
        self.q = queue.Queue(maxsize=1)
        self.stop_ev = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        if not isinstance(base, NullDetector):
            self.thread.start()

    def _run(self):
        while not self.stop_ev.is_set():
            try:
                frame = self.q.get(timeout=0.1)
                if frame is None:
                    break
            except queue.Empty:
                continue
            
            dets = self.base.detect(frame)
            with self.lock:
                self.latest_dets = dets

    def detect(self, bgr: np.ndarray) -> List[Detection]:
        if isinstance(self.base, NullDetector):
            return []
            
        self.frame_count += 1
        if self.frame_count % self.every_n == 1 or self.every_n == 1:
            if not self.q.full():
                self.q.put(bgr.copy())
                
        with self.lock:
            return list(self.latest_dets)
            
    def stop(self):
        self.stop_ev.set()
        if self.thread.is_alive():
            try:
                self.q.put_nowait(None)
            except:
                pass
            self.thread.join(timeout=1.0)


def build_detector(det_cfg: Dict[str, Any], output_dir: str) -> ObstacleDetector:
    if det_cfg["backend"] == "none":
        return NullDetector()
    try:
        wdir = resolve_path(output_dir) / "weights"
        wdir.mkdir(parents=True, exist_ok=True)
        wpath = wdir / Path(det_cfg["weights"]).name
        det = YoloV8nDetector(str(wpath), det_cfg["conf"], det_cfg["class_ids"])
        every_n = det_cfg.get("every_n", 3)
        logger.info("Obstacle detector: YOLO (%s), classes %s, every_n=%d", wpath.name, det_cfg["class_ids"], every_n)
        return ThreadedDetector(det, every_n=every_n)
    except Exception as e:
        logger.warning("YOLO obstacle detector unavailable: %s", str(e)[:120])
        return NullDetector()
