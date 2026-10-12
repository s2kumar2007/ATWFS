import abc
import collections
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Deque

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    cls_id: int
    name: str
    conf: float
    kind: str = "vehicle"
    approach_rate: float = 1.0
    tracker_id: Optional[int] = None
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
        
    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0
        
    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0


class ObstacleDetector(abc.ABC):
    @abc.abstractmethod
    def detect(self, bgr: np.ndarray, frame_idx: int = 0) -> List[Detection]:
        pass

    def stop(self) -> None:
        pass


class NullDetector(ObstacleDetector):
    def detect(self, bgr: np.ndarray, frame_idx: int = 0) -> List[Detection]:
        return []


def merge_boxes(dets: List[Detection], iou_thresh: float = 0.5) -> List[Detection]:
    if not dets:
        return []
    
    boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in dets], dtype=np.float32)
    scores = np.array([d.conf for d in dets], dtype=np.float32)
    
    keep = cv2.dnn.NMSBoxes(
        boxes.tolist(), scores.tolist(), 0.0, iou_thresh
    )
    
    if len(keep) > 0:
        return [dets[i] for i in keep.flatten()]
    return []


class YOLODetector(ObstacleDetector):
    def __init__(self, cfg: Dict[str, Any], output_dir: str):
        from ultralytics import YOLO
        self.model = YOLO(cfg["weights"])
        self.conf = cfg.get("conf", 0.25)
        self.iou = cfg.get("iou", 0.5)
        self.agnostic_nms = cfg.get("agnostic_nms", True)
        self.allowed_classes = set(cfg["class_ids"])
        
        # Specific thresholds
        th = cfg.get("conf_thresholds", {})
        self.thr_person = th.get("person", 0.25)
        self.thr_vehicle = th.get("vehicle", 0.25)
        self.thr_animal = th.get("animal", 0.30)
        
        self.track_histories: Dict[int, Deque[int]] = collections.defaultdict(lambda: collections.deque(maxlen=5))
        self.ema_rates: Dict[int, float] = {}
        
    def _get_kind(self, cls_id: int) -> str:
        if cls_id == 0: return "person"
        if cls_id in [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]: return "animal"
        return "vehicle"
        
    def _get_threshold(self, kind: str) -> float:
        if kind == "person": return self.thr_person
        if kind == "animal": return self.thr_animal
        return self.thr_vehicle

    def detect(self, bgr: np.ndarray, frame_idx: int = 0) -> List[Detection]:
        h, w = bgr.shape[:2]
        
        # First pass: full image (use model.track for consistent IDs)
        # Using persist=True ensures tracking across frames.
        results_full = self.model.track(
            bgr, persist=True, conf=self.conf, iou=self.iou,
            agnostic_nms=self.agnostic_nms, classes=list(self.allowed_classes),
            imgsz=960, verbose=False, tracker="botsort.yaml"
        )[0]
        
        # Second pass: small far objects
        # Crop y from 0.30 to 0.62
        y1_crop, y2_crop = int(0.30 * h), int(0.62 * h)
        crop = bgr[y1_crop:y2_crop, :]
        crop_up = cv2.resize(crop, (w*2, (y2_crop - y1_crop)*2))
        
        results_crop = self.model.predict(
            crop_up, conf=self.conf, iou=self.iou,
            agnostic_nms=self.agnostic_nms, classes=list(self.allowed_classes),
            imgsz=960, verbose=False
        )[0]
        
        dets = []
        
        def process_boxes(boxes, offset_x=0, offset_y=0, scale=1.0):
            if boxes.id is not None:
                ids = boxes.id.cpu().numpy().astype(int)
            else:
                ids = [None] * len(boxes)
                
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            
            for (x1, y1, x2, y2), conf, cls_id, trk_id in zip(xyxy, confs, cls_ids, ids):
                if cls_id not in self.allowed_classes: continue
                
                kind = self._get_kind(cls_id)
                if conf < self._get_threshold(kind): continue
                
                name = self.model.names[cls_id]
                
                rx1 = int((x1 / scale) + offset_x)
                ry1 = int((y1 / scale) + offset_y)
                rx2 = int((x2 / scale) + offset_x)
                ry2 = int((y2 / scale) + offset_y)
                
                dets.append(Detection(rx1, ry1, rx2, ry2, cls_id, name, conf, kind, 1.0, trk_id))
        
        if results_full.boxes is not None:
            process_boxes(results_full.boxes)
            
        if results_crop.boxes is not None:
            process_boxes(results_crop.boxes, offset_y=y1_crop, scale=2.0)
            
        # Merge duplicates (IoU > 0.5)
        merged = merge_boxes(dets, iou_thresh=0.5)
        
        # Compute approach rate for merged boxes
        for d in merged:
            if d.tracker_id is not None:
                hist = self.track_histories[d.tracker_id]
                hist.append(d.height)
                
                if len(hist) > 1:
                    raw_rate = d.height / hist[0]
                    prev_rate = self.ema_rates.get(d.tracker_id, 1.0)
                    rate = 0.3 * raw_rate + 0.7 * prev_rate
                    self.ema_rates[d.tracker_id] = rate
                    d.approach_rate = rate
        
        # Cleanup old tracks
        active_ids = {d.tracker_id for d in merged if d.tracker_id is not None}
        for trk_id in list(self.track_histories.keys()):
            if trk_id not in active_ids:
                if len(self.track_histories[trk_id]) > 0:
                    # just clear it if not seen? Wait, YOLO tracks persist, but if it disappears for a while, we can clear to save memory.
                    pass # let's just let it grow for now or rely on tracker.
        
        logger.debug(f"Frame {frame_idx}: Detected {len(merged)} objects.")
        return merged


class ThreadedDetector(ObstacleDetector):
    def __init__(self, backend: ObstacleDetector, every_n: int):
        self.backend = backend
        self.every_n = every_n
        self.lock = threading.Lock()
        self.latest: List[Detection] = []
        self.stop_event = threading.Event()
        self.frame_bgr: Optional[np.ndarray] = None
        self.frame_idx = 0
        self.cond = threading.Condition()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self) -> None:
        while not self.stop_event.is_set():
            with self.cond:
                self.cond.wait_for(lambda: self.frame_bgr is not None or self.stop_event.is_set())
                if self.stop_event.is_set():
                    break
                bgr = self.frame_bgr.copy()
                idx = self.frame_idx
                self.frame_bgr = None
            
            dets = self.backend.detect(bgr, idx)
            with self.lock:
                self.latest = dets

    def detect(self, bgr: np.ndarray, frame_idx: int = 0) -> List[Detection]:
        if frame_idx % self.every_n == 0:
            with self.cond:
                self.frame_bgr = bgr
                self.frame_idx = frame_idx
                self.cond.notify()
        with self.lock:
            return list(self.latest)

    def stop(self) -> None:
        self.stop_event.set()
        with self.cond:
            self.cond.notify()
        self.thread.join()


def build_detector(cfg: Dict[str, Any], output_dir: str, live_camera: bool = False) -> ObstacleDetector:
    backend_type = cfg.get("backend", "null").lower()
    if backend_type == "null":
        return NullDetector()
        
    if "yolo" in backend_type:
        det = YOLODetector(cfg, output_dir)
        
        # Synchronous for video files, Threaded for live cameras
        if live_camera and cfg.get("every_n", 1) > 1:
            return ThreadedDetector(det, cfg["every_n"])
        return det
        
    raise ValueError(f"Unknown detector backend: {backend_type}")
