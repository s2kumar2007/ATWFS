"""Obstacle detector using YOLOv8 with standalone BYTETracker (no GMC, no BoT-SORT).

Features:
- Dual-pass detection (full frame + optional high-resolution road-band crop) using ``model.predict()``
- Standalone BYTETracker operating strictly in full-frame coordinates
- Frame skip support: propagates tracked boxes on skipped frames with Kalman filter
- Strict frame size consistency check (fails fast if input resolution changes mid-stream)
- TensorRT caching and fallback
"""
from __future__ import annotations

import abc
import collections
from dataclasses import dataclass
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Deque, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import torch

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
    def width(self) -> int:
        return self.x2 - self.x1

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
    """Merge duplicate detections using class-agnostic Non-Maximum Suppression."""
    if not dets:
        return []

    boxes = [[float(d.x1), float(d.y1), float(d.width), float(d.height)] for d in dets]
    scores = [float(d.conf) for d in dets]

    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=float(iou_thresh))
    if len(indices) > 0:
        flat = indices.flatten() if hasattr(indices, "flatten") else [i[0] for i in indices]
        return [dets[int(i)] for i in flat]
    return []


class _DetResultsWrapper:
    """Lightweight duck-typed container expected by BYTETracker.update()."""

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask: np.ndarray) -> "_DetResultsWrapper":
        return _DetResultsWrapper(self.xywh[mask], self.conf[mask], self.cls[mask])


class YOLODetector(ObstacleDetector):
    def __init__(
        self,
        cfg: Dict[str, Any],
        output_dir: str,
        device: Optional[torch.device] = None,
        use_tensorrt: bool = False,
    ):
        from ultralytics import YOLO
        from ultralytics.trackers.byte_tracker import BYTETracker

        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device selection: match segmentation model
        if device is None:
            self.device_str = "0" if torch.cuda.is_available() else "cpu"
            self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device_type = device.type
            self.device_str = str(device.index if device.index is not None else (0 if device.type == "cuda" else "cpu"))

        weights_path = cfg.get("weights", "yolov8n.pt")
        self.use_tensorrt = use_tensorrt or cfg.get("tensorrt", False)
        self.model = self._load_yolo_model(weights_path, YOLO)

        self.conf = float(cfg.get("conf", 0.25))
        self.iou = float(cfg.get("iou", 0.5))
        self.agnostic_nms = bool(cfg.get("agnostic_nms", True))
        self.allowed_classes: Set[int] = set(cfg.get("class_ids", [0, 1, 2, 3, 5, 7, 16, 17, 18, 19]))

        th = cfg.get("conf_thresholds", {})
        self.thr_person = float(th.get("person", 0.25))
        self.thr_vehicle = float(th.get("vehicle", 0.25))
        self.thr_animal = float(th.get("animal", 0.30))

        self.imgsz = int(cfg.get("imgsz", 960 if self.device_type == "cuda" else 640))
        self.crop_every_n = int(cfg.get("crop_every_n", 6 if self.device_type == "cuda" else 0))
        self.crop_upscale = float(cfg.get("crop_upscale", 2.0 if self.device_type == "cuda" else 1.0))
        self.every_n = max(1, int(cfg.get("every_n", 1)))

        # Standalone BYTETracker setup
        tracker_args = SimpleNamespace(
            track_high_thresh=self.conf,
            track_low_thresh=max(0.1, self.conf * 0.5),
            new_track_thresh=self.conf,
            track_buffer=30,
            match_thresh=0.8,
            fuse_score=True,
        )
        self.tracker = BYTETracker(tracker_args)

        self.track_histories: Dict[int, Deque[int]] = collections.defaultdict(lambda: collections.deque(maxlen=5))
        self.ema_rates: Dict[int, float] = {}
        self.last_seen: Dict[int, int] = {}

        self._last_shape: Optional[Tuple[int, int]] = None
        self._last_active_dets: List[Detection] = []
        self._next_track_id = 10000

    def _load_yolo_model(self, weights_path: str, yolo_cls: Any) -> Any:
        ckpt_dir = self.output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        p = Path(weights_path)

        if self.use_tensorrt and self.device_type == "cuda":
            engine_name = f"{p.stem}_{self.imgsz}_fp16.engine"
            engine_path = ckpt_dir / engine_name
            if engine_path.is_file():
                try:
                    logger.info("Loading cached TensorRT engine from %s", engine_path)
                    return yolo_cls(str(engine_path))
                except Exception as exc:
                    logger.warning("Failed loading TensorRT engine (%s); falling back to PyTorch YOLO.", exc)

            # Try export once
            try:
                base_model = yolo_cls(weights_path)
                logger.info("Exporting YOLO to half-precision TensorRT engine at %s...", engine_path)
                exported_p = base_model.export(format="engine", half=True, imgsz=self.imgsz, device=self.device_str)
                if Path(exported_p).is_file():
                    Path(exported_p).replace(engine_path)
                    return yolo_cls(str(engine_path))
            except Exception as exc:
                logger.warning("TensorRT export unavailable (%s); falling back to PyTorch YOLO.", exc)

        return yolo_cls(weights_path)

    def _get_kind(self, cls_id: int) -> str:
        if cls_id == 0:
            return "person"
        if cls_id in (14, 15, 16, 17, 18, 19, 20, 21, 22, 23):
            return "animal"
        return "vehicle"

    def _get_threshold(self, kind: str) -> float:
        if kind == "person":
            return self.thr_person
        if kind == "animal":
            return self.thr_animal
        return self.thr_vehicle

    def detect(self, bgr: np.ndarray, frame_idx: int = 0) -> List[Detection]:
        """Detect obstacles with dual-pass predict + standalone BYTETracker."""
        h, w = bgr.shape[:2]

        # Fail fast if consecutive frames ever differ in size
        if self._last_shape is not None and (h, w) != self._last_shape:
            raise ValueError(
                f"Frame size changed from {self._last_shape} to {(h, w)}. "
                f"Detector requires constant resolution across video frames."
            )
        self._last_shape = (h, w)

        # Honour detector.every_n: on skipped frames propagate active boxes with tracker Kalman state
        if self.every_n > 1 and (frame_idx % self.every_n != 0) and self._last_active_dets:
            return self._propagate_skipped_frame(frame_idx)

        is_half = (self.device_type == "cuda")
        predict_kwargs: Dict[str, Any] = {
            "conf": self.conf,
            "iou": self.iou,
            "agnostic_nms": self.agnostic_nms,
            "classes": list(self.allowed_classes),
            "imgsz": self.imgsz,
            "device": self.device_str,
            "verbose": False,
        }
        if is_half:
            predict_kwargs["half"] = True

        dets: List[Detection] = []

        def extract_boxes(boxes: Any, offset_x: float = 0.0, offset_y: float = 0.0, scale: float = 1.0) -> None:
            if boxes is None or len(boxes) == 0:
                return
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, cls_ids):
                if cls_id not in self.allowed_classes:
                    continue
                kind = self._get_kind(cls_id)
                if conf < self._get_threshold(kind):
                    continue

                name = self.model.names[cls_id] if hasattr(self.model, "names") else str(cls_id)
                rx1 = int(round((x1 / scale) + offset_x))
                ry1 = int(round((y1 / scale) + offset_y))
                rx2 = int(round((x2 / scale) + offset_x))
                ry2 = int(round((y2 / scale) + offset_y))

                dets.append(Detection(rx1, ry1, rx2, ry2, cls_id, name, float(conf), kind, 1.0, None))

        if self.device_type == "cpu":
            # Road band pass (y from 0.28 to 0.65 of the frame) with rectangular inference at imgsz 640
            y1_road, y2_road = int(0.28 * h), int(0.65 * h)
            road_crop = bgr[y1_road:y2_road, :]
            crop_kwargs = dict(predict_kwargs)
            crop_kwargs["imgsz"] = 640
            crop_kwargs["rect"] = True
            results_crop = self.model.predict(road_crop, **crop_kwargs)[0]
            extract_boxes(results_crop.boxes, offset_y=y1_road)

            # Merge with low-resolution full-frame pass only when fewer than 2 vehicles are found
            num_vehicles = sum(1 for d in dets if d.kind == "vehicle")
            if num_vehicles < 2:
                full_kwargs = dict(predict_kwargs)
                full_kwargs["imgsz"] = 416
                full_kwargs["rect"] = True
                results_full = self.model.predict(bgr, **full_kwargs)[0]
                extract_boxes(results_full.boxes)
        else:
            # Pass 1: Full-frame predict (NEVER model.track)
            results_full = self.model.predict(bgr, **predict_kwargs)[0]
            extract_boxes(results_full.boxes)

            # Pass 2: Optional small-object road-band crop
            num_vehicles = sum(1 for d in dets if d.kind == "vehicle")
            should_run_crop = (self.crop_every_n > 0) and (
                (frame_idx % self.crop_every_n == 0) or (num_vehicles < 2)
            )

            if should_run_crop:
                y1_crop, y2_crop = int(0.30 * h), int(0.62 * h)
                crop = bgr[y1_crop:y2_crop, :]
                scale = self.crop_upscale
                if abs(scale - 1.0) > 1e-3:
                    crop_in = cv2.resize(crop, (int(round(w * scale)), int(round((y2_crop - y1_crop) * scale))))
                else:
                    crop_in = crop
                    scale = 1.0

                results_crop = self.model.predict(crop_in, **predict_kwargs)[0]
                extract_boxes(results_crop.boxes, offset_y=y1_crop, scale=scale)

        # Merge boxes in full-frame coordinates with IoU 0.5 class-agnostic NMS
        merged = merge_boxes(dets, iou_thresh=0.5)

        # Standalone BYTETracker update strictly on full-frame merged boxes
        if merged:
            xywh = np.array([[d.center_x, d.center_y, float(d.width), float(d.height)] for d in merged], dtype=np.float32)
            scores = np.array([float(d.conf) for d in merged], dtype=np.float32)
            classes = np.array([float(d.cls_id) for d in merged], dtype=np.float32)
            wrapper = _DetResultsWrapper(xywh, scores, classes)
            self.tracker.update(wrapper)

            # Match tracker IDs to detections via track.idx
            matched_indices = set()
            for track in self.tracker.tracked_stracks:
                idx = getattr(track, "idx", None)
                if idx is not None and 0 <= idx < len(merged) and idx not in matched_indices:
                    merged[idx].tracker_id = int(track.track_id)
                    matched_indices.add(idx)

            # Assign fresh ID to any unmatched detection
            for idx, d in enumerate(merged):
                if d.tracker_id is None:
                    d.tracker_id = self._next_track_id
                    self._next_track_id += 1

            # Update approach rate & track history
            for d in merged:
                tid = d.tracker_id
                self.last_seen[tid] = frame_idx
                hist = self.track_histories[tid]
                hist.append(d.height)
                if len(hist) > 1:
                    raw_rate = d.height / max(1.0, hist[0])
                    prev_rate = self.ema_rates.get(tid, 1.0)
                    rate = 0.3 * raw_rate + 0.7 * prev_rate
                    self.ema_rates[tid] = rate
                    d.approach_rate = rate
                else:
                    d.approach_rate = self.ema_rates.get(tid, 1.0)
        else:
            # Empty detection set: update tracker with empty array
            wrapper = _DetResultsWrapper(
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
            self.tracker.update(wrapper)

        # Clean up tracks unseen for 30 frames
        stale_ids = [tid for tid, last_f in self.last_seen.items() if frame_idx - last_f > 30]
        for tid in stale_ids:
            self.track_histories.pop(tid, None)
            self.ema_rates.pop(tid, None)
            self.last_seen.pop(tid, None)

        self._last_active_dets = merged
        return merged

    def _propagate_skipped_frame(self, frame_idx: int) -> List[Detection]:
        """Propagate tracked boxes on skipped frames using BYTETracker Kalman state."""
        if hasattr(self.tracker, "multi_predict") and self.tracker.tracked_stracks:
            self.tracker.multi_predict(self.tracker.tracked_stracks)

        prop_dets: List[Detection] = []
        track_map = {int(t.track_id): t for t in self.tracker.tracked_stracks}

        for old_d in self._last_active_dets:
            tid = old_d.tracker_id
            t = track_map.get(tid)
            if t is not None and hasattr(t, "xyxy"):
                coords = [float(v) for v in t.xyxy]
                x1, y1, x2, y2 = int(round(coords[0])), int(round(coords[1])), int(round(coords[2])), int(round(coords[3]))
            else:
                x1, y1, x2, y2 = old_d.x1, old_d.y1, old_d.x2, old_d.y2

            new_d = Detection(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                cls_id=old_d.cls_id,
                name=old_d.name,
                conf=old_d.conf,
                kind=old_d.kind,
                approach_rate=old_d.approach_rate,
                tracker_id=tid,
            )

            # Update approach rate on skipped frame
            hist = self.track_histories[tid]
            hist.append(new_d.height)
            if len(hist) > 1:
                raw_rate = new_d.height / max(1.0, hist[0])
                prev_rate = self.ema_rates.get(tid, 1.0)
                rate = 0.3 * raw_rate + 0.7 * prev_rate
                self.ema_rates[tid] = rate
                new_d.approach_rate = rate

            self.last_seen[tid] = frame_idx
            prop_dets.append(new_d)

        self._last_active_dets = prop_dets
        return prop_dets


class ThreadedDetector(ObstacleDetector):
    def __init__(self, backend: ObstacleDetector, every_n: int):
        self.backend = backend
        self.every_n = max(1, every_n)
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


def build_detector(
    cfg: Dict[str, Any],
    output_dir: str,
    live_camera: bool = False,
    device: Optional[torch.device] = None,
    use_tensorrt: bool = False,
) -> ObstacleDetector:
    backend_type = cfg.get("backend", "null").lower()
    if backend_type == "null":
        return NullDetector()

    if "yolo" in backend_type:
        det = YOLODetector(cfg, output_dir, device=device, use_tensorrt=use_tensorrt)
        if live_camera and cfg.get("every_n", 1) > 1:
            return ThreadedDetector(det, cfg["every_n"])
        return det

    raise ValueError(f"Unknown detector backend: {backend_type}")
