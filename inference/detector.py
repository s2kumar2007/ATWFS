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
import queue
import threading
import time
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
    last_update_frame: int = 0
    consecutive_detections: int = 1

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

    def close(self) -> None:
        self.stop()


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
        self.last_real_detection_frame: Dict[int, int] = {}
        self.consecutive_detection_count: Dict[int, int] = {}

        self._last_shape: Optional[Tuple[int, int]] = None
        self._last_active_dets: List[Detection] = []
        self._next_track_id = 10000
        self._last_tracker_frame: Optional[int] = None

        self.active_pass_samples: List[float] = []
        self.overhead_samples: List[float] = []

        # Warm-up pass once so first video frame does not stall
        try:
            dummy = np.zeros((160, 160, 3), dtype=np.uint8)
            self.model.predict(dummy, imgsz=160, device=self.device_str, verbose=False)
        except Exception:
            pass

        # Background worker for active YOLO inference
        self._worker_queue: queue.Queue[Optional[Tuple[np.ndarray, int]]] = queue.Queue(maxsize=1)
        self._result_queue: queue.Queue[Tuple[int, List[Detection], float]] = queue.Queue()
        self._worker_busy = threading.Event()
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._yolo_worker_loop, daemon=True)
        self._worker_thread.start()

    def _yolo_worker_loop(self) -> None:
        try:
            torch.set_num_threads(4)
        except Exception:
            pass
        while not self._stop_event.is_set():
            try:
                task = self._worker_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if task is None:
                break
            bgr, frame_idx = task
            try:
                t0 = time.perf_counter()
                dets = self._run_active_pass(bgr, frame_idx)
                active_ms = (time.perf_counter() - t0) * 1000.0
                self._result_queue.put((frame_idx, dets, active_ms))
            except Exception as exc:
                logger.error("Error in YOLO active pass worker: %s", exc)
            finally:
                self._worker_busy.clear()

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

    def _run_active_pass(self, bgr: np.ndarray, frame_idx: int) -> List[Detection]:
        h, w = bgr.shape[:2]
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

                raw_name = self.model.names[cls_id] if hasattr(self.model, "names") else str(cls_id)
                if "auto" in raw_name.lower() or "rickshaw" in raw_name.lower():
                    disp_name = "auto"
                elif kind == "vehicle":
                    disp_name = "vehicle"
                elif kind == "person":
                    disp_name = "person"
                elif kind == "animal":
                    disp_name = "animal"
                else:
                    disp_name = raw_name

                rx1 = int(round((x1 / scale) + offset_x))
                ry1 = int(round((y1 / scale) + offset_y))
                rx2 = int(round((x2 / scale) + offset_x))
                ry2 = int(round((y2 / scale) + offset_y))

                # Clamp to frame bounds
                rx1 = max(0, min(w - 1, rx1))
                ry1 = max(0, min(h - 1, ry1))
                rx2 = max(0, min(w - 1, rx2))
                ry2 = max(0, min(h - 1, ry2))
                if (rx2 - rx1) < 5 or (ry2 - ry1) < 5:
                    continue

                dets.append(
                    Detection(
                        rx1, ry1, rx2, ry2, cls_id, disp_name, float(conf), kind, 1.0, None,
                        last_update_frame=frame_idx, consecutive_detections=1,
                    )
                )

        if self.device_type == "cpu":
            # Road band pass on full road region (y from 0.25 to 0.85) with rectangular inference at imgsz 480
            y1_road, y2_road = int(0.25 * h), int(0.85 * h)
            road_crop = bgr[y1_road:y2_road, :]
            crop_kwargs = dict(predict_kwargs)
            crop_kwargs["imgsz"] = 480
            crop_kwargs["rect"] = True
            results_crop = self.model.predict(road_crop, **crop_kwargs)[0]
            extract_boxes(results_crop.boxes, offset_y=y1_road)

            # Low-resolution full-frame pass at most every 8th frame and only when fewer than 2 vehicles are found
            num_vehicles = sum(1 for d in dets if d.kind == "vehicle")
            if (frame_idx % 8 == 0) and (num_vehicles < 2):
                full_kwargs = dict(predict_kwargs)
                full_kwargs["imgsz"] = 416
                full_kwargs["rect"] = True
                results_full = self.model.predict(bgr, **full_kwargs)[0]
                extract_boxes(results_full.boxes)
        else:
            # Pass 1: Full-frame predict (CUDA)
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

        merged = merge_boxes(dets, iou_thresh=0.5)
        return merged

    def detect(self, bgr: np.ndarray, frame_idx: int = 0) -> List[Detection]:
        """Detect obstacles without stalling the main pipeline thread."""
        t0 = time.perf_counter()
        h, w = bgr.shape[:2]

        # Fail fast if consecutive frames ever differ in size
        if self._last_shape is not None and (h, w) != self._last_shape:
            raise ValueError(
                f"Frame size changed from {self._last_shape} to {(h, w)}. "
                f"Detector requires constant resolution across video frames."
            )
        self._last_shape = (h, w)

        # 1. Drain newly completed active pass results from worker
        received_new_result = False
        while not self._result_queue.empty():
            try:
                res_idx, res_dets, active_ms = self._result_queue.get_nowait()
                self.active_pass_samples.append(active_ms)
                # Update tracker with newly arrived detections at res_idx
                self._update_tracker_with_dets(res_dets, res_idx)
                # Fast-forward tracker from res_idx + 1 up to current frame_idx
                for ff_idx in range(res_idx + 1, frame_idx + 1):
                    self._step_tracker_forward(ff_idx)
                received_new_result = True
            except queue.Empty:
                break

        # 2. Trigger worker on cadence if worker is idle
        if frame_idx % self.every_n == 0:
            if not self._worker_busy.is_set():
                self._worker_busy.set()
                try:
                    self._worker_queue.put_nowait((bgr.copy(), frame_idx))
                except queue.Full:
                    self._worker_busy.clear()

        # 3. If no new result was received or tracker needs propagation to current frame
        if not received_new_result:
            if self._last_tracker_frame is None or self._last_tracker_frame < frame_idx:
                start_f = frame_idx if self._last_tracker_frame is None else self._last_tracker_frame + 1
                for f in range(start_f, frame_idx + 1):
                    self._step_tracker_forward(f)

        t_ms = (time.perf_counter() - t0) * 1000.0
        self.overhead_samples.append(t_ms)
        # Every box drawn or used for decisions must come from a detection or tracker update within the last 3 frames;
        # Never draw a box that hasn't been updated.
        valid_dets = [
            d for d in self._last_active_dets
            if (frame_idx - self.last_real_detection_frame.get(d.tracker_id, -999)) <= 3
        ]
        return list(valid_dets)

    def _update_tracker_with_dets(self, merged: List[Detection], frame_idx: int) -> None:
        """Update BYTETracker with merged full-frame detections at frame_idx."""
        if merged:
            xywh = np.array([[d.center_x, d.center_y, float(d.width), float(d.height)] for d in merged], dtype=np.float32)
            scores = np.array([float(d.conf) for d in merged], dtype=np.float32)
            classes = np.array([float(d.cls_id) for d in merged], dtype=np.float32)
            wrapper = _DetResultsWrapper(xywh, scores, classes)
            self.tracker.update(wrapper)

            # Match tracker IDs to detections via track.idx
            matched_indices = set()
            active_tids = set()
            for track in self.tracker.tracked_stracks:
                idx = getattr(track, "idx", None)
                if idx is not None and 0 <= idx < len(merged) and idx not in matched_indices:
                    tid = int(track.track_id)
                    merged[idx].tracker_id = tid
                    matched_indices.add(idx)
                    active_tids.add(tid)

            # Assign fresh ID to any unmatched detection
            for idx, d in enumerate(merged):
                if d.tracker_id is None:
                    d.tracker_id = self._next_track_id
                    active_tids.add(self._next_track_id)
                    self._next_track_id += 1

            # Update approach rate, track history, and real detection timestamps
            for d in merged:
                tid = d.tracker_id
                self.last_seen[tid] = frame_idx
                self.last_real_detection_frame[tid] = frame_idx
                self.consecutive_detection_count[tid] = self.consecutive_detection_count.get(tid, 0) + 1
                d.last_update_frame = frame_idx
                d.consecutive_detections = self.consecutive_detection_count[tid]

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

            # Reset consecutive count for tracks not detected in this pass
            for tid in list(self.consecutive_detection_count.keys()):
                if tid not in active_tids:
                    self.consecutive_detection_count[tid] = 0
        else:
            wrapper = _DetResultsWrapper(
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
            self.tracker.update(wrapper)
            # Reset consecutive counts on empty frame
            for tid in list(self.consecutive_detection_count.keys()):
                self.consecutive_detection_count[tid] = 0

        # Clean up stale tracks (tracks lost for more than 8 frames are removed)
        stale_ids = [tid for tid, last_f in self.last_real_detection_frame.items() if frame_idx - last_f > 8]
        for tid in stale_ids:
            self.track_histories.pop(tid, None)
            self.ema_rates.pop(tid, None)
            self.last_seen.pop(tid, None)
            self.last_real_detection_frame.pop(tid, None)
            self.consecutive_detection_count.pop(tid, None)
            if hasattr(self.tracker, "tracked_stracks"):
                self.tracker.tracked_stracks = [t for t in self.tracker.tracked_stracks if int(t.track_id) != tid]
            if hasattr(self.tracker, "lost_stracks"):
                self.tracker.lost_stracks = [t for t in self.tracker.lost_stracks if int(t.track_id) != tid]

        self._last_active_dets = merged
        self._last_tracker_frame = frame_idx

    def _step_tracker_forward(self, frame_idx: int) -> None:
        """Fast-forward / propagate tracked boxes using BYTETracker Kalman state."""
        if hasattr(self.tracker, "multi_predict") and self.tracker.tracked_stracks:
            self.tracker.multi_predict(self.tracker.tracked_stracks)

        prop_dets: List[Detection] = []
        track_map = {int(t.track_id): t for t in self.tracker.tracked_stracks}

        for old_d in self._last_active_dets:
            tid = old_d.tracker_id
            # Propagated boxes older than 3 frames are dropped
            age = frame_idx - self.last_real_detection_frame.get(tid, -999)
            if age > 3:
                continue

            t = track_map.get(tid)
            if t is not None and hasattr(t, "xyxy"):
                coords = [float(v) for v in t.xyxy]
                x1, y1, x2, y2 = int(round(coords[0])), int(round(coords[1])), int(round(coords[2])), int(round(coords[3]))
            else:
                x1, y1, x2, y2 = old_d.x1, old_d.y1, old_d.x2, old_d.y2

            # Clamp coordinates
            if self._last_shape:
                h, w = self._last_shape
                x1, y1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
                x2, y2 = max(0, min(w - 1, x2)), max(0, min(h - 1, y2))
                if (x2 - x1) < 5 or (y2 - y1) < 5:
                    continue

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
                last_update_frame=self.last_real_detection_frame.get(tid, old_d.last_update_frame),
                consecutive_detections=self.consecutive_detection_count.get(tid, 0),
            )

            # Update approach rate on propagated frame
            hist = self.track_histories[tid]
            hist.append(new_d.height)
            if len(hist) > 1:
                raw_rate = new_d.height / max(1.0, hist[0])
                prev_rate = self.ema_rates.get(tid, 1.0)
                rate = 0.3 * raw_rate + 0.7 * prev_rate
                self.ema_rates[tid] = rate
                new_d.approach_rate = rate

            prop_dets.append(new_d)

        self._last_active_dets = prop_dets
        self._last_tracker_frame = frame_idx

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._worker_queue.put_nowait(None)
        except Exception:
            pass
        if hasattr(self, "_worker_thread") and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)


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
