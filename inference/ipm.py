"""Self-calibrating Inverse Perspective Mapping (IPM) to Bird's-Eye View (BEV).

Provides:
- Horizon / vanishing point estimation from road lines using Hough transform + EMA smoothing.
- IMU pitch and roll estimation with a complementary filter.
- Dynamic homography computation from camera height, FOV, pitch, and roll.
- Metric road width and lateral offset computation in BEV.
- Fallback hierarchy: IMU -> horizon only -> fixed fraction fallback.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


@dataclass
class IPMResult:
    """Result of IPM processing for a frame."""
    bev_mask: Optional[np.ndarray]
    homography: Optional[np.ndarray]
    pitch_deg: float
    roll_deg: float
    road_width_m: Optional[float]
    lateral_offset_m: Optional[float]
    method: str  # "imu_fused", "horizon_only", "nominal", or "failed"
    valid: bool


class ComplementaryFilter:
    """Complementary filter for fusing accelerometer and gyroscope signals into pitch/roll."""

    def __init__(self, alpha: float = 0.98, nominal_pitch_deg: float = 0.0, nominal_roll_deg: float = 0.0) -> None:
        self.alpha = float(alpha)
        self.pitch_rad = math.radians(nominal_pitch_deg)
        self.roll_rad = math.radians(nominal_roll_deg)
        self.initialized = False

    def update(self, ax: float, ay: float, az: float, gx: float, gy: float, gz: float, dt: float) -> Tuple[float, float]:
        """Update pitch and roll estimates.

        Args:
            ax, ay, az: Accelerometer readings (m/s^2 or g).
            gx, gy, gz: Gyroscope readings (rad/s).
            dt: Time step in seconds.

        Returns:
            (pitch_rad, roll_rad).
        """
        norm_a = math.sqrt(ax * ax + ay * ay + az * az)
        if norm_a < 1e-4:
            # Fall back to pure gyro integration
            self.pitch_rad += gy * dt
            self.roll_rad += gx * dt
            return self.pitch_rad, self.roll_rad

        # Accelerometer angles (assuming z-down or z-up standard body frame: x-forward, y-right, z-down)
        pitch_acc = math.atan2(-ax, math.sqrt(ay * ay + az * az))
        roll_acc = math.atan2(ay, az)

        if not self.initialized:
            self.pitch_rad = pitch_acc
            self.roll_rad = roll_acc
            self.initialized = True
        else:
            dt = max(1e-4, min(0.5, dt))
            # Gyro integration + accelerometer correction
            self.pitch_rad = self.alpha * (self.pitch_rad + gy * dt) + (1.0 - self.alpha) * pitch_acc
            self.roll_rad = self.alpha * (self.roll_rad + gx * dt) + (1.0 - self.alpha) * roll_acc

        return self.pitch_rad, self.roll_rad


class IMUReader:
    """Reads IMU measurements from a CSV file or provides stream samples."""

    def __init__(self, csv_path: Optional[Union[str, Path]] = None) -> None:
        self.rows: List[Dict[str, float]] = []
        self.cursor = 0
        if csv_path and Path(csv_path).exists():
            self._load_csv(Path(csv_path))

    def _load_csv(self, path: Path) -> None:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    self.rows.append({
                        "timestamp": float(r.get("timestamp", 0.0)),
                        "ax": float(r.get("ax", 0.0)),
                        "ay": float(r.get("ay", 0.0)),
                        "az": float(r.get("az", 9.81)),
                        "gx": float(r.get("gx", 0.0)),
                        "gy": float(r.get("gy", 0.0)),
                        "gz": float(r.get("gz", 0.0)),
                    })
                except (ValueError, TypeError):
                    continue

    def get_sample(self, frame_idx: int) -> Optional[Dict[str, float]]:
        if not self.rows:
            return None
        if frame_idx < len(self.rows):
            return self.rows[frame_idx]
        return self.rows[-1]


class HorizonEstimator:
    """Estimates vanishing point and horizon line from road boundary lines in image."""

    def __init__(self, ema_alpha: float = 0.2) -> None:
        self.ema_alpha = float(ema_alpha)
        self.smoothed_vp_y: Optional[float] = None

    def estimate_horizon_y(self, frame_bgr: np.ndarray, mask: Optional[np.ndarray] = None) -> Optional[float]:
        """Estimate the vanishing point vertical coordinate (horizon_y) in pixels."""
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Focus on lower 75% of the frame
        roi_y1 = int(h * 0.25)
        gray_roi = gray[roi_y1:, :]
        
        edges = cv2.Canny(gray_roi, 50, 150)
        if mask is not None:
            m_roi = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)[roi_y1:, :]
            # Dilate mask slightly to capture edges along road boundary
            dilated = cv2.dilate(m_roi, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
            edges = cv2.bitwise_and(edges, edges, mask=dilated)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=int(h * 0.1), maxLineGap=20)
        if lines is None or len(lines) < 2:
            return self.smoothed_vp_y

        left_lines = []
        right_lines = []
        for line in lines:
            x1, y1, x2, y2 = [float(v) for v in np.asarray(line).reshape(-1)[:4]]
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) < 1e-4:
                continue
            slope = dy / dx
            angle_deg = math.degrees(math.atan(slope))
            # Plausible left lane line: negative angle between -75 and -20 deg
            if -75 <= angle_deg <= -20:
                left_lines.append((x1, y1 + roi_y1, x2, y2 + roi_y1))
            # Plausible right lane line: positive angle between 20 and 75 deg
            elif 20 <= angle_deg <= 75:
                right_lines.append((x1, y1 + roi_y1, x2, y2 + roi_y1))

        if not left_lines or not right_lines:
            return self.smoothed_vp_y

        # Average line coordinates
        def line_coeffs(lines_list):
            pts = np.array(lines_list, dtype=np.float32)
            fit = cv2.fitLine(pts.reshape(-1, 2), cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
            vx, vy, x0, y0 = float(fit[0]), float(fit[1]), float(fit[2]), float(fit[3])
            # Line equation: (y - y0) = (vy/vx)*(x - x0) -> vy*x - vx*y + (vx*y0 - vy*x0) = 0
            a = float(vy)
            b = float(-vx)
            c = float(vx * y0 - vy * x0)
            return a, b, c

        a1, b1, c1 = line_coeffs(left_lines)
        a2, b2, c2 = line_coeffs(right_lines)

        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-5:
            return self.smoothed_vp_y

        vp_y = (a2 * c1 - a1 * c2) / det
        # Vanishing point must be reasonably in the upper-middle of the image
        if 0.1 * h <= vp_y <= 0.8 * h:
            if self.smoothed_vp_y is None:
                self.smoothed_vp_y = float(vp_y)
            else:
                self.smoothed_vp_y = self.ema_alpha * float(vp_y) + (1.0 - self.ema_alpha) * self.smoothed_vp_y

        return self.smoothed_vp_y


class InversePerspectiveMapping:
    """Computes Bird's-Eye View (BEV) homography and extracts real-world road metrics."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        cam = cfg.get("camera", {})
        self.height_m = float(cam.get("height_m", 1.3))
        self.nominal_pitch_deg = float(cam.get("pitch_deg", 0.0))
        self.nominal_roll_deg = float(cam.get("roll_deg", 0.0))
        self.hfov_deg = float(cam.get("hfov_deg", 75.0))
        self.vfov_deg = float(cam.get("vfov_deg", 45.0))
        
        # BEV grid specs
        self.bev_width_m = float(cam.get("bev_width_m", 10.0))    # lateral span: [-5m, +5m]
        self.bev_height_m = float(cam.get("bev_height_m", 25.0))  # longitudinal span: [1m, 26m]
        self.bev_min_dist_m = float(cam.get("bev_min_dist_m", 1.5))
        self.bev_resolution = float(cam.get("bev_resolution", 0.05)) # 0.05m = 5cm/px
        
        self.bev_w_px = int(round(self.bev_width_m / self.bev_resolution))
        self.bev_h_px = int(round(self.bev_height_m / self.bev_resolution))
        
        self.filter = ComplementaryFilter(
            alpha=float(cam.get("filter_alpha", 0.98)),
            nominal_pitch_deg=self.nominal_pitch_deg,
            nominal_roll_deg=self.nominal_roll_deg,
        )
        self.horizon_est = HorizonEstimator(ema_alpha=0.25)
        self.current_pitch_deg = self.nominal_pitch_deg
        self.current_roll_deg = self.nominal_roll_deg

    def compute_intrinsics(self, img_w: int, img_h: int) -> np.ndarray:
        """Compute camera intrinsic matrix K."""
        fx = (img_w / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)
        fy = (img_h / 2.0) / math.tan(math.radians(self.vfov_deg) / 2.0)
        cx = img_w / 2.0
        cy = img_h / 2.0
        return np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    def compute_homography(self, pitch_rad: float, roll_rad: float, img_w: int, img_h: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute BEV-to-Image and Image-to-BEV homography matrices.

        Ground coordinate system:
        - X_g: lateral (right positive), in [-bev_width_m/2, bev_width_m/2]
        - Y_g: longitudinal (ahead positive), in [bev_min_dist_m, bev_min_dist_m + bev_height_m]
        - Ground elevation: 0.0 (camera is at height self.height_m).
        """
        K = self.compute_intrinsics(img_w, img_h)
        
        # Camera rotation matrix:
        # Pitch rotates about X axis (tilting camera down -> positive pitch)
        cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
        R_pitch = np.array([
            [1.0, 0.0, 0.0],
            [0.0, cp, -sp],
            [0.0, sp, cp]
        ], dtype=np.float64)
        
        # Roll rotates about Z axis
        cr, sr = math.cos(roll_rad), math.sin(roll_rad)
        R_roll = np.array([
            [cr, -sr, 0.0],
            [sr, cr, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        
        R = R_roll @ R_pitch
        
        # 4 control points on the ground plane in BEV space
        y_min = self.bev_min_dist_m
        y_max = self.bev_min_dist_m + self.bev_height_m
        x_min = -self.bev_width_m / 2.0
        x_max = self.bev_width_m / 2.0
        
        ground_pts = np.array([
            [x_min, self.height_m, y_min],
            [x_max, self.height_m, y_min],
            [x_max, self.height_m, y_max],
            [x_min, self.height_m, y_max],
        ], dtype=np.float64)
        
        # Project 3D ground points into camera image plane
        cam_pts = (R @ ground_pts.T).T
        proj_pts = (K @ cam_pts.T).T
        img_pts = proj_pts[:, :2] / np.maximum(1e-6, proj_pts[:, 2:3])
        
        # Corresponding pixel coordinates in BEV image (origin top-left, x right, y down)
        bev_px_pts = np.array([
            [0.0, self.bev_h_px - 1.0],
            [self.bev_w_px - 1.0, self.bev_h_px - 1.0],
            [self.bev_w_px - 1.0, 0.0],
            [0.0, 0.0],
        ], dtype=np.float32)
        
        img_px_pts = img_pts.astype(np.float32)
        
        H_bev2img = cv2.getPerspectiveTransform(bev_px_pts, img_px_pts)
        H_img2bev = np.linalg.inv(H_bev2img)
        H_img2bev /= (H_img2bev[2, 2] if abs(H_img2bev[2, 2]) > 1e-8 else 1.0)
        
        return H_img2bev, H_bev2img

    def warp_mask_to_bev(self, mask: np.ndarray, H_img2bev: np.ndarray) -> np.ndarray:
        """Warp drivable mask into Bird's-Eye View representation."""
        return cv2.warpPerspective(mask, H_img2bev, (self.bev_w_px, self.bev_h_px), flags=cv2.INTER_NEAREST)

    def extract_road_metrics(self, bev_mask: np.ndarray, distance_m: float = 6.0) -> Tuple[Optional[float], Optional[float]]:
        """Extract physical road width (m) and lateral offset (m) from BEV mask at distance_m ahead."""
        # Convert longitudinal distance into BEV row index
        # y_bev = 0 is y_max, y_bev = bev_h_px is y_min
        y_frac = (distance_m - self.bev_min_dist_m) / max(1e-4, self.bev_height_m)
        if not (0.0 <= y_frac <= 1.0):
            return None, None
            
        row_idx = int(round((1.0 - y_frac) * (self.bev_h_px - 1)))
        # Sample band of 5 rows around row_idx for stability
        r1 = max(0, row_idx - 2)
        r2 = min(self.bev_h_px, row_idx + 3)
        band = bev_mask[r1:r2]
        
        # Check active drivable columns
        col_active = (band > 0).any(axis=0)
        nz_indices = np.where(col_active)[0]
        if len(nz_indices) < 3:
            return None, None
            
        width_px = nz_indices[-1] - nz_indices[0] + 1
        width_m = float(width_px * self.bev_resolution)
        
        # Lateral offset from vehicle center line (center is bev_w_px / 2.0)
        center_px = float(np.mean(nz_indices))
        offset_px = center_px - (self.bev_w_px / 2.0)
        offset_m = float(offset_px * self.bev_resolution)
        
        return width_m, offset_m

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        mask: np.ndarray,
        dt: float = 0.04,
        imu_sample: Optional[Dict[str, float]] = None
    ) -> IPMResult:
        """Run self-calibrating IPM on the current frame."""
        h, w = frame_bgr.shape[:2]
        pitch_rad = math.radians(self.nominal_pitch_deg)
        roll_rad = math.radians(self.nominal_roll_deg)
        method = "nominal"
        
        # 1. Try IMU input if available
        if imu_sample is not None:
            pitch_rad, roll_rad = self.filter.update(
                imu_sample["ax"], imu_sample["ay"], imu_sample["az"],
                imu_sample["gx"], imu_sample["gy"], imu_sample["gz"],
                dt
            )
            method = "imu_fused"
        else:
            # 2. Try image horizon estimation
            vp_y = self.horizon_est.estimate_horizon_y(frame_bgr, mask)
            if vp_y is not None:
                K = self.compute_intrinsics(w, h)
                fy = K[1, 1]
                cy = K[1, 2]
                pitch_rad = math.atan2(vp_y - cy, fy)
                method = "horizon_only"
                
        self.current_pitch_deg = math.degrees(pitch_rad)
        self.current_roll_deg = math.degrees(roll_rad)
        
        try:
            H_img2bev, _ = self.compute_homography(pitch_rad, roll_rad, w, h)
            bev_mask = self.warp_mask_to_bev(mask, H_img2bev)
            width_m, offset_m = self.extract_road_metrics(bev_mask, distance_m=6.0)
            valid = (width_m is not None and width_m > 1.0)
            return IPMResult(
                bev_mask=bev_mask,
                homography=H_img2bev,
                pitch_deg=self.current_pitch_deg,
                roll_deg=self.current_roll_deg,
                road_width_m=width_m,
                lateral_offset_m=offset_m,
                method=method,
                valid=valid
            )
        except Exception:
            return IPMResult(
                bev_mask=None,
                homography=None,
                pitch_deg=self.current_pitch_deg,
                roll_deg=self.current_roll_deg,
                road_width_m=None,
                lateral_offset_m=None,
                method="failed",
                valid=False
            )
