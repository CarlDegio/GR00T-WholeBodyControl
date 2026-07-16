#!/usr/bin/env python3
"""Visualize G1 MID-360 points and REASEN rays in the robot body frame."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import json
import sys
import threading
import time
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import view_mid360_open3d as mid360  # noqa: E402


DEFAULT_ESTIMATOR = Path(
    "/home/user/Project/REASAN/training/ray_predictor/ray_predictor/"
    "runs/envelope_5x_history10_20260716/ray_predictor_g1_10f.onnx"
)


def rpy_matrix_deg(rpy_deg: tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad(np.asarray(rpy_deg, dtype=np.float64))
    return mid360.rpy_to_rotation_matrix(float(roll), float(pitch), float(yaw))


def transform_sensor_to_body(
    points_sensor: np.ndarray,
    translation: tuple[float, float, float],
    rpy_deg: tuple[float, float, float],
) -> np.ndarray:
    """Apply torso_link <- Mid360 rigid transform to row-vector points."""
    if points_sensor.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    rotation = rpy_matrix_deg(rpy_deg)
    result = points_sensor.astype(np.float64, copy=False) @ rotation.T
    result += np.asarray(translation, dtype=np.float64)
    return result.astype(np.float32)


def filter_sensor_min_range(points_sensor: np.ndarray, min_range: float) -> np.ndarray:
    """Remove the MID-360 near field before applying any sensor extrinsics."""
    if points_sensor.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    points = np.asarray(points_sensor, dtype=np.float32)
    distance_sensor = np.linalg.norm(points, axis=1)
    valid = np.isfinite(points).all(axis=1) & np.isfinite(distance_sensor) & (distance_sensor >= min_range)
    return points[valid]


def filter_ground_plane(
    points_body: np.ndarray,
    *,
    distance_threshold: float,
    candidate_max_z: float,
    max_tilt_deg: float,
    iterations: int = 64,
) -> tuple[np.ndarray, int]:
    """Remove a near-horizontal RANSAC plane from points below the torso."""
    if len(points_body) < 3:
        return points_body, 0
    candidate_indices = np.flatnonzero(np.isfinite(points_body).all(axis=1) & (points_body[:, 2] <= candidate_max_z))
    if candidate_indices.size < 50:
        return points_body, 0

    candidates = points_body[candidate_indices].astype(np.float64, copy=False)
    rng = np.random.default_rng(0)
    best_mask = None
    best_count = 0
    min_vertical_component = float(np.cos(np.deg2rad(max_tilt_deg)))
    for _ in range(iterations):
        sample = candidates[rng.choice(len(candidates), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm < 1.0e-8:
            continue
        normal /= norm
        if abs(float(normal[2])) < min_vertical_component:
            continue
        offset = -float(np.dot(normal, sample[0]))
        inliers = np.abs(candidates @ normal + offset) <= distance_threshold
        count = int(np.count_nonzero(inliers))
        if count > best_count:
            best_mask = inliers
            best_count = count

    if best_mask is None or best_count < 50:
        return points_body, 0
    keep = np.ones(len(points_body), dtype=bool)
    keep[candidate_indices[best_mask]] = False
    return points_body[keep], best_count


def pointcloud_to_spherical_grid(
    points_body: np.ndarray,
    *,
    phi_range: tuple[float, float],
    theta_range: tuple[float, float],
    phi_res_deg: float,
    theta_res_deg: float,
    max_range: float,
) -> np.ndarray:
    """Match REASEN's nearest-range spherical grid convention."""
    phi_bins = int(round((phi_range[1] - phi_range[0]) / phi_res_deg))
    theta_bins = int(round((theta_range[1] - theta_range[0]) / theta_res_deg))
    if phi_bins != 180 or theta_bins != 30:
        raise ValueError(f"REASEN Estimator expects a [30,180] grid, got [{theta_bins},{phi_bins}]")

    grid = np.full((theta_bins, phi_bins), max_range, dtype=np.float32)
    if points_body.size == 0:
        return grid

    xyz = np.asarray(points_body, dtype=np.float32)
    distance = np.linalg.norm(xyz, axis=1)
    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(distance) & (distance > 1.0e-6) & (distance < max_range)
    if not np.any(finite):
        return grid

    xyz = xyz[finite]
    distance = distance[finite]
    phi = np.rad2deg(np.arctan2(xyz[:, 1], xyz[:, 0]))
    theta = np.rad2deg(np.arcsin(np.clip(xyz[:, 2] / distance, -1.0, 1.0)))
    valid = (
        (phi >= phi_range[0])
        & (phi <= phi_range[1])
        & (theta >= theta_range[0])
        & (theta <= theta_range[1])
    )
    if not np.any(valid):
        return grid

    phi_idx = np.floor((phi[valid] - phi_range[0]) / phi_res_deg).astype(np.int64)
    theta_idx = np.floor((theta[valid] - theta_range[0]) / theta_res_deg).astype(np.int64)
    np.clip(phi_idx, 0, phi_bins - 1, out=phi_idx)
    np.clip(theta_idx, 0, theta_bins - 1, out=theta_idx)
    np.minimum.at(grid, (theta_idx, phi_idx), distance[valid])
    return grid


def direct_actor_profile(grid: np.ndarray, max_range: float) -> np.ndarray:
    """Nearest observed point per azimuth; diagnostic, not Estimator ground truth."""
    return np.clip(np.min(grid, axis=0) / max_range, 0.0, 1.0).astype(np.float32)


def rays_to_points(rays_normalized: np.ndarray, max_range: float, z: float = 0.0) -> np.ndarray:
    rays = np.clip(np.asarray(rays_normalized).reshape(-1), 0.0, 1.0)
    if rays.size != 180:
        raise ValueError(f"Expected 180 ActorRay values, got {rays.size}")
    azimuth = np.deg2rad(-179.0 + 2.0 * np.arange(180, dtype=np.float32))
    distance = rays * max_range
    return np.column_stack((distance * np.cos(azimuth), distance * np.sin(azimuth), np.full(180, z)))


def make_radial_lines(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = len(points)
    vertices = np.empty((count * 2, 3), dtype=np.float64)
    vertices[0::2] = 0.0
    vertices[1::2] = points
    lines = np.column_stack((np.arange(0, count * 2, 2), np.arange(1, count * 2, 2))).astype(np.int32)
    return vertices, lines


class ImuFeatureReceiver:
    """Receive the two Estimator IMU features: projected gravity and body gyro."""

    def __init__(self, topic: str) -> None:
        mid360.import_unitree_sdk2()
        self._lock = threading.Lock()
        self._gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self._gyro = np.zeros(3, dtype=np.float32)
        self._received_at: float | None = None
        self._subscriber = mid360.ChannelSubscriber(topic, mid360.HGIMUState_)
        self._subscriber.Init(self._on_imu, 1)

    def _on_imu(self, msg: Any) -> None:
        rpy = np.asarray(getattr(msg, "rpy", (0.0, 0.0, 0.0)), dtype=np.float64)
        gyro = np.asarray(getattr(msg, "gyroscope", (0.0, 0.0, 0.0)), dtype=np.float32)
        if rpy.shape[0] < 3 or gyro.shape[0] < 3 or not np.isfinite(rpy[:3]).all() or not np.isfinite(gyro[:3]).all():
            return
        rotation_w_from_b = mid360.rpy_to_rotation_matrix(*map(float, rpy[:3]))
        gravity_b = rotation_w_from_b.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
        with self._lock:
            self._gravity = gravity_b.astype(np.float32)
            self._gyro = gyro[:3].copy()
            self._received_at = time.monotonic()

    def latest(self) -> np.ndarray:
        with self._lock:
            return np.concatenate((self._gravity, self._gyro)).astype(np.float32)

    def snapshot(self) -> tuple[np.ndarray, bool, float | None]:
        with self._lock:
            features = np.concatenate((self._gravity, self._gyro)).astype(np.float32)
            received_at = self._received_at
        age = None if received_at is None else max(0.0, time.monotonic() - received_at)
        return features, received_at is not None, age

    def close(self) -> None:
        self._subscriber.Close()


class OnnxRayEstimator:
    def __init__(self, model_path: Path) -> None:
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise SystemExit("Estimator mode requires onnxruntime: python3 -m pip install onnxruntime-gpu") from exc

        available = ort.get_available_providers()
        preferred = [name for name in ("CUDAExecutionProvider", "CPUExecutionProvider") if name in available]
        self.session = ort.InferenceSession(str(model_path), providers=preferred or available)
        inputs = self.session.get_inputs()
        grid_inputs = [item for item in inputs if len(item.shape) == 4]
        imu_inputs = [item for item in inputs if len(item.shape) == 3]
        if len(grid_inputs) != 1 or len(imu_inputs) != 1:
            raise ValueError(f"Expected one rank-4 grid input and one rank-3 IMU input, got {[i.shape for i in inputs]}")
        self.grid_name = grid_inputs[0].name
        self.imu_name = imu_inputs[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.history_frames = int(grid_inputs[0].shape[1])
        if self.history_frames <= 0:
            raise ValueError(f"Estimator ONNX has invalid history dimension: {grid_inputs[0].shape}")
        self.grid_history: deque[np.ndarray] = deque(maxlen=self.history_frames)
        self.imu_history: deque[np.ndarray] = deque(maxlen=self.history_frames)
        print(
            f"[estimator] {model_path} | H={self.history_frames} | "
            f"{self.grid_name}{grid_inputs[0].shape}, {self.imu_name}{imu_inputs[0].shape} -> {self.output_name}"
        )

    def infer(self, grid_normalized: np.ndarray, imu: np.ndarray) -> np.ndarray:
        if not self.grid_history:
            for _ in range(self.history_frames):
                self.grid_history.append(grid_normalized.copy())
                self.imu_history.append(imu.copy())
        else:
            self.grid_history.append(grid_normalized.copy())
            self.imu_history.append(imu.copy())
        grid_batch = np.stack(self.grid_history, axis=0)[None].astype(np.float32)
        imu_batch = np.stack(self.imu_history, axis=0)[None].astype(np.float32)
        output = self.session.run([self.output_name], {self.grid_name: grid_batch, self.imu_name: imu_batch})[0]
        return np.clip(np.asarray(output).reshape(-1), 0.0, 1.0).astype(np.float32)


class RayMedianFilter:
    """Per-azimuth median over a short history of 180-D ray profiles."""

    def __init__(self, window: int) -> None:
        if window < 1 or window % 2 == 0:
            raise ValueError("Ray median window must be a positive odd integer")
        self.history: deque[np.ndarray] = deque(maxlen=window)

    def update(self, rays: np.ndarray) -> np.ndarray:
        values = np.asarray(rays, dtype=np.float32).reshape(180)
        self.history.append(values.copy())
        return np.median(np.stack(self.history, axis=0), axis=0).astype(np.float32)


def build_actor_ray_message(
    rays_normalized: np.ndarray,
    *,
    sequence: int,
    max_range: float,
    source: str,
    imu_features: np.ndarray | None = None,
    imu_valid: bool = False,
    imu_age_s: float | None = None,
) -> dict[str, Any]:
    rays = np.clip(np.asarray(rays_normalized, dtype=np.float32).reshape(-1), 0.0, 1.0)
    if rays.size != 180:
        raise ValueError(f"Expected 180 ActorRay values, got {rays.size}")
    message = {
        "type": "reasan_actor_ray",
        "version": 1,
        "timestamp_ns": time.time_ns(),
        "sequence": int(sequence),
        "frame_id": "torso_link",
        "source": source,
        "count": 180,
        "angle_min_deg": -179.0,
        "angle_max_deg": 179.0,
        "angle_increment_deg": 2.0,
        "range_max_m": float(max_range),
        "normalized": rays.tolist(),
        "ranges_m": (rays * max_range).tolist(),
        "imu_valid": bool(imu_valid),
    }
    if imu_features is not None:
        imu = np.asarray(imu_features, dtype=np.float32).reshape(-1)
        if imu.size != 6 or not np.isfinite(imu).all():
            raise ValueError("IMU features must be [projected_gravity(3), angular_velocity(3)]")
        message["projected_gravity"] = imu[:3].tolist()
        message["angular_velocity"] = imu[3:].tolist()
    if imu_age_s is not None:
        message["imu_age_s"] = float(imu_age_s)
    return message


class ActorRayZmqPublisher:
    def __init__(self, endpoint: str) -> None:
        try:
            import zmq
        except Exception as exc:
            raise SystemExit("ZMQ publishing requires pyzmq: python3 -m pip install pyzmq") from exc
        self.zmq = zmq
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.SNDHWM, 1)
        self.socket.bind(endpoint)
        self.endpoint = endpoint
        print(f"[zmq] publishing 180-D ActorRay JSON on {endpoint}")

    def publish(self, message: dict[str, Any]) -> bool:
        try:
            self.socket.send_string(json.dumps(message, separators=(",", ":")), flags=self.zmq.NOBLOCK)
            return True
        except self.zmq.Again:
            return False

    def close(self) -> None:
        self.socket.close(linger=0)


class ReasanOccupancyViewer:
    """OpenCV bird's-eye view matching play_g1_filter.py's occupancy layout."""

    PANEL_SIZE = 500
    CENTER = (250, 260)
    DRAW_RADIUS_PX = 205
    WINDOW_NAME = "G1 REASEN | physical points (m) vs normalized ActorRay"

    def __init__(
        self,
        max_range: float,
        *,
        create_window: bool = True,
        window_left: int = 920,
    ) -> None:
        try:
            import cv2
        except Exception as exc:
            raise SystemExit("2-D visualization requires OpenCV: python3 -m pip install opencv-python") from exc
        self.cv2 = cv2
        self.max_range = max_range
        self.window_created = create_window
        if create_window:
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.WINDOW_NAME, self.PANEL_SIZE * 2, self.PANEL_SIZE)
            cv2.moveWindow(self.WINDOW_NAME, window_left, 30)

    def _base_panel(self, title: str, radius_label: str) -> np.ndarray:
        cv2 = self.cv2
        panel = np.full((self.PANEL_SIZE, self.PANEL_SIZE, 3), 20, dtype=np.uint8)
        for fraction in (0.25, 0.5, 0.75, 1.0):
            cv2.circle(panel, self.CENTER, int(self.DRAW_RADIUS_PX * fraction), (65, 65, 65), 1, cv2.LINE_AA)
        cv2.line(
            panel,
            (self.CENTER[0], self.CENTER[1] - self.DRAW_RADIUS_PX),
            (self.CENTER[0], self.CENTER[1] + self.DRAW_RADIUS_PX),
            (50, 50, 50),
            1,
        )
        cv2.line(
            panel,
            (self.CENTER[0] - self.DRAW_RADIUS_PX, self.CENTER[1]),
            (self.CENTER[0] + self.DRAW_RADIUS_PX, self.CENTER[1]),
            (50, 50, 50),
            1,
        )
        cv2.putText(panel, title, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(panel, radius_label, (14, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 170, 170), 1, cv2.LINE_AA)
        cv2.circle(panel, self.CENTER, 7, (255, 210, 80), -1, cv2.LINE_AA)
        cv2.arrowedLine(
            panel,
            self.CENTER,
            (self.CENTER[0], self.CENTER[1] - 30),
            (255, 210, 80),
            2,
            cv2.LINE_AA,
            tipLength=0.3,
        )
        cv2.putText(
            panel,
            "+X",
            (self.CENTER[0] + 8, self.CENTER[1] - 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 210, 80),
            1,
        )
        return panel

    def _xy_to_pixels(self, xy: np.ndarray, physical_radius: float) -> tuple[np.ndarray, np.ndarray]:
        scale = self.DRAW_RADIUS_PX / physical_radius
        # Same convention as G1 Play: body +X=image up and body +Y=image left.
        px = np.rint(self.CENTER[0] - xy[:, 1] * scale).astype(np.int32)
        py = np.rint(self.CENTER[1] - xy[:, 0] * scale).astype(np.int32)
        return px, py

    def physical_panel(self, points_body: np.ndarray) -> np.ndarray:
        cv2 = self.cv2
        panel = self._base_panel("Physical occupancy from MID-360", f"metric radius: {self.max_range:.1f} m")
        if points_body.size:
            finite = np.isfinite(points_body).all(axis=1)
            distance = np.linalg.norm(points_body, axis=1)
            inside = finite & (distance < self.max_range)
            xy = points_body[inside, :2]
            if len(xy):
                px, py = self._xy_to_pixels(xy, self.max_range)
                valid = (px >= 0) & (px < self.PANEL_SIZE) & (py >= 0) & (py < self.PANEL_SIZE)
                occupancy = np.zeros_like(panel)
                occupancy[py[valid], px[valid]] = (0, 95, 255)
                occupancy = cv2.dilate(occupancy, np.ones((3, 3), dtype=np.uint8), iterations=1)
                mask = np.any(occupancy != 0, axis=-1)
                panel[mask] = occupancy[mask]
        cv2.putText(
            panel,
            "orange: raw MID-360 points within clipping radius",
            (14, self.PANEL_SIZE - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (0, 165, 255),
            1,
            cv2.LINE_AA,
        )
        return panel

    def actor_panel(self, actor_rays: np.ndarray, direct_rays: np.ndarray | None) -> np.ndarray:
        cv2 = self.cv2
        rays = np.clip(np.asarray(actor_rays).reshape(-1), 0.0, 1.0)
        panel = self._base_panel("Normalized ActorRay received by Filter", f"normalized radius: 1.0 == {self.max_range:.1f} m")
        azimuth = np.deg2rad(-179.0 + 2.0 * np.arange(180, dtype=np.float32))
        directions = np.column_stack((np.cos(azimuth), np.sin(azimuth)))

        if direct_rays is not None:
            direct_endpoints = directions * np.asarray(direct_rays).reshape(-1, 1)
            dx, dy = self._xy_to_pixels(direct_endpoints, 1.0)
            for index in range(180):
                cv2.circle(panel, (int(dx[index]), int(dy[index])), 1, (80, 190, 80), -1, cv2.LINE_AA)

        endpoints = directions * rays[:, None]
        px, py = self._xy_to_pixels(endpoints, 1.0)
        for index in range(180):
            endpoint = (int(px[index]), int(py[index]))
            cv2.line(panel, self.CENTER, endpoint, (55, 72, 72), 1, cv2.LINE_AA)
            color = (60, 80, 235) if rays[index] < 0.995 else (110, 110, 110)
            cv2.circle(panel, endpoint, 2, color, -1, cv2.LINE_AA)
        cv2.putText(
            panel,
            f"min={rays.min():.3f}  mean={rays.mean():.3f}  max={rays.max():.3f}",
            (14, self.PANEL_SIZE - 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
        legend = "red: active ActorRay; green: raw direct profile" if direct_rays is not None else "red: occupied before max range; gray: no hit"
        cv2.putText(
            panel,
            legend,
            (14, self.PANEL_SIZE - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (80, 130, 235),
            1,
            cv2.LINE_AA,
        )
        return panel

    def compose(self, points_body: np.ndarray, actor_rays: np.ndarray, direct_rays: np.ndarray | None) -> np.ndarray:
        return np.concatenate((self.physical_panel(points_body), self.actor_panel(actor_rays, direct_rays)), axis=1)

    def render(self, points_body: np.ndarray, actor_rays: np.ndarray, direct_rays: np.ndarray | None) -> bool:
        frame = self.compose(points_body, actor_rays, direct_rays)
        if not self.window_created:
            return True
        self.cv2.imshow(self.WINDOW_NAME, frame)
        return self.cv2.waitKey(1) & 0xFF != ord("q")

    def close(self) -> None:
        if self.window_created:
            self.cv2.destroyWindow(self.WINDOW_NAME)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Visualize MID-360 point cloud and REASEN ActorRay/Estimator output.")
    result.add_argument("--interface", "-i", help="NIC connected to G1, e.g. enp3s0")
    result.add_argument("--domain-id", type=int, default=0)
    result.add_argument("--topic", default=mid360.DEFAULT_TOPIC)
    result.add_argument("--imu-topic", default=mid360.DEFAULT_IMU_TOPIC)
    result.add_argument("--ray-source", choices=("direct", "estimator", "both"), default="both")
    result.add_argument("--estimator", type=Path, default=DEFAULT_ESTIMATOR)
    result.add_argument("--min-range", type=float, default=0.3, help="Ignore points closer than this distance in meters")
    result.add_argument("--max-range", type=float, default=3.0)
    result.add_argument("--filter-ground", dest="filter_ground", action="store_true", default=True)
    result.add_argument("--no-filter-ground", dest="filter_ground", action="store_false")
    result.add_argument("--ground-distance", type=float, default=0.04, help="RANSAC ground-plane thickness in meters")
    result.add_argument("--ground-candidate-max-z", type=float, default=-0.25, help="Only fit ground using body-frame points below this Z")
    result.add_argument("--ground-max-tilt-deg", type=float, default=25.0)
    result.add_argument(
        "--ray-median-window",
        type=int,
        default=3,
        help="Odd number of Direct ActorRay frames used for per-direction median filtering",
    )
    result.add_argument(
        "--zmq-endpoint",
        default="tcp://*:5562",
        help="ZMQ PUB endpoint for the active 180-D ActorRay; empty string disables publishing",
    )
    result.add_argument("--phi-range", nargs=2, type=float, default=(-180.0, 180.0), metavar=("MIN", "MAX"))
    result.add_argument("--theta-range", nargs=2, type=float, default=(-55.0, 5.0), metavar=("MIN", "MAX"))
    result.add_argument("--phi-res-deg", type=float, default=2.0)
    result.add_argument("--theta-res-deg", type=float, default=2.0)
    result.add_argument("--lidar-offset", nargs=3, type=float, default=(0.0002835, 0.00003, 0.41618), metavar=("X", "Y", "Z"))
    result.add_argument("--lidar-rpy-deg", nargs=3, type=float, default=(0.0, 177.674, 179.995), metavar=("R", "P", "Y"))
    result.add_argument("--stride", type=int, default=1)
    result.add_argument("--point-size", type=float, default=2.0)
    result.add_argument("--ray-line-width", type=float, default=2.0)
    result.add_argument("--width", type=int, default=900)
    result.add_argument("--height", type=int, default=700)
    result.add_argument("--show-rays-3d", action="store_true", help="Overlay ray lines in Open3D; off keeps raw 3-D points clean")
    result.add_argument("--no-cv", action="store_true", help="Disable the Play-style dynamic 2-D occupancy window")
    result.add_argument("--update-hz", type=float, default=20.0)
    result.add_argument("--print-every", type=float, default=1.0)
    result.add_argument("--self-test", action="store_true")
    return result


def cloud_receiver_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        topic=args.topic,
        stride=max(1, args.stride),
        max_range=0.0,
        intensity_max=255.0,
        color_mode="white",
        pool_seconds=0.0,
        accumulate_frames=1,
        max_points=0,
        once=False,
        print_every=args.print_every,
    )


def run(args: argparse.Namespace) -> None:
    try:
        import open3d as o3d
    except Exception as exc:
        raise SystemExit("Install visualization dependencies: python3 -m pip install open3d") from exc

    estimator = None
    if args.ray_source in ("estimator", "both"):
        if not args.estimator.is_file():
            raise SystemExit(f"Estimator ONNX not found: {args.estimator}")
        estimator = OnnxRayEstimator(args.estimator)
    ray_median_filter = RayMedianFilter(args.ray_median_window)
    ray_publisher = ActorRayZmqPublisher(args.zmq_endpoint) if args.zmq_endpoint else None

    mid360.init_dds(args.domain_id, args.interface)
    receiver = mid360.Mid360CloudReceiver(cloud_receiver_args(args))
    imu_receiver = ImuFeatureReceiver(args.imu_topic) if estimator is not None or ray_publisher is not None else None

    vis = o3d.visualization.Visualizer()
    vis.create_window("G1 MID-360 raw 3-D point cloud", width=args.width, height=args.height, left=10, top=30, visible=True)
    option = vis.get_render_option()
    option.background_color = np.array([0.02, 0.02, 0.02])
    option.point_size = args.point_size
    option.line_width = args.ray_line_width

    cloud = o3d.geometry.PointCloud()
    direct_lines = o3d.geometry.LineSet()
    estimator_lines = o3d.geometry.LineSet()
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis.add_geometry(cloud)
    vis.add_geometry(direct_lines)
    vis.add_geometry(estimator_lines)
    vis.add_geometry(axis)
    mid360.set_default_view(vis, 0.65)
    occupancy_viewer = (
        None
        if args.no_cv
        else ReasanOccupancyViewer(args.max_range, window_left=args.width + 30)
    )

    print("[colors] point cloud=white, direct profile=green, Estimator ActorRay=magenta")
    last_seq = 0
    next_update = 0.0
    update_period = 1.0 / max(args.update_hz, 1.0)
    try:
        while vis.poll_events():
            now = time.monotonic()
            frame = receiver.latest_after(last_seq)
            if frame is not None and now >= next_update:
                points_sensor = filter_sensor_min_range(frame.points, args.min_range)
                points_body = transform_sensor_to_body(points_sensor, tuple(args.lidar_offset), tuple(args.lidar_rpy_deg))
                ground_points_removed = 0
                if args.filter_ground:
                    points_body, ground_points_removed = filter_ground_plane(
                        points_body,
                        distance_threshold=args.ground_distance,
                        candidate_max_z=args.ground_candidate_max_z,
                        max_tilt_deg=args.ground_max_tilt_deg,
                    )
                grid = pointcloud_to_spherical_grid(
                    points_body,
                    phi_range=tuple(args.phi_range),
                    theta_range=tuple(args.theta_range),
                    phi_res_deg=args.phi_res_deg,
                    theta_res_deg=args.theta_res_deg,
                    max_range=args.max_range,
                )
                direct_raw = direct_actor_profile(grid, args.max_range)
                direct = ray_median_filter.update(direct_raw)
                predicted = None
                imu_features = None
                imu_valid = False
                imu_age_s = None
                if imu_receiver is not None:
                    imu_features, imu_valid, imu_age_s = imu_receiver.snapshot()
                if estimator is not None and imu_receiver is not None:
                    predicted = estimator.infer(grid / args.max_range, imu_features)
                actor_for_filter = predicted if predicted is not None else direct
                if ray_publisher is not None:
                    ray_publisher.publish(
                        build_actor_ray_message(
                            actor_for_filter,
                            sequence=frame.seq,
                            max_range=args.max_range,
                            source="estimator" if predicted is not None else "direct_median",
                            imu_features=imu_features,
                            imu_valid=imu_valid,
                            imu_age_s=imu_age_s,
                        )
                    )

                cloud.points = o3d.utility.Vector3dVector(points_body.astype(np.float64))
                cloud.colors = o3d.utility.Vector3dVector(np.ones_like(points_body, dtype=np.float64))
                vis.update_geometry(cloud)

                if args.show_rays_3d and args.ray_source in ("direct", "both"):
                    vertices, lines = make_radial_lines(rays_to_points(direct, args.max_range, z=0.02))
                    direct_lines.points = o3d.utility.Vector3dVector(vertices)
                    direct_lines.lines = o3d.utility.Vector2iVector(lines)
                    direct_lines.colors = o3d.utility.Vector3dVector(np.tile([0.1, 1.0, 0.1], (180, 1)))
                    vis.update_geometry(direct_lines)

                if args.show_rays_3d and predicted is not None:
                    vertices, lines = make_radial_lines(rays_to_points(predicted, args.max_range, z=0.06))
                    estimator_lines.points = o3d.utility.Vector3dVector(vertices)
                    estimator_lines.lines = o3d.utility.Vector2iVector(lines)
                    estimator_lines.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.1, 0.8], (180, 1)))
                    vis.update_geometry(estimator_lines)

                if occupancy_viewer is not None:
                    comparison = direct_raw if predicted is None or args.ray_source == "both" else None
                    if not occupancy_viewer.render(points_body, actor_for_filter, comparison):
                        break

                occupied = int(np.count_nonzero(grid < args.max_range))
                if args.print_every > 0 and frame.seq % max(1, round(args.update_hz * args.print_every)) == 0:
                    print(
                        f"[frame {frame.seq}] kept_points={len(points_body)} "
                        f"ground_removed={ground_points_removed} occupied_bins={occupied}/5400"
                    )
                last_seq = frame.seq
                next_update = now + update_period

            vis.update_renderer()
            time.sleep(0.002)
    except KeyboardInterrupt:
        pass
    finally:
        vis.destroy_window()
        if occupancy_viewer is not None:
            occupancy_viewer.close()
        if imu_receiver is not None:
            imu_receiver.close()
        if ray_publisher is not None:
            ray_publisher.close()
        receiver.close()


def self_test() -> None:
    phi = np.deg2rad(np.array([-179.0, -1.0, 1.0, 179.0]))
    points = np.column_stack((np.cos(phi), np.sin(phi), np.zeros_like(phi))).astype(np.float32)
    grid = pointcloud_to_spherical_grid(
        points,
        phi_range=(-180.0, 180.0),
        theta_range=(-55.0, 5.0),
        phi_res_deg=2.0,
        theta_res_deg=2.0,
        max_range=3.0,
    )
    assert grid.shape == (30, 180)
    profile = direct_actor_profile(grid, 3.0)
    assert profile.shape == (180,)
    assert np.count_nonzero(profile < 1.0) == 4
    sensor_points = np.array([[0.2, 0.0, 0.0], [0.3, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    filtered = filter_sensor_min_range(sensor_points, 0.3)
    assert filtered.shape == (2, 3)
    assert np.allclose(filtered[:, 0], [0.3, 1.0])
    gx, gy = np.meshgrid(np.linspace(-2.0, 2.0, 30), np.linspace(-2.0, 2.0, 30))
    ground = np.column_stack((gx.ravel(), gy.ravel(), np.full(gx.size, -0.8))).astype(np.float32)
    obstacle = np.array([[1.0, 0.0, -0.4], [1.0, 0.0, 0.0], [1.0, 0.0, 0.4]], dtype=np.float32)
    without_ground, removed = filter_ground_plane(
        np.concatenate((ground, obstacle), axis=0),
        distance_threshold=0.04,
        candidate_max_z=-0.25,
        max_tilt_deg=25.0,
    )
    assert removed == len(ground)
    assert len(without_ground) == len(obstacle)
    ray_points = rays_to_points(profile, 3.0)
    assert ray_points.shape == (180, 3)
    median_filter = RayMedianFilter(3)
    baseline = np.ones(180, dtype=np.float32)
    assert np.allclose(median_filter.update(baseline), 1.0)
    impulse = baseline.copy()
    impulse[20] = 0.1
    assert np.isclose(median_filter.update(impulse)[20], 0.55)
    assert np.isclose(median_filter.update(baseline)[20], 1.0)
    imu = np.array([0.0, 0.0, -1.0, 0.1, 0.2, 0.3], dtype=np.float32)
    message = build_actor_ray_message(
        baseline,
        sequence=7,
        max_range=3.0,
        source="direct_median",
        imu_features=imu,
        imu_valid=True,
        imu_age_s=0.01,
    )
    assert message["count"] == 180
    assert len(message["normalized"]) == 180
    assert len(message["ranges_m"]) == 180
    assert message["ranges_m"][0] == 3.0
    assert message["imu_valid"] is True
    assert np.allclose(message["projected_gravity"], [0.0, 0.0, -1.0])
    assert np.allclose(message["angular_velocity"], [0.1, 0.2, 0.3])
    json.dumps(message)
    print("self-test passed: grid=[30,180], ActorRay=[180]")


def main() -> int:
    args = parser().parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.min_range < 0 or args.max_range <= args.min_range:
        raise SystemExit("Require 0 <= --min-range < --max-range")
    if args.ray_median_window < 1 or args.ray_median_window % 2 == 0:
        raise SystemExit("--ray-median-window must be a positive odd integer")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
