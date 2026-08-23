"""SensorGateway ingress and NavDP HTTP request helpers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import logging
import math
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np

from gear_sonic.camera.constants import PRODUCTION_JPEG_QUALITY
from gear_sonic.runtime.gateway.rgbd import materialize_rgbd
from gear_sonic.runtime.gateway.sensor_client import (
    MaterializedSnapshot,
    SensorGatewayClient,
)
from gear_sonic.runtime.gateway.polling_ingress import PollingSensorIngress
from gear_sonic.runtime.profile import load_component_config
from gear_sonic.utils.inference.navdp.control import LatestMessageWorker
from gear_sonic.utils.inference.navdp.navigation import Pose2D, update_slam_map
from gear_sonic.utils.inference.navdp.visualization import filter_livox_points
from gear_sonic.utils.math3d.quaternions import yaw_from_quaternion_xyzw

LOGGER = logging.getLogger("sonic.navdp")


@dataclass
class NavDPPlannerConfig:
    sensor_gateway_poll_hz: float
    sensor_gateway_request_timeout_ms: int
    sensor_gateway_max_age_ms: float
    sensor_gateway_max_skew_ms: float
    rgb_stream: str
    depth_stream: str
    control_hz: float
    mpc_hz: float
    mpc_result_timeout_s: float
    heading_preview_s: float
    goal_tolerance_m: float
    stop_threshold: float
    odometry_timeout_s: float
    trajectory_timeout_s: float
    request_timeout_s: float
    heading_angular_speed_rad_s: float
    heading_fine_angular_speed_rad_s: float
    heading_slowdown_angle_rad: float
    heading_goal_tolerance_rad: float
    heading_orientation_timeout_s: float
    profile: str = ""
    overlay: tuple[str, ...] = ()


def load_navdp_planner_config(
    profile: str = "", overlays: tuple[str, ...] = ()
) -> NavDPPlannerConfig:
    return load_component_config(
        NavDPPlannerConfig,
        "navdp",
        profile or None,
        overlays=overlays,
        ignored_fields=("root", "checkpoint"),
    )


def _encode_navdp_frames(rgb_bgr: np.ndarray, depth_m: np.ndarray) -> tuple[bytes, bytes]:
    import cv2

    depth = np.asarray(depth_m, dtype=np.float32).copy()
    depth[(depth < 0.1) | (depth > 5.0) | ~np.isfinite(depth)] = 0.0
    ok_rgb, rgb_encoded = cv2.imencode(
        ".jpg", rgb_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), PRODUCTION_JPEG_QUALITY]
    )
    ok_depth, depth_png = cv2.imencode(".png", np.rint(depth * 10000.0).astype(np.uint16))
    if not ok_rgb or not ok_depth:
        raise RuntimeError("failed to encode NavDP RGB-D")
    return rgb_encoded.tobytes(), depth_png.tobytes()


@dataclass(frozen=True)
class GatewayCameraFrame:
    rgb: np.ndarray
    depth_m: np.ndarray
    camera_info: Mapping[str, Any]
    source_timestamp_s: float


def _update_odometry_state(
    sensors: _SharedSensors,
    values: Sequence[float],
    *,
    source_timestamp_s: float,
    received_monotonic_s: float,
) -> None:
    state = np.asarray(values, dtype=np.float64).reshape(-1)
    if state.shape != (13,):
        raise ValueError(f"odometry vector must have 13 values, got {state.shape}")
    pose = Pose2D(
        float(state[0]),
        float(state[1]),
        yaw_from_quaternion_xyzw(state[3:7]),
    )
    timestamp_s = float(source_timestamp_s)
    if timestamp_s <= 0.0:
        timestamp_s = time.time()
    with sensors.lock:
        sensors.pose = pose
        sensors.pose_time = float(received_monotonic_s)
        sensors.pose_history.append((timestamp_s, pose))
        sample = np.asarray([[pose.x, pose.y]], dtype=np.float32)
        if (
            not len(sensors.robot_history)
            or np.linalg.norm(sample[0] - sensors.robot_history[-1]) >= 0.02
        ):
            sensors.robot_history = np.concatenate(
                (sensors.robot_history, sample), axis=0
            )[-5000:]


def _update_lidar_state(
    sensors: _SharedSensors,
    raw_points: np.ndarray,
) -> None:
    points = _remove_ground(filter_livox_points(raw_points))
    with sensors.lock:
        sensors.points = points


def _update_slam_cloud_state(
    sensors: _SharedSensors,
    xyz: np.ndarray,
    *,
    received_monotonic_s: float | None,
    reset_after_s: float | None = None,
) -> None:
    values = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    timestamp_s = (
        time.monotonic()
        if received_monotonic_s is None
        else float(received_monotonic_s)
    )
    with sensors.lock:
        pose = sensors.pose
        previous = sensors.slam_map_xy
        reset_map = bool(
            reset_after_s is not None
            and sensors.slam_map_time > 0.0
            and timestamp_s - sensors.slam_map_time > float(reset_after_s)
        )
        if reset_map:
            previous = np.empty((0, 2), dtype=np.float32)
    if pose is None:
        return
    updated = update_slam_map(previous, values[:, :2], center_xy=(pose.x, pose.y))
    with sensors.lock:
        sensors.slam_map_xy = updated
        sensors.slam_map_time = timestamp_s
        if reset_map:
            current_pose = sensors.pose or pose
            sensors.robot_history = np.asarray(
                [[current_pose.x, current_pose.y]], dtype=np.float32
            )


def _gateway_camera_frame(
    snapshot: MaterializedSnapshot,
    *,
    rgb_stream: str = "camera/chest_view",
    depth_stream: str = "camera/chest_view_depth",
) -> GatewayCameraFrame:
    unavailable = (
        f"fresh aligned NavDP RGB-D unavailable for {rgb_stream!r} and "
        f"{depth_stream!r}"
    )
    decoded = materialize_rgbd(
        snapshot,
        rgb_stream=rgb_stream,
        depth_stream=depth_stream,
        prefer_depth_info=False,
        prefer_depth_timestamp=False,
        validate_dtypes=False,
        rgb_label="fresh aligned NavDP RGB",
        depth_label="fresh aligned NavDP depth",
        mismatch_message=unavailable,
    )
    rgb = decoded.rgb
    depth_raw = decoded.depth_raw
    assert depth_raw is not None
    camera_info = dict(decoded.camera_info)
    timestamp_ns = decoded.source_timestamp_ns
    depth_m = depth_raw.astype(np.float32) * float(
        camera_info.get("depth_scale_m", 0.001)
    )
    return GatewayCameraFrame(
        rgb=rgb,
        depth_m=depth_m,
        camera_info=camera_info,
        source_timestamp_s=(
            float(timestamp_ns) * 1.0e-9 if timestamp_ns > 0 else time.time()
        ),
    )


class NavDPSensorGatewayIngress(PollingSensorIngress):
    """Materialize SensorGateway camera, odometry, and point-cloud streams for NavDP."""

    CAMERA_STREAMS = ("camera/chest_view", "camera/chest_view_depth")
    ODOMETRY_STREAM = "ros/odometry"
    LIDAR_STREAM = "ros/livox_lidar_xyz"
    SLAM_CLOUD_STREAM = "ros/registered_cloud_xyz"

    def __init__(
        self,
        endpoint: str,
        sensors: _SharedSensors,
        *,
        poll_hz: float = 20.0,
        request_timeout_ms: int = 100,
        max_age_ms: float = 1000.0,
        max_skew_ms: float = 5.0,
        rgb_stream: str = CAMERA_STREAMS[0],
        depth_stream: str = CAMERA_STREAMS[1],
        client: SensorGatewayClient | None = None,
    ) -> None:
        if not str(rgb_stream).strip() or not str(depth_stream).strip():
            raise ValueError("NavDP RGB and depth streams are required")
        if str(rgb_stream).strip() == str(depth_stream).strip():
            raise ValueError("NavDP RGB and depth streams must be different")
        self.sensors = sensors
        self.camera_streams = (
            str(rgb_stream).strip(), str(depth_stream).strip(),
        )
        super().__init__(
            endpoint,
            thread_name="navdp-sensor-gateway",
            error_prefix="NavDP",
            poll_hz=poll_hz,
            request_timeout_ms=request_timeout_ms,
            max_age_ms=max_age_ms,
            max_skew_ms=max_skew_ms,
            client=client,
        )
        self._camera_lock = threading.Lock()
        self._camera: GatewayCameraFrame | None = None
        self._camera_version = 0
        self._consumed_camera_version = 0
        self._lidar_worker = LatestMessageWorker(self._process_lidar)
        self._slam_worker = LatestMessageWorker(self._process_slam_cloud)

    def _pollers(self):
        return (
            self._poll_camera,
            self._poll_odometry,
            self._poll_lidar,
            self._poll_slam_cloud,
        )

    def _poll_camera(self) -> None:
        snapshot = self._request(
            self.camera_streams, max_skew_ms=self.max_skew_ms,
        )
        rgb_frame = snapshot.snapshot.frames[self.camera_streams[0]]
        depth_frame = snapshot.snapshot.frames[self.camera_streams[1]]
        rgb_is_new = self._is_new(rgb_frame)
        depth_is_new = self._is_new(depth_frame)
        if not (rgb_is_new or depth_is_new):
            return
        camera = _gateway_camera_frame(
            snapshot,
            rgb_stream=self.camera_streams[0],
            depth_stream=self.camera_streams[1],
        )
        with self._camera_lock:
            self._camera = camera
            self._camera_version += 1

    def _poll_odometry(self) -> None:
        snapshot = self._request((self.ODOMETRY_STREAM,), max_skew_ms=0.0)
        frame = snapshot.snapshot.frames[self.ODOMETRY_STREAM]
        if not self._is_new(frame):
            return
        _update_odometry_state(
            self.sensors,
            snapshot.arrays[self.ODOMETRY_STREAM],
            source_timestamp_s=float(frame.source_timestamp_ns) * 1.0e-9,
            received_monotonic_s=float(frame.metadata.timestamp_ns) * 1.0e-9,
        )

    def _poll_lidar(self) -> None:
        snapshot = self._request((self.LIDAR_STREAM,), max_skew_ms=0.0)
        frame = snapshot.snapshot.frames[self.LIDAR_STREAM]
        if self._is_new(frame):
            self._lidar_worker.submit(snapshot.arrays[self.LIDAR_STREAM])

    def _poll_slam_cloud(self) -> None:
        snapshot = self._request((self.SLAM_CLOUD_STREAM,), max_skew_ms=0.0)
        frame = snapshot.snapshot.frames[self.SLAM_CLOUD_STREAM]
        if self._is_new(frame):
            self._slam_worker.submit(
                (
                    snapshot.arrays[self.SLAM_CLOUD_STREAM],
                    float(frame.metadata.timestamp_ns) * 1.0e-9,
                )
            )

    def _process_lidar(self, points: np.ndarray) -> None:
        _update_lidar_state(self.sensors, points)

    def _process_slam_cloud(self, item: tuple[np.ndarray, float]) -> None:
        points, received_s = item
        _update_slam_cloud_state(
            self.sensors,
            points,
            received_monotonic_s=received_s,
            reset_after_s=self.max_age_ms * 1.0e-3,
        )

    def _report_errors(self, errors: list[Exception]) -> None:
        message = "; ".join(dict.fromkeys(str(error) for error in errors))
        if message != self._last_error:
            if message:
                LOGGER.warning("SensorGateway waiting: %s", message)
            elif self._last_error:
                LOGGER.info("SensorGateway recovered")
            self._last_error = message

    def poll_camera(self) -> GatewayCameraFrame | None:
        with self._camera_lock:
            if self._camera_version == self._consumed_camera_version:
                return None
            self._consumed_camera_version = self._camera_version
            camera = self._camera
            if camera is None:
                return None
            return camera

    def _close_resources(self) -> None:
        self._lidar_worker.close()
        self._slam_worker.close()


def point_plane_distances(
    points: np.ndarray,
    normal: np.ndarray,
    offset: float,
) -> np.ndarray:
    """Evaluate a 3-D plane without dispatching tiny products to threaded BLAS."""
    values = np.asarray(points)
    direction = np.asarray(normal)
    projection = (
        values[:, 0] * direction[0]
        + values[:, 1] * direction[1]
        + values[:, 2] * direction[2]
    )
    return np.abs(projection - float(offset))


def _remove_ground(points: np.ndarray) -> np.ndarray:
    """Conservative RANSAC plane removal using only low base-frame candidates."""
    values = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    candidate_indices = np.flatnonzero(values[:, 2] <= -0.25)
    candidates = values[candidate_indices]
    if len(candidates) < 50:
        return values
    rng = np.random.default_rng(0)
    best: np.ndarray | None = None
    best_count = 0
    for _ in range(64):
        sample = candidates[rng.choice(len(candidates), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-6:
            continue
        normal /= norm
        if abs(float(normal[2])) < math.cos(math.radians(25.0)):
            continue
        plane_offset = float(
            sample[0, 0] * normal[0]
            + sample[0, 1] * normal[1]
            + sample[0, 2] * normal[2]
        )
        distance = point_plane_distances(candidates, normal, plane_offset)
        inliers = distance <= 0.04
        count = int(inliers.sum())
        if count > best_count:
            best = inliers
            best_count = count
    if best is None or best_count < 50:
        return values
    keep = np.ones(len(values), dtype=bool)
    keep[candidate_indices[best]] = False
    return values[keep]


class _SharedSensors:
    def __init__(self) -> None:
        import threading

        self.lock = threading.Lock()
        self.pose: Pose2D | None = None
        self.pose_time = 0.0
        self.pose_history: deque[tuple[float, Pose2D]] = deque(maxlen=200)
        self.points = np.empty((0, 3), dtype=np.float32)
        self.slam_map_xy = np.empty((0, 2), dtype=np.float32)
        self.slam_map_time = 0.0
        self.robot_history = np.empty((0, 2), dtype=np.float32)


def _control_freshness_snapshot(
    sensors: _SharedSensors,
) -> tuple[Pose2D | None, float, np.ndarray]:
    """Read timestamps after potentially blocking MPC work, not before it."""
    with sensors.lock:
        return sensors.pose, sensors.pose_time, sensors.points.copy()


def _navdp_request(
    server: str,
    rgb: np.ndarray,
    depth: np.ndarray,
    goal: tuple[float, float],
    *,
    pose: Pose2D | None = None,
    timeout: float = 10.0,
) -> np.ndarray:
    import requests

    rgb_encoded, depth_png = _encode_navdp_frames(rgb, depth)
    goal = (float(np.clip(goal[0], 0.0, 10.0)), float(np.clip(goal[1], -10.0, 10.0)))
    data = {"goal_data": json.dumps({"goal_x": [goal[0]], "goal_y": [goal[1]]})}
    if pose is not None:
        half_yaw = 0.5 * float(pose.yaw)
        data["state_data"] = json.dumps(
            {
                "robot_pos": [[float(pose.x), float(pose.y), 0.0]],
                "robot_quat": [
                    [0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)]
                ],
            }
        )
    response = requests.post(
        f"{server.rstrip('/')}/pointgoal_step",
        files={"image": ("image.jpg", rgb_encoded), "depth": ("depth.png", depth_png)},
        data=data,
        timeout=timeout,
    )
    response.raise_for_status()
    trajectory = np.asarray(response.json()["trajectory"], dtype=np.float32).squeeze()
    if trajectory.ndim != 2 or trajectory.shape[1] < 2 or len(trajectory) < 2:
        raise ValueError("invalid NavDP trajectory")
    return trajectory[:, :2]


def _reset_navdp(
    server: str,
    info: Mapping[str, Any],
    *,
    stop_threshold: float = -4.0,
) -> None:
    import requests

    intrinsic = [
        [float(info["fx"]), 0.0, float(info["cx"])],
        [0.0, float(info["fy"]), float(info["cy"])],
        [0.0, 0.0, 1.0],
    ]
    response = requests.post(
        f"{server.rstrip('/')}/navigator_reset",
        json={"intrinsic": intrinsic, "stop_threshold": [float(stop_threshold)], "batch_size": 1},
        timeout=30.0,
    )
    response.raise_for_status()
