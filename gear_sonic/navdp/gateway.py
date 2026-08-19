"""SensorGateway ingress and NavDP HTTP request helpers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np

from gear_sonic.camera.constants import PRODUCTION_JPEG_QUALITY
from gear_sonic.navdp.control import LatestMessageWorker
from gear_sonic.navdp.navigation import Pose2D, update_slam_map
from gear_sonic.navdp.visualization import filter_livox_points
from gear_sonic.runtime.client import MaterializedSnapshot, SensorGatewayClient
from gear_sonic.runtime.config import load_runtime_profile
from gear_sonic.runtime.contracts import SharedMemoryFrame
from gear_sonic.runtime.snapshot import SnapshotRequest

_DEFAULT_PROFILE = load_runtime_profile()
_NAVDP_DEFAULTS = _DEFAULT_PROFILE.component("navdp")


def _bind_endpoint(name: str) -> str:
    endpoint = _DEFAULT_PROFILE.endpoint(name)
    return f"tcp://*:{endpoint.port}"


@dataclass
class NavDPPlannerConfig:
    command_endpoint: str = _DEFAULT_PROFILE.endpoint_uri("navigation_command")
    status_endpoint: str = _bind_endpoint("navigation_status")
    output_endpoint: str = _bind_endpoint("navdp_velocity")
    navdp_server: str = _DEFAULT_PROFILE.endpoint_uri("xnavdp_http")
    sensor_gateway_endpoint: str = _DEFAULT_PROFILE.endpoint_uri(
        "sensor_gateway_metadata"
    )
    sensor_gateway_poll_hz: float = float(_NAVDP_DEFAULTS["sensor_gateway_poll_hz"])
    sensor_gateway_request_timeout_ms: int = int(
        _NAVDP_DEFAULTS["sensor_gateway_request_timeout_ms"]
    )
    sensor_gateway_max_age_ms: float = float(
        _NAVDP_DEFAULTS["sensor_gateway_max_age_ms"]
    )
    sensor_gateway_max_skew_ms: float = float(
        _NAVDP_DEFAULTS["sensor_gateway_max_skew_ms"]
    )
    control_hz: float = float(_NAVDP_DEFAULTS["control_hz"])
    mpc_hz: float = float(_NAVDP_DEFAULTS["mpc_hz"])
    mpc_result_timeout_s: float = float(_NAVDP_DEFAULTS["mpc_result_timeout_s"])
    heading_preview_s: float = float(_NAVDP_DEFAULTS["heading_preview_s"])
    goal_tolerance_m: float = float(_NAVDP_DEFAULTS["goal_tolerance_m"])
    navdp_stop_threshold: float = float(_NAVDP_DEFAULTS["stop_threshold"])
    odom_timeout_s: float = float(_NAVDP_DEFAULTS["odometry_timeout_s"])
    trajectory_timeout_s: float = float(_NAVDP_DEFAULTS["trajectory_timeout_s"])
    navdp_request_timeout_s: float = float(_NAVDP_DEFAULTS["request_timeout_s"])
    visualize: bool = bool(_NAVDP_DEFAULTS["visualize"])
    visualization_gateway_endpoint: str = ""
    record_actorray: bool = bool(_NAVDP_DEFAULTS["record_actorray"])
    actorray_output_dir: str = str(_NAVDP_DEFAULTS["actorray_output_dir"])
    actorray_record_fps: float = float(_NAVDP_DEFAULTS["actorray_record_fps"])


def _quaternion_yaw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


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


def _extract_camera_frame(message: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    images = message.get("images", {})
    info = message.get("camera_info", {}).get("ego_view", {})
    rgb = np.asarray(images.get("ego_view"))
    depth_raw = np.asarray(images.get("ego_view_depth"))
    if rgb.ndim != 3 or depth_raw.ndim != 2 or rgb.shape[:2] != depth_raw.shape:
        raise ValueError("fresh aligned ego-view RGB-D unavailable")
    scale = float(info.get("depth_scale_m", 0.001))
    return rgb.copy(), depth_raw.astype(np.float32) * scale, info


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
        _quaternion_yaw(*map(float, state[3:7])),
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
    *,
    received_monotonic_s: float | None,
) -> None:
    points = _remove_ground(filter_livox_points(raw_points))
    timestamp_s = (
        time.monotonic()
        if received_monotonic_s is None
        else float(received_monotonic_s)
    )
    with sensors.lock:
        sensors.points = points
        sensors.points_time = timestamp_s


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


def _gateway_camera_frame(snapshot: MaterializedSnapshot) -> GatewayCameraFrame:
    rgb_stream = "camera/ego_view"
    depth_stream = "camera/ego_view_depth"
    rgb_frame = snapshot.snapshot.frames[rgb_stream]
    depth_frame = snapshot.snapshot.frames[depth_stream]
    camera_info = dict(rgb_frame.attributes.get("camera_info", {}))
    timestamp_ns = rgb_frame.source_timestamp_ns or depth_frame.source_timestamp_ns
    message = {
        "images": {
            "ego_view": snapshot.arrays[rgb_stream],
            "ego_view_depth": snapshot.arrays[depth_stream],
        },
        "camera_info": {"ego_view": camera_info},
    }
    rgb, depth_m, info = _extract_camera_frame(message)
    return GatewayCameraFrame(
        rgb=rgb,
        depth_m=depth_m,
        camera_info=info,
        source_timestamp_s=(
            float(timestamp_ns) * 1.0e-9 if timestamp_ns > 0 else time.time()
        ),
    )


class NavDPSensorGatewayIngress:
    """Materialize SensorGateway camera, odometry, and point-cloud streams for NavDP."""

    CAMERA_STREAMS = ("camera/ego_view", "camera/ego_view_depth")
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
        client: SensorGatewayClient | None = None,
    ) -> None:
        if poll_hz <= 0.0:
            raise ValueError("sensor gateway poll_hz must be positive")
        if request_timeout_ms <= 0:
            raise ValueError("sensor gateway request_timeout_ms must be positive")
        if max_age_ms < 0.0 or max_skew_ms < 0.0:
            raise ValueError("sensor gateway age and skew cannot be negative")
        self.sensors = sensors
        self.poll_hz = float(poll_hz)
        self.max_age_ms = float(max_age_ms)
        self.max_skew_ms = float(max_skew_ms)
        self.client = client or SensorGatewayClient(
            endpoint,
            request_timeout_ms=request_timeout_ms,
        )
        self._owns_client = client is None
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="navdp-sensor-gateway",
            daemon=True,
        )
        self._camera_lock = threading.Lock()
        self._camera: GatewayCameraFrame | None = None
        self._camera_version = 0
        self._consumed_camera_version = 0
        self._last_sequences: dict[str, int] = {}
        self._last_error = ""
        self._last_error_print_s = 0.0
        self._lidar_worker = LatestMessageWorker(self._process_lidar)
        self._slam_worker = LatestMessageWorker(self._process_slam_cloud)
        self._closed = False

    def _request(self, streams: tuple[str, ...], *, max_skew_ms: float):
        return self.client.read_snapshot(
            SnapshotRequest(
                streams=streams,
                max_age_ms=self.max_age_ms,
                max_skew_ms=max_skew_ms,
            ),
            retries=0,
        )

    def _is_new(self, frame: SharedMemoryFrame) -> bool:
        sequence = frame.metadata.sequence
        if self._last_sequences.get(frame.stream) == sequence:
            return False
        self._last_sequences[frame.stream] = sequence
        return True

    def _poll_camera(self) -> None:
        snapshot = self._request(self.CAMERA_STREAMS, max_skew_ms=self.max_skew_ms)
        rgb_frame = snapshot.snapshot.frames[self.CAMERA_STREAMS[0]]
        depth_frame = snapshot.snapshot.frames[self.CAMERA_STREAMS[1]]
        rgb_is_new = self._is_new(rgb_frame)
        depth_is_new = self._is_new(depth_frame)
        if not (rgb_is_new or depth_is_new):
            return
        camera = _gateway_camera_frame(snapshot)
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
            self._lidar_worker.submit(
                (
                    snapshot.arrays[self.LIDAR_STREAM],
                    float(frame.metadata.timestamp_ns) * 1.0e-9,
                )
            )

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

    def _process_lidar(self, item: tuple[np.ndarray, float]) -> None:
        points, received_s = item
        _update_lidar_state(
            self.sensors,
            points,
            received_monotonic_s=received_s,
        )

    def _process_slam_cloud(self, item: tuple[np.ndarray, float]) -> None:
        points, received_s = item
        _update_slam_cloud_state(
            self.sensors,
            points,
            received_monotonic_s=received_s,
            reset_after_s=self.max_age_ms * 1.0e-3,
        )

    def _report_error(self, exc: Exception) -> None:
        message = str(exc)
        now = time.monotonic()
        if message != self._last_error or now - self._last_error_print_s >= 2.0:
            print(f"[NavDP] SensorGateway waiting: {message}", flush=True)
            self._last_error = message
            self._last_error_print_s = now

    def _run(self) -> None:
        period_s = 1.0 / self.poll_hz
        pollers = (
            self._poll_camera,
            self._poll_odometry,
            self._poll_lidar,
            self._poll_slam_cloud,
        )
        while not self._stop.is_set():
            started = time.monotonic()
            for poll in pollers:
                if self._stop.is_set():
                    break
                try:
                    poll()
                except Exception as exc:
                    self._report_error(exc)
            self._stop.wait(max(0.0, period_s - (time.monotonic() - started)))

    def start(self) -> None:
        self._thread.start()

    def poll_camera(self) -> GatewayCameraFrame | None:
        with self._camera_lock:
            if self._camera_version == self._consumed_camera_version:
                return None
            self._consumed_camera_version = self._camera_version
            camera = self._camera
            if camera is None:
                return None
            return GatewayCameraFrame(
                rgb=camera.rgb.copy(),
                depth_m=camera.depth_m.copy(),
                camera_info=dict(camera.camera_info),
                source_timestamp_s=camera.source_timestamp_s,
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._lidar_worker.close()
        self._slam_worker.close()
        if self._owns_client:
            self.client.close()


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
        self.points_time = 0.0
        self.slam_map_xy = np.empty((0, 2), dtype=np.float32)
        self.slam_map_time = 0.0
        self.robot_history = np.empty((0, 2), dtype=np.float32)


def _control_freshness_snapshot(
    sensors: _SharedSensors,
) -> tuple[Pose2D | None, float, np.ndarray, float]:
    """Read timestamps after potentially blocking MPC work, not before it."""
    with sensors.lock:
        return (
            sensors.pose,
            sensors.pose_time,
            sensors.points.copy(),
            sensors.points_time,
        )


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
