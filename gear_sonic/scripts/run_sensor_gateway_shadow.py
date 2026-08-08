#!/usr/bin/env python3
"""Compare legacy camera/ROS2 inputs with SensorGateway copies without control."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import signal
import threading
import time
from typing import Any, Mapping

import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq

from gear_sonic.camera.sensor_server import ImageMessageSchema
from gear_sonic.runtime.client import SensorGatewayClient, SensorGatewayClientError
from gear_sonic.runtime.config import load_runtime_profile
from gear_sonic.runtime.sensor_gateway import (
    imu_array,
    livox_xyz_array,
    odometry_array,
    pointcloud2_xyz_array,
    ros_stamp_ns,
)
from gear_sonic.runtime.shadow import SensorGatewayShadowComparator


CONTROL_OUTPUTS_ENABLED = False
ROS_STREAMS = (
    "ros/livox_lidar_xyz",
    "ros/livox_imu",
    "ros/odometry",
    "ros/registered_cloud_xyz",
)


@dataclass(frozen=True)
class ShadowSettings:
    profile_name: str
    gateway_endpoint: str
    camera_endpoint: str
    cpp_state_endpoint: str
    ros_topics: dict[str, str]
    camera_streams: tuple[str, ...]
    sample_hz: float
    request_timeout_ms: int
    max_age_ms: float
    max_source_skew_ms: float
    snapshot_retries: int
    startup_timeout_s: float
    summary_interval_s: float
    duration_s: float
    enable_camera: bool
    enable_cpp_state: bool
    enable_ros: bool


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="")
    parser.add_argument("--overlay", action="append", default=[])
    parser.add_argument("--gateway-host", default="")
    parser.add_argument("--gateway-port", type=int, default=0)
    parser.add_argument("--camera-host", default="")
    parser.add_argument("--camera-port", type=int, default=0)
    parser.add_argument("--cpp-state-host", default="")
    parser.add_argument("--cpp-state-port", type=int, default=0)
    parser.add_argument("--sample-hz", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument(
        "--enable-camera",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enable-cpp-state",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enable-ros",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def resolve_shadow_settings(args: argparse.Namespace) -> ShadowSettings:
    profile = load_runtime_profile(args.profile or None, overlays=tuple(args.overlay))
    component = profile.component("sensor_shadow")

    def endpoint(name: str, host_override: str, port_override: int) -> str:
        address = profile.endpoint(name)
        return f"tcp://{host_override or address.host}:{port_override or address.port}"

    sample_hz = float(args.sample_hz or component["sample_hz"])
    if sample_hz <= 0.0:
        raise ValueError("sample_hz must be positive")
    if args.duration_s < 0.0:
        raise ValueError("duration_s cannot be negative")
    request_timeout_ms = int(component["request_timeout_ms"])
    max_age_ms = float(component["max_age_ms"])
    max_source_skew_ms = float(component["max_source_skew_ms"])
    snapshot_retries = int(component["snapshot_retries"])
    startup_timeout_s = float(component["startup_timeout_s"])
    summary_interval_s = float(component["summary_interval_s"])
    camera_streams = tuple(str(name) for name in component["camera_streams"])
    if request_timeout_ms <= 0:
        raise ValueError("request_timeout_ms must be positive")
    if max_age_ms < 0.0 or max_source_skew_ms < 0.0:
        raise ValueError("snapshot age and skew limits cannot be negative")
    if snapshot_retries < 0:
        raise ValueError("snapshot_retries cannot be negative")
    if startup_timeout_s <= 0.0:
        raise ValueError("startup_timeout_s must be positive")
    if summary_interval_s < 0.0:
        raise ValueError("summary_interval_s cannot be negative")
    if args.enable_camera and not camera_streams:
        raise ValueError("camera_streams cannot be empty when camera comparison is enabled")
    return ShadowSettings(
        profile_name=profile.name,
        gateway_endpoint=endpoint(
            "sensor_gateway_metadata",
            args.gateway_host,
            args.gateway_port,
        ),
        camera_endpoint=endpoint("camera_server", args.camera_host, args.camera_port),
        cpp_state_endpoint=endpoint(
            "cpp_state",
            args.cpp_state_host,
            args.cpp_state_port,
        ),
        ros_topics={
            "lidar": str(profile.ros_topics["lidar"]),
            "imu": str(profile.ros_topics["lidar_imu"]),
            "odometry": str(profile.ros_topics["odometry"]),
            "registered_cloud": str(profile.ros_topics["registered_cloud"]),
        },
        camera_streams=camera_streams,
        sample_hz=sample_hz,
        request_timeout_ms=request_timeout_ms,
        max_age_ms=max_age_ms,
        max_source_skew_ms=max_source_skew_ms,
        snapshot_retries=snapshot_retries,
        startup_timeout_s=startup_timeout_s,
        summary_interval_s=summary_interval_s,
        duration_s=float(args.duration_s),
        enable_camera=bool(args.enable_camera),
        enable_cpp_state=bool(args.enable_cpp_state),
        enable_ros=bool(args.enable_ros),
    )


class DirectCppStateIngress:
    """Retain the newest untouched legacy ``g1_debug`` payload for comparison."""

    def __init__(
        self,
        context: zmq.Context,
        endpoint: str,
        *,
        topic: str = "g1_debug",
    ) -> None:
        self.topic = topic.encode("utf-8")
        self.socket = context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, self.topic)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(endpoint)
        self._latest: tuple[np.ndarray, int] | None = None

    def poll(self) -> None:
        if not self.socket.poll(0, zmq.POLLIN):
            return
        raw = self.socket.recv(zmq.NOBLOCK)
        payload = raw[len(self.topic) :]
        state = msgpack.unpackb(payload, raw=False, object_hook=mnp.decode)
        timestamp_s = float(state.get("ros_timestamp", 0.0))
        self._latest = (
            np.frombuffer(payload, dtype=np.uint8).copy(),
            max(0, int(timestamp_s * 1_000_000_000)),
        )

    def latest(self) -> tuple[np.ndarray, int] | None:
        return self._latest

    def close(self) -> None:
        self.socket.close(linger=0)


class DirectRosIngress:
    """Retain only the newest legacy ROS message for low-rate shadow sampling."""

    def __init__(self, topics: Mapping[str, str]) -> None:
        import rclpy
        from livox_ros_driver2.msg import CustomMsg
        from nav_msgs.msg import Odometry
        from rclpy.executors import SingleThreadedExecutor
        from sensor_msgs.msg import Imu, PointCloud2
        from sensor_msgs_py import point_cloud2

        self._rclpy = rclpy
        self._point_cloud2 = point_cloud2
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init(args=None)
        self.node = rclpy.create_node("sonic_sensor_gateway_shadow")
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self._lock = threading.Lock()
        self._latest: dict[str, Any] = {}
        self._closed = False
        self._thread = threading.Thread(
            target=self.executor.spin,
            name="sensor-shadow-ros2",
            daemon=True,
        )

        def retain(stream: str):
            def callback(message: Any) -> None:
                with self._lock:
                    self._latest[stream] = message

            return callback

        self._subscriptions = (
            self.node.create_subscription(
                CustomMsg,
                topics["lidar"],
                retain("ros/livox_lidar_xyz"),
                10,
            ),
            self.node.create_subscription(
                Imu,
                topics["imu"],
                retain("ros/livox_imu"),
                10,
            ),
            self.node.create_subscription(
                Odometry,
                topics["odometry"],
                retain("ros/odometry"),
                10,
            ),
            self.node.create_subscription(
                PointCloud2,
                topics["registered_cloud"],
                retain("ros/registered_cloud_xyz"),
                2,
            ),
        )

    def start(self) -> None:
        self._thread.start()

    def latest(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._latest)

    def materialize(self, stream: str, message: Any) -> tuple[np.ndarray, int]:
        converters = {
            "ros/livox_lidar_xyz": livox_xyz_array,
            "ros/livox_imu": imu_array,
            "ros/odometry": odometry_array,
            "ros/registered_cloud_xyz": lambda value: pointcloud2_xyz_array(
                value,
                self._point_cloud2,
            ),
        }
        return converters[stream](message), ros_stamp_ns(message)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.executor.shutdown(timeout_sec=1.0)
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.executor.remove_node(self.node)
        self.node.destroy_node()
        if self._owns_context and self._rclpy.ok():
            self._rclpy.shutdown()


def _summary_lines(stats: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return [
        f"{stream}=exact:{state['matches']}/{state['compared']} "
        f"coverage:{100.0 * state['coverage_rate']:.1f}%"
        for stream, state in sorted(stats.items())
    ]


def run_shadow(settings: ShadowSettings) -> None:
    if CONTROL_OUTPUTS_ENABLED:
        raise RuntimeError("SensorGateway shadow comparator must remain read-only")
    context = zmq.Context()
    camera_socket: zmq.Socket | None = None
    cpp_state: DirectCppStateIngress | None = None
    ros: DirectRosIngress | None = None
    stop = threading.Event()

    def request_stop(_signum=None, _frame=None) -> None:
        stop.set()

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    client = SensorGatewayClient(
        settings.gateway_endpoint,
        request_timeout_ms=settings.request_timeout_ms,
        context=context,
    )
    comparator = SensorGatewayShadowComparator(
        client,
        max_age_ms=settings.max_age_ms,
        max_source_skew_ms=settings.max_source_skew_ms,
        retries=settings.snapshot_retries,
    )
    try:
        startup_deadline = time.monotonic() + settings.startup_timeout_s
        while True:
            try:
                if client.ping():
                    break
            except SensorGatewayClientError:
                pass
            if time.monotonic() >= startup_deadline:
                raise RuntimeError(
                    f"SensorGateway did not become ready within {settings.startup_timeout_s:g}s"
                )
            stop.wait(0.2)
        if settings.enable_camera:
            camera_socket = context.socket(zmq.SUB)
            camera_socket.setsockopt(zmq.SUBSCRIBE, b"")
            camera_socket.setsockopt(zmq.CONFLATE, 1)
            camera_socket.setsockopt(zmq.LINGER, 0)
            camera_socket.connect(settings.camera_endpoint)
        if settings.enable_cpp_state:
            cpp_state = DirectCppStateIngress(context, settings.cpp_state_endpoint)
        if settings.enable_ros:
            ros = DirectRosIngress(settings.ros_topics)
            ros.start()

        print("[SensorShadow] READ-ONLY comparison; no control output sockets exist")
        print(f"[SensorShadow] profile: {settings.profile_name}")
        print(f"[SensorShadow] gateway: {settings.gateway_endpoint}")
        if cpp_state is not None:
            print(f"[SensorShadow] C++ state SUB: {settings.cpp_state_endpoint}")
        print(f"[SensorShadow] sample rate: {settings.sample_hz:g}Hz")
        started = time.monotonic()
        next_sample = started
        next_summary = started + settings.summary_interval_s
        last_source_timestamps: dict[str, int] = {}
        latest_camera: ImageMessageSchema | None = None

        while not stop.is_set():
            now = time.monotonic()
            if settings.duration_s and now - started >= settings.duration_s:
                break
            if camera_socket is not None and camera_socket.poll(0, zmq.POLLIN):
                payload = msgpack.unpackb(camera_socket.recv(), raw=False)
                latest_camera = ImageMessageSchema.deserialize(payload)
            if cpp_state is not None:
                cpp_state.poll()

            if now >= next_sample:
                samples: list[tuple[str, np.ndarray, int]] = []
                if latest_camera is not None:
                    for name in settings.camera_streams:
                        image = latest_camera.images.get(name)
                        timestamp_s = float(latest_camera.timestamps.get(name, 0.0))
                        if isinstance(image, np.ndarray) and timestamp_s > 0.0:
                            samples.append(
                                (
                                    f"camera/{name}",
                                    image,
                                    int(timestamp_s * 1_000_000_000),
                                )
                            )
                if ros is not None:
                    for stream, message in ros.latest().items():
                        try:
                            values, source_ns = ros.materialize(stream, message)
                        except Exception as exc:
                            print(f"[SensorShadow] {stream} conversion failed: {exc}")
                            continue
                        samples.append((stream, values, source_ns))
                if cpp_state is not None and cpp_state.latest() is not None:
                    values, source_ns = cpp_state.latest()
                    samples.append(("cpp/state_msgpack", values, source_ns))

                for stream, values, source_ns in samples:
                    if source_ns <= 0 or last_source_timestamps.get(stream) == source_ns:
                        continue
                    result = comparator.compare(
                        stream,
                        values,
                        source_timestamp_ns=source_ns,
                    )
                    last_source_timestamps[stream] = source_ns
                    if not result.comparable:
                        print(
                            "[SensorShadow] UNPAIRED "
                            + json.dumps(result.to_dict(), separators=(",", ":"))
                        )
                    elif not result.matched:
                        print(
                            "[SensorShadow] MISMATCH "
                            + json.dumps(result.to_dict(), separators=(",", ":"))
                        )
                next_sample = now + 1.0 / settings.sample_hz

            if settings.summary_interval_s > 0.0 and now >= next_summary:
                lines = _summary_lines(comparator.stats.to_dict())
                print("[SensorShadow] " + " | ".join(lines or ["waiting for samples"]))
                next_summary = now + settings.summary_interval_s
            stop.wait(0.002)
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        if ros is not None:
            ros.close()
        if cpp_state is not None:
            cpp_state.close()
        if camera_socket is not None:
            camera_socket.close(linger=0)
        client.close()
        context.term()
        print("[SensorShadow] stopped")


def main() -> None:
    args = build_argument_parser().parse_args()
    run_shadow(resolve_shadow_settings(args))


if __name__ == "__main__":
    main()
