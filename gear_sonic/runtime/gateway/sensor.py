"""Read-only SensorGateway core, ZMQ ingress adapters, and Snapshot RPC."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import threading
import time
from typing import Any, Mapping

import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq

from gear_sonic.camera.sensor_server import ImageMessageSchema
from gear_sonic.runtime.gateway.diagnostics import EndpointHealthMonitor
from gear_sonic.runtime.gateway.rgb_preview import RgbPreviewWorker
from gear_sonic.runtime.gateway.shared_memory import SharedMemoryRing
from gear_sonic.runtime.gateway.snapshot import SensorSnapshotStore, SnapshotRequest
from gear_sonic.runtime.gateway.visualization import (
    VISUALIZATION_SCHEMA,
    VISUALIZATION_STREAMS,
)


DEPTH_ANYTHING_STATUS_TYPE = "sonic.depth_anything_status"


@dataclass
class _RetiredRing:
    expires_ns: int
    ring: SharedMemoryRing


class SensorGatewayCore:
    """Thread-safe sensor array publication without any control output."""

    def __init__(
        self,
        *,
        slot_count: int = 8,
        history_size: int = 64,
        frame_ttl_ms: int = 1000,
        retired_ring_ttl_s: float = 2.0,
    ) -> None:
        if slot_count < 2:
            raise ValueError("slot_count must be at least two")
        if retired_ring_ttl_s < 0.0:
            raise ValueError("retired_ring_ttl_s cannot be negative")
        self.slot_count = int(slot_count)
        self.frame_ttl_ms = int(frame_ttl_ms)
        self.retired_ring_ttl_ns = int(retired_ring_ttl_s * 1_000_000_000)
        self.snapshots = SensorSnapshotStore(history_size=history_size)
        self._rings: dict[str, SharedMemoryRing] = {}
        self._health: dict[str, EndpointHealthMonitor] = {}
        self._retired: list[_RetiredRing] = []
        self._lock = threading.RLock()

    def register_endpoint(
        self,
        endpoint: str,
        *,
        expected_hz: float,
        started_ns: int | None = None,
    ) -> None:
        with self._lock:
            self._health.setdefault(
                endpoint,
                EndpointHealthMonitor(
                    endpoint,
                    expected_hz=expected_hz,
                    started_ns=started_ns,
                ),
            )

    def observe_endpoint(
        self,
        endpoint: str,
        *,
        expected_hz: float,
        received_ns: int | None = None,
        source_timestamp_ns: int | None = None,
    ) -> None:
        receive_time = time.monotonic_ns() if received_ns is None else int(received_ns)
        with self._lock:
            self.register_endpoint(
                endpoint,
                expected_hz=expected_hz,
                started_ns=receive_time,
            )
            self._health[endpoint].observe(
                received_ns=receive_time,
                source_timestamp_ns=source_timestamp_ns,
            )

    def set_endpoint_idle(self, endpoint: str, idle: bool) -> None:
        with self._lock:
            monitor = self._health.get(endpoint)
            if monitor is None:
                raise KeyError(f"endpoint is not registered: {endpoint}")
            monitor.set_idle(idle)

    def publish_array(
        self,
        stream: str,
        array: np.ndarray,
        *,
        received_ns: int | None = None,
        source_timestamp_ns: int = 0,
        source_clock: str = "unknown",
        expected_hz: float = 10.0,
        attributes: Mapping[str, Any] | None = None,
    ):
        receive_time = time.monotonic_ns() if received_ns is None else int(received_ns)
        values = np.ascontiguousarray(array)
        with self._lock:
            ring = self._ensure_ring(
                stream,
                max(values.nbytes, 1),
                receive_time,
                expected_hz,
            )
        # The ring owns its write lock. Large RGB-D and point-cloud copies must
        # not hold the global metadata lock or delay Snapshot/health RPCs.
        frame = ring.write(
            values,
            received_ns=receive_time,
            source_timestamp_ns=source_timestamp_ns,
            source_clock=source_clock,
            attributes={} if attributes is None else dict(attributes),
        )
        with self._lock:
            self.snapshots.add(frame)
            monitor = self._health.get(stream)
            if monitor is None:
                monitor = EndpointHealthMonitor(
                    stream,
                    expected_hz=expected_hz,
                    started_ns=receive_time,
                )
                self._health[stream] = monitor
            monitor.observe(
                received_ns=receive_time,
                sequence=frame.metadata.sequence,
                source_timestamp_ns=(
                    source_timestamp_ns
                    if source_clock == "gateway_monotonic" and source_timestamp_ns > 0
                    else None
                ),
            )
            self._cleanup_retired(receive_time)
            return frame

    def record_failure(self, stream: str, error: str, *, expected_hz: float = 10.0) -> None:
        with self._lock:
            monitor = self._health.get(stream)
            if monitor is None:
                monitor = EndpointHealthMonitor(stream, expected_hz=expected_hz)
                self._health[stream] = monitor
            monitor.record_failure(error)

    def select(self, request: SnapshotRequest, *, now_ns: int | None = None):
        current_time = time.monotonic_ns() if now_ns is None else int(now_ns)
        with self._lock:
            return self.snapshots.select(request, now_ns=current_time)

    def health_payload(self, *, now_ns: int | None = None) -> dict[str, Any]:
        current_time = time.monotonic_ns() if now_ns is None else int(now_ns)
        with self._lock:
            self._cleanup_retired(current_time)
            return {
                "type": "sonic.sensor_gateway_health",
                "version": 1,
                "timestamp_ns": current_time,
                "streams": {
                    name: monitor.snapshot(now_ns=current_time).to_dict()
                    for name, monitor in sorted(self._health.items())
                },
                "shared_memory": {
                    name: {
                        "name": ring.name,
                        "slot_count": ring.slot_count,
                        "slot_size_bytes": ring.slot_size_bytes,
                    }
                    for name, ring in sorted(self._rings.items())
                },
                "retired_ring_count": len(self._retired),
            }

    def _ensure_ring(
        self,
        stream: str,
        required_bytes: int,
        now_ns: int,
        expected_hz: float,
    ) -> SharedMemoryRing:
        retained_frames = min(
            self.snapshots.history_size,
            math.ceil(float(expected_hz) * self.frame_ttl_ms / 1000.0) + 2,
        )
        required_slots = max(self.slot_count, retained_frames)
        ring = self._rings.get(stream)
        if (
            ring is not None
            and required_bytes <= ring.slot_size_bytes
            and required_slots <= ring.slot_count
        ):
            return ring
        initial_sequence = 0
        if ring is not None:
            initial_sequence = ring.next_sequence
            self._retired.append(
                _RetiredRing(
                    expires_ns=now_ns + self.retired_ring_ttl_ns,
                    ring=ring,
                )
            )
            self.snapshots.discard_stream(stream)
        capacity = required_bytes if ring is None else max(required_bytes, ring.slot_size_bytes * 2)
        replacement = SharedMemoryRing(
            stream,
            slot_count=required_slots,
            slot_size_bytes=capacity,
            ttl_ms=self.frame_ttl_ms,
            initial_sequence=initial_sequence,
        )
        self._rings[stream] = replacement
        self.snapshots.configure_stream(stream, history_size=required_slots)
        return replacement

    def _cleanup_retired(self, now_ns: int) -> None:
        retained: list[_RetiredRing] = []
        for item in self._retired:
            if now_ns >= item.expires_ns:
                item.ring.shutdown()
            else:
                retained.append(item)
        self._retired = retained

    def close(self) -> None:
        with self._lock:
            for ring in self._rings.values():
                ring.shutdown()
            for item in self._retired:
                item.ring.shutdown()
            self._rings.clear()
            self._retired.clear()


class SensorGatewayRpc:
    """JSON REQ/REP interface for snapshots, health, and readiness checks."""

    def __init__(self, context: zmq.Context, endpoint: str, core: SensorGatewayCore) -> None:
        self.core = core
        self.socket = context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(endpoint)

    def serve_once(self, timeout_ms: int = 0) -> bool:
        if not self.socket.poll(timeout_ms, zmq.POLLIN):
            return False
        try:
            request = self.socket.recv_json()
            message_type = request.get("type")
            if message_type == "sonic.snapshot_request":
                response = self.core.select(SnapshotRequest.from_dict(request)).to_dict()
            elif message_type == "sonic.sensor_gateway_health_request":
                response = self.core.health_payload()
            elif message_type == "ping":
                response = {"type": "pong", "version": 1, "service": "sensor_gateway"}
            else:
                raise ValueError(f"unsupported SensorGateway request: {message_type!r}")
        except Exception as exc:
            response = {
                "type": "sonic.sensor_gateway_error",
                "version": 1,
                "error": str(exc),
            }
        self.socket.send_json(response)
        return True

    def close(self) -> None:
        self.socket.close(linger=0)


class SensorGatewayRpcServer:
    """Own the REP socket on a dedicated thread so ingress cannot starve RPC."""

    def __init__(
        self,
        context: zmq.Context,
        endpoint: str,
        core: SensorGatewayCore,
        *,
        poll_timeout_ms: int = 10,
    ) -> None:
        if poll_timeout_ms <= 0:
            raise ValueError("RPC poll timeout must be positive")
        self.context = context
        self.endpoint = endpoint
        self.core = core
        self.poll_timeout_ms = int(poll_timeout_ms)
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._error: Exception | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="sensor-gateway-rpc",
            daemon=True,
        )
        self._closed = False

    def _run(self) -> None:
        rpc: SensorGatewayRpc | None = None
        try:
            rpc = SensorGatewayRpc(self.context, self.endpoint, self.core)
            self._ready.set()
            while not self._stop.is_set():
                rpc.serve_once(timeout_ms=self.poll_timeout_ms)
        except Exception as exc:
            self._error = exc
            self._ready.set()
        finally:
            if rpc is not None:
                rpc.close()

    def start(self, *, timeout_s: float = 2.0) -> None:
        self._thread.start()
        if not self._ready.wait(timeout_s):
            raise RuntimeError("SensorGateway RPC thread did not become ready")
        if self._error is not None:
            raise RuntimeError(f"SensorGateway RPC failed to start: {self._error}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._thread.join(timeout=1.0)
        if self._thread.is_alive():
            raise RuntimeError("SensorGateway RPC thread did not stop")


class CameraZmqIngress:
    """Read existing camera PUB messages without republishing or acknowledging them."""

    def __init__(
        self,
        context: zmq.Context,
        endpoint: str,
        core: SensorGatewayCore,
        *,
        expected_hz: float = 30.0,
        preview_rgb: bool = False,
        preview_worker: RgbPreviewWorker | None = None,
    ) -> None:
        self.core = core
        self.expected_hz = float(expected_hz)
        self.socket = context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(endpoint)
        self.core.register_endpoint(
            "source/camera_server",
            expected_hz=self.expected_hz,
        )
        self._preview = preview_worker if preview_worker is not None else (
            RgbPreviewWorker(encoded=True) if preview_rgb else None
        )
        if self._preview is not None:
            self._preview.start()

    def poll_once(self, timeout_ms: int = 0) -> int:
        if not self.socket.poll(timeout_ms, zmq.POLLIN):
            return 0
        received_ns = time.monotonic_ns()
        payload = msgpack.unpackb(self.socket.recv(), raw=False)
        self.core.observe_endpoint(
            "source/camera_server",
            expected_hz=self.expected_hz,
            received_ns=received_ns,
        )
        encoded_schema = ImageMessageSchema.deserialize(payload, decode_images=False)
        count = 0
        preview_images: dict[str, bytes | str] = {}
        for name, encoded in encoded_schema.images.items():
            if name.endswith("_depth") or not isinstance(encoded, bytes | bytearray | str):
                continue
            preview_images[name] = bytes(encoded) if isinstance(encoded, bytearray) else encoded
            if isinstance(encoded, str):
                encoded_array = np.frombuffer(encoded.encode("utf-8"), dtype=np.uint8).copy()
                wire_encoding = "base64_jpeg"
            else:
                encoded_array = np.frombuffer(encoded, dtype=np.uint8).copy()
                wire_encoding = "jpeg_bytes"
            timestamp_s = float(encoded_schema.timestamps.get(name, 0.0))
            self.core.publish_array(
                f"camera_encoded/{name}",
                encoded_array,
                received_ns=received_ns,
                source_timestamp_ns=max(0, int(timestamp_s * 1_000_000_000)),
                source_clock="camera_unix" if timestamp_s > 0.0 else "unknown",
                expected_hz=self.expected_hz,
                attributes={
                    "encoding": wire_encoding,
                    "decoded_color_order": "RGB",
                    "image_shape": list(encoded_schema.image_shapes.get(name, ())),
                    "camera_info": dict(encoded_schema.camera_info.get(name, {})),
                },
            )
        if self._preview is not None and preview_images:
            self._preview.publish(preview_images)

        schema = ImageMessageSchema.deserialize(payload)
        for name, image in schema.images.items():
            if not isinstance(image, np.ndarray):
                continue
            timestamp_s = float(schema.timestamps.get(name, 0.0))
            base_name = name[: -len("_depth")] if name.endswith("_depth") else name
            attributes = {
                "encoding": "numpy",
                "camera_info": dict(schema.camera_info.get(base_name, {})),
            }
            if not name.endswith("_depth"):
                attributes["color_order"] = "RGB"
            self.core.publish_array(
                f"camera/{name}",
                image,
                received_ns=received_ns,
                source_timestamp_ns=max(0, int(timestamp_s * 1_000_000_000)),
                source_clock="camera_unix" if timestamp_s > 0.0 else "unknown",
                expected_hz=self.expected_hz,
                attributes=attributes,
            )
            count += 1
        return count

    def close(self) -> None:
        if self._preview is not None:
            self._preview.close()
        self.socket.close(linger=0)


class DepthAnythingZmqIngress:
    """Register RGB-estimated metric chest depth as a derived stream."""

    STREAM = "derived/depth_anything/chest_view"

    def __init__(
        self,
        context: zmq.Context,
        endpoint: str,
        core: SensorGatewayCore,
        *,
        expected_hz: float = 2.0,
    ) -> None:
        self.core = core
        self.expected_hz = float(expected_hz)
        self.socket = context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(endpoint)
        self.core.register_endpoint(
            "source/depth_anything", expected_hz=self.expected_hz
        )

    def poll_once(self, timeout_ms: int = 0) -> int:
        if not self.socket.poll(timeout_ms, zmq.POLLIN):
            return 0
        received_ns = time.monotonic_ns()
        payload = msgpack.unpackb(self.socket.recv(), raw=False)
        if payload.get("type") == DEPTH_ANYTHING_STATUS_TYPE:
            active = bool(payload.get("active", False))
            self.core.observe_endpoint(
                "source/depth_anything",
                expected_hz=self.expected_hz,
                received_ns=received_ns,
            )
            self.core.set_endpoint_idle("source/depth_anything", not active)
            return 1
        schema = ImageMessageSchema.deserialize(payload)
        depth = schema.images.get("chest_view_depth")
        if not isinstance(depth, np.ndarray):
            raise ValueError("Depth Anything payload is missing chest_view_depth")
        if depth.ndim != 2 or depth.dtype != np.uint16:
            raise ValueError(
                f"Depth Anything depth must be HxW uint16, got "
                f"{depth.shape} {depth.dtype}"
            )
        timestamp_s = float(
            schema.timestamps.get(
                "chest_view_depth", schema.timestamps.get("chest_view", 0.0)
            )
        )
        camera_info = dict(schema.camera_info.get("chest_view", {}))
        depth_source = str(camera_info.get("depth_source", ""))
        if not depth_source.startswith("depth-anything-v2-metric-"):
            raise ValueError(
                "Depth Anything payload has invalid depth_source: "
                f"{depth_source!r}"
            )
        if not bool(camera_info.get("metric_model_output", False)):
            raise ValueError("Depth Anything payload is not marked as metric output")
        if bool(camera_info.get("uses_raw_depth", True)):
            raise ValueError("Depth Anything payload must not use raw depth")
        if float(camera_info.get("depth_scale_m", 0.0)) != 0.001:
            raise ValueError("Depth Anything uint16 payload must use millimetres")
        if camera_info.get("inference_owner") not in {"lavira", "base_pose"}:
            raise ValueError("Depth Anything payload has no active inference owner")
        self.core.set_endpoint_idle("source/depth_anything", False)
        self.core.observe_endpoint(
            "source/depth_anything",
            expected_hz=self.expected_hz,
            received_ns=received_ns,
        )
        self.core.publish_array(
            self.STREAM,
            depth,
            received_ns=received_ns,
            source_timestamp_ns=max(0, int(timestamp_s * 1_000_000_000)),
            source_clock="camera_unix" if timestamp_s > 0.0 else "unknown",
            expected_hz=self.expected_hz,
            attributes={
                "encoding": "numpy",
                "camera_info": camera_info,
                "depth_source": depth_source,
            },
        )
        return 1

    def close(self) -> None:
        self.socket.close(linger=0)


class CppStateZmqIngress:
    """Copy untouched C++ state and robot-config msgpack into shared memory."""

    def __init__(
        self,
        context: zmq.Context,
        endpoint: str,
        core: SensorGatewayCore,
        *,
        topic: str = "g1_debug",
        expected_hz: float = 50.0,
    ) -> None:
        self.core = core
        self.topic = topic.encode("utf-8")
        self.expected_hz = float(expected_hz)
        self.socket = context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, self.topic)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(endpoint)
        self.config_topic = b"robot_config"
        self.config_socket = context.socket(zmq.SUB)
        self.config_socket.setsockopt(zmq.SUBSCRIBE, self.config_topic)
        self.config_socket.setsockopt(zmq.CONFLATE, 1)
        self.config_socket.setsockopt(zmq.LINGER, 0)
        self.config_socket.connect(endpoint)
        self.core.register_endpoint(
            "source/cpp_state",
            expected_hz=self.expected_hz,
        )

    def poll_once(self, timeout_ms: int = 0) -> int:
        count = 0
        if self.socket.poll(timeout_ms, zmq.POLLIN):
            received_ns = time.monotonic_ns()
            raw = self.socket.recv()
            payload = raw[len(self.topic) :]
            state = msgpack.unpackb(payload, raw=False, object_hook=mnp.decode)
            timestamp_s = float(state.get("ros_timestamp", 0.0))
            self.core.observe_endpoint(
                "source/cpp_state",
                expected_hz=self.expected_hz,
                received_ns=received_ns,
            )
            self.core.publish_array(
                "cpp/state_msgpack",
                np.frombuffer(payload, dtype=np.uint8).copy(),
                received_ns=received_ns,
                source_timestamp_ns=max(0, int(timestamp_s * 1_000_000_000)),
                source_clock="ros_time" if timestamp_s > 0.0 else "unknown",
                expected_hz=self.expected_hz,
                attributes={
                    "encoding": "msgpack",
                    "topic": self.topic.decode("utf-8"),
                    "upstream_index": int(state.get("index", -1)),
                },
            )
            count += 1

        if self.config_socket.poll(0, zmq.POLLIN):
            received_ns = time.monotonic_ns()
            raw = self.config_socket.recv()
            payload = raw[len(self.config_topic) :]
            msgpack.unpackb(payload, raw=False, object_hook=mnp.decode)
            self.core.publish_array(
                "cpp/robot_config_msgpack",
                np.frombuffer(payload, dtype=np.uint8).copy(),
                received_ns=received_ns,
                source_clock="unknown",
                attributes={
                    "encoding": "msgpack",
                    "topic": self.config_topic.decode("utf-8"),
                },
            )
            count += 1
        return count

    def close(self) -> None:
        self.config_socket.close(linger=0)
        self.socket.close(linger=0)


class VisualizationZmqIngress:
    """Receive pre-rendered JPEG panels without coupling GUI to producers."""

    def __init__(
        self,
        context: zmq.Context,
        endpoint: str,
        core: SensorGatewayCore,
        *,
        expected_hz: float = 20.0,
    ) -> None:
        self.core = core
        self.expected_hz = float(expected_hz)
        self.socket = context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVHWM, 4)
        self.socket.bind(endpoint)
        self.core.register_endpoint(
            "source/visualization_ingress", expected_hz=self.expected_hz
        )

    def poll_once(self, timeout_ms: int = 0) -> int:
        if not self.socket.poll(timeout_ms, zmq.POLLIN):
            return 0
        received_ns = time.monotonic_ns()
        metadata_raw, jpeg = self.socket.recv_multipart()
        metadata = json.loads(metadata_raw)
        if metadata.get("type") != VISUALIZATION_SCHEMA or int(
            metadata.get("version", -1)
        ) != 1:
            raise ValueError("unsupported visualization frame")
        stream = str(metadata.get("stream", ""))
        if stream not in VISUALIZATION_STREAMS:
            raise ValueError(f"unsupported visualization stream: {stream}")
        encoded = np.frombuffer(jpeg, dtype=np.uint8).copy()
        self.core.observe_endpoint(
            "source/visualization_ingress",
            expected_hz=self.expected_hz,
            received_ns=received_ns,
        )
        self.core.publish_array(
            stream,
            encoded,
            received_ns=received_ns,
            source_timestamp_ns=max(0, int(metadata.get("timestamp_ns", 0))),
            source_clock="gateway_monotonic",
            expected_hz=self.expected_hz,
            attributes={
                "encoding": "jpeg",
                "source_sequence": int(metadata.get("sequence", -1)),
                "source_shape": list(metadata.get("shape", [])),
            },
        )
        return 1

    def close(self) -> None:
        self.socket.close(linger=0)


class Ros2SensorIngress:
    """Receive the existing ROS2 sensor topics on a dedicated executor thread."""

    def __init__(
        self,
        core: SensorGatewayCore,
        *,
        lidar_topic: str,
        imu_topic: str,
        odometry_topic: str,
        registered_cloud_topic: str,
        lidar_expected_hz: float = 10.0,
        imu_expected_hz: float = 200.0,
        odometry_expected_hz: float = 100.0,
        registered_cloud_expected_hz: float = 10.0,
    ) -> None:
        # Keep ROS imports out of module import time so non-ROS gateway tests and
        # configuration tools continue to work in ordinary Python environments.
        from livox_ros_driver2.msg import CustomMsg
        from nav_msgs.msg import Odometry
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
        from sensor_msgs.msg import Imu, PointCloud2
        from sensor_msgs_py import point_cloud2

        self.core = core
        self._rclpy = rclpy
        self._point_cloud2 = point_cloud2
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init(args=None)
        self.node = rclpy.create_node("sonic_sensor_gateway")
        self.executor = MultiThreadedExecutor(num_threads=2)
        self.executor.add_node(self.node)
        self._thread = threading.Thread(
            target=self.executor.spin,
            name="sensor-gateway-ros2",
            daemon=True,
        )
        self._closed = False

        self._expected_hz = {
            "source/ros_lidar": float(lidar_expected_hz),
            "source/ros_imu": float(imu_expected_hz),
            "source/ros_odometry": float(odometry_expected_hz),
            "source/ros_registered_cloud": float(registered_cloud_expected_hz),
        }
        for endpoint, expected_hz in self._expected_hz.items():
            self.core.register_endpoint(endpoint, expected_hz=expected_hz)

        self._subscriptions = (
            self.node.create_subscription(CustomMsg, lidar_topic, self._on_lidar, 10),
            self.node.create_subscription(Imu, imu_topic, self._on_imu, 10),
            self.node.create_subscription(Odometry, odometry_topic, self._on_odometry, 10),
            self.node.create_subscription(
                PointCloud2,
                registered_cloud_topic,
                self._on_registered_cloud,
                2,
            ),
        )
        self._topics = {
            "lidar": lidar_topic,
            "imu": imu_topic,
            "odometry": odometry_topic,
            "registered_cloud": registered_cloud_topic,
        }

    def _observe_source(self, endpoint: str, received_ns: int) -> None:
        self.core.observe_endpoint(
            endpoint,
            expected_hz=self._expected_hz[endpoint],
            received_ns=received_ns,
        )

    def _record_error(self, endpoint: str, stream: str, exc: Exception) -> None:
        expected_hz = self._expected_hz[endpoint]
        self.core.record_failure(endpoint, str(exc), expected_hz=expected_hz)
        self.core.record_failure(stream, str(exc), expected_hz=expected_hz)

    def _on_lidar(self, message: Any) -> None:
        received_ns = time.monotonic_ns()
        try:
            values = livox_xyz_array(message)
            self._observe_source("source/ros_lidar", received_ns)
            self.core.publish_array(
                "ros/livox_lidar_xyz",
                values,
                received_ns=received_ns,
                source_timestamp_ns=ros_stamp_ns(message),
                source_clock="ros_time",
                expected_hz=self._expected_hz["source/ros_lidar"],
                attributes={
                    "encoding": "float32_xyz",
                    "frame_id": str(getattr(message.header, "frame_id", "")),
                    "topic": self._topics["lidar"],
                },
            )
        except Exception as exc:
            self._record_error("source/ros_lidar", "ros/livox_lidar_xyz", exc)

    def _on_imu(self, message: Any) -> None:
        received_ns = time.monotonic_ns()
        try:
            self._observe_source("source/ros_imu", received_ns)
            self.core.publish_array(
                "ros/livox_imu",
                imu_array(message),
                received_ns=received_ns,
                source_timestamp_ns=ros_stamp_ns(message),
                source_clock="ros_time",
                expected_hz=self._expected_hz["source/ros_imu"],
                attributes={
                    "encoding": "float64_vector",
                    "fields": [
                        "orientation_x",
                        "orientation_y",
                        "orientation_z",
                        "orientation_w",
                        "angular_velocity_x",
                        "angular_velocity_y",
                        "angular_velocity_z",
                        "linear_acceleration_x",
                        "linear_acceleration_y",
                        "linear_acceleration_z",
                    ],
                    "frame_id": str(getattr(message.header, "frame_id", "")),
                    "topic": self._topics["imu"],
                },
            )
        except Exception as exc:
            self._record_error("source/ros_imu", "ros/livox_imu", exc)

    def _on_odometry(self, message: Any) -> None:
        received_ns = time.monotonic_ns()
        try:
            self._observe_source("source/ros_odometry", received_ns)
            self.core.publish_array(
                "ros/odometry",
                odometry_array(message),
                received_ns=received_ns,
                source_timestamp_ns=ros_stamp_ns(message),
                source_clock="ros_time",
                expected_hz=self._expected_hz["source/ros_odometry"],
                attributes={
                    "encoding": "float64_vector",
                    "fields": [
                        "position_x",
                        "position_y",
                        "position_z",
                        "orientation_x",
                        "orientation_y",
                        "orientation_z",
                        "orientation_w",
                        "linear_velocity_x",
                        "linear_velocity_y",
                        "linear_velocity_z",
                        "angular_velocity_x",
                        "angular_velocity_y",
                        "angular_velocity_z",
                    ],
                    "frame_id": str(getattr(message.header, "frame_id", "")),
                    "child_frame_id": str(getattr(message, "child_frame_id", "")),
                    "topic": self._topics["odometry"],
                },
            )
        except Exception as exc:
            self._record_error("source/ros_odometry", "ros/odometry", exc)

    def _on_registered_cloud(self, message: Any) -> None:
        received_ns = time.monotonic_ns()
        try:
            values = pointcloud2_xyz_array(message, self._point_cloud2)
            self._observe_source("source/ros_registered_cloud", received_ns)
            self.core.publish_array(
                "ros/registered_cloud_xyz",
                values,
                received_ns=received_ns,
                source_timestamp_ns=ros_stamp_ns(message),
                source_clock="ros_time",
                expected_hz=self._expected_hz["source/ros_registered_cloud"],
                attributes={
                    "encoding": "float32_xyz",
                    "frame_id": str(getattr(message.header, "frame_id", "")),
                    "topic": self._topics["registered_cloud"],
                },
            )
        except Exception as exc:
            self._record_error(
                "source/ros_registered_cloud",
                "ros/registered_cloud_xyz",
                exc,
            )

    def start(self) -> None:
        self._thread.start()

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


def ros_stamp_ns(message: Any) -> int:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return 0
    return max(0, int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec))


def livox_xyz_array(message: Any) -> np.ndarray:
    values = np.fromiter(
        (
            coordinate
            for point in message.points
            for coordinate in (point.x, point.y, point.z)
        ),
        dtype=np.float32,
        count=3 * len(message.points),
    )
    return values.reshape(-1, 3)


def pointcloud2_xyz_array(message: Any, point_cloud2_module: Any) -> np.ndarray:
    values = np.asarray(
        point_cloud2_module.read_points(
            message,
            field_names=("x", "y", "z"),
            skip_nans=True,
        )
    )
    if values.dtype.names:
        values = np.column_stack(tuple(values[name] for name in ("x", "y", "z")))
    return np.ascontiguousarray(values, dtype=np.float32).reshape(-1, 3)


def odometry_array(message: Any) -> np.ndarray:
    pose = message.pose.pose
    twist = message.twist.twist
    return np.asarray(
        (
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
            twist.linear.x,
            twist.linear.y,
            twist.linear.z,
            twist.angular.x,
            twist.angular.y,
            twist.angular.z,
        ),
        dtype=np.float64,
    )


def imu_array(message: Any) -> np.ndarray:
    return np.asarray(
        (
            message.orientation.x,
            message.orientation.y,
            message.orientation.z,
            message.orientation.w,
            message.angular_velocity.x,
            message.angular_velocity.y,
            message.angular_velocity.z,
            message.linear_acceleration.x,
            message.linear_acceleration.y,
            message.linear_acceleration.z,
        ),
        dtype=np.float64,
    )
