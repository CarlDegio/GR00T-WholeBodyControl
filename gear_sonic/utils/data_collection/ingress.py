"""Cached SensorGateway adapter for the data-exporter sensor interfaces."""

from __future__ import annotations

import threading
import time
from typing import Any

import msgpack
import msgpack_numpy as mnp
import numpy as np

from gear_sonic.runtime.gateway.sensor_client import (
    MaterializedSnapshot,
    SensorGatewayClient,
)
from gear_sonic.runtime.gateway.polling_ingress import (
    PollingSensorIngress,
    copy_numpy_tree,
)
from gear_sonic.runtime.protocol import decode_cpp_state_array

DATA_EXPORTER_STATE_STREAM = "cpp/state_msgpack"
DATA_EXPORTER_ROBOT_CONFIG_STREAM = "cpp/robot_config_msgpack"


def decode_robot_config_array(values: np.ndarray) -> dict[str, Any]:
    """Decode robot config without changing its legacy list/scalar value types."""
    payload = np.asarray(values, dtype=np.uint8).reshape(-1).tobytes()
    decoded = msgpack.unpackb(payload, raw=False, object_hook=mnp.decode)
    if not isinstance(decoded, dict):
        raise ValueError("robot_config payload must decode to a mapping")
    return decoded


def data_exporter_camera_message_from_snapshot(
    snapshot: MaterializedSnapshot,
    camera_names: tuple[str, ...],
    *,
    encoded: bool,
) -> dict[str, Any]:
    """Recreate the camera-client dictionary consumed by DataExporter."""
    prefix = "camera_encoded" if encoded else "camera"
    images: dict[str, Any] = {}
    timestamps: dict[str, float] = {}
    camera_info: dict[str, dict[str, Any]] = {}
    for name in camera_names:
        stream = f"{prefix}/{name}"
        frame = snapshot.snapshot.frames[stream]
        values = np.asarray(snapshot.arrays[stream])
        if encoded:
            if values.ndim != 1 or values.dtype != np.uint8:
                raise ValueError(
                    f"encoded camera {name!r} must be 1-D uint8, "
                    f"got {values.shape} {values.dtype}"
                )
            wire_encoding = frame.attributes.get("encoding")
            payload = values.tobytes()
            if wire_encoding == "jpeg_bytes":
                images[name] = payload
            elif wire_encoding == "base64_jpeg":
                images[name] = payload.decode("utf-8")
            else:
                raise ValueError(
                    f"encoded camera {name!r} has unsupported encoding {wire_encoding!r}"
                )
        else:
            if values.ndim != 3 or values.shape[-1] != 3 or values.dtype != np.uint8:
                raise ValueError(
                    f"camera {name!r} must be HxWx3 uint8, "
                    f"got {values.shape} {values.dtype}"
                )
            images[name] = values.copy()
        timestamps[name] = (
            float(frame.source_timestamp_ns) * 1.0e-9
            if frame.source_timestamp_ns > 0
            else time.time()
        )
        camera_info[name] = dict(frame.attributes.get("camera_info", {}))
    return {
        "images": images,
        "timestamps": timestamps,
        "camera_info": camera_info,
    }


class DataExporterSensorGatewayIngress(PollingSensorIngress):
    """Background Gateway reader exposing non-blocking exporter sensor caches."""

    def __init__(
        self,
        endpoint: str,
        *,
        camera_names: tuple[str, ...],
        defer_video_encoding: bool,
        poll_hz: float = 50.0,
        request_timeout_ms: int = 100,
        max_age_ms: float = 1000.0,
        max_skew_ms: float = 5.0,
        client: SensorGatewayClient | None = None,
    ) -> None:
        if not camera_names or len(set(camera_names)) != len(camera_names):
            raise ValueError("DataExporter camera names must be non-empty and unique")
        self.camera_names = tuple(camera_names)
        self.defer_video_encoding = bool(defer_video_encoding)
        camera_prefix = "camera_encoded" if self.defer_video_encoding else "camera"
        self.camera_streams = tuple(
            f"{camera_prefix}/{name}" for name in self.camera_names
        )
        super().__init__(
            endpoint,
            thread_name="data-exporter-sensor-gateway",
            error_prefix="DataExporter",
            poll_hz=poll_hz,
            request_timeout_ms=request_timeout_ms,
            max_age_ms=max_age_ms,
            max_skew_ms=max_skew_ms,
            client=client,
        )
        self._condition = threading.Condition()
        self._camera: dict[str, Any] | None = None
        self._camera_received_ns = 0
        self._state: dict[str, Any] | None = None
        self._state_received_ns = 0
        self._robot_config: dict[str, Any] | None = None

    def _pollers(self):
        return self._poll_camera, self._poll_state, self._poll_robot_config

    def _poll_camera(self) -> None:
        snapshot = self._request(self.camera_streams, max_skew_ms=self.max_skew_ms)
        frames = [snapshot.snapshot.frames[stream] for stream in self.camera_streams]
        frame_updates = [self._is_new(frame) for frame in frames]
        if not any(frame_updates):
            return
        message = data_exporter_camera_message_from_snapshot(
            snapshot,
            self.camera_names,
            encoded=self.defer_video_encoding,
        )
        with self._condition:
            self._camera = message
            self._camera_received_ns = max(
                frame.metadata.timestamp_ns for frame in frames
            )

    def _poll_state(self) -> None:
        snapshot = self._request((DATA_EXPORTER_STATE_STREAM,), max_skew_ms=0.0)
        frame = snapshot.snapshot.frames[DATA_EXPORTER_STATE_STREAM]
        if not self._is_new(frame):
            return
        state = decode_cpp_state_array(snapshot.arrays[DATA_EXPORTER_STATE_STREAM])
        with self._condition:
            self._state = state
            self._state_received_ns = frame.metadata.timestamp_ns

    def _poll_robot_config(self) -> None:
        with self._condition:
            if self._robot_config is not None:
                return
        snapshot = self._request((DATA_EXPORTER_ROBOT_CONFIG_STREAM,), max_skew_ms=0.0)
        config = decode_robot_config_array(
            snapshot.arrays[DATA_EXPORTER_ROBOT_CONFIG_STREAM]
        )
        with self._condition:
            self._robot_config = config
            self._condition.notify_all()

    def read_camera(self) -> dict[str, Any] | None:
        with self._condition:
            if self._camera is None or not self._fresh(self._camera_received_ns):
                return None
            return copy_numpy_tree(self._camera)

    def read_state(self, *, clear: bool = True) -> dict[str, Any] | None:
        with self._condition:
            if self._state is None or not self._fresh(self._state_received_ns):
                return None
            state = self._state
            if clear:
                self._state = None
                self._state_received_ns = 0
            return copy_numpy_tree(state)

    def wait_for_robot_config(self, timeout_s: float = 0.0) -> dict[str, Any]:
        if timeout_s < 0.0:
            raise ValueError("robot_config timeout cannot be negative")
        deadline = None if timeout_s == 0.0 else time.monotonic() + timeout_s
        with self._condition:
            while self._robot_config is None:
                if self._closed:
                    raise RuntimeError("DataExporter SensorGateway ingress closed while waiting")
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0.0:
                    raise TimeoutError(
                        f"No robot_config received from SensorGateway {self.endpoint} "
                        f"within {timeout_s}s"
                    )
                self._condition.wait(remaining)
            return copy_numpy_tree(self._robot_config)

    def _wake_waiters(self) -> None:
        with self._condition:
            self._condition.notify_all()
