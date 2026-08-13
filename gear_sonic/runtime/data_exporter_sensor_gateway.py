"""Cached SensorGateway adapter for the data-exporter sensor interfaces."""

from __future__ import annotations

import copy
import threading
import time
from typing import Any

import msgpack
import msgpack_numpy as mnp
import numpy as np

from gear_sonic.runtime.client import MaterializedSnapshot, SensorGatewayClient
from gear_sonic.runtime.contracts import SharedMemoryFrame
from gear_sonic.runtime.snapshot import SnapshotRequest
from gear_sonic.runtime.vla_sensor_gateway import decode_cpp_state_array

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


def _copy_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _copy_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_value(item) for item in value]
    return copy.deepcopy(value)


class DataExporterSensorGatewayIngress:
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
        if poll_hz <= 0.0:
            raise ValueError("DataExporter SensorGateway poll_hz must be positive")
        if request_timeout_ms <= 0:
            raise ValueError("DataExporter request_timeout_ms must be positive")
        if max_age_ms < 0.0 or max_skew_ms < 0.0:
            raise ValueError("DataExporter Gateway age and skew cannot be negative")
        self.endpoint = endpoint
        self.camera_names = tuple(camera_names)
        self.defer_video_encoding = bool(defer_video_encoding)
        camera_prefix = "camera_encoded" if self.defer_video_encoding else "camera"
        self.camera_streams = tuple(
            f"{camera_prefix}/{name}" for name in self.camera_names
        )
        self.poll_hz = float(poll_hz)
        self.max_age_ms = float(max_age_ms)
        self.max_skew_ms = float(max_skew_ms)
        self.client = client or SensorGatewayClient(
            endpoint,
            request_timeout_ms=request_timeout_ms,
        )
        self._owns_client = client is None
        self._condition = threading.Condition()
        self._camera: dict[str, Any] | None = None
        self._camera_received_ns = 0
        self._state: dict[str, Any] | None = None
        self._state_received_ns = 0
        self._robot_config: dict[str, Any] | None = None
        self._last_sequences: dict[str, int] = {}
        self._last_error = ""
        self._last_error_print_s = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="data-exporter-sensor-gateway",
            daemon=True,
        )
        self._started = False
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

    def _report_error(self, exc: Exception) -> None:
        message = str(exc)
        now = time.monotonic()
        if message != self._last_error or now - self._last_error_print_s >= 2.0:
            print(f"[DataExporter] SensorGateway waiting: {message}", flush=True)
            self._last_error = message
            self._last_error_print_s = now

    def _run(self) -> None:
        period_s = 1.0 / self.poll_hz
        while not self._stop.is_set():
            started = time.monotonic()
            for poll in (self._poll_camera, self._poll_state, self._poll_robot_config):
                if self._stop.is_set():
                    break
                try:
                    poll()
                except Exception as exc:
                    self._report_error(exc)
            self._stop.wait(max(0.0, period_s - (time.monotonic() - started)))

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("DataExporter SensorGateway ingress is closed")
        if self._started:
            return
        self._started = True
        self._thread.start()

    def _fresh(self, received_ns: int) -> bool:
        return bool(
            received_ns > 0
            and (time.monotonic_ns() - received_ns) / 1_000_000.0 <= self.max_age_ms
        )

    def read_camera(self) -> dict[str, Any] | None:
        with self._condition:
            if self._camera is None or not self._fresh(self._camera_received_ns):
                return None
            return _copy_value(self._camera)

    def read_state(self, *, clear: bool = True) -> dict[str, Any] | None:
        with self._condition:
            if self._state is None or not self._fresh(self._state_received_ns):
                return None
            state = self._state
            if clear:
                self._state = None
                self._state_received_ns = 0
            return _copy_value(state)

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
            return _copy_value(self._robot_config)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._started:
            self._thread.join(timeout=max(1.0, 2.0 / self.poll_hz))
            if self._thread.is_alive():
                raise RuntimeError("DataExporter SensorGateway worker did not stop")
        if self._owns_client:
            self.client.close()
