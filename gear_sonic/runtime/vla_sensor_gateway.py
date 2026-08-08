"""Cached SensorGateway adapters for the existing VLA sensor interfaces."""

from __future__ import annotations

import copy
import threading
import time
from typing import Any, Mapping

import msgpack
import msgpack_numpy as mnp
import numpy as np

from gear_sonic.runtime.client import MaterializedSnapshot, SensorGatewayClient
from gear_sonic.runtime.contracts import SharedMemoryFrame
from gear_sonic.runtime.snapshot import SnapshotRequest


VLA_CAMERA_NAMES = ("ego_view", "chest_view", "left_wrist", "right_wrist")
VLA_CAMERA_STREAMS = tuple(f"camera/{name}" for name in VLA_CAMERA_NAMES)
VLA_STATE_STREAM = "cpp/state_msgpack"


def _convert_lists_to_numpy(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _convert_lists_to_numpy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    return value


def decode_cpp_state_array(values: np.ndarray) -> dict[str, Any]:
    """Decode Gateway's untouched C++ payload like ``ZMQStateSubscriber``."""
    payload = np.asarray(values, dtype=np.uint8).reshape(-1).tobytes()
    decoded = msgpack.unpackb(payload, raw=False, object_hook=mnp.decode)
    if not isinstance(decoded, dict):
        raise ValueError("C++ state payload must decode to a mapping")
    return _convert_lists_to_numpy(decoded)


def camera_message_from_snapshot(snapshot: MaterializedSnapshot) -> dict[str, Any]:
    """Recreate the subset of ``ImageMessageSchema.asdict`` consumed by VLA."""
    images: dict[str, np.ndarray] = {}
    timestamps: dict[str, float] = {}
    camera_info: dict[str, Mapping[str, Any]] = {}
    for name, stream in zip(VLA_CAMERA_NAMES, VLA_CAMERA_STREAMS, strict=True):
        frame = snapshot.snapshot.frames[stream]
        image = np.asarray(snapshot.arrays[stream])
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(
                f"VLA camera {name!r} must be HxWx3 uint8, got {image.shape} {image.dtype}"
            )
        images[name] = image.copy()
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


def _copy_sensor_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _copy_sensor_value(item) for key, item in value.items()}
    return copy.deepcopy(value)


class VlaSensorGatewayIngress:
    """Background Gateway reader exposing VLA's legacy camera/state methods.

    Gateway RPC and shared-memory copies stay on this object's worker thread.
    The 50 Hz VLA control loop only reads already materialized local caches.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        poll_hz: float = 50.0,
        request_timeout_ms: int = 100,
        max_age_ms: float = 1000.0,
        max_skew_ms: float = 5.0,
        client: SensorGatewayClient | None = None,
    ) -> None:
        if poll_hz <= 0.0:
            raise ValueError("VLA SensorGateway poll_hz must be positive")
        if request_timeout_ms <= 0:
            raise ValueError("VLA SensorGateway request_timeout_ms must be positive")
        if max_age_ms < 0.0 or max_skew_ms < 0.0:
            raise ValueError("VLA SensorGateway age and skew cannot be negative")
        self.poll_hz = float(poll_hz)
        self.max_age_ms = float(max_age_ms)
        self.max_skew_ms = float(max_skew_ms)
        self.client = client or SensorGatewayClient(
            endpoint,
            request_timeout_ms=request_timeout_ms,
        )
        self._owns_client = client is None
        self._lock = threading.Lock()
        self._camera: dict[str, Any] | None = None
        self._camera_received_ns = 0
        self._state: dict[str, Any] | None = None
        self._state_received_ns = 0
        self._last_sequences: dict[str, int] = {}
        self._last_error = ""
        self._last_error_print_s = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="vla-sensor-gateway",
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
        snapshot = self._request(VLA_CAMERA_STREAMS, max_skew_ms=self.max_skew_ms)
        frames = [snapshot.snapshot.frames[stream] for stream in VLA_CAMERA_STREAMS]
        frame_updates = [self._is_new(frame) for frame in frames]
        if not any(frame_updates):
            return
        message = camera_message_from_snapshot(snapshot)
        with self._lock:
            self._camera = message
            self._camera_received_ns = max(frame.metadata.timestamp_ns for frame in frames)

    def _poll_state(self) -> None:
        snapshot = self._request((VLA_STATE_STREAM,), max_skew_ms=0.0)
        frame = snapshot.snapshot.frames[VLA_STATE_STREAM]
        if not self._is_new(frame):
            return
        state = decode_cpp_state_array(snapshot.arrays[VLA_STATE_STREAM])
        with self._lock:
            self._state = state
            self._state_received_ns = frame.metadata.timestamp_ns

    def _report_error(self, exc: Exception) -> None:
        message = str(exc)
        now = time.monotonic()
        if message != self._last_error or now - self._last_error_print_s >= 2.0:
            print(f"[VLA] SensorGateway waiting: {message}", flush=True)
            self._last_error = message
            self._last_error_print_s = now

    def _run(self) -> None:
        period_s = 1.0 / self.poll_hz
        while not self._stop.is_set():
            started = time.monotonic()
            for poll in (self._poll_camera, self._poll_state):
                if self._stop.is_set():
                    break
                try:
                    poll()
                except Exception as exc:
                    self._report_error(exc)
            self._stop.wait(max(0.0, period_s - (time.monotonic() - started)))

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("VLA SensorGateway ingress is closed")
        if self._started:
            return
        self._started = True
        self._thread.start()

    def _fresh(self, received_ns: int) -> bool:
        return bool(
            received_ns > 0
            and (time.monotonic_ns() - received_ns) / 1_000_000.0 <= self.max_age_ms
        )

    def read(self, blocking: bool = False, **_kwargs) -> dict[str, Any] | None:
        """Match ``ComposedCameraClientSensor.read`` without doing I/O here."""
        if blocking:
            raise ValueError("blocking reads are unsupported by SensorGateway cache")
        with self._lock:
            if self._camera is None or not self._fresh(self._camera_received_ns):
                return None
            return _copy_sensor_value(self._camera)

    def get_msg(self, clear: bool = True) -> dict[str, Any] | None:
        """Match ``ZMQStateSubscriber.get_msg`` against the local state cache."""
        with self._lock:
            if self._state is None or not self._fresh(self._state_received_ns):
                return None
            state = self._state
            if clear:
                self._state = None
                self._state_received_ns = 0
            return _copy_sensor_value(state)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._started:
            self._thread.join(timeout=max(1.0, 2.0 / self.poll_hz))
            if self._thread.is_alive():
                raise RuntimeError("VLA SensorGateway worker did not stop")
        if self._owns_client:
            self.client.close()
