"""Cached SensorGateway adapters for the existing VLA sensor interfaces."""

from __future__ import annotations

import threading
import time
from typing import Any, Mapping

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

VLA_CAMERA_NAMES = ("ego_view", "chest_view", "left_wrist", "right_wrist")
VLA_CAMERA_STREAMS = tuple(f"camera_encoded/{name}" for name in VLA_CAMERA_NAMES)
VLA_STATE_STREAM = "cpp/state_msgpack"


def camera_message_from_snapshot(snapshot: MaterializedSnapshot) -> dict[str, Any]:
    """Recreate the subset of ``ImageMessageSchema.asdict`` consumed by VLA."""
    images: dict[str, bytes] = {}
    image_shapes: dict[str, tuple[int, ...]] = {}
    timestamps: dict[str, float] = {}
    camera_info: dict[str, Mapping[str, Any]] = {}
    for name, stream in zip(VLA_CAMERA_NAMES, VLA_CAMERA_STREAMS, strict=True):
        frame = snapshot.snapshot.frames[stream]
        payload = np.asarray(snapshot.arrays[stream], dtype=np.uint8).reshape(-1).tobytes()
        encoding = frame.attributes.get("encoding")
        if encoding != "jpeg_bytes":
            raise ValueError(
                f"VLA camera {name!r} has unsupported encoding {encoding!r}"
            )
        images[name] = payload
        image_shapes[name] = tuple(frame.attributes["image_shape"])
        timestamps[name] = (
            float(frame.source_timestamp_ns) * 1.0e-9
            if frame.source_timestamp_ns > 0
            else time.time()
        )
        camera_info[name] = dict(frame.attributes.get("camera_info", {}))
    return {
        "images": images,
        "image_shapes": image_shapes,
        "timestamps": timestamps,
        "camera_info": camera_info,
    }


class VlaSensorGatewayIngress(PollingSensorIngress):
    """Background Gateway reader exposing typed camera and robot-state caches.

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
        super().__init__(
            endpoint,
            thread_name="vla-sensor-gateway",
            error_prefix="VLA",
            poll_hz=poll_hz,
            request_timeout_ms=request_timeout_ms,
            max_age_ms=max_age_ms,
            max_skew_ms=max_skew_ms,
            client=client,
        )
        self._lock = threading.Lock()
        self._camera: dict[str, Any] | None = None
        self._camera_received_ns = 0
        self._state: dict[str, Any] | None = None
        self._state_received_ns = 0

    def _pollers(self):
        return self._poll_camera, self._poll_state

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

    def read_camera(self) -> dict[str, Any] | None:
        """Return the latest fresh four-camera message without blocking."""
        with self._lock:
            if self._camera is None or not self._fresh(self._camera_received_ns):
                return None
            return copy_numpy_tree(self._camera)

    def read_state(self, *, clear: bool = True) -> dict[str, Any] | None:
        """Return the latest fresh robot state, optionally consuming it."""
        with self._lock:
            if self._state is None or not self._fresh(self._state_received_ns):
                return None
            state = self._state
            if clear:
                self._state = None
                self._state_received_ns = 0
            return copy_numpy_tree(state)
