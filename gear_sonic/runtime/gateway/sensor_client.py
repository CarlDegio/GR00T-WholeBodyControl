"""Local client for SensorGateway health and shared-memory snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np
import zmq

from gear_sonic.runtime.protocol import SharedMemoryFrame
from gear_sonic.runtime.gateway.shared_memory import (
    FrameOverwrittenError,
    read_shared_memory_frame,
)
from gear_sonic.runtime.gateway.snapshot import SensorSnapshot, SnapshotRequest


class SensorGatewayClientError(RuntimeError):
    """Base error for local SensorGateway access."""


class SensorGatewayTimeoutError(SensorGatewayClientError):
    """The metadata RPC did not respond within its configured deadline."""


class SnapshotUnavailableError(SensorGatewayClientError):
    """No readable snapshot satisfied the requested age and skew limits."""


@dataclass(frozen=True)
class MaterializedSnapshot:
    """One validated metadata snapshot and copied NumPy arrays."""

    snapshot: SensorSnapshot
    arrays: Mapping[str, np.ndarray]
    attempts: int


class SensorGatewayClient:
    """Thread-safe local RPC client with whole-snapshot overwrite retries."""

    def __init__(
        self,
        endpoint: str,
        *,
        request_timeout_ms: int = 1000,
        context: zmq.Context | None = None,
        frame_reader: Callable[[SharedMemoryFrame], np.ndarray] = read_shared_memory_frame,
    ) -> None:
        if not endpoint:
            raise ValueError("SensorGateway endpoint cannot be empty")
        if request_timeout_ms <= 0:
            raise ValueError("request_timeout_ms must be positive")
        self.endpoint = endpoint
        self.request_timeout_ms = int(request_timeout_ms)
        self._context = zmq.Context() if context is None else context
        self._owns_context = context is None
        self._frame_reader = frame_reader
        self._socket: zmq.Socket | None = None
        self._lock = threading.Lock()
        self._closed = False

    def _new_socket(self) -> zmq.Socket:
        socket = self._context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDTIMEO, self.request_timeout_ms)
        socket.setsockopt(zmq.RCVTIMEO, self.request_timeout_ms)
        socket.connect(self.endpoint)
        return socket

    def _reset_socket(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
        self._socket = None

    def _rpc(self, request: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise SensorGatewayClientError("SensorGateway client is closed")
            if self._socket is None:
                self._socket = self._new_socket()
            try:
                self._socket.send_json(dict(request))
                response = self._socket.recv_json()
            except zmq.Again as exc:
                self._reset_socket()
                raise SensorGatewayTimeoutError(
                    f"SensorGateway did not respond within {self.request_timeout_ms}ms"
                ) from exc
            except zmq.ZMQError as exc:
                self._reset_socket()
                raise SensorGatewayClientError(f"SensorGateway RPC failed: {exc}") from exc
        if not isinstance(response, dict):
            raise SensorGatewayClientError("SensorGateway response must be a JSON object")
        if response.get("type") == "sonic.sensor_gateway_error":
            raise SensorGatewayClientError(str(response.get("error", "unknown gateway error")))
        return response

    def ping(self) -> bool:
        response = self._rpc({"type": "ping", "version": 1})
        return response.get("type") == "pong" and response.get("service") == "sensor_gateway"

    def health(self) -> dict[str, Any]:
        response = self._rpc(
            {"type": "sonic.sensor_gateway_health_request", "version": 1}
        )
        if response.get("type") != "sonic.sensor_gateway_health" or int(
            response.get("version", -1)
        ) != 1:
            raise SensorGatewayClientError("unsupported SensorGateway health response")
        return response

    def request_snapshot(self, request: SnapshotRequest) -> SensorSnapshot:
        return SensorSnapshot.from_dict(self._rpc(request.to_dict()))

    @staticmethod
    def _validate_snapshot(
        request: SnapshotRequest,
        snapshot: SensorSnapshot,
        *,
        now_ns: int,
    ) -> None:
        if not snapshot.complete:
            raise SnapshotUnavailableError(snapshot.reason or "incomplete snapshot")
        missing = set(request.streams) - set(snapshot.frames)
        if missing:
            raise SnapshotUnavailableError(
                f"complete snapshot omitted streams: {', '.join(sorted(missing))}"
            )
        if snapshot.skew_ms is None or snapshot.skew_ms > request.max_skew_ms:
            raise SnapshotUnavailableError("snapshot exceeds requested skew")
        stale = [
            stream
            for stream, frame in snapshot.frames.items()
            if frame.metadata.age_ms(now_ns) > request.max_age_ms
            or frame.metadata.is_expired(now_ns)
        ]
        if stale:
            raise SnapshotUnavailableError(
                f"snapshot expired before materialization: {', '.join(sorted(stale))}"
            )

    def read_snapshot(
        self,
        request: SnapshotRequest,
        *,
        retries: int = 2,
        retry_backoff_s: float = 0.005,
    ) -> MaterializedSnapshot:
        if retries < 0:
            raise ValueError("retries cannot be negative")
        if retry_backoff_s < 0.0:
            raise ValueError("retry_backoff_s cannot be negative")

        last_error: Exception | None = None
        for attempt in range(1, retries + 2):
            try:
                snapshot = self.request_snapshot(request)
                self._validate_snapshot(request, snapshot, now_ns=time.monotonic_ns())
                arrays = {
                    stream: self._frame_reader(snapshot.frames[stream])
                    for stream in request.streams
                }
                self._validate_snapshot(request, snapshot, now_ns=time.monotonic_ns())
                return MaterializedSnapshot(
                    snapshot=snapshot,
                    arrays=MappingProxyType(arrays),
                    attempts=attempt,
                )
            except (
                FileNotFoundError,
                FrameOverwrittenError,
                SensorGatewayTimeoutError,
                SnapshotUnavailableError,
            ) as exc:
                last_error = exc
                if attempt <= retries and retry_backoff_s:
                    time.sleep(retry_backoff_s)

        raise SnapshotUnavailableError(
            f"snapshot unavailable after {retries + 1} attempts: {last_error}"
        ) from last_error

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._reset_socket()
            if self._owns_context:
                self._context.term()

    def __enter__(self) -> "SensorGatewayClient":
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()
