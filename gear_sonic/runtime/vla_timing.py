"""Non-blocking VLA timing telemetry for the fixed SensorGateway dashboard."""

from __future__ import annotations

from collections import deque
import json
import math
import time
from typing import Callable, Mapping

import zmq


VLA_TIMING_SCHEMA = "sonic.vla_timing"
VLA_TIMING_SEGMENTS = (
    "frame_age",
    "camera_read",
    "state_read",
    "jpeg_encode",
    "observation_build",
    "request_pack",
    "policy_roundtrip",
    "response_unpack",
    "action_postprocess",
    "worker_total",
)


def _validate_segments(values: Mapping[str, float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, raw_value in values.items():
        if name not in VLA_TIMING_SEGMENTS:
            continue
        value = float(raw_value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"invalid VLA timing {name}: {raw_value}")
        result[name] = value
    if not result:
        raise ValueError("VLA timing sample has no recognized segments")
    return result


class VlaTimingPublisher:
    """Best-effort PUSH publisher that can never block the VLA worker."""

    def __init__(self, endpoint: str) -> None:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.SNDHWM, 4)
        self.socket.setsockopt(zmq.IMMEDIATE, 1)
        self.socket.connect(endpoint)
        self.sequence = 0

    def publish(self, segments_ms: Mapping[str, float]) -> bool:
        payload = {
            "type": VLA_TIMING_SCHEMA,
            "version": 1,
            "sequence": self.sequence,
            "timestamp_ns": time.monotonic_ns(),
            "segments_ms": _validate_segments(segments_ms),
        }
        self.sequence += 1
        try:
            self.socket.send_json(payload, flags=zmq.DONTWAIT)
            return True
        except zmq.Again:
            return False

    def close(self) -> None:
        self.socket.close(linger=0)
        self.context.term()


def _percentile(values: deque[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    alpha = position - lower
    return ordered[lower] * (1.0 - alpha) + ordered[upper] * alpha


class VlaTimingWindow:
    """Rolling timing distribution rendered by SensorGateway."""

    def __init__(self, window_size: int = 100) -> None:
        if window_size < 2:
            raise ValueError("VLA timing window_size must be at least two")
        self.window_size = int(window_size)
        self.samples = {
            name: deque(maxlen=self.window_size) for name in VLA_TIMING_SEGMENTS
        }
        self.sample_count = 0
        self.last_received_ns: int | None = None

    def record(
        self,
        segments_ms: Mapping[str, float],
        *,
        received_ns: int | None = None,
    ) -> None:
        values = _validate_segments(segments_ms)
        for name, value in values.items():
            self.samples[name].append(value)
        self.sample_count += 1
        self.last_received_ns = (
            time.monotonic_ns() if received_ns is None else int(received_ns)
        )

    def snapshot(self, *, now_ns: int | None = None) -> dict:
        current_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
        age_ms = (
            None
            if self.last_received_ns is None
            else max(0, current_ns - self.last_received_ns) / 1_000_000.0
        )
        return {
            "sample_count": self.sample_count,
            "last_sample_age_ms": age_ms,
            "window_size": self.window_size,
            "segments_ms": {
                name: {
                    "last": values[-1] if values else None,
                    "mean": sum(values) / len(values) if values else None,
                    "p50": _percentile(values, 0.50),
                    "p95": _percentile(values, 0.95),
                    "count": len(values),
                }
                for name, values in self.samples.items()
            },
        }


class VlaTimingIngress:
    """Receive VLA timing samples on the SensorGateway process thread."""

    def __init__(
        self,
        context: zmq.Context,
        endpoint: str,
        window: VlaTimingWindow,
        *,
        observe: Callable[[int], None] | None = None,
    ) -> None:
        self.window = window
        self.observe = observe
        self.socket = context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVHWM, 16)
        self.socket.bind(endpoint)

    def poll_once(self, timeout_ms: int = 0) -> int:
        if not self.socket.poll(timeout_ms, zmq.POLLIN):
            return 0
        received_ns = time.monotonic_ns()
        payload = self.socket.recv_json()
        if payload.get("type") != VLA_TIMING_SCHEMA or int(payload.get("version", -1)) != 1:
            raise ValueError("invalid VLA timing message")
        self.window.record(payload.get("segments_ms", {}), received_ns=received_ns)
        if self.observe is not None:
            self.observe(received_ns)
        return 1

    def close(self) -> None:
        self.socket.close(linger=0)
