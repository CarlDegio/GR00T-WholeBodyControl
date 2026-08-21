"""Passive endpoint health metrics for tests and operator tooling."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import time


class EndpointState(str, Enum):
    WAITING = "waiting"
    HEALTHY = "healthy"
    IDLE = "idle"
    STALE = "stale"
    DOWN = "down"


@dataclass(frozen=True)
class EndpointHealth:
    endpoint: str
    state: EndpointState
    message_count: int
    dropped_messages: int
    out_of_order_messages: int
    failure_count: int
    consecutive_failures: int
    rate_hz: float
    last_message_age_ms: float | None
    latency_ms: float | None
    last_error: str

    def to_dict(self) -> dict[str, object]:
        return {
            "endpoint": self.endpoint,
            "state": self.state.value,
            "message_count": self.message_count,
            "dropped_messages": self.dropped_messages,
            "out_of_order_messages": self.out_of_order_messages,
            "failure_count": self.failure_count,
            "consecutive_failures": self.consecutive_failures,
            "rate_hz": self.rate_hz,
            "last_message_age_ms": self.last_message_age_ms,
            "latency_ms": self.latency_ms,
            "last_error": self.last_error,
        }


class EndpointHealthMonitor:
    """Track one endpoint without probing or modifying its wire messages."""

    def __init__(
        self,
        endpoint: str,
        *,
        expected_hz: float,
        stale_after_s: float | None = None,
        down_after_s: float | None = None,
        rate_window: int = 64,
        started_ns: int | None = None,
    ) -> None:
        if not endpoint:
            raise ValueError("endpoint cannot be empty")
        if expected_hz <= 0.0:
            raise ValueError("expected_hz must be positive")
        expected_period_s = 1.0 / float(expected_hz)
        self.endpoint = endpoint
        self.stale_after_s = (
            3.0 * expected_period_s if stale_after_s is None else float(stale_after_s)
        )
        self.down_after_s = (
            10.0 * expected_period_s if down_after_s is None else float(down_after_s)
        )
        if self.stale_after_s <= 0.0:
            raise ValueError("stale_after_s must be positive")
        if self.down_after_s <= self.stale_after_s:
            raise ValueError("down_after_s must be greater than stale_after_s")
        if rate_window < 2:
            raise ValueError("rate_window must be at least two")

        self.started_ns = time.monotonic_ns() if started_ns is None else int(started_ns)
        self._receive_times_ns: deque[int] = deque(maxlen=rate_window)
        self._last_sequence: int | None = None
        self._latency_ms: float | None = None
        self._message_count = 0
        self._dropped_messages = 0
        self._out_of_order_messages = 0
        self._failure_count = 0
        self._consecutive_failures = 0
        self._last_error = ""
        self._idle = False

    def set_idle(self, idle: bool) -> None:
        """Mark a reachable model endpoint as intentionally not producing data."""

        self._idle = bool(idle)

    def observe(
        self,
        *,
        received_ns: int | None = None,
        sequence: int | None = None,
        source_timestamp_ns: int | None = None,
        round_trip_ms: float | None = None,
    ) -> None:
        """Record a passive receive or an explicit request/response success."""

        receive_time = time.monotonic_ns() if received_ns is None else int(received_ns)
        if receive_time < 0:
            raise ValueError("received_ns cannot be negative")
        if self._receive_times_ns and receive_time < self._receive_times_ns[-1]:
            raise ValueError("receive timestamps must be monotonic")

        if sequence is not None:
            sequence = int(sequence)
            if sequence < 0:
                raise ValueError("sequence cannot be negative")
            if self._last_sequence is None:
                self._last_sequence = sequence
            elif sequence > self._last_sequence:
                if sequence > self._last_sequence + 1:
                    self._dropped_messages += sequence - self._last_sequence - 1
                self._last_sequence = sequence
            else:
                self._out_of_order_messages += 1

        if source_timestamp_ns is not None:
            source_time = int(source_timestamp_ns)
            self._latency_ms = max(0, receive_time - source_time) / 1_000_000.0
        elif round_trip_ms is not None:
            round_trip = float(round_trip_ms)
            if round_trip < 0.0:
                raise ValueError("round_trip_ms cannot be negative")
            self._latency_ms = round_trip

        self._receive_times_ns.append(receive_time)
        self._message_count += 1
        self._consecutive_failures = 0
        self._last_error = ""

    def record_failure(self, error: str) -> None:
        self._failure_count += 1
        self._consecutive_failures += 1
        self._last_error = str(error)

    def snapshot(self, *, now_ns: int | None = None) -> EndpointHealth:
        current_time = time.monotonic_ns() if now_ns is None else int(now_ns)
        if current_time < self.started_ns:
            raise ValueError("now_ns cannot be before monitor startup")

        last_age_s: float | None = None
        if self._receive_times_ns:
            last_age_s = max(0, current_time - self._receive_times_ns[-1]) / 1_000_000_000.0

        if self._consecutive_failures:
            state = EndpointState.DOWN
        elif self._idle and last_age_s is not None and last_age_s < self.down_after_s:
            state = EndpointState.IDLE
        elif last_age_s is None:
            startup_age_s = (current_time - self.started_ns) / 1_000_000_000.0
            state = EndpointState.WAITING if startup_age_s < self.down_after_s else EndpointState.DOWN
        elif last_age_s >= self.down_after_s:
            state = EndpointState.DOWN
        elif last_age_s >= self.stale_after_s:
            state = EndpointState.STALE
        else:
            state = EndpointState.HEALTHY

        rate_hz = 0.0
        if len(self._receive_times_ns) >= 2:
            duration_s = (
                self._receive_times_ns[-1] - self._receive_times_ns[0]
            ) / 1_000_000_000.0
            if duration_s > 0.0:
                rate_hz = (len(self._receive_times_ns) - 1) / duration_s

        return EndpointHealth(
            endpoint=self.endpoint,
            state=state,
            message_count=self._message_count,
            dropped_messages=self._dropped_messages,
            out_of_order_messages=self._out_of_order_messages,
            failure_count=self._failure_count,
            consecutive_failures=self._consecutive_failures,
            rate_hz=rate_hz,
            last_message_age_ms=None if last_age_s is None else last_age_s * 1000.0,
            latency_ms=self._latency_ms,
            last_error=self._last_error,
        )
