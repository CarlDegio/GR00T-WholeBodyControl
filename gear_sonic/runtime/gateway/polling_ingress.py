"""Shared polling lifecycle for SensorGateway-backed consumers."""

from __future__ import annotations

import copy
import threading
import time
from typing import Any, Callable, Sequence

import numpy as np

from gear_sonic.runtime.gateway.sensor_client import SensorGatewayClient
from gear_sonic.runtime.gateway.snapshot import SnapshotRequest
from gear_sonic.runtime.protocol import SharedMemoryFrame


def copy_numpy_tree(value: Any) -> Any:
    """Copy nested sensor data without sharing mutable NumPy storage."""
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: copy_numpy_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [copy_numpy_tree(item) for item in value]
    return copy.deepcopy(value)


class PollingSensorIngress:
    """Own a SensorGateway client and run subclass pollers on one worker thread."""

    def __init__(
        self,
        endpoint: str,
        *,
        thread_name: str,
        error_prefix: str,
        poll_hz: float,
        request_timeout_ms: int,
        max_age_ms: float,
        max_skew_ms: float,
        client: SensorGatewayClient | None,
    ) -> None:
        if poll_hz <= 0.0:
            raise ValueError("SensorGateway poll_hz must be positive")
        if request_timeout_ms <= 0:
            raise ValueError("SensorGateway request_timeout_ms must be positive")
        if max_age_ms < 0.0 or max_skew_ms < 0.0:
            raise ValueError("SensorGateway age and skew cannot be negative")
        self.endpoint = endpoint
        self.poll_hz = float(poll_hz)
        self.max_age_ms = float(max_age_ms)
        self.max_skew_ms = float(max_skew_ms)
        self.client = client or SensorGatewayClient(
            endpoint,
            request_timeout_ms=request_timeout_ms,
        )
        self._owns_client = client is None
        self._error_prefix = error_prefix
        self._last_sequences: dict[str, int] = {}
        self._last_error = ""
        self._last_error_print_s = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=thread_name,
            daemon=True,
        )
        self._started = False
        self._closed = False

    def _pollers(self) -> Sequence[Callable[[], None]]:
        raise NotImplementedError

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

    def _fresh(self, received_ns: int) -> bool:
        return bool(
            received_ns > 0
            and (time.monotonic_ns() - received_ns) / 1_000_000.0
            <= self.max_age_ms
        )

    def _report_errors(self, errors: list[Exception]) -> None:
        for error in errors:
            message = str(error)
            now = time.monotonic()
            if message != self._last_error or now - self._last_error_print_s >= 2.0:
                print(
                    f"[{self._error_prefix}] SensorGateway waiting: {message}",
                    flush=True,
                )
                self._last_error = message
                self._last_error_print_s = now

    def _run(self) -> None:
        period_s = 1.0 / self.poll_hz
        while not self._stop.is_set():
            started = time.monotonic()
            errors = []
            for poll in self._pollers():
                if self._stop.is_set():
                    break
                try:
                    poll()
                except Exception as exc:
                    errors.append(exc)
            self._report_errors(errors)
            self._stop.wait(max(0.0, period_s - (time.monotonic() - started)))

    def start(self) -> None:
        if self._closed:
            raise RuntimeError(f"{self._error_prefix} SensorGateway ingress is closed")
        if self._started:
            return
        self._started = True
        self._thread.start()

    def _wake_waiters(self) -> None:
        pass

    def _close_resources(self) -> None:
        pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._wake_waiters()
        if self._started:
            self._thread.join(timeout=max(1.0, 2.0 / self.poll_hz))
            if self._thread.is_alive():
                raise RuntimeError(
                    f"{self._error_prefix} SensorGateway worker did not stop"
                )
        self._close_resources()
        if self._owns_client:
            self.client.close()
