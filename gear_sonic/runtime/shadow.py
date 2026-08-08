"""Read-only comparisons between legacy sensor inputs and SensorGateway copies."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any

import numpy as np

from gear_sonic.runtime.client import (
    SensorGatewayClient,
    SensorGatewayClientError,
)
from gear_sonic.runtime.contracts import SharedMemoryFrame
from gear_sonic.runtime.snapshot import SnapshotRequest, TimestampBasis


@dataclass(frozen=True)
class ArrayComparison:
    stream: str
    comparable: bool
    matched: bool
    exact_values: bool
    shape_equal: bool
    dtype_equal: bool
    legacy_shape: tuple[int, ...]
    gateway_shape: tuple[int, ...]
    legacy_dtype: str
    gateway_dtype: str
    source_skew_ms: float | None
    max_abs_error: float | None
    mean_abs_error: float | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stream": self.stream,
            "comparable": self.comparable,
            "matched": self.matched,
            "exact_values": self.exact_values,
            "shape_equal": self.shape_equal,
            "dtype_equal": self.dtype_equal,
            "legacy_shape": list(self.legacy_shape),
            "gateway_shape": list(self.gateway_shape),
            "legacy_dtype": self.legacy_dtype,
            "gateway_dtype": self.gateway_dtype,
            "source_skew_ms": self.source_skew_ms,
            "max_abs_error": self.max_abs_error,
            "mean_abs_error": self.mean_abs_error,
            "reason": self.reason,
        }


def compare_sensor_arrays(
    stream: str,
    legacy: np.ndarray,
    gateway: np.ndarray,
    *,
    legacy_source_timestamp_ns: int,
    gateway_frame: SharedMemoryFrame,
    max_source_skew_ms: float,
) -> ArrayComparison:
    legacy_values = np.asarray(legacy)
    gateway_values = np.asarray(gateway)
    shape_equal = legacy_values.shape == gateway_values.shape
    dtype_equal = legacy_values.dtype == gateway_values.dtype
    exact_values = bool(
        shape_equal and np.array_equal(legacy_values, gateway_values, equal_nan=True)
    )
    source_skew_ms = (
        abs(int(legacy_source_timestamp_ns) - gateway_frame.source_timestamp_ns)
        / 1_000_000.0
        if legacy_source_timestamp_ns > 0 and gateway_frame.source_timestamp_ns > 0
        else None
    )

    max_abs_error: float | None = None
    mean_abs_error: float | None = None
    if shape_equal and np.issubdtype(legacy_values.dtype, np.number) and np.issubdtype(
        gateway_values.dtype,
        np.number,
    ):
        difference = np.abs(
            legacy_values.astype(np.float64, copy=False)
            - gateway_values.astype(np.float64, copy=False)
        )
        finite_difference = difference[np.isfinite(difference)]
        if finite_difference.size:
            max_abs_error = float(np.max(finite_difference))
            mean_abs_error = float(np.mean(finite_difference))

    reasons: list[str] = []
    if not shape_equal:
        reasons.append("shape mismatch")
    if not dtype_equal:
        reasons.append("dtype mismatch")
    if not exact_values:
        reasons.append("value mismatch")
    if source_skew_ms is None:
        reasons.append("missing source timestamp")
    elif source_skew_ms > max_source_skew_ms:
        reasons.append("source timestamp mismatch")
    matched = not reasons
    return ArrayComparison(
        stream=stream,
        comparable=True,
        matched=matched,
        exact_values=exact_values,
        shape_equal=shape_equal,
        dtype_equal=dtype_equal,
        legacy_shape=tuple(int(value) for value in legacy_values.shape),
        gateway_shape=tuple(int(value) for value in gateway_values.shape),
        legacy_dtype=legacy_values.dtype.str,
        gateway_dtype=gateway_values.dtype.str,
        source_skew_ms=source_skew_ms,
        max_abs_error=max_abs_error,
        mean_abs_error=mean_abs_error,
        reason=", ".join(reasons),
    )


def unavailable_comparison(
    stream: str,
    legacy: np.ndarray,
    reason: str,
) -> ArrayComparison:
    values = np.asarray(legacy)
    return ArrayComparison(
        stream=stream,
        comparable=False,
        matched=False,
        exact_values=False,
        shape_equal=False,
        dtype_equal=False,
        legacy_shape=tuple(int(value) for value in values.shape),
        gateway_shape=(),
        legacy_dtype=values.dtype.str,
        gateway_dtype="",
        source_skew_ms=None,
        max_abs_error=None,
        mean_abs_error=None,
        reason=reason,
    )


class ShadowComparisonStats:
    """Thread-safe aggregate suitable for the future diagnostics GUI."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._streams: dict[str, dict[str, Any]] = {}

    def record(self, comparison: ArrayComparison) -> None:
        with self._lock:
            state = self._streams.setdefault(
                comparison.stream,
                {
                    "samples": 0,
                    "matches": 0,
                    "mismatches": 0,
                    "unpaired": 0,
                    "last": {},
                },
            )
            state["samples"] += 1
            if not comparison.comparable:
                state["unpaired"] += 1
            elif comparison.matched:
                state["matches"] += 1
            else:
                state["mismatches"] += 1
            state["last"] = comparison.to_dict()

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                stream: {
                    "samples": int(state["samples"]),
                    "matches": int(state["matches"]),
                    "mismatches": int(state["mismatches"]),
                    "unpaired": int(state["unpaired"]),
                    "compared": int(state["matches"] + state["mismatches"]),
                    "match_rate": (
                        float(state["matches"])
                        / float(state["matches"] + state["mismatches"])
                        if state["matches"] + state["mismatches"]
                        else 0.0
                    ),
                    "coverage_rate": (
                        float(state["matches"] + state["mismatches"])
                        / float(state["samples"])
                        if state["samples"]
                        else 0.0
                    ),
                    "last": dict(state["last"]),
                }
                for stream, state in self._streams.items()
            }


class SensorGatewayShadowComparator:
    """Match one direct legacy sample to the gateway copy by source time."""

    def __init__(
        self,
        client: SensorGatewayClient,
        *,
        max_age_ms: float = 1000.0,
        max_source_skew_ms: float = 1.0,
        retries: int = 2,
    ) -> None:
        if max_age_ms < 0.0 or max_source_skew_ms < 0.0:
            raise ValueError("shadow comparison limits cannot be negative")
        if retries < 0:
            raise ValueError("shadow comparison retries cannot be negative")
        self.client = client
        self.max_age_ms = float(max_age_ms)
        self.max_source_skew_ms = float(max_source_skew_ms)
        self.retries = int(retries)
        self.stats = ShadowComparisonStats()

    def compare(
        self,
        stream: str,
        legacy: np.ndarray,
        *,
        source_timestamp_ns: int,
    ) -> ArrayComparison:
        try:
            materialized = self.client.read_snapshot(
                SnapshotRequest(
                    streams=(stream,),
                    max_age_ms=self.max_age_ms,
                    max_skew_ms=self.max_source_skew_ms,
                    anchor_timestamp_ns=int(source_timestamp_ns),
                    timestamp_basis=TimestampBasis.SOURCE,
                ),
                retries=self.retries,
            )
            comparison = compare_sensor_arrays(
                stream,
                legacy,
                materialized.arrays[stream],
                legacy_source_timestamp_ns=source_timestamp_ns,
                gateway_frame=materialized.snapshot.frames[stream],
                max_source_skew_ms=self.max_source_skew_ms,
            )
        except SensorGatewayClientError as exc:
            comparison = unavailable_comparison(stream, legacy, str(exc))
        self.stats.record(comparison)
        return comparison
