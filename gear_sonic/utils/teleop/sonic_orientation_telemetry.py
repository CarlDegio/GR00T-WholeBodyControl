"""Shared contracts for SONIC target-heading and measured-yaw feedback."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from typing import Any


ORIENTATION_TELEMETRY_TYPE = "sonic_orientation_telemetry"
ORIENTATION_TELEMETRY_VERSION = 1
_TWO_PI = 2.0 * math.pi
_HEADING_ZERO_TOLERANCE = 1.0e-9


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _optional_finite_float(value: Any, *, name: str) -> float | None:
    return None if value is None else _finite_float(value, name=name)


def _wrapped(value: float) -> float:
    return math.remainder(float(value), _TWO_PI)


def quaternion_yaw_wxyz(quaternion: Any) -> float:
    """Return wrapped yaw from a finite, nonzero scalar-first quaternion."""
    if isinstance(quaternion, (str, bytes)):
        raise ValueError("base quaternion must contain four finite values")
    try:
        values = tuple(float(item) for item in quaternion)
    except (TypeError, ValueError) as exc:
        raise ValueError("base quaternion must contain four finite values") from exc
    if len(values) != 4 or not all(math.isfinite(item) for item in values):
        raise ValueError("base quaternion must contain four finite values")
    norm = math.sqrt(sum(item * item for item in values))
    if norm <= 1.0e-12:
        raise ValueError("base quaternion norm must be positive")
    qw, qx, qy, qz = (item / norm for item in values)
    return _wrapped(
        math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
    )


@dataclass(frozen=True)
class OrientationTelemetrySample:
    emitted_at_monotonic_s: float
    actual_yaw_rad: float | None
    actual_heading_rad: float | None
    heading_setpoint_rad: float
    heading_lag_rad: float | None
    state_age_s: float | None


def _validated_sample(sample: OrientationTelemetrySample) -> OrientationTelemetrySample:
    emitted = _finite_float(
        sample.emitted_at_monotonic_s,
        name="emitted_at_monotonic_s",
    )
    actual_yaw = _optional_finite_float(
        sample.actual_yaw_rad,
        name="actual_yaw_rad",
    )
    actual_heading = _optional_finite_float(
        sample.actual_heading_rad,
        name="actual_heading_rad",
    )
    heading_setpoint = _finite_float(
        sample.heading_setpoint_rad,
        name="heading_setpoint_rad",
    )
    heading_lag = _optional_finite_float(
        sample.heading_lag_rad,
        name="heading_lag_rad",
    )
    state_age = _optional_finite_float(sample.state_age_s, name="state_age_s")
    if state_age is not None and state_age < 0.0:
        raise ValueError("state_age_s must be non-negative")
    if actual_yaw is None and (
        actual_heading is not None or heading_lag is not None or state_age is not None
    ):
        raise ValueError("actual heading, lag, and state age require actual yaw")
    if actual_yaw is not None and state_age is None:
        raise ValueError("actual yaw requires state age")
    if actual_heading is None and heading_lag is not None:
        raise ValueError("heading lag requires actual heading")
    return OrientationTelemetrySample(
        emitted_at_monotonic_s=emitted,
        actual_yaw_rad=None if actual_yaw is None else _wrapped(actual_yaw),
        actual_heading_rad=(
            None if actual_heading is None else _wrapped(actual_heading)
        ),
        heading_setpoint_rad=_wrapped(heading_setpoint),
        heading_lag_rad=None if heading_lag is None else _wrapped(heading_lag),
        state_age_s=state_age,
    )


def encode_orientation_telemetry(sample: OrientationTelemetrySample) -> str:
    """Serialize one validated orientation sample for the local PUB channel."""
    value = _validated_sample(sample)
    return json.dumps(
        {
            "type": ORIENTATION_TELEMETRY_TYPE,
            "version": ORIENTATION_TELEMETRY_VERSION,
            "emitted_at_monotonic_s": value.emitted_at_monotonic_s,
            "actual_yaw_rad": value.actual_yaw_rad,
            "actual_heading_rad": value.actual_heading_rad,
            "heading_setpoint_rad": value.heading_setpoint_rad,
            "heading_lag_rad": value.heading_lag_rad,
            "state_age_s": value.state_age_s,
        },
        separators=(",", ":"),
        allow_nan=False,
    )


def decode_orientation_telemetry(raw: bytes | str) -> OrientationTelemetrySample:
    """Validate and decode one orientation-telemetry message."""
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
        raise ValueError(f"invalid orientation telemetry JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("orientation telemetry must be an object")
    if (
        value.get("type") != ORIENTATION_TELEMETRY_TYPE
        or value.get("version") != ORIENTATION_TELEMETRY_VERSION
    ):
        raise ValueError("unsupported orientation telemetry type or version")
    try:
        sample = OrientationTelemetrySample(
            emitted_at_monotonic_s=value["emitted_at_monotonic_s"],
            actual_yaw_rad=value["actual_yaw_rad"],
            actual_heading_rad=value["actual_heading_rad"],
            heading_setpoint_rad=value["heading_setpoint_rad"],
            heading_lag_rad=value["heading_lag_rad"],
            state_age_s=value["state_age_s"],
        )
    except KeyError as exc:
        raise ValueError(f"orientation telemetry missing {exc.args[0]}") from exc
    return _validated_sample(sample)


class OrientationTracker:
    """Track measured base yaw in the executor's zero-based heading frame."""

    def __init__(self) -> None:
        self._origin_yaw_rad: float | None = None
        self._origin_unavailable = False
        self._actual_yaw_rad: float | None = None
        self._state_received_at_monotonic_s: float | None = None

    def _note_heading(self, heading_setpoint_rad: float) -> float:
        heading = _wrapped(
            _finite_float(heading_setpoint_rad, name="heading_setpoint_rad")
        )
        if self._origin_yaw_rad is None and abs(heading) > _HEADING_ZERO_TOLERANCE:
            self._origin_unavailable = True
        return heading

    def update_state(
        self,
        state: Mapping[str, Any],
        *,
        received_at_monotonic_s: float,
        heading_setpoint_rad: float,
    ) -> None:
        if not isinstance(state, Mapping) or "base_quat" not in state:
            raise ValueError("g1_debug state is missing base_quat")
        received_at = _finite_float(
            received_at_monotonic_s,
            name="received_at_monotonic_s",
        )
        heading = self._note_heading(heading_setpoint_rad)
        yaw = quaternion_yaw_wxyz(state["base_quat"])
        if (
            self._origin_yaw_rad is None
            and not self._origin_unavailable
            and abs(heading) <= _HEADING_ZERO_TOLERANCE
        ):
            self._origin_yaw_rad = yaw
        self._actual_yaw_rad = yaw
        self._state_received_at_monotonic_s = received_at

    def sample(
        self,
        now_monotonic_s: float,
        heading_setpoint_rad: float,
    ) -> OrientationTelemetrySample:
        now = _finite_float(now_monotonic_s, name="now_monotonic_s")
        heading = self._note_heading(heading_setpoint_rad)
        actual_heading = None
        heading_lag = None
        if self._actual_yaw_rad is not None and self._origin_yaw_rad is not None:
            actual_heading = _wrapped(self._actual_yaw_rad - self._origin_yaw_rad)
            heading_lag = _wrapped(heading - actual_heading)
        state_age = (
            None
            if self._state_received_at_monotonic_s is None
            else max(0.0, now - self._state_received_at_monotonic_s)
        )
        return OrientationTelemetrySample(
            emitted_at_monotonic_s=now,
            actual_yaw_rad=self._actual_yaw_rad,
            actual_heading_rad=actual_heading,
            heading_setpoint_rad=heading,
            heading_lag_rad=heading_lag,
            state_age_s=state_age,
        )


class LatestOrientationTelemetry:
    """Retain the newest decoded executor sample for the visual controller."""

    def __init__(self) -> None:
        self._sample: OrientationTelemetrySample | None = None

    def update(self, raw: bytes | str) -> None:
        sample = decode_orientation_telemetry(raw)
        if (
            self._sample is None
            or sample.emitted_at_monotonic_s >= self._sample.emitted_at_monotonic_s
        ):
            self._sample = sample

    def diagnostics(self, now_monotonic_s: float) -> dict[str, float | None] | None:
        if self._sample is None:
            return None
        now = _finite_float(now_monotonic_s, name="now_monotonic_s")
        sample = self._sample
        return {
            "actual_yaw_rad": sample.actual_yaw_rad,
            "actual_heading_rad": sample.actual_heading_rad,
            "heading_setpoint_rad": sample.heading_setpoint_rad,
            "heading_lag_rad": sample.heading_lag_rad,
            "state_age_s": sample.state_age_s,
            "telemetry_age_s": max(
                0.0,
                now - sample.emitted_at_monotonic_s,
            ),
        }
