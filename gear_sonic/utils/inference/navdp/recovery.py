"""Pure malicious-drift detection used by the FAST-LIO lifecycle supervisor."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable, Mapping


@dataclass(frozen=True)
class SlamRecoveryLimits:
    """Thresholds that distinguish a broken pose estimate from normal motion."""

    startup_grace_s: float
    max_planar_speed_m_s: float
    max_position_step_m: float
    max_yaw_rate_rad_s: float
    consecutive_samples: int

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "SlamRecoveryLimits":
        limits = cls(
            startup_grace_s=float(values["startup_grace_s"]),
            max_planar_speed_m_s=float(values["max_planar_speed_m_s"]),
            max_position_step_m=float(values["max_position_step_m"]),
            max_yaw_rate_rad_s=float(values["max_yaw_rate_rad_s"]),
            consecutive_samples=int(values["consecutive_samples"]),
        )
        if limits.startup_grace_s < 0.0:
            raise ValueError("startup_grace_s cannot be negative")
        if limits.max_planar_speed_m_s <= 0.0:
            raise ValueError("max_planar_speed_m_s must be positive")
        if limits.max_position_step_m <= 0.0:
            raise ValueError("max_position_step_m must be positive")
        if limits.max_yaw_rate_rad_s <= 0.0:
            raise ValueError("max_yaw_rate_rad_s must be positive")
        if limits.consecutive_samples < 1:
            raise ValueError("consecutive_samples must be at least one")
        return limits


@dataclass(frozen=True)
class OdometrySample:
    """The subset of FAST-LIO odometry needed for drift detection."""

    timestamp_s: float
    x: float
    y: float
    yaw: float
    velocity_x: float
    velocity_y: float
    yaw_rate: float


def _wrapped_angle_delta(current: float, previous: float) -> float:
    return math.atan2(math.sin(current - previous), math.cos(current - previous))


class MaliciousDriftDetector:
    """Require persistent impossible motion while catching large jumps immediately."""

    def __init__(
        self,
        limits: SlamRecoveryLimits,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.limits = limits
        self._monotonic = monotonic
        self._started_s = float(monotonic())
        self._previous: OdometrySample | None = None
        self._speed_violations = 0
        self._yaw_rate_violations = 0

    def reset(self, *, started_s: float | None = None) -> None:
        self._started_s = (
            float(self._monotonic()) if started_s is None else float(started_s)
        )
        self._previous = None
        self._speed_violations = 0
        self._yaw_rate_violations = 0

    def observe(
        self,
        sample: OdometrySample,
        *,
        monotonic_s: float | None = None,
    ) -> str | None:
        now_s = float(self._monotonic()) if monotonic_s is None else float(monotonic_s)
        values = (
            sample.timestamp_s,
            sample.x,
            sample.y,
            sample.yaw,
            sample.velocity_x,
            sample.velocity_y,
            sample.yaw_rate,
        )
        if now_s - self._started_s < self.limits.startup_grace_s:
            self._previous = sample if all(math.isfinite(value) for value in values) else None
            return None
        if not all(math.isfinite(value) for value in values):
            return "non_finite_odometry"

        previous = self._previous
        self._previous = sample
        reported_speed = math.hypot(sample.velocity_x, sample.velocity_y)
        reported_yaw_rate = abs(sample.yaw_rate)
        derived_speed = 0.0
        derived_yaw_rate = 0.0
        if previous is not None:
            dt = sample.timestamp_s - previous.timestamp_s
            if dt <= 0.0:
                self._speed_violations = 0
                self._yaw_rate_violations = 0
                return None
            step = math.hypot(sample.x - previous.x, sample.y - previous.y)
            if step > self.limits.max_position_step_m:
                return f"position_jump:{step:.3f}m/{dt:.3f}s"
            derived_speed = step / dt
            derived_yaw_rate = abs(_wrapped_angle_delta(sample.yaw, previous.yaw)) / dt

        planar_speed = max(reported_speed, derived_speed)
        yaw_rate = max(reported_yaw_rate, derived_yaw_rate)
        self._speed_violations = (
            self._speed_violations + 1
            if planar_speed > self.limits.max_planar_speed_m_s
            else 0
        )
        self._yaw_rate_violations = (
            self._yaw_rate_violations + 1
            if yaw_rate > self.limits.max_yaw_rate_rad_s
            else 0
        )
        if self._speed_violations >= self.limits.consecutive_samples:
            return f"planar_speed:{planar_speed:.3f}m/s"
        if self._yaw_rate_violations >= self.limits.consecutive_samples:
            return f"yaw_rate:{yaw_rate:.3f}rad/s"
        return None
