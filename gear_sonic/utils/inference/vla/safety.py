"""Fail-closed VLA POSE safety gate shared with planner velocity execution."""

from __future__ import annotations

from dataclasses import dataclass
import math

from gear_sonic.utils.planner_control.executor import SafetySnapshot
from gear_sonic.utils.planner_control.safety import motion_safety_reason


@dataclass(frozen=True)
class VlaSafetyGate:
    radar_timeout_s: float = 0.75
    robot_state_timeout_s: float = 0.75

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (self.radar_timeout_s, self.robot_state_timeout_s)
        ):
            raise ValueError("VLA safety timeouts must be finite and positive")

    def reason(
        self,
        *,
        now: float,
        safety: SafetySnapshot,
        robot_state_timestamp_s: float,
    ) -> str:
        state_age = float(now) - float(robot_state_timestamp_s)
        if (
            robot_state_timestamp_s <= 0.0
            or state_age < 0.0
            or state_age > self.robot_state_timeout_s
        ):
            return "robot_state_timeout"
        return motion_safety_reason(
            now=now,
            radar_timestamp_s=safety.radar_timestamp_s,
            radar_timeout_s=self.radar_timeout_s,
            depth_m=safety.depth_m,
        )

    def ready(self, **kwargs: object) -> bool:
        return self.reason(**kwargs) == "clear"  # type: ignore[arg-type]
