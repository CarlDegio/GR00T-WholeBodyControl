"""Pure arbitration and safety core for final planner velocity execution."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np

from gear_sonic.runtime.protocol import NavigationCommand, PlannerVelocityCommand
from gear_sonic.utils.planner_control.safety import motion_safety_reason
from gear_sonic.utils.planner_control.sonic import SonicPlannerState


@dataclass(frozen=True)
class SafetySnapshot:
    radar_timestamp_s: float = 0.0
    depth_m: np.ndarray | None = None


@dataclass(frozen=True)
class PlannerExecutorDecision:
    generation: int
    segment_id: int
    skill_id: int
    source: str
    requested_velocity: tuple[float, float, float]
    velocity: tuple[float, float, float]
    reason: str
    message: bytes


class PlannerVelocityExecutorCore:
    """Select a producer, apply common safety, and build one SONIC packet."""

    def __init__(
        self,
        *,
        control_hz: float = 20.0,
        manual_timeout_s: float = 0.70,
        navdp_timeout_s: float = 0.30,
        radar_timeout_s: float = 0.75,
        sonic: SonicPlannerState | None = None,
    ) -> None:
        values = (
            control_hz,
            manual_timeout_s,
            navdp_timeout_s,
            radar_timeout_s,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("planner executor timing values must be finite and positive")
        self.period_s = 1.0 / float(control_hz)
        self.manual_timeout_s = float(manual_timeout_s)
        self.navdp_timeout_s = float(navdp_timeout_s)
        self.radar_timeout_s = float(radar_timeout_s)
        self.sonic = sonic or SonicPlannerState()
        self.generation = 0
        self.segment_id = 0
        self.skill_id = 0
        self.mode: Literal[
            "stop", "manual_velocity", "nav_goal", "heading_goal"
        ] = "stop"
        self.source = "stop"
        self.velocity = (0.0, 0.0, 0.0)
        self.command_timestamp_s = 0.0
        self.navdp_heading_target_rad: float | None = None
        self.navdp_heading_reference_rad: float | None = None
        self.navdp_heading_offset_rad: float | None = None

    def accept_navigation(self, command: NavigationCommand, *, now: float) -> bool:
        if command.generation < self.generation or (
            command.generation == self.generation
            and (command.skill_id, command.segment_id)
            < (self.skill_id, self.segment_id)
        ):
            return False
        self.generation = command.generation
        self.segment_id = command.segment_id
        self.skill_id = command.skill_id
        self.mode = command.mode
        self.command_timestamp_s = float(now)
        self.navdp_heading_target_rad = None
        self.navdp_heading_reference_rad = None
        self.navdp_heading_offset_rad = None
        if command.mode == "manual_velocity":
            self.source = command.source or "manual"
            self.velocity = command.velocity or (0.0, 0.0, 0.0)
        elif command.mode in {"nav_goal", "heading_goal"}:
            self.source = "navdp"
            self.velocity = (0.0, 0.0, 0.0)
        else:
            self.source = "stop"
            self.velocity = (0.0, 0.0, 0.0)
        return True

    def accept_planner_velocity(
        self, command: PlannerVelocityCommand, *, now: float
    ) -> bool:
        if (
            self.mode != "nav_goal"
            and self.mode != "heading_goal"
            or command.generation != self.generation
            or command.segment_id != self.segment_id
            or command.skill_id != self.skill_id
            or command.source != "navdp"
        ):
            return False
        self.velocity = command.velocity
        self.command_timestamp_s = float(now)
        self.navdp_heading_target_rad = command.heading_target_rad
        self.navdp_heading_reference_rad = command.heading_reference_rad
        return True

    def _requested(self, now: float) -> tuple[tuple[float, float, float], str]:
        if self.mode == "stop":
            return (0.0, 0.0, 0.0), "stopped"
        timeout = (
            self.manual_timeout_s if self.mode == "manual_velocity" else self.navdp_timeout_s
        )
        if float(now) - self.command_timestamp_s > timeout:
            return (0.0, 0.0, 0.0), f"{self.source}_velocity_timeout"
        return self.velocity, "clear"

    def decide(
        self,
        *,
        now: float,
        safety: SafetySnapshot,
    ) -> PlannerExecutorDecision:
        requested, reason = self._requested(now)
        velocity = requested
        safety_reason = motion_safety_reason(
            now=now,
            radar_timestamp_s=safety.radar_timestamp_s,
            radar_timeout_s=self.radar_timeout_s,
            depth_m=safety.depth_m,
        )
        if safety_reason == "radar_timeout":
            velocity = (0.0, 0.0, 0.0)
            reason = "radar_timeout"
        elif safety_reason == "depth_hard_stop":
            velocity = (0.0, 0.0, 0.0)
            reason = "depth_hard_stop"

        # A blocked/stale cycle must freeze both translation and the SONIC
        # heading setpoint.  Otherwise a fresh heading target could still turn
        # the robot while lidar/depth safety is holding velocity at zero.
        if (
            self.mode in {"nav_goal", "heading_goal"}
            and reason == "clear"
            and self.navdp_heading_target_rad is not None
        ):
            if self.navdp_heading_offset_rad is None:
                assert self.navdp_heading_reference_rad is not None
                self.navdp_heading_offset_rad = math.remainder(
                    self.sonic.heading - self.navdp_heading_reference_rad,
                    2.0 * math.pi,
                )
            self.sonic.heading = math.remainder(
                self.navdp_heading_target_rad + self.navdp_heading_offset_rad,
                2.0 * math.pi,
            )
        message = self.sonic.message(
            velocity,
            dt=self.period_s if self.mode == "manual_velocity" else 0.0,
        )
        return PlannerExecutorDecision(
            generation=self.generation,
            segment_id=self.segment_id,
            skill_id=self.skill_id,
            source=self.source,
            requested_velocity=requested,
            velocity=velocity,
            reason=reason,
            message=message,
        )
