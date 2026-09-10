"""Accumulate servo velocities into executable, bounded OVMM waypoints."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from gear_sonic.utils.inference.base_pose.servo import ServoCommand


@dataclass(frozen=True)
class ActionLimits:
    min_displacement_m: float = 0.1
    max_displacement_m: float = 1.0
    min_turn_degrees: float = 5.0
    max_turn_degrees: float = 180.0
    allow_lateral_movement: bool = True
    allow_simultaneous_turn: bool = True
    allow_back: bool = True
    discrete_movement: bool = False
    constraint_base_in_manip_mode: bool = False


class ResidualActionAdapter:
    def __init__(
        self,
        limits: ActionLimits,
        *,
        dt_s: float = 0.05,
        max_displacement_m: float = 0.25,
        max_turn_degrees: float = 30.0,
    ):
        self.limits = limits
        self.dt_s = float(dt_s)
        self.translation_cap = min(float(max_displacement_m), limits.max_displacement_m)
        self.rotation_cap = math.radians(
            min(float(max_turn_degrees), limits.max_turn_degrees)
        )
        self.translation_min = float(limits.min_displacement_m)
        self.rotation_min = math.radians(limits.min_turn_degrees)
        values = (
            self.dt_s,
            self.translation_cap,
            self.rotation_cap,
            self.translation_min,
            self.rotation_min,
        )
        if not all(math.isfinite(x) and x > 0 for x in values):
            raise ValueError("Action time and limits must be finite and positive")
        if (
            self.translation_min > self.translation_cap
            or self.rotation_min > self.rotation_cap
        ):
            raise ValueError(
                "The action cap cannot be below Habitat's effective minimum"
            )
        if not (
            limits.allow_lateral_movement
            and limits.allow_simultaneous_turn
            and limits.allow_back
        ):
            raise ValueError(
                "Faithful BasePose requires the standard OVMM lateral/turn/back actions"
            )
        if limits.discrete_movement or limits.constraint_base_in_manip_mode:
            raise ValueError(
                "Expected the standard continuous OVMM waypoint action configuration"
            )
        self.reset()

    def reset(self) -> None:
        self.residual = np.zeros(3, dtype=np.float64)
        self.previous = np.zeros(3, dtype=np.float64)
        self.phase = None

    @staticmethod
    def _above_float32_deadzone(
        value: np.ndarray, minimum: float, cap: float
    ) -> np.ndarray:
        # Habitat normalizes, casts to float32, then rescales. A value at the
        # boundary must not round just below its strict minimum-distance test.
        length = float(np.linalg.norm(value))
        desired = min(max(length, minimum * (1.0 + 1e-6)), cap)
        return value * (desired / length)

    def step(self, command: ServoCommand, *, phase: str) -> np.ndarray:
        velocity = np.asarray(command.velocity, dtype=np.float64)
        if not np.isfinite(velocity).all():
            self.reset()
            raise ValueError("Servo command must be finite")
        if phase != self.phase:
            self.reset()
            self.phase = phase
        clear = (velocity == 0) | (velocity * self.previous < 0)
        self.residual[clear] = 0.0
        self.previous = velocity.copy()
        self.residual += velocity * self.dt_s
        waypoint = np.zeros(3, dtype=np.float64)
        norm = float(np.linalg.norm(self.residual[:2]))
        if norm + 1e-12 >= self.translation_min:
            # Conservative norm cap also respects Habitat's per-axis cap.
            waypoint[:2] = self._above_float32_deadzone(
                self.residual[:2], self.translation_min, self.translation_cap
            )
            self.residual[:2] -= waypoint[:2]
            self.residual[:2][self.residual[:2] * velocity[:2] < 0] = 0.0
        angle = float(self.residual[2])
        if abs(angle) + 1e-12 >= self.rotation_min:
            waypoint[2] = math.copysign(
                min(max(abs(angle), self.rotation_min * (1 + 1e-6)), self.rotation_cap),
                angle,
            )
            self.residual[2] -= waypoint[2]
            if self.residual[2] * velocity[2] < 0:
                self.residual[2] = 0.0
        return waypoint
