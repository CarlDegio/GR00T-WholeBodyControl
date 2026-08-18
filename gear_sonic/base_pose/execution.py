"""Convert validated base-pose plans into bounded planner velocities."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from gear_sonic.base_pose.policy import BasePoseResult, validate_base_pose_plan


STOP_VELOCITY = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class VelocityCommand:
    vx: float
    vy: float
    wz: float
    duration: float

    @property
    def velocity(self) -> tuple[float, float, float]:
        return self.vx, self.vy, self.wz


@dataclass(frozen=True)
class MotionSegment:
    action: str
    command: VelocityCommand


def plan_to_segments(
    plan: dict[str, Any],
    *,
    rotation_speed: float = 0.4,
    translation_speed: float = 0.3,
    rotation_scale: float = 1.0,
    translation_scale: float = 1.0,
) -> tuple[MotionSegment, ...]:
    """Map an agent-near plan to fixed-speed, time-bounded commands."""

    values = (
        rotation_speed,
        translation_speed,
        rotation_scale,
        translation_scale,
    )
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("base-pose speeds and scales must be finite and positive")
    validated = validate_base_pose_plan(plan)
    if validated["status"] != "ADJUST":
        return ()
    segments: list[MotionSegment] = []
    for item in validated["command_sequence"]:
        action = str(item["action"])
        scale = rotation_scale if action.startswith("ROTATE_") else translation_scale
        value = float(item["value"]) * scale
        if action == "ROTATE_LEFT":
            command = VelocityCommand(
                0.0, 0.0, rotation_speed, math.radians(value) / rotation_speed
            )
        elif action == "ROTATE_RIGHT":
            command = VelocityCommand(
                0.0, 0.0, -rotation_speed, math.radians(value) / rotation_speed
            )
        elif action == "MOVE_FORWARD":
            command = VelocityCommand(
                translation_speed, 0.0, 0.0, value / translation_speed
            )
        elif action == "MOVE_BACKWARD":
            command = VelocityCommand(
                -translation_speed, 0.0, 0.0, value / translation_speed
            )
        else:  # pragma: no cover - validation rejects unsupported actions
            raise ValueError(f"unsupported base-pose action: {action}")
        segments.append(MotionSegment(action=action, command=command))
    return tuple(segments)


class BasePoseSequenceController:
    """Advance one sequence and insert a zero-velocity pause between segments."""

    def __init__(
        self,
        *,
        rotation_speed: float = 0.4,
        translation_speed: float = 0.3,
        rotation_scale: float = 1.0,
        translation_scale: float = 1.0,
        transition_pause: float = 0.5,
        stop_duration: float = 0.05,
    ) -> None:
        values = (
            rotation_speed,
            translation_speed,
            rotation_scale,
            translation_scale,
            stop_duration,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("controller speeds, scales, and stop duration must be positive")
        if not math.isfinite(transition_pause) or transition_pause < 0.0:
            raise ValueError("transition_pause must be finite and non-negative")
        self.rotation_speed = float(rotation_speed)
        self.translation_speed = float(translation_speed)
        self.rotation_scale = float(rotation_scale)
        self.translation_scale = float(translation_scale)
        self.transition_pause = float(transition_pause)
        self.stop_duration = float(stop_duration)
        self.cancel()

    @property
    def active(self) -> bool:
        return self.phase != "idle"

    @property
    def segment_index(self) -> int | None:
        return self._segment_index if self.active else None

    @staticmethod
    def _timestamp(now: float) -> float:
        if isinstance(now, bool) or not isinstance(now, (int, float)):
            raise ValueError("now must be finite")
        value = float(now)
        if not math.isfinite(value):
            raise ValueError("now must be finite")
        return value

    def start(self, result: BasePoseResult, now: float) -> bool:
        if self.active:
            raise RuntimeError("base-pose sequence is already active")
        segments = plan_to_segments(
            result.plan,
            rotation_speed=self.rotation_speed,
            translation_speed=self.translation_speed,
            rotation_scale=self.rotation_scale,
            translation_scale=self.translation_scale,
        )
        if not segments:
            self.cancel()
            return False
        timestamp = self._timestamp(now)
        self._segments = segments
        self._segment_index = 0
        self.phase = "motion"
        self._deadline = timestamp + segments[0].command.duration
        return True

    def cancel(self) -> None:
        self.phase = "idle"
        self._segments: tuple[MotionSegment, ...] = ()
        self._segment_index = 0
        self._deadline = 0.0

    def stop_command(self) -> VelocityCommand:
        return VelocityCommand(*STOP_VELOCITY, self.stop_duration)

    def step(self, now: float) -> tuple[str, VelocityCommand]:
        timestamp = self._timestamp(now)
        while self.active and timestamp + 1.0e-12 >= self._deadline:
            if self.phase == "motion":
                if self._segment_index + 1 >= len(self._segments):
                    self.cancel()
                    return "hold", self.stop_command()
                self.phase = "pause"
                self._deadline = timestamp + self.transition_pause
                return "hold", self.stop_command()
            self._segment_index += 1
            self.phase = "motion"
            self._deadline = timestamp + self._segments[self._segment_index].command.duration
        if self.phase == "motion":
            segment = self._segments[self._segment_index]
            return segment.action, segment.command
        return "hold", self.stop_command()
