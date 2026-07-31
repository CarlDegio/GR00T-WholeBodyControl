"""Validate and schedule Uni-LaViRA ObjectNav commands for Sonic planner mode."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

LOCOMOTION_MODE_SLOW_WALK = 1
_STOP_VECTOR = (0.0, 0.0, 0.0)
_ZERO_TOLERANCE = 1e-9


class CommandValidationError(ValueError):
    """Raised before motion when an ObjectNav command batch is unsafe."""


class PlannerBusyError(RuntimeError):
    """Raised when a second batch is started while one is active."""


@dataclass(frozen=True)
class VelocityCommand:
    vx: float
    vy: float
    wz: float
    duration: float


@dataclass(frozen=True)
class ObjectNavBatch:
    rotation: VelocityCommand
    translation: VelocityCommand


@dataclass(frozen=True)
class PlannerOutput:
    mode: int
    movement: tuple[float, float, float]
    facing: tuple[float, float, float]
    speed: float
    height: float
    phase: str


def _finite_number(command: Mapping[str, Any], field: str, index: int) -> float:
    if field not in command:
        raise CommandValidationError(f"commands[{index}] missing {field}")
    value = command[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CommandValidationError(f"commands[{index}].{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CommandValidationError(f"commands[{index}].{field} must be finite")
    return result


def _velocity_command(value: Any, index: int) -> VelocityCommand:
    if not isinstance(value, Mapping):
        raise CommandValidationError(f"commands[{index}] must be an object")
    return VelocityCommand(
        vx=_finite_number(value, "vx", index),
        vy=_finite_number(value, "vy", index),
        wz=_finite_number(value, "wz", index),
        duration=_finite_number(value, "duration", index),
    )


def validate_object_nav_batch(
    payload: Any,
    *,
    max_speed: float = 0.5,
    max_duration: float = 30.0,
    max_abs_yaw: float = math.pi,
) -> ObjectNavBatch:
    """Return a validated fixed rotate-then-translate command batch."""
    if not isinstance(payload, Mapping):
        raise CommandValidationError("request must be a JSON object")
    commands = payload.get("commands")
    if not isinstance(commands, list) or len(commands) != 2:
        raise CommandValidationError("commands must contain exactly two entries")
    limits = (max_speed, max_duration, max_abs_yaw)
    if not all(
        not isinstance(value, bool)
        and isinstance(value, (int, float)) and math.isfinite(value) and value > 0
        for value in limits
    ):
        raise ValueError("planner safety limits must be finite and positive")

    rotation = _velocity_command(commands[0], 0)
    translation = _velocity_command(commands[1], 1)
    for index, command in enumerate((rotation, translation)):
        if command.duration < 0:
            raise CommandValidationError(f"commands[{index}].duration must be non-negative")
        if command.duration > max_duration:
            raise CommandValidationError(f"commands[{index}].duration exceeds limit")

    if math.hypot(rotation.vx, rotation.vy) > _ZERO_TOLERANCE:
        raise CommandValidationError("commands[0] must be pure rotation")
    if abs(translation.wz) > _ZERO_TOLERANCE:
        raise CommandValidationError("commands[1] must be pure translation")
    if math.hypot(translation.vx, translation.vy) > max_speed:
        raise CommandValidationError("commands[1] speed exceeds limit")
    if abs(rotation.wz * rotation.duration) > max_abs_yaw:
        raise CommandValidationError("commands[0] yaw exceeds limit")

    return ObjectNavBatch(rotation=rotation, translation=translation)


class UniLaviraPlannerExecutor:
    """Advance one validated ObjectNav batch without blocking the caller."""

    def __init__(
        self,
        *,
        transition_pause: float = 0.5,
        max_speed: float = 0.5,
        max_duration: float = 30.0,
        max_abs_yaw: float = math.pi,
    ):
        if (
            not isinstance(transition_pause, (int, float))
            or not math.isfinite(transition_pause)
            or transition_pause < 0
        ):
            raise ValueError("transition_pause must be finite and non-negative")
        limits = (max_speed, max_duration, max_abs_yaw)
        if not all(
            not isinstance(value, bool)
            and isinstance(value, (int, float)) and math.isfinite(value) and value > 0
            for value in limits
        ):
            raise ValueError("planner safety limits must be finite and positive")
        self.transition_pause = float(transition_pause)
        self.max_speed = float(max_speed)
        self.max_duration = float(max_duration)
        self.max_abs_yaw = float(max_abs_yaw)
        self._heading_rad = 0.0
        self._phase = "stopped"
        self._phase_deadline = 0.0
        self._batch: ObjectNavBatch | None = None
        self._movement = _STOP_VECTOR
        self._speed = 0.0
        self._active = False
        self._just_completed = False
        self._abort_reason: str | None = None

    @property
    def active(self) -> bool:
        return self._active

    @property
    def just_completed(self) -> bool:
        return self._just_completed

    @property
    def heading_rad(self) -> float:
        return self._heading_rad

    @property
    def abort_reason(self) -> str | None:
        return self._abort_reason

    def start(self, payload: Any, now: float) -> PlannerOutput:
        if self._active:
            raise PlannerBusyError("a planner request is already active")
        batch = validate_object_nav_batch(
            payload,
            max_speed=self.max_speed,
            max_duration=self.max_duration,
            max_abs_yaw=self.max_abs_yaw,
        )
        timestamp = float(now)
        if not math.isfinite(timestamp):
            raise ValueError("now must be finite")

        self._batch = batch
        self._heading_rad += batch.rotation.wz * batch.rotation.duration
        self._movement, self._speed = self._translation_state(batch.translation)
        self._active = True
        self._just_completed = False
        self._abort_reason = None
        if batch.rotation.duration > 0:
            self._phase = "rotating"
            self._phase_deadline = timestamp + batch.rotation.duration
        else:
            self._phase = "transition_pause"
            self._phase_deadline = timestamp + self.transition_pause
        return self._output()

    def tick(self, now: float) -> PlannerOutput:
        timestamp = float(now)
        if not math.isfinite(timestamp):
            raise ValueError("now must be finite")
        self._just_completed = False
        while self._active and timestamp + 1e-12 >= self._phase_deadline:
            if self._phase == "rotating":
                self._phase = "transition_pause"
                self._phase_deadline += self.transition_pause
                continue
            if self._phase == "transition_pause":
                assert self._batch is not None
                if self._batch.translation.duration > 0:
                    self._phase = "translating"
                    self._phase_deadline += self._batch.translation.duration
                    continue
                self._complete()
                break
            if self._phase == "translating":
                self._complete()
                break
            raise RuntimeError(f"unknown planner phase: {self._phase}")
        return self._output()

    def abort(self, reason: str) -> PlannerOutput:
        self._active = False
        self._just_completed = False
        self._abort_reason = str(reason)
        self._phase = "stopped"
        return self._output()

    def reset_heading(self) -> None:
        if self._active:
            raise PlannerBusyError("cannot reset heading while a request is active")
        self._heading_rad = 0.0
        self._phase = "stopped"
        self._movement = _STOP_VECTOR
        self._speed = 0.0
        self._just_completed = False
        self._abort_reason = None

    def _complete(self) -> None:
        self._active = False
        self._just_completed = True
        self._phase = "stopped"

    def _facing(self) -> tuple[float, float, float]:
        return (math.cos(self._heading_rad), math.sin(self._heading_rad), 0.0)

    def _translation_state(
        self, command: VelocityCommand
    ) -> tuple[tuple[float, float, float], float]:
        speed = math.hypot(command.vx, command.vy)
        if speed <= _ZERO_TOLERANCE:
            return _STOP_VECTOR, 0.0
        facing = self._facing()
        world_x = command.vx * facing[0] - command.vy * facing[1]
        world_y = command.vx * facing[1] + command.vy * facing[0]
        return (world_x / speed, world_y / speed, 0.0), speed

    def _output(self) -> PlannerOutput:
        walking = self._active and self._phase == "translating"
        return PlannerOutput(
            mode=LOCOMOTION_MODE_SLOW_WALK,
            movement=self._movement if walking else _STOP_VECTOR,
            facing=self._facing(),
            speed=self._speed if walking else 0.0,
            height=-1.0,
            phase=self._phase,
        )


class PlannerModeTracker:
    """Mirror the operator's k/i/o control state from keyboard messages."""

    def __init__(self):
        self._running = False
        self._mode = "OFF"

    @property
    def running(self) -> bool:
        return self._running

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def planner_ready(self) -> bool:
        return self._running and self._mode == "PLANNER"

    def apply(self, command: str) -> str | None:
        normalized = str(command).strip().lower()
        if normalized == "k":
            if not self._running:
                self._running = True
                self._mode = "PLANNER"
                return None
            left_planner = self._mode == "PLANNER"
            self._running = False
            self._mode = "OFF"
            return "control_stopped" if left_planner else None

        if not self._running:
            return None
        if normalized == "i":
            if self._mode == "PLANNER":
                self._mode = "POSE"
                return "pose_mode_requested"
            self._mode = "POSE"
            return None
        if normalized == "o":
            self._mode = "PLANNER"
        return None




class UniLaviraJsonBridge:
    """Coordinate one REP exchange with the non-blocking planner executor."""

    def __init__(self, socket: Any, executor: UniLaviraPlannerExecutor):
        self.socket = socket
        self.executor = executor
        self.pending_reply = False
        self._completion_waiting = False
        self._has_output = False

    def step(self, *, now: float, planner_ready: bool) -> PlannerOutput | None:
        if self.pending_reply:
            if not planner_ready:
                return self.abort("left_planner_mode")
            output = self.executor.tick(now)
            if self.executor.just_completed:
                self._completion_waiting = True
            return output

        if self.socket.poll(0):
            try:
                request = self.socket.recv_json()
            except Exception as exc:
                self.socket.send_json(
                    {
                        "status": "rejected",
                        "reason": "invalid_json",
                        "detail": str(exc),
                    }
                )
                return self._held_output(now, planner_ready)

            if not planner_ready:
                self.socket.send_json(
                    {"status": "rejected", "reason": "not_in_planner"}
                )
                return self._held_output(now, planner_ready)

            try:
                output = self.executor.start(request, now)
            except (CommandValidationError, ValueError) as exc:
                self.socket.send_json(
                    {
                        "status": "rejected",
                        "reason": "invalid_commands",
                        "detail": str(exc),
                    }
                )
                return self._held_output(now, planner_ready)

            self._completion_waiting = False
            self.pending_reply = True
            self._has_output = True
            return output

        return self._held_output(now, planner_ready)

    def acknowledge_output_published(self) -> None:
        """Reply completed only after the terminal stopped output was sent."""
        if not self._completion_waiting:
            return
        if not self.pending_reply:
            raise RuntimeError("terminal planner output has no pending REP reply")
        self.socket.send_json(
            {
                "status": "completed",
                "heading_rad": self.executor.heading_rad,
            }
        )
        self.pending_reply = False
        self._completion_waiting = False

    def abort(self, reason: str) -> PlannerOutput | None:
        if not self.pending_reply and not self._has_output:
            return None
        output = self.executor.abort(reason)
        self._completion_waiting = False
        if self.pending_reply:
            self.pending_reply = False
            self.socket.send_json({"status": "aborted", "reason": str(reason)})
        return output

    def reset_control_session(self) -> None:
        if self.pending_reply:
            self.abort("control_session_reset")
        elif self.executor.active:
            self.executor.abort("control_session_reset")
        self.executor.reset_heading()
        self._completion_waiting = False
        self._has_output = False

    def _held_output(
        self, now: float, planner_ready: bool
    ) -> PlannerOutput | None:
        if not self._has_output or not planner_ready:
            return None
        return self.executor.tick(now)
