"""Wire contracts shared by velocity producers and the final executor."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import time
from typing import Any, Literal, Mapping, Sequence, cast

COMMAND_TYPE = "sonic_navigation_command"
STATUS_TYPE = "sonic_navigation_status"
VELOCITY_TYPE = "sonic_planner_velocity"
RUNTIME_STATUS_TYPE = "sonic_navigation_runtime_status"


def _generation(value: int) -> int:
    result = int(value)
    if result < 0:
        raise ValueError("generation cannot be negative")
    return result


def _segment_id(value: int) -> int:
    result = int(value)
    if result < 0:
        raise ValueError("segment_id cannot be negative")
    return result


def _skill_id(value: int) -> int:
    result = int(value)
    if result < 0:
        raise ValueError("skill_id cannot be negative")
    return result


def _finite_scalar(value: float, *, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _finite_tuple(values: Sequence[float], *, field: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{field} requires three values")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{field} must be finite")
    return result  # type: ignore[return-value]


@dataclass(frozen=True)
class NavigationCommand:
    mode: Literal["manual_velocity", "nav_goal", "heading_goal", "stop"]
    generation: int
    timestamp: float
    segment_id: int = 0
    skill_id: int = 0
    velocity: tuple[float, float, float] | None = None
    goal_base: tuple[float, float] | None = None
    heading_delta_rad: float | None = None
    heading_turn_direction: Literal["left", "right"] | None = None
    heading_max_angular_speed_rad_s: float | None = None
    heading_max_duration_s: float | None = None
    target: str = ""
    target_type: str = ""
    confidence: float = 0.0
    source: str = ""

    def __post_init__(self) -> None:
        if self.mode not in {"manual_velocity", "nav_goal", "heading_goal", "stop"}:
            raise ValueError("invalid navigation mode")
        _generation(self.generation)
        _segment_id(self.segment_id)
        _skill_id(self.skill_id)
        _finite_scalar(self.timestamp, field="navigation timestamp")
        _finite_scalar(self.confidence, field="navigation confidence")
        if self.velocity is not None:
            _finite_tuple(self.velocity, field="velocity")
        if self.goal_base is not None:
            if len(self.goal_base) != 2:
                raise ValueError("goal_base requires two values")
            _finite_scalar(self.goal_base[0], field="goal_base.x")
            _finite_scalar(self.goal_base[1], field="goal_base.y")
        if self.heading_delta_rad is not None:
            _finite_scalar(self.heading_delta_rad, field="heading_delta_rad")
        if self.heading_turn_direction not in {None, "left", "right"}:
            raise ValueError("heading_turn_direction must be left or right")
        if self.heading_max_angular_speed_rad_s is not None:
            speed = _finite_scalar(
                self.heading_max_angular_speed_rad_s,
                field="heading_max_angular_speed_rad_s",
            )
            if speed <= 0.0:
                raise ValueError(
                    "heading_max_angular_speed_rad_s must be positive"
                )
        if self.heading_max_duration_s is not None:
            duration = _finite_scalar(
                self.heading_max_duration_s,
                field="heading_max_duration_s",
            )
            if duration <= 0.0:
                raise ValueError("heading_max_duration_s must be positive")
        if self.mode == "heading_goal" and self.heading_delta_rad is None:
            raise ValueError("heading_goal requires heading_delta_rad")


def build_navigation_message(
    *,
    mode: str,
    generation: int,
    segment_id: int = 0,
    skill_id: int = 0,
    timestamp: float | None = None,
    velocity: Sequence[float] | None = None,
    goal_base: Sequence[float] | None = None,
    heading_delta_rad: float | None = None,
    heading_turn_direction: Literal["left", "right"] | None = None,
    heading_max_angular_speed_rad_s: float | None = None,
    heading_max_duration_s: float | None = None,
    target: str = "",
    target_type: str = "",
    confidence: float = 0.0,
    source: str = "",
) -> str:
    if mode not in {"manual_velocity", "nav_goal", "heading_goal", "stop"}:
        raise ValueError("invalid navigation mode")
    payload: dict[str, Any] = {
        "type": COMMAND_TYPE,
        "version": 1,
        "generation": _generation(generation),
        "segment_id": _segment_id(segment_id),
        "skill_id": _skill_id(skill_id),
        "mode": mode,
        "timestamp": _finite_scalar(
            time.time() if timestamp is None else timestamp,
            field="navigation timestamp",
        ),
    }
    if velocity is not None:
        payload["velocity"] = dict(
            zip(("vx", "vy", "wz"), _finite_tuple(velocity, field="velocity"))
        )
    if goal_base is not None:
        if len(goal_base) != 2:
            raise ValueError("goal_base requires two values")
        payload["goal_base"] = {
            "x": _finite_scalar(goal_base[0], field="goal_base.x"),
            "y": _finite_scalar(goal_base[1], field="goal_base.y"),
        }
        payload.update(
            target=str(target),
            target_type=str(target_type),
            confidence=_finite_scalar(confidence, field="navigation confidence"),
        )
    if heading_delta_rad is not None:
        payload["heading_delta_rad"] = _finite_scalar(
            heading_delta_rad, field="heading_delta_rad"
        )
    if heading_turn_direction is not None:
        if heading_turn_direction not in {"left", "right"}:
            raise ValueError("heading_turn_direction must be left or right")
        payload["heading_turn_direction"] = heading_turn_direction
    if heading_max_angular_speed_rad_s is not None:
        speed = _finite_scalar(
            heading_max_angular_speed_rad_s,
            field="heading_max_angular_speed_rad_s",
        )
        if speed <= 0.0:
            raise ValueError("heading_max_angular_speed_rad_s must be positive")
        payload["heading_max_angular_speed_rad_s"] = speed
    if heading_max_duration_s is not None:
        duration = _finite_scalar(
            heading_max_duration_s, field="heading_max_duration_s"
        )
        if duration <= 0.0:
            raise ValueError("heading_max_duration_s must be positive")
        payload["heading_max_duration_s"] = duration
    if mode == "heading_goal" and heading_delta_rad is None:
        raise ValueError("heading_goal requires heading_delta_rad")
    if source:
        payload["source"] = str(source)
    return json.dumps(payload, allow_nan=False)


def decode_navigation_message(message: str | bytes | Mapping[str, Any]) -> NavigationCommand:
    payload = json.loads(message) if isinstance(message, (str, bytes)) else dict(message)
    if payload.get("type") != COMMAND_TYPE or payload.get("version") != 1:
        raise ValueError("unsupported navigation command")
    mode = payload.get("mode")
    if mode not in {"manual_velocity", "nav_goal", "heading_goal", "stop"}:
        raise ValueError("invalid navigation mode")
    velocity = payload.get("velocity")
    goal = payload.get("goal_base")
    return NavigationCommand(
        mode=mode,
        generation=int(payload["generation"]),
        timestamp=float(payload["timestamp"]),
        segment_id=int(payload.get("segment_id", 0)),
        skill_id=int(payload.get("skill_id", 0)),
        velocity=None
        if velocity is None
        else _finite_tuple(
            tuple(float(velocity[key]) for key in ("vx", "vy", "wz")),
            field="velocity",
        ),
        goal_base=None if goal is None else (float(goal["x"]), float(goal["y"])),
        heading_delta_rad=(
            None
            if payload.get("heading_delta_rad") is None
            else float(payload["heading_delta_rad"])
        ),
        heading_turn_direction=(
            None
            if payload.get("heading_turn_direction") is None
            else cast(
                Literal["left", "right"],
                str(payload["heading_turn_direction"]),
            )
        ),
        heading_max_angular_speed_rad_s=(
            None
            if payload.get("heading_max_angular_speed_rad_s") is None
            else float(payload["heading_max_angular_speed_rad_s"])
        ),
        heading_max_duration_s=(
            None
            if payload.get("heading_max_duration_s") is None
            else float(payload["heading_max_duration_s"])
        ),
        target=str(payload.get("target", "")),
        target_type=str(payload.get("target_type", "")),
        confidence=float(payload.get("confidence", 0.0)),
        source=str(payload.get("source", "")),
    )


@dataclass(frozen=True)
class PlannerVelocityCommand:
    generation: int
    timestamp: float
    source: str
    velocity: tuple[float, float, float]
    segment_id: int = 0
    skill_id: int = 0
    heading_target_rad: float | None = None
    heading_reference_rad: float | None = None

    def __post_init__(self) -> None:
        _generation(self.generation)
        _segment_id(self.segment_id)
        _skill_id(self.skill_id)
        _finite_scalar(self.timestamp, field="planner velocity timestamp")
        if not self.source:
            raise ValueError("planner velocity source cannot be empty")
        _finite_tuple(self.velocity, field="planner velocity")
        if (self.heading_target_rad is None) != (self.heading_reference_rad is None):
            raise ValueError("heading target and reference must be supplied together")
        if self.heading_target_rad is not None:
            _finite_scalar(self.heading_target_rad, field="heading target")
            _finite_scalar(self.heading_reference_rad, field="heading reference")


def build_planner_velocity_message(
    *,
    generation: int,
    segment_id: int = 0,
    skill_id: int = 0,
    source: str,
    velocity: Sequence[float],
    timestamp: float | None = None,
    heading_target_rad: float | None = None,
    heading_reference_rad: float | None = None,
) -> str:
    if not source:
        raise ValueError("planner velocity source cannot be empty")
    command = _finite_tuple(velocity, field="planner velocity")
    payload: dict[str, Any] = {
        "type": VELOCITY_TYPE,
        "version": 1,
        "generation": _generation(generation),
        "segment_id": _segment_id(segment_id),
        "skill_id": _skill_id(skill_id),
        "timestamp": _finite_scalar(
            time.time() if timestamp is None else timestamp,
            field="planner velocity timestamp",
        ),
        "source": str(source),
        "velocity": dict(zip(("vx", "vy", "wz"), command)),
    }
    if (heading_target_rad is None) != (heading_reference_rad is None):
        raise ValueError("heading target and reference must be supplied together")
    if heading_target_rad is not None:
        target = float(heading_target_rad)
        reference = float(heading_reference_rad)
        if not math.isfinite(target) or not math.isfinite(reference):
            raise ValueError("planner headings must be finite")
        payload["heading"] = {"target_rad": target, "reference_rad": reference}
    return json.dumps(payload, separators=(",", ":"), allow_nan=False)


def decode_planner_velocity_message(
    message: str | bytes | Mapping[str, Any],
) -> PlannerVelocityCommand:
    payload = json.loads(message) if isinstance(message, (str, bytes)) else dict(message)
    if payload.get("type") != VELOCITY_TYPE or payload.get("version") != 1:
        raise ValueError("unsupported planner velocity command")
    source = str(payload.get("source", ""))
    if not source:
        raise ValueError("planner velocity source cannot be empty")
    raw_velocity = payload.get("velocity")
    if not isinstance(raw_velocity, Mapping):
        raise ValueError("planner velocity object is missing")
    velocity = _finite_tuple(
        tuple(float(raw_velocity[key]) for key in ("vx", "vy", "wz")),
        field="planner velocity",
    )
    heading = payload.get("heading")
    target: float | None = None
    reference: float | None = None
    if heading is not None:
        if not isinstance(heading, Mapping):
            raise ValueError("planner heading must be an object")
        target = float(heading["target_rad"])
        reference = float(heading["reference_rad"])
        if not math.isfinite(target) or not math.isfinite(reference):
            raise ValueError("planner headings must be finite")
    return PlannerVelocityCommand(
        generation=int(payload["generation"]),
        timestamp=float(payload["timestamp"]),
        source=source,
        velocity=velocity,
        segment_id=int(payload.get("segment_id", 0)),
        skill_id=int(payload.get("skill_id", 0)),
        heading_target_rad=target,
        heading_reference_rad=reference,
    )


@dataclass(frozen=True)
class NavigationRuntimeStatus:
    """Final planner command after the shared executor safety boundary."""

    generation: int
    timestamp: float
    mode: Literal["manual_velocity", "nav_goal", "heading_goal", "stop"]
    source: str
    requested_velocity: tuple[float, float, float]
    velocity: tuple[float, float, float]
    reason: str
    segment_id: int = 0
    skill_id: int = 0

    def __post_init__(self) -> None:
        _generation(self.generation)
        _segment_id(self.segment_id)
        _skill_id(self.skill_id)
        _finite_scalar(self.timestamp, field="runtime status timestamp")
        if self.mode not in {"manual_velocity", "nav_goal", "heading_goal", "stop"}:
            raise ValueError("invalid runtime navigation mode")
        if not self.source:
            raise ValueError("runtime status source cannot be empty")
        _finite_tuple(self.requested_velocity, field="requested velocity")
        _finite_tuple(self.velocity, field="final velocity")
        if not self.reason:
            raise ValueError("runtime status reason cannot be empty")


def build_navigation_runtime_status_message(
    *,
    generation: int,
    segment_id: int = 0,
    skill_id: int = 0,
    mode: str,
    source: str,
    requested_velocity: Sequence[float],
    velocity: Sequence[float],
    reason: str,
    timestamp: float | None = None,
) -> str:
    if mode not in {"manual_velocity", "nav_goal", "heading_goal", "stop"}:
        raise ValueError("invalid runtime navigation mode")
    status = NavigationRuntimeStatus(
        generation=_generation(generation),
        segment_id=_segment_id(segment_id),
        skill_id=_skill_id(skill_id),
        timestamp=_finite_scalar(
            time.time() if timestamp is None else timestamp,
            field="runtime status timestamp",
        ),
        mode=cast(Literal["manual_velocity", "nav_goal", "heading_goal", "stop"], mode),
        source=str(source),
        requested_velocity=_finite_tuple(
            requested_velocity,
            field="requested velocity",
        ),
        velocity=_finite_tuple(velocity, field="final velocity"),
        reason=str(reason),
    )
    return json.dumps(
        {
            "type": RUNTIME_STATUS_TYPE,
            "version": 1,
            "generation": status.generation,
            "segment_id": status.segment_id,
            "skill_id": status.skill_id,
            "timestamp": status.timestamp,
            "mode": status.mode,
            "source": status.source,
            "requested_velocity": dict(
                zip(("vx", "vy", "wz"), status.requested_velocity)
            ),
            "velocity": dict(zip(("vx", "vy", "wz"), status.velocity)),
            "reason": status.reason,
        },
        separators=(",", ":"),
        allow_nan=False,
    )


def decode_navigation_runtime_status_message(
    message: str | bytes | Mapping[str, Any],
) -> NavigationRuntimeStatus:
    payload = json.loads(message) if isinstance(message, (str, bytes)) else dict(message)
    if payload.get("type") != RUNTIME_STATUS_TYPE or payload.get("version") != 1:
        raise ValueError("unsupported navigation runtime status")
    requested = payload.get("requested_velocity")
    velocity = payload.get("velocity")
    if not isinstance(requested, Mapping) or not isinstance(velocity, Mapping):
        raise ValueError("runtime status velocity objects are missing")
    mode = payload.get("mode")
    if mode not in {"manual_velocity", "nav_goal", "heading_goal", "stop"}:
        raise ValueError("invalid runtime navigation mode")
    return NavigationRuntimeStatus(
        generation=int(payload["generation"]),
        timestamp=float(payload["timestamp"]),
        segment_id=int(payload.get("segment_id", 0)),
        skill_id=int(payload.get("skill_id", 0)),
        mode=mode,
        source=str(payload.get("source", "")),
        requested_velocity=_finite_tuple(
            tuple(float(requested[name]) for name in ("vx", "vy", "wz")),
            field="requested velocity",
        ),
        velocity=_finite_tuple(
            tuple(float(velocity[name]) for name in ("vx", "vy", "wz")),
            field="final velocity",
        ),
        reason=str(payload.get("reason", "")),
    )
