"""Wire contracts shared by velocity producers and the final executor."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import time
from typing import Any, Literal, Mapping, Sequence

COMMAND_TYPE = "sonic_navigation_command"
STATUS_TYPE = "sonic_navigation_status"
VELOCITY_TYPE = "sonic_planner_velocity"


def _generation(value: int) -> int:
    result = int(value)
    if result < 0:
        raise ValueError("generation cannot be negative")
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
    mode: Literal["manual_velocity", "nav_goal", "stop"]
    generation: int
    timestamp: float
    velocity: tuple[float, float, float] | None = None
    goal_base: tuple[float, float] | None = None
    target: str = ""
    target_type: str = ""
    confidence: float = 0.0
    source: str = ""

    def __post_init__(self) -> None:
        if self.mode not in {"manual_velocity", "nav_goal", "stop"}:
            raise ValueError("invalid navigation mode")
        _generation(self.generation)
        _finite_scalar(self.timestamp, field="navigation timestamp")
        _finite_scalar(self.confidence, field="navigation confidence")
        if self.velocity is not None:
            _finite_tuple(self.velocity, field="velocity")
        if self.goal_base is not None:
            if len(self.goal_base) != 2:
                raise ValueError("goal_base requires two values")
            _finite_scalar(self.goal_base[0], field="goal_base.x")
            _finite_scalar(self.goal_base[1], field="goal_base.y")


def build_navigation_message(
    *,
    mode: str,
    generation: int,
    timestamp: float | None = None,
    velocity: Sequence[float] | None = None,
    goal_base: Sequence[float] | None = None,
    target: str = "",
    target_type: str = "",
    confidence: float = 0.0,
    source: str = "",
) -> str:
    if mode not in {"manual_velocity", "nav_goal", "stop"}:
        raise ValueError("invalid navigation mode")
    payload: dict[str, Any] = {
        "type": COMMAND_TYPE,
        "version": 1,
        "generation": _generation(generation),
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
    if source:
        payload["source"] = str(source)
    return json.dumps(payload, allow_nan=False)


def decode_navigation_message(message: str | bytes | Mapping[str, Any]) -> NavigationCommand:
    payload = json.loads(message) if isinstance(message, (str, bytes)) else dict(message)
    if payload.get("type") != COMMAND_TYPE or payload.get("version") != 1:
        raise ValueError("unsupported navigation command")
    mode = payload.get("mode")
    if mode not in {"manual_velocity", "nav_goal", "stop"}:
        raise ValueError("invalid navigation mode")
    velocity = payload.get("velocity")
    goal = payload.get("goal_base")
    return NavigationCommand(
        mode=mode,
        generation=int(payload["generation"]),
        timestamp=float(payload["timestamp"]),
        velocity=None
        if velocity is None
        else _finite_tuple(
            tuple(float(velocity[key]) for key in ("vx", "vy", "wz")),
            field="velocity",
        ),
        goal_base=None if goal is None else (float(goal["x"]), float(goal["y"])),
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
    heading_target_rad: float | None = None
    heading_reference_rad: float | None = None

    def __post_init__(self) -> None:
        _generation(self.generation)
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
        heading_target_rad=target,
        heading_reference_rad=reference,
    )
