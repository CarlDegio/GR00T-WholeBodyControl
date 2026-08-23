"""Transport-neutral navigation command and status contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import time
from typing import Any, Literal, Mapping, Sequence, cast


COMMAND_TYPE = "sonic_navigation_command"
STATUS_TYPE = "sonic_navigation_status"
VELOCITY_TYPE = "sonic_planner_velocity"
RUNTIME_STATUS_TYPE = "sonic_navigation_runtime_status"
NavigationMode = Literal["manual_velocity", "nav_goal", "heading_goal", "stop"]
TurnDirection = Literal["left", "right"]
_MODES = {"manual_velocity", "nav_goal", "heading_goal", "stop"}


def _nonnegative(value: int, field: str) -> int:
    result = int(value)
    if result < 0:
        raise ValueError(f"{field} cannot be negative")
    return result


def _finite(value: float, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _vector(values: Sequence[float], field: str, size: int = 3) -> tuple[float, ...]:
    if len(values) != size:
        raise ValueError(f"{field} requires {size} values")
    return tuple(_finite(value, field) for value in values)


def _mode(value: str, field: str = "navigation mode") -> NavigationMode:
    if value not in _MODES:
        raise ValueError(f"invalid {field}")
    return cast(NavigationMode, value)


def _decode(
    message: str | bytes | Mapping[str, Any], expected_type: str
) -> dict[str, Any]:
    value = json.loads(message) if isinstance(message, (str, bytes)) else dict(message)
    if value.get("type") != expected_type or value.get("version") != 1:
        raise ValueError(f"unsupported {expected_type}")
    value.pop("type")
    value.pop("version")
    return value


def _encode(type_name: str, value: Any) -> dict[str, Any]:
    return {"type": type_name, "version": 1, **asdict(value)}


def _json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), allow_nan=False)


def _timestamped(values: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(values)
    if result.get("timestamp") is None:
        result["timestamp"] = time.time()
    return result


@dataclass(frozen=True)
class NavigationCommand:
    mode: NavigationMode
    generation: int
    timestamp: float
    segment_id: int = 0
    skill_id: int = 0
    velocity: tuple[float, float, float] | None = None
    goal_base: tuple[float, float] | None = None
    heading_delta_rad: float | None = None
    heading_turn_direction: TurnDirection | None = None
    heading_max_angular_speed_rad_s: float | None = None
    heading_max_duration_s: float | None = None
    target: str = ""
    target_type: str = ""
    confidence: float = 0.0
    source: str = ""

    def __post_init__(self) -> None:
        _mode(self.mode)
        for name in ("generation", "segment_id", "skill_id"):
            _nonnegative(getattr(self, name), name)
        _finite(self.timestamp, "navigation timestamp")
        _finite(self.confidence, "navigation confidence")
        if self.velocity is not None:
            _vector(self.velocity, "velocity")
        if self.goal_base is not None:
            _vector(self.goal_base, "goal_base", 2)
        for name in (
            "heading_delta_rad",
            "heading_max_angular_speed_rad_s",
            "heading_max_duration_s",
        ):
            value = getattr(self, name)
            if value is not None:
                _finite(value, name)
        if self.heading_turn_direction not in {None, "left", "right"}:
            raise ValueError("heading_turn_direction must be left or right")
        if (
            self.heading_max_angular_speed_rad_s is not None
            and self.heading_max_angular_speed_rad_s <= 0
        ):
            raise ValueError("heading_max_angular_speed_rad_s must be positive")
        if self.heading_max_duration_s is not None and self.heading_max_duration_s <= 0:
            raise ValueError("heading_max_duration_s must be positive")
        if self.mode == "heading_goal" and self.heading_delta_rad is None:
            raise ValueError("heading_goal requires heading_delta_rad")

    def to_dict(self) -> dict[str, Any]:
        payload = _encode(COMMAND_TYPE, self)
        if self.velocity is None:
            payload.pop("velocity")
        else:
            payload["velocity"] = dict(zip(("vx", "vy", "wz"), self.velocity))
        if self.goal_base is None:
            for name in ("goal_base", "target", "target_type", "confidence"):
                payload.pop(name)
        else:
            payload["goal_base"] = dict(zip(("x", "y"), self.goal_base))
        for name in (
            "heading_delta_rad",
            "heading_turn_direction",
            "heading_max_angular_speed_rad_s",
            "heading_max_duration_s",
        ):
            if payload[name] is None:
                payload.pop(name)
        if not self.source:
            payload.pop("source")
        return payload

    def to_json(self) -> str:
        return _json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NavigationCommand":
        value = _decode(payload, COMMAND_TYPE)
        velocity = value.pop("velocity", None)
        goal = value.pop("goal_base", None)
        if velocity is not None:
            if not isinstance(velocity, Mapping):
                raise ValueError("navigation velocity must be an object")
            value["velocity"] = tuple(velocity[name] for name in ("vx", "vy", "wz"))
        if goal is not None:
            if not isinstance(goal, Mapping):
                raise ValueError("navigation goal must be an object")
            value["goal_base"] = (goal["x"], goal["y"])
        value["mode"] = _mode(str(value.get("mode")))
        if value.get("heading_turn_direction") is not None:
            value["heading_turn_direction"] = str(value["heading_turn_direction"])
        return cls(**value)

    @classmethod
    def from_json(cls, message: str | bytes) -> "NavigationCommand":
        return cls.from_dict(json.loads(message))


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
        for name in ("generation", "segment_id", "skill_id"):
            _nonnegative(getattr(self, name), name)
        _finite(self.timestamp, "planner velocity timestamp")
        if not self.source:
            raise ValueError("planner velocity source cannot be empty")
        _vector(self.velocity, "planner velocity")
        if (self.heading_target_rad is None) != (self.heading_reference_rad is None):
            raise ValueError("heading target and reference must be supplied together")
        if self.heading_target_rad is not None:
            _finite(self.heading_target_rad, "heading target")
            _finite(self.heading_reference_rad, "heading reference")

    def to_dict(self) -> dict[str, Any]:
        payload = _encode(VELOCITY_TYPE, self)
        payload["velocity"] = dict(zip(("vx", "vy", "wz"), self.velocity))
        payload["heading"] = (
            None
            if self.heading_target_rad is None
            else {
                "target_rad": self.heading_target_rad,
                "reference_rad": self.heading_reference_rad,
            }
        )
        payload.pop("heading_target_rad")
        payload.pop("heading_reference_rad")
        if payload["heading"] is None:
            payload.pop("heading")
        return payload

    def to_json(self) -> str:
        return _json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlannerVelocityCommand":
        value = _decode(payload, VELOCITY_TYPE)
        velocity = value.pop("velocity", None)
        if not isinstance(velocity, Mapping):
            raise ValueError("planner velocity object is missing")
        value["velocity"] = tuple(velocity[name] for name in ("vx", "vy", "wz"))
        heading = value.pop("heading", None)
        if heading is not None:
            if not isinstance(heading, Mapping):
                raise ValueError("planner heading must be an object")
            value["heading_target_rad"] = heading["target_rad"]
            value["heading_reference_rad"] = heading["reference_rad"]
        return cls(**value)

    @classmethod
    def from_json(cls, message: str | bytes) -> "PlannerVelocityCommand":
        return cls.from_dict(json.loads(message))


@dataclass(frozen=True)
class NavigationRuntimeStatus:
    generation: int
    timestamp: float
    mode: NavigationMode
    source: str
    requested_velocity: tuple[float, float, float]
    velocity: tuple[float, float, float]
    reason: str
    segment_id: int = 0
    skill_id: int = 0

    def __post_init__(self) -> None:
        for name in ("generation", "segment_id", "skill_id"):
            _nonnegative(getattr(self, name), name)
        _finite(self.timestamp, "runtime status timestamp")
        _mode(self.mode, "runtime navigation mode")
        if not self.source or not self.reason:
            raise ValueError("runtime status source and reason cannot be empty")
        _vector(self.requested_velocity, "requested velocity")
        _vector(self.velocity, "final velocity")

    def to_dict(self) -> dict[str, Any]:
        payload = _encode(RUNTIME_STATUS_TYPE, self)
        payload["requested_velocity"] = dict(
            zip(("vx", "vy", "wz"), self.requested_velocity)
        )
        payload["velocity"] = dict(zip(("vx", "vy", "wz"), self.velocity))
        return payload

    def to_json(self) -> str:
        return _json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NavigationRuntimeStatus":
        value = _decode(payload, RUNTIME_STATUS_TYPE)
        for field in ("requested_velocity", "velocity"):
            vector = value.get(field)
            if not isinstance(vector, Mapping):
                raise ValueError("runtime status velocity objects are missing")
            value[field] = tuple(vector[name] for name in ("vx", "vy", "wz"))
        value["mode"] = _mode(str(value.get("mode")), "runtime navigation mode")
        return cls(**value)

    @classmethod
    def from_json(cls, message: str | bytes) -> "NavigationRuntimeStatus":
        return cls.from_dict(json.loads(message))


def build_navigation_message(**values: Any) -> str:
    return NavigationCommand(**_timestamped(values)).to_json()


def decode_navigation_message(
    message: str | bytes | Mapping[str, Any],
) -> NavigationCommand:
    return NavigationCommand.from_dict(
        json.loads(message) if isinstance(message, (str, bytes)) else message
    )


def build_planner_velocity_message(**values: Any) -> str:
    return PlannerVelocityCommand(**_timestamped(values)).to_json()


def decode_planner_velocity_message(
    message: str | bytes | Mapping[str, Any],
) -> PlannerVelocityCommand:
    return PlannerVelocityCommand.from_dict(
        json.loads(message) if isinstance(message, (str, bytes)) else message
    )


def build_navigation_runtime_status_message(**values: Any) -> str:
    return NavigationRuntimeStatus(**_timestamped(values)).to_json()


def decode_navigation_runtime_status_message(
    message: str | bytes | Mapping[str, Any],
) -> NavigationRuntimeStatus:
    return NavigationRuntimeStatus.from_dict(
        json.loads(message) if isinstance(message, (str, bytes)) else message
    )
