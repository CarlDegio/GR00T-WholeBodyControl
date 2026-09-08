"""Pure typed control-ingress logic shared by the console and GUI."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping

from gear_sonic.runtime.protocol import MessageMetadata, OperatorCommand

CONSOLE_COMMAND_NAMES = {
    "c": "start_recording",
    "e": "stop_recording_success",
    "f": "stop_recording_failure",
    "g": "complete_agent_success",
    "i": "select_pose_mode",
    "k": "toggle_control_loop",
    "o": "select_planner_mode",
    "p": "toggle_policy_pause",
    "[": "toggle_left_hand_initial_pose",
    "]": "toggle_right_hand_initial_pose",
}
PROMPT_PREFIX = "prompt:"
NAVIGATION_KEYS = frozenset({"w", "a", "s", "d", "q", "e", "n", "b", " "})
MANUAL_NAVIGATION_VELOCITIES = {
    "w": (0.3, 0.0, 0.0),
    "s": (-0.3, 0.0, 0.0),
    "a": (0.0, 0.15, 0.0),
    "d": (0.0, -0.15, 0.0),
    "q": (0.0, 0.0, 0.5),
    "e": (0.0, 0.0, -0.5),
}
BASE_POSE_RUNTIME_STATUS_COMMAND = "base_pose_runtime_status"
RECORDING_ALIASES = {
    "record-start": "c",
    "record-success": "e",
    "record-failure": "f",
}
def operator_command_from_console_input(
    message: str,
    *,
    metadata: MessageMetadata,
    command_id: str,
) -> OperatorCommand:
    """Translate one normalized console input into a typed command."""

    if message.startswith(PROMPT_PREFIX):
        name = "set_prompt"
        parameters = {"prompt": message[len(PROMPT_PREFIX) :]}
    else:
        name = CONSOLE_COMMAND_NAMES.get(message, "unsupported_console_input")
        parameters = {}
    return OperatorCommand(
        metadata=metadata,
        command_id=command_id,
        name=name,
        parameters=parameters,
    )


@dataclass(frozen=True)
class ControlIngressEvent:
    """One typed console event."""

    command: OperatorCommand


class ControlGatewayCore:
    """Sequence and normalize operator input without owning robot state."""

    def __init__(
        self,
        *,
        source: str = "operator_console",
        ttl_ms: int = 1000,
        monotonic_ns: Callable[[], int] | None = None,
    ) -> None:
        if not source:
            raise ValueError("source cannot be empty")
        if ttl_ms < 0:
            raise ValueError("ttl_ms cannot be negative")
        if monotonic_ns is None:
            import time

            monotonic_ns = time.monotonic_ns
        self.source = source
        self.ttl_ms = int(ttl_ms)
        self._monotonic_ns = monotonic_ns
        self._sequence = 0

    def accept_console_line(self, line: str) -> ControlIngressEvent:
        if line.startswith("t "):
            line = PROMPT_PREFIX + line[2:]
        return self.accept_console_message(line)

    def accept_console_message(self, message: str) -> ControlIngressEvent:
        """Create a typed event for a normalized console message."""
        metadata, command_id = self._next_identity()
        command = operator_command_from_console_input(
            message,
            metadata=metadata,
            command_id=command_id,
        )
        return ControlIngressEvent(command=command)

    def accept_command(
        self,
        name: str,
        *,
        parameters: dict,
    ) -> ControlIngressEvent:
        """Create an explicit typed command for non-ambiguous GUI controls."""

        metadata, command_id = self._next_identity()
        return ControlIngressEvent(
            command=OperatorCommand(
                metadata=metadata,
                command_id=command_id,
                name=name,
                parameters=dict(parameters),
            ),
        )

    def _next_identity(self) -> tuple[MessageMetadata, str]:
        sequence = self._sequence
        self._sequence += 1
        metadata = MessageMetadata(
            source=self.source,
            sequence=sequence,
            timestamp_ns=int(self._monotonic_ns()),
            ttl_ms=self.ttl_ms,
        )
        return metadata, f"{self.source}-{sequence}"


class OperatorConsoleRouter:
    """Interpret one CLI line using the same POSE/PLANNER key context."""

    def __init__(self) -> None:
        self.control_mode = "PLANNER"
        self.control_running = False

    def accept_line(
        self,
        line: str,
        *,
        core: ControlGatewayCore,
    ) -> ControlIngressEvent:
        value = line.lower()
        if value == "g":
            line = value
        if value == "space":
            value = " "
        if value in RECORDING_ALIASES:
            value = RECORDING_ALIASES[value]
            return core.accept_console_message(value)
        if self.control_mode == "PLANNER" and value in NAVIGATION_KEYS:
            return core.accept_command(
                "navigation_key",
                parameters={"key": value},
            )
        event = core.accept_console_line(line)
        if event.command.name == "toggle_control_loop":
            if not self.control_running:
                self.control_mode = "PLANNER"
            self.control_running = not self.control_running
        elif event.command.name == "select_pose_mode" and self.control_running:
            self.control_mode = "POSE"
        elif event.command.name in {"select_planner_mode", "complete_agent_success"} and self.control_running:
            self.control_mode = "PLANNER"
        return event


@dataclass(frozen=True)
class NavigationControlAction:
    generation: int
    mode: str
    velocity: tuple[float, float, float] | None = None
    agent_event: str | None = None
    reason: str = ""
    segment_id: int = 0
    skill_id: int = 0


class NavigationControlState:
    """Own manual-key timing and the generation shared by Agent and NavDP."""

    def __init__(
        self,
        *,
        manual_hold_s: float = 0.55,
        base_pose_command_timeout_s: float = 0.35,
    ) -> None:
        if manual_hold_s <= 0.0 or base_pose_command_timeout_s <= 0.0:
            raise ValueError("navigation command timeouts must be positive")
        self.manual_hold_s = float(manual_hold_s)
        self.base_pose_command_timeout_s = float(base_pose_command_timeout_s)
        self.generation = 0
        self.skill_id = 0
        self.segment_id = 0
        self.mode = "listen_wasd"
        self.manual_velocity = (0.0, 0.0, 0.0)
        self.manual_deadline = 0.0
        self.lavira_task_active = False
        self.task_started_at: float | None = None
        self.window_id = 0

    @property
    def owner(self) -> str | None:
        if self.mode.startswith("lavira_"):
            return "lavira"
        if self.mode.startswith("base_pose_"):
            return "base_pose"
        return None

    def _busy(self) -> NavigationControlAction:
        return NavigationControlAction(
            self.generation,
            "ignored",
            reason=f"navigation_busy:{self.owner or self.mode}",
            skill_id=self.skill_id,
            segment_id=max(0, self.segment_id),
        )

    def handle_key(
        self,
        key: str,
        *,
        now: float,
        cancel_reason: str = "",
    ) -> NavigationControlAction:
        normalized = key.lower()
        if normalized not in NAVIGATION_KEYS:
            raise ValueError(f"unsupported navigation key: {key!r}")
        if normalized in {"n", "b"}:
            if self.mode != "listen_wasd":
                return self._busy()
            self.generation += 1
            self.skill_id = 0
            self.segment_id = -1 if normalized == "n" else 0
            self.window_id = 0
            self.manual_velocity = (0.0, 0.0, 0.0)
            self.manual_deadline = 0.0
            is_lavira = normalized == "n"
            self.lavira_task_active = is_lavira
            self.task_started_at = float(now) if is_lavira else None
            self.mode = "lavira_pending" if is_lavira else "base_pose_inference"
            return NavigationControlAction(
                self.generation,
                "stop",
                agent_event="start_navigation" if is_lavira else "start_base_pose",
                skill_id=self.skill_id,
            )
        if normalized == " ":
            self.generation += 1
            self.skill_id = 0
            self.segment_id = 0
            self.window_id = 0
            self.manual_velocity = (0.0, 0.0, 0.0)
            self.manual_deadline = 0.0
            self.mode = "listen_wasd"
            self.lavira_task_active = False
            self.task_started_at = None
            return NavigationControlAction(
                self.generation,
                "stop",
                agent_event="cancel_navigation",
                reason=str(cancel_reason),
                skill_id=self.skill_id,
            )
        if self.mode != "listen_wasd":
            return self._busy()
        self.manual_velocity = MANUAL_NAVIGATION_VELOCITIES[normalized]
        self.manual_deadline = float(now) + self.manual_hold_s
        return NavigationControlAction(
            self.generation,
            "manual_velocity",
            velocity=self.manual_velocity,
            skill_id=self.skill_id,
        )

    def tick(self, *, now: float) -> NavigationControlAction | None:
        if (
            self.mode == "listen_wasd"
            and self.manual_deadline > 0.0
            and float(now) >= self.manual_deadline
        ):
            self.manual_velocity = (0.0, 0.0, 0.0)
            self.manual_deadline = 0.0
            return NavigationControlAction(
                self.generation,
                "manual_velocity",
                velocity=self.manual_velocity,
                skill_id=self.skill_id,
            )
        if (
            self.mode == "base_pose_motion"
            and self.manual_deadline > 0.0
            and float(now) >= self.manual_deadline
        ):
            self.generation += 1
            self.skill_id = 0
            self.manual_velocity = (0.0, 0.0, 0.0)
            self.manual_deadline = 0.0
            self.mode = "listen_wasd"
            self.lavira_task_active = False
            self.window_id = 0
            self.task_started_at = None
            return NavigationControlAction(
                self.generation,
                "stop",
                agent_event="cancel_navigation",
                reason="base_pose_velocity_timeout",
                skill_id=self.skill_id,
            )
        return None

    def accept_goal(self, parameters: Mapping[str, object]) -> NavigationControlAction:
        generation = int(parameters["generation"])
        skill_id = int(parameters.get("skill_id", 0))
        segment_id = int(parameters.get("segment_id", 0))
        if (
            generation != self.generation
            or self.mode != "lavira_pending"
            or (skill_id, segment_id) <= (self.skill_id, self.segment_id)
        ):
            raise ValueError("stale or unexpected navigation goal")
        self.skill_id = skill_id
        self.segment_id = segment_id
        self.mode = "lavira_nav"
        return NavigationControlAction(
            generation,
            "nav_goal",
            segment_id=segment_id,
            skill_id=skill_id,
        )

    def accept_heading_goal(
        self, parameters: Mapping[str, object]
    ) -> NavigationControlAction:
        action = self.accept_goal(parameters)
        return NavigationControlAction(
            action.generation,
            "heading_goal",
            segment_id=action.segment_id,
            skill_id=action.skill_id,
        )

    def accept_lavira_depth_request(self, parameters: Mapping[str, object]) -> bool:
        generation = int(parameters["generation"])
        skill_id = int(parameters.get("skill_id", 0))
        segment_id = int(parameters.get("segment_id", self.segment_id))
        return (
            generation == self.generation
            and self.mode == "lavira_pending"
            and skill_id == self.skill_id
            and segment_id == self.segment_id
        )

    def accept_lavira_rgbd_captured(self, parameters: Mapping[str, object]) -> bool:
        """Validate LaViRA's DA lease release without changing navigation state."""

        generation = int(parameters["generation"])
        skill_id = int(parameters.get("skill_id", 0))
        segment_id = int(parameters.get("segment_id", self.segment_id))
        return (
            generation == self.generation
            and self.mode == "lavira_pending"
            and skill_id == self.skill_id
            and segment_id == self.segment_id
        )

    def accept_base_pose_start(
        self, parameters: Mapping[str, object]
    ) -> NavigationControlAction:
        generation = int(parameters["generation"])
        skill_id = int(parameters.get("skill_id", 0))
        segment_id = int(parameters.get("segment_id", 0))
        if (
            generation != self.generation
            or self.mode != "lavira_pending"
            or (skill_id, segment_id) <= (self.skill_id, self.segment_id)
        ):
            raise ValueError("stale or unexpected BasePose start")
        self.skill_id = skill_id
        self.segment_id = segment_id
        self.mode = "base_pose_inference"
        return NavigationControlAction(
            generation,
            "stop",
            segment_id=segment_id,
            skill_id=skill_id,
            agent_event="start_base_pose",
        )

    def accept_vla_command(
        self, name: str, parameters: Mapping[str, object]
    ) -> bool:
        generation = int(parameters.get("generation", -1))
        skill_id = int(parameters.get("skill_id", 0))
        if generation != self.generation or not self.lavira_task_active:
            return False
        if name == "start_vla_task":
            if self.mode == "lavira_manipulate" and skill_id == self.skill_id:
                return True
            if self.mode != "lavira_pending" or skill_id < self.skill_id:
                return False
            if skill_id > self.skill_id:
                self.segment_id = 0
            self.skill_id = skill_id
            self.window_id = int(parameters.get("window_id", 0))
            self.mode = "lavira_manipulate"
            return True
        if self.mode != "lavira_manipulate" or skill_id != self.skill_id:
            return False
        window_id = int(parameters.get("window_id", 0))
        if window_id < self.window_id:
            return False
        self.window_id = window_id
        return name in {"hold_vla_task", "resume_vla_task", "stop_vla_task"}

    def accept_base_pose_velocity(
        self,
        parameters: Mapping[str, object],
        *,
        now: float,
    ) -> NavigationControlAction:
        generation = int(parameters["generation"])
        skill_id = int(parameters.get("skill_id", 0))
        segment_id = int(parameters.get("segment_id", self.segment_id))
        if generation != self.generation or self.mode not in {
            "base_pose_inference",
            "base_pose_motion",
            "base_pose_stopping",
        }:
            raise ValueError("stale or unexpected base-pose velocity")
        if skill_id != self.skill_id or segment_id != self.segment_id:
            raise ValueError("stale base-pose skill or segment")
        raw_velocity = parameters.get("velocity")
        if not isinstance(raw_velocity, (list, tuple)) or len(raw_velocity) != 3:
            raise ValueError("base_pose_velocity requires [vx, vy, wz]")
        velocity = tuple(float(value) for value in raw_velocity)
        if not all(math.isfinite(value) for value in velocity):
            raise ValueError("base-pose velocity must be finite")
        vx, vy, wz = velocity
        motion_profile = str(parameters.get("motion_profile", "sequence"))
        if motion_profile == "sequence":
            unsafe = abs(vx) > 0.300001 or abs(vy) > 0.000001 or abs(wz) > 0.400001
        elif motion_profile == "yoloe_servo":
            unsafe = (
                abs(vx) > 0.400001
                or abs(vy) > 0.400001
                or abs(wz) > 0.300001
                or (abs(vx) > 0.000001 and abs(vy) > 0.000001)
            )
        else:
            raise ValueError("unsupported base-pose motion profile")
        if unsafe:
            raise ValueError("base-pose velocity exceeds the planner safety envelope")
        action = str(parameters.get("action", "visual_servo"))
        if action not in {"hold", "visual_servo", "stop"}:
            raise ValueError("unsupported base-pose action")
        if action in {"hold", "stop"} and any(
            abs(value) > 0.000001 for value in velocity
        ):
            raise ValueError(f"base-pose {action} action must command zero velocity")
        self.manual_velocity = velocity
        if action == "visual_servo":
            self.mode = "base_pose_motion"
            self.manual_deadline = float(now) + self.base_pose_command_timeout_s
            output_mode = "manual_velocity"
        else:
            # Initial/recovery inference may legitimately spend seconds in a
            # remote model call.  A zero hold is already fail-safe, so only an
            # actual visual-servo motion lease uses the 350 ms watchdog.
            self.mode = (
                "base_pose_inference" if action == "hold" else "base_pose_stopping"
            )
            self.manual_deadline = 0.0
            output_mode = "stop"
        return NavigationControlAction(
            generation,
            output_mode,
            velocity=velocity,
            segment_id=segment_id,
            skill_id=skill_id,
        )

    def accept_status(
        self,
        payload: Mapping[str, object],
        *,
        owner: str | None = None,
        agent_final: bool = True,
    ) -> bool:
        if int(payload.get("generation", -1)) != self.generation:
            return False
        if int(payload.get("skill_id", 0)) != self.skill_id:
            return False
        if owner is not None and self.owner != owner:
            return False
        if owner == "lavira" and not agent_final:
            if int(payload.get("segment_id", 0)) != self.segment_id:
                return False
            if self.mode != "lavira_nav":
                return False
        if payload.get("state") in {"reached", "failed", "stopped"}:
            if owner == "lavira" and agent_final:
                self.mode = "listen_wasd"
                self.lavira_task_active = False
                self.task_started_at = None
            elif self.lavira_task_active:
                self.mode = "lavira_pending"
            else:
                self.mode = "listen_wasd"
            self.manual_velocity = (0.0, 0.0, 0.0)
            self.manual_deadline = 0.0
        return True


@dataclass(frozen=True)
class RoutedControlCommand:
    command: OperatorCommand
    accepted: bool
    reason: str


class ControlGatewayRouter:
    """Validate ordering/TTL before an intent reaches deployed consumers."""

    def __init__(self, *, monotonic_ns: Callable[[], int] | None = None) -> None:
        if monotonic_ns is None:
            import time

            monotonic_ns = time.monotonic_ns
        self._monotonic_ns = monotonic_ns
        self._last_sequence_by_source: dict[str, int] = {}

    def route(self, command: OperatorCommand) -> RoutedControlCommand:
        now_ns = int(self._monotonic_ns())
        if command.metadata.is_expired(now_ns):
            return RoutedControlCommand(
                command, False, "expired operator intent"
            )
        previous = self._last_sequence_by_source.get(command.metadata.source)
        if previous is not None and command.metadata.sequence <= previous:
            return RoutedControlCommand(
                command, False, "duplicate or out-of-order intent"
            )
        self._last_sequence_by_source[command.metadata.source] = command.metadata.sequence
        return RoutedControlCommand(
            command,
            True,
            "forwarded",
        )
