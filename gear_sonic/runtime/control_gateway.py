"""Pure control-ingress logic shared by the console and future GUI.

This first migration stage is intentionally a protocol adapter.  It assigns
identity and lifetime to operator input while preserving the exact legacy
string that existing VLA and data-exporter subscribers consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from gear_sonic.runtime.contracts import MessageMetadata, OperatorCommand


LEGACY_COMMAND_NAMES = {
    "c": "start_recording",
    "e": "stop_recording_success",
    "f": "stop_recording_failure",
    "i": "select_pose_mode",
    "k": "toggle_control_loop",
    "o": "select_planner_mode",
    "p": "toggle_policy_pause",
    "[": "toggle_left_hand_initial_pose",
    "]": "toggle_right_hand_initial_pose",
}
PROMPT_PREFIX = "prompt:"
NAVIGATION_KEYS = frozenset({"w", "a", "s", "d", "q", "e", "n", " "})
MANUAL_NAVIGATION_VELOCITIES = {
    "w": (0.3, 0.0, 0.0),
    "s": (-0.3, 0.0, 0.0),
    "a": (0.0, 0.15, 0.0),
    "d": (0.0, -0.15, 0.0),
    "q": (0.0, 0.0, 0.5),
    "e": (0.0, 0.0, -0.5),
}
RECORDING_ALIASES = {
    "record-start": "c",
    "record-success": "e",
    "record-failure": "f",
}


def legacy_message_from_console_line(line: str) -> str:
    """Apply the launcher's existing ``t <prompt>`` console convention."""

    if line.startswith("t "):
        return PROMPT_PREFIX + line[2:]
    return line


def operator_command_from_legacy(
    message: str,
    *,
    metadata: MessageMetadata,
    command_id: str,
) -> OperatorCommand:
    """Describe one legacy message without changing or executing it."""

    if message.startswith(PROMPT_PREFIX):
        name = "set_prompt"
        parameters = {
            "prompt": message[len(PROMPT_PREFIX) :],
            "legacy_message": message,
        }
    else:
        name = LEGACY_COMMAND_NAMES.get(message, "legacy_passthrough")
        parameters = {"legacy_message": message}
    return OperatorCommand(
        metadata=metadata,
        command_id=command_id,
        name=name,
        parameters=parameters,
    )


@dataclass(frozen=True)
class ControlIngressEvent:
    """One console event represented in both current and future protocols."""

    legacy_message: str
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

    @property
    def next_sequence(self) -> int:
        return self._sequence

    def accept_console_line(self, line: str) -> ControlIngressEvent:
        return self.accept_legacy_message(legacy_message_from_console_line(line))

    def accept_legacy_message(self, legacy_message: str) -> ControlIngressEvent:
        """Create a typed event for a message already in the deployed format."""
        metadata, command_id = self._next_identity()
        command = operator_command_from_legacy(
            legacy_message,
            metadata=metadata,
            command_id=command_id,
        )
        return ControlIngressEvent(
            legacy_message=legacy_message,
            command=command,
        )

    def accept_command(
        self,
        name: str,
        *,
        parameters: dict,
        legacy_message: str = "",
        mirror_legacy: bool = False,
    ) -> ControlIngressEvent:
        """Create an explicit typed command for non-ambiguous GUI controls."""

        metadata, command_id = self._next_identity()
        values = dict(parameters)
        values["legacy_message"] = legacy_message
        values["mirror_legacy"] = bool(mirror_legacy)
        return ControlIngressEvent(
            legacy_message=legacy_message,
            command=OperatorCommand(
                metadata=metadata,
                command_id=command_id,
                name=name,
                parameters=values,
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
        if value == "space":
            value = " "
        if value in RECORDING_ALIASES:
            value = RECORDING_ALIASES[value]
            return core.accept_legacy_message(value)
        if self.control_mode == "PLANNER" and value in NAVIGATION_KEYS:
            return core.accept_command(
                "navigation_key",
                parameters={"key": value},
                legacy_message=value,
                mirror_legacy=False,
            )
        event = core.accept_console_line(line)
        if event.command.name == "toggle_control_loop":
            if not self.control_running:
                self.control_mode = "PLANNER"
            self.control_running = not self.control_running
        elif event.command.name == "select_pose_mode" and self.control_running:
            self.control_mode = "POSE"
        elif event.command.name == "select_planner_mode" and self.control_running:
            self.control_mode = "PLANNER"
        return event


@dataclass(frozen=True)
class NavigationControlAction:
    generation: int
    mode: str
    velocity: tuple[float, float, float] | None = None
    agent_event: str | None = None


class NavigationControlState:
    """Own manual-key timing and the generation shared by Agent and NavDP."""

    def __init__(self, *, manual_hold_s: float = 0.55) -> None:
        if manual_hold_s <= 0.0:
            raise ValueError("manual_hold_s must be positive")
        self.manual_hold_s = float(manual_hold_s)
        self.generation = 0
        self.mode = "listen_wasd"
        self.manual_velocity = (0.0, 0.0, 0.0)
        self.manual_deadline = 0.0

    def handle_key(self, key: str, *, now: float) -> NavigationControlAction:
        normalized = key.lower()
        if normalized not in NAVIGATION_KEYS:
            raise ValueError(f"unsupported navigation key: {key!r}")
        if normalized == "n":
            if self.mode != "listen_wasd":
                return NavigationControlAction(self.generation, "ignored")
            self.generation += 1
            self.manual_velocity = (0.0, 0.0, 0.0)
            self.manual_deadline = 0.0
            self.mode = "nav_pending"
            return NavigationControlAction(
                self.generation,
                "stop",
                agent_event="start_navigation",
            )
        if normalized == " ":
            self.generation += 1
            self.manual_velocity = (0.0, 0.0, 0.0)
            self.manual_deadline = 0.0
            self.mode = "listen_wasd"
            return NavigationControlAction(
                self.generation,
                "stop",
                agent_event="cancel_navigation",
            )
        if self.mode != "listen_wasd":
            return NavigationControlAction(self.generation, "ignored")
        self.manual_velocity = MANUAL_NAVIGATION_VELOCITIES[normalized]
        self.manual_deadline = float(now) + self.manual_hold_s
        return NavigationControlAction(
            self.generation,
            "manual_velocity",
            velocity=self.manual_velocity,
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
            )
        return None

    def accept_goal(self, parameters: Mapping[str, object]) -> NavigationControlAction:
        generation = int(parameters["generation"])
        if generation != self.generation or self.mode != "nav_pending":
            raise ValueError("stale or unexpected navigation goal")
        self.mode = "nav"
        return NavigationControlAction(generation, "nav_goal")

    def accept_status(self, payload: Mapping[str, object]) -> bool:
        if int(payload.get("generation", -1)) != self.generation:
            return False
        if payload.get("state") in {"reached", "failed", "stopped"}:
            self.mode = "listen_wasd"
        return True


@dataclass(frozen=True)
class RoutedControlCommand:
    command: OperatorCommand
    legacy_message: str | None
    accepted: bool
    reason: str
    mirror_legacy: bool


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
                command, None, False, "expired operator intent", False
            )
        previous = self._last_sequence_by_source.get(command.metadata.source)
        if previous is not None and command.metadata.sequence <= previous:
            return RoutedControlCommand(
                command, None, False, "duplicate or out-of-order intent", False
            )
        try:
            legacy_message = str(command.parameters["legacy_message"])
        except KeyError:
            return RoutedControlCommand(command, None, False, "missing legacy_message", False)
        self._last_sequence_by_source[command.metadata.source] = command.metadata.sequence
        return RoutedControlCommand(
            command,
            legacy_message,
            True,
            "forwarded",
            bool(command.parameters.get("mirror_legacy", True)),
        )
