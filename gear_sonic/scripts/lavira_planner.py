#!/usr/bin/env python3
"""Publish one-shot LaViRA ObjectNav commands through the REASAN input."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import math
import queue
import select
import signal
import sys
import termios
import threading
import time
import tty
import types
from typing import Any, Callable, Iterator, Mapping, TextIO

import zmq

from gear_sonic.utils.inference.object_nav import (
    ObjectNavConfig,
    ObjectNavResult,
    ObjectNavRunner,
)


_ZERO_TOLERANCE = 1.0e-9
_STOP_VELOCITY = (0.0, 0.0, 0.0)
_MANUAL_KEYS = frozenset("wsadqe")
_PHASE_LABELS = {
    "idle": "IDLE",
    "inferencing": "INFERENCING",
    "rotating": "ROTATING",
    "transition_pause": "STOP_GAP",
    "translating": "FORWARD",
    "final_stop": "FINAL_STOP",
}


class CommandValidationError(ValueError):
    """Raised before motion when an ObjectNav command batch is unsafe."""


class PlannerBusyError(RuntimeError):
    """Raised when a second command batch is started while one is active."""


class _PlannerTermination(BaseException):
    """Immediately unwind main-thread control flow into planner cleanup."""


class _TerminationControl:
    def __init__(self) -> None:
        self.requested = False

    def running(self) -> bool:
        return not self.requested

    def request(self, _signum: int, _frame: Any) -> None:
        if self.requested:
            return
        self.requested = True
        raise _PlannerTermination


def _install_termination_handlers(
    termination: _TerminationControl,
    *,
    register: Callable[[int, Any], Any] = signal.signal,
) -> None:
    register(signal.SIGHUP, termination.request)
    register(signal.SIGINT, termination.request)
    register(signal.SIGTERM, termination.request)


@dataclass
class LaviraPlannerConfig:
    mission: str
    global_target: str
    debug: bool = False
    host: str = "*"
    port: int = 5558
    planner_hz: float = 20.0
    transition_pause: float = 0.5
    final_stop_count: int = 3
    max_speed: float = 0.5
    max_duration: float = 30.0
    max_abs_yaw: float = math.pi
    camera_host: str = "localhost"
    camera_port: int = 5555
    camera_timeout_ms: int = 3000
    codex_timeout_seconds: float = 180.0
    min_confidence: float = 0.6
    rotation_speed: float = 0.4
    forward_speed: float = 0.3
    target_standoff_distance: float = 0.0
    max_direct_travel: float = 8.0
    output_root: str = "outputs/object_nav"


@dataclass(frozen=True)
class VelocityCommand:
    vx: float
    vy: float
    wz: float
    duration: float

    @property
    def velocity(self) -> tuple[float, float, float]:
        return (self.vx, self.vy, self.wz)


@dataclass(frozen=True)
class ObjectNavBatch:
    rotation: VelocityCommand
    translation: VelocityCommand


@dataclass(frozen=True)
class WorkerResult:
    generation: int
    result: ObjectNavResult | None
    error: str | None


def _keyboard_module() -> Any:
    """Load the unchanged keyboard helpers when the CLI-only tyro is absent."""
    module_name = "gear_sonic.scripts.keyboard_planner_thread_server"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != "tyro":
            raise
        placeholder = types.ModuleType("tyro")
        placeholder.cli = None  # type: ignore[attr-defined]
        sys.modules["tyro"] = placeholder
        try:
            return importlib.import_module(module_name)
        finally:
            if sys.modules.get("tyro") is placeholder:
                del sys.modules["tyro"]


def build_reasan_velocity_message(
    command: VelocityCommand, *, action: str
) -> str:
    """Adapt a velocity command through the existing keyboard JSON builder."""
    return _keyboard_module().build_navila_message(
        action,
        command.velocity,
        command.duration,
        raw_text="lavira object nav",
    )


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


def _valid_positive_limit(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value > 0.0
    )


def validate_object_nav_batch(
    payload: Any,
    *,
    max_speed: float = 0.5,
    max_duration: float = 30.0,
    max_abs_yaw: float = math.pi,
    min_positive_duration: float = 0.05,
) -> ObjectNavBatch:
    """Return one validated pure-rotation then pure-translation batch."""
    if not isinstance(payload, Mapping):
        raise CommandValidationError("request must be a JSON object")
    commands = payload.get("commands")
    if not isinstance(commands, list) or len(commands) != 2:
        raise CommandValidationError("commands must contain exactly two entries")
    if not all(
        _valid_positive_limit(value)
        for value in (max_speed, max_duration, max_abs_yaw)
    ):
        raise ValueError("planner safety limits must be finite and positive")
    if (
        isinstance(min_positive_duration, bool)
        or not isinstance(min_positive_duration, (int, float))
        or not math.isfinite(min_positive_duration)
        or min_positive_duration < 0.0
    ):
        raise ValueError("min_positive_duration must be finite and non-negative")

    rotation = _velocity_command(commands[0], 0)
    translation = _velocity_command(commands[1], 1)
    for index, command in enumerate((rotation, translation)):
        if command.duration < 0.0:
            raise CommandValidationError(
                f"commands[{index}].duration must be non-negative"
            )
        if command.duration > max_duration:
            raise CommandValidationError(
                f"commands[{index}].duration exceeds limit"
            )
        if 0.0 < command.duration < min_positive_duration:
            raise CommandValidationError(
                f"commands[{index}].duration is shorter than publish period"
            )

    if math.hypot(rotation.vx, rotation.vy) > _ZERO_TOLERANCE:
        raise CommandValidationError("commands[0] must be pure rotation")
    if abs(translation.wz) > _ZERO_TOLERANCE:
        raise CommandValidationError("commands[1] must be pure translation")
    if math.hypot(translation.vx, translation.vy) > max_speed:
        raise CommandValidationError("commands[1] speed exceeds limit")
    if abs(rotation.wz * rotation.duration) > max_abs_yaw:
        raise CommandValidationError("commands[0] yaw exceeds limit")
    return ObjectNavBatch(rotation=rotation, translation=translation)


class LaviraPlannerController:
    """Advance one validated ObjectNav result without blocking the main thread."""

    def __init__(
        self,
        *,
        transition_pause: float = 0.5,
        final_stop_count: int = 3,
        max_speed: float = 0.5,
        max_duration: float = 30.0,
        max_abs_yaw: float = math.pi,
        min_positive_duration: float = 0.05,
    ):
        if (
            isinstance(transition_pause, bool)
            or not isinstance(transition_pause, (int, float))
            or not math.isfinite(transition_pause)
            or transition_pause < 0.0
        ):
            raise ValueError("transition_pause must be finite and non-negative")
        if (
            isinstance(final_stop_count, bool)
            or not isinstance(final_stop_count, int)
            or final_stop_count < 3
        ):
            raise ValueError("final_stop_count must be an integer of at least three")
        if not _valid_positive_limit(min_positive_duration):
            raise ValueError("min_positive_duration must be finite and positive")
        validate_object_nav_batch(
            {"commands": [
                {"vx": 0.0, "vy": 0.0, "wz": 0.0, "duration": 0.0},
                {"vx": 0.0, "vy": 0.0, "wz": 0.0, "duration": 0.0},
            ]},
            max_speed=max_speed,
            max_duration=max_duration,
            max_abs_yaw=max_abs_yaw,
            min_positive_duration=min_positive_duration,
        )
        self.transition_pause = float(transition_pause)
        self.final_stop_count = final_stop_count
        self.max_speed = float(max_speed)
        self.max_duration = float(max_duration)
        self.max_abs_yaw = float(max_abs_yaw)
        self.min_positive_duration = float(min_positive_duration)
        self.phase = "idle"
        self.failure_reason: str | None = None
        self._batch: ObjectNavBatch | None = None
        self._deadline = 0.0
        self._final_stops_remaining = 0

    @property
    def active(self) -> bool:
        return self.phase != "idle"

    def start(self, result: ObjectNavResult, now: float) -> None:
        if self.active:
            raise PlannerBusyError("a planner request is already active")
        timestamp = self._timestamp(now)
        if getattr(result, "outcome", None) != "NAVIGATE":
            self._enter_final_stop(f"object_nav_{getattr(result, 'outcome', 'invalid')}")
            return
        try:
            batch = validate_object_nav_batch(
                result.commands,
                max_speed=self.max_speed,
                max_duration=self.max_duration,
                max_abs_yaw=self.max_abs_yaw,
                min_positive_duration=self.min_positive_duration,
            )
        except (CommandValidationError, TypeError, ValueError, AttributeError) as exc:
            self._enter_final_stop(f"invalid_commands: {exc}")
            return

        self._batch = batch
        self.failure_reason = None
        if batch.rotation.duration > 0.0:
            self.phase = "rotating"
            self._deadline = timestamp + batch.rotation.duration
        else:
            self.phase = "transition_pause"
            self._deadline = timestamp + self.transition_pause

    def cancel(self, reason: str) -> None:
        self._enter_final_stop(str(reason))

    def step(self, now: float) -> VelocityCommand:
        timestamp = self._timestamp(now)
        while self.phase in {"rotating", "transition_pause", "translating"}:
            if timestamp + 1.0e-12 < self._deadline:
                break
            if self.phase == "rotating":
                self.phase = "transition_pause"
                self._deadline = timestamp + self.transition_pause
                break
            if self.phase == "transition_pause":
                assert self._batch is not None
                if self._batch.translation.duration > 0.0:
                    self.phase = "translating"
                    self._deadline += self._batch.translation.duration
                    continue
                self._enter_final_stop(None)
                break
            self._enter_final_stop(None)
            break

        if self.phase == "rotating":
            assert self._batch is not None
            return self._batch.rotation
        if self.phase == "translating":
            assert self._batch is not None
            return self._batch.translation
        if self.phase == "final_stop":
            command = self._stop_command()
            self._final_stops_remaining -= 1
            if self._final_stops_remaining == 0:
                self.phase = "idle"
                self._batch = None
            return command
        return self._stop_command()

    @staticmethod
    def _timestamp(now: float) -> float:
        if isinstance(now, bool) or not isinstance(now, (int, float)):
            raise ValueError("now must be finite")
        timestamp = float(now)
        if not math.isfinite(timestamp):
            raise ValueError("now must be finite")
        return timestamp

    def _stop_command(self) -> VelocityCommand:
        return VelocityCommand(*_STOP_VELOCITY, self.min_positive_duration)

    def _enter_final_stop(self, reason: str | None) -> None:
        self.phase = "final_stop"
        self.failure_reason = reason
        self._final_stops_remaining = self.final_stop_count


class LaviraPlannerRuntime:
    """Coordinate keyboard, generation-tagged inference, and publication."""

    def __init__(
        self,
        config: LaviraPlannerConfig,
        *,
        publish: Callable[[str], None],
        sleep: Callable[[float], None] = time.sleep,
        request_queue: queue.Queue[int | None] | None = None,
        result_queue: queue.Queue[WorkerResult] | None = None,
        worker_stop_event: threading.Event | None = None,
        logger: Callable[[str], None] = print,
    ):
        self._validate_config(config)
        self.config = config
        self.publish = publish
        self._sleep = sleep
        self.request_queue = request_queue or queue.Queue(maxsize=1)
        self.result_queue = result_queue or queue.Queue(maxsize=1)
        self.worker_stop_event = worker_stop_event or threading.Event()
        self._logger = logger
        self.controller = LaviraPlannerController(
            transition_pause=config.transition_pause,
            final_stop_count=config.final_stop_count,
            max_speed=config.max_speed,
            max_duration=config.max_duration,
            max_abs_yaw=config.max_abs_yaw,
            min_positive_duration=1.0 / config.planner_hz,
        )
        self.generation = 0
        self._pending_generation: int | None = None
        self._next_publish_at: float | None = None
        self._last_logged_phase: str | None = None
        self._log_phase_transition()

    @property
    def phase(self) -> str:
        if self._pending_generation is not None:
            return "inferencing"
        return self.controller.phase

    def handle_key(
        self,
        key: str,
        *,
        now: float,
        running: Callable[[], bool] = lambda: True,
    ) -> str:
        normalized = str(key).lower()
        if normalized == "n":
            if self.phase != "idle" or self.worker_stop_event.is_set():
                self._logger("[LaViRA] BUSY navigation request rejected")
                return "busy"
            candidate = self.generation + 1
            try:
                self.request_queue.put_nowait(candidate)
            except queue.Full:
                self._logger("[LaViRA] BUSY navigation request rejected")
                return "busy"
            self.generation = candidate
            self._pending_generation = candidate
            self._log_phase_transition()
            return "started"
        if normalized == "x":
            self._cancel_and_publish_stops("exit", now)
            return "exit"
        if key == " ":
            self._cancel_and_publish_stops("operator_stop", now)
            return "cancelled"
        if normalized in _MANUAL_KEYS:
            self._cancel_and_publish_stops(f"manual_{normalized}", now)
            if not running():
                return "cancelled"
            manual_config, commands = self._manual_commands()
            action, velocity = commands[normalized]
            command = VelocityCommand(*velocity, manual_config.duration)
            message = build_reasan_velocity_message(command, action=action)
            if not running():
                return "cancelled"
            self.publish(message)
            return "manual"
        return "ignored"

    def accept_worker_result(self, item: WorkerResult, *, now: float) -> bool:
        if (
            self._pending_generation is None
            or item.generation != self._pending_generation
            or item.generation != self.generation
        ):
            return False
        self._pending_generation = None
        if item.error is not None or item.result is None:
            reason = item.error or "inference returned no result"
            self._logger(f"[LaViRA] FAILURE {reason}")
            self.controller.cancel(reason)
        else:
            self.controller.start(item.result, now)
            if self.controller.failure_reason is not None:
                self._logger(f"[LaViRA] FAILURE {self.controller.failure_reason}")
        self._next_publish_at = self.controller._timestamp(now)
        self._log_phase_transition()
        return True

    def poll_worker_results(self, *, now: float) -> int:
        accepted = 0
        while True:
            try:
                item = self.result_queue.get_nowait()
            except queue.Empty:
                return accepted
            accepted += int(self.accept_worker_result(item, now=now))

    def publish_due(
        self,
        now: float,
        *,
        running: Callable[[], bool] = lambda: True,
    ) -> VelocityCommand | None:
        timestamp = self.controller._timestamp(now)
        if not self.controller.active:
            self._next_publish_at = None
            return None
        if self._next_publish_at is None:
            self._next_publish_at = timestamp
        if timestamp + 1.0e-12 < self._next_publish_at:
            return None
        command = self.controller.step(timestamp)
        message = build_reasan_velocity_message(command, action=self._action(command))
        if not running():
            self.controller.cancel("termination_requested")
            self._logger("[LaViRA] CANCEL termination_requested")
            self._log_phase_transition()
            self._next_publish_at = None
            return None
        self.publish(message)
        self._log_phase_transition()
        self._next_publish_at = timestamp + 1.0 / self.config.planner_hz
        return command

    def shutdown(self, reason: str = "shutdown") -> None:
        self.worker_stop_event.set()
        self._replace_requests_with_shutdown()
        self._cancel_and_publish_stops(reason, time.monotonic())

    def _replace_requests_with_shutdown(self) -> None:
        while True:
            while True:
                try:
                    self.request_queue.get_nowait()
                except queue.Empty:
                    break
            try:
                self.request_queue.put_nowait(None)
                return
            except queue.Full:
                continue

    def _cancel_and_publish_stops(self, reason: str, now: float) -> None:
        self.generation += 1
        self._pending_generation = None
        self.controller.cancel(reason)
        self._logger(f"[LaViRA] CANCEL {reason}")
        self._log_phase_transition()
        for _ in range(self.config.final_stop_count):
            command = self.controller.step(now)
            self.publish(build_reasan_velocity_message(command, action="stop"))
            self._log_phase_transition()
            self._sleep(1.0 / self.config.planner_hz)
        self._next_publish_at = None

    def _log_phase_transition(self) -> None:
        phase = self.phase
        if phase == self._last_logged_phase:
            return
        self._last_logged_phase = phase
        self._logger(f"[LaViRA] STATE {_PHASE_LABELS[phase]}")

    def _manual_commands(
        self,
    ) -> tuple[Any, dict[str, tuple[str, tuple[float, float, float]]]]:
        keyboard = _keyboard_module()
        manual_config = keyboard.KeyboardPlannerConfig()
        return manual_config, keyboard.key_commands(manual_config)

    @staticmethod
    def _action(command: VelocityCommand) -> str:
        if all(abs(value) <= _ZERO_TOLERANCE for value in command.velocity):
            return "stop"
        if command.wz > _ZERO_TOLERANCE:
            return "turn_left"
        if command.wz < -_ZERO_TOLERANCE:
            return "turn_right"
        return "move_forward"

    @staticmethod
    def _validate_config(config: LaviraPlannerConfig) -> None:
        if not config.mission.strip() or not config.global_target.strip():
            raise ValueError("mission and global_target are required")
        for name in (
            "planner_hz",
            "max_speed",
            "max_duration",
            "max_abs_yaw",
        ):
            if not _valid_positive_limit(getattr(config, name)):
                raise ValueError(f"{name} must be finite and positive")
        if isinstance(config.final_stop_count, bool) or not isinstance(
            config.final_stop_count, int
        ) or config.final_stop_count < 3:
            raise ValueError("final_stop_count must be an integer of at least three")


def _offer_worker_result(
    results: queue.Queue[WorkerResult], item: WorkerResult
) -> None:
    try:
        results.put_nowait(item)
        return
    except queue.Full:
        pass
    try:
        results.get_nowait()
    except queue.Empty:
        pass
    results.put_nowait(item)


def run_inference_worker(
    runner_factory: Callable[[], ObjectNavRunner],
    requests: queue.Queue[int | None],
    results: queue.Queue[WorkerResult],
    stop_event: threading.Event | None = None,
) -> None:
    """Run one AgentNav request at a time and tag every result by generation."""
    stopping = stop_event or threading.Event()
    runner: ObjectNavRunner | None = None
    try:
        while not stopping.is_set():
            generation = requests.get()
            if generation is None or stopping.is_set():
                return
            try:
                if runner is None:
                    runner = runner_factory()
                if stopping.is_set():
                    return
                nav_result = runner.run_once()
                item = WorkerResult(generation, nav_result, None)
            except Exception as exc:
                item = WorkerResult(generation, None, str(exc))
            if stopping.is_set():
                return
            _offer_worker_result(results, item)
    finally:
        if runner is not None:
            runner.close()


def run_planner_loop(
    runtime: LaviraPlannerRuntime,
    *,
    read_key: Callable[[], str | None],
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    running: Callable[[], bool] = lambda: True,
) -> None:
    """Run the responsive main-thread event loop with guaranteed shutdown stops."""
    try:
        try:
            while running():
                now = monotonic()
                key = read_key()
                decision = (
                    runtime.handle_key(key, now=now, running=running)
                    if key is not None
                    else None
                )
                if not running():
                    break
                if decision == "exit":
                    break
                runtime.poll_worker_results(now=now)
                if not running():
                    break
                runtime.publish_due(now, running=running)
                if not running():
                    break
                sleep(min(0.01, 0.25 / runtime.config.planner_hz))
        except (_PlannerTermination, KeyboardInterrupt):
            pass
    finally:
        runtime.shutdown("loop_exit")


def read_key_nonblocking(stream: TextIO = sys.stdin) -> str | None:
    readable, _, _ = select.select([stream], [], [], 0.0)
    return stream.read(1) if readable else None


@contextmanager
def cbreak_terminal(stream: TextIO = sys.stdin) -> Iterator[None]:
    if not stream.isatty():
        raise RuntimeError("LaViRA keyboard input requires an interactive TTY/tmux pane")
    descriptor = stream.fileno()
    original = termios.tcgetattr(descriptor)
    try:
        tty.setcbreak(descriptor)
        yield
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, original)


def _runner_factory(config: LaviraPlannerConfig) -> ObjectNavRunner:
    return ObjectNavRunner(
        ObjectNavConfig(
            mission=config.mission,
            global_target=config.global_target,
            camera_host=config.camera_host,
            camera_port=config.camera_port,
            camera_timeout_ms=config.camera_timeout_ms,
            codex_timeout_seconds=config.codex_timeout_seconds,
            min_confidence=config.min_confidence,
            rotation_speed=config.rotation_speed,
            forward_speed=config.forward_speed,
            target_standoff_distance=config.target_standoff_distance,
            max_direct_travel=config.max_direct_travel,
            output_root=config.output_root,
        )
    )


def main(config: LaviraPlannerConfig) -> None:
    context = zmq.Context.instance()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.LINGER, 0)
    endpoint = f"tcp://{config.host}:{config.port}"
    socket.bind(endpoint)
    runtime = LaviraPlannerRuntime(config, publish=socket.send_string)
    worker = threading.Thread(
        target=run_inference_worker,
        args=(
            lambda: _runner_factory(config),
            runtime.request_queue,
            runtime.result_queue,
            runtime.worker_stop_event,
        ),
        name="lavira-object-nav",
        daemon=True,
    )
    worker.start()
    termination = _TerminationControl()
    try:
        try:
            _install_termination_handlers(termination)
            print(f"[LaViRA] PUB bound to {endpoint}; mission={config.mission!r}")
            print("[LaViRA] N navigate | W/S/A/D/Q/E manual | Space stop | X exit")
            with cbreak_terminal():
                run_planner_loop(
                    runtime,
                    read_key=read_key_nonblocking,
                    running=termination.running,
                )
        except (_PlannerTermination, KeyboardInterrupt):
            pass
    finally:
        runtime.shutdown("main_exit")
        socket.close()
        print("[LaViRA] Stopped")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(LaviraPlannerConfig))
