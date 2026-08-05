#!/usr/bin/env python3
"""Run model-authored head-camera base-pose adjustment through SONIC."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import json
import math
import queue
import select
import signal
import sys
import termios
import threading
import time
import tty
from typing import Any, Callable, Iterator, TextIO

import zmq

from gear_sonic.utils.inference.base_pose import (
    BasePoseConfig,
    BasePoseResult,
    BasePoseRunner,
    validate_base_pose_plan,
)


MESSAGE_TYPE = "navila_reasan_velocity_command"
STOP_VELOCITY = (0.0, 0.0, 0.0)


@dataclass
class BasePosePlannerConfig:
    task: str
    mode: str = "rgb"
    model: str = "gpt-5.6-sol"
    reasoning_effort: str = "max"
    host: str = "*"
    port: int = 5558
    planner_hz: float = 20.0
    transition_pause: float = 0.5
    final_stop_count: int = 3
    rotation_speed: float = 0.4
    translation_speed: float = 0.3
    camera_host: str = "localhost"
    camera_port: int = 5555
    camera_timeout_ms: int = 15000
    camera_stream: str = "ego_view"
    camera_height_m: float = 1.2
    camera_pitch_deg: float = -47.6
    vertical_fov_deg: float = 55.2
    camera_forward_offset_m: float = 0.0
    camera_lateral_offset_m: float = 0.0
    depth_visual_max_m: float = 3.0
    codex_timeout_seconds: float = 600.0
    output_root: str = "outputs/base_pose_adjustment"


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
class MotionSegment:
    action: str
    command: VelocityCommand


@dataclass(frozen=True)
class WorkerResult:
    generation: int
    result: BasePoseResult | None
    error: str | None


def build_velocity_message(command: VelocityCommand, *, action: str) -> str:
    vx, vy, wz = command.velocity
    duration = command.duration
    payload = {
        "action": action,
        "angle_rad": abs(wz) * duration,
        "distance_m": math.hypot(vx, vy) * duration,
        "duration_s": duration,
        "raw_text": "gpt-5.6 base pose",
        "segments": [{"duration_s": duration, "vx": vx, "vy": vy, "wz": wz}],
        "source": "base_pose",
        "status": "ok",
        "type": MESSAGE_TYPE,
        "velocity": {"vx": vx, "vy": vy, "wz": wz},
        "version": 1,
    }
    return json.dumps(payload, separators=(",", ":"))


def plan_to_segments(
    plan: dict[str, Any],
    *,
    rotation_speed: float = 0.4,
    translation_speed: float = 0.3,
) -> tuple[MotionSegment, ...]:
    """Map validated model values to fixed-speed commands without changing them."""
    if not math.isfinite(rotation_speed) or rotation_speed <= 0.0:
        raise ValueError("rotation_speed must be finite and positive")
    if not math.isfinite(translation_speed) or translation_speed <= 0.0:
        raise ValueError("translation_speed must be finite and positive")
    validated = validate_base_pose_plan(plan)
    if validated["status"] != "ADJUST":
        return ()
    segments: list[MotionSegment] = []
    for item in validated["command_sequence"]:
        value = float(item["value"])
        action = str(item["action"])
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
        else:  # pragma: no cover - validate_base_pose_plan rejects this first
            raise ValueError(f"unsupported base-pose action: {action}")
        segments.append(MotionSegment(action=action, command=command))
    return tuple(segments)


class BasePoseSequenceController:
    """Advance one model-authored sequence with an IDLE pause between segments."""

    def __init__(
        self,
        *,
        rotation_speed: float = 0.4,
        translation_speed: float = 0.3,
        transition_pause: float = 0.5,
        stop_duration: float = 0.05,
    ):
        values = (rotation_speed, translation_speed, stop_duration)
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("controller speeds and stop_duration must be positive")
        if not math.isfinite(transition_pause) or transition_pause < 0.0:
            raise ValueError("transition_pause must be finite and non-negative")
        self.rotation_speed = rotation_speed
        self.translation_speed = translation_speed
        self.transition_pause = transition_pause
        self.stop_duration = stop_duration
        self.phase = "idle"
        self._segments: tuple[MotionSegment, ...] = ()
        self._segment_index = 0
        self._deadline = 0.0

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
        self._segments = ()
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
            self._deadline = (
                timestamp + self._segments[self._segment_index].command.duration
            )
        if self.phase == "motion":
            segment = self._segments[self._segment_index]
            return segment.action, segment.command
        return "hold", self.stop_command()


class BasePosePlannerRuntime:
    """Coordinate N inference, Space cancel-and-hold, and 20 Hz publication."""

    def __init__(
        self,
        config: BasePosePlannerConfig,
        *,
        publish: Callable[[str], None],
        request_queue: queue.Queue[int | None] | None = None,
        result_queue: queue.Queue[WorkerResult] | None = None,
        stop_event: threading.Event | None = None,
        logger: Callable[[str], None] = print,
    ):
        if not config.task.strip():
            raise ValueError("task must be non-empty")
        if not math.isfinite(config.planner_hz) or config.planner_hz <= 0.0:
            raise ValueError("planner_hz must be finite and positive")
        if config.final_stop_count < 3:
            raise ValueError("final_stop_count must be at least three")
        self.config = config
        self.publish = publish
        self.request_queue = request_queue or queue.Queue(maxsize=1)
        self.result_queue = result_queue or queue.Queue(maxsize=1)
        self.stop_event = stop_event or threading.Event()
        self.logger = logger
        self.controller = BasePoseSequenceController(
            rotation_speed=config.rotation_speed,
            translation_speed=config.translation_speed,
            transition_pause=config.transition_pause,
            stop_duration=1.0 / config.planner_hz,
        )
        self.generation = 0
        self.pending_generation: int | None = None
        self.next_publish_at: float | None = None
        self.current_output_dir: str | None = None
        self._shutdown = False

    @property
    def phase(self) -> str:
        if self.pending_generation is not None:
            return "inference"
        if self.controller.active:
            return "motion"
        return "idle"

    def _publish_stop(self) -> None:
        self.publish(
            build_velocity_message(self.controller.stop_command(), action="stop")
        )

    def _publish_stop_sequence(self) -> None:
        for _ in range(self.config.final_stop_count):
            self._publish_stop()

    def _discard_queued_requests(self) -> None:
        while True:
            try:
                self.request_queue.get_nowait()
            except queue.Empty:
                return

    def _record_event(self, event: str, **fields: Any) -> None:
        directory = (
            Path(self.current_output_dir)
            if self.current_output_dir is not None
            else Path(self.config.output_root).resolve()
        )
        try:
            directory.mkdir(parents=True, exist_ok=True)
            record = {
                "timestamp_unix_s": time.time(),
                "event": event,
                "generation": self.generation,
                **fields,
            }
            with (directory / "runtime_events.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            self.logger(f"[BasePose] diagnostic write failed: {exc}")

    def cancel_and_stop(self, reason: str, now: float) -> None:
        self.controller._timestamp(now)
        previous_phase = self.phase
        self._record_event("cancel_and_stop", reason=reason, phase=previous_phase)
        self.generation += 1
        self.pending_generation = None
        self._discard_queued_requests()
        self.controller.cancel()
        self._publish_stop_sequence()
        self.next_publish_at = float(now) + 1.0 / self.config.planner_hz
        self.logger(f"[BasePose] STOP {reason}")

    def handle_key(self, key: str, *, now: float) -> str:
        normalized = str(key).lower()
        if normalized == "x":
            self.cancel_and_stop("exit", now)
            return "exit"
        if key == " ":
            self.cancel_and_stop("operator_space", now)
            return "cancelled"
        if normalized == "n":
            if self.phase != "idle" or self.stop_event.is_set():
                self.logger("[BasePose] BUSY request rejected")
                return "busy"
            candidate = self.generation + 1
            try:
                self.request_queue.put_nowait(candidate)
            except queue.Full:
                self.logger("[BasePose] BUSY request rejected")
                return "busy"
            self.current_output_dir = None
            self.generation = candidate
            self.pending_generation = candidate
            self._publish_stop()
            self.next_publish_at = float(now) + 1.0 / self.config.planner_hz
            self.logger(f"[BasePose] INFERENCE generation={candidate}")
            return "started"
        return "ignored"

    def accept_worker_result(self, item: WorkerResult, *, now: float) -> bool:
        if (
            self.pending_generation is None
            or item.generation != self.pending_generation
            or item.generation != self.generation
        ):
            self._record_event(
                "late_result_discarded",
                result_generation=item.generation,
                result_output_dir=(
                    None if item.result is None else item.result.output_dir
                ),
                error=item.error,
            )
            return False
        self.pending_generation = None
        if item.error is not None or item.result is None:
            self.logger(f"[BasePose] FAILURE {item.error or 'no result'}")
            self.controller.cancel()
            self._publish_stop_sequence()
        else:
            self.current_output_dir = item.result.output_dir
            segments = plan_to_segments(
                item.result.plan,
                rotation_speed=self.config.rotation_speed,
                translation_speed=self.config.translation_speed,
            )
            execution = {
                "rotation_speed_rad_s": self.config.rotation_speed,
                "translation_speed_m_s": self.config.translation_speed,
                "transition_pause_s": self.config.transition_pause,
                "segments": [
                    {
                        "step": index,
                        "action": segment.action,
                        "vx": segment.command.vx,
                        "vy": segment.command.vy,
                        "wz": segment.command.wz,
                        "duration_s": segment.command.duration,
                    }
                    for index, segment in enumerate(segments, start=1)
                ],
                "total_motion_s": sum(segment.command.duration for segment in segments),
                "total_pause_s": self.config.transition_pause
                * max(0, len(segments) - 1),
            }
            try:
                Path(item.result.output_dir, "velocity_segments.json").write_text(
                    json.dumps(execution, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            except OSError as exc:
                self.logger(f"[BasePose] diagnostic write failed: {exc}")
            started = self.controller.start(item.result, now)
            status = item.result.plan["status"]
            self.logger(
                f"[BasePose] RESULT status={status} output={item.result.output_dir}"
            )
            if not started:
                self._publish_stop_sequence()
                self._record_event("plan_finished_without_motion", status=status)
        self.next_publish_at = float(now)
        return True

    def poll_worker_results(self, *, now: float) -> int:
        accepted = 0
        while True:
            try:
                item = self.result_queue.get_nowait()
            except queue.Empty:
                return accepted
            accepted += int(self.accept_worker_result(item, now=now))

    def publish_due(self, now: float) -> VelocityCommand | None:
        timestamp = self.controller._timestamp(now)
        if self.next_publish_at is None:
            self.next_publish_at = timestamp
        if timestamp + 1.0e-12 < self.next_publish_at:
            return None
        if self.controller.active:
            was_active = True
            action, command = self.controller.step(timestamp)
        else:
            was_active = False
            action, command = "hold", self.controller.stop_command()
        self.publish(build_velocity_message(command, action=action))
        if was_active and not self.controller.active:
            self._record_event("motion_sequence_completed")
        self.next_publish_at = timestamp + 1.0 / self.config.planner_hz
        return command

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self.stop_event.set()
        self.cancel_and_stop("shutdown", time.monotonic())
        try:
            self.request_queue.put_nowait(None)
        except queue.Full:
            pass


def _offer_result(results: queue.Queue[WorkerResult], item: WorkerResult) -> None:
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
    runner_factory: Callable[[], BasePoseRunner],
    requests: queue.Queue[int | None],
    results: queue.Queue[WorkerResult],
    stop_event: threading.Event,
) -> None:
    runner: BasePoseRunner | None = None
    try:
        while not stop_event.is_set():
            generation = requests.get()
            if generation is None or stop_event.is_set():
                return
            try:
                if runner is None:
                    runner = runner_factory()
                result = runner.run_once()
                item = WorkerResult(generation, result, None)
            except Exception as exc:
                item = WorkerResult(generation, None, str(exc))
            if not stop_event.is_set():
                _offer_result(results, item)
    finally:
        if runner is not None:
            runner.close()


def run_planner_loop(
    runtime: BasePosePlannerRuntime,
    *,
    read_key: Callable[[], str | None],
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    running: Callable[[], bool] = lambda: True,
) -> None:
    try:
        while running():
            now = monotonic()
            key = read_key()
            if key is not None and runtime.handle_key(key, now=now) == "exit":
                break
            runtime.poll_worker_results(now=now)
            runtime.publish_due(now)
            sleep(min(0.01, 0.25 / runtime.config.planner_hz))
    except KeyboardInterrupt:
        pass
    finally:
        runtime.shutdown()


def read_key_nonblocking(stream: TextIO = sys.stdin) -> str | None:
    readable, _, _ = select.select([stream], [], [], 0.0)
    return stream.read(1) if readable else None


@contextmanager
def cbreak_terminal(stream: TextIO = sys.stdin) -> Iterator[None]:
    if not stream.isatty():
        raise RuntimeError("base-pose keyboard input requires an interactive TTY")
    descriptor = stream.fileno()
    original = termios.tcgetattr(descriptor)
    try:
        tty.setcbreak(descriptor)
        yield
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, original)


def _runner_factory(config: BasePosePlannerConfig) -> BasePoseRunner:
    return BasePoseRunner(
        BasePoseConfig(
            task=config.task,
            mode=config.mode,  # type: ignore[arg-type]
            model=config.model,
            reasoning_effort=config.reasoning_effort,
            camera_host=config.camera_host,
            camera_port=config.camera_port,
            camera_timeout_ms=config.camera_timeout_ms,
            camera_stream=config.camera_stream,
            camera_height_m=config.camera_height_m,
            camera_pitch_deg=config.camera_pitch_deg,
            vertical_fov_deg=config.vertical_fov_deg,
            camera_forward_offset_m=config.camera_forward_offset_m,
            camera_lateral_offset_m=config.camera_lateral_offset_m,
            depth_visual_max_m=config.depth_visual_max_m,
            codex_timeout_seconds=config.codex_timeout_seconds,
            output_root=config.output_root,
        )
    )


def main(config: BasePosePlannerConfig) -> None:
    context = zmq.Context.instance()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.LINGER, 0)
    endpoint = f"tcp://{config.host}:{config.port}"
    socket.bind(endpoint)
    runtime = BasePosePlannerRuntime(config, publish=socket.send_string)
    worker = threading.Thread(
        target=run_inference_worker,
        args=(
            lambda: _runner_factory(config),
            runtime.request_queue,
            runtime.result_queue,
            runtime.stop_event,
        ),
        name="base-pose-inference",
        daemon=True,
    )
    worker.start()
    running = True

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGHUP, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        print(
            f"[BasePose] PUB bound to {endpoint}; mode={config.mode}; task={config.task!r}"
        )
        print("[BasePose] N plan | Space cancel-and-stop | X stop-and-exit")
        with cbreak_terminal():
            run_planner_loop(
                runtime, read_key=read_key_nonblocking, running=lambda: running
            )
    finally:
        runtime.shutdown()
        worker.join(timeout=1.0)
        socket.close()
        print("[BasePose] Stopped")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(BasePosePlannerConfig))
