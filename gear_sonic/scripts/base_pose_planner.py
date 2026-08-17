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
from typing import Any, Callable, Iterator, Literal, TextIO

import zmq

from gear_sonic.camera.calibration import DEFAULT_CAMERA_INTRINSICS_PATH
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
    vision_backend: Literal["codex", "qwenvl"] = "codex"
    model: str = "gpt-5.6-sol"
    qwenvl_model: str = "qwen3-vl-plus"
    qwenvl_base_url: str = (
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    )
    qwenvl_thinking_budget: int = 500
    reasoning_effort: str = "xhigh"
    codex_fast: bool = True
    host: str = "*"
    port: int = 5558
    planner_hz: float = 20.0
    transition_pause: float = 0.5
    final_stop_count: int = 3
    rotation_speed: float = 0.4
    translation_speed: float = 0.3
    rotation_scale: float = 1.0
    translation_scale: float = 1.0
    camera_host: str = "localhost"
    camera_port: int = 5555
    camera_timeout_ms: int = 15000
    camera_stream: str = "ego_view"
    camera_intrinsics_path: str = str(DEFAULT_CAMERA_INTRINSICS_PATH)
    camera_height_m: float = 1.2
    camera_pitch_deg: float = -38.0
    vertical_fov_deg: float = 43.077882
    camera_roll_deg: float = 0.0
    camera_yaw_deg: float = 0.0
    camera_forward_offset_m: float = 0.0
    camera_lateral_offset_m: float = 0.0
    dual_head_camera_stream: str = "ego_view"
    dual_chest_camera_stream: str = "chest_view"
    dual_chest_camera_height_m: float = 1.0
    dual_chest_camera_pitch_deg: float = -3.0
    dual_chest_camera_roll_deg: float = 0.0
    dual_chest_camera_yaw_deg: float = 0.0
    dual_chest_camera_forward_offset_m: float = 0.0
    dual_chest_camera_lateral_offset_m: float = 0.0
    dual_match_tolerance_frames: int = 30
    dual_initialization_grace_s: float = 30.0
    depth_visual_max_m: float = 3.0
    codex_timeout_seconds: float = 600.0
    output_root: str = "outputs/base_pose_adjustment"
    raw_yoloe_model_path: str = "tools/yoloe26m/weights/yoloe-26m-seg.pt"
    raw_yoloe_device: str = "0"
    raw_yoloe_confidence: float = 0.25
    raw_yoloe_imgsz: int = 640
    raw_reference_update_interval_frames: int = 5
    raw_reference_update_min_confidence: float = 0.35
    raw_reference_update_min_iou: float = 0.50
    raw_reference_update_freeze_y2_px: float = 75.0
    raw_reference_update_resume_y2_px: float = 110.0
    raw_servo_hz: float = 10.0
    raw_target_distance_m: float = 0.80
    raw_forward_tolerance_m: float = 0.10
    raw_lateral_tolerance_m: float = 0.10
    raw_max_lateral_speed_m_s: float = 0.16
    raw_horizontal_guard_fraction: float = 0.25
    raw_horizontal_recovery_fraction: float = 0.30
    raw_orientation_telemetry_source: str = "tcp://127.0.0.1:5565"
    raw_command_ttl_s: float = 0.15
    raw_camera_stale_s: float = 0.4
    raw_max_run_s: float = 180.0


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
    rotation_scale: float = 1.0,
    translation_scale: float = 1.0,
) -> tuple[MotionSegment, ...]:
    """Scale validated model values and map them to fixed-speed commands."""
    if not math.isfinite(rotation_speed) or rotation_speed <= 0.0:
        raise ValueError("rotation_speed must be finite and positive")
    if not math.isfinite(translation_speed) or translation_speed <= 0.0:
        raise ValueError("translation_speed must be finite and positive")
    if not math.isfinite(rotation_scale) or rotation_scale <= 0.0:
        raise ValueError("rotation_scale must be finite and positive")
    if not math.isfinite(translation_scale) or translation_scale <= 0.0:
        raise ValueError("translation_scale must be finite and positive")
    validated = validate_base_pose_plan(plan)
    if validated["status"] != "ADJUST":
        return ()
    segments: list[MotionSegment] = []
    for item in validated["command_sequence"]:
        action = str(item["action"])
        scale = (
            rotation_scale
            if action.startswith("ROTATE_")
            else translation_scale
        )
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
        rotation_scale: float = 1.0,
        translation_scale: float = 1.0,
        transition_pause: float = 0.5,
        stop_duration: float = 0.05,
    ):
        values = (
            rotation_speed,
            translation_speed,
            rotation_scale,
            translation_scale,
            stop_duration,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError(
                "controller speeds, scales, and stop_duration must be positive"
            )
        if not math.isfinite(transition_pause) or transition_pause < 0.0:
            raise ValueError("transition_pause must be finite and non-negative")
        self.rotation_speed = rotation_speed
        self.translation_speed = translation_speed
        self.rotation_scale = rotation_scale
        self.translation_scale = translation_scale
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
            rotation_scale=config.rotation_scale,
            translation_scale=config.translation_scale,
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
                rotation_scale=self.config.rotation_scale,
                translation_scale=self.config.translation_scale,
            )
            execution = {
                "rotation_scale": self.config.rotation_scale,
                "translation_scale": self.config.translation_scale,
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
            vision_backend=config.vision_backend,  # type: ignore[arg-type]
            model=config.model,
            qwenvl_model=config.qwenvl_model,
            qwenvl_base_url=config.qwenvl_base_url,
            qwenvl_thinking_budget=config.qwenvl_thinking_budget,
            reasoning_effort=config.reasoning_effort,
            codex_fast=config.codex_fast,
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


def _raw_servo_main(
    config: BasePosePlannerConfig,
    socket: zmq.Socket,
    endpoint: str,
) -> None:
    from gear_sonic.utils.inference.base_pose_visual_servo import (
        RawServoRuntime,
        run_raw_servo_loop,
        run_raw_servo_worker,
        validate_raw_servo_dependencies,
    )
    from gear_sonic.utils.inference.base_pose_dual_visual_servo import (
        run_dual_raw_servo_worker,
    )
    from gear_sonic.utils.teleop.sonic_orientation_telemetry import (
        LatestOrientationTelemetry,
    )

    validate_raw_servo_dependencies(config)
    orientation_socket = None
    orientation_provider = None
    if config.raw_orientation_telemetry_source:
        orientation_socket = zmq.Context.instance().socket(zmq.SUB)
        orientation_socket.setsockopt(zmq.SUBSCRIBE, b"")
        orientation_socket.setsockopt(zmq.CONFLATE, 1)
        orientation_socket.setsockopt(zmq.LINGER, 0)
        orientation_socket.connect(config.raw_orientation_telemetry_source)
        latest_orientation = LatestOrientationTelemetry()
        last_warning_at = -math.inf

        def read_orientation(now: float) -> dict[str, float | None] | None:
            nonlocal last_warning_at
            while True:
                try:
                    raw = orientation_socket.recv(zmq.NOBLOCK)
                except zmq.Again:
                    break
                try:
                    latest_orientation.update(raw)
                except ValueError as exc:
                    if now - last_warning_at >= 1.0:
                        print(
                            "[RawServo] WARNING ignored orientation telemetry: "
                            f"{exc}"
                        )
                        last_warning_at = now
            return latest_orientation.diagnostics(now)

        orientation_provider = read_orientation
    runtime = RawServoRuntime(
        config,
        publish=socket.send_string,
        orientation_provider=orientation_provider,
    )
    worker_target = (
        run_dual_raw_servo_worker
        if config.mode == "dual_raw_yoloe_servo"
        else run_raw_servo_worker
    )
    worker_kwargs = {
        "observation_events": runtime.observation_events,
        "diagnostics": runtime.diagnostics,
    }
    if config.mode == "dual_raw_yoloe_servo":
        worker_kwargs["table_required"] = (
            lambda: runtime.controller.table_required
        )
    worker = threading.Thread(
        target=worker_target,
        args=(
            config,
            runtime.requests,
            runtime.events,
            runtime.gate,
            runtime.stop_event,
        ),
        name="base-pose-raw-yoloe-servo",
        daemon=True,
        kwargs=worker_kwargs,
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
            f"[RawServo] PUB bound to {endpoint}; YOLOE={config.raw_yoloe_model_path}; "
            f"visual_hz={config.raw_servo_hz}; "
            f"standoff={config.raw_target_distance_m:.3f}m; task={config.task!r}"
        )
        print(
            "[RawServo] N initialize+align | Space cancel-and-stop | X stop-and-exit"
        )
        with cbreak_terminal():
            run_raw_servo_loop(
                runtime, read_key=read_key_nonblocking, running=lambda: running
            )
    finally:
        runtime.shutdown()
        worker.join()
        runtime.flush_diagnostics()
        if orientation_socket is not None:
            orientation_socket.close()
        print("[RawServo] Stopped")


def main(config: BasePosePlannerConfig) -> None:
    context = zmq.Context.instance()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.LINGER, 0)
    endpoint = f"tcp://{config.host}:{config.port}"
    socket.bind(endpoint)
    if config.mode in {"raw_yoloe_servo", "dual_raw_yoloe_servo"}:
        try:
            _raw_servo_main(config, socket, endpoint)
        finally:
            socket.close()
        return
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
        codex_fast = "on" if config.codex_fast else "off"
        print(
            f"[BasePose] PUB bound to {endpoint}; mode={config.mode}; "
            f"backend={config.vision_backend}; codex_fast={codex_fast}; "
            f"qwenvl_thinking_budget={config.qwenvl_thinking_budget}; "
            f"task={config.task!r}"
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
