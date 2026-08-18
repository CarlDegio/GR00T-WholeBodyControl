#!/usr/bin/env python3
"""Run raw-depth YOLOE base-pose visual servo through SONIC."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import math
import select
import signal
import subprocess
import sys
import termios
import threading
import time
import tty
from typing import Any, Callable, Iterator, Literal, TextIO

import zmq

from gear_sonic.camera.calibration import DEFAULT_CAMERA_INTRINSICS_PATH
@dataclass
class BasePosePlannerConfig:
    task: str
    mode: str = "raw_yoloe_servo"
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
    final_stop_count: int = 3
    camera_host: str = "localhost"
    camera_port: int = 5555
    camera_timeout_ms: int = 15000
    camera_stream: str = "ego_view"
    camera_intrinsics_path: str = str(DEFAULT_CAMERA_INTRINSICS_PATH)
    camera_pitch_deg: float = -38.0
    camera_roll_deg: float = 0.0
    camera_yaw_deg: float = 0.0
    camera_forward_offset_m: float = 0.0
    camera_lateral_offset_m: float = 0.0
    dual_head_camera_stream: str = "ego_view"
    dual_chest_camera_stream: str = "chest_view"
    dual_chest_camera_pitch_deg: float = -3.0
    dual_chest_camera_roll_deg: float = 0.0
    dual_chest_camera_yaw_deg: float = 0.0
    dual_chest_camera_forward_offset_m: float = 0.0
    dual_chest_camera_lateral_offset_m: float = 0.0
    dual_match_tolerance_frames: int = 30
    dual_initialization_grace_s: float = 30.0
    dual_qwenvl_fallback_model: str = "qwen3-vl-8b-instruct"
    codex_timeout_seconds: float = 600.0
    output_root: str = "outputs/base_pose_adjustment"
    raw_yoloe_model_path: str = "tools/yoloe26m/weights/yoloe-26m-seg.pt"
    raw_yoloe_device: str = "0"
    raw_yoloe_confidence: float = 0.25
    raw_yoloe_imgsz: int = 640
    raw_reference_update_interval_frames: int = 5
    raw_reference_update_min_confidence: float = 0.35
    raw_reference_update_min_iou: float = 0.50
    raw_servo_hz: float = 10.0
    raw_head_target_distance_m: float = 0.90
    raw_chest_target_distance_m: float = 0.80
    raw_forward_tolerance_m: float = 0.10
    raw_lateral_tolerance_m: float = 0.10
    raw_min_linear_speed_m_s: float = 0.40
    raw_max_lateral_speed_m_s: float = 0.40
    raw_min_yaw_speed_rad_s: float = 0.10
    raw_yaw_tolerance_deg: float = 8.0
    raw_yaw_coarse_speed_rad_s: float = 0.30
    raw_yaw_trim_speed_rad_s: float = 0.20
    raw_forward_recenter_yaw_speed_rad_s: float = 0.30
    raw_live_camera_viewer: bool = True
    raw_horizontal_guard_fraction: float = 0.25
    raw_horizontal_recovery_fraction: float = 0.30
    raw_orientation_telemetry_source: str = "tcp://127.0.0.1:5565"
    raw_command_ttl_s: float = 0.15
    raw_camera_stale_s: float = 0.4
    raw_max_run_s: float = 180.0
    raw_post_stop_sample_s: float = 3.0
    raw_allow_missing_table: Literal[0, 1] = 0


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


def build_raw_servo_camera_viewer_command(
    config: BasePosePlannerConfig,
) -> list[str]:
    """Build the two-camera OpenCV viewer command used during raw servo."""
    repo_root = Path(__file__).resolve().parents[2]
    viewer_python = repo_root / ".venv_teleop" / "bin" / "python"
    viewer = Path(__file__).with_name("run_camera_viewer.py")
    streams = ",".join(
        (
            str(config.dual_head_camera_stream),
            str(config.dual_chest_camera_stream),
        )
    )
    return [
        str(viewer_python),
        str(viewer),
        "--camera-host",
        str(config.camera_host),
        "--camera-port",
        str(config.camera_port),
        "--camera-streams",
        streams,
        "--window-name",
        "Base Pose Cameras",
        "--status-endpoint",
        _raw_servo_viewer_status_endpoint(config),
    ]


def _raw_servo_viewer_status_endpoint(config: BasePosePlannerConfig) -> str:
    """Return a local subscriber endpoint for the planner's PUB socket."""
    host = str(config.host).strip()
    if host in {"", "*", "0.0.0.0", "::", "[::]"}:
        host = "127.0.0.1"
    return f"tcp://{host}:{int(config.port)}"


class RawServoCameraViewer:
    """Own one OpenCV viewer process for the active N-triggered run."""

    def __init__(
        self,
        config: BasePosePlannerConfig,
        *,
        popen: Callable[..., Any] | None = None,
        logger: Callable[[str], None] = print,
    ):
        self.config = config
        self._popen = subprocess.Popen if popen is None else popen
        self.logger = logger
        self.process: Any | None = None

    def start(self) -> None:
        if not self.config.raw_live_camera_viewer:
            return
        if self.process is not None and self.process.poll() is None:
            return
        self.process = self._popen(
            build_raw_servo_camera_viewer_command(self.config),
            start_new_session=True,
        )
        self.logger("[RawServo] opened live head/chest camera window")

    def stop(self) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1.0)


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
    camera_viewer = RawServoCameraViewer(config)
    runtime = RawServoRuntime(
        config,
        publish=socket.send_string,
        orientation_provider=orientation_provider,
        navigation_started=camera_viewer.start,
        navigation_finished=camera_viewer.stop,
    )
    worker_target = (
        run_dual_raw_servo_worker
        if config.mode == "dual_raw_yoloe_servo"
        else run_raw_servo_worker
    )
    worker_kwargs = {
        "observation_events": runtime.observation_events,
        "diagnostics": runtime.diagnostics,
        "table_required": lambda: runtime.controller.table_required,
    }
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
            f"head_standoff={config.raw_head_target_distance_m:.3f}m; "
            f"chest_standoff={config.raw_chest_target_distance_m:.3f}m; "
            f"task={config.task!r}"
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
        camera_viewer.stop()
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
    socket.close()
    raise ValueError(f"unsupported base-pose YOLOE mode: {config.mode}")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(BasePosePlannerConfig))
