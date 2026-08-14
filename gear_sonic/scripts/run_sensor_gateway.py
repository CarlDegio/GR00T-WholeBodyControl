#!/usr/bin/env python3
"""Run the read-only SONIC sensor aggregation and Snapshot service."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import signal
import sys
import threading
import time

import zmq

from gear_sonic.runtime.config import load_runtime_profile
from gear_sonic.runtime.sensor_gateway import (
    CameraZmqIngress,
    CppStateZmqIngress,
    LingBotDepthZmqIngress,
    Ros2SensorIngress,
    SensorGatewayCore,
    SensorGatewayRpcServer,
    VisualizationZmqIngress,
)
from gear_sonic.runtime.vla_timing import (
    VLA_TIMING_SEGMENTS,
    VlaTimingIngress,
    VlaTimingWindow,
)


CONTROL_OUTPUTS_ENABLED = False


@dataclass(frozen=True)
class SensorGatewaySettings:
    profile_name: str
    camera_endpoint: str
    lingbot_depth_endpoint: str
    cpp_state_endpoint: str
    rpc_bind_endpoint: str
    visualization_bind_endpoint: str
    vla_timing_bind_endpoint: str
    ros_topics: dict[str, str]
    slot_count: int
    history_size: int
    frame_ttl_ms: int
    retired_ring_ttl_s: float
    loop_hz: float
    health_print_interval_s: float
    expected_hz: dict[str, float]
    enable_camera: bool
    enable_lingbot_depth: bool
    enable_cpp_state: bool
    enable_ros: bool
    enable_visualization: bool
    enable_vla_timing: bool
    enable_rgb_preview: bool
    vla_timing_window_size: int


def _positive(value: float, name: str) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def resolve_sensor_gateway_settings(args: argparse.Namespace) -> SensorGatewaySettings:
    profile = load_runtime_profile(
        args.profile or None,
        overlays=tuple(args.overlay),
    )
    component = profile.component("sensor_gateway")

    def endpoint_uri(name: str, host_override: str, port_override: int) -> str:
        address = profile.endpoint(name)
        host = host_override or address.host
        port = int(port_override) if port_override else address.port
        return f"tcp://{host}:{port}"

    expected_hz = {
        "camera": _positive(component["camera_expected_hz"], "camera_expected_hz"),
        "cpp_state": _positive(
            component["cpp_state_expected_hz"],
            "cpp_state_expected_hz",
        ),
        "lidar": _positive(component["lidar_expected_hz"], "lidar_expected_hz"),
        "imu": _positive(component["imu_expected_hz"], "imu_expected_hz"),
        "odometry": _positive(
            component["odometry_expected_hz"],
            "odometry_expected_hz",
        ),
        "registered_cloud": _positive(
            component["registered_cloud_expected_hz"],
            "registered_cloud_expected_hz",
        ),
        "visualization": _positive(
            component["visualization_expected_hz"],
            "visualization_expected_hz",
        ),
        "vla_timing": _positive(
            component["vla_timing_expected_hz"],
            "vla_timing_expected_hz",
        ),
        "lingbot_depth": _positive(
            profile.component("lingbot_depth")["inference_hz"],
            "lingbot_depth.inference_hz",
        ),
    }
    slot_count = int(component["shared_memory_slots"])
    history_size = int(component["snapshot_history_size"])
    frame_ttl_ms = int(component["frame_ttl_ms"])
    if slot_count < 2 or history_size < 2 or frame_ttl_ms < 0:
        raise ValueError("invalid SensorGateway shared-memory configuration")

    return SensorGatewaySettings(
        profile_name=profile.name,
        camera_endpoint=endpoint_uri(
            "camera_server",
            args.camera_host,
            args.camera_port,
        ),
        lingbot_depth_endpoint=endpoint_uri(
            "lingbot_depth",
            args.lingbot_depth_host,
            args.lingbot_depth_port,
        ),
        cpp_state_endpoint=endpoint_uri(
            "cpp_state",
            args.cpp_state_host,
            args.cpp_state_port,
        ),
        rpc_bind_endpoint=endpoint_uri(
            "sensor_gateway_metadata",
            args.rpc_bind_host,
            args.rpc_port,
        ),
        visualization_bind_endpoint=endpoint_uri(
            "sensor_gateway_visualization_ingress",
            args.visualization_bind_host,
            args.visualization_port,
        ),
        vla_timing_bind_endpoint=endpoint_uri(
            "vla_timing_ingress",
            args.vla_timing_bind_host,
            args.vla_timing_port,
        ),
        ros_topics={
            "lidar": str(profile.ros_topics["lidar"]),
            "imu": str(profile.ros_topics["lidar_imu"]),
            "odometry": str(profile.ros_topics["odometry"]),
            "registered_cloud": str(profile.ros_topics["registered_cloud"]),
        },
        slot_count=slot_count,
        history_size=history_size,
        frame_ttl_ms=frame_ttl_ms,
        retired_ring_ttl_s=float(component["retired_ring_ttl_s"]),
        loop_hz=_positive(component["loop_hz"], "loop_hz"),
        health_print_interval_s=float(args.health_print_interval_s),
        expected_hz=expected_hz,
        enable_camera=bool(args.enable_camera),
        enable_lingbot_depth=bool(args.enable_lingbot_depth),
        enable_cpp_state=bool(args.enable_cpp_state),
        enable_ros=bool(args.enable_ros),
        enable_visualization=bool(args.enable_visualization),
        enable_vla_timing=bool(args.enable_vla_timing),
        enable_rgb_preview=bool(args.enable_rgb_preview),
        vla_timing_window_size=int(component["vla_timing_window_size"]),
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="", help="Base runtime YAML profile")
    parser.add_argument(
        "--overlay",
        action="append",
        default=[],
        help="Partial YAML overlay; may be repeated and is applied left-to-right",
    )
    parser.add_argument("--camera-host", default="")
    parser.add_argument("--camera-port", type=int, default=0)
    parser.add_argument("--lingbot-depth-host", default="")
    parser.add_argument("--lingbot-depth-port", type=int, default=0)
    parser.add_argument("--cpp-state-host", default="")
    parser.add_argument("--cpp-state-port", type=int, default=0)
    parser.add_argument("--rpc-bind-host", default="")
    parser.add_argument("--rpc-port", type=int, default=0)
    parser.add_argument("--visualization-bind-host", default="")
    parser.add_argument("--visualization-port", type=int, default=0)
    parser.add_argument("--vla-timing-bind-host", default="")
    parser.add_argument("--vla-timing-port", type=int, default=0)
    parser.add_argument("--health-print-interval-s", type=float, default=0.5)
    parser.add_argument(
        "--enable-camera",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enable-lingbot-depth",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enable-cpp-state",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enable-ros",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enable-visualization",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enable-vla-timing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enable-rgb-preview",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def _health_line(payload: dict) -> str:
    sources = {
        name: value
        for name, value in payload["streams"].items()
        if name.startswith("source/")
    }
    return " | ".join(
        f"{name.removeprefix('source/')}={value['state']} "
        f"{value['rate_hz']:.1f}Hz age={value['last_message_age_ms']}ms"
        for name, value in sources.items()
    )


def _metric(value: float | None, *, suffix: str = "") -> str:
    return "-" if value is None else f"{float(value):.1f}{suffix}"


def _health_dashboard_text(
    payload: dict,
    settings: SensorGatewaySettings,
    vla_timing: dict | None = None,
) -> str:
    """Build a fixed-width live view without inventing clock-based latency."""

    sources = payload.get("streams", {})
    rows = (
        ("Camera", settings.camera_endpoint, "source/camera_server"),
        ("LingBot depth", settings.lingbot_depth_endpoint, "source/lingbot_depth"),
        ("C++ state", settings.cpp_state_endpoint, "source/cpp_state"),
        ("Visualization", settings.visualization_bind_endpoint, "source/visualization_ingress"),
        ("VLA timing", settings.vla_timing_bind_endpoint, "source/vla_timing"),
        ("LiDAR", settings.ros_topics["lidar"], "source/ros_lidar"),
        ("LiDAR IMU", settings.ros_topics["imu"], "source/ros_imu"),
        ("Odometry", settings.ros_topics["odometry"], "source/ros_odometry"),
        (
            "Point cloud",
            settings.ros_topics["registered_cloud"],
            "source/ros_registered_cloud",
        ),
    )
    lines = [
        "SONIC SensorGateway — live source timing (fixed display)",
        f"Snapshot RPC: {settings.rpc_bind_endpoint}   updated: {time.strftime('%H:%M:%S')}",
        "",
        f"{'SOURCE':<15} {'ENDPOINT / TOPIC':<40} {'STATE':<8} {'RATE':>9} {'AGE':>10} {'LATENCY':>10}",
        "-" * 98,
    ]
    for label, endpoint, health_name in rows:
        value = sources.get(health_name, {})
        state = str(value.get("state", "disabled"))
        rate = _metric(value.get("rate_hz"), suffix="Hz")
        age = _metric(value.get("last_message_age_ms"), suffix="ms")
        latency = _metric(value.get("latency_ms"), suffix="ms")
        lines.append(
            f"{label:<15} {endpoint:<40.40} {state:<8.8} {rate:>9} {age:>10} {latency:>10}"
        )
    timing = vla_timing or {}
    segment_stats = timing.get("segments_ms", {})
    lines.extend(
        (
            "",
            "VLA segmented latency (milliseconds; rolling window)",
            f"samples={timing.get('sample_count', 0)}  "
            f"window={timing.get('window_size', settings.vla_timing_window_size)}  "
            f"sample_age={_metric(timing.get('last_sample_age_ms'), suffix='ms')}",
            f"{'SEGMENT':<22} {'LAST':>10} {'MEAN':>10} {'P50':>10} {'P95':>10}",
            "-" * 66,
        )
    )
    for name in VLA_TIMING_SEGMENTS:
        stats = segment_stats.get(name, {})
        lines.append(
            f"{name:<22} {_metric(stats.get('last')):>10} "
            f"{_metric(stats.get('mean')):>10} {_metric(stats.get('p50')):>10} "
            f"{_metric(stats.get('p95')):>10}"
        )
    lines.extend(
        (
            "",
            "AGE = time since Gateway last received a message; LATENCY is shown only when clocks are comparable.",
            "policy_roundtrip includes local transport plus remote server processing.",
            "Ctrl-C stops SensorGateway. This process is read-only; robot control uses the separate ControlGateway/C++ path.",
        )
    )
    return "\n".join(lines)


class _HealthDisplay:
    """Use an alternate terminal screen in tmux so health output never scrolls."""

    def __init__(self) -> None:
        self.interactive = bool(sys.stdout.isatty())
        self.active = False

    def render(
        self,
        payload: dict,
        settings: SensorGatewaySettings,
        vla_timing: dict | None = None,
    ) -> None:
        if not self.interactive:
            line = _health_line(payload)
            print(f"[SensorGateway] {line or 'waiting for enabled sources'}")
            return
        if not self.active:
            sys.stdout.write("\x1b[?1049h\x1b[?25l")
            self.active = True
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write(_health_dashboard_text(payload, settings, vla_timing))
        sys.stdout.flush()

    def close(self) -> None:
        if self.active:
            sys.stdout.write("\x1b[?25h\x1b[?1049l")
            sys.stdout.flush()
            self.active = False


def run_sensor_gateway(settings: SensorGatewaySettings) -> None:
    if CONTROL_OUTPUTS_ENABLED:
        raise RuntimeError("SensorGateway must remain read-only")
    if settings.health_print_interval_s < 0.0:
        raise ValueError("health_print_interval_s cannot be negative")

    context = zmq.Context()
    core = SensorGatewayCore(
        slot_count=settings.slot_count,
        history_size=settings.history_size,
        frame_ttl_ms=settings.frame_ttl_ms,
        retired_ring_ttl_s=settings.retired_ring_ttl_s,
    )
    rpc: SensorGatewayRpcServer | None = None
    camera: CameraZmqIngress | None = None
    lingbot_depth: LingBotDepthZmqIngress | None = None
    cpp_state: CppStateZmqIngress | None = None
    ros: Ros2SensorIngress | None = None
    visualization: VisualizationZmqIngress | None = None
    vla_timing_ingress: VlaTimingIngress | None = None
    vla_timing_window = VlaTimingWindow(settings.vla_timing_window_size)
    stop = threading.Event()
    health_display = _HealthDisplay()

    def request_stop(_signum=None, _frame=None) -> None:
        stop.set()

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        rpc = SensorGatewayRpcServer(context, settings.rpc_bind_endpoint, core)
        rpc.start()
        if settings.enable_visualization:
            visualization = VisualizationZmqIngress(
                context,
                settings.visualization_bind_endpoint,
                core,
                expected_hz=settings.expected_hz["visualization"],
            )
        if settings.enable_vla_timing:
            vla_timing_ingress = VlaTimingIngress(
                context,
                settings.vla_timing_bind_endpoint,
                vla_timing_window,
                observe=lambda received_ns: core.observe_endpoint(
                    "source/vla_timing",
                    received_ns=received_ns,
                    expected_hz=settings.expected_hz["vla_timing"],
                ),
            )
        if settings.enable_camera:
            camera = CameraZmqIngress(
                context,
                settings.camera_endpoint,
                core,
                expected_hz=settings.expected_hz["camera"],
                preview_rgb=settings.enable_rgb_preview,
            )
        if settings.enable_lingbot_depth:
            lingbot_depth = LingBotDepthZmqIngress(
                context,
                settings.lingbot_depth_endpoint,
                core,
                expected_hz=settings.expected_hz["lingbot_depth"],
            )
        if settings.enable_cpp_state:
            cpp_state = CppStateZmqIngress(
                context,
                settings.cpp_state_endpoint,
                core,
                expected_hz=settings.expected_hz["cpp_state"],
            )
        if settings.enable_ros:
            ros = Ros2SensorIngress(
                core,
                lidar_topic=settings.ros_topics["lidar"],
                imu_topic=settings.ros_topics["imu"],
                odometry_topic=settings.ros_topics["odometry"],
                registered_cloud_topic=settings.ros_topics["registered_cloud"],
                lidar_expected_hz=settings.expected_hz["lidar"],
                imu_expected_hz=settings.expected_hz["imu"],
                odometry_expected_hz=settings.expected_hz["odometry"],
                registered_cloud_expected_hz=settings.expected_hz["registered_cloud"],
            )
            ros.start()

        print("[SensorGateway] READ-ONLY mode; no control output sockets exist")
        print(f"[SensorGateway] profile: {settings.profile_name}")
        print(f"[SensorGateway] Snapshot/health REP: {settings.rpc_bind_endpoint}")
        if visualization is not None:
            print(
                "[SensorGateway] visualization PULL: "
                f"{settings.visualization_bind_endpoint}"
            )
        if vla_timing_ingress is not None:
            print(f"[SensorGateway] VLA timing PULL: {settings.vla_timing_bind_endpoint}")
        if camera is not None:
            print(f"[SensorGateway] camera SUB: {settings.camera_endpoint}")
        if lingbot_depth is not None:
            print(
                "[SensorGateway] LingBot depth SUB: "
                f"{settings.lingbot_depth_endpoint} -> derived/lingbot_depth"
            )
        if cpp_state is not None:
            print(f"[SensorGateway] C++ state SUB: {settings.cpp_state_endpoint}")
        if ros is not None:
            print(f"[SensorGateway] ROS2 topics: {json.dumps(settings.ros_topics)}")

        period_s = 1.0 / settings.loop_hz
        next_health = time.monotonic()
        while not stop.is_set():
            loop_started = time.monotonic()
            if camera is not None:
                try:
                    camera.poll_once()
                except Exception as exc:
                    core.record_failure(
                        "source/camera_server",
                        str(exc),
                        expected_hz=settings.expected_hz["camera"],
                    )
            if lingbot_depth is not None:
                try:
                    lingbot_depth.poll_once()
                except Exception as exc:
                    core.record_failure(
                        "source/lingbot_depth",
                        str(exc),
                        expected_hz=settings.expected_hz["lingbot_depth"],
                    )
            if cpp_state is not None:
                try:
                    cpp_state.poll_once()
                except Exception as exc:
                    core.record_failure(
                        "source/cpp_state",
                        str(exc),
                        expected_hz=settings.expected_hz["cpp_state"],
                    )
            if visualization is not None:
                try:
                    while visualization.poll_once():
                        pass
                except Exception as exc:
                    core.record_failure(
                        "source/visualization_ingress",
                        str(exc),
                        expected_hz=settings.expected_hz["visualization"],
                    )
            if vla_timing_ingress is not None:
                try:
                    while vla_timing_ingress.poll_once():
                        pass
                except Exception as exc:
                    core.record_failure(
                        "source/vla_timing",
                        str(exc),
                        expected_hz=settings.expected_hz["vla_timing"],
                    )
            now = time.monotonic()
            if settings.health_print_interval_s > 0.0 and now >= next_health:
                health_display.render(
                    core.health_payload(),
                    settings,
                    vla_timing_window.snapshot(),
                )
                next_health = now + settings.health_print_interval_s
            stop.wait(max(0.0, period_s - (time.monotonic() - loop_started)))
    finally:
        health_display.close()
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        if ros is not None:
            ros.close()
        if visualization is not None:
            visualization.close()
        if vla_timing_ingress is not None:
            vla_timing_ingress.close()
        if cpp_state is not None:
            cpp_state.close()
        if camera is not None:
            camera.close()
        if lingbot_depth is not None:
            lingbot_depth.close()
        if rpc is not None:
            rpc.close()
        core.close()
        context.term()
        print("[SensorGateway] stopped; shared-memory segments released")


def main() -> None:
    args = build_argument_parser().parse_args()
    run_sensor_gateway(resolve_sensor_gateway_settings(args))


if __name__ == "__main__":
    main()
