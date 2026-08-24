#!/usr/bin/env python3
"""Run the read-only SONIC sensor aggregation and Snapshot service."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import signal
import sys
import threading
import time

import zmq

from gear_sonic.runtime.profile import load_runtime_profile
from gear_sonic.runtime.gateway.sensor import (
    CameraZmqIngress,
    CppStateZmqIngress,
    DepthAnythingZmqIngress,
    Ros2SensorIngress,
    SensorGatewayCore,
    SensorGatewayRpcServer,
    VisualizationZmqIngress,
)
from gear_sonic.runtime.telemetry import (
    RUNTIME_METRIC_SEGMENTS,
    configure_file_logging,
    create_metrics_window,
    metrics_snapshot,
    poll_metrics,
)
from gear_sonic.runtime.zmq_sockets import bind_pull


_SHUTDOWN_SIGNALS = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)


@dataclass(frozen=True)
class SensorGatewaySettings:
    profile_name: str
    camera_endpoint: str
    depth_anything_endpoint: str
    cpp_state_endpoint: str
    rpc_bind_endpoint: str
    visualization_bind_endpoint: str
    runtime_metrics_bind_endpoint: str
    ros_topics: dict[str, str]
    slot_count: int
    history_size: int
    frame_ttl_ms: int
    retired_ring_ttl_s: float
    loop_hz: float
    health_print_interval_s: float
    expected_hz: dict[str, float]
    enable_camera: bool
    enable_depth_anything: bool
    enable_cpp_state: bool
    enable_ros: bool
    enable_visualization: bool
    enable_runtime_metrics: bool
    runtime_metrics_window_size: int


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
        "depth_anything": _positive(
            profile.component("depth_anything")["inference_hz"],
            "depth_anything.inference_hz",
        ),
    }
    slot_count = int(component["shared_memory_slots"])
    history_size = int(component["snapshot_history_size"])
    frame_ttl_ms = int(component["frame_ttl_ms"])
    if slot_count < 2 or history_size < 2 or frame_ttl_ms < 0:
        raise ValueError("invalid SensorGateway shared-memory configuration")

    return SensorGatewaySettings(
        profile_name=profile.name,
        camera_endpoint=profile.endpoint_uri("camera_server"),
        depth_anything_endpoint=profile.endpoint_uri("depth_anything"),
        cpp_state_endpoint=profile.endpoint_uri("cpp_state"),
        rpc_bind_endpoint=profile.endpoint_uri("sensor_gateway_metadata"),
        visualization_bind_endpoint=profile.endpoint_uri(
            "sensor_gateway_visualization_ingress"
        ),
        runtime_metrics_bind_endpoint=profile.endpoint_uri("runtime_metrics_ingress"),
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
        enable_depth_anything=bool(args.enable_depth_anything),
        enable_cpp_state=bool(args.enable_cpp_state),
        enable_ros=bool(args.enable_ros),
        enable_visualization=bool(args.enable_visualization),
        enable_runtime_metrics=bool(args.enable_runtime_metrics),
        runtime_metrics_window_size=int(component["runtime_metrics_window_size"]),
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
    parser.add_argument("--health-print-interval-s", type=float, default=0.5)
    parser.add_argument(
        "--enable-camera",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enable-depth-anything",
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
        "--enable-runtime-metrics",
        "--enable-vla-timing",
        dest="enable_runtime_metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _source_state_changes(
    payload: dict,
    previous: dict[str, str],
) -> list[tuple[str, str, str | None, str]]:
    changes = []
    for name, value in payload.get("streams", {}).items():
        if not name.startswith("source/"):
            continue
        state = str(value.get("state", "waiting"))
        prior = previous.get(name)
        previous[name] = state
        if state in {"stale", "down"} and state != prior:
            changes.append((name, state, prior, str(value.get("last_error", ""))))
        elif state == "healthy" and prior in {"stale", "down"}:
            changes.append((name, state, prior, ""))
    return changes


def _metric(value: float | None, *, suffix: str = "") -> str:
    return "-" if value is None else f"{float(value):.1f}{suffix}"


def _health_dashboard_text(
    payload: dict,
    settings: SensorGatewaySettings,
    active_component: str | None = None,
    runtime_metrics: dict | None = None,
    metrics_error: str = "",
) -> str:
    """Build a fixed-width live view without inventing clock-based latency."""

    sources = payload.get("streams", {})
    rows = (
        ("Camera", settings.camera_endpoint, "source/camera_server"),
        (
            "DA metric depth",
            settings.depth_anything_endpoint,
            "source/depth_anything",
        ),
        ("C++ state", settings.cpp_state_endpoint, "source/cpp_state"),
        ("Visualization", settings.visualization_bind_endpoint, "source/visualization_ingress"),
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
    timing = runtime_metrics or {}
    segment_stats = timing.get("values", {})
    component_label = (
        "LaViRA / ObjectNav"
        if active_component == "lavira"
        else active_component.upper() if active_component else "waiting"
    )
    lines.extend(
        (
            "",
            f"ACTIVE TASK: {component_label}",
            f"Runtime metrics: {settings.runtime_metrics_bind_endpoint}",
            f"samples={timing.get('sample_count', 0)}  "
            f"window={timing.get('window_size', settings.runtime_metrics_window_size)}  "
            f"sample_age={_metric(timing.get('last_sample_age_ms'), suffix='ms')}",
            f"{'SEGMENT':<22} {'LAST':>10} {'MEAN':>10} {'P50':>10} {'P95':>10}",
            "-" * 66,
        )
    )
    for name in RUNTIME_METRIC_SEGMENTS.get(active_component, ()):
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
            f"metrics_error={metrics_error or '-'}",
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
        active_component: str | None = None,
        runtime_metrics: dict | None = None,
        metrics_error: str = "",
    ) -> None:
        if not self.interactive:
            return
        if not self.active:
            sys.stdout.write("\x1b[?1049h\x1b[?25l")
            self.active = True
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write(
            _health_dashboard_text(
                payload, settings, active_component, runtime_metrics, metrics_error
            )
        )
        sys.stdout.flush()

    def close(self) -> None:
        if self.active:
            sys.stdout.write("\x1b[?25h\x1b[?1049l")
            sys.stdout.flush()
            self.active = False


def _install_shutdown_signal_handlers(
    stop: threading.Event,
) -> dict[signal.Signals, object]:
    """Route terminal and tmux shutdown through the normal cleanup path."""

    def request_stop(_signum=None, _frame=None) -> None:
        stop.set()

    return {
        signum: signal.signal(signum, request_stop)
        for signum in _SHUTDOWN_SIGNALS
    }


def _restore_signal_handlers(
    previous_handlers: dict[signal.Signals, object],
) -> None:
    for signum, handler in previous_handlers.items():
        signal.signal(signum, handler)


def run_sensor_gateway(settings: SensorGatewaySettings) -> None:
    if settings.health_print_interval_s < 0.0:
        raise ValueError("health_print_interval_s cannot be negative")

    logger = configure_file_logging("sensor_gateway")
    context = zmq.Context()
    core = SensorGatewayCore(
        slot_count=settings.slot_count,
        history_size=settings.history_size,
        frame_ttl_ms=settings.frame_ttl_ms,
        retired_ring_ttl_s=settings.retired_ring_ttl_s,
    )
    rpc: SensorGatewayRpcServer | None = None
    camera: CameraZmqIngress | None = None
    depth_anything: DepthAnythingZmqIngress | None = None
    cpp_state: CppStateZmqIngress | None = None
    ros: Ros2SensorIngress | None = None
    visualization: VisualizationZmqIngress | None = None
    metrics_socket: zmq.Socket | None = None
    metrics_windows = {
        component: create_metrics_window(names, settings.runtime_metrics_window_size)
        for component, names in RUNTIME_METRIC_SEGMENTS.items()
    }
    active_metrics_component: str | None = None
    metrics_error = ""
    previous_source_states: dict[str, str] = {}
    stop = threading.Event()
    health_display = _HealthDisplay()

    previous_signal_handlers = _install_shutdown_signal_handlers(stop)
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
        if settings.enable_runtime_metrics:
            metrics_socket = bind_pull(
                context, settings.runtime_metrics_bind_endpoint,
                high_water_mark=16, linger_ms=0,
            )
        if settings.enable_camera:
            camera = CameraZmqIngress(
                context,
                settings.camera_endpoint,
                core,
                expected_hz=settings.expected_hz["camera"],
            )
        if settings.enable_depth_anything:
            depth_anything = DepthAnythingZmqIngress(
                context,
                settings.depth_anything_endpoint,
                core,
                expected_hz=settings.expected_hz["depth_anything"],
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

        logger.info(
            "READY profile=%s rpc=%s metrics=%s",
            settings.profile_name,
            settings.rpc_bind_endpoint,
            settings.runtime_metrics_bind_endpoint,
        )

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
            if depth_anything is not None:
                try:
                    depth_anything.poll_once()
                except Exception as exc:
                    core.record_failure(
                        "source/depth_anything",
                        str(exc),
                        expected_hz=settings.expected_hz["depth_anything"],
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
            if metrics_socket is not None:
                try:
                    while sample := poll_metrics(metrics_socket, metrics_windows):
                        active_metrics_component, _received_ns = sample
                        if metrics_error:
                            logger.info("METRICS_RECOVERED")
                        metrics_error = ""
                except Exception as exc:
                    error = str(exc)
                    if error != metrics_error:
                        logger.warning("METRICS_INVALID error=%s", error)
                    metrics_error = error
            now = time.monotonic()
            if settings.health_print_interval_s > 0.0 and now >= next_health:
                current_metrics = (
                    metrics_snapshot(metrics_windows[active_metrics_component])
                    if active_metrics_component
                    else None
                )
                health_payload = core.health_payload()
                for source, state, prior, error in _source_state_changes(
                    health_payload, previous_source_states
                ):
                    name = source.removeprefix("source/")
                    if state == "down":
                        logger.error(
                            "SOURCE_DOWN source=%s previous=%s error=%s",
                            name,
                            prior,
                            error or "-",
                        )
                    elif state == "stale":
                        logger.warning(
                            "SOURCE_STALE source=%s previous=%s", name, prior
                        )
                    else:
                        logger.info(
                            "SOURCE_RECOVERED source=%s previous=%s", name, prior
                        )
                health_display.render(
                    health_payload,
                    settings,
                    active_metrics_component,
                    current_metrics,
                    metrics_error,
                )
                next_health = now + settings.health_print_interval_s
            stop.wait(max(0.0, period_s - (time.monotonic() - loop_started)))
    finally:
        try:
            health_display.close()
            if ros is not None:
                ros.close()
            if visualization is not None:
                visualization.close()
            if metrics_socket is not None:
                metrics_socket.close(linger=0)
            if cpp_state is not None:
                cpp_state.close()
            if camera is not None:
                camera.close()
            if depth_anything is not None:
                depth_anything.close()
            if rpc is not None:
                rpc.close()
            core.close()
            context.term()
            logger.info("STOPPED shared-memory segments released")
        finally:
            # Keep SIGHUP routed to request_stop until every shared-memory
            # name is gone; a repeated pane shutdown cannot interrupt cleanup.
            _restore_signal_handlers(previous_signal_handlers)


def main() -> None:
    args = build_argument_parser().parse_args()
    run_sensor_gateway(resolve_sensor_gateway_settings(args))


if __name__ == "__main__":
    main()
