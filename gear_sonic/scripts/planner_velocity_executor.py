#!/usr/bin/env python3
"""Execute all planner velocity sources through one shared safety boundary."""

from __future__ import annotations

from dataclasses import dataclass
import signal
import threading
import time
from typing import Any, Callable, Mapping

import numpy as np
import zmq

from gear_sonic.planner_control import (
    PlannerVelocityExecutorCore,
    SafetySnapshot,
    decode_navigation_message,
    decode_planner_velocity_message,
)
from gear_sonic.runtime.client import SensorGatewayClient
from gear_sonic.runtime.config import load_runtime_profile
from gear_sonic.runtime.cpp_state import decode_cpp_state_array
from gear_sonic.runtime.snapshot import SnapshotRequest
from gear_sonic.utils.teleop.sonic_orientation_telemetry import (
    OrientationTracker,
    encode_orientation_telemetry,
)

_PROFILE = load_runtime_profile()
_DEFAULTS = _PROFILE.component("planner_executor")


def _bind_endpoint(name: str) -> str:
    return f"tcp://*:{_PROFILE.endpoint(name).port}"


@dataclass
class PlannerVelocityExecutorConfig:
    command_endpoint: str = _PROFILE.endpoint_uri("navigation_command")
    navdp_velocity_endpoint: str = _PROFILE.endpoint_uri("navdp_velocity")
    output_endpoint: str = _bind_endpoint("planner_relay")
    sensor_gateway_endpoint: str = _PROFILE.endpoint_uri("sensor_gateway_metadata")
    control_hz: float = float(_DEFAULTS["control_hz"])
    manual_velocity_timeout_s: float = float(
        _DEFAULTS["manual_velocity_timeout_s"]
    )
    navdp_velocity_timeout_s: float = float(_DEFAULTS["navdp_velocity_timeout_s"])
    radar_timeout_s: float = float(_DEFAULTS["radar_timeout_s"])
    sensor_gateway_poll_hz: float = float(_DEFAULTS["sensor_gateway_poll_hz"])
    sensor_gateway_request_timeout_ms: int = int(
        _DEFAULTS["sensor_gateway_request_timeout_ms"]
    )
    sensor_gateway_max_age_ms: float = float(
        _DEFAULTS["sensor_gateway_max_age_ms"]
    )
    orientation_output_endpoint: str = ""


@dataclass(frozen=True)
class RobotOrientationSnapshot:
    sequence: int = -1
    received_at_s: float = 0.0
    state: Mapping[str, Any] | None = None


class PlannerSafetySensorMonitor:
    """Cache only the last-mile safety streams needed by the common executor."""

    LIDAR_STREAM = "ros/livox_lidar_xyz"
    DEPTH_STREAM = "camera/ego_view_depth"
    ROBOT_STATE_STREAM = "cpp/state_msgpack"

    def __init__(
        self,
        endpoint: str,
        *,
        poll_hz: float,
        request_timeout_ms: int,
        max_age_ms: float,
        include_robot_state: bool = False,
        client: SensorGatewayClient | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if poll_hz <= 0.0:
            raise ValueError("safety sensor poll_hz must be positive")
        self.poll_hz = float(poll_hz)
        self.max_age_ms = float(max_age_ms)
        self.include_robot_state = bool(include_robot_state)
        self.client = client or SensorGatewayClient(
            endpoint, request_timeout_ms=int(request_timeout_ms)
        )
        self._owns_client = client is None
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="planner-safety-sensors",
            daemon=True,
        )
        self._snapshot = SafetySnapshot()
        self._orientation = RobotOrientationSnapshot()
        self._last_error = ""
        self._last_error_time = 0.0

    def _request(self, stream: str):
        return self.client.read_snapshot(
            SnapshotRequest(
                streams=(stream,),
                max_age_ms=self.max_age_ms,
                max_skew_ms=0.0,
            ),
            retries=0,
        )

    def _poll_lidar(self) -> None:
        lidar = self.client.request_snapshot(
            SnapshotRequest(
                streams=(self.LIDAR_STREAM,),
                max_age_ms=self.max_age_ms,
                max_skew_ms=0.0,
            )
        )
        if not lidar.complete or self.LIDAR_STREAM not in lidar.frames:
            raise ValueError("SensorGateway omitted the LiDAR safety stream")
        lidar_frame = lidar.frames[self.LIDAR_STREAM]
        radar_time = float(lidar_frame.metadata.timestamp_ns) * 1.0e-9
        with self._lock:
            self._snapshot = SafetySnapshot(
                radar_time,
                self._snapshot.depth_m,
            )

    def _poll_depth(self) -> None:
        depth = self._request(self.DEPTH_STREAM)
        depth_raw = np.asarray(depth.arrays[self.DEPTH_STREAM])
        if depth_raw.ndim != 2:
            raise ValueError("planner safety depth must be an HxW image")
        frame = depth.snapshot.frames[self.DEPTH_STREAM]
        info = dict(frame.attributes.get("camera_info", {}))
        scale = float(info.get("depth_scale_m", 0.001))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("planner safety depth scale must be positive")
        depth_m = depth_raw.astype(np.float32) * scale
        with self._lock:
            self._snapshot = SafetySnapshot(
                self._snapshot.radar_timestamp_s,
                depth_m,
            )

    def _poll_robot_state(self) -> None:
        snapshot = self._request(self.ROBOT_STATE_STREAM)
        frame = snapshot.snapshot.frames[self.ROBOT_STATE_STREAM]
        sequence = int(frame.metadata.sequence)
        with self._lock:
            if sequence == self._orientation.sequence:
                return
        state = decode_cpp_state_array(snapshot.arrays[self.ROBOT_STATE_STREAM])
        try:
            base_quat = tuple(state["base_quat"])
        except (KeyError, TypeError) as exc:
            raise ValueError("g1_debug state is missing base_quat") from exc
        with self._lock:
            self._orientation = RobotOrientationSnapshot(
                sequence=sequence,
                received_at_s=float(frame.metadata.timestamp_ns) * 1.0e-9,
                state={"base_quat": base_quat},
            )

    def poll_once(self) -> None:
        self._poll_lidar()
        self._poll_depth()
        if self.include_robot_state:
            self._poll_robot_state()

    def _report(self, exc: Exception) -> None:
        message = str(exc)
        now = self._monotonic()
        if message != self._last_error or now - self._last_error_time >= 2.0:
            print(f"[PlannerExecutor] SensorGateway waiting: {message}", flush=True)
            self._last_error = message
            self._last_error_time = now

    def _run(self) -> None:
        period = 1.0 / self.poll_hz
        pollers = [self._poll_lidar, self._poll_depth]
        if self.include_robot_state:
            pollers.append(self._poll_robot_state)
        while not self._stop.is_set():
            started = self._monotonic()
            for poll in pollers:
                if self._stop.is_set():
                    break
                try:
                    poll()
                except Exception as exc:
                    self._report(exc)
            self._stop.wait(max(0.0, period - (self._monotonic() - started)))

    def start(self) -> None:
        self._thread.start()

    def snapshot(self) -> SafetySnapshot:
        with self._lock:
            return SafetySnapshot(
                radar_timestamp_s=self._snapshot.radar_timestamp_s,
                depth_m=(
                    None
                    if self._snapshot.depth_m is None
                    else self._snapshot.depth_m.copy()
                ),
            )

    def orientation_snapshot(self) -> RobotOrientationSnapshot:
        with self._lock:
            value = self._orientation
            return RobotOrientationSnapshot(
                sequence=value.sequence,
                received_at_s=value.received_at_s,
                state=None if value.state is None else dict(value.state),
            )

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._owns_client:
            self.client.close()


def main(config: PlannerVelocityExecutorConfig) -> None:
    context = zmq.Context.instance()
    navigation = context.socket(zmq.SUB)
    navdp = context.socket(zmq.SUB)
    output = context.socket(zmq.PUB)
    for socket in (navigation, navdp, output):
        socket.setsockopt(zmq.LINGER, 0)
    navigation.setsockopt_string(zmq.SUBSCRIBE, "")
    navdp.setsockopt_string(zmq.SUBSCRIBE, "")
    navigation.connect(config.command_endpoint)
    navdp.connect(config.navdp_velocity_endpoint)
    output.bind(config.output_endpoint)
    orientation_output = None
    orientation_tracker = None
    if config.orientation_output_endpoint:
        orientation_output = context.socket(zmq.PUB)
        orientation_output.setsockopt(zmq.LINGER, 0)
        orientation_output.bind(config.orientation_output_endpoint)
        orientation_tracker = OrientationTracker()
    sensors = PlannerSafetySensorMonitor(
        config.sensor_gateway_endpoint,
        poll_hz=config.sensor_gateway_poll_hz,
        request_timeout_ms=config.sensor_gateway_request_timeout_ms,
        max_age_ms=config.sensor_gateway_max_age_ms,
        include_robot_state=orientation_tracker is not None,
    )
    sensors.start()
    core = PlannerVelocityExecutorCore(
        control_hz=config.control_hz,
        manual_timeout_s=config.manual_velocity_timeout_s,
        navdp_timeout_s=config.navdp_velocity_timeout_s,
        radar_timeout_s=config.radar_timeout_s,
    )
    running = True

    def stop(_signum: int, _frame: Any) -> None:
        nonlocal running
        running = False

    for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop)
    print(
        f"[PlannerExecutor] command={config.command_endpoint} "
        f"navdp={config.navdp_velocity_endpoint} output={config.output_endpoint}"
    )
    if config.orientation_output_endpoint:
        print(
            "[PlannerExecutor] orientation telemetry="
            f"{config.orientation_output_endpoint}"
        )
    period = 1.0 / config.control_hz
    last_reason = ""
    last_orientation_sequence = -1
    try:
        while running:
            started = time.monotonic()
            while navigation.poll(0):
                try:
                    core.accept_navigation(
                        decode_navigation_message(navigation.recv()),
                        now=time.monotonic(),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    print(f"[PlannerExecutor] rejected navigation command: {exc}")
            while navdp.poll(0):
                try:
                    core.accept_planner_velocity(
                        decode_planner_velocity_message(navdp.recv()),
                        now=time.monotonic(),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    print(f"[PlannerExecutor] rejected planner velocity: {exc}")
            now = time.monotonic()
            if orientation_tracker is not None:
                orientation = sensors.orientation_snapshot()
                if (
                    orientation.state is not None
                    and orientation.sequence != last_orientation_sequence
                ):
                    last_orientation_sequence = orientation.sequence
                    try:
                        orientation_tracker.update_state(
                            orientation.state,
                            received_at_monotonic_s=orientation.received_at_s,
                            heading_setpoint_rad=core.sonic.heading,
                        )
                    except ValueError as exc:
                        print(
                            "[PlannerExecutor] ignored robot orientation: "
                            f"{exc}",
                            flush=True,
                        )
            decision = core.decide(now=now, safety=sensors.snapshot())
            output.send(decision.message)
            if orientation_output is not None and orientation_tracker is not None:
                orientation_output.send_string(
                    encode_orientation_telemetry(
                        orientation_tracker.sample(now, core.sonic.heading)
                    )
                )
            if decision.reason != last_reason:
                print(
                    f"[PlannerExecutor] generation={decision.generation} "
                    f"source={decision.source} safety={decision.reason} "
                    f"velocity={decision.velocity}",
                    flush=True,
                )
                last_reason = decision.reason
            delay = period - (time.monotonic() - started)
            if delay > 0.0:
                time.sleep(delay)
    finally:
        for _ in range(3):
            output.send(core.sonic.message((0.0, 0.0, 0.0)))
        sensors.close()
        navigation.close(0)
        navdp.close(0)
        output.close(0)
        if orientation_output is not None:
            orientation_output.close(0)


if __name__ == "__main__":
    import tyro

    main(tyro.cli(PlannerVelocityExecutorConfig))
