#!/usr/bin/env python3
"""Execute all planner velocity sources through one shared safety boundary."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import signal
import threading
import time
from typing import Any, Callable, Mapping

import numpy as np
import zmq

from gear_sonic.runtime.gateway.sensor_client import SensorGatewayClient
from gear_sonic.runtime.gateway.snapshot import SnapshotRequest
from gear_sonic.runtime.profile import load_component_config, load_runtime_profile
from gear_sonic.runtime.protocol import decode_cpp_state_array
from gear_sonic.runtime.telemetry import (
    build_event,
    configure_file_logging,
    emit_event,
    open_telemetry_publisher,
)
from gear_sonic.runtime.zmq_sockets import bind_publisher, connect_subscriber
from gear_sonic.utils.planner_control import (
    PlannerVelocityExecutorCore,
    SafetySnapshot,
    build_navigation_runtime_status_message,
    decode_navigation_message,
    decode_planner_velocity_message,
)
from gear_sonic.utils.teleop.sonic_orientation_telemetry import (
    OrientationTracker,
    encode_orientation_telemetry,
)

LOGGER = logging.getLogger("sonic.planner_executor")


@dataclass
class PlannerVelocityExecutorConfig:
    control_hz: float
    manual_velocity_timeout_s: float
    navdp_velocity_timeout_s: float
    radar_timeout_s: float
    sensor_gateway_poll_hz: float
    sensor_gateway_request_timeout_ms: int
    sensor_gateway_max_age_ms: float
    profile: str = ""
    overlay: tuple[str, ...] = ()


def load_planner_velocity_executor_config(
    profile: str = "", overlays: tuple[str, ...] = ()
) -> PlannerVelocityExecutorConfig:
    return load_component_config(
        PlannerVelocityExecutorConfig,
        "planner_executor",
        profile or None,
        overlays=overlays,
    )


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
        lidar_client: SensorGatewayClient | None = None,
        auxiliary_client: SensorGatewayClient | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if poll_hz <= 0.0:
            raise ValueError("safety sensor poll_hz must be positive")
        self.poll_hz = float(poll_hz)
        self.max_age_ms = float(max_age_ms)
        self.include_robot_state = bool(include_robot_state)
        if client is not None and (
            lidar_client is not None or auxiliary_client is not None
        ):
            raise ValueError(
                "client cannot be combined with dedicated safety clients"
            )
        self._owned_clients: list[SensorGatewayClient] = []
        if client is not None:
            # Preserve the injectable single-client path used by synchronous
            # tests. Production uses independent sockets so a slow RGB-D/state
            # materialization cannot delay the safety-critical LiDAR refresh.
            self._lidar_client = client
            self._auxiliary_client = client
        else:
            if lidar_client is None:
                lidar_client = SensorGatewayClient(
                    endpoint, request_timeout_ms=int(request_timeout_ms)
                )
                self._owned_clients.append(lidar_client)
            if auxiliary_client is None:
                auxiliary_client = SensorGatewayClient(
                    endpoint, request_timeout_ms=int(request_timeout_ms)
                )
                self._owned_clients.append(auxiliary_client)
            self._lidar_client = lidar_client
            self._auxiliary_client = auxiliary_client
        # Keep the historical attribute for callers that inspect the monitor.
        self.client = self._auxiliary_client
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._lidar_thread = threading.Thread(
            target=self._run_lidar,
            name="planner-safety-lidar",
            daemon=True,
        )
        self._auxiliary_thread = threading.Thread(
            target=self._run_auxiliary,
            name="planner-safety-depth-state",
            daemon=True,
        )
        self._snapshot = SafetySnapshot()
        self._orientation = RobotOrientationSnapshot()
        self._last_error = ""
        self._last_error_time = 0.0

    def _request(self, stream: str):
        return self._auxiliary_client.read_snapshot(
            SnapshotRequest(
                streams=(stream,),
                max_age_ms=self.max_age_ms,
                max_skew_ms=0.0,
            ),
            retries=0,
        )

    def _poll_lidar(self) -> None:
        lidar = self._lidar_client.request_snapshot(
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
            LOGGER.warning("SensorGateway waiting: %s", message)
            self._last_error = message
            self._last_error_time = now

    def _run_pollers(self, pollers) -> None:
        period = 1.0 / self.poll_hz
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

    def _run_lidar(self) -> None:
        self._run_pollers((self._poll_lidar,))

    def _run_auxiliary(self) -> None:
        pollers = [self._poll_depth]
        if self.include_robot_state:
            pollers.append(self._poll_robot_state)
        self._run_pollers(pollers)

    def start(self) -> None:
        self._lidar_thread.start()
        self._auxiliary_thread.start()

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
        for thread in (self._lidar_thread, self._auxiliary_thread):
            if thread.is_alive():
                thread.join(timeout=1.0)
        for owned_client in self._owned_clients:
            owned_client.close()


def main(
    config: PlannerVelocityExecutorConfig, *, publish_orientation: bool = False
) -> None:
    configure_file_logging("planner_executor")
    profile = load_runtime_profile(config.profile or None, overlays=config.overlay)
    command_endpoint = profile.endpoint_uri("navigation_command")
    navdp_endpoint = profile.endpoint_uri("navdp_velocity")
    output_endpoint = profile.endpoint_uri("planner_relay")
    status_endpoint = profile.endpoint_uri("navigation_runtime_status")
    orientation_endpoint = profile.endpoint_uri("orientation_telemetry")
    context = zmq.Context.instance()
    navigation = connect_subscriber(context, command_endpoint, linger_ms=0)
    navdp = connect_subscriber(context, navdp_endpoint, linger_ms=0)
    output = bind_publisher(context, output_endpoint, linger_ms=0)
    runtime_status = bind_publisher(
        context, status_endpoint, high_water_mark=1, linger_ms=0,
    )
    event_socket = open_telemetry_publisher(
        profile.endpoint_uri("runtime_event_ingress")
    )

    def report_event(
        level: int, code: str, message: str, **fields: object
    ) -> None:
        emit_event(
            build_event("planner_executor", level, code, message, **fields),
            socket=event_socket,
            logger=LOGGER,
        )

    orientation_output = None
    orientation_tracker = None
    if publish_orientation:
        orientation_output = bind_publisher(
            context, orientation_endpoint, linger_ms=0,
        )
        orientation_tracker = OrientationTracker()
    sensors = PlannerSafetySensorMonitor(
        profile.endpoint_uri("sensor_gateway_metadata"),
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
    report_event(
        logging.INFO,
        "READY",
        "Planner executor ready",
        command=command_endpoint,
        navdp=navdp_endpoint,
        output=output_endpoint,
        status=status_endpoint,
    )
    if publish_orientation:
        LOGGER.info(
            "orientation telemetry=%s",
            orientation_endpoint,
        )
    period = 1.0 / config.control_hz
    last_reason = ""
    last_orientation_error = ""
    last_owner: tuple[int, int, str, str] | None = None
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
                    report_event(
                        logging.WARNING,
                        "INVALID_NAVIGATION_COMMAND",
                        "rejected navigation command",
                        error=str(exc),
                    )
            while navdp.poll(0):
                try:
                    core.accept_planner_velocity(
                        decode_planner_velocity_message(navdp.recv()),
                        now=time.monotonic(),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    report_event(
                        logging.WARNING,
                        "INVALID_PLANNER_VELOCITY",
                        "rejected planner velocity",
                        error=str(exc),
                    )
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
                        last_orientation_error = ""
                    except ValueError as exc:
                        message = str(exc)
                        if message != last_orientation_error:
                            report_event(
                                logging.WARNING,
                                "INVALID_ORIENTATION",
                                "ignored robot orientation",
                                error=message,
                            )
                            last_orientation_error = message
            decision = core.decide(now=now, safety=sensors.snapshot())
            owner = (
                decision.generation,
                decision.skill_id,
                decision.segment_id,
                decision.source,
                core.mode,
            )
            if owner != last_owner:
                LOGGER.info(
                    "owner generation=%d segment=%d source=%s mode=%s",
                    decision.generation,
                    decision.segment_id,
                    decision.source,
                    core.mode,
                )
                last_owner = owner
            output.send(decision.message)
            try:
                runtime_status.send_string(
                    build_navigation_runtime_status_message(
                        generation=decision.generation,
                        skill_id=decision.skill_id,
                        segment_id=decision.segment_id,
                        mode=core.mode,
                        source=decision.source,
                        requested_velocity=decision.requested_velocity,
                        velocity=decision.velocity,
                        reason=decision.reason,
                    ),
                    flags=zmq.DONTWAIT,
                )
            except zmq.Again:
                pass
            if orientation_output is not None and orientation_tracker is not None:
                orientation_output.send_string(
                    encode_orientation_telemetry(
                        orientation_tracker.sample(now, core.sonic.heading)
                    )
                )
            if decision.reason != last_reason:
                fields = {
                    "generation": decision.generation,
                    "segment_id": decision.segment_id,
                    "source": decision.source,
                    "requested_velocity": decision.requested_velocity,
                    "velocity": decision.velocity,
                }
                if decision.reason not in {"clear", "stopped"}:
                    report_event(
                        logging.WARNING,
                        "VELOCITY_BLOCKED",
                        "planner velocity blocked",
                        reason=decision.reason,
                        **fields,
                    )
                elif last_reason not in {"", "clear", "stopped"}:
                    report_event(
                        logging.INFO,
                        "VELOCITY_RECOVERED",
                        "planner velocity safety recovered",
                        **fields,
                    )
                else:
                    LOGGER.info(
                        "safety generation=%d source=%s reason=%s velocity=%s",
                        decision.generation,
                        decision.source,
                        decision.reason,
                        decision.velocity,
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
        runtime_status.close(0)
        if orientation_output is not None:
            orientation_output.close(0)
        report_event(logging.INFO, "STOPPED", "Planner executor stopped")
        event_socket.close(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="")
    parser.add_argument("--overlay", action="append", default=[])
    parser.add_argument("--publish-orientation", action="store_true")
    args = parser.parse_args()
    main(
        load_planner_velocity_executor_config(args.profile, tuple(args.overlay)),
        publish_orientation=args.publish_orientation,
    )
