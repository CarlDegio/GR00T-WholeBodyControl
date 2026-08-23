#!/usr/bin/env python3
"""Continuous NavDP orchestration entry point."""

from __future__ import annotations

import logging
import queue
import time
from typing import Mapping

import numpy as np

from gear_sonic.runtime.inference_service import InferenceServiceContext
from gear_sonic.runtime.telemetry import NAVDP_TIMING_SEGMENTS
from gear_sonic.runtime.zmq_sockets import bind_publisher, connect_subscriber
from gear_sonic.utils.inference.navdp.control import (
    AsyncMpcSolver,
)
from gear_sonic.utils.inference.navdp.gateway import (
    NavDPPlannerConfig,
    NavDPSensorGatewayIngress,
    _control_freshness_snapshot,
    _navdp_request,
    _SharedSensors,
    load_navdp_planner_config,
)
from gear_sonic.utils.inference.navdp.navigation import (
    STATUS_TYPE,
    HeadingGoalController,
    Pose2D,
)
from gear_sonic.utils.inference.navdp.runtime import (
    _NavDPRuntimeState,
    _NavDPSensorCycle,
    _compute_control,
    _consume_inference_results,
    _drain_navigation_commands,
    _drain_orientation_messages,
    _fresh_sonic_yaw,
    _refresh_gateway_camera,
    _schedule_navigation_inference,
    _snapshot_sensor_cycle,
)
from gear_sonic.utils.inference.navdp.visualization import (
    actor_ray_from_points,
    install_shutdown_signal_handlers,
    render_actor_ray_panel,
    render_slam_world_panel,
)
from gear_sonic.utils.planner_control import build_planner_velocity_message

LOGGER = logging.getLogger("sonic.navdp")


def main(config: NavDPPlannerConfig) -> None:
    service = InferenceServiceContext("navdp", config)
    profile = service.profile
    command_endpoint = profile.endpoint_uri("navigation_command")
    output_endpoint = profile.endpoint_uri("navdp_velocity")
    navdp_endpoint = profile.endpoint_uri("xnavdp_http")
    sensor_endpoint = profile.endpoint_uri("sensor_gateway_metadata")
    orientation_endpoint = profile.endpoint_uri("orientation_telemetry")
    import zmq

    from gear_sonic.runtime.gateway.visualization import (
        NAVDP_ACTOR_RAY_STREAM,
        NAVDP_SLAM_2D_STREAM,
        VisualizationPublisher,
    )

    install_shutdown_signal_handlers()

    context = zmq.Context.instance()
    commands = connect_subscriber(context, command_endpoint)
    orientation = connect_subscriber(
        context,
        orientation_endpoint,
        conflate=True,
        high_water_mark=1,
    )
    status = bind_publisher(
        context,
        profile.endpoint_uri("navigation_status"),
    )
    output = bind_publisher(context, output_endpoint)
    visualization_publisher = (
        VisualizationPublisher(
            profile.endpoint_uri("sensor_gateway_visualization_ingress")
        )
    )
    sensors = _SharedSensors()
    gateway = NavDPSensorGatewayIngress(
        sensor_endpoint,
        sensors,
        poll_hz=config.sensor_gateway_poll_hz,
        request_timeout_ms=config.sensor_gateway_request_timeout_ms,
        max_age_ms=config.sensor_gateway_max_age_ms,
        max_skew_ms=config.sensor_gateway_max_skew_ms,
        rgb_stream=config.rgb_stream,
        depth_stream=config.depth_stream,
    )
    gateway.start()
    LOGGER.info(
        "SensorGateway camera rgb=%s depth=%s goal_tolerance_m=%.3f",
        config.rgb_stream,
        config.depth_stream,
        config.goal_tolerance_m,
    )
    state = _NavDPRuntimeState()
    inference_result: queue.Queue[
        tuple[int, int, np.ndarray | None, Pose2D | None, str | None, float]
    ] = queue.Queue(maxsize=1)
    mpc_solver = AsyncMpcSolver()
    heading_controller = HeadingGoalController(
        angular_speed_rad_s=config.heading_angular_speed_rad_s,
        fine_angular_speed_rad_s=config.heading_fine_angular_speed_rad_s,
        slowdown_angle_rad=config.heading_slowdown_angle_rad,
        tolerance_rad=config.heading_goal_tolerance_rad,
    )

    def report_event(level: int, code: str, message: str, **fields: object) -> None:
        service.event(level, code, message, **fields)

    def send_metrics(values: Mapping[str, float], *, activate: bool = False) -> None:
        service.publish_metrics(
            values,
            allowed_names=NAVDP_TIMING_SEGMENTS,
            activate=activate,
        )

    def send_status(status_state: str, reason: str) -> None:
        status.send_json(
            {
                "type": STATUS_TYPE,
                "version": 1,
                "generation": state.generation,
                "skill_id": state.skill_id,
                "segment_id": state.segment_id,
                "state": status_state,
                "reason": reason,
            }
        )
        LOGGER.log(
            logging.ERROR if status_state == "failed" else logging.INFO,
            "navigation status=%s generation=%d segment=%d reason=%s",
            status_state,
            state.generation,
            state.segment_id,
            reason,
        )

    def infer(
        request_generation: int,
        request_segment_id: int,
        rgb: np.ndarray,
        depth: np.ndarray,
        goal: tuple[float, float],
        inference_pose: Pose2D | None,
    ) -> None:
        started = time.monotonic()
        try:
            result = _navdp_request(
                navdp_endpoint,
                rgb,
                depth,
                goal,
                pose=inference_pose,
                timeout=config.request_timeout_s,
            )
            item = (
                request_generation,
                request_segment_id,
                result,
                inference_pose,
                None,
                time.monotonic() - started,
            )
        except Exception as exc:
            item = (
                request_generation,
                request_segment_id,
                None,
                inference_pose,
                str(exc),
                time.monotonic() - started,
            )
        try:
            inference_result.put_nowait(item)
        except queue.Full:
            pass
        state.inference_busy = False

    def publish_cycle(
        velocity: tuple[float, float, float],
        *,
        zero_action_aborted: bool,
        emit_heading_target: bool,
        terminal_heading_status: tuple[str, str] | None,
        sensor_cycle: _NavDPSensorCycle,
        loop_started: float,
    ) -> None:
        pose, pose_time, points = _control_freshness_snapshot(sensors)
        now = time.monotonic()
        stale_reason = None
        if (
            state.mode == "nav_goal"
            and now - pose_time > config.odometry_timeout_s
        ):
            stale_reason = "odometry_timeout"
        elif state.mode == "nav_goal" and (
            not len(state.trajectory)
            or now - state.trajectory_time > config.trajectory_timeout_s
        ):
            stale_reason = "trajectory_stale"
        if stale_reason:
            velocity = (0.0, 0.0, 0.0)
        if (
            state.mode == "nav_goal"
            and stale_reason != state.last_stale_reason
        ):
            if stale_reason:
                report_event(
                    logging.WARNING,
                    "NAVIGATION_BLOCKED",
                    "NavDP output blocked by stale input",
                    reason=stale_reason,
                )
            elif state.last_stale_reason:
                report_event(
                    logging.INFO,
                    "NAVIGATION_RESUMED",
                    "NavDP input freshness recovered",
                    previous_reason=state.last_stale_reason,
                )
            state.last_stale_reason = stale_reason
        current_rays = actor_ray_from_points(points)
        heading_kwargs = (
            {
                "heading_target_rad": state.heading_target_yaw,
                "heading_reference_rad": state.heading_reference_yaw,
            }
            if (
                state.mode in {"nav_goal", "heading_goal"}
                or emit_heading_target
            )
            and state.heading_target_yaw is not None
            and state.heading_reference_yaw is not None
            else {}
        )
        output.send_string(
            build_planner_velocity_message(
                generation=state.generation,
                skill_id=state.skill_id,
                segment_id=state.segment_id,
                source="navdp",
                velocity=velocity,
                **heading_kwargs,
            )
        )
        if terminal_heading_status is not None:
            send_status(*terminal_heading_status)
        if zero_action_aborted:
            stop_reason = "navdp_zero_action"
            send_status("stopped", stop_reason)
            state.mode = "stop"
            state.trajectory = np.empty((0, 2), dtype=np.float32)
            state.mpc_reference = np.empty((0, 2), dtype=np.float64)
            state.mpc_reference_version += 1
            state.mpc_linear_velocity = 0.0
            state.mpc_angular_velocity = 0.0
            state.mpc_result_time = 0.0
        actor_ray_panel = render_actor_ray_panel(
            current_rays,
            trajectory=state.trajectory,
            velocity=velocity,
        )
        slam_2d_panel = render_slam_world_panel(
            sensor_cycle.slam_map_xy,
            pose=pose,
            world_goal=state.world_goal,
            trajectory_world=None,
            robot_history=sensor_cycle.robot_history,
        )
        visualization_publisher.publish(NAVDP_ACTOR_RAY_STREAM, actor_ray_panel)
        visualization_publisher.publish(NAVDP_SLAM_2D_STREAM, slam_2d_panel)
        if state.mode == "nav_goal":
            now = time.monotonic()
            timing_ms = {"control_loop": (now - loop_started) * 1000.0}
            if pose_time > 0.0:
                timing_ms["odometry_age"] = (
                    max(0.0, now - pose_time) * 1000.0
                )
            if state.trajectory_time > 0.0:
                timing_ms["trajectory_age"] = (
                    max(0.0, now - state.trajectory_time) * 1000.0
                )
            send_metrics(timing_ms)

    LOGGER.info(
        "command=%s output=%s sensors=%s orientation=%s",
        command_endpoint,
        output_endpoint,
        sensor_endpoint,
        orientation_endpoint,
    )
    period = 1.0 / config.control_hz
    try:
        while True:
            loop_started = time.monotonic()
            _drain_orientation_messages(orientation, state, report_event)
            _drain_navigation_commands(
                commands,
                state,
                sensors=sensors,
                config=config,
                heading_controller=heading_controller,
                send_metrics=send_metrics,
                send_status=send_status,
            )

            _refresh_gateway_camera(
                state,
                gateway=gateway,
                navdp_endpoint=navdp_endpoint,
                config=config,
                report_event=report_event,
            )
            _consume_inference_results(
                state,
                inference_result,
                send_metrics=send_metrics,
                send_status=send_status,
                report_event=report_event,
            )

            now = time.monotonic()
            sensor_cycle = _snapshot_sensor_cycle(
                sensors,
                now=now,
                slam_max_age_s=(config.sensor_gateway_max_age_ms * 1.0e-3),
            )
            _schedule_navigation_inference(
                state,
                sensor_cycle,
                config=config,
                infer=infer,
                send_status=send_status,
            )

            (
                velocity,
                zero_action_aborted,
                emit_heading_target,
                terminal_heading_status,
            ) = _compute_control(
                state,
                now=now,
                pose=sensor_cycle.pose,
                config=config,
                mpc_solver=mpc_solver,
                heading_controller=heading_controller,
                send_metrics=send_metrics,
                report_event=report_event,
            )
            publish_cycle(
                velocity,
                zero_action_aborted=zero_action_aborted,
                emit_heading_target=emit_heading_target,
                terminal_heading_status=terminal_heading_status,
                sensor_cycle=sensor_cycle,
                loop_started=loop_started,
            )
            delay = period - (time.monotonic() - loop_started)
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        pass
    finally:
        mpc_solver.close()
        for _ in range(3):
            output.send_string(
                build_planner_velocity_message(
                    generation=state.generation,
                    skill_id=state.skill_id,
                    segment_id=state.segment_id,
                    source="navdp",
                    velocity=(0.0, 0.0, 0.0),
                )
            )
        gateway.close()
        commands.close(0)
        orientation.close(0)
        status.close(0)
        output.close(0)
        service.close()
        visualization_publisher.close()


if __name__ == "__main__":
    from gear_sonic.runtime.profile import parse_component_config

    try:
        main(
            parse_component_config(
                NavDPPlannerConfig,
                "navdp",
                ignored_fields=("root", "checkpoint"),
            )
        )
    except Exception:
        LOGGER.exception("fatal NavDP planner failure")
        raise
