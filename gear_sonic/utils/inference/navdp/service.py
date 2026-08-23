#!/usr/bin/env python3
"""Continuous NavDP orchestration entry point."""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from typing import Any, Mapping

import numpy as np

from gear_sonic.runtime.inference_service import InferenceServiceContext
from gear_sonic.runtime.telemetry import NAVDP_TIMING_SEGMENTS
from gear_sonic.runtime.zmq_sockets import bind_publisher, connect_subscriber
from gear_sonic.utils.inference.navdp.control import (
    AsyncMpcSolver,
    MpcSolveRequest,
    fastlio_heading_target_from_mpc,
    fresh_mpc_control,
    prepare_internnav_world_reference,
    should_abort_nav_for_zero_action,
    xnavdp_control_to_body_velocity,
)
from gear_sonic.utils.inference.navdp.gateway import (
    NavDPPlannerConfig,
    NavDPSensorGatewayIngress,
    _control_freshness_snapshot,
    _navdp_request,
    _reset_navdp,
    _SharedSensors,
    load_navdp_planner_config,
)
from gear_sonic.utils.inference.navdp.navigation import (
    STATUS_TYPE,
    HeadingGoalController,
    Pose2D,
    base_goal_to_world,
    closest_timestamped_pose,
    decode_navigation_message,
    local_goal_from_world,
)
from gear_sonic.utils.inference.navdp.visualization import (
    actor_ray_from_points,
    install_shutdown_signal_handlers,
    render_actor_ray_panel,
    render_slam_world_panel,
)
from gear_sonic.utils.planner_control import build_planner_velocity_message
from gear_sonic.utils.teleop.sonic_orientation_telemetry import (
    OrientationTelemetrySample,
    decode_orientation_telemetry,
)

LOGGER = logging.getLogger("sonic.navdp")


def _fresh_sonic_yaw(
    sample: OrientationTelemetrySample | None,
    *,
    now_s: float,
    timeout_s: float,
) -> float | None:
    """Return the executor's measured SONIC yaw only while its state is fresh."""
    if (
        sample is None
        or sample.actual_yaw_rad is None
        or sample.state_age_s is None
    ):
        return None
    total_age_s = max(0.0, float(now_s) - sample.emitted_at_monotonic_s)
    total_age_s += sample.state_age_s
    if total_age_s > float(timeout_s):
        return None
    return sample.actual_yaw_rad


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
    generation = 0
    skill_id = 0
    segment_id = 0
    mode = "stop"
    world_goal: tuple[float, float] | None = None
    current_local_goal = (0.0, 0.0)
    trajectory = np.empty((0, 2), dtype=np.float32)
    trajectory_time = 0.0
    invalid_count = 0
    latest_rgb: np.ndarray | None = None
    latest_depth: np.ndarray | None = None
    camera_info: Mapping[str, Any] | None = None
    navdp_initialized = False
    inference_busy = False
    inference_result: queue.Queue[
        tuple[int, int, np.ndarray | None, Pose2D | None, str | None, float]
    ] = queue.Queue(maxsize=1)
    heading_reference_yaw: float | None = None
    heading_target_yaw: float | None = None
    mpc_solver = AsyncMpcSolver()
    mpc_reference = np.empty((0, 2), dtype=np.float64)
    mpc_reference_version = 0
    mpc_linear_velocity = 0.0
    mpc_angular_velocity = 0.0
    mpc_result_time = 0.0
    next_mpc_update = 0.0
    latest_camera_timestamp = 0.0
    server_error = ""
    mpc_error = ""
    last_stale_reason: str | None = None
    heading_controller = HeadingGoalController(
        angular_speed_rad_s=config.heading_angular_speed_rad_s,
        fine_angular_speed_rad_s=config.heading_fine_angular_speed_rad_s,
        slowdown_angle_rad=config.heading_slowdown_angle_rad,
        tolerance_rad=config.heading_goal_tolerance_rad,
    )
    heading_result = None
    latest_orientation: OrientationTelemetrySample | None = None
    orientation_error = ""

    def report_event(level: int, code: str, message: str, **fields: object) -> None:
        service.event(level, code, message, **fields)

    def send_metrics(values: Mapping[str, float], *, activate: bool = False) -> None:
        service.publish_metrics(
            values,
            allowed_names=NAVDP_TIMING_SEGMENTS,
            activate=activate,
        )

    def send_status(state: str, reason: str) -> None:
        status.send_json(
            {
                "type": STATUS_TYPE,
                "version": 1,
                "generation": generation,
                "skill_id": skill_id,
                "segment_id": segment_id,
                "state": state,
                "reason": reason,
            }
        )
        LOGGER.log(
            logging.ERROR if state == "failed" else logging.INFO,
            "navigation status=%s generation=%d segment=%d reason=%s",
            state,
            generation,
            segment_id,
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
        nonlocal inference_busy
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
        inference_busy = False

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
            while orientation.poll(0):
                try:
                    sample = decode_orientation_telemetry(orientation.recv())
                except ValueError as exc:
                    message = str(exc)
                    if message != orientation_error:
                        report_event(
                            logging.WARNING,
                            "SONIC_ORIENTATION_INVALID",
                            "discarding invalid SONIC orientation telemetry",
                            error=message,
                        )
                    orientation_error = message
                    continue
                if (
                    latest_orientation is None
                    or sample.emitted_at_monotonic_s
                    >= latest_orientation.emitted_at_monotonic_s
                ):
                    latest_orientation = sample
                if orientation_error:
                    report_event(
                        logging.INFO,
                        "SONIC_ORIENTATION_RECOVERED",
                        "SONIC orientation telemetry recovered",
                    )
                    orientation_error = ""
            while commands.poll(0):
                command = decode_navigation_message(commands.recv())
                if command.generation < generation or (
                    command.generation == generation
                    and (command.skill_id, command.segment_id)
                    < (skill_id, segment_id)
                ):
                    continue
                generation = command.generation
                skill_id = command.skill_id
                segment_id = command.segment_id
                mode = command.mode
                heading_result = None
                trajectory = np.empty((0, 2), dtype=np.float32)
                trajectory_time = 0.0
                invalid_count = 0
                last_stale_reason = None
                mpc_error = ""
                mpc_reference = np.empty((0, 2), dtype=np.float64)
                mpc_reference_version += 1
                mpc_linear_velocity = 0.0
                mpc_angular_velocity = 0.0
                mpc_result_time = 0.0
                next_mpc_update = 0.0
                if mode == "manual_velocity":
                    # Manual/BasePose velocity is consumed only by the common
                    # planner executor. NavDP clears its own navigation state.
                    mode = "stop"
                    world_goal = None
                elif mode == "stop":
                    world_goal = None
                elif mode == "nav_goal":
                    with sensors.lock:
                        pose = sensors.pose
                    if pose is None or command.goal_base is None:
                        mode = "stop"
                        send_status("failed", "odometry_unavailable")
                    else:
                        world_goal = base_goal_to_world(command.goal_base, pose)
                        heading_reference_yaw = pose.yaw
                        heading_target_yaw = pose.yaw
                        send_metrics({}, activate=True)
                        send_status("active", "goal_accepted")
                else:
                    now = time.monotonic()
                    sonic_yaw = _fresh_sonic_yaw(
                        latest_orientation,
                        now_s=now,
                        timeout_s=config.heading_orientation_timeout_s,
                    )
                    if sonic_yaw is None or command.heading_delta_rad is None:
                        mode = "stop"
                        send_status("failed", "sonic_orientation_unavailable")
                    else:
                        world_goal = None
                        # These protocol fields are source-neutral. For a heading
                        # goal they carry SONIC yaw so the executor can rebase it
                        # into its command-facing frame.
                        heading_reference_yaw = sonic_yaw
                        heading_target_yaw = None
                        heading_controller.start(
                            current_yaw=sonic_yaw,
                            delta_rad=command.heading_delta_rad,
                            turn_direction=command.heading_turn_direction,
                            now=now,
                            max_angular_speed_rad_s=(
                                command.heading_max_angular_speed_rad_s
                            ),
                            max_duration_s=command.heading_max_duration_s,
                        )
                        send_metrics({}, activate=True)
                        send_status("active", "heading_goal_accepted")

            gateway_camera = gateway.poll_camera()
            if gateway_camera is not None:
                latest_rgb = gateway_camera.rgb
                latest_depth = gateway_camera.depth_m
                camera_info = gateway_camera.camera_info
                latest_camera_timestamp = gateway_camera.source_timestamp_s
            if camera_info is not None and not navdp_initialized:
                try:
                    _reset_navdp(
                        navdp_endpoint,
                        camera_info,
                        stop_threshold=config.stop_threshold,
                    )
                    navdp_initialized = True
                    report_event(
                        logging.INFO,
                        "SERVER_READY",
                        "NavDP server initialized",
                        recovered=bool(server_error),
                    )
                    server_error = ""
                except Exception as exc:
                    message = str(exc)
                    if message != server_error:
                        report_event(
                            logging.WARNING,
                            "SERVER_UNAVAILABLE",
                            "waiting for NavDP server",
                            error=message,
                        )
                        server_error = message

            while not inference_result.empty():
                (
                    result_generation,
                    result_segment_id,
                    result,
                    inference_pose,
                    error,
                    elapsed_s,
                ) = (
                    inference_result.get_nowait()
                )
                if (
                    result_generation != generation
                    or result_segment_id != segment_id
                    or mode != "nav_goal"
                ):
                    continue
                send_metrics({"policy_inference": elapsed_s * 1000.0})
                if error or result is None:
                    invalid_count += 1
                    report_event(
                        logging.WARNING,
                        "INFERENCE_FAILED",
                        "NavDP inference failed",
                        attempt=invalid_count,
                        error=error,
                    )
                    if invalid_count >= 3:
                        mode = "stop"
                        send_status("failed", "three_invalid_trajectories")
                else:
                    world_reference = (
                        prepare_internnav_world_reference(result, inference_pose)
                        if inference_pose is not None
                        else np.empty((0, 2), dtype=np.float64)
                    )
                    if len(world_reference) < 2:
                        invalid_count += 1
                        report_event(
                            logging.WARNING,
                            "INVALID_TRAJECTORY",
                            "NavDP trajectory has no usable MPC reference",
                            attempt=invalid_count,
                        )
                    else:
                        invalid_count = 0
                        trajectory = result
                        trajectory_time = time.monotonic()
                        mpc_reference = world_reference.copy()
                        mpc_reference_version += 1
                        LOGGER.info(
                            "trajectory generation=%d points=%d local_goal=(%.3f, %.3f)",
                            generation,
                            len(trajectory),
                            current_local_goal[0],
                            current_local_goal[1],
                        )

            now = time.monotonic()
            with sensors.lock:
                pose, pose_time = sensors.pose, sensors.pose_time
                pose_history = list(sensors.pose_history)
                slam_map_xy = sensors.slam_map_xy.copy()
                slam_map_time = sensors.slam_map_time
                robot_history = sensors.robot_history.copy()
            slam_max_age_s = config.sensor_gateway_max_age_ms * 1.0e-3
            if slam_map_time <= 0.0 or now - slam_map_time > slam_max_age_s:
                # Do not keep presenting the last map as live after FAST-LIO or
                # SensorGateway has stopped.  The ingress also starts a fresh
                # map when registered-cloud samples resume after this gap.
                slam_map_xy = np.empty((0, 2), dtype=np.float32)
                robot_history = np.empty((0, 2), dtype=np.float32)
            if mode == "nav_goal" and pose is not None and world_goal is not None:
                current_local_goal = local_goal_from_world(world_goal, pose)
                if math.hypot(*current_local_goal) <= config.goal_tolerance_m:
                    mode = "stop"
                    send_status("reached", f"goal_within_{config.goal_tolerance_m:g}m")
                elif (
                    not inference_busy
                    and navdp_initialized
                    and latest_rgb is not None
                    and latest_depth is not None
                ):
                    inference_busy = True
                    inference_pose = closest_timestamped_pose(
                        pose_history, latest_camera_timestamp
                    ) or pose
                    threading.Thread(
                        target=infer,
                        args=(
                            generation,
                            segment_id,
                            latest_rgb.copy(),
                            latest_depth.copy(),
                            current_local_goal,
                            inference_pose,
                        ),
                        daemon=True,
                    ).start()

            zero_action_aborted = False
            new_mpc_solution = False
            emit_heading_target = False
            terminal_heading_status: tuple[str, str] | None = None
            result = mpc_solver.poll_latest()
            if (
                result is not None
                and result.generation == generation
                and result.reference_version == mpc_reference_version
            ):
                send_metrics({"mpc_solve": result.elapsed_s * 1000.0})
                if result.error is None and result.control is not None:
                    mpc_linear_velocity, mpc_angular_velocity = result.control
                    mpc_result_time = result.completed_time
                    new_mpc_solution = True
                    if mpc_error:
                        report_event(
                            logging.INFO,
                            "MPC_RECOVERED",
                            "NavDP MPC solver recovered",
                        )
                        mpc_error = ""
                else:
                    message = f"{result.return_status}: {result.error}"
                    if message != mpc_error:
                        report_event(
                            logging.WARNING,
                            "MPC_FAILED",
                            "NavDP MPC solve failed",
                            elapsed_ms=result.elapsed_s * 1000.0,
                            status=result.return_status,
                            error=result.error,
                        )
                        mpc_error = message
            if mode == "nav_goal":
                if now >= next_mpc_update and len(mpc_reference) and pose is not None:
                    next_mpc_update = now + 1.0 / config.mpc_hz
                    mpc_solver.submit(
                        MpcSolveRequest(
                            generation=generation,
                            reference_version=mpc_reference_version,
                            world_reference=mpc_reference.copy(),
                            pose=pose,
                        )
                    )
                effective_control = fresh_mpc_control(
                    (mpc_linear_velocity, mpc_angular_velocity),
                    result_time=mpc_result_time,
                    now=now,
                    timeout_s=config.mpc_result_timeout_s,
                )
                mpc_linear_velocity, mpc_angular_velocity = effective_control
                mpc_solution_available = mpc_result_time > 0.0
                if new_mpc_solution:
                    zero_action_aborted = should_abort_nav_for_zero_action(
                        mode=mode,
                        selected_command=(
                            mpc_linear_velocity,
                            0.0,
                            mpc_angular_velocity,
                        ),
                        command_available=mpc_solution_available,
                    )
                    heading_target_yaw = fastlio_heading_target_from_mpc(
                        fastlio_yaw=pose.yaw,
                        mpc_angular_velocity=mpc_angular_velocity,
                        heading_preview_s=config.heading_preview_s,
                    )
                velocity = xnavdp_control_to_body_velocity(
                    mpc_linear_velocity,
                    mpc_angular_velocity,
                )
            elif mode == "heading_goal":
                emit_heading_target = True
                sonic_yaw = _fresh_sonic_yaw(
                    latest_orientation,
                    now_s=now,
                    timeout_s=config.heading_orientation_timeout_s,
                )
                if sonic_yaw is None:
                    mode = "stop"
                    velocity = (0.0, 0.0, 0.0)
                    terminal_heading_status = (
                        "failed", "sonic_orientation_timeout",
                    )
                else:
                    heading_result = heading_controller.update(
                        current_yaw=sonic_yaw,
                        now=now,
                    )
                    heading_target_yaw = heading_result.target_rad
                    heading_reference_yaw = heading_result.reference_rad
                    velocity = (
                        0.0,
                        0.0,
                        heading_result.angular_velocity_rad_s,
                    )
                    if heading_result.state != "active":
                        LOGGER.log(
                            logging.ERROR
                            if heading_result.state == "failed"
                            else logging.INFO,
                            "heading finished state=%s requested_rad=%.6f "
                            "accumulated_sonic_yaw_rad=%.6f remaining_rad=%.6f "
                            "reason=%s",
                            heading_result.state,
                            heading_controller.turn_delta_rad,
                            heading_controller.accumulated_yaw_rad,
                            heading_result.remaining_rad,
                            heading_result.reason,
                        )
                        mode = "stop"
                        velocity = (0.0, 0.0, 0.0)
                        terminal_heading_status = (
                            heading_result.state, heading_result.reason,
                        )
            else:
                velocity = (0.0, 0.0, 0.0)
            pose, pose_time, points = _control_freshness_snapshot(sensors)
            now = time.monotonic()
            stale_reason = None
            if mode == "nav_goal" and now - pose_time > config.odometry_timeout_s:
                stale_reason = "odometry_timeout"
            elif mode == "nav_goal" and (
                not len(trajectory)
                or now - trajectory_time > config.trajectory_timeout_s
            ):
                stale_reason = "trajectory_stale"
            if stale_reason:
                velocity = (0.0, 0.0, 0.0)
            if mode == "nav_goal" and stale_reason != last_stale_reason:
                if stale_reason:
                    report_event(
                        logging.WARNING,
                        "NAVIGATION_BLOCKED",
                        "NavDP output blocked by stale input",
                        reason=stale_reason,
                    )
                elif last_stale_reason:
                    report_event(
                        logging.INFO,
                        "NAVIGATION_RESUMED",
                        "NavDP input freshness recovered",
                        previous_reason=last_stale_reason,
                    )
                last_stale_reason = stale_reason
            current_rays = actor_ray_from_points(points)
            heading_kwargs = (
                {
                    "heading_target_rad": heading_target_yaw,
                    "heading_reference_rad": heading_reference_yaw,
                }
                if (mode in {"nav_goal", "heading_goal"} or emit_heading_target)
                and heading_target_yaw is not None
                and heading_reference_yaw is not None
                else {}
            )
            output.send_string(
                build_planner_velocity_message(
                    generation=generation,
                    skill_id=skill_id,
                    segment_id=segment_id,
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
                mode = "stop"
                trajectory = np.empty((0, 2), dtype=np.float32)
                mpc_reference = np.empty((0, 2), dtype=np.float64)
                mpc_reference_version += 1
                mpc_linear_velocity = 0.0
                mpc_angular_velocity = 0.0
                mpc_result_time = 0.0
            actor_ray_panel = render_actor_ray_panel(
                current_rays,
                trajectory=trajectory,
                velocity=velocity,
            )
            slam_2d_panel = render_slam_world_panel(
                slam_map_xy,
                pose=pose,
                world_goal=world_goal,
                trajectory_world=None,
                robot_history=robot_history,
            )
            visualization_publisher.publish(NAVDP_ACTOR_RAY_STREAM, actor_ray_panel)
            visualization_publisher.publish(NAVDP_SLAM_2D_STREAM, slam_2d_panel)
            if mode == "nav_goal":
                now = time.monotonic()
                timing_ms = {"control_loop": (now - loop_started) * 1000.0}
                if pose_time > 0.0:
                    timing_ms["odometry_age"] = (
                        max(0.0, now - pose_time) * 1000.0
                    )
                if trajectory_time > 0.0:
                    timing_ms["trajectory_age"] = (
                        max(0.0, now - trajectory_time) * 1000.0
                    )
                send_metrics(timing_ms)
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
                    generation=generation,
                    skill_id=skill_id,
                    segment_id=segment_id,
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
