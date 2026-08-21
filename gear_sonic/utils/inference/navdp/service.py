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

from gear_sonic.runtime.profile import load_runtime_profile
from gear_sonic.runtime.telemetry import (
    NAVDP_TIMING_SEGMENTS,
    build_event,
    configure_file_logging,
    emit_event,
    open_telemetry_publisher,
    publish_metrics,
)
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
    load_navdp_planner_config,
    _control_freshness_snapshot,
    _navdp_request,
    _reset_navdp,
    _SharedSensors,
)
from gear_sonic.utils.inference.navdp.navigation import (
    STATUS_TYPE,
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

LOGGER = logging.getLogger("sonic.navdp")


def main(config: NavDPPlannerConfig) -> None:
    configure_file_logging("navdp")
    profile = load_runtime_profile(config.profile or None, overlays=config.overlay)
    command_endpoint = profile.endpoint_uri("navigation_command")
    output_endpoint = profile.endpoint_uri("navdp_velocity")
    navdp_endpoint = profile.endpoint_uri("xnavdp_http")
    sensor_endpoint = profile.endpoint_uri("sensor_gateway_metadata")
    import zmq
    from gear_sonic.runtime.gateway.visualization import (
        NAVDP_ACTOR_RAY_STREAM,
        NAVDP_SLAM_2D_STREAM,
        VisualizationPublisher,
    )

    install_shutdown_signal_handlers()

    context = zmq.Context.instance()
    commands = context.socket(zmq.SUB)
    commands.connect(command_endpoint)
    commands.setsockopt_string(zmq.SUBSCRIBE, "")
    status = context.socket(zmq.PUB)
    status.bind(profile.endpoint_uri("navigation_status"))
    output = context.socket(zmq.PUB)
    output.bind(output_endpoint)
    event_socket = open_telemetry_publisher(
        profile.endpoint_uri("runtime_event_ingress")
    )
    metrics_socket = open_telemetry_publisher(
        profile.endpoint_uri("runtime_metrics_ingress")
    )
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
    )
    gateway.start()
    generation = 0
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
        tuple[int, np.ndarray | None, Pose2D | None, str | None, float]
    ] = queue.Queue(maxsize=1)
    nav_fastlio_reference_yaw: float | None = None
    fastlio_target_heading: float | None = None
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

    def report_event(level: int, code: str, message: str, **fields: object) -> None:
        emit_event(
            build_event("navdp", level, code, message, **fields),
            socket=event_socket,
            logger=LOGGER,
        )

    def send_metrics(values: Mapping[str, float], *, activate: bool = False) -> None:
        publish_metrics(
            metrics_socket,
            "navdp",
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
                "state": state,
                "reason": reason,
            }
        )
        LOGGER.log(
            logging.ERROR if state == "failed" else logging.INFO,
            "navigation status=%s generation=%d reason=%s",
            state,
            generation,
            reason,
        )

    def infer(
        request_generation: int,
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
                result,
                inference_pose,
                None,
                time.monotonic() - started,
            )
        except Exception as exc:
            item = (
                request_generation,
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
        "command=%s output=%s sensors=%s",
        command_endpoint,
        output_endpoint,
        sensor_endpoint,
    )
    period = 1.0 / config.control_hz
    try:
        while True:
            loop_started = time.monotonic()
            while commands.poll(0):
                command = decode_navigation_message(commands.recv())
                if command.generation < generation:
                    continue
                generation = command.generation
                mode = command.mode
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
                else:
                    with sensors.lock:
                        pose = sensors.pose
                    if pose is None or command.goal_base is None:
                        mode = "stop"
                        send_status("failed", "odometry_unavailable")
                    else:
                        world_goal = base_goal_to_world(command.goal_base, pose)
                        nav_fastlio_reference_yaw = pose.yaw
                        fastlio_target_heading = pose.yaw
                        send_metrics({}, activate=True)
                        send_status("active", "goal_accepted")

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
                result_generation, result, inference_pose, error, elapsed_s = (
                    inference_result.get_nowait()
                )
                if result_generation != generation or mode != "nav_goal":
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
                            latest_rgb.copy(),
                            latest_depth.copy(),
                            current_local_goal,
                            inference_pose,
                        ),
                        daemon=True,
                    ).start()

            zero_action_aborted = False
            new_mpc_solution = False
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
                    fastlio_target_heading = fastlio_heading_target_from_mpc(
                        fastlio_yaw=pose.yaw,
                        mpc_angular_velocity=mpc_angular_velocity,
                        heading_preview_s=config.heading_preview_s,
                    )
                velocity = xnavdp_control_to_body_velocity(
                    mpc_linear_velocity,
                    mpc_angular_velocity,
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
                    "heading_target_rad": fastlio_target_heading,
                    "heading_reference_rad": nav_fastlio_reference_yaw,
                }
                if mode == "nav_goal"
                and fastlio_target_heading is not None
                and nav_fastlio_reference_yaw is not None
                else {}
            )
            output.send_string(
                build_planner_velocity_message(
                    generation=generation,
                    source="navdp",
                    velocity=velocity,
                    **heading_kwargs,
                )
            )
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
                    source="navdp",
                    velocity=(0.0, 0.0, 0.0),
                )
            )
        gateway.close()
        commands.close(0)
        status.close(0)
        output.close(0)
        event_socket.close(linger=0)
        metrics_socket.close(linger=0)
        visualization_publisher.close()


if __name__ == "__main__":
    import tyro
    from gear_sonic.runtime.profile import RuntimeProfileSelection

    selection = tyro.cli(RuntimeProfileSelection)
    main(load_navdp_planner_config(selection.profile, selection.overlay))
