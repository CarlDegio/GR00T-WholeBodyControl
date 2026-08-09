#!/usr/bin/env python3
"""Continuous NavDP orchestration entry point."""

from __future__ import annotations

import math
import queue
import threading
import time
from typing import Any, Mapping

import numpy as np

from gear_sonic.navdp.control import (
    XNAVDP_G1_MPC_DEFAULTS,
    AsyncMpcSolver,
    InternNavMpcController,
    LatestMessageWorker,
    MpcSolveRequest,
    MpcSolveResult,
    SonicPlannerState,
    apply_hard_safety,
    fresh_mpc_control,
    prepare_internnav_world_reference,
    should_abort_nav_for_lidar,
    should_abort_nav_for_zero_action,
    sonic_heading_from_mpc,
    xnavdp_adaptive_speed,
    xnavdp_control_to_body_velocity,
)
from gear_sonic.navdp.gateway import (
    GatewayCameraFrame,
    NavDPPlannerConfig,
    NavDPSensorGatewayIngress,
    _control_freshness_snapshot,
    _encode_navdp_frames,
    _extract_camera_frame,
    _gateway_camera_frame,
    _navdp_request,
    _quaternion_yaw,
    _remove_ground,
    _reset_navdp,
    _SharedSensors,
    _update_lidar_state,
    _update_odometry_state,
    _update_slam_cloud_state,
    point_plane_distances,
)
from gear_sonic.navdp.navigation import (
    COMMAND_TYPE,
    STATUS_TYPE,
    NavigationCommand,
    Pose2D,
    base_goal_to_world,
    build_navigation_message,
    closest_timestamped_pose,
    decode_navigation_message,
    local_goal_from_world,
    local_trajectory_to_world,
    update_slam_map,
)
from gear_sonic.navdp.visualization import (
    _VIZ_CENTER,
    _VIZ_RADIUS,
    _VIZ_SIZE,
    ActorRayRecordingSession,
    ActorRayVideoRecorder,
    _reasan_base_panel,
    _viz_pixels,
    actor_ray_from_points,
    actor_ray_velocity_arrow,
    compose_head_rgbd_view,
    compose_reasan_navigation_view,
    depth_requires_stop,
    filter_livox_points,
    format_actor_ray_control_text,
    format_direction_chain_diagnostics,
    format_navigation_diagnostics,
    install_shutdown_signal_handlers,
    integrate_velocity_path,
    render_actor_ray_panel,
    render_head_depth_panel,
    render_slam_world_panel,
)

__all__ = [
    "COMMAND_TYPE",
    "STATUS_TYPE",
    "XNAVDP_G1_MPC_DEFAULTS",
    "ActorRayRecordingSession",
    "ActorRayVideoRecorder",
    "AsyncMpcSolver",
    "GatewayCameraFrame",
    "InternNavMpcController",
    "LatestMessageWorker",
    "MpcSolveRequest",
    "MpcSolveResult",
    "NavDPPlannerConfig",
    "NavDPSensorGatewayIngress",
    "NavigationCommand",
    "Pose2D",
    "SonicPlannerState",
    "_SharedSensors",
    "_VIZ_CENTER",
    "_VIZ_RADIUS",
    "_VIZ_SIZE",
    "_control_freshness_snapshot",
    "_encode_navdp_frames",
    "_extract_camera_frame",
    "_gateway_camera_frame",
    "_navdp_request",
    "_quaternion_yaw",
    "_reasan_base_panel",
    "_remove_ground",
    "_reset_navdp",
    "_update_lidar_state",
    "_update_odometry_state",
    "_update_slam_cloud_state",
    "_viz_pixels",
    "actor_ray_from_points",
    "actor_ray_velocity_arrow",
    "apply_hard_safety",
    "base_goal_to_world",
    "build_navigation_message",
    "closest_timestamped_pose",
    "compose_head_rgbd_view",
    "compose_reasan_navigation_view",
    "decode_navigation_message",
    "depth_requires_stop",
    "filter_livox_points",
    "format_actor_ray_control_text",
    "format_direction_chain_diagnostics",
    "format_navigation_diagnostics",
    "fresh_mpc_control",
    "install_shutdown_signal_handlers",
    "integrate_velocity_path",
    "local_goal_from_world",
    "local_trajectory_to_world",
    "main",
    "point_plane_distances",
    "prepare_internnav_world_reference",
    "render_actor_ray_panel",
    "render_head_depth_panel",
    "render_slam_world_panel",
    "should_abort_nav_for_lidar",
    "should_abort_nav_for_zero_action",
    "sonic_heading_from_mpc",
    "update_slam_map",
    "xnavdp_adaptive_speed",
    "xnavdp_control_to_body_velocity",
]


def main(config: NavDPPlannerConfig) -> None:
    import cv2
    import zmq
    from gear_sonic.runtime.visualization import VisualizationPublisher

    install_shutdown_signal_handlers()

    context = zmq.Context.instance()
    commands = context.socket(zmq.SUB)
    commands.connect(config.command_endpoint)
    commands.setsockopt_string(zmq.SUBSCRIBE, "")
    status = context.socket(zmq.PUB)
    status.bind(config.status_endpoint)
    output = context.socket(zmq.PUB)
    output.bind(config.output_endpoint)
    visualization_publisher = (
        VisualizationPublisher(config.visualization_gateway_endpoint)
        if config.visualization_gateway_endpoint
        else None
    )
    sensors = _SharedSensors()
    gateway = NavDPSensorGatewayIngress(
        config.sensor_gateway_endpoint,
        sensors,
        poll_hz=config.sensor_gateway_poll_hz,
        request_timeout_ms=config.sensor_gateway_request_timeout_ms,
        max_age_ms=config.sensor_gateway_max_age_ms,
        max_skew_ms=config.sensor_gateway_max_skew_ms,
    )
    gateway.start()
    actorray_recording = (
        ActorRayRecordingSession(
            config.actorray_output_dir, fps=config.actorray_record_fps
        )
        if config.record_actorray
        else None
    )

    generation = 0
    mode = "stop"
    manual = (0.0, 0.0, 0.0)
    world_goal: tuple[float, float] | None = None
    current_local_goal = (0.0, 0.0)
    nav_confidence = 0.0
    trajectory = np.empty((0, 2), dtype=np.float32)
    trajectory_time = 0.0
    invalid_count = 0
    latest_rgb: np.ndarray | None = None
    latest_depth: np.ndarray | None = None
    camera_info: Mapping[str, Any] | None = None
    navdp_initialized = False
    inference_busy = False
    inference_result: queue.Queue[
        tuple[int, np.ndarray | None, Pose2D | None, str | None]
    ] = queue.Queue(maxsize=1)
    trajectory_log_pending = False
    last_safety_blocked = False
    sonic_planner = SonicPlannerState()
    sonic_fastlio_yaw_offset = 0.0
    mpc_solver = AsyncMpcSolver()
    mpc_reference = np.empty((0, 2), dtype=np.float64)
    mpc_reference_version = 0
    mpc_linear_velocity = 0.0
    mpc_angular_velocity = 0.0
    mpc_result_time = 0.0
    next_mpc_update = 0.0
    latest_camera_timestamp = 0.0
    last_direction_log_time = 0.0
    previous_direction_pose_yaw: float | None = None

    def send_status(state: str, reason: str) -> None:
        status.send_json({"type": STATUS_TYPE, "version": 1, "generation": generation, "state": state, "reason": reason})

    def infer(
        request_generation: int,
        rgb: np.ndarray,
        depth: np.ndarray,
        goal: tuple[float, float],
        inference_pose: Pose2D | None,
    ) -> None:
        nonlocal inference_busy
        try:
            result = _navdp_request(
                config.navdp_server,
                rgb,
                depth,
                goal,
                pose=inference_pose,
                timeout=config.navdp_request_timeout_s,
            )
            item = (request_generation, result, inference_pose, None)
        except Exception as exc:
            item = (request_generation, None, inference_pose, str(exc))
        try:
            inference_result.put_nowait(item)
        except queue.Full:
            pass
        inference_busy = False

    print(f"[NavDP] command={config.command_endpoint} output={config.output_endpoint}")
    print(f"[NavDP] sensors=gateway endpoint={config.sensor_gateway_endpoint}")
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
                if actorray_recording is not None:
                    actorray_recording.stop()
                trajectory = np.empty((0, 2), dtype=np.float32)
                invalid_count = 0
                trajectory_log_pending = False
                last_safety_blocked = False
                mpc_reference = np.empty((0, 2), dtype=np.float64)
                mpc_reference_version += 1
                mpc_linear_velocity = 0.0
                mpc_angular_velocity = 0.0
                mpc_result_time = 0.0
                next_mpc_update = 0.0
                if mode == "manual_velocity":
                    manual = command.velocity or (0.0, 0.0, 0.0)
                    nav_confidence = 0.0
                elif mode == "stop":
                    manual = (0.0, 0.0, 0.0)
                    world_goal = None
                    nav_confidence = 0.0
                else:
                    with sensors.lock:
                        pose = sensors.pose
                    if pose is None or command.goal_base is None:
                        mode = "stop"
                        send_status("failed", "odometry_unavailable")
                    else:
                        nav_confidence = command.confidence
                        world_goal = base_goal_to_world(command.goal_base, pose)
                        sonic_fastlio_yaw_offset = math.remainder(
                            sonic_planner.heading - pose.yaw, 2.0 * math.pi
                        )
                        if actorray_recording is not None:
                            actorray_recording.start(generation)
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
                        config.navdp_server,
                        camera_info,
                        stop_threshold=config.navdp_stop_threshold,
                    )
                    navdp_initialized = True
                    print("[NavDP] server initialized")
                except Exception as exc:
                    print(f"[NavDP] waiting for server: {exc}")

            while not inference_result.empty():
                result_generation, result, inference_pose, error = inference_result.get_nowait()
                if result_generation != generation or mode != "nav_goal":
                    continue
                if error or result is None:
                    invalid_count += 1
                    print(f"[NavDP] inference failed ({invalid_count}/3): {error}")
                    if invalid_count >= 3:
                        mode = "stop"
                        if actorray_recording is not None:
                            actorray_recording.stop()
                        send_status("failed", "three_invalid_trajectories")
                else:
                    world_reference = (
                        prepare_internnav_world_reference(result, inference_pose)
                        if inference_pose is not None
                        else np.empty((0, 2), dtype=np.float64)
                    )
                    if len(world_reference) < 2:
                        invalid_count += 1
                        print("[NavDP] inference trajectory has no usable MPC reference")
                    else:
                        invalid_count = 0
                        trajectory = result
                        trajectory_time = time.monotonic()
                        trajectory_log_pending = True
                        mpc_reference = world_reference.copy()
                        mpc_reference_version += 1

            now = time.monotonic()
            with sensors.lock:
                pose, pose_time = sensors.pose, sensors.pose_time
                pose_history = list(sensors.pose_history)
                points, points_time = sensors.points.copy(), sensors.points_time
                slam_map_xy = sensors.slam_map_xy.copy()
                robot_history = sensors.robot_history.copy()
            if mode == "nav_goal" and pose is not None and world_goal is not None:
                current_local_goal = local_goal_from_world(world_goal, pose)
                if math.hypot(*current_local_goal) <= config.goal_tolerance_m:
                    mode = "stop"
                    if actorray_recording is not None:
                        actorray_recording.stop()
                    send_status("reached", f"goal_within_{config.goal_tolerance_m:g}m")
                elif not inference_busy and navdp_initialized and latest_rgb is not None and latest_depth is not None:
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
            mpc_solution_available = False
            new_mpc_solution = False
            result = mpc_solver.poll_latest()
            if (
                result is not None
                and result.generation == generation
                and result.reference_version == mpc_reference_version
            ):
                if result.error is None and result.control is not None:
                    mpc_linear_velocity, mpc_angular_velocity = result.control
                    mpc_result_time = result.completed_time
                    mpc_solution_available = True
                    new_mpc_solution = True
                else:
                    print(
                        f"[NavDP] MPC solve failed after {result.elapsed_s:.3f}s "
                        f"status={result.return_status}: {result.error}",
                        flush=True,
                    )
            if mode == "manual_velocity":
                velocity = manual
            elif mode == "nav_goal":
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
                    sonic_planner.heading = sonic_heading_from_mpc(
                        fastlio_yaw=pose.yaw,
                        fastlio_to_sonic_offset=sonic_fastlio_yaw_offset,
                        mpc_angular_velocity=mpc_angular_velocity,
                        heading_preview_s=config.heading_preview_s,
                    )
                    if now - last_direction_log_time >= 1.0 and len(trajectory):
                        fastlio_yaw_delta = (
                            0.0
                            if previous_direction_pose_yaw is None
                            else math.remainder(
                                pose.yaw - previous_direction_pose_yaw,
                                2.0 * math.pi,
                            )
                        )
                        print(
                            format_direction_chain_diagnostics(
                                trajectory=trajectory,
                                mpc_angular_velocity=mpc_angular_velocity,
                                fastlio_yaw=pose.yaw,
                                fastlio_yaw_delta=fastlio_yaw_delta,
                                sonic_target_heading=sonic_planner.heading,
                            ),
                            flush=True,
                        )
                        previous_direction_pose_yaw = pose.yaw
                        last_direction_log_time = now
                velocity = xnavdp_control_to_body_velocity(
                    mpc_linear_velocity,
                    mpc_angular_velocity,
                )
            else:
                velocity = (0.0, 0.0, 0.0)
            pose, pose_time, points, points_time = _control_freshness_snapshot(
                sensors
            )
            now = time.monotonic()
            candidate_velocity = velocity
            stale_reason = None
            if now - points_time > config.radar_timeout_s:
                stale_reason = "radar_timeout"
            elif mode == "nav_goal" and now - pose_time > config.odom_timeout_s:
                stale_reason = "odometry_timeout"
            elif mode == "nav_goal" and (not len(trajectory) or now - trajectory_time > config.trajectory_timeout_s):
                stale_reason = "trajectory_stale"
            if stale_reason:
                velocity = (0.0, 0.0, 0.0)
            camera_stop = latest_depth is not None and depth_requires_stop(latest_depth)
            current_rays = actor_ray_from_points(points)
            angles = np.deg2rad(-179.0 + 2.0 * np.arange(180, dtype=np.float32))
            hit = current_rays < 3.0
            safety_points = np.column_stack((
                current_rays[hit] * np.cos(angles[hit]),
                current_rays[hit] * np.sin(angles[hit]),
                np.zeros(int(hit.sum())),
            )).astype(np.float32)
            before_safety = velocity
            velocity = apply_hard_safety(velocity, safety_points, camera_stop=camera_stop)
            if actorray_recording is not None and mode == "nav_goal":
                actorray_recording.write(
                    render_actor_ray_panel(
                        current_rays,
                        trajectory=trajectory,
                        velocity=velocity,
                    )
                )
            lidar_aborted = should_abort_nav_for_lidar(
                mode=mode,
                before_safety=before_safety,
                after_safety=velocity,
                camera_stop=camera_stop,
            )
            output.send(
                sonic_planner.message(
                    velocity,
                    dt=period if mode == "manual_velocity" else 0.0,
                )
            )
            safety_blocked = not np.allclose(velocity, candidate_velocity, atol=1.0e-6)
            if lidar_aborted or zero_action_aborted:
                stop_reason = "lidar_hard_stop" if lidar_aborted else "navdp_zero_action"
                print(f"[NavDP] navigation stopped: {stop_reason}", flush=True)
                send_status("stopped", stop_reason)
                mode = "stop"
                if actorray_recording is not None:
                    actorray_recording.stop()
                trajectory = np.empty((0, 2), dtype=np.float32)
                mpc_reference = np.empty((0, 2), dtype=np.float64)
                mpc_reference_version += 1
                mpc_linear_velocity = 0.0
                mpc_angular_velocity = 0.0
                mpc_result_time = 0.0
            if (
                mode == "nav_goal"
                and world_goal is not None
                and len(trajectory)
                and (trajectory_log_pending or safety_blocked != last_safety_blocked)
            ):
                reason = stale_reason or ("depth_hard_stop" if camera_stop else "radar_hard_stop" if safety_blocked else "clear")
                print(
                    format_navigation_diagnostics(
                        generation=generation,
                        confidence=nav_confidence,
                        local_goal=current_local_goal,
                        world_goal=world_goal,
                        trajectory=trajectory,
                        velocity=velocity,
                    )
                    + f"\n  safety_state={reason}",
                    flush=True,
                )
                trajectory_log_pending = False
            last_safety_blocked = safety_blocked

            if config.visualize or visualization_publisher is not None:
                canvas = compose_reasan_navigation_view(
                    points,
                    current_rays,
                    trajectory,
                    slam_map_xy=slam_map_xy,
                    pose=pose,
                    world_goal=world_goal,
                    robot_history=robot_history,
                    velocity=velocity,
                )
                head_rgbd = compose_head_rgbd_view(latest_rgb, latest_depth)
                if visualization_publisher is not None:
                    visualization_publisher.publish(
                        "visualization/navdp_navigation", canvas
                    )
                    visualization_publisher.publish(
                        "visualization/navdp_head_rgbd", head_rgbd
                    )
                if config.visualize:
                    cv2.imshow("NavDP + MID360 + FAST-LIO world map", canvas)
                    cv2.imshow("NavDP Head RGB-D", head_rgbd)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
            delay = period - (time.monotonic() - loop_started)
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        pass
    finally:
        mpc_solver.close()
        for _ in range(3):
            output.send(sonic_planner.message((0.0, 0.0, 0.0)))
        gateway.close()
        commands.close(0)
        status.close(0)
        output.close(0)
        if actorray_recording is not None:
            actorray_recording.stop()
        if visualization_publisher is not None:
            visualization_publisher.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import tyro

    main(tyro.cli(NavDPPlannerConfig))
