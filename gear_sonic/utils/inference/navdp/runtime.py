"""State and single-cycle stages for the NavDP service."""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

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
    _reset_navdp,
    _SharedSensors,
)
from gear_sonic.utils.inference.navdp.navigation import (
    HeadingGoalController,
    Pose2D,
    base_goal_to_world,
    closest_timestamped_pose,
    decode_navigation_message,
    local_goal_from_world,
)
from gear_sonic.utils.teleop.sonic_orientation_telemetry import (
    OrientationTelemetrySample,
    decode_orientation_telemetry,
)

LOGGER = logging.getLogger("sonic.navdp")


@dataclass
class _NavDPRuntimeState:
    generation: int = 0
    skill_id: int = 0
    segment_id: int = 0
    mode: str = "stop"
    world_goal: tuple[float, float] | None = None
    current_local_goal: tuple[float, float] = (0.0, 0.0)
    trajectory: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float32)
    )
    trajectory_time: float = 0.0
    invalid_count: int = 0
    latest_rgb: np.ndarray | None = None
    latest_depth: np.ndarray | None = None
    camera_info: Mapping[str, Any] | None = None
    navdp_initialized: bool = False
    inference_busy: bool = False
    heading_reference_yaw: float | None = None
    heading_target_yaw: float | None = None
    mpc_reference: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float64)
    )
    mpc_reference_version: int = 0
    mpc_linear_velocity: float = 0.0
    mpc_angular_velocity: float = 0.0
    mpc_result_time: float = 0.0
    next_mpc_update: float = 0.0
    latest_camera_timestamp: float = 0.0
    server_error: str = ""
    mpc_error: str = ""
    last_stale_reason: str | None = None
    heading_result: Any = None
    latest_orientation: OrientationTelemetrySample | None = None
    orientation_error: str = ""


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


def _drain_orientation_messages(
    orientation,
    state: _NavDPRuntimeState,
    report_event,
) -> None:
    while orientation.poll(0):
        try:
            sample = decode_orientation_telemetry(orientation.recv())
        except ValueError as exc:
            message = str(exc)
            if message != state.orientation_error:
                report_event(
                    logging.WARNING,
                    "SONIC_ORIENTATION_INVALID",
                    "discarding invalid SONIC orientation telemetry",
                    error=message,
                )
            state.orientation_error = message
            continue
        if (
            state.latest_orientation is None
            or sample.emitted_at_monotonic_s
            >= state.latest_orientation.emitted_at_monotonic_s
        ):
            state.latest_orientation = sample
        if state.orientation_error:
            report_event(
                logging.INFO,
                "SONIC_ORIENTATION_RECOVERED",
                "SONIC orientation telemetry recovered",
            )
            state.orientation_error = ""


def _reset_command_state(state: _NavDPRuntimeState) -> None:
    state.heading_result = None
    state.trajectory = np.empty((0, 2), dtype=np.float32)
    state.trajectory_time = 0.0
    state.invalid_count = 0
    state.last_stale_reason = None
    state.mpc_error = ""
    state.mpc_reference = np.empty((0, 2), dtype=np.float64)
    state.mpc_reference_version += 1
    state.mpc_linear_velocity = 0.0
    state.mpc_angular_velocity = 0.0
    state.mpc_result_time = 0.0
    state.next_mpc_update = 0.0


def _apply_navigation_command(
    command,
    state: _NavDPRuntimeState,
    *,
    sensors: _SharedSensors,
    config: NavDPPlannerConfig,
    heading_controller: HeadingGoalController,
    send_metrics,
    send_status,
) -> None:
    if command.generation < state.generation or (
        command.generation == state.generation
        and (command.skill_id, command.segment_id)
        < (state.skill_id, state.segment_id)
    ):
        return
    state.generation = command.generation
    state.skill_id = command.skill_id
    state.segment_id = command.segment_id
    state.mode = command.mode
    _reset_command_state(state)
    if state.mode == "manual_velocity":
        # Manual/BasePose velocity is consumed only by the common planner
        # executor. NavDP clears its own navigation state.
        state.mode = "stop"
        state.world_goal = None
    elif state.mode == "stop":
        state.world_goal = None
    elif state.mode == "nav_goal":
        with sensors.lock:
            pose = sensors.pose
        if pose is None or command.goal_base is None:
            state.mode = "stop"
            send_status("failed", "odometry_unavailable")
        else:
            state.world_goal = base_goal_to_world(command.goal_base, pose)
            state.heading_reference_yaw = pose.yaw
            state.heading_target_yaw = pose.yaw
            send_metrics({}, activate=True)
            send_status("active", "goal_accepted")
    else:
        now = time.monotonic()
        sonic_yaw = _fresh_sonic_yaw(
            state.latest_orientation,
            now_s=now,
            timeout_s=config.heading_orientation_timeout_s,
        )
        if sonic_yaw is None or command.heading_delta_rad is None:
            state.mode = "stop"
            send_status("failed", "sonic_orientation_unavailable")
        else:
            state.world_goal = None
            # These protocol fields are source-neutral. For a heading goal they
            # carry SONIC yaw so the executor can rebase it into its command
            # facing frame.
            state.heading_reference_yaw = sonic_yaw
            state.heading_target_yaw = None
            heading_controller.start(
                current_yaw=sonic_yaw,
                delta_rad=command.heading_delta_rad,
                turn_direction=command.heading_turn_direction,
                now=now,
                max_angular_speed_rad_s=command.heading_max_angular_speed_rad_s,
                max_duration_s=command.heading_max_duration_s,
            )
            send_metrics({}, activate=True)
            send_status("active", "heading_goal_accepted")


def _drain_navigation_commands(
    commands,
    state: _NavDPRuntimeState,
    **apply_kwargs,
) -> None:
    while commands.poll(0):
        _apply_navigation_command(
            decode_navigation_message(commands.recv()),
            state,
            **apply_kwargs,
        )


def _compute_control(
    state: _NavDPRuntimeState,
    *,
    now: float,
    pose: Pose2D | None,
    config: NavDPPlannerConfig,
    mpc_solver: AsyncMpcSolver,
    heading_controller: HeadingGoalController,
    send_metrics,
    report_event,
) -> tuple[
    tuple[float, float, float],
    bool,
    bool,
    tuple[str, str] | None,
]:
    zero_action_aborted = False
    new_mpc_solution = False
    emit_heading_target = False
    terminal_heading_status: tuple[str, str] | None = None
    result = mpc_solver.poll_latest()
    if (
        result is not None
        and result.generation == state.generation
        and result.reference_version == state.mpc_reference_version
    ):
        send_metrics({"mpc_solve": result.elapsed_s * 1000.0})
        if result.error is None and result.control is not None:
            state.mpc_linear_velocity, state.mpc_angular_velocity = result.control
            state.mpc_result_time = result.completed_time
            new_mpc_solution = True
            if state.mpc_error:
                report_event(
                    logging.INFO,
                    "MPC_RECOVERED",
                    "NavDP MPC solver recovered",
                )
                state.mpc_error = ""
        else:
            message = f"{result.return_status}: {result.error}"
            if message != state.mpc_error:
                report_event(
                    logging.WARNING,
                    "MPC_FAILED",
                    "NavDP MPC solve failed",
                    elapsed_ms=result.elapsed_s * 1000.0,
                    status=result.return_status,
                    error=result.error,
                )
                state.mpc_error = message
    if state.mode == "nav_goal":
        if (
            now >= state.next_mpc_update
            and len(state.mpc_reference)
            and pose is not None
        ):
            state.next_mpc_update = now + 1.0 / config.mpc_hz
            mpc_solver.submit(
                MpcSolveRequest(
                    generation=state.generation,
                    reference_version=state.mpc_reference_version,
                    world_reference=state.mpc_reference.copy(),
                    pose=pose,
                )
            )
        effective_control = fresh_mpc_control(
            (state.mpc_linear_velocity, state.mpc_angular_velocity),
            result_time=state.mpc_result_time,
            now=now,
            timeout_s=config.mpc_result_timeout_s,
        )
        state.mpc_linear_velocity, state.mpc_angular_velocity = effective_control
        mpc_solution_available = state.mpc_result_time > 0.0
        if new_mpc_solution:
            zero_action_aborted = should_abort_nav_for_zero_action(
                mode=state.mode,
                selected_command=(
                    state.mpc_linear_velocity,
                    0.0,
                    state.mpc_angular_velocity,
                ),
                command_available=mpc_solution_available,
            )
            state.heading_target_yaw = fastlio_heading_target_from_mpc(
                fastlio_yaw=pose.yaw,
                mpc_angular_velocity=state.mpc_angular_velocity,
                heading_preview_s=config.heading_preview_s,
            )
        velocity = xnavdp_control_to_body_velocity(
            state.mpc_linear_velocity,
            state.mpc_angular_velocity,
        )
    elif state.mode == "heading_goal":
        emit_heading_target = True
        sonic_yaw = _fresh_sonic_yaw(
            state.latest_orientation,
            now_s=now,
            timeout_s=config.heading_orientation_timeout_s,
        )
        if sonic_yaw is None:
            state.mode = "stop"
            velocity = (0.0, 0.0, 0.0)
            terminal_heading_status = (
                "failed", "sonic_orientation_timeout",
            )
        else:
            state.heading_result = heading_controller.update(
                current_yaw=sonic_yaw,
                now=now,
            )
            state.heading_target_yaw = state.heading_result.target_rad
            state.heading_reference_yaw = state.heading_result.reference_rad
            velocity = (
                0.0,
                0.0,
                state.heading_result.angular_velocity_rad_s,
            )
            if state.heading_result.state != "active":
                LOGGER.log(
                    logging.ERROR
                    if state.heading_result.state == "failed"
                    else logging.INFO,
                    "heading finished state=%s requested_rad=%.6f "
                    "accumulated_sonic_yaw_rad=%.6f remaining_rad=%.6f "
                    "reason=%s",
                    state.heading_result.state,
                    heading_controller.turn_delta_rad,
                    heading_controller.accumulated_yaw_rad,
                    state.heading_result.remaining_rad,
                    state.heading_result.reason,
                )
                state.mode = "stop"
                velocity = (0.0, 0.0, 0.0)
                terminal_heading_status = (
                    state.heading_result.state,
                    state.heading_result.reason,
                )
    else:
        velocity = (0.0, 0.0, 0.0)
    return (
        velocity,
        zero_action_aborted,
        emit_heading_target,
        terminal_heading_status,
    )


def _refresh_gateway_camera(
    state: _NavDPRuntimeState,
    *,
    gateway: NavDPSensorGatewayIngress,
    navdp_endpoint: str,
    config: NavDPPlannerConfig,
    report_event,
) -> None:
    gateway_camera = gateway.poll_camera()
    if gateway_camera is not None:
        state.latest_rgb = gateway_camera.rgb
        state.latest_depth = gateway_camera.depth_m
        state.camera_info = gateway_camera.camera_info
        state.latest_camera_timestamp = gateway_camera.source_timestamp_s
    if state.camera_info is None or state.navdp_initialized:
        return
    try:
        _reset_navdp(
            navdp_endpoint,
            state.camera_info,
            stop_threshold=config.stop_threshold,
        )
        state.navdp_initialized = True
        report_event(
            logging.INFO,
            "SERVER_READY",
            "NavDP server initialized",
            recovered=bool(state.server_error),
        )
        state.server_error = ""
    except Exception as exc:
        message = str(exc)
        if message != state.server_error:
            report_event(
                logging.WARNING,
                "SERVER_UNAVAILABLE",
                "waiting for NavDP server",
                error=message,
            )
            state.server_error = message


def _consume_inference_results(
    state: _NavDPRuntimeState,
    inference_result: queue.Queue,
    *,
    send_metrics,
    send_status,
    report_event,
) -> None:
    while not inference_result.empty():
        (
            result_generation,
            result_segment_id,
            result,
            inference_pose,
            error,
            elapsed_s,
        ) = inference_result.get_nowait()
        if (
            result_generation != state.generation
            or result_segment_id != state.segment_id
            or state.mode != "nav_goal"
        ):
            continue
        send_metrics({"policy_inference": elapsed_s * 1000.0})
        if error or result is None:
            state.invalid_count += 1
            report_event(
                logging.WARNING,
                "INFERENCE_FAILED",
                "NavDP inference failed",
                attempt=state.invalid_count,
                error=error,
            )
            if state.invalid_count >= 3:
                state.mode = "stop"
                send_status("failed", "three_invalid_trajectories")
            continue
        world_reference = (
            prepare_internnav_world_reference(result, inference_pose)
            if inference_pose is not None
            else np.empty((0, 2), dtype=np.float64)
        )
        if len(world_reference) < 2:
            state.invalid_count += 1
            report_event(
                logging.WARNING,
                "INVALID_TRAJECTORY",
                "NavDP trajectory has no usable MPC reference",
                attempt=state.invalid_count,
            )
            continue
        state.invalid_count = 0
        state.trajectory = result
        state.trajectory_time = time.monotonic()
        state.mpc_reference = world_reference.copy()
        state.mpc_reference_version += 1
        LOGGER.info(
            "trajectory generation=%d points=%d local_goal=(%.3f, %.3f)",
            state.generation,
            len(state.trajectory),
            state.current_local_goal[0],
            state.current_local_goal[1],
        )


@dataclass(frozen=True)
class _NavDPSensorCycle:
    pose: Pose2D | None
    pose_time: float
    pose_history: list
    slam_map_xy: np.ndarray
    robot_history: np.ndarray


def _snapshot_sensor_cycle(
    sensors: _SharedSensors,
    *,
    now: float,
    slam_max_age_s: float,
) -> _NavDPSensorCycle:
    with sensors.lock:
        pose, pose_time = sensors.pose, sensors.pose_time
        pose_history = list(sensors.pose_history)
        slam_map_xy = sensors.slam_map_xy.copy()
        slam_map_time = sensors.slam_map_time
        robot_history = sensors.robot_history.copy()
    if slam_map_time <= 0.0 or now - slam_map_time > slam_max_age_s:
        # Do not keep presenting the last map as live after FAST-LIO or the
        # SensorGateway has stopped.
        slam_map_xy = np.empty((0, 2), dtype=np.float32)
        robot_history = np.empty((0, 2), dtype=np.float32)
    return _NavDPSensorCycle(
        pose=pose,
        pose_time=pose_time,
        pose_history=pose_history,
        slam_map_xy=slam_map_xy,
        robot_history=robot_history,
    )


def _schedule_navigation_inference(
    state: _NavDPRuntimeState,
    sensor_cycle: _NavDPSensorCycle,
    *,
    config: NavDPPlannerConfig,
    infer,
    send_status,
) -> None:
    pose = sensor_cycle.pose
    if state.mode != "nav_goal" or pose is None or state.world_goal is None:
        return
    state.current_local_goal = local_goal_from_world(state.world_goal, pose)
    if math.hypot(*state.current_local_goal) <= config.goal_tolerance_m:
        state.mode = "stop"
        send_status("reached", f"goal_within_{config.goal_tolerance_m:g}m")
        return
    if (
        state.inference_busy
        or not state.navdp_initialized
        or state.latest_rgb is None
        or state.latest_depth is None
    ):
        return
    state.inference_busy = True
    inference_pose = closest_timestamped_pose(
        sensor_cycle.pose_history, state.latest_camera_timestamp
    ) or pose
    threading.Thread(
        target=infer,
        args=(
            state.generation,
            state.segment_id,
            state.latest_rgb.copy(),
            state.latest_depth.copy(),
            state.current_local_goal,
            inference_pose,
        ),
        daemon=True,
    ).start()
