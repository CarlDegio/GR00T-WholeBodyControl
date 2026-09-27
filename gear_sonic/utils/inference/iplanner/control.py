"""LaViRA G1 truncation, replanning and pure pursuit on the SONIC runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import math
import queue
import threading
import time

import numpy as np

from gear_sonic.runtime.profile import load_component_config
from gear_sonic.utils.inference.navdp.navigation import (
    closest_timestamped_pose,
    local_goal_from_world,
    local_trajectory_to_world,
)


@dataclass(frozen=True)
class _LegacyIPlannerNavigationConfig:
    target_speed: float
    max_forward_speed: float
    max_yaw_speed: float
    lookahead_dist: float
    goal_tolerance: float
    safe_distance: float
    replan_interval: float
    max_execution_time: float
    yaw_bias: float

    def __post_init__(self):
        for name in ("target_speed", "max_forward_speed", "max_yaw_speed", "lookahead_dist",
                     "goal_tolerance", "replan_interval", "max_execution_time"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"iPlanner {name} must be finite and positive")
        if not math.isfinite(self.safe_distance) or self.safe_distance < 0:
            raise ValueError("iPlanner safe_distance must be finite and nonnegative")
        if not math.isfinite(self.yaw_bias):
            raise ValueError("iPlanner yaw_bias must be finite")


@dataclass(frozen=True)
class IPlannerNavigationConfig(_LegacyIPlannerNavigationConfig):
    min_forward_speed: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        if (not math.isfinite(self.min_forward_speed)
                or not 0 <= self.min_forward_speed <= self.max_forward_speed):
            raise ValueError("iPlanner min_forward_speed must be between zero and max_forward_speed")


def load_navigation_config(profile):
    # Frozen runtimes made before the configurable floor retain the original
    # 0.1 m/s minimum, with all other fields still validated by the strict loader.
    config_type = (IPlannerNavigationConfig if "min_forward_speed" in profile.component("iplanner_navigation")
                   else _LegacyIPlannerNavigationConfig)
    config = load_component_config(
        config_type, "iplanner_navigation", profile.source_files[0],
        overlays=profile.source_files[1:],
    )
    return config if isinstance(config, IPlannerNavigationConfig) else IPlannerNavigationConfig(**vars(config))


def truncate_trajectory(path, safe_distance):
    """Apply G1 VLNTask's safety buffer to the first trajectory of a segment."""
    path = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    arc = np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))
    if not len(arc) or arc[-1] <= safe_distance:
        return np.empty((0, 2))
    cutoff = int(np.searchsorted(arc, arc[-1] - safe_distance, side="left")) + 1
    return path[:cutoff + 1].copy()


def pure_pursuit(path, goal_distance, config, *, current_index=0):
    """Return (vx, vy, wz) and the original G1 forward-search waypoint index."""
    path = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    if not len(path) or not np.isfinite(path).all() or not math.isfinite(goal_distance):
        return (0.0, 0.0, 0.0), 0
    if goal_distance < config.goal_tolerance:
        return (0.0, 0.0, 0.0), current_index
    index = len(path) - 1
    for candidate in range(min(current_index, len(path) - 1), len(path)):
        if np.linalg.norm(path[candidate]) > config.lookahead_dist:
            index = candidate
            break
    dx, dy = path[index]
    distance = math.hypot(dx, dy)
    # The G1 upstream disables steering inside 2 m. On this integration that
    # also discards a known lateral goal error after a coarse panorama turn.
    # Continue following the path until the existing arrival tolerance stops it.
    alpha = math.atan2(dy, dx)
    yaw_speed = float(np.clip(
        2.0 * config.target_speed * math.sin(alpha) / max(distance, 0.1),
        -config.max_yaw_speed, config.max_yaw_speed,
    ))
    speed = min(config.max_forward_speed, max(
        config.min_forward_speed, config.target_speed * (1.0 - abs(alpha) / math.pi),
    ))
    return (speed, 0.0, yaw_speed + config.yaw_bias), index


class IPlannerNavigation:
    def __init__(self, config):
        self.config = config
        self.key = None
        self.world_path = np.empty((0, 2))
        self.first_plan = True
        self.current_index = 0
        self.started_at = 0.0
        self.next_inference = 0.0
        self.requested_world_goal = None
        self.last_plan_event_at = -math.inf
        self.last_control_event_at = -math.inf

    def sync_command(self, state, *, now):
        key = (state.generation, state.skill_id, state.segment_id, state.mode)
        if key == self.key and not (len(self.world_path) and state.trajectory_time == 0.0):
            return
        self.key = key
        self.world_path = np.empty((0, 2))
        self.first_plan = True
        self.current_index = 0
        self.started_at = now
        self.next_inference = now
        self.requested_world_goal = state.world_goal
        self.last_plan_event_at = -math.inf
        self.last_control_event_at = -math.inf

    def consume_results(self, state, results, *, send_metrics, send_status, report_event):
        while True:
            try:
                generation, segment, path, pose, error, elapsed = results.get_nowait()
            except queue.Empty:
                break
            if generation != state.generation or segment != state.segment_id or state.mode != "nav_goal":
                continue
            send_metrics({"policy_inference": elapsed * 1000.0})
            if not error and path is not None:
                path = np.asarray(path, dtype=np.float64)
                if (path.ndim != 2 or path.shape[1:] != (2,) or len(path) < 2
                        or not np.isfinite(path).all()
                        or np.linalg.norm(np.diff(path, axis=0), axis=1).sum() <= 1e-6):
                    error = "iPlanner returned an invalid or zero-length trajectory"
                elif self.first_plan:
                    path = truncate_trajectory(path, self.config.safe_distance)
                    if len(path) < 2:
                        error = "iPlanner trajectory does not extend beyond the configured safety buffer"
            if error or path is None or pose is None:
                state.invalid_count += 1
                report_event(logging.WARNING, "INFERENCE_FAILED", "iPlanner inference failed",
                             attempt=state.invalid_count, error=error)
                if state.invalid_count >= 3:
                    state.mode = "stop"
                    send_status("failed", "three_invalid_trajectories")
                continue
            initial_plan = self.first_plan
            self.world_path = local_trajectory_to_world(path, pose)
            # A predicted path can stop short or bend away from the requested
            # waypoint. Its endpoint must not silently become the arrival goal.
            # Keep the VA waypoint fixed for replanning and distance checks.
            self.first_plan = False
            self.current_index = 0
            state.trajectory = np.asarray(path, dtype=np.float32)
            state.trajectory_time = time.monotonic()
            state.invalid_count = 0
            if state.trajectory_time - self.last_plan_event_at >= 1.0:
                self.last_plan_event_at = state.trajectory_time
                report_event(
                    logging.INFO, "IPLANNER_PLAN_ACCEPTED", "iPlanner trajectory and coordinate reference",
                    generation=state.generation, skill_id=state.skill_id, segment_id=state.segment_id,
                    initial_plan=initial_plan, inference_pose=[pose.x, pose.y, pose.yaw],
                    requested_world_goal=self.requested_world_goal, execution_world_goal=state.world_goal,
                    trajectory_local=state.trajectory.tolist(), trajectory_world=self.world_path.tolist(),
                )

    def schedule(self, state, cycle, *, config, infer, send_status):
        if state.mode != "nav_goal" or cycle.pose is None or state.world_goal is None:
            return
        now = time.monotonic()
        state.current_local_goal = local_goal_from_world(state.world_goal, cycle.pose)
        if math.hypot(*state.current_local_goal) < self.config.goal_tolerance:
            state.mode = "stop"
            send_status("reached", "iplanner_goal_tolerance")
            return
        if now - self.started_at > self.config.max_execution_time:
            state.mode = "stop"
            send_status("failed", "iplanner_execution_timeout")
            return
        if (now < self.next_inference or state.inference_busy or not state.navdp_initialized
                or state.latest_rgb is None or state.latest_depth is None):
            return
        state.inference_busy = True
        self.next_inference = now + self.config.replan_interval
        pose = closest_timestamped_pose(cycle.pose_history, state.latest_camera_timestamp) or cycle.pose
        threading.Thread(
            target=infer,
            args=(state.generation, state.segment_id, state.latest_rgb.copy(), state.latest_depth.copy(),
                  local_goal_from_world(state.world_goal, pose), pose),
            daemon=True,
        ).start()

    def control(self, state, *, pose, now, config, report_event=None):
        if (state.mode != "nav_goal" or pose is None or state.world_goal is None
                or not len(self.world_path) or now - state.trajectory_time > config.trajectory_timeout_s):
            return (0.0, 0.0, 0.0)
        # Preserve Groot's camera/odometry alignment between replans while
        # applying G1's point selection and velocity equations in the body frame.
        local_path = np.asarray([local_goal_from_world(point, pose) for point in self.world_path])
        goal_distance = math.hypot(*local_goal_from_world(state.world_goal, pose))
        velocity, self.current_index = pure_pursuit(
            local_path, goal_distance, self.config, current_index=self.current_index,
        )
        state.heading_target_yaw = math.remainder(pose.yaw + velocity[2] * config.heading_preview_s, 2 * math.pi)
        if report_event is not None and now - self.last_control_event_at >= 1.0:
            self.last_control_event_at = now
            orientation = state.latest_orientation
            report_event(
                logging.INFO, "IPLANNER_CONTROL", "iPlanner path tracking in the current body frame",
                generation=state.generation, skill_id=state.skill_id, segment_id=state.segment_id,
                pose=[pose.x, pose.y, pose.yaw], local_goal=local_goal_from_world(state.world_goal, pose),
                lookahead=local_path[self.current_index].tolist(), goal_distance_m=goal_distance,
                goal_tolerance_m=self.config.goal_tolerance,
                path_endpoint_distance_m=float(np.linalg.norm(local_path[-1])),
                velocity=velocity, heading_target_rad=state.heading_target_yaw,
                heading_reference_rad=state.heading_reference_yaw,
                camera_timestamp_s=state.latest_camera_timestamp,
                sonic_orientation=(None if orientation is None else {
                    **asdict(orientation),
                    "telemetry_age_s": max(0.0, now - orientation.emitted_at_monotonic_s),
                }),
            )
        return velocity
