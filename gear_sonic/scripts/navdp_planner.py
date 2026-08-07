#!/usr/bin/env python3
"""Continuous NavDP trajectory executor with integrated hard safety guards."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import queue
import signal
import threading
import time
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_planner_message


COMMAND_TYPE = "sonic_navigation_command"
STATUS_TYPE = "sonic_navigation_status"
PLANNER_TYPE = "navila_reasan_velocity_command"


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class NavigationCommand:
    mode: Literal["manual_velocity", "nav_goal", "stop"]
    generation: int
    timestamp: float
    velocity: tuple[float, float, float] | None = None
    goal_base: tuple[float, float] | None = None
    target: str = ""
    target_type: str = ""
    confidence: float = 0.0


def build_navigation_message(
    *,
    mode: str,
    generation: int,
    timestamp: float | None = None,
    velocity: Sequence[float] | None = None,
    goal_base: Sequence[float] | None = None,
    target: str = "",
    target_type: str = "",
    confidence: float = 0.0,
) -> str:
    payload: dict[str, Any] = {
        "type": COMMAND_TYPE,
        "version": 1,
        "generation": int(generation),
        "mode": mode,
        "timestamp": time.time() if timestamp is None else float(timestamp),
    }
    if velocity is not None:
        payload["velocity"] = dict(zip(("vx", "vy", "wz"), map(float, velocity)))
    if goal_base is not None:
        payload["goal_base"] = {"x": float(goal_base[0]), "y": float(goal_base[1])}
        payload.update(
            target=str(target), target_type=str(target_type), confidence=float(confidence)
        )
    return json.dumps(payload)


def decode_navigation_message(message: str | bytes | Mapping[str, Any]) -> NavigationCommand:
    payload = json.loads(message) if isinstance(message, (str, bytes)) else dict(message)
    if payload.get("type") != COMMAND_TYPE or payload.get("version") != 1:
        raise ValueError("unsupported navigation command")
    mode = payload.get("mode")
    if mode not in {"manual_velocity", "nav_goal", "stop"}:
        raise ValueError("invalid navigation mode")
    velocity = payload.get("velocity")
    goal = payload.get("goal_base")
    return NavigationCommand(
        mode=mode,
        generation=int(payload["generation"]),
        timestamp=float(payload["timestamp"]),
        velocity=None
        if velocity is None
        else tuple(float(velocity[key]) for key in ("vx", "vy", "wz")),
        goal_base=None if goal is None else (float(goal["x"]), float(goal["y"])),
        target=str(payload.get("target", "")),
        target_type=str(payload.get("target_type", "")),
        confidence=float(payload.get("confidence", 0.0)),
    )


def camera_point_to_base(point: Sequence[float]) -> tuple[float, float, float]:
    """Optical x-right/y-down/z-forward to base x-forward/y-left/z-up."""
    x_cam, y_cam, z_cam = map(float, point)
    return z_cam, -x_cam, -y_cam + 0.40


def base_goal_to_world(goal: Sequence[float], pose: Pose2D) -> tuple[float, float]:
    x, y = map(float, goal)
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    return pose.x + c * x - s * y, pose.y + s * x + c * y


def local_goal_from_world(goal: Sequence[float], pose: Pose2D) -> tuple[float, float]:
    dx, dy = float(goal[0]) - pose.x, float(goal[1]) - pose.y
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    return c * dx + s * dy, -s * dx + c * dy


def local_trajectory_to_world(trajectory: np.ndarray, pose: Pose2D) -> np.ndarray:
    """Transform a robot-local x-forward/y-left trajectory into odometry XY."""
    points = np.asarray(trajectory, dtype=np.float32).reshape(-1, 2)
    if not len(points):
        return points.copy()
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
    return points @ rotation.T + np.array([pose.x, pose.y], dtype=np.float32)


def closest_timestamped_pose(
    history: Sequence[tuple[float, Pose2D]], timestamp: float
) -> Pose2D | None:
    """Return the Fast-LIO pose closest to a camera capture timestamp."""
    if not history:
        return None
    return min(history, key=lambda sample: abs(sample[0] - float(timestamp)))[1]


def update_slam_map(
    existing_xy: np.ndarray,
    incoming_xy: np.ndarray,
    *,
    center_xy: Sequence[float],
    voxel_size_m: float = 0.08,
    retain_radius_m: float = 20.0,
    max_points: int = 120_000,
) -> np.ndarray:
    """Merge absolute XY samples into a bounded, voxelized display-only map."""
    old = np.asarray(existing_xy, dtype=np.float32).reshape(-1, 2)
    new = np.asarray(incoming_xy, dtype=np.float32).reshape(-1, 2)
    combined = np.concatenate((new, old), axis=0)
    if not len(combined):
        return combined
    center = np.asarray(center_xy, dtype=np.float32)
    keep = np.isfinite(combined).all(axis=1)
    keep &= np.linalg.norm(combined - center, axis=1) <= float(retain_radius_m)
    combined = combined[keep]
    if not len(combined):
        return combined
    voxel = np.floor(combined / float(voxel_size_m)).astype(np.int64)
    _, indices = np.unique(voxel, axis=0, return_index=True)
    result = combined[np.sort(indices)]
    if len(result) > max_points:
        result = result[-max_points:]
    return result.astype(np.float32, copy=False)




def filter_livox_points(
    points: np.ndarray,
    *,
    min_range_m: float = 0.25,
    lidar_translation: Sequence[float] = (0.0002835, 0.00003, 0.41618),
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32).reshape(-1, 3).copy()
    ranges = np.linalg.norm(values, axis=1)
    values = values[
        np.isfinite(values).all(axis=1)
        & (ranges >= float(min_range_m))
    ]
    values += np.asarray(lidar_translation, dtype=np.float32)
    return values


def livox_custom_points_to_numpy(message: Any) -> np.ndarray:
    """Convert livox_ros_driver2/CustomMsg points to an Nx3 float array."""
    values = np.fromiter(
        (coordinate for point in message.points for coordinate in (point.x, point.y, point.z)),
        dtype=np.float32,
        count=3 * len(message.points),
    )
    return values.reshape(-1, 3)


def actor_ray_from_points(
    points_base: np.ndarray, *, bins: int = 180, max_range_m: float = 3.0
) -> np.ndarray:
    points = np.asarray(points_base, dtype=np.float32).reshape(-1, 3)
    rays = np.full(bins, float(max_range_m), dtype=np.float32)
    if len(points):
        angles = np.arctan2(points[:, 1], points[:, 0])
        indices = np.floor((angles + np.pi) * bins / (2.0 * np.pi)).astype(int) % bins
        ranges = np.linalg.norm(points, axis=1)
        valid = np.isfinite(ranges) & (ranges > 1.0e-6) & (ranges < max_range_m)
        np.minimum.at(rays, indices[valid], ranges[valid])
    return rays


_VIZ_SIZE = 500
_VIZ_CENTER = (250, 260)
_VIZ_RADIUS = 205


def _reasan_base_panel(title: str, subtitle: str) -> np.ndarray:
    import cv2

    panel = np.full((_VIZ_SIZE, _VIZ_SIZE, 3), 20, dtype=np.uint8)
    for fraction in (0.25, 0.5, 0.75, 1.0):
        cv2.circle(panel, _VIZ_CENTER, int(_VIZ_RADIUS * fraction), (65, 65, 65), 1, cv2.LINE_AA)
    cv2.line(panel, (_VIZ_CENTER[0], _VIZ_CENTER[1] - _VIZ_RADIUS), (_VIZ_CENTER[0], _VIZ_CENTER[1] + _VIZ_RADIUS), (50, 50, 50), 1)
    cv2.line(panel, (_VIZ_CENTER[0] - _VIZ_RADIUS, _VIZ_CENTER[1]), (_VIZ_CENTER[0] + _VIZ_RADIUS, _VIZ_CENTER[1]), (50, 50, 50), 1)
    cv2.putText(panel, title, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(panel, subtitle, (14, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 170, 170), 1, cv2.LINE_AA)
    cv2.circle(panel, _VIZ_CENTER, 7, (255, 210, 80), -1, cv2.LINE_AA)
    cv2.arrowedLine(panel, _VIZ_CENTER, (_VIZ_CENTER[0], _VIZ_CENTER[1] - 30), (255, 210, 80), 2, cv2.LINE_AA, tipLength=0.3)
    cv2.putText(panel, "+X", (_VIZ_CENTER[0] + 8, _VIZ_CENTER[1] - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 210, 80), 1)
    return panel


def _viz_pixels(xy: np.ndarray, radius_m: float) -> tuple[np.ndarray, np.ndarray]:
    scale = _VIZ_RADIUS / radius_m
    px = np.rint(_VIZ_CENTER[0] - xy[:, 1] * scale).astype(np.int32)
    py = np.rint(_VIZ_CENTER[1] - xy[:, 0] * scale).astype(np.int32)
    return px, py


def integrate_velocity_path(
    velocity: Sequence[float], *, horizon_s: float = 1.5, dt_s: float = 0.05
) -> np.ndarray:
    """Integrate a constant body-frame holonomic command from the current pose."""
    vx, vy, wz = map(float, velocity)
    steps = max(1, int(round(float(horizon_s) / float(dt_s))))
    path = np.zeros((steps + 1, 2), dtype=np.float32)
    yaw = 0.0
    for index in range(steps):
        c, s = math.cos(yaw), math.sin(yaw)
        path[index + 1, 0] = path[index, 0] + (c * vx - s * vy) * dt_s
        path[index + 1, 1] = path[index, 1] + (s * vx + c * vy) * dt_s
        yaw += wz * dt_s
    return path


def format_navigation_diagnostics(
    *,
    generation: int,
    confidence: float,
    local_goal: Sequence[float],
    world_goal: Sequence[float],
    trajectory: np.ndarray,
    velocity: Sequence[float],
) -> str:
    """Format one non-spamming diagnostic snapshot at a planner state boundary."""
    local_x, local_y = map(float, local_goal)
    world_x, world_y = map(float, world_goal)
    vx, vy, wz = map(float, velocity)
    navdp_path = np.asarray(trajectory, dtype=np.float32).reshape(-1, 2)
    sent_path = integrate_velocity_path((vx, vy, wz))
    array_options = dict(precision=3, suppress_small=True, max_line_width=160)
    return (
        f"[NavDP trajectory] generation={int(generation)} "
        f"lavira_confidence={float(confidence):.3f}\n"
        f"  local_goal=({local_x:.3f}, {local_y:.3f}) "
        f"world_goal=({world_x:.3f}, {world_y:.3f})\n"
        f"  navdp_local_trajectory={np.array2string(navdp_path, **array_options)}\n"
        f"  sent_velocity=({vx:.3f}, {vy:.3f}, {wz:.3f})\n"
        f"  sent_velocity_path_30x0.05s={np.array2string(sent_path, **array_options)}"
    )


def format_direction_chain_diagnostics(
    *,
    trajectory: np.ndarray,
    mpc_angular_velocity: float,
    fastlio_yaw: float,
    fastlio_yaw_delta: float,
    sonic_target_heading: float,
) -> str:
    """Expose the lateral/yaw signs at every navigation control boundary."""
    path = np.asarray(trajectory, dtype=np.float32).reshape(-1, 2)
    path_dy = float(path[1, 1] - path[0, 1]) if len(path) >= 2 else 0.0
    return (
        "[NavDP direction chain] "
        f"path_dy={path_dy:+.3f} "
        f"mpc_wz={float(mpc_angular_velocity):+.3f} "
        f"fastlio_yaw={float(fastlio_yaw):+.3f} "
        f"fastlio_dyaw={float(fastlio_yaw_delta):+.3f} "
        f"sonic_heading={float(sonic_target_heading):+.3f}"
    )


def format_actor_ray_control_text(velocity: Sequence[float]) -> str:
    """Format the body command actually sent to SONIC."""
    vx, vy, wz = map(float, velocity)
    return f"sent speed={math.hypot(vx, vy):.3f} m/s   wz={wz:+.3f} rad/s"


def actor_ray_velocity_arrow(
    velocity: Sequence[float],
    *,
    max_speed_mps: float = 0.30,
    max_length_px: int = 100,
    preview_s: float = 1.0,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Draw unicycle speed: vx controls length and wz controls deflection."""
    vx, _vy, wz = map(float, velocity)
    if abs(vx) <= 1.0e-9:
        return _VIZ_CENTER, _VIZ_CENTER
    length = min(abs(vx) / float(max_speed_mps), 1.0) * int(max_length_px)
    heading = wz * float(preview_s) + (math.pi if vx < 0.0 else 0.0)
    end = (
        int(round(_VIZ_CENTER[0] - math.sin(heading) * length)),
        int(round(_VIZ_CENTER[1] - math.cos(heading) * length)),
    )
    return _VIZ_CENTER, end


def render_actor_ray_panel(
    ranges_m: np.ndarray,
    *,
    max_range_m: float = 3.0,
    trajectory: np.ndarray | None = None,
    velocity: Sequence[float] | None = None,
) -> np.ndarray:
    """Render the exact REASEN-style normalized 180-ray polar panel."""
    import cv2

    ranges = np.clip(np.asarray(ranges_m, dtype=np.float32).reshape(-1), 0.0, max_range_m)
    if ranges.size != 180:
        raise ValueError("ActorRay visualization requires exactly 180 bins")
    normalized = ranges / max_range_m
    panel = _reasan_base_panel(
        "Normalized ActorRay received by planner", f"normalized radius: 1.0 == {max_range_m:.1f} m"
    )
    azimuth = np.deg2rad(-179.0 + 2.0 * np.arange(180, dtype=np.float32))
    endpoints = np.column_stack((np.cos(azimuth), np.sin(azimuth))) * normalized[:, None]
    px, py = _viz_pixels(endpoints, 1.0)
    for index in range(180):
        endpoint = (int(px[index]), int(py[index]))
        cv2.line(panel, _VIZ_CENTER, endpoint, (55, 72, 72), 1, cv2.LINE_AA)
        color = (60, 80, 235) if normalized[index] < 0.995 else (110, 110, 110)
        cv2.circle(panel, endpoint, 2, color, -1, cv2.LINE_AA)
    path = (
        np.empty((0, 2), dtype=np.float32)
        if trajectory is None
        else np.asarray(trajectory, dtype=np.float32).reshape(-1, 2)[:24]
    )
    if len(path):
        path_x, path_y = _viz_pixels(path, max_range_m)
        pixels = np.column_stack((path_x, path_y)).astype(np.int32)
        for point in pixels:
            if 0 <= point[0] < _VIZ_SIZE and 0 <= point[1] < _VIZ_SIZE:
                cv2.circle(panel, tuple(point), 3, (0, 255, 255), -1, cv2.LINE_AA)
    if velocity is not None:
        arrow_start, arrow_end = actor_ray_velocity_arrow(velocity)
        if arrow_end == arrow_start:
            cv2.circle(panel, arrow_start, 5, (255, 255, 0), -1, cv2.LINE_AA)
        else:
            cv2.arrowedLine(
                panel,
                arrow_start,
                arrow_end,
                (255, 255, 0),
                4,
                cv2.LINE_AA,
                tipLength=0.22,
            )
        cv2.putText(
            panel,
            format_actor_ray_control_text(velocity),
            (14, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (80, 255, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(panel, f"min={normalized.min():.3f}  mean={normalized.mean():.3f}  max={normalized.max():.3f}", (14, _VIZ_SIZE - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(panel, "red: occupied | yellow: NavDP | cyan arrow: sent velocity", (14, _VIZ_SIZE - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (210, 210, 210), 1, cv2.LINE_AA)
    return panel


class ActorRayVideoRecorder:
    """Atomically publish a finalized MP4 after the active recording closes."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        fps: float = 20.0,
        generation: int | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.fps = float(fps)
        self.generation = generation
        self.output_path: Path | None = None
        self.working_path: Path | None = None
        self._writer = None

    def write(self, panel: np.ndarray) -> None:
        frame = np.asarray(panel, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("ActorRay recording frame must be an HxWx3 BGR image")
        if self._writer is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            prefix = (
                "actorray_"
                if self.generation is None
                else f"actorray_g{self.generation:06d}_"
            )
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            stem = f"{prefix}{timestamp}_{time.time_ns() % 1_000_000_000:09d}"
            self.output_path = self.output_dir / f"{stem}.mp4"
            self.working_path = self.output_dir / f".{stem}.recording.mp4"
            height, width = frame.shape[:2]
            import imageio_ffmpeg

            writer = imageio_ffmpeg.write_frames(
                str(self.working_path),
                (width, height),
                pix_fmt_in="bgr24",
                pix_fmt_out="yuv420p",
                fps=self.fps,
                codec="libx264",
                quality=6,
                macro_block_size=2,
                output_params=["-movflags", "+faststart"],
            )
            writer.send(None)
            self._writer = writer
            print(f"[NavDP] recording ActorRay video: {self.working_path}", flush=True)
        self._writer.send(np.ascontiguousarray(frame))

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            if self.working_path is not None and self.output_path is not None:
                self.working_path.replace(self.output_path)
                print(f"[NavDP] finalized ActorRay video: {self.output_path}", flush=True)


class ActorRayRecordingSession:
    """Own exactly one ActorRay video for each active navigation generation."""

    def __init__(self, output_dir: str | Path, *, fps: float = 20.0) -> None:
        self.output_dir = Path(output_dir)
        self.fps = float(fps)
        self._recorder: ActorRayVideoRecorder | None = None
        self.output_path: Path | None = None

    def start(self, generation: int) -> None:
        self.stop()
        self._recorder = ActorRayVideoRecorder(
            self.output_dir,
            fps=self.fps,
            generation=generation,
        )
        self.output_path = None

    def write(self, panel: np.ndarray) -> None:
        if self._recorder is None:
            return
        self._recorder.write(panel)
        self.output_path = self._recorder.output_path

    def stop(self) -> None:
        if self._recorder is not None:
            self._recorder.close()
            self.output_path = self._recorder.output_path
            self._recorder = None


def install_shutdown_signal_handlers() -> None:
    """Route tmux/process termination signals through the normal cleanup path."""

    def request_shutdown(_signum, _frame) -> None:
        raise KeyboardInterrupt

    for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, request_shutdown)


def compose_reasan_navigation_view(
    points_base: np.ndarray,
    ranges_m: np.ndarray,
    trajectory: np.ndarray,
    *,
    max_range_m: float = 3.0,
    slam_map_xy: np.ndarray | None = None,
    pose: Pose2D | None = None,
    world_goal: Sequence[float] | None = None,
    robot_history: np.ndarray | None = None,
    world_span_m: float = 10.0,
    velocity: Sequence[float] | None = None,
) -> np.ndarray:
    import cv2

    physical = _reasan_base_panel("Physical occupancy", f"metric radius: {max_range_m:.1f} m")
    points = np.asarray(points_base, dtype=np.float32).reshape(-1, 3)
    if len(points):
        inside = np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) < max_range_m)
        xy = points[inside, :2]
        if len(xy):
            px, py = _viz_pixels(xy, max_range_m)
            valid = (px >= 0) & (px < _VIZ_SIZE) & (py >= 0) & (py < _VIZ_SIZE)
            occupancy = np.zeros_like(physical)
            occupancy[py[valid], px[valid]] = (0, 95, 255)
            occupancy = cv2.dilate(occupancy, np.ones((3, 3), np.uint8), iterations=1)
            mask = np.any(occupancy, axis=-1)
            physical[mask] = occupancy[mask]
    path = np.asarray(trajectory, dtype=np.float32).reshape(-1, 2)
    envelope_px = round(_VIZ_RADIUS * 0.3 / max_range_m)
    cv2.circle(physical, _VIZ_CENTER, envelope_px, (255, 220, 80), 2, cv2.LINE_AA)
    cv2.putText(physical, "orange: MID-360 | blue circle: 0.3 m envelope", (14, _VIZ_SIZE - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (0, 200, 255), 1, cv2.LINE_AA)
    world = render_slam_world_panel(
        np.empty((0, 2), dtype=np.float32) if slam_map_xy is None else slam_map_xy,
        pose=pose,
        world_goal=world_goal,
        trajectory_world=None,
        robot_history=robot_history,
        span_m=world_span_m,
    )
    return np.concatenate(
        (
            physical,
            render_actor_ray_panel(
                ranges_m,
                max_range_m=max_range_m,
                trajectory=path,
                velocity=velocity,
            ),
            world,
        ),
        axis=1,
    )


def render_slam_world_panel(
    slam_map_xy: np.ndarray,
    *,
    pose: Pose2D | None,
    world_goal: Sequence[float] | None,
    trajectory_world: np.ndarray | None,
    robot_history: np.ndarray | None,
    span_m: float = 10.0,
) -> np.ndarray:
    """Render a north-up FAST-LIO map while retaining absolute odometry coordinates."""
    import cv2

    panel = np.full((_VIZ_SIZE, _VIZ_SIZE, 3), 20, dtype=np.uint8)
    center = np.array([pose.x, pose.y] if pose is not None else [0.0, 0.0], dtype=np.float32)
    scale = (_VIZ_SIZE - 70) / float(span_m)

    def pixels(values: np.ndarray) -> np.ndarray:
        xy = np.asarray(values, dtype=np.float32).reshape(-1, 2)
        return np.column_stack(
            (np.rint(_VIZ_CENTER[0] - (xy[:, 1] - center[1]) * scale),
             np.rint(_VIZ_CENTER[1] - (xy[:, 0] - center[0]) * scale))
        ).astype(np.int32)

    for offset in np.arange(-span_m / 2, span_m / 2 + 0.01, 2.0):
        delta = int(round(offset * scale))
        cv2.line(panel, (_VIZ_CENTER[0] + delta, 45), (_VIZ_CENTER[0] + delta, 475), (43, 43, 43), 1)
        cv2.line(panel, (35, _VIZ_CENTER[1] + delta), (465, _VIZ_CENTER[1] + delta), (43, 43, 43), 1)

    map_xy = np.asarray(slam_map_xy, dtype=np.float32).reshape(-1, 2)
    if len(map_xy):
        relative = map_xy - center
        within_span = np.max(np.abs(relative), axis=1) <= float(span_m) / 2.0
        map_xy = map_xy[within_span]
    if len(map_xy):
        pix = pixels(map_xy)
        valid = (pix[:, 0] >= 0) & (pix[:, 0] < _VIZ_SIZE) & (pix[:, 1] >= 0) & (pix[:, 1] < _VIZ_SIZE)
        panel[pix[valid, 1], pix[valid, 0]] = (105, 105, 105)

    history = np.empty((0, 2), dtype=np.float32) if robot_history is None else np.asarray(robot_history, dtype=np.float32).reshape(-1, 2)
    if len(history) >= 2:
        cv2.polylines(panel, [pixels(history)], False, (245, 245, 245), 2, cv2.LINE_AA)
    predicted = np.empty((0, 2), dtype=np.float32) if trajectory_world is None else np.asarray(trajectory_world, dtype=np.float32).reshape(-1, 2)
    if len(predicted) >= 2:
        cv2.polylines(panel, [pixels(predicted)], False, (0, 255, 255), 3, cv2.LINE_AA)

    if world_goal is not None:
        goal = np.asarray(world_goal, dtype=np.float32).reshape(1, 2)
        goal_px = pixels(goal)[0]
        goal_px = np.clip(goal_px, [42, 55], [_VIZ_SIZE - 42, _VIZ_SIZE - 35])
        cv2.drawMarker(panel, tuple(goal_px), (60, 230, 60), cv2.MARKER_DIAMOND, 18, 3, cv2.LINE_AA)

    if pose is not None:
        heading = np.array([math.cos(pose.yaw), math.sin(pose.yaw)], dtype=np.float32)
        tip = pixels(np.array([[pose.x, pose.y], [pose.x, pose.y] + 0.7 * heading]))
        cv2.circle(panel, tuple(tip[0]), 7, (255, 255, 0), -1, cv2.LINE_AA)
        cv2.arrowedLine(panel, tuple(tip[0]), tuple(tip[1]), (255, 255, 0), 3, cv2.LINE_AA, tipLength=0.3)

    cv2.putText(panel, "FAST-LIO world map (north-up)", (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(panel, f"center odom=({center[0]:.2f}, {center[1]:.2f}) m", (14, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
    cv2.putText(panel, "gray map | cyan robot | green goal | white movement history", (14, _VIZ_SIZE - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (200, 200, 200), 1, cv2.LINE_AA)
    return panel


def depth_requires_stop(
    depth_m: np.ndarray, *, stop_distance_m: float = 0.10, min_area_pixels: int = 2000
) -> bool:
    mask = np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m < stop_distance_m)
    try:
        import cv2

        count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        return bool(count > 1 and int(stats[1:, cv2.CC_STAT_AREA].max()) > min_area_pixels)
    except ImportError:
        return int(mask.sum()) > min_area_pixels


def render_head_depth_panel(
    depth_m: np.ndarray, *, max_depth_m: float = 5.0
) -> np.ndarray:
    """Render head depth with a fixed metric color scale."""
    import cv2

    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    normalized[valid] = np.rint(
        np.clip(depth[valid] / float(max_depth_m), 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    panel = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
    panel[~valid] = 0
    return panel


def compose_head_rgbd_view(
    rgb: np.ndarray | None,
    depth_m: np.ndarray | None,
    *,
    panel_size: tuple[int, int] = (640, 480),
) -> np.ndarray:
    """Place head RGB and metric depth side by side with black fallbacks."""
    import cv2

    width, height = map(int, panel_size)
    black = np.zeros((height, width, 3), dtype=np.uint8)
    rgb_panel = black.copy()
    depth_panel = black.copy()
    if rgb is not None:
        value = np.asarray(rgb)
        if value.ndim == 3 and value.shape[2] == 3:
            rgb_panel = cv2.resize(
                cv2.cvtColor(value, cv2.COLOR_RGB2BGR),
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )
    if depth_m is not None:
        value = np.asarray(depth_m)
        if value.ndim == 2:
            depth_panel = cv2.resize(
                render_head_depth_panel(value),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
    return np.concatenate((rgb_panel, depth_panel), axis=1)


def apply_hard_safety(
    velocity: Sequence[float],
    points_base: np.ndarray,
    *,
    camera_stop: bool,
    stop_distance_m: float = 0.10,
    half_sector_deg: float = 67.5,
) -> tuple[float, float, float]:
    command = tuple(map(float, velocity))
    if camera_stop:
        return 0.0, 0.0, 0.0
    if math.hypot(command[0], command[1]) <= 1e-9:
        return command
    points = np.asarray(points_base, dtype=np.float32).reshape(-1, 3)
    if not len(points):
        return command
    rays = actor_ray_from_points(points)
    angles = np.linspace(-180.0, 180.0, len(rays), endpoint=False)
    motion_angle = math.degrees(math.atan2(command[1], command[0]))
    angle_error = (angles - motion_angle + 180.0) % 360.0 - 180.0
    if np.any((np.abs(angle_error) <= half_sector_deg) & (rays <= stop_distance_m)):
        return 0.0, 0.0, 0.0
    return command


def arc_target_to_body_velocity(
    *, speed: float, target_heading: float, actual_heading: float
) -> tuple[float, float, float]:
    """Express a world-heading arc target in the measured robot body frame."""
    error = math.remainder(
        float(target_heading) - float(actual_heading), 2.0 * math.pi
    )
    return (
        float(speed) * math.cos(error),
        float(speed) * math.sin(error),
        0.0,
    )


def world_path_tangent_heading(
    world_reference: np.ndarray,
    pose: Pose2D,
    *,
    lookahead_m: float = 0.15,
) -> float:
    """Return the world heading from the robot toward a nearby forward path point."""
    points = np.asarray(world_reference, dtype=np.float64).reshape(-1, 2)
    if not len(points):
        return float(pose.yaw)
    position = np.array([pose.x, pose.y], dtype=np.float64)
    closest = int(np.argmin(np.linalg.norm(points - position, axis=1)))
    forward = points[closest:]
    distances = np.linalg.norm(forward - position, axis=1)
    eligible = np.flatnonzero(distances >= float(lookahead_m))
    target = forward[int(eligible[0])] if len(eligible) else forward[-1]
    delta = target - position
    if float(np.linalg.norm(delta)) <= 1.0e-9:
        return float(pose.yaw)
    return math.atan2(float(delta[1]), float(delta[0]))


XNAVDP_G1_MPC_DEFAULTS = {
    "horizon_steps": 30,
    "desired_velocity": 0.3,
    "max_linear_velocity": 0.3,
    "max_angular_velocity": 0.8,
    "reference_gap": 3,
    "dt": 0.1,
    "reference_trajectory_length_m": 2.0,
    "minimum_desired_velocity": 0.0,
    "interpolation_ratio": 50,
    "lookahead_points": 10,
    "linear_control_weight": 5.0,
    "angular_control_weight": 0.02,
    "curvature_speed_gain": 0.15,
}


def xnavdp_control_to_body_velocity(
    linear_velocity: float, angular_velocity: float
) -> tuple[float, float, float]:
    """Map X-NavDP's unicycle control directly to the SONIC body twist."""
    return float(linear_velocity), 0.0, float(angular_velocity)


def sonic_heading_from_mpc(
    *,
    fastlio_yaw: float,
    fastlio_to_sonic_offset: float,
    mpc_angular_velocity: float,
    heading_preview_s: float,
) -> float:
    """Preview SONIC's absolute heading far enough to survive planner smoothing."""
    sonic_wz = xnavdp_control_to_body_velocity(0.0, mpc_angular_velocity)[2]
    return math.remainder(
        float(fastlio_yaw)
        + float(fastlio_to_sonic_offset)
        + sonic_wz * float(heading_preview_s),
        2.0 * math.pi,
    )


def xnavdp_adaptive_speed(
    trajectory_length_m: float,
    maximum_curvature: float,
    *,
    desired_velocity: float = 0.3,
    max_linear_velocity: float = 0.3,
    max_angular_velocity: float = 0.8,
    reference_trajectory_length_m: float = 2.0,
    minimum_desired_velocity: float = 0.0,
    curvature_speed_gain: float = 0.15,
) -> float:
    """Match X-NavDP's length- and curvature-limited reference speed."""
    length_scale = min(
        max(float(trajectory_length_m), 0.0) / float(reference_trajectory_length_m),
        1.0,
    )
    length_limit = float(desired_velocity) * length_scale
    curvature = max(float(maximum_curvature), 1.0e-6)
    curvature_limit = min(
        float(max_linear_velocity),
        float(curvature_speed_gain) * float(max_angular_velocity) / curvature,
    )
    return max(
        float(minimum_desired_velocity),
        min(float(max_linear_velocity), length_limit, curvature_limit),
    )


def prepare_internnav_world_reference(
    trajectory: np.ndarray,
    inference_pose: Pose2D,
    *,
    skip_points: int = 3,
    interpolation_ratio: int = 50,
) -> np.ndarray:
    """Match InternNav real-world preprocessing for a NavDP local trajectory."""
    points = np.asarray(trajectory, dtype=np.float64).reshape(-1, 2)
    points = points[np.isfinite(points).all(axis=1)]
    points = points[int(skip_points) :]
    if len(points) < 2:
        return np.empty((0, 2), dtype=np.float64)
    world = local_trajectory_to_world(points, inference_pose).astype(np.float64)
    ratio = max(1, int(interpolation_ratio))
    if ratio == 1:
        return world
    source = np.arange(len(world), dtype=np.float64)
    target = np.linspace(0.0, len(world) - 1, len(world) * ratio)
    return np.column_stack(
        (np.interp(target, source, world[:, 0]), np.interp(target, source, world[:, 1]))
    )


class InternNavMpcController:
    """X-NavDP G1 nonlinear MPC adapted to SONIC's 10 Hz planner."""

    def __init__(
        self,
        world_reference: np.ndarray,
        *,
        horizon_steps: int = 30,
        desired_velocity: float = 0.3,
        max_linear_velocity: float = 0.3,
        max_angular_velocity: float = 0.8,
        reference_gap: int = 3,
        dt: float = 0.1,
        reference_trajectory_length_m: float = 2.0,
        minimum_desired_velocity: float = 0.0,
        lookahead_points: int = 10,
        linear_control_weight: float = 5.0,
        angular_control_weight: float = 0.02,
        curvature_speed_gain: float = 0.15,
    ) -> None:
        import casadi as ca

        self.horizon_steps = int(horizon_steps)
        self.desired_velocity = float(desired_velocity)
        self.max_linear_velocity = float(max_linear_velocity)
        self.max_angular_velocity = float(max_angular_velocity)
        self.reference_gap = int(reference_gap)
        self.dt = float(dt)
        self.reference_trajectory_length_m = float(reference_trajectory_length_m)
        self.minimum_desired_velocity = float(minimum_desired_velocity)
        self.lookahead_points = int(lookahead_points)
        self.linear_control_weight = float(linear_control_weight)
        self.angular_control_weight = float(angular_control_weight)
        self.curvature_speed_gain = float(curvature_speed_gain)
        self.reference_count = self.horizon_steps // self.reference_gap + 1
        self.world_reference = np.asarray(world_reference, dtype=np.float64).reshape(-1, 2)
        if len(self.world_reference) < 2:
            raise ValueError("MPC world reference requires at least two points")

        optimizer = ca.Opti()
        controls = optimizer.variable(self.horizon_steps, 2)
        states = optimizer.variable(self.horizon_steps + 1, 3)
        initial_state = optimizer.parameter(3)
        reference_states = optimizer.parameter(3 * self.reference_count)
        optimizer.subject_to(states[0, :] == initial_state.T)
        for index in range(self.horizon_steps):
            x, y, yaw = states[index, 0], states[index, 1], states[index, 2]
            linear, angular = controls[index, 0], controls[index, 1]
            state = ca.vertcat(x, y, yaw)
            control = ca.vertcat(linear, angular)

            def dynamics(value):
                return ca.vertcat(
                    control[0] * ca.cos(value[2]),
                    control[0] * ca.sin(value[2]),
                    control[1],
                )

            k1 = dynamics(state)
            k2 = dynamics(state + 0.5 * self.dt * k1)
            k3 = dynamics(state + 0.5 * self.dt * k2)
            k4 = dynamics(state + self.dt * k3)
            next_state = (state + self.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6).T
            optimizer.subject_to(states[index + 1, :] == next_state)

        state_weight = np.diag([10.0, 10.0, 0.0])
        control_weight = np.diag(
            [self.linear_control_weight, self.angular_control_weight]
        )
        objective = 0
        for index in range(self.horizon_steps):
            objective += ca.mtimes(
                [controls[index, :], control_weight, controls[index, :].T]
            )
            if index % self.reference_gap == 0:
                ref_index = index // self.reference_gap
                error = states[index, :] - reference_states[
                    ref_index * 3 : ref_index * 3 + 3
                ].T
                objective += ca.mtimes([error, state_weight, error.T])
        terminal_error = states[-1, :] - reference_states[-3:].T
        objective += ca.mtimes([terminal_error, state_weight, terminal_error.T])
        optimizer.minimize(objective)
        optimizer.subject_to(
            optimizer.bounded(
                -self.max_linear_velocity,
                controls[:, 0],
                self.max_linear_velocity,
            )
        )
        optimizer.subject_to(
            optimizer.bounded(
                -self.max_angular_velocity,
                controls[:, 1],
                self.max_angular_velocity,
            )
        )
        optimizer.solver(
            "ipopt",
            {
                "ipopt.max_iter": 100,
                "ipopt.print_level": 0,
                "print_time": 0,
                "ipopt.acceptable_tol": 1.0e-8,
                "ipopt.acceptable_obj_change_tol": 1.0e-6,
            },
        )
        self._optimizer = optimizer
        self._controls = controls
        self._states = states
        self._initial_state = initial_state
        self._reference_states = reference_states
        self._last_controls: np.ndarray | None = None
        self._last_states: np.ndarray | None = None
        self._update_desired_velocity()

    def update_reference(self, world_reference: np.ndarray) -> None:
        reference = np.asarray(world_reference, dtype=np.float64).reshape(-1, 2)
        if len(reference) < 2:
            raise ValueError("MPC world reference requires at least two points")
        self.world_reference = reference
        self._update_desired_velocity()

    def _update_desired_velocity(self) -> None:
        delta = np.diff(self.world_reference, axis=0)
        trajectory_length = float(np.linalg.norm(delta, axis=1).sum())
        dx = np.gradient(self.world_reference[:, 0])
        dy = np.gradient(self.world_reference[:, 1])
        if len(dy):
            dy[0] = 0.0
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        denominator = np.maximum((dx * dx + dy * dy) ** 1.5, 1.0e-6)
        curvature = np.abs(dx * ddy - dy * ddx) / denominator
        curvature = np.convolve(curvature, np.ones(3) / 3.0, mode="same")
        maximum_curvature = float(np.max(curvature[:12])) if len(curvature) else 0.0
        self.desired_velocity = xnavdp_adaptive_speed(
            trajectory_length,
            maximum_curvature,
            desired_velocity=self.max_linear_velocity,
            max_linear_velocity=self.max_linear_velocity,
            max_angular_velocity=self.max_angular_velocity,
            reference_trajectory_length_m=self.reference_trajectory_length_m,
            minimum_desired_velocity=self.minimum_desired_velocity,
            curvature_speed_gain=self.curvature_speed_gain,
        )

    def _select_reference(self, pose: Pose2D) -> np.ndarray:
        distances = np.linalg.norm(
            self.world_reference - np.array([pose.x, pose.y]), axis=1
        )
        nearest = int(np.argmin(distances))
        start = min(nearest + self.lookahead_points, len(self.world_reference) - 1)
        remaining = self.world_reference[start:]
        arc = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(remaining, axis=0), axis=1)))
        )
        spacing = self.desired_velocity * self.reference_gap * self.dt
        indices = [int(np.searchsorted(arc, spacing * index, side="left")) for index in range(self.reference_count)]
        indices = np.clip(indices, 0, len(remaining) - 1)
        xy = remaining[indices]
        return np.column_stack((xy, np.zeros(self.reference_count)))

    def solve(self, pose: Pose2D) -> tuple[float, float]:
        reference = self._select_reference(pose)
        self._optimizer.set_value(
            self._initial_state, np.array([pose.x, pose.y, pose.yaw])
        )
        self._optimizer.set_value(self._reference_states, reference.reshape(-1))
        controls_guess = (
            np.zeros((self.horizon_steps, 2))
            if self._last_controls is None
            else self._last_controls
        )
        states_guess = (
            np.zeros((self.horizon_steps + 1, 3))
            if self._last_states is None
            else self._last_states
        )
        self._optimizer.set_initial(self._controls, controls_guess)
        self._optimizer.set_initial(self._states, states_guess)
        solution = self._optimizer.solve()
        self._last_controls = np.asarray(solution.value(self._controls))
        self._last_states = np.asarray(solution.value(self._states))
        return float(self._last_controls[0, 0]), float(self._last_controls[0, 1])


@dataclass(frozen=True)
class MpcSolveRequest:
    generation: int
    reference_version: int
    world_reference: np.ndarray
    pose: Pose2D


@dataclass(frozen=True)
class MpcSolveResult:
    generation: int
    reference_version: int
    control: tuple[float, float] | None
    completed_time: float
    elapsed_s: float
    error: str | None = None
    return_status: str = "unknown"


class AsyncMpcSolver:
    """Own the CasADi controller on a worker thread and keep only latest work."""

    def __init__(self, *, controller_factory=InternNavMpcController) -> None:
        self._controller_factory = controller_factory
        self._requests: queue.Queue[MpcSolveRequest | None] = queue.Queue(maxsize=1)
        self._results: queue.Queue[MpcSolveResult] = queue.Queue(maxsize=1)
        self._closed = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @staticmethod
    def _replace(queue_: queue.Queue, item) -> None:
        try:
            queue_.get_nowait()
        except queue.Empty:
            pass
        queue_.put_nowait(item)

    def submit(self, request: MpcSolveRequest) -> None:
        if not self._closed:
            self._replace(self._requests, request)

    def poll_latest(self) -> MpcSolveResult | None:
        latest = None
        while True:
            try:
                latest = self._results.get_nowait()
            except queue.Empty:
                return latest

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._replace(self._requests, None)
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        controller = None
        reference_version = -1
        while True:
            request = self._requests.get()
            if request is None:
                return
            started = time.perf_counter()
            control = None
            error = None
            return_status = "unknown"
            try:
                if controller is None:
                    controller = self._controller_factory(request.world_reference)
                elif request.reference_version != reference_version:
                    controller.update_reference(request.world_reference)
                reference_version = request.reference_version
                control = tuple(map(float, controller.solve(request.pose)))
                try:
                    return_status = controller._optimizer.stats().get(
                        "return_status", "unknown"
                    )
                except Exception:
                    return_status = "success"
            except Exception as exc:
                error = str(exc)
                try:
                    return_status = controller._optimizer.stats().get(
                        "return_status", "unknown"
                    )
                    controller._last_controls = None
                    controller._last_states = None
                except Exception:
                    pass
            self._replace(
                self._results,
                MpcSolveResult(
                    generation=request.generation,
                    reference_version=request.reference_version,
                    control=control,
                    completed_time=time.monotonic(),
                    elapsed_s=time.perf_counter() - started,
                    error=error,
                    return_status=return_status,
                ),
            )


class LatestMessageWorker:
    """Process expensive sensor messages off the ROS executor, newest first."""

    def __init__(self, processor) -> None:
        self._processor = processor
        self._messages: queue.Queue[Any | None] = queue.Queue(maxsize=1)
        self._closed = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, message: Any) -> None:
        if self._closed:
            return
        try:
            self._messages.get_nowait()
        except queue.Empty:
            pass
        self._messages.put_nowait(message)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._messages.get_nowait()
        except queue.Empty:
            pass
        self._messages.put_nowait(None)
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while True:
            message = self._messages.get()
            if message is None:
                return
            try:
                self._processor(message)
            except Exception as exc:
                print(f"[NavDP] sensor processing failed: {exc}", flush=True)


def fresh_mpc_control(
    control: tuple[float, float],
    *,
    result_time: float,
    now: float,
    timeout_s: float,
) -> tuple[float, float]:
    """Use a successful MPC result only for a bounded amount of time."""
    return control if now - result_time <= timeout_s else (0.0, 0.0)


def should_abort_nav_for_lidar(
    *,
    mode: str,
    before_safety: Sequence[float],
    after_safety: Sequence[float],
    camera_stop: bool,
) -> bool:
    """Return true only when MID-360 hard safety blocked active translation."""
    before = tuple(map(float, before_safety))
    after = tuple(map(float, after_safety))
    return bool(
        mode == "nav_goal"
        and not camera_stop
        and math.hypot(before[0], before[1]) > 1.0e-9
        and math.hypot(after[0], after[1]) <= 1.0e-9
    )


def should_abort_nav_for_zero_action(
    *,
    mode: str,
    selected_command: Sequence[float],
    command_available: bool,
) -> bool:
    """Treat an explicit zero MPC macro action as completion of this NAV task."""
    command = tuple(map(float, selected_command))
    return bool(
        mode == "nav_goal"
        and command_available
        and np.allclose(command, (0.0, 0.0, 0.0), atol=1.0e-9)
    )


def build_planner_velocity_message(
    velocity: Sequence[float], *, timestamp: float | None = None
) -> dict[str, Any]:
    vx, vy, wz = map(float, velocity)
    return {
        "type": PLANNER_TYPE,
        "version": 1,
        "source": "navdp",
        "status": "ok",
        "action": "continuous_nav",
        "duration_s": 0.05,
        "timestamp": time.time() if timestamp is None else float(timestamp),
        "velocity": {"vx": vx, "vy": vy, "wz": wz},
        "segments": [{"duration_s": 0.05, "vx": vx, "vy": vy, "wz": wz}],
    }


@dataclass
class SonicPlannerState:
    """Convert body-frame velocity commands into SONIC's planner wire protocol."""

    heading: float = 0.0

    def directional_message(
        self,
        *,
        speed: float,
        movement_heading: float,
        facing_heading: float,
    ) -> bytes:
        """Build a command with independent world movement and facing directions."""
        self.heading = math.remainder(float(facing_heading), 2.0 * math.pi)
        movement = (
            math.cos(float(movement_heading)),
            math.sin(float(movement_heading)),
            0.0,
        )
        facing = (math.cos(self.heading), math.sin(self.heading), 0.0)
        return build_planner_message(
            1,
            movement if speed > 1.0e-6 else (0.0, 0.0, 0.0),
            facing,
            speed=max(0.0, float(speed)),
            height=-1.0,
        )

    def arc_message(self, *, speed: float, heading: float) -> bytes:
        """Build a forward arc command with coincident movement and facing."""
        return self.directional_message(
            speed=speed,
            movement_heading=heading,
            facing_heading=heading,
        )

    def message(self, velocity: Sequence[float], dt: float = 0.0) -> bytes:
        vx, vy, wz = map(float, velocity)
        if dt:
            self.heading = math.remainder(
                self.heading + wz * float(dt), 2.0 * math.pi
            )
        cosine, sine = math.cos(self.heading), math.sin(self.heading)
        world_x = cosine * vx - sine * vy
        world_y = sine * vx + cosine * vy
        speed = math.hypot(world_x, world_y)
        movement = (
            (0.0, 0.0, 0.0)
            if speed < 1.0e-6
            else (world_x / speed, world_y / speed, 0.0)
        )
        facing = (cosine, sine, 0.0)
        return build_planner_message(1, movement, facing, speed=speed, height=-1.0)


@dataclass
class NavDPPlannerConfig:
    command_endpoint: str = "tcp://127.0.0.1:5558"
    status_endpoint: str = "tcp://*:5559"
    output_endpoint: str = "tcp://*:5563"
    navdp_server: str = "http://127.0.0.1:19999"
    camera_host: str = "192.168.123.164"
    camera_port: int = 5555
    lidar_topic: str = "/livox/lidar"
    odom_topic: str = "/Odometry_loc"
    slam_cloud_topic: str = "/cloud_registered_1"
    control_hz: float = 20.0
    mpc_hz: float = 10.0
    mpc_result_timeout_s: float = 0.30
    heading_preview_s: float = 0.6
    goal_tolerance_m: float = 1.0
    navdp_stop_threshold: float = -4.0
    radar_timeout_s: float = 0.75
    odom_timeout_s: float = 0.30
    trajectory_timeout_s: float = 2.50
    navdp_request_timeout_s: float = 10.0
    visualize: bool = True
    record_actorray: bool = False
    actorray_output_dir: str = "outputs/navdp"
    actorray_record_fps: float = 20.0


def _quaternion_yaw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _encode_navdp_frames(rgb_bgr: np.ndarray, depth_m: np.ndarray) -> tuple[bytes, bytes]:
    import cv2

    depth = np.asarray(depth_m, dtype=np.float32).copy()
    depth[(depth < 0.1) | (depth > 5.0) | ~np.isfinite(depth)] = 0.0
    ok_rgb, rgb_encoded = cv2.imencode(".jpg", rgb_bgr)
    ok_depth, depth_png = cv2.imencode(".png", np.rint(depth * 10000.0).astype(np.uint16))
    if not ok_rgb or not ok_depth:
        raise RuntimeError("failed to encode NavDP RGB-D")
    return rgb_encoded.tobytes(), depth_png.tobytes()


def _extract_camera_frame(message: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    images = message.get("images", {})
    info = message.get("camera_info", {}).get("ego_view", {})
    rgb = np.asarray(images.get("ego_view"))
    depth_raw = np.asarray(images.get("ego_view_depth"))
    if rgb.ndim != 3 or depth_raw.ndim != 2 or rgb.shape[:2] != depth_raw.shape:
        raise ValueError("fresh aligned ego-view RGB-D unavailable")
    scale = float(info.get("depth_scale_m", 0.001))
    return rgb.copy(), depth_raw.astype(np.float32) * scale, info


def point_plane_distances(
    points: np.ndarray,
    normal: np.ndarray,
    offset: float,
) -> np.ndarray:
    """Evaluate a 3-D plane without dispatching tiny products to threaded BLAS."""
    values = np.asarray(points)
    direction = np.asarray(normal)
    projection = (
        values[:, 0] * direction[0]
        + values[:, 1] * direction[1]
        + values[:, 2] * direction[2]
    )
    return np.abs(projection - float(offset))


def _remove_ground(points: np.ndarray) -> np.ndarray:
    """Conservative RANSAC plane removal using only low base-frame candidates."""
    values = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    candidate_indices = np.flatnonzero(values[:, 2] <= -0.25)
    candidates = values[candidate_indices]
    if len(candidates) < 50:
        return values
    rng = np.random.default_rng(0)
    best: np.ndarray | None = None
    best_count = 0
    for _ in range(64):
        sample = candidates[rng.choice(len(candidates), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-6:
            continue
        normal /= norm
        if abs(float(normal[2])) < math.cos(math.radians(25.0)):
            continue
        plane_offset = float(
            sample[0, 0] * normal[0]
            + sample[0, 1] * normal[1]
            + sample[0, 2] * normal[2]
        )
        distance = point_plane_distances(candidates, normal, plane_offset)
        inliers = distance <= 0.04
        count = int(inliers.sum())
        if count > best_count:
            best = inliers
            best_count = count
    if best is None or best_count < 50:
        return values
    keep = np.ones(len(values), dtype=bool)
    keep[candidate_indices[best]] = False
    return values[keep]


class _SharedSensors:
    def __init__(self) -> None:
        import threading

        self.lock = threading.Lock()
        self.pose: Pose2D | None = None
        self.pose_time = 0.0
        self.pose_history: deque[tuple[float, Pose2D]] = deque(maxlen=200)
        self.points = np.empty((0, 3), dtype=np.float32)
        self.points_time = 0.0
        self.slam_map_xy = np.empty((0, 2), dtype=np.float32)
        self.slam_map_time = 0.0
        self.robot_history = np.empty((0, 2), dtype=np.float32)


def _control_freshness_snapshot(
    sensors: _SharedSensors,
) -> tuple[Pose2D | None, float, np.ndarray, float]:
    """Read timestamps after potentially blocking MPC work, not before it."""
    with sensors.lock:
        return (
            sensors.pose,
            sensors.pose_time,
            sensors.points.copy(),
            sensors.points_time,
        )


def _start_ros(config: NavDPPlannerConfig, sensors: _SharedSensors):
    import rclpy
    from livox_ros_driver2.msg import CustomMsg
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py import point_cloud2

    rclpy.init(args=None)
    node = rclpy.create_node("sonic_navdp_planner")

    def odom_callback(message: Odometry) -> None:
        p, q = message.pose.pose.position, message.pose.pose.orientation
        pose = Pose2D(p.x, p.y, _quaternion_yaw(q.x, q.y, q.z, q.w))
        stamp = float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1.0e-9
        if stamp <= 0.0:
            stamp = time.time()
        with sensors.lock:
            sensors.pose = pose
            sensors.pose_time = time.monotonic()
            sensors.pose_history.append((stamp, pose))
            sample = np.array([[p.x, p.y]], dtype=np.float32)
            if not len(sensors.robot_history) or np.linalg.norm(sample[0] - sensors.robot_history[-1]) >= 0.02:
                sensors.robot_history = np.concatenate((sensors.robot_history, sample), axis=0)[-5000:]

    def process_lidar(message: CustomMsg) -> None:
        points = filter_livox_points(livox_custom_points_to_numpy(message))
        points = _remove_ground(points)
        with sensors.lock:
            sensors.points = points
            sensors.points_time = time.monotonic()

    def process_slam_cloud(message: PointCloud2) -> None:
        raw = np.asarray(
            point_cloud2.read_points(message, field_names=("x", "y", "z"), skip_nans=True)
        )
        if raw.dtype.names:
            xyz = np.column_stack(tuple(raw[name] for name in ("x", "y", "z"))).astype(np.float32)
        else:
            xyz = raw.astype(np.float32, copy=False).reshape(-1, 3)
        with sensors.lock:
            pose = sensors.pose
            previous = sensors.slam_map_xy
        if pose is None:
            return
        updated = update_slam_map(previous, xyz[:, :2], center_xy=(pose.x, pose.y))
        with sensors.lock:
            sensors.slam_map_xy = updated
            sensors.slam_map_time = time.monotonic()

    lidar_worker = LatestMessageWorker(process_lidar)
    slam_worker = LatestMessageWorker(process_slam_cloud)

    def lidar_callback(message: CustomMsg) -> None:
        lidar_worker.submit(message)

    def slam_cloud_callback(message: PointCloud2) -> None:
        slam_worker.submit(message)

    node.create_subscription(Odometry, config.odom_topic, odom_callback, 10)
    node.create_subscription(CustomMsg, config.lidar_topic, lidar_callback, 10)
    node.create_subscription(PointCloud2, config.slam_cloud_topic, slam_cloud_callback, 2)
    return rclpy, node, (lidar_worker, slam_worker)


def _navdp_request(
    server: str,
    rgb: np.ndarray,
    depth: np.ndarray,
    goal: tuple[float, float],
    *,
    pose: Pose2D | None = None,
    timeout: float = 10.0,
) -> np.ndarray:
    import requests

    rgb_encoded, depth_png = _encode_navdp_frames(rgb, depth)
    goal = (float(np.clip(goal[0], 0.0, 10.0)), float(np.clip(goal[1], -10.0, 10.0)))
    data = {"goal_data": json.dumps({"goal_x": [goal[0]], "goal_y": [goal[1]]})}
    if pose is not None:
        half_yaw = 0.5 * float(pose.yaw)
        data["state_data"] = json.dumps(
            {
                "robot_pos": [[float(pose.x), float(pose.y), 0.0]],
                "robot_quat": [
                    [0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)]
                ],
            }
        )
    response = requests.post(
        f"{server.rstrip('/')}/pointgoal_step",
        files={"image": ("image.jpg", rgb_encoded), "depth": ("depth.png", depth_png)},
        data=data,
        timeout=timeout,
    )
    response.raise_for_status()
    trajectory = np.asarray(response.json()["trajectory"], dtype=np.float32).squeeze()
    if trajectory.ndim != 2 or trajectory.shape[1] < 2 or len(trajectory) < 2:
        raise ValueError("invalid NavDP trajectory")
    return trajectory[:, :2]


def _reset_navdp(
    server: str,
    info: Mapping[str, Any],
    *,
    stop_threshold: float = -4.0,
) -> None:
    import requests

    intrinsic = [
        [float(info["fx"]), 0.0, float(info["cx"])],
        [0.0, float(info["fy"]), float(info["cy"])],
        [0.0, 0.0, 1.0],
    ]
    response = requests.post(
        f"{server.rstrip('/')}/navigator_reset",
        json={"intrinsic": intrinsic, "stop_threshold": [float(stop_threshold)], "batch_size": 1},
        timeout=30.0,
    )
    response.raise_for_status()


def shutdown_ros_context(rclpy_module) -> None:
    """Shut ROS down once when signal handling already began teardown."""
    if rclpy_module.ok():
        rclpy_module.shutdown()


def spin_ros(rclpy_module, node) -> None:
    """Spin until shutdown, suppressing only the expected invalid-context exit."""
    try:
        rclpy_module.spin(node)
    except Exception:
        if rclpy_module.ok():
            raise


def main(config: NavDPPlannerConfig) -> None:
    import cv2
    import threading
    import zmq
    from gear_sonic.camera.composed_camera import ComposedCameraClientSensor

    install_shutdown_signal_handlers()

    context = zmq.Context.instance()
    commands = context.socket(zmq.SUB)
    commands.connect(config.command_endpoint)
    commands.setsockopt_string(zmq.SUBSCRIBE, "")
    status = context.socket(zmq.PUB)
    status.bind(config.status_endpoint)
    output = context.socket(zmq.PUB)
    output.bind(config.output_endpoint)
    camera = ComposedCameraClientSensor(config.camera_host, config.camera_port)
    sensors = _SharedSensors()
    rclpy, node, sensor_workers = _start_ros(config, sensors)
    ros_thread = threading.Thread(target=spin_ros, args=(rclpy, node), daemon=True)
    ros_thread.start()
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
    mpc_target_heading: float | None = None
    mpc_movement_heading: float | None = None
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
    print(
        f"[NavDP] ROS lidar={config.lidar_topic} odom={config.odom_topic} "
        f"slam={config.slam_cloud_topic}"
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
                mpc_target_heading = None
                mpc_movement_heading = None
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

            frame = camera.read(blocking=False)
            if frame is not None:
                try:
                    latest_rgb, latest_depth, camera_info = _extract_camera_frame(frame)
                    timestamps = frame.get("timestamps", {})
                    latest_camera_timestamp = float(
                        timestamps.get(
                            "ego_view",
                            max(timestamps.values()) if timestamps else time.time(),
                        )
                    )
                except ValueError:
                    pass
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
                mpc_target_heading = None
                mpc_movement_heading = None
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

            if config.visualize:
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
                cv2.imshow("NavDP + MID360 + FAST-LIO world map", canvas)
                cv2.imshow(
                    "NavDP Head RGB-D",
                    compose_head_rgbd_view(latest_rgb, latest_depth),
                )
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            delay = period - (time.monotonic() - loop_started)
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        pass
    finally:
        mpc_solver.close()
        for worker in sensor_workers:
            worker.close()
        for _ in range(3):
            output.send(sonic_planner.message((0.0, 0.0, 0.0)))
        camera.close()
        shutdown_ros_context(rclpy)
        ros_thread.join(timeout=1.0)
        node.destroy_node()
        commands.close(0)
        status.close(0)
        output.close(0)
        if actorray_recording is not None:
            actorray_recording.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import tyro

    main(tyro.cli(NavDPPlannerConfig))
