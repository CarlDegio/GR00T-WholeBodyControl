"""NavDP sensor filtering, diagnostics, and visualization helpers."""

from __future__ import annotations

import math
import signal
from typing import Sequence

import numpy as np

from gear_sonic.utils.inference.navdp.navigation import Pose2D

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
    cv2.line(
        panel,
        (_VIZ_CENTER[0], _VIZ_CENTER[1] - _VIZ_RADIUS),
        (_VIZ_CENTER[0], _VIZ_CENTER[1] + _VIZ_RADIUS),
        (50, 50, 50),
        1,
    )
    cv2.line(
        panel,
        (_VIZ_CENTER[0] - _VIZ_RADIUS, _VIZ_CENTER[1]),
        (_VIZ_CENTER[0] + _VIZ_RADIUS, _VIZ_CENTER[1]),
        (50, 50, 50),
        1,
    )
    cv2.putText(panel, title, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(panel, subtitle, (14, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 170, 170), 1, cv2.LINE_AA)
    cv2.circle(panel, _VIZ_CENTER, 7, (255, 210, 80), -1, cv2.LINE_AA)
    cv2.arrowedLine(
        panel,
        _VIZ_CENTER,
        (_VIZ_CENTER[0], _VIZ_CENTER[1] - 30),
        (255, 210, 80),
        2,
        cv2.LINE_AA,
        tipLength=0.3,
    )
    cv2.putText(
        panel,
        "+X",
        (_VIZ_CENTER[0] + 8, _VIZ_CENTER[1] - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 210, 80),
        1,
    )
    return panel


def _viz_pixels(xy: np.ndarray, radius_m: float) -> tuple[np.ndarray, np.ndarray]:
    scale = _VIZ_RADIUS / radius_m
    px = np.rint(_VIZ_CENTER[0] - xy[:, 1] * scale).astype(np.int32)
    py = np.rint(_VIZ_CENTER[1] - xy[:, 0] * scale).astype(np.int32)
    return px, py


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
    cv2.putText(
        panel,
        f"min={normalized.min():.3f}  mean={normalized.mean():.3f}  max={normalized.max():.3f}",
        (14, _VIZ_SIZE - 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "red: occupied | yellow: NavDP | cyan arrow: sent velocity",
        (14, _VIZ_SIZE - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    return panel


def install_shutdown_signal_handlers() -> None:
    """Route tmux/process termination signals through the normal cleanup path."""

    def request_shutdown(_signum, _frame) -> None:
        raise KeyboardInterrupt

    for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, request_shutdown)


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

    history = (
        np.empty((0, 2), dtype=np.float32)
        if robot_history is None
        else np.asarray(robot_history, dtype=np.float32).reshape(-1, 2)
    )
    if len(history) >= 2:
        cv2.polylines(panel, [pixels(history)], False, (245, 245, 245), 2, cv2.LINE_AA)
    predicted = (
        np.empty((0, 2), dtype=np.float32)
        if trajectory_world is None
        else np.asarray(trajectory_world, dtype=np.float32).reshape(-1, 2)
    )
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

    cv2.putText(
        panel,
        "FAST-LIO world map (north-up)",
        (14, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"center odom=({center[0]:.2f}, {center[1]:.2f}) m",
        (14, 49),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (170, 170, 170),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "gray map | cyan robot | green goal | white movement history",
        (14, _VIZ_SIZE - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return panel
