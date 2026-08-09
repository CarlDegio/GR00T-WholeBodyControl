"""NavDP sensor filtering, diagnostics, and visualization helpers."""

from __future__ import annotations

import math
from pathlib import Path
import signal
import time
from typing import Sequence

import numpy as np

from gear_sonic.navdp.navigation import Pose2D


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
