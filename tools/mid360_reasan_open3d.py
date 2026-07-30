#!/usr/bin/env python3
"""Visualize G1 MID-360 points and REASEN rays in the robot body frame."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import json
import sys
import time
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import view_mid360_open3d as mid360  # noqa: E402


def rpy_matrix_deg(rpy_deg: tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad(np.asarray(rpy_deg, dtype=np.float64))
    return mid360.rpy_to_rotation_matrix(float(roll), float(pitch), float(yaw))


def transform_sensor_to_body(
    points_sensor: np.ndarray,
    translation: tuple[float, float, float],
    rpy_deg: tuple[float, float, float],
) -> np.ndarray:
    """Apply torso_link <- Mid360 rigid transform to row-vector points."""
    if points_sensor.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    rotation = rpy_matrix_deg(rpy_deg)
    result = points_sensor.astype(np.float64, copy=False) @ rotation.T
    result += np.asarray(translation, dtype=np.float64)
    return result.astype(np.float32)


def filter_sensor_min_range(points_sensor: np.ndarray, min_range: float) -> np.ndarray:
    """Remove the MID-360 near field before applying any sensor extrinsics."""
    if points_sensor.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    points = np.asarray(points_sensor, dtype=np.float32)
    distance_sensor = np.linalg.norm(points, axis=1)
    valid = np.isfinite(points).all(axis=1) & np.isfinite(distance_sensor) & (distance_sensor >= min_range)
    return points[valid]


def filter_ground_plane(
    points_body: np.ndarray,
    *,
    distance_threshold: float,
    candidate_max_z: float,
    max_tilt_deg: float,
    iterations: int = 64,
) -> tuple[np.ndarray, int]:
    """Remove a near-horizontal RANSAC plane from points below the torso."""
    if len(points_body) < 3:
        return points_body, 0
    candidate_indices = np.flatnonzero(np.isfinite(points_body).all(axis=1) & (points_body[:, 2] <= candidate_max_z))
    if candidate_indices.size < 50:
        return points_body, 0

    candidates = points_body[candidate_indices].astype(np.float64, copy=False)
    rng = np.random.default_rng(0)
    best_mask = None
    best_count = 0
    min_vertical_component = float(np.cos(np.deg2rad(max_tilt_deg)))
    for _ in range(iterations):
        sample = candidates[rng.choice(len(candidates), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm < 1.0e-8:
            continue
        normal /= norm
        if abs(float(normal[2])) < min_vertical_component:
            continue
        offset = -float(np.dot(normal, sample[0]))
        inliers = np.abs(candidates @ normal + offset) <= distance_threshold
        count = int(np.count_nonzero(inliers))
        if count > best_count:
            best_mask = inliers
            best_count = count

    if best_mask is None or best_count < 50:
        return points_body, 0
    keep = np.ones(len(points_body), dtype=bool)
    keep[candidate_indices[best_mask]] = False
    return points_body[keep], best_count


def direct_actor_profile_from_points(points_body: np.ndarray, max_range: float) -> np.ndarray:
    """Take the nearest 3-D range per 2-degree azimuth without vertical cropping."""
    if max_range <= 0.0:
        raise ValueError("max_range must be positive")
    profile = np.full(180, max_range, dtype=np.float32)
    if points_body.size == 0:
        return profile / max_range

    xyz = np.asarray(points_body, dtype=np.float32)
    distance = np.linalg.norm(xyz, axis=1)
    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(distance) & (distance > 1.0e-6) & (distance < max_range)
    if not np.any(finite):
        return profile / max_range

    xyz = xyz[finite]
    distance = distance[finite]
    phi = np.rad2deg(np.arctan2(xyz[:, 1], xyz[:, 0]))
    phi_idx = np.floor((phi + 180.0) / 2.0).astype(np.int64)
    np.clip(phi_idx, 0, 179, out=phi_idx)
    np.minimum.at(profile, phi_idx, distance)
    return np.clip(profile / max_range, 0.0, 1.0)


def body_envelope_radius_px(body_radius_m: float, max_range_m: float, draw_radius_px: int) -> int:
    """Convert a metric body envelope to the 2-D occupancy-panel scale."""
    if body_radius_m < 0.0 or max_range_m <= 0.0 or draw_radius_px <= 0:
        raise ValueError("body radius must be non-negative and panel scale must be positive")
    return round(body_radius_m / max_range_m * draw_radius_px)


class RayMedianFilter:
    """Per-azimuth median over a short history of 180-D ray profiles."""

    def __init__(self, window: int) -> None:
        if window < 1 or window % 2 == 0:
            raise ValueError("Ray median window must be a positive odd integer")
        self.history: deque[np.ndarray] = deque(maxlen=window)

    def update(self, rays: np.ndarray) -> np.ndarray:
        values = np.asarray(rays, dtype=np.float32).reshape(180)
        self.history.append(values.copy())
        return np.median(np.stack(self.history, axis=0), axis=0).astype(np.float32)


def build_actor_ray_message(
    rays_normalized: np.ndarray,
    *,
    sequence: int,
    max_range: float,
) -> dict[str, Any]:
    rays = np.clip(np.asarray(rays_normalized, dtype=np.float32).reshape(-1), 0.0, 1.0)
    if rays.size != 180:
        raise ValueError(f"Expected 180 ActorRay values, got {rays.size}")
    message = {
        "type": "reasan_actor_ray",
        "version": 1,
        "timestamp_ns": time.time_ns(),
        "sequence": int(sequence),
        "frame_id": "torso_link",
        "source": "direct_all_vertical_median",
        "count": 180,
        "angle_min_deg": -179.0,
        "angle_max_deg": 179.0,
        "angle_increment_deg": 2.0,
        "range_max_m": float(max_range),
        "normalized": rays.tolist(),
        "ranges_m": (rays * max_range).tolist(),
    }
    return message


class ActorRayZmqPublisher:
    def __init__(self, endpoint: str) -> None:
        try:
            import zmq
        except Exception as exc:
            raise SystemExit("ZMQ publishing requires pyzmq: python3 -m pip install pyzmq") from exc
        self.zmq = zmq
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.SNDHWM, 1)
        self.socket.bind(endpoint)
        self.endpoint = endpoint
        print(f"[zmq] publishing 180-D ActorRay JSON on {endpoint}")

    def publish(self, message: dict[str, Any]) -> bool:
        try:
            self.socket.send_string(json.dumps(message, separators=(",", ":")), flags=self.zmq.NOBLOCK)
            return True
        except self.zmq.Again:
            return False

    def close(self) -> None:
        self.socket.close(linger=0)


class ReasanOccupancyViewer:
    """OpenCV bird's-eye view matching play_g1_filter.py's occupancy layout."""

    PANEL_SIZE = 500
    CENTER = (250, 260)
    DRAW_RADIUS_PX = 205
    BODY_ENVELOPE_RADIUS_M = 0.3
    WINDOW_NAME = "G1 REASEN | physical points (m) vs normalized ActorRay"

    def __init__(
        self,
        max_range: float,
        *,
        create_window: bool = True,
        window_left: int = 920,
    ) -> None:
        try:
            import cv2
        except Exception as exc:
            raise SystemExit("2-D visualization requires OpenCV: python3 -m pip install opencv-python") from exc
        self.cv2 = cv2
        self.max_range = max_range
        self.window_created = create_window
        if create_window:
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.WINDOW_NAME, self.PANEL_SIZE * 2, self.PANEL_SIZE)
            cv2.moveWindow(self.WINDOW_NAME, window_left, 30)

    def _base_panel(self, title: str, radius_label: str) -> np.ndarray:
        cv2 = self.cv2
        panel = np.full((self.PANEL_SIZE, self.PANEL_SIZE, 3), 20, dtype=np.uint8)
        for fraction in (0.25, 0.5, 0.75, 1.0):
            cv2.circle(panel, self.CENTER, int(self.DRAW_RADIUS_PX * fraction), (65, 65, 65), 1, cv2.LINE_AA)
        cv2.line(
            panel,
            (self.CENTER[0], self.CENTER[1] - self.DRAW_RADIUS_PX),
            (self.CENTER[0], self.CENTER[1] + self.DRAW_RADIUS_PX),
            (50, 50, 50),
            1,
        )
        cv2.line(
            panel,
            (self.CENTER[0] - self.DRAW_RADIUS_PX, self.CENTER[1]),
            (self.CENTER[0] + self.DRAW_RADIUS_PX, self.CENTER[1]),
            (50, 50, 50),
            1,
        )
        cv2.putText(panel, title, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(panel, radius_label, (14, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 170, 170), 1, cv2.LINE_AA)
        envelope_radius = body_envelope_radius_px(
            self.BODY_ENVELOPE_RADIUS_M, self.max_range, self.DRAW_RADIUS_PX
        )
        cv2.circle(panel, self.CENTER, envelope_radius, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(
            panel,
            "0.3 m body envelope",
            (self.CENTER[0] + envelope_radius + 5, self.CENTER[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.circle(panel, self.CENTER, 7, (255, 210, 80), -1, cv2.LINE_AA)
        cv2.arrowedLine(
            panel,
            self.CENTER,
            (self.CENTER[0], self.CENTER[1] - 30),
            (255, 210, 80),
            2,
            cv2.LINE_AA,
            tipLength=0.3,
        )
        cv2.putText(
            panel,
            "+X",
            (self.CENTER[0] + 8, self.CENTER[1] - 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 210, 80),
            1,
        )
        return panel

    def _xy_to_pixels(self, xy: np.ndarray, physical_radius: float) -> tuple[np.ndarray, np.ndarray]:
        scale = self.DRAW_RADIUS_PX / physical_radius
        # Same convention as G1 Play: body +X=image up and body +Y=image left.
        px = np.rint(self.CENTER[0] - xy[:, 1] * scale).astype(np.int32)
        py = np.rint(self.CENTER[1] - xy[:, 0] * scale).astype(np.int32)
        return px, py

    def physical_panel(self, points_body: np.ndarray) -> np.ndarray:
        cv2 = self.cv2
        panel = self._base_panel("Physical occupancy from MID-360", f"metric radius: {self.max_range:.1f} m")
        if points_body.size:
            finite = np.isfinite(points_body).all(axis=1)
            distance = np.linalg.norm(points_body, axis=1)
            inside = finite & (distance < self.max_range)
            xy = points_body[inside, :2]
            if len(xy):
                px, py = self._xy_to_pixels(xy, self.max_range)
                valid = (px >= 0) & (px < self.PANEL_SIZE) & (py >= 0) & (py < self.PANEL_SIZE)
                occupancy = np.zeros_like(panel)
                occupancy[py[valid], px[valid]] = (0, 95, 255)
                occupancy = cv2.dilate(occupancy, np.ones((3, 3), dtype=np.uint8), iterations=1)
                mask = np.any(occupancy != 0, axis=-1)
                panel[mask] = occupancy[mask]
        cv2.putText(
            panel,
            "orange: raw MID-360 points within clipping radius",
            (14, self.PANEL_SIZE - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (0, 165, 255),
            1,
            cv2.LINE_AA,
        )
        return panel

    def actor_panel(self, actor_rays: np.ndarray, direct_rays: np.ndarray | None) -> np.ndarray:
        cv2 = self.cv2
        rays = np.clip(np.asarray(actor_rays).reshape(-1), 0.0, 1.0)
        panel = self._base_panel("Normalized ActorRay received by Filter", f"normalized radius: 1.0 == {self.max_range:.1f} m")
        azimuth = np.deg2rad(-179.0 + 2.0 * np.arange(180, dtype=np.float32))
        directions = np.column_stack((np.cos(azimuth), np.sin(azimuth)))

        if direct_rays is not None:
            direct_endpoints = directions * np.asarray(direct_rays).reshape(-1, 1)
            dx, dy = self._xy_to_pixels(direct_endpoints, 1.0)
            for index in range(180):
                cv2.circle(panel, (int(dx[index]), int(dy[index])), 1, (80, 190, 80), -1, cv2.LINE_AA)

        endpoints = directions * rays[:, None]
        px, py = self._xy_to_pixels(endpoints, 1.0)
        for index in range(180):
            endpoint = (int(px[index]), int(py[index]))
            cv2.line(panel, self.CENTER, endpoint, (55, 72, 72), 1, cv2.LINE_AA)
            color = (60, 80, 235) if rays[index] < 0.995 else (110, 110, 110)
            cv2.circle(panel, endpoint, 2, color, -1, cv2.LINE_AA)
        cv2.putText(
            panel,
            f"min={rays.min():.3f}  mean={rays.mean():.3f}  max={rays.max():.3f}",
            (14, self.PANEL_SIZE - 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
        legend = "red: active ActorRay; green: raw direct profile" if direct_rays is not None else "red: occupied before max range; gray: no hit"
        cv2.putText(
            panel,
            legend,
            (14, self.PANEL_SIZE - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (80, 130, 235),
            1,
            cv2.LINE_AA,
        )
        return panel

    def compose(self, points_body: np.ndarray, actor_rays: np.ndarray, direct_rays: np.ndarray | None) -> np.ndarray:
        return np.concatenate((self.physical_panel(points_body), self.actor_panel(actor_rays, direct_rays)), axis=1)

    def render(self, points_body: np.ndarray, actor_rays: np.ndarray, direct_rays: np.ndarray | None) -> bool:
        frame = self.compose(points_body, actor_rays, direct_rays)
        if not self.window_created:
            return True
        self.cv2.imshow(self.WINDOW_NAME, frame)
        return self.cv2.waitKey(1) & 0xFF != ord("q")

    def close(self) -> None:
        if self.window_created:
            self.cv2.destroyWindow(self.WINDOW_NAME)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Visualize MID-360 occupancy and REASEN ActorRay in 2-D.")
    result.add_argument("--interface", "-i", help="NIC connected to G1, e.g. enp3s0")
    result.add_argument("--domain-id", type=int, default=0)
    result.add_argument("--topic", default=mid360.DEFAULT_TOPIC)
    result.add_argument("--min-range", type=float, default=0.3, help="Ignore points closer than this distance in meters")
    result.add_argument("--max-range", type=float, default=3.0)
    result.add_argument("--filter-ground", dest="filter_ground", action="store_true", default=True)
    result.add_argument("--no-filter-ground", dest="filter_ground", action="store_false")
    result.add_argument("--ground-distance", type=float, default=0.04, help="RANSAC ground-plane thickness in meters")
    result.add_argument("--ground-candidate-max-z", type=float, default=-0.25, help="Only fit ground using body-frame points below this Z")
    result.add_argument("--ground-max-tilt-deg", type=float, default=25.0)
    result.add_argument(
        "--ray-median-window",
        type=int,
        default=3,
        help="Odd number of Direct ActorRay frames used for per-direction median filtering",
    )
    result.add_argument(
        "--zmq-endpoint",
        default="tcp://*:5562",
        help="ZMQ PUB endpoint for the active 180-D ActorRay; empty string disables publishing",
    )
    result.add_argument("--lidar-offset", nargs=3, type=float, default=(0.0002835, 0.00003, 0.41618), metavar=("X", "Y", "Z"))
    result.add_argument("--lidar-rpy-deg", nargs=3, type=float, default=(0.0, 177.674, 179.995), metavar=("R", "P", "Y"))
    result.add_argument("--stride", type=int, default=1)
    result.add_argument("--update-hz", type=float, default=20.0)
    result.add_argument("--print-every", type=float, default=1.0)
    result.add_argument("--self-test", action="store_true")
    return result


def cloud_receiver_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        topic=args.topic,
        stride=max(1, args.stride),
        max_range=0.0,
        intensity_max=255.0,
        color_mode="white",
        pool_seconds=0.0,
        accumulate_frames=1,
        max_points=0,
        once=False,
        print_every=args.print_every,
    )


def run(args: argparse.Namespace) -> None:
    ray_median_filter = RayMedianFilter(args.ray_median_window)
    ray_publisher = ActorRayZmqPublisher(args.zmq_endpoint) if args.zmq_endpoint else None

    mid360.init_dds(args.domain_id, args.interface)
    receiver = mid360.Mid360CloudReceiver(cloud_receiver_args(args))

    occupancy_viewer = ReasanOccupancyViewer(args.max_range, window_left=10)

    print("[2-D colors] physical points=orange, direct profile=green, active ActorRay=red, envelope=yellow")
    last_seq = 0
    next_update = 0.0
    update_period = 1.0 / max(args.update_hz, 1.0)
    try:
        while True:
            now = time.monotonic()
            frame = receiver.latest_after(last_seq)
            if frame is not None and now >= next_update:
                points_sensor = filter_sensor_min_range(frame.points, args.min_range)
                points_body = transform_sensor_to_body(points_sensor, tuple(args.lidar_offset), tuple(args.lidar_rpy_deg))
                ground_points_removed = 0
                if args.filter_ground:
                    points_body, ground_points_removed = filter_ground_plane(
                        points_body,
                        distance_threshold=args.ground_distance,
                        candidate_max_z=args.ground_candidate_max_z,
                        max_tilt_deg=args.ground_max_tilt_deg,
                    )
                direct_raw = direct_actor_profile_from_points(points_body, args.max_range)
                actor_for_filter = ray_median_filter.update(direct_raw)
                if ray_publisher is not None:
                    ray_publisher.publish(
                        build_actor_ray_message(
                            actor_for_filter,
                            sequence=frame.seq,
                            max_range=args.max_range,
                        )
                    )

                if not occupancy_viewer.render(points_body, actor_for_filter, direct_raw):
                    break

                occupied = int(np.count_nonzero(actor_for_filter < 1.0))
                if args.print_every > 0 and frame.seq % max(1, round(args.update_hz * args.print_every)) == 0:
                    print(
                        f"[frame {frame.seq}] kept_points={len(points_body)} "
                        f"ground_removed={ground_points_removed} occupied_rays={occupied}/180"
                    )
                last_seq = frame.seq
                next_update = now + update_period

            time.sleep(0.002)
    except KeyboardInterrupt:
        pass
    finally:
        occupancy_viewer.close()
        if ray_publisher is not None:
            ray_publisher.close()
        receiver.close()


def self_test() -> None:
    points = np.array(
        [[1.0, 0.0, 1.0], [0.6, 0.0, -0.6], [-2.0, 0.0, 0.0]], dtype=np.float32
    )
    profile = direct_actor_profile_from_points(points, 3.0)
    assert profile.shape == (180,)
    assert np.count_nonzero(profile < 1.0) == 2
    assert np.isclose(profile[90], np.linalg.norm(points[1]) / 3.0)
    sensor_points = np.array([[0.2, 0.0, 0.0], [0.3, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    filtered = filter_sensor_min_range(sensor_points, 0.3)
    assert filtered.shape == (2, 3)
    assert np.allclose(filtered[:, 0], [0.3, 1.0])
    gx, gy = np.meshgrid(np.linspace(-2.0, 2.0, 30), np.linspace(-2.0, 2.0, 30))
    ground = np.column_stack((gx.ravel(), gy.ravel(), np.full(gx.size, -0.8))).astype(np.float32)
    obstacle = np.array([[1.0, 0.0, -0.4], [1.0, 0.0, 0.0], [1.0, 0.0, 0.4]], dtype=np.float32)
    without_ground, removed = filter_ground_plane(
        np.concatenate((ground, obstacle), axis=0),
        distance_threshold=0.04,
        candidate_max_z=-0.25,
        max_tilt_deg=25.0,
    )
    assert removed == len(ground)
    assert len(without_ground) == len(obstacle)
    median_filter = RayMedianFilter(3)
    baseline = np.ones(180, dtype=np.float32)
    assert np.allclose(median_filter.update(baseline), 1.0)
    impulse = baseline.copy()
    impulse[20] = 0.1
    assert np.isclose(median_filter.update(impulse)[20], 0.55)
    assert np.isclose(median_filter.update(baseline)[20], 1.0)
    message = build_actor_ray_message(
        baseline,
        sequence=7,
        max_range=3.0,
    )
    assert message["count"] == 180
    assert len(message["normalized"]) == 180
    assert len(message["ranges_m"]) == 180
    assert message["ranges_m"][0] == 3.0
    assert message["source"] == "direct_all_vertical_median"
    json.dumps(message)
    print("self-test passed: all-vertical Direct ActorRay=[180]")


def main() -> int:
    args = parser().parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.min_range < 0 or args.max_range <= args.min_range:
        raise SystemExit("Require 0 <= --min-range < --max-range")
    if args.ray_median_window < 1 or args.ray_median_window % 2 == 0:
        raise SystemExit("--ray-median-window must be a positive odd integer")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
