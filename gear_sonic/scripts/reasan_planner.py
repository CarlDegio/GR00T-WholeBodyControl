#!/usr/bin/env python3
"""Apply a rule-based forward protective stop and publish SONIC commands."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import signal
import time
from typing import Any

import numpy as np
import zmq


from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    build_planner_message,
)


RAY_MESSAGE_TYPE = "reasan_actor_ray"
COMMAND_MESSAGE_TYPE = "navila_reasan_velocity_command"
COMMAND_LOWER = np.array([-0.5, -0.15, -1.0], dtype=np.float32)
COMMAND_UPPER = np.array([1.0, 0.15, 1.0], dtype=np.float32)
FORWARD_STOP_DISTANCE_M = 0.5
FORWARD_HALF_ANGLE_DEG = 45.0
RAW_DEPTH_STOP_DISTANCE_M = 0.30
RAW_DEPTH_STOP_MIN_AREA_PIXELS = 2000


@dataclass
class LatestValue:
    value: Any = None
    received_at: float | None = None

    def update(self, value: Any, now: float) -> None:
        self.value = value
        self.received_at = now

    def age(self, now: float) -> float:
        return math.inf if self.received_at is None else max(0.0, now - self.received_at)


def decode_actor_ray(raw: bytes | str) -> dict[str, Any]:
    try:
        message = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
        raise ValueError(f"invalid ActorRay JSON: {exc}") from exc
    if not isinstance(message, dict) or message.get("type") != RAY_MESSAGE_TYPE or message.get("version") != 1:
        raise ValueError("unsupported ActorRay message")
    rays = np.asarray(message.get("normalized"), dtype=np.float32)
    if rays.shape != (180,) or not np.isfinite(rays).all() or np.any((rays < 0.0) | (rays > 1.0)):
        raise ValueError("ActorRay must contain 180 finite normalized values in [0,1]")
    try:
        angle_min = float(message["angle_min_deg"])
        angle_increment = float(message["angle_increment_deg"])
        range_max = float(message["range_max_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("ActorRay angular/range metadata is missing") from exc
    if (
        not all(math.isfinite(value) for value in (angle_min, angle_increment, range_max))
        or angle_increment <= 0.0
        or range_max <= 0.0
    ):
        raise ValueError("ActorRay angular/range metadata is invalid")
    angles = angle_min + angle_increment * np.arange(rays.size, dtype=np.float32)
    return {
        "sequence": int(message["sequence"]),
        "source": str(message.get("source", "unknown")),
        "rays": rays,
        "angles_deg": angles,
        "distances": rays * range_max,
    }


def decode_velocity_command(raw: bytes | str) -> dict[str, Any]:
    try:
        message = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
        raise ValueError(f"invalid velocity-command JSON: {exc}") from exc
    if not isinstance(message, dict) or message.get("type") != COMMAND_MESSAGE_TYPE:
        raise ValueError("unsupported velocity-command message")
    if message.get("version") != 1 or message.get("status") != "ok":
        raise ValueError("velocity command must have version=1 and status='ok'")
    segments = message.get("segments")
    if not isinstance(segments, list) or not segments or not isinstance(segments[0], dict):
        raise ValueError("velocity command must contain at least one segment")
    segment = segments[0]
    try:
        duration = float(segment["duration_s"])
        velocity = np.asarray([segment["vx"], segment["vy"], segment["wz"]], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid velocity segment: {exc}") from exc
    if duration <= 0.0 or not math.isfinite(duration) or not np.isfinite(velocity).all():
        raise ValueError("velocity and duration must be finite; duration must be positive")
    return {
        "action": str(message.get("action", "unknown")),
        "duration": duration,
        "velocity": np.clip(velocity, COMMAND_LOWER, COMMAND_UPPER),
    }


def raw_depth_requires_stop(
    depth: np.ndarray,
    *,
    depth_scale_m: float,
    stop_distance_m: float = RAW_DEPTH_STOP_DISTANCE_M,
    min_area_pixels: int = RAW_DEPTH_STOP_MIN_AREA_PIXELS,
) -> bool:
    """Return whether one connected near-depth region exceeds a fixed pixel area."""
    import cv2

    image = np.asarray(depth)
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[..., 0]
    if image.ndim != 2 or image.size == 0:
        raise ValueError(f"raw chest depth must be a non-empty 2-D image, got {image.shape}")
    if depth_scale_m <= 0.0 or stop_distance_m <= 0.0:
        raise ValueError("depth scale and stop distance must be positive")
    if isinstance(min_area_pixels, bool) or min_area_pixels <= 0:
        raise ValueError("minimum connected area must be a positive pixel count")

    depth_m = image.astype(np.float32) * float(depth_scale_m)
    near = np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m < stop_distance_m)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        near.astype(np.uint8), connectivity=8
    )
    if component_count <= 1:
        return False
    largest_area = int(stats[1:, cv2.CC_STAT_AREA].max())
    return largest_area > int(min_area_pixels)


def apply_rule_based_safety(
    command: np.ndarray,
    ray: dict[str, Any],
    *,
    stop_distance_m: float = FORWARD_STOP_DISTANCE_M,
    half_angle_deg: float = FORWARD_HALF_ANGLE_DEG,
    camera_stop: bool = False,
) -> np.ndarray:
    """Stop only positive-X motion when an obstacle enters the front sector."""
    velocity = np.clip(
        np.asarray(command, dtype=np.float32).reshape(3), COMMAND_LOWER, COMMAND_UPPER
    )
    if camera_stop:
        return np.zeros(3, dtype=np.float32)
    if float(velocity[0]) <= 1.0e-6:
        return velocity.copy()
    angles = np.asarray(ray["angles_deg"], dtype=np.float32)
    distances = np.asarray(ray["distances"], dtype=np.float32)
    in_front = np.abs(angles) <= float(half_angle_deg)
    if np.any(in_front & (distances <= float(stop_distance_m))):
        return np.zeros(3, dtype=np.float32)
    return velocity.copy()


@dataclass
class PlannerState:
    heading: float = 0.0

    def message(self, velocity: np.ndarray, dt: float) -> bytes:
        vx, vy, wz = map(float, velocity)
        self.heading = math.remainder(self.heading + wz * dt, 2.0 * math.pi)
        cosine, sine = math.cos(self.heading), math.sin(self.heading)
        world_x = cosine * vx - sine * vy
        world_y = sine * vx + cosine * vy
        speed = math.hypot(world_x, world_y)
        movement = (0.0, 0.0, 0.0) if speed < 1.0e-6 else (world_x / speed, world_y / speed, 0.0)
        facing = (cosine, sine, 0.0)
        return build_planner_message(1, movement, facing, speed=speed, height=-1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ray-endpoint", default="tcp://127.0.0.1:5562")
    parser.add_argument("--keyboard-endpoint", default="tcp://127.0.0.1:5558")
    parser.add_argument("--output-endpoint", default="tcp://*:5563")
    parser.add_argument("--camera-host", default="127.0.0.1")
    parser.add_argument("--camera-port", type=int, default=5555)
    parser.add_argument("--output-hz", type=float, default=20.0)
    parser.add_argument("--ray-timeout", type=float, default=0.35)
    parser.add_argument("--depth-timeout", type=float, default=0.35)
    parser.add_argument("--keyboard-timeout", type=float, default=0.7)
    parser.add_argument("--forward-stop-distance", type=float, default=FORWARD_STOP_DISTANCE_M)
    parser.add_argument("--forward-half-angle", type=float, default=FORWARD_HALF_ANGLE_DEG)
    parser.add_argument("--depth-stop-distance", type=float, default=RAW_DEPTH_STOP_DISTANCE_M)
    parser.add_argument(
        "--depth-stop-min-area-pixels", type=int, default=RAW_DEPTH_STOP_MIN_AREA_PIXELS
    )
    parser.add_argument("--status-hz", type=float, default=2.0)
    return parser.parse_args()


def make_latest_subscriber(context: zmq.Context, endpoint: str) -> zmq.Socket:
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(endpoint)
    return socket


def main() -> None:
    args = parse_args()
    positive = (
        args.output_hz,
        args.ray_timeout,
        args.depth_timeout,
        args.keyboard_timeout,
        args.forward_stop_distance,
        args.forward_half_angle,
        args.status_hz,
        args.camera_port,
    )
    if (
        min(positive) <= 0.0
        or args.forward_half_angle > 180.0
        or args.depth_stop_min_area_pixels <= 0
    ):
        raise ValueError("frequencies, timeouts and safety-sector values must be valid")
    from gear_sonic.camera.composed_camera import ComposedCameraClientSensor

    context = zmq.Context.instance()
    ray_socket = make_latest_subscriber(context, args.ray_endpoint)
    keyboard_socket = make_latest_subscriber(context, args.keyboard_endpoint)
    camera = ComposedCameraClientSensor(server_ip=args.camera_host, port=args.camera_port)
    output = context.socket(zmq.PUB)
    output.setsockopt(zmq.LINGER, 0)
    output.bind(args.output_endpoint)
    poller = zmq.Poller()
    poller.register(ray_socket, zmq.POLLIN)
    poller.register(keyboard_socket, zmq.POLLIN)
    latest_ray, latest_command, latest_depth = LatestValue(), LatestValue(), LatestValue()
    last_depth_timestamp: float | None = None
    planner = PlannerState()
    safe_velocity = np.zeros(3, dtype=np.float32)
    running = True

    def stop(_signum=None, _frame=None) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    output_period = 1.0 / args.output_hz
    next_output = next_status = time.monotonic()
    print(f"[REASEN Planner] rays={args.ray_endpoint} keyboard={args.keyboard_endpoint}")
    print(f"[REASEN Planner] raw chest depth={args.camera_host}:{args.camera_port}")
    print(f"[REASEN Planner] SONIC PUB={args.output_endpoint}, output={args.output_hz:g} Hz")
    print(
        f"[REASEN Planner] rule=front-stop distance={args.forward_stop_distance:g} m "
        f"sector=+/-{args.forward_half_angle:g} deg"
    )
    print(
        f"[REASEN Planner] raw-depth-stop distance={args.depth_stop_distance:g} m "
        f"largest-area>{args.depth_stop_min_area_pixels} pixels"
    )
    try:
        while running:
            events = dict(poller.poll(1))
            now = time.monotonic()
            if ray_socket in events:
                try:
                    latest_ray.update(decode_actor_ray(ray_socket.recv()), now)
                except ValueError as exc:
                    print(f"[REASEN Planner] Ignored ActorRay: {exc}")
            if keyboard_socket in events:
                try:
                    latest_command.update(decode_velocity_command(keyboard_socket.recv()), now)
                except ValueError as exc:
                    print(f"[REASEN Planner] Ignored keyboard command: {exc}")

            packet = camera.read(blocking=False)
            if packet is not None:
                depth = packet.get("images", {}).get("chest_view_depth")
                timestamp = packet.get("timestamps", {}).get("chest_view_depth")
                if depth is not None and timestamp is not None:
                    timestamp = float(timestamp)
                    if math.isfinite(timestamp) and timestamp != last_depth_timestamp:
                        info = packet.get("camera_info", {}).get("chest_view", {})
                        latest_depth.update(
                            {
                                "image": depth,
                                "scale_m": float(info.get("depth_scale_m", 0.001)),
                            },
                            now,
                        )
                        last_depth_timestamp = timestamp

            ray_ok = latest_ray.age(now) <= args.ray_timeout
            command_age = latest_command.age(now)
            command_ok = latest_command.value is not None and command_age <= min(
                args.keyboard_timeout, latest_command.value["duration"]
            )
            healthy = bool(ray_ok and latest_command.value is not None)
            command = latest_command.value["velocity"] if command_ok else np.zeros(3, dtype=np.float32)
            camera_stop = False
            if latest_depth.age(now) <= args.depth_timeout:
                try:
                    camera_stop = raw_depth_requires_stop(
                        latest_depth.value["image"],
                        depth_scale_m=latest_depth.value["scale_m"],
                        stop_distance_m=args.depth_stop_distance,
                        min_area_pixels=args.depth_stop_min_area_pixels,
                    )
                except ValueError as exc:
                    print(f"[REASEN Planner] Ignored raw chest depth: {exc}")
            if now >= next_output:
                safe_velocity = (
                    apply_rule_based_safety(
                        command,
                        latest_ray.value,
                        stop_distance_m=args.forward_stop_distance,
                        half_angle_deg=args.forward_half_angle,
                        camera_stop=camera_stop,
                    )
                    if healthy
                    else np.zeros(3, dtype=np.float32)
                )
                output.send(planner.message(safe_velocity, output_period))
                next_output = now + output_period
            if now >= next_status:
                protective_stop = (
                    healthy
                    and np.any(np.abs(command) > 1.0e-6)
                    and not np.any(safe_velocity)
                    and (camera_stop or float(command[0]) > 1.0e-6)
                )
                status = "PROTECTIVE-STOP" if protective_stop else ("PASS" if healthy else "SENSOR-STOP")
                print(
                    f"[REASEN Planner] {status} ray_age={latest_ray.age(now):.3f}s "
                    f"depth_age={latest_depth.age(now):.3f}s depth_stop={camera_stop} "
                    f"cmd_age={command_age:.3f}s in={command.tolist()} out={safe_velocity.tolist()}"
                )
                next_status = now + 1.0 / args.status_hz
    finally:
        zero = np.zeros(3, dtype=np.float32)
        for _ in range(3):
            output.send(planner.message(zero, output_period))
            time.sleep(0.02)
        output.send(build_command_message(start=False, stop=True, planner=True))
        ray_socket.close()
        keyboard_socket.close()
        camera.stop_client()
        output.close()


if __name__ == "__main__":
    main()
