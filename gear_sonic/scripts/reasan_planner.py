#!/usr/bin/env python3
"""Apply a TTC/distance potential field to ActorRay and publish SONIC planner messages."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import signal
import sys
import time
from typing import Any

import numpy as np
import zmq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (  # noqa: E402
    build_command_message,
    build_planner_message,
)


RAY_MESSAGE_TYPE = "reasan_actor_ray"
COMMAND_MESSAGE_TYPE = "navila_reasan_velocity_command"
COMMAND_LOWER = np.array([-0.5, -0.3, -1.0], dtype=np.float32)
COMMAND_UPPER = np.array([1.0, 0.3, 1.0], dtype=np.float32)


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
    return {
        "sequence": int(message["sequence"]),
        "source": str(message.get("source", "unknown")),
        "rays": rays,
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


@dataclass(frozen=True)
class PotentialFieldResult:
    velocity: np.ndarray
    danger: float
    obstacle_direction: np.ndarray
    avoidance_side: float


class TtcPotentialField:
    """NumPy equivalent of REASEN's training/play safe-velocity field."""

    def __init__(self, control_dt: float) -> None:
        if control_dt <= 0.0:
            raise ValueError("control_dt must be positive")
        angles = np.arange(180, dtype=np.float32) * (2.0 * np.pi / 180.0) - np.pi
        self.directions = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        self.control_dt = control_dt
        self.previous_side = 0.0
        self.yaw_rate = 0.0

    def reset(self) -> None:
        self.previous_side = 0.0
        self.yaw_rate = 0.0

    def step_normalized(self, command: np.ndarray, normalized_rays: np.ndarray) -> PotentialFieldResult:
        rays = np.asarray(normalized_rays, dtype=np.float32)
        if rays.shape != (180,) or not np.isfinite(rays).all():
            raise ValueError("normalized ActorRay must contain 180 finite values")
        return self.step(command, np.clip(rays, 0.0, 1.0) * 3.0)

    def step(self, command: np.ndarray, ray_distances: np.ndarray) -> PotentialFieldResult:
        command = np.clip(np.asarray(command, dtype=np.float32), COMMAND_LOWER, COMMAND_UPPER)
        distances = np.nan_to_num(np.asarray(ray_distances, dtype=np.float32), posinf=3.0)
        if distances.shape != (180,):
            raise ValueError("ray_distances must have shape [180]")
        distances = np.clip(distances, 0.0, 3.0)
        command_xy = command[:2]
        command_speed = float(np.linalg.norm(command_xy))
        if command_speed <= 1.0e-6:
            self.reset()
            return PotentialFieldResult(
                np.zeros(3, dtype=np.float32), 0.0, np.zeros(2, dtype=np.float32), 0.0
            )

        command_direction = command_xy / command_speed
        obstacle_vectors = distances[:, None] * self.directions
        along = obstacle_vectors @ command_direction
        lateral_sq = np.maximum(distances**2 - along**2, 0.0)
        intersects = (along > 0.0) & (lateral_sq < 0.3**2)
        contact_offset = np.sqrt(np.maximum(0.3**2 - lateral_sq, 0.0))
        distance_to_contact = np.maximum(along - contact_offset, 0.0)
        ttc = np.full(180, np.inf, dtype=np.float32)
        ttc[intersects] = distance_to_contact[intersects] / command_speed
        ttc_risk = np.clip((3.0 - ttc) / 3.0, 0.0, 1.0) ** 4
        clearance = np.maximum(distances - 0.3, 0.0)
        distance_risk = np.clip((1.2 - clearance) / 1.2, 0.0, 1.0) ** 4
        risk = 0.7 * ttc_risk + 0.3 * distance_risk

        obstacle_sum = np.sum(risk[:, None] * self.directions, axis=0)
        obstacle_mass = max(float(np.sum(risk)), 1.0e-6)
        obstacle_norm = float(np.linalg.norm(obstacle_sum))
        obstacle_direction = obstacle_sum / max(obstacle_norm, 1.0e-6)
        coherence = float(np.clip(obstacle_norm / obstacle_mass, 0.0, 1.0))
        danger = float(np.max(risk) * coherence)

        radial_velocity = -0.625 * danger * obstacle_direction
        closing_speed = max(float(command_xy @ obstacle_direction), 0.0)
        removed_velocity = command_xy - danger * closing_speed * obstacle_direction

        approach = np.maximum(self.directions @ command_direction, 0.0)
        cross = command_direction[0] * self.directions[:, 1] - command_direction[1] * self.directions[:, 0]
        forward = approach > 1.0e-6
        left = forward & (cross > 1.0e-6)
        right = forward & (cross < -1.0e-6)
        left_clearance = float(np.sum(distances[left]) / max(int(np.sum(left)), 1))
        right_clearance = float(np.sum(distances[right]) / max(int(np.sum(right)), 1))
        side = float(np.tanh((left_clearance - right_clearance + 0.15 * self.previous_side) / 0.25))
        self.previous_side = float(np.clip(side, -1.0, 1.0))

        left_tangent = np.array([-obstacle_direction[1], obstacle_direction[0]], dtype=np.float32)
        tangent_velocity = 0.5 * danger * command_speed * self.previous_side * left_tangent
        safe_xy = removed_velocity + tangent_velocity + radial_velocity
        reverse = min(float(safe_xy @ command_direction), 0.0)
        safe_xy -= reverse * command_direction
        safe_xy = np.clip(safe_xy, COMMAND_LOWER[:2], COMMAND_UPPER[:2])

        yaw_target = 0.0
        if float(np.linalg.norm(safe_xy)) < 0.15:
            safe_xy.fill(0.0)
            turn_side = float(np.sign(self.previous_side)) or 1.0
            yaw_target = 0.5 * turn_side
        max_yaw_delta = 2.0 * self.control_dt
        self.yaw_rate += float(np.clip(yaw_target - self.yaw_rate, -max_yaw_delta, max_yaw_delta))
        self.yaw_rate = float(np.clip(self.yaw_rate, COMMAND_LOWER[2], COMMAND_UPPER[2]))
        velocity = np.array([safe_xy[0], safe_xy[1], self.yaw_rate], dtype=np.float32)
        return PotentialFieldResult(velocity, danger, obstacle_direction.astype(np.float32), self.previous_side)


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
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--output-hz", type=float, default=20.0)
    parser.add_argument("--ray-timeout", type=float, default=0.35)
    parser.add_argument("--keyboard-timeout", type=float, default=0.7)
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
    positive = (args.control_hz, args.output_hz, args.ray_timeout, args.keyboard_timeout, args.status_hz)
    if min(positive) <= 0.0:
        raise ValueError("frequencies and timeouts must be positive")
    potential_field = TtcPotentialField(control_dt=1.0 / args.control_hz)
    context = zmq.Context.instance()
    ray_socket = make_latest_subscriber(context, args.ray_endpoint)
    keyboard_socket = make_latest_subscriber(context, args.keyboard_endpoint)
    output = context.socket(zmq.PUB)
    output.setsockopt(zmq.LINGER, 0)
    output.bind(args.output_endpoint)
    poller = zmq.Poller()
    poller.register(ray_socket, zmq.POLLIN)
    poller.register(keyboard_socket, zmq.POLLIN)
    latest_ray, latest_command = LatestValue(), LatestValue()
    planner = PlannerState()
    safe_velocity = np.zeros(3, dtype=np.float32)
    running, healthy_last = True, False

    def stop(_signum=None, _frame=None) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    control_period, output_period = 1.0 / args.control_hz, 1.0 / args.output_hz
    next_control = next_output = next_status = time.monotonic()
    print(f"[REASEN Planner] rays={args.ray_endpoint} keyboard={args.keyboard_endpoint}")
    print(
        f"[REASEN Planner] SONIC PUB={args.output_endpoint}, TTC/distance field={args.control_hz:g} Hz, "
        f"output={args.output_hz:g} Hz"
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

            ray_ok = latest_ray.age(now) <= args.ray_timeout
            command_age = latest_command.age(now)
            command_ok = latest_command.value is not None and command_age <= min(
                args.keyboard_timeout, latest_command.value["duration"]
            )
            healthy = bool(ray_ok and command_ok)
            command = latest_command.value["velocity"] if command_ok else np.zeros(3, dtype=np.float32)
            if now >= next_control:
                if healthy:
                    field = potential_field.step_normalized(command, latest_ray.value["rays"])
                    safe_velocity = field.velocity
                else:
                    safe_velocity.fill(0.0)
                    if healthy_last:
                        potential_field.reset()
                healthy_last = healthy
                next_control = now + control_period
            if now >= next_output:
                output.send(planner.message(safe_velocity, output_period))
                next_output = now + output_period
            if now >= next_status:
                status = "OK" if healthy else "SAFE-STOP"
                print(
                    f"[REASEN Planner] {status} ray_age={latest_ray.age(now):.3f}s "
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
        output.close()


if __name__ == "__main__":
    main()
