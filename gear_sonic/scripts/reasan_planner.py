#!/usr/bin/env python3
"""Run the G1 REASEN Filter ONNX and publish SONIC planner messages."""

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
import onnxruntime as ort
import zmq


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (  # noqa: E402
    build_command_message,
    build_planner_message,
)


RAY_MESSAGE_TYPE = "reasan_actor_ray"
COMMAND_MESSAGE_TYPE = "navila_reasan_velocity_command"
DEFAULT_FILTER = Path(
    "/home/user/Project/REASAN/training/logs/rsl_rl/g1_filter/"
    "g1_filter_bigger_z_speed/exported/filter_g1_model_19998.onnx"
)
COMMAND_LOWER = np.array([-0.5, -0.15, -1.0], dtype=np.float32)
COMMAND_UPPER = np.array([1.0, 0.15, 1.0], dtype=np.float32)


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
    gravity = np.asarray(message.get("projected_gravity"), dtype=np.float32)
    angular_velocity = np.asarray(message.get("angular_velocity"), dtype=np.float32)
    if rays.shape != (180,) or not np.isfinite(rays).all() or np.any((rays < 0.0) | (rays > 1.0)):
        raise ValueError("ActorRay must contain 180 finite normalized values in [0,1]")
    if gravity.shape != (3,) or angular_velocity.shape != (3,):
        raise ValueError("ActorRay message is missing projected_gravity/angular_velocity")
    if not np.isfinite(gravity).all() or not np.isfinite(angular_velocity).all():
        raise ValueError("IMU features must be finite")
    if not bool(message.get("imu_valid", False)):
        raise ValueError("IMU has not received a valid sample")
    imu_age = float(message.get("imu_age_s", math.inf))
    if not math.isfinite(imu_age) or imu_age < 0.0:
        raise ValueError("invalid IMU age")
    return {
        "sequence": int(message["sequence"]),
        "source": str(message.get("source", "unknown")),
        "rays": rays,
        "gravity": gravity,
        "angular_velocity": angular_velocity,
        "imu_age": imu_age,
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


class ReasanFilterOnnx:
    def __init__(self, model_path: Path, action_ema_alpha: float) -> None:
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        expected_inputs = {"proprio_obs", "ray_obs", "h_in", "c_in"}
        expected_outputs = {"actions", "h_out", "c_out"}
        if {item.name for item in self.session.get_inputs()} != expected_inputs:
            raise ValueError(f"unexpected Filter inputs: {[item.name for item in self.session.get_inputs()]}")
        if {item.name for item in self.session.get_outputs()} != expected_outputs:
            raise ValueError(f"unexpected Filter outputs: {[item.name for item in self.session.get_outputs()]}")
        self.action_ema_alpha = action_ema_alpha
        self.hidden = np.zeros((1, 1, 256), dtype=np.float32)
        self.cell = np.zeros((1, 1, 256), dtype=np.float32)
        self.previous_action = np.zeros(3, dtype=np.float32)
        print(f"[REASEN Filter] {model_path} | providers={self.session.get_providers()}")

    def reset(self) -> None:
        self.hidden.fill(0.0)
        self.cell.fill(0.0)
        self.previous_action.fill(0.0)

    def infer(self, command: np.ndarray, ray: dict[str, Any], suppress_zero: bool) -> np.ndarray:
        command = np.clip(np.asarray(command, dtype=np.float32), COMMAND_LOWER, COMMAND_UPPER)
        proprio = np.concatenate(
            (ray["angular_velocity"] * 0.25, ray["gravity"], command, self.previous_action)
        )[None].astype(np.float32)
        actions, self.hidden, self.cell = self.session.run(
            ["actions", "h_out", "c_out"],
            {"proprio_obs": proprio, "ray_obs": ray["rays"][None], "h_in": self.hidden, "c_in": self.cell},
        )
        action = np.clip(np.asarray(actions, dtype=np.float32).reshape(3), COMMAND_LOWER, COMMAND_UPPER)
        if suppress_zero and np.all(np.abs(command) <= 1.0e-6):
            action.fill(0.0)
            self.previous_action.fill(0.0)
            return action
        if self.action_ema_alpha > 0.0:
            action = self.action_ema_alpha * self.previous_action + (1.0 - self.action_ema_alpha) * action
        self.previous_action = action.copy()
        return action


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
    parser.add_argument("--filter", type=Path, default=DEFAULT_FILTER)
    parser.add_argument("--ray-endpoint", default="tcp://127.0.0.1:5562")
    parser.add_argument("--keyboard-endpoint", default="tcp://127.0.0.1:5558")
    parser.add_argument("--output-endpoint", default="tcp://*:5563")
    parser.add_argument("--filter-hz", type=float, default=50.0)
    parser.add_argument("--output-hz", type=float, default=20.0)
    parser.add_argument("--ray-timeout", type=float, default=0.35)
    parser.add_argument("--imu-timeout", type=float, default=0.15)
    parser.add_argument("--keyboard-timeout", type=float, default=0.7)
    parser.add_argument("--action-ema-alpha", type=float, default=0.0)
    parser.add_argument("--suppress-output-on-zero-input", action="store_true")
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
    positive = (args.filter_hz, args.output_hz, args.ray_timeout, args.imu_timeout, args.keyboard_timeout, args.status_hz)
    if min(positive) <= 0.0 or not 0.0 <= args.action_ema_alpha < 1.0:
        raise ValueError("frequencies/timeouts must be positive and EMA alpha must be in [0,1)")
    if not args.filter.is_file():
        raise FileNotFoundError(f"Filter ONNX not found: {args.filter}")
    filter_model = ReasanFilterOnnx(args.filter, args.action_ema_alpha)
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
    running, healthy_last, turn_bypass_last = True, False, False

    def stop(_signum=None, _frame=None) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    filter_period, output_period = 1.0 / args.filter_hz, 1.0 / args.output_hz
    next_filter = next_output = next_status = time.monotonic()
    print(f"[REASEN Planner] rays={args.ray_endpoint} keyboard={args.keyboard_endpoint}")
    print(f"[REASEN Planner] SONIC PUB={args.output_endpoint}, Filter={args.filter_hz:g} Hz, output={args.output_hz:g} Hz")
    print(f"[REASEN Planner] EMA={args.action_ema_alpha:g}, suppress-zero={args.suppress_output_on_zero_input}")
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
            imu_ok = ray_ok and latest_ray.value["imu_age"] + latest_ray.age(now) <= args.imu_timeout
            command_age = latest_command.age(now)
            command_ok = latest_command.value is not None and command_age <= min(
                args.keyboard_timeout, latest_command.value["duration"]
            )
            healthy = bool(ray_ok and imu_ok and latest_command.value is not None)
            command = latest_command.value["velocity"] if command_ok else np.zeros(3, dtype=np.float32)
            turn_bypass = bool(
                command_ok
                and abs(float(command[0])) <= 1.0e-6
                and abs(float(command[1])) <= 1.0e-6
                and abs(float(command[2])) > 1.0e-6
            )
            if now >= next_filter:
                if turn_bypass:
                    if not turn_bypass_last:
                        filter_model.reset()
                    safe_velocity = command.copy()
                elif healthy:
                    if turn_bypass_last:
                        filter_model.reset()
                    safe_velocity = filter_model.infer(command, latest_ray.value, args.suppress_output_on_zero_input)
                else:
                    safe_velocity.fill(0.0)
                    if healthy_last or turn_bypass_last:
                        filter_model.reset()
                healthy_last = healthy
                turn_bypass_last = turn_bypass
                next_filter = now + filter_period
            if now >= next_output:
                output.send(planner.message(safe_velocity, output_period))
                next_output = now + output_period
            if now >= next_status:
                status = "TURN-BYPASS" if turn_bypass else ("OK" if healthy else "SAFE-STOP")
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
