#!/usr/bin/env python3
"""Relay LaViRA velocity JSON directly to SONIC planner ZMQ messages."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import signal
import time
from typing import Any

import numpy as np
import zmq

from gear_sonic.utils.inference.initial_poses import UPPER_BODY_MUJOCO_INDICES
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    build_planner_message,
)


COMMAND_MESSAGE_TYPE = "navila_reasan_velocity_command"
COMMAND_LOWER = np.array([-0.5, -0.15, -1.0], dtype=np.float32)
COMMAND_UPPER = np.array([1.0, 0.15, 1.0], dtype=np.float32)


def decode_velocity_command(raw: bytes | str) -> dict[str, Any]:
    try:
        message = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
        raise ValueError(f"invalid LaViRA velocity JSON: {exc}") from exc
    if not isinstance(message, dict) or message.get("type") != COMMAND_MESSAGE_TYPE:
        raise ValueError("unsupported LaViRA velocity message")
    if message.get("version") != 1 or message.get("status") != "ok":
        raise ValueError("velocity command must have version=1 and status='ok'")
    segments = message.get("segments")
    if (
        not isinstance(segments, list)
        or not segments
        or not isinstance(segments[0], dict)
    ):
        raise ValueError("velocity command must contain at least one segment")
    segment = segments[0]
    try:
        duration = float(segment["duration_s"])
        velocity = np.asarray(
            [segment["vx"], segment["vy"], segment["wz"]], dtype=np.float32
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid velocity segment: {exc}") from exc
    if (
        duration <= 0.0
        or not math.isfinite(duration)
        or not np.isfinite(velocity).all()
    ):
        raise ValueError(
            "velocity and duration must be finite; duration must be positive"
        )
    return {
        "duration": duration,
        "velocity": np.clip(velocity, COMMAND_LOWER, COMMAND_UPPER),
    }


@dataclass
class LatestCommand:
    value: dict[str, Any] | None = None
    received_at: float | None = None

    def update(self, value: dict[str, Any], now: float) -> None:
        self.value = value
        self.received_at = now

    def velocity(self, now: float, timeout: float) -> np.ndarray:
        if self.value is None or self.received_at is None:
            return np.zeros(3, dtype=np.float32)
        age = max(0.0, now - self.received_at)
        if age > min(timeout, float(self.value["duration"])):
            return np.zeros(3, dtype=np.float32)
        return np.asarray(self.value["velocity"], dtype=np.float32).copy()


@dataclass(frozen=True)
class FrozenPlannerPose:
    upper_body_position: tuple[float, ...]
    left_hand_position: tuple[float, ...]
    right_hand_position: tuple[float, ...]


def extract_frozen_planner_pose(state: Any) -> FrozenPlannerPose:
    """Validate one g1_debug sample and select the planner-order hold targets."""
    if not isinstance(state, dict):
        raise ValueError("g1_debug state must be an object")
    try:
        body = np.asarray(state["body_q_measured"], dtype=np.float64)
        left = np.asarray(state["left_hand_q_measured"], dtype=np.float64)
        right = np.asarray(state["right_hand_q_measured"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("g1_debug measured body/hand state is incomplete") from exc
    if body.shape != (29,) or left.shape != (7,) or right.shape != (7,):
        raise ValueError("g1_debug measured state must have shapes body=29, hands=7+7")
    if (
        not np.isfinite(body).all()
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        raise ValueError("g1_debug measured state must be finite")
    upper = tuple(float(body[index]) for index in UPPER_BODY_MUJOCO_INDICES)
    return FrozenPlannerPose(
        upper_body_position=upper,
        left_hand_position=tuple(map(float, left)),
        right_hand_position=tuple(map(float, right)),
    )


@dataclass
class PlannerState:
    heading: float = 0.0

    def message(
        self,
        velocity: np.ndarray,
        dt: float,
        *,
        frozen_pose: FrozenPlannerPose | None = None,
    ) -> bytes:
        vx, vy, wz = map(float, velocity)
        self.heading = math.remainder(self.heading + wz * dt, 2.0 * math.pi)
        cosine, sine = math.cos(self.heading), math.sin(self.heading)
        world_x = cosine * vx - sine * vy
        world_y = sine * vx + cosine * vy
        speed = math.hypot(world_x, world_y)
        movement = (
            (0.0, 0.0, 0.0)
            if speed < 1.0e-6
            else (world_x / speed, world_y / speed, 0.0)
        )
        mode = (
            0 if frozen_pose is not None and speed < 1.0e-6 and abs(wz) < 1.0e-6 else 1
        )
        return build_planner_message(
            mode,
            movement,
            (cosine, sine, 0.0),
            speed=speed,
            height=-1.0,
            upper_body_position=(
                None if frozen_pose is None else frozen_pose.upper_body_position
            ),
            upper_body_velocity=(None if frozen_pose is None else [0.0] * 17),
            left_hand_position=(
                None if frozen_pose is None else frozen_pose.left_hand_position
            ),
            right_hand_position=(
                None if frozen_pose is None else frozen_pose.right_hand_position
            ),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="tcp://127.0.0.1:5558")
    parser.add_argument("--output", default="tcp://*:5563")
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--timeout", type=float, default=0.7)
    parser.add_argument("--freeze-current-upper-body", action="store_true")
    parser.add_argument("--state-host", default="localhost")
    parser.add_argument("--state-port", type=int, default=5557)
    parser.add_argument("--hold-ready-file", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hz <= 0.0 or args.timeout <= 0.0:
        raise ValueError("hz and timeout must be positive")
    context = zmq.Context.instance()
    source = context.socket(zmq.SUB)
    source.setsockopt(zmq.SUBSCRIBE, b"")
    source.setsockopt(zmq.CONFLATE, 1)
    source.setsockopt(zmq.LINGER, 0)
    source.connect(args.source)
    output = context.socket(zmq.PUB)
    output.setsockopt(zmq.LINGER, 0)
    output.bind(args.output)
    planner = PlannerState()
    latest = LatestCommand()
    state_subscriber = None
    frozen_pose: FrozenPlannerPose | None = None
    ready_path = Path(args.hold_ready_file).resolve() if args.hold_ready_file else None
    request_path = Path(f"{ready_path}.request") if ready_path is not None else None
    if ready_path is not None:
        ready_path.unlink(missing_ok=True)
    if request_path is not None:
        request_path.unlink(missing_ok=True)
    if args.freeze_current_upper_body:
        from gear_sonic.utils.data_collection.zmq_state_subscriber import (
            ZMQStateSubscriber,
        )

        state_subscriber = ZMQStateSubscriber(
            host=args.state_host, port=args.state_port
        )
    period = 1.0 / args.hz
    running = True

    def stop(_signum=None, _frame=None) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print(
        f"[LaViRA Relay] {args.source} -> {args.output}, "
        f"{args.hz:g} Hz, timeout={args.timeout:g}s"
    )
    next_tick = time.monotonic()
    try:
        while running:
            while True:
                try:
                    raw = source.recv(zmq.NOBLOCK)
                except zmq.Again:
                    break
                try:
                    latest.update(decode_velocity_command(raw), time.monotonic())
                except ValueError as exc:
                    print(f"[LaViRA Relay] Ignored command: {exc}")
            now = time.monotonic()
            if now >= next_tick:
                latch_completed = False
                latch_token = "ready\n"
                latch_requested = (request_path is None and frozen_pose is None) or (
                    request_path is not None and request_path.is_file()
                )
                if state_subscriber is not None and latch_requested:
                    if request_path is not None:
                        try:
                            latch_token = request_path.read_text(encoding="utf-8")
                        except OSError:
                            latch_requested = False
                    state = state_subscriber.get_msg(clear=True)
                    if latch_requested and state is not None:
                        try:
                            frozen_pose = extract_frozen_planner_pose(state)
                        except ValueError as exc:
                            print(f"[LaViRA Relay] Waiting for valid hold state: {exc}")
                        else:
                            latch_completed = True
                            print("[LaViRA Relay] Current upper body and hands latched")
                output.send(
                    planner.message(
                        latest.velocity(now, args.timeout),
                        period,
                        frozen_pose=frozen_pose,
                    )
                )
                if latch_completed:
                    request_still_valid = True
                    if request_path is not None:
                        try:
                            request_still_valid = (
                                request_path.read_text(encoding="utf-8")
                                == latch_token
                            )
                        except OSError:
                            request_still_valid = False
                    if ready_path is not None and request_still_valid:
                        ready_path.parent.mkdir(parents=True, exist_ok=True)
                        ready_path.write_text(latch_token, encoding="utf-8")
                next_tick = now + period
            time.sleep(min(0.002, max(0.0, next_tick - time.monotonic())))
    finally:
        zero = np.zeros(3, dtype=np.float32)
        for _ in range(3):
            output.send(planner.message(zero, period, frozen_pose=frozen_pose))
            time.sleep(0.02)
        output.send(build_command_message(start=False, stop=True, planner=True))
        if state_subscriber is not None:
            state_subscriber.close()
        if ready_path is not None:
            ready_path.unlink(missing_ok=True)
        if request_path is not None:
            request_path.unlink(missing_ok=True)
        source.close()
        output.close()


if __name__ == "__main__":
    main()
