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
from gear_sonic.utils.teleop.sonic_orientation_telemetry import (
    OrientationTracker,
    encode_orientation_telemetry,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    build_planner_message,
)


COMMAND_MESSAGE_TYPE = "navila_reasan_velocity_command"
DEFAULT_MAX_LATERAL_SPEED_M_S = 0.16
COMMAND_LOWER = np.array(
    [-0.5, -DEFAULT_MAX_LATERAL_SPEED_M_S, -1.0], dtype=np.float32
)
COMMAND_UPPER = np.array(
    [1.0, DEFAULT_MAX_LATERAL_SPEED_M_S, 1.0], dtype=np.float32
)


def _bound_velocity_preserving_linear_ratio(
    velocity: np.ndarray, *, max_lateral_speed_m_s: float
) -> np.ndarray:
    limit = float(max_lateral_speed_m_s)
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("maximum lateral speed must be finite and positive")
    representable_limit = np.float32(
        min(limit, float(np.finfo(np.float32).max))
    )
    if float(representable_limit) > limit:
        representable_limit = np.nextafter(
            representable_limit, np.float32(0.0)
        )
    safe_limit = float(representable_limit)
    vx, vy, wz = map(float, velocity)
    scale = 1.0
    if vx > float(COMMAND_UPPER[0]):
        scale = min(scale, float(COMMAND_UPPER[0]) / vx)
    elif vx < float(COMMAND_LOWER[0]):
        scale = min(scale, float(COMMAND_LOWER[0]) / vx)
    if abs(vy) > safe_limit:
        scale = min(scale, safe_limit / abs(vy))
    return np.asarray(
        [vx * scale, vy * scale, np.clip(wz, COMMAND_LOWER[2], COMMAND_UPPER[2])],
        dtype=np.float32,
    )


def decode_velocity_command(
    raw: bytes | str,
    *,
    max_lateral_speed_m_s: float = DEFAULT_MAX_LATERAL_SPEED_M_S,
) -> dict[str, Any]:
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
        "velocity": _bound_velocity_preserving_linear_ratio(
            velocity, max_lateral_speed_m_s=max_lateral_speed_m_s
        ),
    }


@dataclass
class LatestCommand:
    value: dict[str, Any] | None = None
    received_at: float | None = None

    def update(self, value: dict[str, Any], now: float) -> None:
        self.value = value
        self.received_at = now

    def is_fresh(self, now: float, timeout: float) -> bool:
        if self.value is None or self.received_at is None:
            return False
        age = max(0.0, now - self.received_at)
        return age <= min(timeout, float(self.value["duration"]))

    def velocity(self, now: float, timeout: float) -> np.ndarray:
        if not self.is_fresh(now, timeout):
            return np.zeros(3, dtype=np.float32)
        assert self.value is not None
        return np.asarray(self.value["velocity"], dtype=np.float32).copy()


def select_velocity(
    automatic: LatestCommand,
    manual: LatestCommand | None,
    *,
    now: float,
    timeout: float,
) -> np.ndarray:
    if manual is not None and manual.is_fresh(now, timeout):
        return manual.velocity(now, timeout)
    return automatic.velocity(now, timeout)


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


def process_robot_state(
    state: Any,
    *,
    now_monotonic_s: float,
    heading_setpoint_rad: float,
    orientation_tracker: OrientationTracker | None,
    freeze_current_upper_body: bool,
) -> FrozenPlannerPose | None:
    """Update diagnostic orientation and independently honor pose freezing."""
    if orientation_tracker is not None:
        try:
            orientation_tracker.update_state(
                state,
                received_at_monotonic_s=now_monotonic_s,
                heading_setpoint_rad=heading_setpoint_rad,
            )
        except ValueError:
            # Orientation is observational. A malformed quaternion must not
            # block planner output or an explicitly requested pose latch.
            pass
    if not freeze_current_upper_body:
        return None
    return extract_frozen_planner_pose(state)


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
        # SONIC Planner accepts a target facing direction, not yaw velocity.
        # Treat upstream wz as a heading-setpoint slew rate and integrate it.
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
    parser.add_argument("--manual-source", default="")
    parser.add_argument("--output", default="tcp://*:5563")
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--timeout", type=float, default=0.7)
    parser.add_argument(
        "--max-lateral-speed-m-s",
        type=float,
        default=DEFAULT_MAX_LATERAL_SPEED_M_S,
    )
    parser.add_argument("--freeze-current-upper-body", action="store_true")
    parser.add_argument("--state-host", default="localhost")
    parser.add_argument("--state-port", type=int, default=5557)
    parser.add_argument("--hold-ready-file", default="")
    parser.add_argument("--orientation-telemetry-output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hz <= 0.0 or args.timeout <= 0.0:
        raise ValueError("hz and timeout must be positive")
    if (
        not math.isfinite(args.max_lateral_speed_m_s)
        or args.max_lateral_speed_m_s <= 0.0
    ):
        raise ValueError("maximum lateral speed must be finite and positive")
    context = zmq.Context.instance()
    source = context.socket(zmq.SUB)
    source.setsockopt(zmq.SUBSCRIBE, b"")
    source.setsockopt(zmq.CONFLATE, 1)
    source.setsockopt(zmq.LINGER, 0)
    source.connect(args.source)
    manual_source = None
    if args.manual_source:
        manual_source = context.socket(zmq.SUB)
        manual_source.setsockopt(zmq.SUBSCRIBE, b"")
        manual_source.setsockopt(zmq.CONFLATE, 1)
        manual_source.setsockopt(zmq.LINGER, 0)
        manual_source.connect(args.manual_source)
    output = context.socket(zmq.PUB)
    output.setsockopt(zmq.LINGER, 0)
    output.bind(args.output)
    orientation_output = None
    orientation_tracker = None
    if args.orientation_telemetry_output:
        orientation_output = context.socket(zmq.PUB)
        orientation_output.setsockopt(zmq.LINGER, 0)
        orientation_output.bind(args.orientation_telemetry_output)
        orientation_tracker = OrientationTracker()
    planner = PlannerState()
    latest = LatestCommand()
    manual_latest = LatestCommand() if manual_source is not None else None
    state_subscriber = None
    frozen_pose: FrozenPlannerPose | None = None
    ready_path = Path(args.hold_ready_file).resolve() if args.hold_ready_file else None
    request_path = Path(f"{ready_path}.request") if ready_path is not None else None
    if ready_path is not None:
        ready_path.unlink(missing_ok=True)
    if request_path is not None:
        request_path.unlink(missing_ok=True)
    if args.freeze_current_upper_body or orientation_tracker is not None:
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
        f"{args.hz:g} Hz, timeout={args.timeout:g}s, "
        f"max_lateral={args.max_lateral_speed_m_s:g}m/s"
    )
    if args.manual_source:
        print(f"[LaViRA Relay] exclusive manual source <- {args.manual_source}")
    if args.orientation_telemetry_output:
        print(
            "[LaViRA Relay] orientation telemetry -> "
            f"{args.orientation_telemetry_output}"
        )
    next_tick = time.monotonic()
    try:
        while running:
            inputs = [("automatic", source, latest)]
            if manual_source is not None and manual_latest is not None:
                inputs.append(("manual", manual_source, manual_latest))
            for label, input_socket, command_state in inputs:
                while True:
                    try:
                        raw = input_socket.recv(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    try:
                        command_state.update(
                            decode_velocity_command(
                                raw,
                                max_lateral_speed_m_s=args.max_lateral_speed_m_s,
                            ),
                            time.monotonic(),
                        )
                    except ValueError as exc:
                        print(f"[LaViRA Relay] Ignored {label} command: {exc}")
            now = time.monotonic()
            if now >= next_tick:
                latch_completed = False
                latch_token = "ready\n"
                latch_requested = args.freeze_current_upper_body and (
                    (request_path is None and frozen_pose is None)
                    or (request_path is not None and request_path.is_file())
                )
                if state_subscriber is not None:
                    if latch_requested and request_path is not None:
                        try:
                            latch_token = request_path.read_text(encoding="utf-8")
                        except OSError:
                            latch_requested = False
                    state = state_subscriber.get_msg(clear=True)
                    if state is not None:
                        sampled_pose = None
                        try:
                            sampled_pose = process_robot_state(
                                state,
                                now_monotonic_s=now,
                                heading_setpoint_rad=planner.heading,
                                orientation_tracker=orientation_tracker,
                                freeze_current_upper_body=latch_requested,
                            )
                        except ValueError as exc:
                            if latch_requested:
                                print(
                                    "[LaViRA Relay] Waiting for valid hold state: "
                                    f"{exc}"
                                )
                        if sampled_pose is not None:
                            frozen_pose = sampled_pose
                            latch_completed = True
                            print("[LaViRA Relay] Current upper body and hands latched")
                planner_message = planner.message(
                    select_velocity(
                        latest, manual_latest, now=now, timeout=args.timeout
                    ),
                    period,
                    frozen_pose=frozen_pose,
                )
                output.send(planner_message)
                if orientation_output is not None and orientation_tracker is not None:
                    orientation_output.send_string(
                        encode_orientation_telemetry(
                            orientation_tracker.sample(now, planner.heading)
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
        if orientation_output is not None:
            orientation_output.close()
        if manual_source is not None:
            manual_source.close()
        source.close()
        output.close()


if __name__ == "__main__":
    main()
