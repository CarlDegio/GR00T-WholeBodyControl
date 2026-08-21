#!/usr/bin/env python3
"""Relay asynchronous REASEN Filter velocities into SONIC planner messages."""

from __future__ import annotations

import argparse
import math
import signal
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import zmq


MESSAGE_PREFIX = b"filter_velocity"
PACKET_VELOCITY = 0
PACKET_STOP = 1
_PACKET = struct.Struct("<BQfff")

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    build_planner_message,
)


class LatestVelocity:
    def __init__(self, timeout_s: float):
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        self.timeout_s = timeout_s
        self._velocity = (0.0, 0.0, 0.0)
        self._updated_at: float | None = None

    def update(self, velocity: tuple[float, float, float], now: float) -> None:
        self._velocity = velocity
        self._updated_at = now

    def value(self, now: float) -> tuple[float, float, float]:
        if self._updated_at is None or now - self._updated_at >= self.timeout_s:
            return (0.0, 0.0, 0.0)
        return self._velocity


class StartHeartbeat:
    def __init__(self, interval_s: float):
        self.interval_s = interval_s
        self._next_at = float("-inf")

    def due(self, now: float) -> bool:
        if now < self._next_at:
            return False
        self._next_at = now + self.interval_s
        return True

    def reset(self) -> None:
        self._next_at = float("-inf")


class ReadySignal:
    def __init__(self, path: Path | None):
        self.path = path
        self.clear()

    def mark(self) -> None:
        if self.path is not None:
            self.path.touch()

    def clear(self) -> None:
        if self.path is not None:
            self.path.unlink(missing_ok=True)


@dataclass
class PlannerState:
    heading: float = 0.0


def build_planner_from_velocity(
    state: PlannerState, velocity: tuple[float, float, float], dt: float
) -> bytes:
    vx, vy, wz = velocity
    state.heading = math.remainder(state.heading + wz * dt, 2.0 * math.pi)
    cosine = math.cos(state.heading)
    sine = math.sin(state.heading)
    world_x = cosine * vx - sine * vy
    world_y = sine * vx + cosine * vy
    speed = math.hypot(world_x, world_y)
    if speed < 1.0e-6:
        mode = 0
        movement = (0.0, 0.0, 0.0)
        speed = 0.0
    else:
        mode = 1 if speed < 0.8 else 2
        movement = (world_x / speed, world_y / speed, 0.0)
    facing = (cosine, sine, 0.0)
    return build_planner_message(mode, movement, facing, speed=speed, height=-1.0)


def decode_filter_packet(packet: bytes) -> tuple[int, int, tuple[float, float, float]]:
    if not packet.startswith(MESSAGE_PREFIX) or len(packet) != len(MESSAGE_PREFIX) + _PACKET.size:
        raise ValueError("Malformed Filter velocity packet")
    kind, sequence, vx, vy, wz = _PACKET.unpack_from(packet, len(MESSAGE_PREFIX))
    if kind not in (PACKET_VELOCITY, PACKET_STOP):
        raise ValueError(f"Unsupported Filter packet kind: {kind}")
    return kind, sequence, (vx, vy, wz)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--ready-file", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hz <= 0.0:
        raise ValueError("hz must be positive")
    context = zmq.Context.instance()
    source = context.socket(zmq.SUB)
    source.setsockopt(zmq.SUBSCRIBE, MESSAGE_PREFIX)
    source.setsockopt(zmq.CONFLATE, 1)
    source.setsockopt(zmq.LINGER, 0)
    source.connect(args.source)
    output = context.socket(zmq.PUB)
    output.setsockopt(zmq.LINGER, 0)
    output.bind(args.output)
    latest = LatestVelocity(args.timeout)
    planner = PlannerState()
    period = 1.0 / args.hz
    running = True
    control_started = False
    start_heartbeat = StartHeartbeat(0.5)
    ready_signal = ReadySignal(args.ready_file)

    def request_stop(_signum=None, _frame=None):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    time.sleep(0.5)
    next_tick = time.monotonic()
    print(f"[relay] Filter {args.source} -> SONIC {args.output}, {args.hz:g} Hz, timeout={args.timeout:g}s")
    try:
        while running:
            while True:
                try:
                    packet = source.recv(zmq.NOBLOCK)
                except zmq.Again:
                    break
                try:
                    kind, _sequence, velocity = decode_filter_packet(packet)
                except ValueError as error:
                    print(f"[relay] Ignored packet: {error}", file=sys.stderr)
                    continue
                now = time.monotonic()
                if kind == PACKET_STOP:
                    latest.update((0.0, 0.0, 0.0), now)
                    if control_started:
                        output.send(build_planner_from_velocity(planner, (0.0, 0.0, 0.0), period))
                        output.send(build_command_message(start=False, stop=True, planner=True))
                        control_started = False
                        start_heartbeat.reset()
                    continue
                latest.update(velocity, now)
                ready_signal.mark()
                if not control_started:
                    control_started = True

            now = time.monotonic()
            if control_started and start_heartbeat.due(now):
                output.send(build_command_message(start=True, stop=False, planner=True))
            if now >= next_tick:
                output.send(build_planner_from_velocity(planner, latest.value(now), period))
                next_tick += period
                if next_tick <= now:
                    next_tick = now + period
            time.sleep(min(0.002, max(0.0, next_tick - time.monotonic())))
    finally:
        zero = build_planner_from_velocity(planner, (0.0, 0.0, 0.0), period)
        for _ in range(3):
            output.send(zero)
            time.sleep(0.02)
        output.send(build_command_message(start=False, stop=True, planner=True))
        ready_signal.clear()
        source.close()
        output.close()


if __name__ == "__main__":
    main()
