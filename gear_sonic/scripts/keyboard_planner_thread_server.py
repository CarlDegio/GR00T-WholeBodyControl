#!/usr/bin/env python3
"""Publish REASEN/Navila velocity commands from an interactive keyboard."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import sys
import termios
import time
import tty
from typing import Sequence

import tyro
import zmq


MESSAGE_TYPE = "navila_reasan_velocity_command"
QUIT_KEY = "x"


@dataclass
class KeyboardPlannerConfig:
    debug: bool = False
    port: int = 5558
    host: str = "*"
    hz: float = 20.0
    """Compatibility option retained for launch_inference; commands remain key-triggered."""
    duration: float = 0.5
    forward_speed: float = 0.4
    backward_speed: float = 0.4
    lateral_speed: float = 0.4
    yaw_speed: float = 0.5


def build_navila_message(
    action: str,
    velocity: Sequence[float],
    duration_s: float,
    raw_text: str = "keyboard mock",
) -> str:
    vx, vy, wz = map(float, velocity)
    payload = {
        "action": action,
        "angle_rad": abs(wz) * duration_s,
        "distance_m": math.hypot(vx, vy) * duration_s,
        "duration_s": float(duration_s),
        "raw_text": raw_text,
        "segments": [{"duration_s": float(duration_s), "vx": vx, "vy": vy, "wz": wz}],
        "source": "keyboard_mock",
        "status": "ok",
        "type": MESSAGE_TYPE,
        "velocity": {"vx": vx, "vy": vy, "wz": wz},
        "version": 1,
    }
    return json.dumps(payload, separators=(",", ":"))


def key_commands(config: KeyboardPlannerConfig) -> dict[str, tuple[str, tuple[float, float, float]]]:
    return {
        "w": ("forward", (config.forward_speed, 0.0, 0.0)),
        "s": ("backward", (-config.backward_speed, 0.0, 0.0)),
        "a": ("move_left", (0.0, config.lateral_speed, 0.0)),
        "d": ("move_right", (0.0, -config.lateral_speed, 0.0)),
        "q": ("turn_left", (0.0, 0.0, config.yaw_speed)),
        "e": ("turn_right", (0.0, 0.0, -config.yaw_speed)),
        " ": ("stop", (0.0, 0.0, 0.0)),
    }


def main(config: KeyboardPlannerConfig) -> None:
    if config.duration <= 0.0:
        raise ValueError(f"duration must be positive, got {config.duration}")
    if not sys.stdin.isatty():
        raise RuntimeError("Keyboard input requires an interactive TTY/tmux pane")

    endpoint = f"tcp://{config.host}:{config.port}"
    context = zmq.Context.instance()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(endpoint)
    original_terminal = termios.tcgetattr(sys.stdin.fileno())
    commands = key_commands(config)
    stop_message = build_navila_message("stop", (0.0, 0.0, 0.0), config.duration)

    print(f"[REASEN Keyboard] PUB bound to {endpoint}; command duration={config.duration:g}s")
    print("[REASEN Keyboard] W/S forward/back | A/D lateral | Q/E yaw | Space stop | X exit")
    print(
        "[REASEN Keyboard] Translation speed: "
        f"W={config.forward_speed:g}, S={config.backward_speed:g}, "
        f"A/D={config.lateral_speed:g} m/s"
    )
    print("[REASEN Keyboard] JSON type: navila_reasan_velocity_command")
    time.sleep(0.3)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while True:
            key = sys.stdin.read(1).lower()
            if key == QUIT_KEY:
                break
            command = commands.get(key)
            if command is None:
                continue
            action, velocity = command
            message = build_navila_message(action, velocity, config.duration)
            socket.send_string(message)
            print(
                f"\r{action:>10}: vx={velocity[0]:+.2f} vy={velocity[1]:+.2f} "
                f"wz={velocity[2]:+.2f} duration={config.duration:g}s   ",
                end="",
                flush=True,
            )
            if config.debug:
                print(f"\n{message}")
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, original_terminal)
        for _ in range(3):
            socket.send_string(stop_message)
            time.sleep(0.02)
        socket.close()
        print("\n[REASEN Keyboard] Stopped")


if __name__ == "__main__":
    main(tyro.cli(KeyboardPlannerConfig))
