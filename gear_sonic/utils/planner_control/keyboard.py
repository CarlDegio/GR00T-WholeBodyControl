#!/usr/bin/env python3
"""Send interactive navigation keys through SonicControlGateway."""

from __future__ import annotations

from dataclasses import dataclass
import sys
import termios
import time
import tty

from gear_sonic.runtime.profile import load_runtime_profile
from gear_sonic.runtime.gateway.control_client import ControlGatewayIntentClient


QUIT_KEY = "x"
NAVIGATION_KEYS = frozenset(("w", "s", "a", "d", "q", "e", " "))


@dataclass
class KeyboardPlannerConfig:
    profile: str = ""
    overlay: tuple[str, ...] = ()
    debug: bool = False


def main(config: KeyboardPlannerConfig) -> None:
    if not sys.stdin.isatty():
        raise RuntimeError("Keyboard input requires an interactive TTY/tmux pane")

    profile = load_runtime_profile(config.profile or None, overlays=config.overlay)
    endpoint = profile.endpoint_uri("control_gateway_intent")
    intent = ControlGatewayIntentClient(
        endpoint,
        source="keyboard_planner",
    )
    original_terminal = termios.tcgetattr(sys.stdin.fileno())

    print(f"[Planner Keyboard] ControlGateway intent: {endpoint}")
    print("[Planner Keyboard] W/S forward/back | A/D lateral | Q/E yaw | Space stop | X exit")
    time.sleep(0.3)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while True:
            key = sys.stdin.read(1).lower()
            if key == QUIT_KEY:
                break
            if key not in NAVIGATION_KEYS:
                continue
            intent.send("navigation_key", {"key": key})
            label = "Space" if key == " " else key.upper()
            print(f"\rSent navigation key: {label:<5}", end="", flush=True)
            if config.debug:
                print(f"\nControlGateway navigation_key={key!r}")
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, original_terminal)
        for _ in range(3):
            intent.send("navigation_key", {"key": " ", "reason": "keyboard_exit"})
            time.sleep(0.02)
        intent.close()
        print("\n[Planner Keyboard] Stopped")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(KeyboardPlannerConfig))
