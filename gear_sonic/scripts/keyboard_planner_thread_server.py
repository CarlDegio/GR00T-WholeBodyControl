"""
Keyboard planner sidecar for G1 locomotion via ZMQ.

Publishes ``planner`` topic messages to a local PUB socket (default :5558).
``run_vla_inference.py`` relays those bytes to the C++ deploy on :5556.

Locomotion is fixed to SLOW_WALK (LocomotionMode=1). Keys:
  W/S       forward / backward (speed command along current facing)
  A/D       heading left / right (±π/6 rad per stdin event; matches C++ Q/E logic)

Movement for W/S is hold-to-walk / release-to-stop. Planner ``mode`` stays
``SLOW_WALK`` while the sidecar runs; only ``movement``/``speed`` change.
This avoids ``IDLE``↔``SLOW_WALK`` flapping (which triggers replans). Stdin
has no key-up events, so W/S use a longer hold window (~550 ms) to bridge
Linux key-repeat initial delay.

Start/stop and mode switching (``command`` topic) are handled exclusively by
``run_vla_inference.py`` (5580 keyboard: ``k`` start/stop, ``i`` POSE, ``o`` PLANNER).
This sidecar never sends ``command`` messages — only ``planner`` locomotion at :5558.

Typical workflow:
  1. Start ``run_vla_inference`` and C++ deploy
  2. In the inference keyboard terminal, press ``k`` then ``o`` (PLANNER mode)
  3. Focus this tmux pane and use W/S/A/D here
  4. Press ``i`` in the inference terminal to switch to VLA POSE mode

Usage:
    python gear_sonic/scripts/keyboard_planner_thread_server.py
    python gear_sonic/scripts/keyboard_planner_thread_server.py --debug --max-speed 0.5
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import select
import sys
import termios
import time
import tty

import tyro
import zmq

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_planner_message

LOCOMOTION_MODE_SLOW_WALK = 1
LOCOMOTION_MODE_IDLE = 0

HEADING_STEP = math.pi / 6.0
# cbreak has no key-up; bridge Linux key-repeat delay (~500 ms) for W/S hold.
MOVEMENT_KEY_HOLD_TIMEOUT_SEC = 0.55
STDIN_KEY_DEBOUNCE_SEC = 0.12
MOVEMENT_KEYS = frozenset({"w", "s"})


@dataclass
class KeyboardPlannerConfig:
    """CLI config for the keyboard planner sidecar."""

    debug: bool = False
    """Enable debug mode."""

    max_speed: float = 0.3
    """Locomotion speed (m/s) when W/S is active."""

    port: int = 5558
    """ZMQ PUB port (``run_vla_inference`` relay SUB connects here)."""

    host: str = "*"
    """ZMQ bind host."""

    hz: float = 20.0
    """Planner ZMQ publish rate (Hz)."""


class WASDKeyboardHandler:
    """Track WASD from terminal stdin (cbreak). Works in tmux on Wayland/X11."""

    TRACKED_KEYS = frozenset({"w", "a", "s", "d"})

    def __init__(
        self,
        debug: bool = False,
        movement_hold_timeout: float = MOVEMENT_KEY_HOLD_TIMEOUT_SEC,
        debounce_sec: float = STDIN_KEY_DEBOUNCE_SEC,
    ):
        self._debug = debug
        self._movement_hold_timeout = movement_hold_timeout
        self._debounce_sec = debounce_sec
        self._key_times: dict[str, float] = {}
        self._last_accept_time: dict[str, float] = {}
        self._fd: int | None = None
        self._old_settings = None

        if not sys.stdin.isatty():
            raise RuntimeError(
                "Keyboard input requires a TTY. Run in the planner tmux pane, not piped/background."
            )

        fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        self._fd = fd

    def _poll_one(self) -> str | None:
        if self._fd is None:
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            return None
        return os.read(self._fd, 1).decode(errors="ignore")

    def poll(self) -> tuple[int, int]:
        """Drain stdin; return (left_heading_steps, right_heading_steps) for this poll."""
        now = time.monotonic()
        heading_left = 0
        heading_right = 0
        while True:
            key_char = self._poll_one()
            if not key_char:
                break
            normalized = key_char.lower()
            if normalized not in self.TRACKED_KEYS:
                continue

            last_accept = self._last_accept_time.get(normalized, 0.0)
            if now - last_accept < self._debounce_sec:
                continue

            self._last_accept_time[normalized] = now
            self._key_times[normalized] = now
            if normalized == "a":
                heading_right += 1
            elif normalized == "d":
                heading_left += 1
            if self._debug:
                print(f"[KeyboardPlanner] key: {normalized!r}")
        return heading_left, heading_right

    def snapshot(self) -> set[str]:
        now = time.monotonic()
        active = {
            key
            for key, last_seen in self._key_times.items()
            if key in MOVEMENT_KEYS
            and now - last_seen <= self._movement_hold_timeout
        }
        self._key_times = {
            key: last_seen
            for key, last_seen in self._key_times.items()
            if key in active
        }
        return self._resolve_opposing_keys(active)

    def _resolve_opposing_keys(self, keys: set[str]) -> set[str]:
        """Keep the most recently pressed key when W/S or A/D are both active."""
        resolved = set(keys)
        for left, right in (("w", "s"), ("a", "d")):
            if left in resolved and right in resolved:
                left_time = self._key_times.get(left, 0.0)
                right_time = self._key_times.get(right, 0.0)
                resolved.discard(left if right_time >= left_time else right)
        return resolved

    def close(self) -> None:
        if self._fd is not None and self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
            self._fd = None
            self._old_settings = None


class PlannerKeyboardController:
    """Convert held WASD keys into planner movement/facing (SLOW_WALK only)."""

    def __init__(self, debug: bool = False, max_speed: float = 0.5):
        self._debug = debug
        self.max_speed = max_speed
        self.facing_angle = 0.0
        self._walk_active = False
        self._last_facing_deg = 0.0

    def _facing_vector(self) -> list[float]:
        return [math.cos(self.facing_angle), math.sin(self.facing_angle), 0.0]

    def update_from_keys(
        self,
        keys: set[str],
        heading_left_steps: int = 0,
        heading_right_steps: int = 0,
    ) -> tuple[int, list[float], list[float], float, float]:
        # Heading controls are independent of movement momentum (C++ Q/E / planner_heading_*).
        if heading_left_steps and heading_right_steps:
            net = heading_right_steps - heading_left_steps
            self.facing_angle += net * HEADING_STEP
        else:
            self.facing_angle -= heading_left_steps * HEADING_STEP
            self.facing_angle += heading_right_steps * HEADING_STEP

        facing = self._facing_vector()
        walk_active = "w" in keys or "s" in keys

        if "w" in keys:
            movement_out = facing.copy()
        elif "s" in keys:
            movement_out = [-facing[0], -facing[1], 0.0]
        else:
            movement_out = [0.0, 0.0, 0.0]

        mode = LOCOMOTION_MODE_SLOW_WALK
        speed = self.max_speed if walk_active else 0.0

        facing_deg = math.degrees(self.facing_angle)
        if self._debug and walk_active != self._walk_active:
            print(
                f"[KeyboardPlanner] walk {'on' if walk_active else 'off'} | "
                f"keys={sorted(keys)} speed={speed} movement={movement_out} facing={facing}"
            )
            self._walk_active = walk_active
        elif self._debug and (
            heading_left_steps or heading_right_steps or abs(facing_deg - self._last_facing_deg) > 0.01
        ):
            print(
                f"[KeyboardPlanner] heading L={heading_left_steps} R={heading_right_steps} "
                f"keys={sorted(keys)} mode={mode} speed={speed} facing_deg={facing_deg:.1f}"
            )
        self._last_facing_deg = facing_deg

        return mode, movement_out, facing, speed, -1.0


def _sleep_remaining(t_start: float, period: float):
    remaining = period - (time.monotonic() - t_start)
    if remaining > 0:
        time.sleep(remaining)


def main(config: KeyboardPlannerConfig):
    if config.hz <= 0:
        raise ValueError(f"hz must be positive, got {config.hz}")

    period = 1.0 / config.hz
    keyboard_handler = WASDKeyboardHandler(debug=config.debug)
    controller = PlannerKeyboardController(config.debug, max_speed=config.max_speed)

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.LINGER, 0)
    endpoint = f"tcp://{config.host}:{config.port}"
    pub.bind(endpoint)
    time.sleep(0.1)

    print(f"[KeyboardPlanner] PUB bound to {endpoint} at {config.hz:.0f} Hz")
    print("[KeyboardPlanner] Locomotion: SLOW_WALK only")
    print("[KeyboardPlanner] Keys: W/S hold=walk  A/D heading ±30°  mode fixed SLOW_WALK")
    print("[KeyboardPlanner] command topic: use run_vla_inference keyboard (k=start, i=POSE, o=PLANNER)")
    print("[KeyboardPlanner] Press 'k' then 'o' in run_vla_inference first.")
    print("[KeyboardPlanner] Focus this tmux pane, then hold W/A/S/D.")
    print("[KeyboardPlanner] Ctrl+C to exit.")

    try:
        while True:
            t_start = time.monotonic()
            heading_left, heading_right = keyboard_handler.poll()
            keys = keyboard_handler.snapshot()
            mode, movement, facing, speed, height = controller.update_from_keys(
                keys,
                heading_left_steps=heading_left,
                heading_right_steps=heading_right,
            )
            pub.send(
                build_planner_message(
                    mode,
                    movement,
                    facing,
                    speed=speed,
                    height=height,
                )
            )
            _sleep_remaining(t_start, period)
    except KeyboardInterrupt:
        print("\n[KeyboardPlanner] Shutting down...")
    finally:
        try:
            pub.send(
                build_planner_message(
                    LOCOMOTION_MODE_IDLE,
                    [0.0, 0.0, 0.0],
                    controller._facing_vector(),
                    speed=-1.0,
                    height=-1.0,
                )
            )
        except Exception:
            pass
        keyboard_handler.close()
        pub.close()
        ctx.term()
        print("[KeyboardPlanner] Done.")


if __name__ == "__main__":
    main(tyro.cli(KeyboardPlannerConfig))
