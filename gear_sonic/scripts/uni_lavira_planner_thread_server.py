"""Standalone Uni-LaViRA JSON-to-Sonic planner sidecar.

This is an alternative to ``keyboard_planner_thread_server.py``. Both bind
planner PUB port 5558, so run exactly one of them. ``run_vla_inference.py``
keeps relaying the selected sidecar's ``planner`` messages to C++ on port 5556.

Start this process before the operator sends ``k`` on keyboard port 5580.
The sidecar mirrors ``k``/``i``/``o`` state so JSON is accepted only while the
controller is believed to be running in PLANNER mode.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

from gear_sonic.utils.inference.uni_lavira_planner import (
    PlannerModeTracker,
    PlannerOutput,
    UniLaviraJsonBridge,
    UniLaviraPlannerExecutor,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_planner_message


@dataclass
class UniLaviraPlannerConfig:
    """CLI configuration for the standalone JSON planner sidecar."""

    json_host: str = "127.0.0.1"
    """Bind host for acknowledged Uni-LaViRA JSON requests."""

    json_port: int = 5559
    """REP port for Uni-LaViRA JSON requests."""

    planner_host: str = "*"
    """Bind host for planner messages consumed by run_vla_inference."""

    planner_port: int = 5558
    """Planner PUB port; mutually exclusive with the keyboard planner sidecar."""

    keyboard_host: str = "localhost"
    """Host of the operator keyboard publisher."""

    keyboard_port: int = 5580
    """Keyboard PUB port used to mirror k/i/o control state."""

    hz: float = 20.0
    """Planner publication and state-machine update rate."""

    max_speed: float = 0.5
    """Maximum accepted translation speed in metres per second."""

    max_duration: float = 30.0
    """Maximum accepted duration for each command in seconds."""

    max_abs_yaw: float = math.pi
    """Maximum accepted absolute relative yaw in radians."""


def publish_output(socket: Any, output: PlannerOutput) -> None:
    """Publish one planner state using the same wire builder as keyboard mode."""
    socket.send(
        build_planner_message(
            output.mode,
            output.movement,
            output.facing,
            speed=output.speed,
            height=output.height,
        )
    )


def _best_effort_abort(
    bridge: UniLaviraJsonBridge, reason: str
) -> PlannerOutput | None:
    """Stop executor state even when the REP abort reply cannot be delivered."""
    try:
        return bridge.abort(reason)
    except Exception as exc:
        print(f"[UniLaviraPlanner] abort reply failed: {exc}")
        bridge.pending_reply = False
        return bridge.executor.abort(reason)


def drain_keyboard_commands(
    socket: Any,
    tracker: PlannerModeTracker,
    bridge: UniLaviraJsonBridge,
) -> PlannerOutput | None:
    """Drain operator commands, returning the latest required stopped output."""
    stopped = None
    while socket.poll(0):
        command = socket.recv_string()
        was_running = tracker.running
        reason = tracker.apply(command)
        if not was_running and tracker.planner_ready:
            bridge.reset_control_session()
        if reason is not None:
            candidate = _best_effort_abort(bridge, reason)
            if candidate is not None:
                stopped = candidate
        if command.strip().lower() in {"k", "i", "o"}:
            print(
                f"[UniLaviraPlanner] keyboard={command.strip()!r} "
                f"running={tracker.running} mode={tracker.mode}"
            )
    return stopped


def _sleep_remaining(started: float, period: float) -> None:
    remaining = period - (time.monotonic() - started)
    if remaining > 0:
        time.sleep(remaining)


def main(config: UniLaviraPlannerConfig) -> None:
    """Run the non-blocking JSON REP, keyboard SUB, and planner PUB loop."""
    if not math.isfinite(config.hz) or config.hz <= 0:
        raise ValueError(f"hz must be finite and positive, got {config.hz}")

    import zmq

    context = zmq.Context()
    json_rep = context.socket(zmq.REP)
    planner_pub = context.socket(zmq.PUB)
    keyboard_sub = context.socket(zmq.SUB)
    for socket in (json_rep, planner_pub, keyboard_sub):
        socket.setsockopt(zmq.LINGER, 0)

    json_endpoint = f"tcp://{config.json_host}:{config.json_port}"
    planner_endpoint = f"tcp://{config.planner_host}:{config.planner_port}"
    keyboard_endpoint = f"tcp://{config.keyboard_host}:{config.keyboard_port}"
    json_rep.bind(json_endpoint)
    planner_pub.bind(planner_endpoint)
    keyboard_sub.setsockopt_string(zmq.SUBSCRIBE, "")
    keyboard_sub.connect(keyboard_endpoint)

    bridge = UniLaviraJsonBridge(
        json_rep,
        UniLaviraPlannerExecutor(
            max_speed=config.max_speed,
            max_duration=config.max_duration,
            max_abs_yaw=config.max_abs_yaw,
        ),
    )
    tracker = PlannerModeTracker()
    period = 1.0 / config.hz

    time.sleep(0.1)
    print(f"[UniLaviraPlanner] JSON REP bound to {json_endpoint}")
    print(f"[UniLaviraPlanner] planner PUB bound to {planner_endpoint}")
    print(f"[UniLaviraPlanner] keyboard SUB connected to {keyboard_endpoint}")
    print("[UniLaviraPlanner] start this sidecar before pressing k")

    try:
        while True:
            started = time.monotonic()
            keyboard_output = drain_keyboard_commands(
                keyboard_sub, tracker, bridge
            )
            if keyboard_output is not None:
                publish_output(planner_pub, keyboard_output)

            try:
                output = bridge.step(
                    now=time.monotonic(),
                    planner_ready=tracker.planner_ready,
                )
                if output is not None:
                    publish_output(planner_pub, output)
                    bridge.acknowledge_output_published()
            except Exception as exc:
                print(f"[UniLaviraPlanner] bridge failure: {exc}")
                stopped = _best_effort_abort(bridge, "internal_error")
                if stopped is not None:
                    publish_output(planner_pub, stopped)

            _sleep_remaining(started, period)
    except KeyboardInterrupt:
        print("[UniLaviraPlanner] interrupted")
    finally:
        stopped = _best_effort_abort(bridge, "shutdown")
        if stopped is not None:
            for _ in range(5):
                try:
                    publish_output(planner_pub, stopped)
                except Exception as exc:
                    print(f"[UniLaviraPlanner] stop publish failed: {exc}")
                    break
                time.sleep(0.02)
        keyboard_sub.close()
        json_rep.close()
        planner_pub.close()
        context.term()
        print("[UniLaviraPlanner] shutdown complete")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(UniLaviraPlannerConfig))

