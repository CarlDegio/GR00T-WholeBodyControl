"""Standalone Uni-LaViRA JSON-to-Sonic planner controller.

This process replaces ``run_vla_inference.py`` for JSON-only locomotion. It
owns the action PUB port 5556, sends ``command`` messages from keyboard port
5580, and sends Uni-LaViRA ``planner`` messages directly to Sonic C++.

Do not run this process together with ``run_vla_inference.py`` because both
bind the action PUB port 5556.
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
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    build_planner_message,
)


@dataclass
class UniLaviraPlannerConfig:
    """CLI configuration for direct Uni-LaViRA control of Sonic deploy."""

    json_host: str = "127.0.0.1"
    """Bind host for acknowledged Uni-LaViRA JSON requests."""

    json_port: int = 5559
    """REP port for Uni-LaViRA JSON requests."""

    action_host: str = "*"
    """Bind host for command/planner messages consumed by Sonic C++."""

    action_port: int = 5556
    """Direct Sonic C++ action PUB port; conflicts with run_vla_inference."""

    keyboard_host: str = "localhost"
    """Host of the operator keyboard publisher."""

    keyboard_port: int = 5580
    """Keyboard PUB port used for k/i/o control."""

    hz: float = 20.0
    """Planner publication and state-machine update rate."""

    max_speed: float = 0.5
    """Maximum accepted translation speed in metres per second."""

    max_duration: float = 30.0
    """Maximum accepted duration for each command in seconds."""

    max_abs_yaw: float = math.pi
    """Maximum accepted absolute relative yaw in radians."""


class ActionSubscriberTracker:
    """Track whether Sonic C++ subscribed to both direct-control topics."""

    REQUIRED_TOPICS = (b"command", b"planner")

    def __init__(self):
        self._counts = {topic: 0 for topic in self.REQUIRED_TOPICS}

    @property
    def ready(self) -> bool:
        return all(self._counts[topic] > 0 for topic in self.REQUIRED_TOPICS)

    def observe(self, event: bytes) -> None:
        if not event:
            return
        topic = bytes(event[1:])
        if topic not in self._counts:
            return
        if event[0] == 1:
            self._counts[topic] += 1
        elif event[0] == 0:
            self._counts[topic] = max(0, self._counts[topic] - 1)


def drain_action_subscriptions(
    socket: Any, tracker: ActionSubscriberTracker
) -> int:
    count = 0
    while socket.poll(0):
        tracker.observe(socket.recv())
        count += 1
    return count


def publish_output(socket: Any, output: PlannerOutput) -> None:
    """Publish one planner state directly to Sonic C++."""
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
    """Abort motion, exiting fail-closed if the REP reply cannot be sent."""
    try:
        return bridge.abort(reason)
    except Exception as exc:
        print(f"[UniLaviraPlanner] abort reply failed: {exc}")
        raise


def _control_message_for_key(
    command: str, tracker: PlannerModeTracker
) -> bytes | None:
    normalized = command.strip().lower()
    if normalized == "k":
        if not tracker.running:
            return build_command_message(start=True, stop=False, planner=True)
        return build_command_message(
            start=False, stop=True, planner=tracker.mode == "PLANNER"
        )
    if normalized == "i" and tracker.running:
        return build_command_message(
            start=False, stop=True, planner=tracker.mode == "PLANNER"
        )
    if normalized == "o" and tracker.running:
        return build_command_message(start=True, stop=False, planner=True)
    return None


def drain_keyboard_commands(
    socket: Any,
    tracker: PlannerModeTracker,
    bridge: UniLaviraJsonBridge,
    action_socket: Any,
    *,
    action_ready: bool = True,
) -> PlannerOutput | None:
    """Send k/i/o to C++ and mirror only successfully published transitions."""
    stopped = None
    while socket.poll(0):
        command = socket.recv_string()
        if not action_ready:
            print(
                f"[UniLaviraPlanner] ignored {command.strip()!r}: "
                "Sonic command/planner subscribers are not ready"
            )
            continue
        message = _control_message_for_key(command, tracker)
        if message is None:
            continue

        was_running = tracker.running
        action_socket.send(message)
        transition = command
        if command.strip().lower() == "i":
            transition = "k"
        reason = tracker.apply(transition)
        if not was_running and tracker.planner_ready:
            bridge.reset_control_session()
        if reason is not None:
            candidate = _best_effort_abort(bridge, reason)
            if candidate is not None:
                stopped = candidate
        print(
            f"[UniLaviraPlanner] keyboard={command.strip()!r} "
            f"running={tracker.running} mode={tracker.mode}"
        )
    return stopped


def step_bridge_once(
    bridge: UniLaviraJsonBridge,
    *,
    planner_ready: bool,
    action_socket: Any,
    now: float,
) -> PlannerOutput | None:
    """Advance and publish once; any REP/PUB failure is fatal to the loop."""
    output = bridge.step(now=now, planner_ready=planner_ready)
    if output is not None:
        publish_output(action_socket, output)
        bridge.acknowledge_output_published()
    return output


def handle_deploy_disconnect(
    tracker: PlannerModeTracker,
    bridge: UniLaviraJsonBridge,
    action_socket: Any,
) -> None:
    """End the old control session; a reconnected deploy requires a fresh k."""
    was_planner = tracker.mode == "PLANNER"
    action_socket.send(
        build_command_message(
            start=False, stop=True, planner=was_planner
        )
    )
    if tracker.running:
        tracker.apply("k")
    stopped = _best_effort_abort(bridge, "deploy_disconnected")
    if stopped is None:
        stopped = bridge.executor.abort("deploy_disconnected")
    publish_output(action_socket, stopped)


def _sleep_remaining(started: float, period: float) -> None:
    remaining = period - (time.monotonic() - started)
    if remaining > 0:
        time.sleep(remaining)


def main(config: UniLaviraPlannerConfig) -> None:
    """Run direct JSON REP, keyboard SUB, and Sonic action PUB control."""
    if not math.isfinite(config.hz) or config.hz <= 0:
        raise ValueError(f"hz must be finite and positive, got {config.hz}")

    import zmq

    context = zmq.Context()
    json_rep = context.socket(zmq.REP)
    action_pub = context.socket(zmq.XPUB)
    keyboard_sub = context.socket(zmq.SUB)
    for socket in (json_rep, action_pub, keyboard_sub):
        socket.setsockopt(zmq.LINGER, 0)
    action_pub.setsockopt(zmq.XPUB_VERBOSE, 1)

    json_endpoint = f"tcp://{config.json_host}:{config.json_port}"
    action_endpoint = f"tcp://{config.action_host}:{config.action_port}"
    keyboard_endpoint = f"tcp://{config.keyboard_host}:{config.keyboard_port}"
    json_rep.bind(json_endpoint)
    action_pub.bind(action_endpoint)
    keyboard_sub.setsockopt_string(zmq.SUBSCRIBE, "")
    keyboard_sub.connect(keyboard_endpoint)

    bridge = UniLaviraJsonBridge(
        json_rep,
        UniLaviraPlannerExecutor(
            max_speed=config.max_speed,
            max_duration=config.max_duration,
            max_abs_yaw=config.max_abs_yaw,
            min_positive_duration=1.0 / config.hz,
        ),
    )
    tracker = PlannerModeTracker()
    action_subscribers = ActionSubscriberTracker()
    period = 1.0 / config.hz

    time.sleep(0.1)
    print(f"[UniLaviraPlanner] JSON REP bound to {json_endpoint}")
    print(f"[UniLaviraPlanner] Sonic action PUB bound to {action_endpoint}")
    print(f"[UniLaviraPlanner] keyboard SUB connected to {keyboard_endpoint}")
    print("[UniLaviraPlanner] press k after this process and Sonic deploy are ready")

    try:
        while True:
            started = time.monotonic()
            was_action_ready = action_subscribers.ready
            drain_action_subscriptions(action_pub, action_subscribers)
            if action_subscribers.ready and not was_action_ready:
                print("[UniLaviraPlanner] Sonic command/planner subscribers ready")
            elif was_action_ready and not action_subscribers.ready:
                print("[UniLaviraPlanner] Sonic subscriber disconnected")
                handle_deploy_disconnect(
                    tracker, bridge, action_pub
                )

            keyboard_output = drain_keyboard_commands(
                keyboard_sub, tracker, bridge, action_pub,
                action_ready=action_subscribers.ready,
            )
            if keyboard_output is not None:
                publish_output(action_pub, keyboard_output)

            step_bridge_once(
                bridge,
                planner_ready=(
                    tracker.planner_ready and action_subscribers.ready
                ),
                action_socket=action_pub,
                now=time.monotonic(),
            )

            _sleep_remaining(started, period)
    except KeyboardInterrupt:
        print("[UniLaviraPlanner] interrupted")
    finally:
        try:
            stopped = _best_effort_abort(bridge, "shutdown")
        except Exception:
            stopped = bridge.executor.abort("shutdown")
        if stopped is None:
            stopped = bridge.executor.abort("shutdown")
        try:
            action_pub.send(
                build_command_message(start=False, stop=True, planner=False)
            )
            for _ in range(5):
                publish_output(action_pub, stopped)
                time.sleep(0.02)
        except Exception as exc:
            print(f"[UniLaviraPlanner] shutdown publish failed: {exc}")
        keyboard_sub.close()
        json_rep.close()
        action_pub.close()
        context.term()
        print("[UniLaviraPlanner] shutdown complete")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(UniLaviraPlannerConfig))
