"""Orchestrated wrapper around the existing Uni-LaViRA planner server."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

from gear_sonic.scripts import uni_lavira_planner_thread_server as base
from gear_sonic.utils.inference.uni_lavira_planner import (
    PlannerModeTracker,
    UniLaviraJsonBridge,
    UniLaviraPlannerExecutor,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message


@dataclass
class ControlledUniLaviraPlannerConfig(base.UniLaviraPlannerConfig):
    """Add a localhost-only lifecycle endpoint to the existing server."""

    control_host: str = "127.0.0.1"
    control_port: int = 5560


def start_planner(
    tracker: PlannerModeTracker,
    bridge: UniLaviraJsonBridge,
    action_socket: Any,
    *,
    action_ready: bool,
) -> dict[str, Any]:
    """Idempotently start SONIC in PLANNER mode."""
    if not action_ready:
        return {"status": "not_ready", "reason": "action_subscribers_not_ready"}
    if tracker.planner_ready:
        return {"status": "ready", "planner_ready": True}
    if tracker.running:
        return {"status": "rejected", "reason": "control_running_outside_planner"}
    action_socket.send(build_command_message(start=True, stop=False, planner=True))
    tracker.apply("k")
    bridge.reset_control_session()
    print("[UniLaviraPlanner] orchestrator started PLANNER mode")
    return {"status": "ready", "planner_ready": True}


def preserve_planner_mode(
    tracker: PlannerModeTracker,
    bridge: UniLaviraJsonBridge,
    action_socket: Any,
    *,
    action_ready: bool,
    reason: str,
) -> bool:
    """Abort motion to zero velocity while keeping C++ running in PLANNER."""
    try:
        stopped = base._best_effort_abort(bridge, reason)
    except Exception:
        stopped = bridge.executor.abort(reason)
    if stopped is None:
        stopped = bridge.executor.abort(reason)

    action_socket.send(build_command_message(start=True, stop=False, planner=True))
    if not tracker.running:
        tracker.apply("k")
    elif not tracker.planner_ready:
        tracker.apply("o")
    for _ in range(5):
        base.publish_output(action_socket, stopped)
        time.sleep(0.02)
    if action_ready:
        print("[UniLaviraPlanner] zero motion published; C++ remains in PLANNER")
        return True
    print(
        "[UniLaviraPlanner] PLANNER keepalive published best-effort; "
        "subscribers are not ready"
    )
    return False


def handle_control_request(
    socket: Any,
    tracker: PlannerModeTracker,
    bridge: UniLaviraJsonBridge,
    action_socket: Any,
    *,
    action_ready: bool,
) -> str | None:
    """Handle one control request and return the requested exit mode, if any."""
    if not socket.poll(0):
        return None
    try:
        request = socket.recv_json()
    except Exception as exc:
        socket.send_json(
            {"status": "rejected", "reason": "invalid_json", "detail": str(exc)}
        )
        return None
    if not isinstance(request, dict):
        socket.send_json({"status": "rejected", "reason": "invalid_request"})
        return None

    operation = str(request.get("op", "")).strip().lower()
    if operation == "status":
        socket.send_json(
            {
                "status": "ok",
                "action_ready": bool(action_ready),
                "planner_ready": tracker.planner_ready,
                "running": tracker.running,
                "mode": tracker.mode,
                "busy": bridge.pending_reply,
            }
        )
        return None
    if operation == "start":
        socket.send_json(
            start_planner(
                tracker,
                bridge,
                action_socket,
                action_ready=action_ready,
            )
        )
        return None
    if operation in {"handoff", "shutdown"}:
        preserved = preserve_planner_mode(
            tracker,
            bridge,
            action_socket,
            action_ready=action_ready,
            reason=f"orchestrator_{operation}",
        )
        if not preserved:
            socket.send_json(
                {"status": "rejected", "reason": "action_subscribers_not_ready"}
            )
            return None
        status = "handoff" if operation == "handoff" else "stopping"
        socket.send_json({"status": status, "planner_ready": True})
        print(f"[UniLaviraPlanner] orchestrator requested safe {operation}")
        return operation

    socket.send_json({"status": "rejected", "reason": "unknown_operation"})
    return None


def main(config: ControlledUniLaviraPlannerConfig) -> None:
    if not math.isfinite(config.hz) or config.hz <= 0:
        raise ValueError(f"hz must be finite and positive, got {config.hz}")

    import zmq

    context = zmq.Context()
    json_rep = context.socket(zmq.REP)
    control_rep = context.socket(zmq.REP)
    action_pub = context.socket(zmq.XPUB)
    keyboard_sub = context.socket(zmq.SUB)
    for socket in (json_rep, control_rep, action_pub, keyboard_sub):
        socket.setsockopt(zmq.LINGER, 0)
    action_pub.setsockopt(zmq.XPUB_VERBOSE, 1)

    json_endpoint = f"tcp://{config.json_host}:{config.json_port}"
    control_endpoint = f"tcp://{config.control_host}:{config.control_port}"
    action_endpoint = f"tcp://{config.action_host}:{config.action_port}"
    keyboard_endpoint = f"tcp://{config.keyboard_host}:{config.keyboard_port}"
    json_rep.bind(json_endpoint)
    control_rep.bind(control_endpoint)
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
    subscribers = base.ActionSubscriberTracker()
    period = 1.0 / config.hz

    time.sleep(0.1)
    print(f"[UniLaviraPlanner] JSON REP bound to {json_endpoint}")
    print(f"[UniLaviraPlanner] control REP bound to {control_endpoint}")
    print(f"[UniLaviraPlanner] Sonic action PUB bound to {action_endpoint}")
    print(f"[UniLaviraPlanner] keyboard SUB connected to {keyboard_endpoint}")
    print("[UniLaviraPlanner] waiting for orchestrator start after N")

    try:
        exit_mode: str | None = None
        while exit_mode is None:
            started = time.monotonic()
            was_ready = subscribers.ready
            base.drain_action_subscriptions(action_pub, subscribers)
            if subscribers.ready and not was_ready:
                print("[UniLaviraPlanner] Sonic command/planner subscribers ready")
            elif was_ready and not subscribers.ready:
                print("[UniLaviraPlanner] Sonic subscriber disconnected")
                base.handle_deploy_disconnect(tracker, bridge, action_pub)

            exit_mode = handle_control_request(
                control_rep,
                tracker,
                bridge,
                action_pub,
                action_ready=subscribers.ready,
            )
            if exit_mode is not None:
                continue

            keyboard_output = base.drain_keyboard_commands(
                keyboard_sub,
                tracker,
                bridge,
                action_pub,
                action_ready=subscribers.ready,
            )
            if keyboard_output is not None:
                base.publish_output(action_pub, keyboard_output)
            base.step_bridge_once(
                bridge,
                planner_ready=tracker.planner_ready and subscribers.ready,
                action_socket=action_pub,
                now=time.monotonic(),
            )
            base._sleep_remaining(started, period)
    except KeyboardInterrupt:
        print("[UniLaviraPlanner] interrupted")
    finally:
        if exit_mode not in {"handoff", "shutdown"}:
            try:
                preserve_planner_mode(
                    tracker,
                    bridge,
                    action_pub,
                    action_ready=subscribers.ready,
                    reason="server_exit",
                )
            except Exception as exc:
                print(f"[UniLaviraPlanner] failed to confirm PLANNER on exit: {exc}")
        keyboard_sub.close()
        control_rep.close()
        json_rep.close()
        action_pub.close()
        context.term()
        print("[UniLaviraPlanner] server exited; C++ PLANNER state preserved")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(ControlledUniLaviraPlannerConfig))
