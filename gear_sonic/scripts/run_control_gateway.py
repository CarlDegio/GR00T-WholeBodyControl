#!/usr/bin/env python3
"""Route typed operator intents to structured control consumers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import time

import zmq

from gear_sonic.runtime.config import load_runtime_profile
from gear_sonic.runtime.contracts import (
    CommandAck,
    ControlGatewayHealth,
    MessageMetadata,
    OperatorCommand,
)
from gear_sonic.runtime.control_gateway import (
    BASE_POSE_RUNTIME_STATUS_COMMAND,
    ControlGatewayCore,
    ControlGatewayRouter,
    NavigationControlAction,
    NavigationControlState,
)
from gear_sonic.planner_control import build_navigation_message


@dataclass(frozen=True)
class ControlGatewaySettings:
    profile_name: str
    intent_bind_endpoint: str
    dispatch_bind_endpoint: str
    status_bind_endpoint: str
    navigation_bind_endpoint: str
    navigation_status_endpoint: str
    command_ttl_ms: int
    heartbeat_hz: float


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="", help="Base runtime YAML profile")
    parser.add_argument("--overlay", action="append", default=[])
    parser.add_argument("--intent-bind-host", default="")
    parser.add_argument("--intent-port", type=int, default=0)
    parser.add_argument("--dispatch-bind-host", default="")
    parser.add_argument("--dispatch-port", type=int, default=0)
    parser.add_argument("--status-bind-host", default="")
    parser.add_argument("--status-port", type=int, default=0)
    parser.add_argument("--navigation-bind-host", default="")
    parser.add_argument("--navigation-port", type=int, default=0)
    parser.add_argument("--navigation-status-host", default="")
    parser.add_argument("--navigation-status-port", type=int, default=0)
    parser.add_argument("--command-ttl-ms", type=int, default=0)
    return parser


def resolve_control_gateway_settings(args: argparse.Namespace) -> ControlGatewaySettings:
    profile = load_runtime_profile(args.profile or None, overlays=tuple(args.overlay))
    component = profile.component("control_gateway")

    def bind_endpoint(name: str, host_override: str, port_override: int) -> str:
        address = profile.endpoint(name)
        host = host_override or address.host
        if host in {"127.0.0.1", "localhost"}:
            host = "127.0.0.1"
        port = int(port_override) if port_override else address.port
        if not 1 <= port <= 65535:
            raise ValueError(f"invalid {name} port: {port}")
        return f"tcp://{host}:{port}"

    command_ttl_ms = int(args.command_ttl_ms or component["command_ttl_ms"])
    if command_ttl_ms <= 0:
        raise ValueError("command_ttl_ms must be positive")
    heartbeat_hz = float(component["heartbeat_hz"])
    if heartbeat_hz <= 0.0:
        raise ValueError("heartbeat_hz must be positive")
    return ControlGatewaySettings(
        profile_name=profile.name,
        intent_bind_endpoint=bind_endpoint(
            "control_gateway_intent", args.intent_bind_host, args.intent_port
        ),
        dispatch_bind_endpoint=bind_endpoint(
            "control_gateway_dispatch", args.dispatch_bind_host, args.dispatch_port
        ),
        status_bind_endpoint=bind_endpoint(
            "control_gateway_status", args.status_bind_host, args.status_port
        ),
        navigation_bind_endpoint=bind_endpoint(
            "navigation_command",
            getattr(args, "navigation_bind_host", ""),
            getattr(args, "navigation_port", 0),
        ),
        navigation_status_endpoint=bind_endpoint(
            "navigation_status",
            getattr(args, "navigation_status_host", ""),
            getattr(args, "navigation_status_port", 0),
        ),
        command_ttl_ms=command_ttl_ms,
        heartbeat_hz=heartbeat_hz,
    )


def run_control_gateway(settings: ControlGatewaySettings) -> None:
    context = zmq.Context()
    intent_pull = context.socket(zmq.PULL)
    dispatch_pub = context.socket(zmq.PUB)
    status_pub = context.socket(zmq.PUB)
    navigation_pub = context.socket(zmq.PUB)
    navigation_status_sub = context.socket(zmq.SUB)
    for socket in (
        intent_pull,
        dispatch_pub,
        status_pub,
        navigation_pub,
        navigation_status_sub,
    ):
        socket.setsockopt(zmq.LINGER, 0)
    for socket in (intent_pull, navigation_status_sub):
        socket.setsockopt(zmq.RCVHWM, 100)
    for socket in (dispatch_pub, status_pub, navigation_pub):
        socket.setsockopt(zmq.SNDHWM, 100)
    intent_pull.bind(settings.intent_bind_endpoint)
    dispatch_pub.bind(settings.dispatch_bind_endpoint)
    status_pub.bind(settings.status_bind_endpoint)
    navigation_pub.bind(settings.navigation_bind_endpoint)
    navigation_status_sub.setsockopt_string(zmq.SUBSCRIBE, "")
    navigation_status_sub.connect(settings.navigation_status_endpoint)
    router = ControlGatewayRouter()
    navigation = NavigationControlState()
    gateway_events = ControlGatewayCore(source="control_gateway_navigation")
    ack_sequence = 0
    health_sequence = 0
    accepted_commands = 0
    rejected_commands = 0
    last_command_id = ""

    print(f"[ControlGateway] profile={settings.profile_name}")
    print(f"[ControlGateway] intent PULL: {settings.intent_bind_endpoint}")
    print(f"[ControlGateway] dispatch PUB: {settings.dispatch_bind_endpoint}")
    print(f"[ControlGateway] status PUB: {settings.status_bind_endpoint}")
    print(f"[ControlGateway] navigation PUB: {settings.navigation_bind_endpoint}")
    print(f"[ControlGateway] navigation status SUB: {settings.navigation_status_endpoint}")

    def log_control_route(
        source: str,
        command: str,
        destinations: tuple[str, ...],
        **details: object,
    ) -> None:
        detail_text = " ".join(
            f"{key}={json.dumps(value, ensure_ascii=False, separators=(',', ':'))}"
            for key, value in details.items()
            if value is not None
        )
        destination_text = ",".join(destinations) if destinations else "none"
        suffix = f" | {detail_text}" if detail_text else ""
        print(
            f"[ControlRoute] {source} -> {destination_text} | {command}{suffix}",
            flush=True,
        )

    def dispatch_navigation_event(name: str, parameters: dict[str, object]) -> None:
        event = gateway_events.accept_command(
            name,
            parameters=parameters,
        )
        dispatch_pub.send_string(event.command.to_json())

    def publish_navigation_action(
        action: NavigationControlAction,
        *,
        source: str,
        input_key: str | None = None,
    ) -> bool:
        if action.mode == "ignored":
            log_control_route(
                source,
                "navigation_key",
                (),
                result=action.reason or "ignored",
                key=input_key,
                generation=action.generation,
                navigation_state=navigation.mode,
            )
            return False
        navigation_pub.send_string(
            build_navigation_message(
                mode=action.mode,
                generation=action.generation,
                velocity=action.velocity,
                source=source,
            )
        )
        if action.agent_event is not None:
            event_parameters: dict[str, object] = {
                "generation": action.generation,
            }
            if action.reason:
                event_parameters["reason"] = action.reason
            dispatch_navigation_event(
                action.agent_event,
                event_parameters,
            )
        agent_destinations: tuple[str, ...] = ()
        if action.agent_event == "start_navigation":
            agent_destinations = ("LaViRA",)
        elif action.agent_event == "start_base_pose":
            agent_destinations = ("BasePose",)
        elif action.agent_event == "cancel_navigation":
            agent_destinations = ("LaViRA", "BasePose")
        destinations = ("PlannerExecutor", "NavDP") + agent_destinations
        log_control_route(
            source,
            action.mode,
            destinations,
            generation=action.generation,
            key=input_key,
            velocity=action.velocity,
            agent_event=action.agent_event,
            reason=action.reason,
        )
        return True

    try:
        poller = zmq.Poller()
        poller.register(intent_pull, zmq.POLLIN)
        poller.register(navigation_status_sub, zmq.POLLIN)
        next_heartbeat = time.monotonic()
        while True:
            timeout_ms = max(0, int((next_heartbeat - time.monotonic()) * 1000.0))
            events = dict(poller.poll(min(timeout_ms, 100)))
            if intent_pull in events:
                payload = intent_pull.recv()
                try:
                    command = OperatorCommand.from_json(payload)
                    routed = router.route(command)
                    last_command_id = command.command_id
                    command_accepted = routed.accepted
                    command_reason = routed.reason
                    if routed.accepted:
                        if command.name == "navigation_key":
                            key = command.parameters.get("key")
                            if not isinstance(key, str) or len(key) != 1:
                                raise ValueError("navigation_key requires one string key")
                            action = navigation.handle_key(key, now=time.monotonic())
                            command_accepted = publish_navigation_action(
                                action,
                                source=command.metadata.source,
                                input_key="Space" if key == " " else key.upper(),
                            )
                            if not command_accepted:
                                command_reason = action.reason or "navigation request ignored"
                        elif command.name == "navigation_goal":
                            action = navigation.accept_goal(command.parameters)
                            goal = command.parameters.get("goal_base")
                            if not isinstance(goal, (list, tuple)) or len(goal) != 2:
                                raise ValueError("navigation_goal requires goal_base [x, y]")
                            navigation_pub.send_string(
                                build_navigation_message(
                                    mode=action.mode,
                                    generation=action.generation,
                                    goal_base=goal,
                                    target=str(command.parameters.get("target", "")),
                                    target_type=str(
                                        command.parameters.get("target_type", "")
                                    ),
                                    confidence=float(
                                        command.parameters.get("confidence", 0.0)
                                    ),
                                )
                            )
                            log_control_route(
                                command.metadata.source,
                                "navigation_goal",
                                ("PlannerExecutor", "NavDP"),
                                generation=action.generation,
                                goal_base=goal,
                                target=str(command.parameters.get("target", "")),
                                confidence=float(command.parameters.get("confidence", 0.0)),
                            )
                        elif command.name == "navigation_agent_status":
                            if command.metadata.source != "lavira_agent":
                                raise ValueError("navigation status requires lavira_agent source")
                            if not navigation.accept_status(
                                command.parameters, owner="lavira"
                            ):
                                raise ValueError("stale navigation agent status")
                            generation = int(command.parameters["generation"])
                            navigation_pub.send_string(
                                build_navigation_message(
                                    mode="stop",
                                    generation=generation,
                                )
                            )
                            dispatch_navigation_event(
                                "navigation_status",
                                {
                                    "generation": generation,
                                    "state": str(command.parameters.get("state", "failed")),
                                    "reason": str(command.parameters.get("reason", "")),
                                },
                            )
                            log_control_route(
                                command.metadata.source,
                                "navigation_agent_status",
                                ("PlannerExecutor", "NavDP", "LaViRA"),
                                generation=generation,
                                state=str(command.parameters.get("state", "failed")),
                                reason=str(command.parameters.get("reason", "")),
                            )
                        elif command.name == "base_pose_velocity":
                            if command.metadata.source != "base_pose_agent":
                                raise ValueError("base-pose velocity requires base_pose_agent source")
                            action = navigation.accept_base_pose_velocity(
                                command.parameters,
                                now=time.monotonic(),
                            )
                            publish_navigation_action(
                                action,
                                source=command.metadata.source,
                            )
                            dispatch_navigation_event(
                                BASE_POSE_RUNTIME_STATUS_COMMAND,
                                {
                                    "generation": action.generation,
                                    "state": "motion",
                                    "velocity": list(
                                        action.velocity or (0.0, 0.0, 0.0)
                                    ),
                                    "action": str(
                                        command.parameters.get("action", "visual_servo")
                                    ),
                                    "camera_stream": str(
                                        command.parameters.get("camera_stream", "")
                                    ),
                                },
                            )
                        elif command.name == "base_pose_status":
                            if command.metadata.source != "base_pose_agent":
                                raise ValueError("base-pose status requires base_pose_agent source")
                            if not navigation.accept_status(
                                command.parameters, owner="base_pose"
                            ):
                                raise ValueError("stale base-pose status")
                            generation = int(command.parameters["generation"])
                            navigation_pub.send_string(
                                build_navigation_message(
                                    mode="stop",
                                    generation=generation,
                                )
                            )
                            dispatch_navigation_event(
                                BASE_POSE_RUNTIME_STATUS_COMMAND,
                                {
                                    "generation": generation,
                                    "state": str(
                                        command.parameters.get("state", "failed")
                                    ),
                                    "reason": str(
                                        command.parameters.get("reason", "")
                                    ),
                                    "velocity": [0.0, 0.0, 0.0],
                                },
                            )
                            log_control_route(
                                command.metadata.source,
                                "base_pose_status",
                                (
                                    "PlannerExecutor",
                                    "NavDP",
                                    "ControlGateway status subscribers",
                                ),
                                generation=generation,
                                state=str(command.parameters.get("state", "failed")),
                                reason=str(command.parameters.get("reason", "")),
                            )
                        elif command.name == "select_pose_mode":
                            publish_navigation_action(
                                navigation.handle_key(" ", now=time.monotonic()),
                                source=command.metadata.source,
                                input_key="POSE mode",
                            )
                            log_control_route(
                                command.metadata.source,
                                command.name,
                                ("VLA", "DataExporter", "typed subscribers"),
                                navigation_result="cancelled",
                            )
                        else:
                            log_control_route(
                                command.metadata.source,
                                command.name,
                                ("VLA", "DataExporter", "typed subscribers"),
                            )
                        if command.name not in {
                            "navigation_key",
                            "navigation_goal",
                            "navigation_agent_status",
                            "base_pose_velocity",
                            "base_pose_status",
                        }:
                            dispatch_pub.send_string(command.to_json())
                        if command_accepted:
                            accepted_commands += 1
                        else:
                            rejected_commands += 1
                    else:
                        rejected_commands += 1
                    ack = CommandAck(
                        metadata=MessageMetadata.now(
                            source="control_gateway",
                            sequence=ack_sequence,
                            ttl_ms=settings.command_ttl_ms,
                        ),
                        command_id=command.command_id,
                        accepted=command_accepted,
                        system_state="ready",
                        control_mode="unobserved",
                        reason=command_reason,
                    )
                    ack_sequence += 1
                    status_pub.send_string(
                        json.dumps(ack.to_dict(), separators=(",", ":"), allow_nan=False)
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    rejected_commands += 1
                    print(f"[ControlGateway] rejected malformed intent: {exc}")
            if navigation_status_sub in events:
                try:
                    payload = navigation_status_sub.recv_json()
                    if navigation.accept_status(payload):
                        status_pub.send_string(
                            json.dumps(payload, separators=(",", ":"), allow_nan=False)
                        )
                        dispatch_navigation_event("navigation_status", dict(payload))
                        log_control_route(
                            "NavDP",
                            "navigation_status",
                            ("LaViRA", "ControlGateway status subscribers"),
                            generation=payload.get("generation"),
                            state=payload.get("state"),
                            reason=payload.get("reason"),
                        )
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    print(f"[ControlGateway] rejected navigation status: {exc}")
            timeout_action = navigation.tick(now=time.monotonic())
            if timeout_action is not None:
                publish_navigation_action(
                    timeout_action,
                    source="ControlGateway manual-hold timer",
                )
            now = time.monotonic()
            if now >= next_heartbeat:
                health = ControlGatewayHealth(
                    metadata=MessageMetadata.now(
                        source="control_gateway",
                        sequence=health_sequence,
                        ttl_ms=max(1000, int(3000.0 / settings.heartbeat_hz)),
                    ),
                    state="ready",
                    accepted_commands=accepted_commands,
                    rejected_commands=rejected_commands,
                    last_command_id=last_command_id,
                )
                health_sequence += 1
                status_pub.send_string(
                    json.dumps(health.to_dict(), separators=(",", ":"), allow_nan=False)
                )
                next_heartbeat = now + 1.0 / settings.heartbeat_hz
    except KeyboardInterrupt:
        pass
    finally:
        for socket in (
            intent_pull,
            dispatch_pub,
            status_pub,
            navigation_pub,
            navigation_status_sub,
        ):
            socket.close()
        context.term()


def main() -> None:
    run_control_gateway(
        resolve_control_gateway_settings(build_argument_parser().parse_args())
    )


if __name__ == "__main__":
    main()
