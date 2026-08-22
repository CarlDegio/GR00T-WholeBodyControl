#!/usr/bin/env python3
"""Route typed operator intents to structured control consumers."""

from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Mapping

import zmq

from gear_sonic.runtime.gateway.control import (
    BASE_POSE_RUNTIME_STATUS_COMMAND,
    ControlGatewayCore,
    ControlGatewayRouter,
    NavigationControlAction,
    NavigationControlState,
)
from gear_sonic.runtime.profile import RuntimeProfile, load_runtime_profile
from gear_sonic.runtime.protocol import OperatorCommand, build_navigation_message
from gear_sonic.runtime.telemetry import build_event, configure_file_logging, emit_event


def build_base_pose_runtime_status(
    action: NavigationControlAction,
    parameters: Mapping[str, object],
) -> dict[str, object]:
    """Build viewer telemetry without feeding overlays into motion validation."""

    base_pose_action = str(parameters.get("action", "visual_servo"))
    state = {
        "hold": "inference",
        "visual_servo": "motion",
        "stop": "stopping",
    }.get(base_pose_action, "motion")
    status: dict[str, object] = {
        "generation": action.generation,
        "state": state,
        "velocity": list(action.velocity or (0.0, 0.0, 0.0)),
        "action": base_pose_action,
        "camera_stream": str(parameters.get("camera_stream", "")),
    }
    viewer_overlay = parameters.get("viewer_overlay")
    if viewer_overlay is not None:
        if not isinstance(viewer_overlay, Mapping):
            raise ValueError("base-pose viewer_overlay must be an object")
        status["viewer_overlay"] = dict(viewer_overlay)
    return status


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="", help="Base runtime YAML profile")
    parser.add_argument("--overlay", action="append", default=[])
    return parser


def run_control_gateway(profile: RuntimeProfile) -> None:
    logger = configure_file_logging("control_gateway")
    context = zmq.Context()
    intent_pull = context.socket(zmq.PULL)
    event_pull = context.socket(zmq.PULL)
    dispatch_pub = context.socket(zmq.PUB)
    navigation_pub = context.socket(zmq.PUB)
    navigation_status_sub = context.socket(zmq.SUB)
    for socket in (
        intent_pull,
        event_pull,
        dispatch_pub,
        navigation_pub,
        navigation_status_sub,
    ):
        socket.setsockopt(zmq.LINGER, 0)
    for socket in (intent_pull, event_pull, navigation_status_sub):
        socket.setsockopt(zmq.RCVHWM, 100)
    for socket in (dispatch_pub, navigation_pub):
        socket.setsockopt(zmq.SNDHWM, 100)
    intent_pull.bind(profile.endpoint_uri("control_gateway_intent"))
    event_pull.bind(profile.endpoint_uri("runtime_event_ingress"))
    dispatch_pub.bind(profile.endpoint_uri("control_gateway_dispatch"))
    navigation_pub.bind(profile.endpoint_uri("navigation_command"))
    navigation_status_sub.setsockopt_string(zmq.SUBSCRIBE, "")
    navigation_status_sub.connect(profile.endpoint_uri("navigation_status"))
    router = ControlGatewayRouter()
    navigation = NavigationControlState()
    gateway_events = ControlGatewayCore(source="control_gateway_navigation")

    def show_event(level: int, code: str, message: str, **fields: object) -> None:
        emit_event(
            build_event("control_gateway", level, code, message, **fields),
            display=True,
            logger=logger,
        )

    def show_command(command: OperatorCommand, code: str, message: str | None = None,
                     level: int = logging.INFO, **fields: object) -> None:
        details = {
            "source": command.metadata.source,
            "generation": command.parameters.get("generation"),
            "reason": command.parameters.get("reason"),
            **fields,
        }
        show_event(level, code, message or command.name, **details)

    show_event(logging.INFO, "READY", "ControlGateway ready", profile=profile.name)

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
            show_event(logging.WARNING, "CONTROL_IGNORED", action.reason or "ignored",
                       source=source, key=input_key, generation=action.generation)
            return False
        navigation_pub.send_string(
            build_navigation_message(
                mode=action.mode,
                generation=action.generation,
                segment_id=action.segment_id,
                velocity=action.velocity,
                source=source,
            )
        )
        if action.agent_event is not None:
            event_parameters: dict[str, object] = {
                "generation": action.generation,
                "segment_id": action.segment_id,
            }
            if action.reason:
                event_parameters["reason"] = action.reason
            dispatch_navigation_event(
                action.agent_event,
                event_parameters,
            )
        if action.agent_event is not None:
            show_event(logging.INFO, action.agent_event.upper(), action.mode, source=source,
                       generation=action.generation, reason=action.reason or None)
        return True

    try:
        poller = zmq.Poller()
        poller.register(intent_pull, zmq.POLLIN)
        poller.register(event_pull, zmq.POLLIN)
        poller.register(navigation_status_sub, zmq.POLLIN)
        while True:
            events = dict(poller.poll(100))
            if intent_pull in events:
                payload = intent_pull.recv()
                try:
                    command = OperatorCommand.from_json(payload)
                    routed = router.route(command)
                    if routed.accepted:
                        if command.name == "navigation_key":
                            key = command.parameters.get("key")
                            if not isinstance(key, str) or len(key) != 1:
                                raise ValueError("navigation_key requires one string key")
                            action = navigation.handle_key(
                                key,
                                now=time.monotonic(),
                                cancel_reason=str(command.parameters.get("reason", "")),
                            )
                            publish_navigation_action(
                                action,
                                source=command.metadata.source,
                                input_key="Space" if key == " " else key.upper(),
                            )
                        elif command.name == "lavira_depth_request":
                            if command.metadata.source != "lavira_agent":
                                raise ValueError(
                                    "LaViRA depth request requires lavira_agent source"
                                )
                            if not navigation.accept_lavira_depth_request(
                                command.parameters
                            ):
                                raise ValueError("stale LaViRA depth request")
                            dispatch_navigation_event(
                                "lavira_depth_request", dict(command.parameters)
                            )
                            show_command(
                                command,
                                "LAVIRA_DEPTH_REQUEST",
                                "LaViRA depth lease acquired",
                            )
                        elif command.name == "lavira_rgbd_captured":
                            if command.metadata.source != "lavira_agent":
                                raise ValueError(
                                    "LaViRA RGB-D completion requires lavira_agent source"
                                )
                            if not navigation.accept_lavira_rgbd_captured(
                                command.parameters
                            ):
                                raise ValueError("stale LaViRA RGB-D completion")
                            dispatch_navigation_event(
                                "lavira_rgbd_captured", dict(command.parameters)
                            )
                            show_command(command, "LAVIRA_RGBD_CAPTURED", "LaViRA RGB-D captured")
                        elif command.name == "navigation_goal":
                            action = navigation.accept_goal(command.parameters)
                            goal = command.parameters.get("goal_base")
                            if not isinstance(goal, (list, tuple)) or len(goal) != 2:
                                raise ValueError("navigation_goal requires goal_base [x, y]")
                            navigation_pub.send_string(
                                build_navigation_message(
                                    mode=action.mode,
                                    generation=action.generation,
                                    segment_id=action.segment_id,
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
                            dispatch_navigation_event(
                                "navigation_goal",
                                {
                                    "generation": action.generation,
                                    "segment_id": action.segment_id,
                                },
                            )
                            show_command(command, "NAVIGATION_GOAL", "navigation goal accepted",
                                         goal_base=goal, target=command.parameters.get("target"),
                                         confidence=command.parameters.get("confidence"))
                        elif command.name == "navigation_heading_goal":
                            if command.metadata.source != "lavira_agent":
                                raise ValueError(
                                    "heading goal requires lavira_agent source"
                                )
                            action = navigation.accept_heading_goal(command.parameters)
                            heading_delta_rad = float(
                                command.parameters["heading_delta_rad"]
                            )
                            navigation_pub.send_string(
                                build_navigation_message(
                                    mode=action.mode,
                                    generation=action.generation,
                                    segment_id=action.segment_id,
                                    heading_delta_rad=heading_delta_rad,
                                )
                            )
                            dispatch_navigation_event(
                                "navigation_heading_goal",
                                {
                                    "generation": action.generation,
                                    "segment_id": action.segment_id,
                                },
                            )
                            show_command(
                                command,
                                "NAVIGATION_HEADING_GOAL",
                                "heading goal accepted",
                                heading_delta_rad=heading_delta_rad,
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
                                    segment_id=int(
                                        command.parameters.get("segment_id", 0)
                                    ),
                                )
                            )
                            dispatch_navigation_event(
                                "navigation_status",
                                dict(command.parameters),
                            )
                            state = str(command.parameters.get("state", "failed"))
                            level = logging.ERROR if state == "failed" else logging.INFO
                            show_command(command, "NAVIGATION_STATUS", state, level)
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
                                build_base_pose_runtime_status(
                                    action,
                                    command.parameters,
                                ),
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
                            state = str(command.parameters.get("state", "failed"))
                            level = logging.ERROR if state == "failed" else logging.INFO
                            show_command(command, "BASE_POSE_STATUS", state, level)
                        elif command.name == "select_pose_mode":
                            publish_navigation_action(
                                navigation.handle_key(
                                    " ",
                                    now=time.monotonic(),
                                    cancel_reason="select_pose_mode",
                                ),
                                source=command.metadata.source,
                                input_key="POSE mode",
                            )
                            show_command(command, "POSE_MODE", "POSE mode selected")
                        else:
                            show_command(command, "CONTROL_COMMAND")
                        if command.name not in {
                            "navigation_key",
                            "navigation_goal",
                            "navigation_heading_goal",
                            "lavira_depth_request",
                            "lavira_rgbd_captured",
                            "navigation_agent_status",
                            "base_pose_velocity",
                            "base_pose_status",
                        }:
                            dispatch_pub.send_string(command.to_json())
                    else:
                        show_command(command, "CONTROL_REJECTED", level=logging.WARNING,
                                     reason=routed.reason, command_id=command.command_id)
                except (KeyError, TypeError, ValueError) as exc:
                    show_event(logging.WARNING, "INVALID_INTENT", str(exc))
            if navigation_status_sub in events:
                try:
                    payload = navigation_status_sub.recv_json()
                    if navigation.accept_status(
                        payload,
                        owner="lavira",
                        agent_final=False,
                    ):
                        dispatch_navigation_event("navigation_status", dict(payload))
                        state = str(payload.get("state", "unknown"))
                        level = logging.ERROR if state == "failed" else logging.INFO
                        show_event(level, "NAVIGATION_STATUS", state, source="navdp",
                                   generation=payload.get("generation"),
                                   reason=payload.get("reason"))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    show_event(logging.WARNING, "INVALID_NAVIGATION_STATUS", str(exc))
            if event_pull in events:
                try:
                    emit_event(event_pull.recv_json(), display=True)
                except (AttributeError, KeyError, OSError, OverflowError, TypeError, ValueError) as exc:
                    show_event(logging.WARNING, "INVALID_RUNTIME_EVENT", str(exc))
            timeout_action = navigation.tick(now=time.monotonic())
            if timeout_action is not None:
                publish_navigation_action(
                    timeout_action,
                    source="ControlGateway manual-hold timer",
                )
    except KeyboardInterrupt:
        pass
    finally:
        for socket in (
            intent_pull,
            event_pull,
            dispatch_pub,
            navigation_pub,
            navigation_status_sub,
        ):
            socket.close()
        context.term()


def main() -> None:
    args = build_argument_parser().parse_args()
    run_control_gateway(
        load_runtime_profile(args.profile or None, overlays=tuple(args.overlay))
    )


if __name__ == "__main__":
    main()
