from __future__ import annotations

import argparse

from gear_sonic.runtime.contracts import OperatorCommand
from gear_sonic.runtime.control_gateway import (
    ControlGatewayCore,
    ControlGatewayRouter,
    NavigationControlState,
    OperatorConsoleRouter,
    legacy_message_from_console_line,
)
from gear_sonic.scripts.run_control_gateway import resolve_control_gateway_settings


def test_console_translation_preserves_the_deployed_legacy_wire() -> None:
    assert legacy_message_from_console_line("i") == "i"
    assert legacy_message_from_console_line("t pick up the cup") == (
        "prompt:pick up the cup"
    )
    assert legacy_message_from_console_line("") == ""


def test_core_adds_identity_lifetime_and_semantics_without_state_changes() -> None:
    timestamps = iter((100, 200, 300))
    core = ControlGatewayCore(
        source="test_console",
        ttl_ms=750,
        monotonic_ns=lambda: next(timestamps),
    )

    pose = core.accept_console_line("i")
    prompt = core.accept_console_line("t inspect the basket")
    unknown = core.accept_console_line("custom")

    assert pose.legacy_message == "i"
    assert pose.command.name == "select_pose_mode"
    assert pose.command.command_id == "test_console-0"
    assert pose.command.metadata.timestamp_ns == 100
    assert pose.command.metadata.ttl_ms == 750
    assert OperatorCommand.from_json(pose.command.to_json()) == pose.command

    assert prompt.legacy_message == "prompt:inspect the basket"
    assert prompt.command.name == "set_prompt"
    assert prompt.command.parameters["prompt"] == "inspect the basket"
    assert prompt.command.metadata.sequence == 1

    assert unknown.command.name == "legacy_passthrough"
    assert unknown.command.parameters["legacy_message"] == "custom"
    assert core.next_sequence == 3


def test_navigation_key_is_structured_and_never_hits_legacy_recording_wire() -> None:
    core = ControlGatewayCore(monotonic_ns=lambda: 100)
    event = core.accept_command(
        "navigation_key",
        parameters={"key": "e"},
        legacy_message="e",
        mirror_legacy=False,
    )
    routed = ControlGatewayRouter(monotonic_ns=lambda: 101).route(event.command)

    assert routed.accepted
    assert routed.legacy_message == "e"
    assert not routed.mirror_legacy


def test_console_routes_planner_keys_and_pose_recording_without_e_collision() -> None:
    core = ControlGatewayCore(monotonic_ns=lambda: 100)
    console = OperatorConsoleRouter()

    planner_e = console.accept_line("e", core=core)
    start = console.accept_line("k", core=core)
    pose = console.accept_line("i", core=core)
    pose_e = console.accept_line("e", core=core)
    explicit_record = console.accept_line("record-success", core=core)
    planner = console.accept_line("o", core=core)
    stop = console.accept_line("space", core=core)

    assert planner_e.command.name == "navigation_key"
    assert planner_e.command.parameters["key"] == "e"
    assert planner_e.command.parameters["mirror_legacy"] is False
    assert start.command.name == "toggle_control_loop"
    assert pose.command.name == "select_pose_mode"
    assert pose_e.command.name == "stop_recording_success"
    assert explicit_record.command.name == "stop_recording_success"
    assert planner.command.name == "select_planner_mode"
    assert stop.command.name == "navigation_key"
    assert stop.command.parameters["key"] == " "


def test_gateway_settings_use_reserved_profile_endpoints() -> None:
    args = argparse.Namespace(
        profile="",
        overlay=[],
        legacy_bind_host="",
        legacy_port=0,
        intent_bind_host="",
        intent_port=0,
        dispatch_bind_host="",
        dispatch_port=0,
        status_bind_host="",
        status_port=0,
        command_ttl_ms=0,
    )

    settings = resolve_control_gateway_settings(args)

    assert settings.legacy_bind_endpoint == "tcp://127.0.0.1:5580"
    assert settings.intent_bind_endpoint == "tcp://127.0.0.1:5561"
    assert settings.dispatch_bind_endpoint == "tcp://127.0.0.1:5565"
    assert settings.status_bind_endpoint == "tcp://127.0.0.1:5562"
    assert settings.navigation_bind_endpoint == "tcp://127.0.0.1:5558"
    assert settings.navigation_status_endpoint == "tcp://127.0.0.1:5559"
    assert settings.command_ttl_ms == 1000
    assert settings.heartbeat_hz == 2.0


def test_navigation_state_preserves_existing_listen_wasd_mapping() -> None:
    state = NavigationControlState()

    forward = state.handle_key("w", now=1.0)
    assert forward.mode == "manual_velocity"
    assert forward.velocity == (0.3, 0.0, 0.0)
    assert forward.generation == 0
    assert state.mode == "listen_wasd"

    stopped = state.tick(now=1.55)
    assert stopped is not None
    assert stopped.velocity == (0.0, 0.0, 0.0)
    assert state.mode == "listen_wasd"


def test_navigation_can_start_during_the_existing_manual_hold_window() -> None:
    state = NavigationControlState()
    state.handle_key("w", now=1.0)

    started = state.handle_key("n", now=1.1)

    assert started.mode == "stop"
    assert started.agent_event == "start_navigation"
    assert started.generation == 1
    assert state.manual_velocity == (0.0, 0.0, 0.0)
    assert state.mode == "nav_pending"


def test_navigation_state_ignores_motion_and_repeat_n_during_nav() -> None:
    state = NavigationControlState()
    started = state.handle_key("n", now=1.0)
    assert started.agent_event == "start_navigation"
    assert state.mode == "nav_pending"

    for key in ("w", "a", "s", "d", "q", "e", "n"):
        ignored = state.handle_key(key, now=1.1)
        assert ignored.mode == "ignored"
        assert ignored.generation == started.generation
    assert state.mode == "nav_pending"

    cancelled = state.handle_key(" ", now=1.2)
    assert cancelled.mode == "stop"
    assert cancelled.agent_event == "cancel_navigation"
    assert cancelled.generation == started.generation + 1
    assert state.mode == "listen_wasd"
