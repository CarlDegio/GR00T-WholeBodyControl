from __future__ import annotations

import io
import json
import logging
import time

import pytest
import zmq

from gear_sonic.runtime.protocol import OperatorCommand
from gear_sonic.runtime.gateway.control_client import ControlGatewayIntentClient
from gear_sonic.runtime.gateway.control import (
    ControlGatewayCore,
    ControlGatewayRouter,
    NavigationControlState,
    OperatorConsoleRouter,
)
from gear_sonic.runtime.gateway.services.control import (
    EventPaneDisplay,
    build_base_pose_runtime_status,
)
from gear_sonic.runtime.telemetry import build_event


def test_event_pane_keeps_todo_above_general_events_and_has_placeholders() -> None:
    output = io.StringIO()
    display = EventPaneDisplay(stream=output, interactive=False)

    assert "LA TODO · no active task" in display.dashboard_text()
    assert "Press N" in display.dashboard_text()

    display.accept(build_event(
        "lavira", logging.INFO, "TASK_ACCEPTED", "task accepted",
        generation=7,
    ))
    assert "generation=7" in display.dashboard_text()
    assert "Waiting for the Language Agent" in display.dashboard_text()

    display.accept(build_event(
        "navdp", logging.WARNING, "PATH_BLOCKED", "local path blocked",
        generation=7,
    ))
    display.accept(build_event(
        "lavira", logging.INFO, "TODO_UPDATED", "TODO updated",
        generation=7,
        step=3,
        todo_list=(
            "- [x] Find the doorway\n"
            "- [ ] Approach the basket\n"
            "- [ ] Align"
        ),
    ))
    panel = display.dashboard_text(width=100, height=30)
    todo_end = panel.index("RUNTIME EVENTS")
    event_start = panel.index("[WARNING][NAVDP]")
    assert "progress=1/3" in panel[:todo_end]
    assert "✓ Find the doorway" in panel[:todo_end]
    assert "▶ Approach the basket" in panel[:todo_end]
    assert "○ Align" in panel[:todo_end]
    assert event_start > todo_end
    assert "TODO_UPDATED" not in panel[event_start:]

    display.accept(build_event(
        "lavira", logging.INFO, "TASK_COMPLETED", "task complete",
        generation=7,
    ))
    assert "LA TODO · no active task" in display.dashboard_text()


def test_event_pane_limits_long_todo_to_top_quarter_and_keeps_active_item() -> None:
    display = EventPaneDisplay(stream=io.StringIO(), interactive=False)
    todo = "\n".join([
        *(f"- [x] Completed {index}" for index in range(8)),
        "- [ ] Current approach",
        *(f"- [ ] Later {index}" for index in range(8)),
    ])
    display.accept(build_event(
        "lavira", logging.INFO, "TODO_UPDATED", "TODO updated",
        generation=9, step=11, todo_list=todo,
    ))

    lines = display.dashboard_text(width=100, height=20).splitlines()
    separator_index = next(
        index for index, line in enumerate(lines) if set(line) == {"─"}
    )
    assert separator_index <= 5
    assert "▶ Current approach" in "\n".join(lines[:separator_index])
    assert "additional TODO line(s)" in "\n".join(lines[:separator_index])


def test_latest_only_intent_client_conflates_pending_commands() -> None:
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.setsockopt(zmq.LINGER, 0)
    receiver.bind("inproc://latest-only-intent-client")
    client = ControlGatewayIntentClient(
        "inproc://latest-only-intent-client",
        source="base_pose_agent",
        context=context,
        latest_only=True,
    )
    try:
        assert client._socket.getsockopt(zmq.CONFLATE) == 1
        assert client._socket.getsockopt(zmq.SNDHWM) == 1
        for vx in (0.4, -0.4, 0.0):
            client.send("base_pose_velocity", {"velocity": [vx, 0.0, 0.0]})
        time.sleep(0.02)
        payloads = []
        while receiver.poll(0):
            payloads.append(json.loads(receiver.recv()))
        assert len(payloads) == 1
        assert payloads[0]["parameters"]["velocity"] == [0.0, 0.0, 0.0]
    finally:
        client.close()
        receiver.close()
        context.term()


def test_core_adds_identity_lifetime_and_typed_semantics() -> None:
    timestamps = iter((100, 200, 300))
    core = ControlGatewayCore(
        source="test_console",
        ttl_ms=750,
        monotonic_ns=lambda: next(timestamps),
    )

    pose = core.accept_console_line("i")
    prompt = core.accept_console_line("t inspect the basket")
    unknown = core.accept_console_line("custom")

    assert pose.command.name == "select_pose_mode"
    assert pose.command.command_id == "test_console-0"
    assert pose.command.metadata.timestamp_ns == 100
    assert pose.command.metadata.ttl_ms == 750
    assert OperatorCommand.from_json(pose.command.to_json()) == pose.command

    assert prompt.command.name == "set_prompt"
    assert prompt.command.parameters["prompt"] == "inspect the basket"
    assert prompt.command.metadata.sequence == 1

    assert unknown.command.name == "unsupported_console_input"
    assert unknown.command.parameters == {}
    assert unknown.command.metadata.sequence == 2


def test_navigation_key_is_a_structured_typed_command() -> None:
    core = ControlGatewayCore(monotonic_ns=lambda: 100)
    event = core.accept_command(
        "navigation_key",
        parameters={"key": "e"},
    )
    routed = ControlGatewayRouter(monotonic_ns=lambda: 101).route(event.command)

    assert routed.accepted
    assert routed.command.parameters == {"key": "e"}


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
    assert start.command.name == "toggle_control_loop"
    assert pose.command.name == "select_pose_mode"
    assert pose_e.command.name == "stop_recording_success"
    assert explicit_record.command.name == "stop_recording_success"
    assert planner.command.name == "select_planner_mode"
    assert stop.command.name == "navigation_key"
    assert stop.command.parameters["key"] == " "


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
    assert state.mode == "lavira_pending"


def test_navigation_state_ignores_motion_and_repeat_n_during_nav() -> None:
    state = NavigationControlState()
    started = state.handle_key("n", now=1.0)
    assert started.agent_event == "start_navigation"
    assert state.mode == "lavira_pending"

    for key in ("w", "a", "s", "d", "q", "e", "n", "b"):
        ignored = state.handle_key(key, now=1.1)
        assert ignored.mode == "ignored"
        assert ignored.generation == started.generation
        assert ignored.reason == "navigation_busy:lavira"
    assert state.mode == "lavira_pending"

    cancelled = state.handle_key(" ", now=1.2)
    assert cancelled.mode == "stop"
    assert cancelled.agent_event == "cancel_navigation"
    assert cancelled.reason == ""
    assert cancelled.generation == started.generation + 1
    assert state.mode == "listen_wasd"

    reasoned_cancel = state.handle_key(
        " ",
        now=1.3,
        cancel_reason="slam_recovery:fastlio_stale",
    )
    assert reasoned_cancel.reason == "slam_recovery:fastlio_stale"


def test_b_starts_base_pose_and_rejects_lavira_until_space() -> None:
    state = NavigationControlState()

    started = state.handle_key("b", now=1.0)

    assert started.mode == "stop"
    assert started.agent_event == "start_base_pose"
    assert state.mode == "base_pose_inference"
    rejected = state.handle_key("n", now=1.1)
    assert rejected.mode == "ignored"
    assert rejected.reason == "navigation_busy:base_pose"
    assert rejected.generation == started.generation


def test_lavira_rgbd_release_is_valid_only_during_pending_target_confirmation() -> None:
    state = NavigationControlState()
    started = state.handle_key("n", now=1.0)

    assert state.accept_lavira_rgbd_captured({"generation": started.generation})
    assert state.mode == "lavira_pending"
    state.accept_goal({"generation": started.generation})
    assert not state.accept_lavira_rgbd_captured(
        {"generation": started.generation}
    )


def test_lavira_generation_accepts_monotonic_segments_until_agent_final() -> None:
    state = NavigationControlState()
    started = state.handle_key("n", now=1.0)

    heading = state.accept_heading_goal(
        {
            "generation": started.generation,
            "segment_id": 0,
            "heading_delta_rad": 1.57,
        }
    )
    assert heading.mode == "heading_goal"
    assert state.accept_status(
        {
            "generation": started.generation,
            "segment_id": 0,
            "state": "reached",
        },
        owner="lavira",
        agent_final=False,
    )
    assert state.mode == "lavira_pending"
    assert state.accept_lavira_depth_request(
        {"generation": started.generation, "segment_id": 0}
    )
    assert state.accept_lavira_rgbd_captured(
        {"generation": started.generation, "segment_id": 0}
    )

    goal = state.accept_goal(
        {"generation": started.generation, "segment_id": 1}
    )
    assert goal.segment_id == 1
    assert not state.accept_status(
        {
            "generation": started.generation,
            "segment_id": 0,
            "state": "reached",
        },
        owner="lavira",
        agent_final=False,
    )
    assert state.accept_status(
        {
            "generation": started.generation,
            "segment_id": 1,
            "state": "reached",
        },
        owner="lavira",
        agent_final=False,
    )
    assert state.mode == "lavira_pending"
    assert state.accept_status(
        {"generation": started.generation, "state": "reached"},
        owner="lavira",
    )
    assert state.mode == "listen_wasd"


def test_lavira_skill_handoff_rejects_stale_skill_and_enters_terminal_vla() -> None:
    state = NavigationControlState()
    started = state.handle_key("n", now=1.0)
    heading = state.accept_heading_goal({
        "generation": started.generation,
        "skill_id": 1,
        "segment_id": 0,
        "heading_delta_rad": 0.0,
    })
    assert heading.skill_id == 1
    assert state.accept_status({
        "generation": started.generation,
        "skill_id": 1,
        "segment_id": 0,
        "state": "reached",
    }, owner="lavira", agent_final=False)
    with pytest.raises(ValueError, match="stale"):
        state.accept_goal({
            "generation": started.generation,
            "skill_id": 0,
            "segment_id": 99,
        })

    base = state.accept_base_pose_start({
        "generation": started.generation,
        "skill_id": 2,
        "segment_id": 1,
    })
    assert base.skill_id == 2
    assert state.accept_status({
        "generation": started.generation,
        "skill_id": 2,
        "segment_id": 1,
        "state": "reached",
        "reason": "aligned",
    }, owner="base_pose", agent_final=False)
    assert state.mode == "lavira_pending"

    assert state.accept_vla_command("start_vla_task", {
        "generation": started.generation,
        "skill_id": 3,
        "window_id": 0,
    })
    assert state.mode == "lavira_manipulate"
    assert not state.accept_vla_command("start_vla_task", {
        "generation": started.generation,
        "skill_id": 2,
    })
    assert state.accept_vla_command("hold_vla_task", {
        "generation": started.generation,
        "skill_id": 3,
        "window_id": 1,
    })


def test_vla_start_accepts_same_skill_after_automatic_la_panorama() -> None:
    state = NavigationControlState()
    started = state.handle_key("n", now=1.0)
    for segment_id in range(4):
        state.accept_heading_goal({
            "generation": started.generation,
            "skill_id": 1,
            "segment_id": segment_id,
            "heading_delta_rad": 0.0,
        })
        assert state.accept_status({
            "generation": started.generation,
            "skill_id": 1,
            "segment_id": segment_id,
            "state": "reached",
        }, owner="lavira", agent_final=False)

    assert state.accept_vla_command("start_vla_task", {
        "generation": started.generation,
        "skill_id": 1,
        "window_id": 0,
    })
    assert state.mode == "lavira_manipulate"
    assert state.segment_id == 3


def test_base_pose_velocity_is_bounded_and_times_out_safe() -> None:
    state = NavigationControlState(base_pose_command_timeout_s=0.2)
    started = state.handle_key("b", now=1.0)
    action = state.accept_base_pose_velocity(
        {
            "generation": started.generation,
            "velocity": [0.3, 0.0, 0.0],
            "action": "visual_servo",
        },
        now=1.1,
    )

    assert action.mode == "manual_velocity"
    assert state.mode == "base_pose_motion"
    assert state.tick(now=1.29) is None
    timeout = state.tick(now=1.3)
    assert timeout is not None
    assert timeout.mode == "stop"
    assert timeout.agent_event == "cancel_navigation"
    assert timeout.reason == "base_pose_velocity_timeout"
    assert timeout.generation == started.generation + 1
    assert state.mode == "listen_wasd"


def test_base_pose_inference_hold_does_not_arm_motion_watchdog() -> None:
    state = NavigationControlState(base_pose_command_timeout_s=0.2)
    started = state.handle_key("b", now=1.0)
    parameters = {
        "generation": started.generation,
        "velocity": [0.0, 0.0, 0.0],
        "motion_profile": "yoloe_servo",
        "action": "hold",
    }

    action = state.accept_base_pose_velocity(parameters, now=1.1)
    status = build_base_pose_runtime_status(action, parameters)

    assert action.mode == "stop"
    assert action.velocity == (0.0, 0.0, 0.0)
    assert state.mode == "base_pose_inference"
    assert state.manual_deadline == 0.0
    assert state.tick(now=100.0) is None
    assert status["state"] == "inference"
    assert status["action"] == "hold"


def test_base_pose_stop_disarms_existing_motion_watchdog() -> None:
    state = NavigationControlState(base_pose_command_timeout_s=0.2)
    started = state.handle_key("b", now=1.0)
    state.accept_base_pose_velocity(
        {
            "generation": started.generation,
            "velocity": [0.3, 0.0, 0.0],
            "action": "visual_servo",
        },
        now=1.1,
    )
    parameters = {
        "generation": started.generation,
        "velocity": [0.0, 0.0, 0.0],
        "action": "stop",
    }

    action = state.accept_base_pose_velocity(parameters, now=1.2)
    status = build_base_pose_runtime_status(action, parameters)

    assert action.mode == "stop"
    assert state.mode == "base_pose_stopping"
    assert state.manual_deadline == 0.0
    assert state.tick(now=100.0) is None
    assert status["state"] == "stopping"
    assert status["action"] == "stop"


def test_base_pose_hold_and_stop_require_zero_velocity() -> None:
    state = NavigationControlState()
    started = state.handle_key("b", now=1.0)

    for action in ("hold", "stop"):
        with pytest.raises(ValueError, match="must command zero velocity"):
            state.accept_base_pose_velocity(
                {
                    "generation": started.generation,
                    "velocity": [0.1, 0.0, 0.0],
                    "action": action,
                },
                now=1.1,
            )


def test_base_pose_velocity_rejects_stale_or_unsafe_commands() -> None:
    state = NavigationControlState()
    started = state.handle_key("b", now=1.0)

    with pytest.raises(ValueError, match="safety envelope"):
        state.accept_base_pose_velocity(
            {"generation": started.generation, "velocity": [0.31, 0.0, 0.0]},
            now=1.1,
        )
    state.handle_key(" ", now=1.2)
    with pytest.raises(ValueError, match="stale"):
        state.accept_base_pose_velocity(
            {"generation": started.generation, "velocity": [0.0, 0.0, 0.0]},
            now=1.3,
        )


def test_yoloe_base_pose_profile_allows_bounded_single_axis_lateral_servo() -> None:
    state = NavigationControlState()
    started = state.handle_key("b", now=1.0)

    action = state.accept_base_pose_velocity(
        {
            "generation": started.generation,
            "velocity": [0.0, 0.4, -0.3],
            "motion_profile": "yoloe_servo",
        },
        now=1.1,
    )

    assert action.velocity == pytest.approx((0.0, 0.4, -0.3))
    with pytest.raises(ValueError, match="safety envelope"):
        state.accept_base_pose_velocity(
            {
                "generation": started.generation,
                "velocity": [0.1, 0.1, 0.0],
                "motion_profile": "yoloe_servo",
            },
            now=1.2,
        )


def test_base_pose_viewer_overlay_is_telemetry_only() -> None:
    state = NavigationControlState()
    started = state.handle_key("b", now=1.0)
    parameters = {
        "generation": started.generation,
        "velocity": [0.0, 0.0, 0.2],
        "motion_profile": "yoloe_servo",
        "action": "visual_servo",
        "camera_stream": "ego_view",
        "viewer_overlay": {
            "target_bbox_xyxy": [10.0, 20.0, 30.0, 40.0],
            "image_size": [64, 48],
        },
    }

    action = state.accept_base_pose_velocity(parameters, now=1.1)
    status = build_base_pose_runtime_status(action, parameters)

    assert action.velocity == pytest.approx((0.0, 0.0, 0.2))
    assert status["generation"] == started.generation
    assert status["velocity"] == [0.0, 0.0, 0.2]
    assert status["viewer_overlay"] == parameters["viewer_overlay"]
