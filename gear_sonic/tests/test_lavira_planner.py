from __future__ import annotations

import json

import pytest

from gear_sonic.scripts.lavira_planner import (
    LaviraPlannerConfig,
    LaviraPlannerRuntime,
    WorkerResult,
    result_to_goal,
)
from gear_sonic.utils.inference.object_nav import ObjectNavResult


def nav_result(*, distance: float = 2.0, angle_deg: float = -30.0) -> ObjectNavResult:
    return ObjectNavResult(
        outcome="NAVIGATE",
        policy={"target": "chair", "target_type": "global_target", "confidence": 0.9},
        commands={},
        geometry={"mean_range": distance, "angle_deg": angle_deg},
        output_dir="/tmp",
    )


def decoded(messages: list[str]) -> list[dict]:
    return [json.loads(message) for message in messages]


def test_result_goal_maps_camera_right_to_negative_base_y() -> None:
    assert result_to_goal(nav_result()) == pytest.approx((1.73205, -1.0))


def test_n_starts_one_generation_and_rejects_motion_keys_during_nav() -> None:
    messages: list[str] = []
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig("find chair", "chair"), publish=messages.append
    )
    assert runtime.handle_key("n", now=1.0) == "started"
    assert runtime.phase == "nav"
    assert runtime.handle_key("w", now=1.1) == "ignored"
    assert runtime.handle_key("n", now=1.2) == "busy"
    assert decoded(messages)[0]["mode"] == "stop"


def test_worker_result_publishes_goal_not_timed_velocity_sequence() -> None:
    messages: list[str] = []
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig("find chair", "chair"), publish=messages.append
    )
    runtime.handle_key("n", now=1.0)
    runtime.results.put(WorkerResult(1, nav_result(), None))
    runtime.tick(2.0)
    message = decoded(messages)[-1]
    assert message["mode"] == "nav_goal"
    assert message["goal_base"] == pytest.approx({"x": 1.73205, "y": -1.0})
    assert "segments" not in message
    assert "duration_s" not in message


def test_space_invalidates_generation_and_late_worker_result() -> None:
    messages: list[str] = []
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig("find chair", "chair"), publish=messages.append
    )
    runtime.handle_key("n", now=1.0)
    assert runtime.handle_key(" ", now=1.1) == "cancelled"
    runtime.results.put(WorkerResult(1, nav_result(), None))
    runtime.tick(2.0)
    assert runtime.phase == "listen_wasd"
    assert decoded(messages)[-1]["mode"] == "stop"


def test_space_discards_unstarted_request_so_navigation_can_restart() -> None:
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig("find chair", "chair"), publish=lambda _message: None
    )
    assert runtime.handle_key("n", now=1.0) == "started"
    assert runtime.handle_key(" ", now=1.1) == "cancelled"

    assert runtime.requests.empty()
    assert runtime.handle_key("n", now=1.2) == "started"
    assert runtime.requests.get_nowait() == runtime.generation


def test_new_worker_result_replaces_stale_full_queue_entry() -> None:
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig("find chair", "chair"), publish=lambda _message: None
    )
    stale = WorkerResult(1, nav_result(), None)
    current = WorkerResult(3, nav_result(angle_deg=15.0), None)
    runtime.results.put_nowait(stale)

    runtime.publish_worker_result(current)

    assert runtime.results.get_nowait() is current


def test_current_generation_status_returns_to_listen_but_stale_status_is_ignored() -> None:
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig("find chair", "chair"), publish=lambda _message: None
    )
    runtime.handle_key("n", now=1.0)
    assert not runtime.accept_status({"type": "sonic_navigation_status", "generation": 0, "state": "reached"})
    assert runtime.phase == "nav"
    assert runtime.accept_status({"type": "sonic_navigation_status", "generation": 1, "state": "reached"})
    assert runtime.phase == "listen_wasd"


def test_listen_wasd_matches_existing_keyboard_velocities() -> None:
    messages: list[str] = []
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig("find chair", "chair"), publish=messages.append
    )
    assert runtime.handle_key("w", now=1.0) == "manual"
    assert decoded(messages)[-1]["velocity"] == {"vx": 0.3, "vy": 0.0, "wz": 0.0}
