from __future__ import annotations

import logging
import math
import queue
import threading
import time

import pytest

from gear_sonic.utils.inference.lavira.agent import (
    LaViRAAgentCancelled,
    LaViRATaskResult,
)
from gear_sonic.utils.inference.lavira.service import (
    LaviraPlannerConfig,
    LaviraPlannerRuntime,
    LaviraRuntimeEventHandler,
    WorkerResult,
)


def task_result(generation: int) -> LaViRATaskResult:
    return LaViRATaskResult(
        generation=generation,
        state="completed",
        reason="",
        steps=1,
        skill_id=1,
        segment_id=1,
        navigation_mode="object_nav",
    )


def runtime_with_intents():
    intents: list[tuple[str, dict]] = []
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(
            mission="find chair",
            alignment_prompt="align to chair",
            global_target="chair",
        ),
        submit_intent=lambda name, parameters: intents.append((name, dict(parameters))),
    )
    return runtime, intents


def test_lavira_config_exposes_dual_cloud_agent_roles() -> None:
    config = LaviraPlannerConfig(
        mission="find chair",
        alignment_prompt="align to chair",
        global_target="chair",
    )

    assert config.navigation_mode == "object_nav"
    assert config.manipulation_prompt == "find chair"
    assert not hasattr(config, "task_type")
    assert not hasattr(config, "question")
    assert config.la_model == "qwen3.8-max"
    assert config.va_model == "qwen3.5-27b"
    assert config.la_enable_thinking is False
    assert config.va_enable_thinking is False
    assert config.la_base_url == "https://ws-6yzgj1m087a053ip.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    assert config.va_base_url == "https://ws-6yzgj1m087a053ip.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    assert config.nav_handoff_min_depth_m == 0.3
    assert config.nav_handoff_max_depth_m == 3.0
    assert config.alignment_head_camera_stream == "ego_view"
    assert config.heading_settle_seconds == pytest.approx(1.0)
    assert config.heading_settle_samples == 30
    assert config.heading_settle_bad_sample_threshold == 12
    assert config.heading_settle_tolerance_rad == pytest.approx(
        math.radians(5.0)
    )
    assert config.heading_correction_speed_rad_s == pytest.approx(0.2)
    assert config.heading_correction_timeout_seconds == pytest.approx(10.0)
    assert not hasattr(config, "qwenvl_model")
    assert not hasattr(config, "vision_backend")
    assert not hasattr(config, "model")
    assert not hasattr(config, "debug")
    assert not hasattr(config, "output_root")

    with pytest.raises(ValueError, match="alignment_prompt is required"):
        LaviraPlannerConfig(
            mission="find chair",
            alignment_prompt=" ",
            global_target="chair",
        )

    with pytest.raises(TypeError):
        LaviraPlannerConfig(
            "find chair", "align to chair", "chair", task_type="eqa"
        )
    with pytest.raises(TypeError):
        LaviraPlannerConfig(
            "find chair", "align to chair", "chair", question="what color?"
        )
    with pytest.raises(ValueError, match="depth range"):
        LaviraPlannerConfig(
            "find chair", "align to chair", "chair",
            nav_handoff_min_depth_m=3.0,
            nav_handoff_max_depth_m=2.0,
        )
    assert LaviraPlannerConfig(
        "find chair", "align to chair", "chair",
        nav_handoff_max_depth_m=15.0,
    ).nav_handoff_max_depth_m == 15.0
    with pytest.raises(ValueError, match="depth range"):
        LaviraPlannerConfig(
            "find chair", "align to chair", "chair",
            nav_handoff_max_depth_m=15.1,
        )
    with pytest.raises(ValueError, match="limits"):
        LaviraPlannerConfig(
            "find chair", "align to chair", "chair",
            heading_settle_seconds=-0.1,
        )


def test_typed_start_starts_one_generation_and_rejects_duplicate() -> None:
    runtime, _ = runtime_with_intents()
    assert runtime.start_navigation(1)
    assert runtime.phase == "nav"
    assert not runtime.start_navigation(1)


def test_space_invalidates_generation_and_late_worker_result() -> None:
    runtime, intents = runtime_with_intents()
    runtime.start_navigation(1)
    runtime.cancel(2, "operator_stop")
    runtime.results.put(WorkerResult(1, task_result(1), None))
    assert runtime.tick() is None
    assert runtime.phase == "listen_wasd"
    assert intents == []


def test_space_discards_unstarted_request_so_navigation_can_restart() -> None:
    runtime, _ = runtime_with_intents()
    assert runtime.start_navigation(1)
    runtime.cancel(2, "operator_stop")

    assert runtime.requests.empty()
    assert runtime.start_navigation(3)
    assert runtime.requests.get_nowait() == runtime.generation


def test_new_worker_result_replaces_stale_full_queue_entry() -> None:
    runtime, _ = runtime_with_intents()
    stale = WorkerResult(1, task_result(1), None)
    current = WorkerResult(3, task_result(3), None)
    runtime.results.put_nowait(stale)

    runtime.publish_worker_result(current)

    assert runtime.results.get_nowait() is current


def test_current_segment_status_wakes_agent_but_does_not_end_generation() -> None:
    runtime, _ = runtime_with_intents()
    runtime.start_navigation(1)
    assert not runtime.accept_status({"type": "sonic_navigation_status", "generation": 0, "state": "reached"})
    assert runtime.phase == "nav"


def test_vla_active_status_is_available_as_start_ack() -> None:
    runtime, _ = runtime_with_intents()
    runtime.start_navigation(1)

    assert runtime.accept_status(
        {
            "generation": 1,
            "skill_id": 3,
            "segment_id": 0,
            "state": "active",
            "reason": "started",
        },
        channel="vla_task_status",
    )

    status = runtime.wait_status(1, 3, 0, 0.1)
    assert status["state"] == "active"
    assert runtime.accept_status(
        {
            "type": "sonic_navigation_status",
            "generation": 1,
            "segment_id": 0,
            "state": "reached",
        }
    )
    assert runtime.phase == "nav"


def test_navdp_world_goal_survives_status_mailbox_handoff() -> None:
    runtime, _ = runtime_with_intents()
    runtime.start_navigation(1)
    payload = {
        "type": "sonic_navigation_status",
        "version": 1,
        "generation": 1,
        "skill_id": 2,
        "segment_id": 7,
        "state": "reached",
        "reason": "goal_within_2m",
        "goal_world": {"x": 1.25, "y": -3.5},
    }

    assert runtime.accept_status(payload, channel="navigation_status")

    assert runtime.wait_status(1, 2, 7, 0.1) == payload


def test_space_wakes_segment_wait_and_invalidates_late_status() -> None:
    runtime, _ = runtime_with_intents()
    runtime.start_navigation(1)
    errors: list[Exception] = []

    def wait() -> None:
        try:
            runtime.wait_status(1, 0, 3, 10.0)
        except Exception as exc:
            errors.append(exc)

    waiter = threading.Thread(target=wait)
    waiter.start()
    time.sleep(0.01)
    runtime.cancel(2, "operator_stop")
    waiter.join(timeout=0.5)

    assert len(errors) == 1
    assert isinstance(errors[0], LaViRAAgentCancelled)
    assert not runtime.accept_status(
        {"generation": 1, "segment_id": 3, "state": "reached"}
    )


def test_warning_and_error_logs_are_promoted_to_structured_runtime_events() -> None:
    pending: queue.SimpleQueue[dict[str, object]] = queue.SimpleQueue()
    handler = LaviraRuntimeEventHandler(pending)

    warning = logging.LogRecord(
        "sonic.lavira",
        logging.WARNING,
        __file__,
        1,
        "model retry %s",
        ("timeout",),
        None,
    )
    handler.handle(warning)
    payload = pending.get_nowait()
    assert payload["type"] == "sonic.runtime_event"
    assert payload["component"] == "lavira"
    assert payload["code"] == "LOG_WARNING"
    assert payload["message"] == "model retry timeout"

    already_reported = logging.LogRecord(
        "sonic.lavira",
        logging.ERROR,
        __file__,
        1,
        "explicit task error",
        (),
        None,
    )
    already_reported.runtime_event_emitted = True
    handler.handle(already_reported)
    assert pending.empty()


def test_runtime_reports_todo_as_structured_event() -> None:
    events = []
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(
            mission="find chair",
            alignment_prompt="align to chair",
            global_target="chair",
        ),
        submit_intent=lambda _name, _parameters: None,
        report_event=lambda level, code, message, **fields: events.append(
            (level, code, message, fields)
        ),
    )

    runtime.start_navigation(7)
    events.clear()
    runtime.report_todo(7, 3, "- [x] Find doorway\n- [ ] Approach chair")

    assert events == [(
        logging.INFO,
        "TODO_UPDATED",
        "LaViRA TODO updated",
        {
            "generation": 7,
            "step": 3,
            "todo_list": "- [x] Find doorway\n- [ ] Approach chair",
        },
    )]


def test_manual_success_discards_late_worker_failure_and_todo():
    runtime, intents = runtime_with_intents()
    events = []
    runtime.report_event = lambda level, code, message, **fields: events.append((code, fields))
    runtime.start_navigation(1)
    runtime.cancel(2, "operator_success")
    events.clear()

    runtime.report_agent_event(logging.ERROR, "TASK_FAILED", "late failure", generation=1)
    runtime.report_agent_event(logging.WARNING, "TASK_CANCELLED", "late cancel", generation=1)
    runtime.report_todo(1, 3, "- [ ] Old manipulation")
    assert events == []
    runtime.publish_worker_result(WorkerResult(1, None, "late VA failure"))
    runtime.tick()
    assert not intents and runtime.phase == "listen_wasd"

    assert runtime.start_navigation(3)
    events.clear()
    runtime.report_agent_event(logging.INFO, "SKILL_STARTED", "new task", generation=3)
    assert [code for code, _ in events] == ["SKILL_STARTED"]
