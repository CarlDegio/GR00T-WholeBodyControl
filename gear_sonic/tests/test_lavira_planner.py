from __future__ import annotations

import threading
import time

import pytest

from gear_sonic.utils.inference.lavira.agent import LaViRAAgentCancelled
from gear_sonic.utils.inference.lavira.object_nav import ObjectNavResult
from gear_sonic.utils.inference.lavira.service import (
    LaviraPlannerConfig,
    LaviraPlannerRuntime,
    WorkerResult,
    result_to_goal,
    run_inference_worker,
)


def nav_result(
    *,
    distance: float = 2.0,
    angle_deg: float = -30.0,
) -> ObjectNavResult:
    geometry = {"mean_range": distance, "angle_deg": angle_deg}
    return ObjectNavResult(
        outcome="NAVIGATE",
        policy={"target": "chair", "target_type": "global_target", "confidence": 0.9},
        geometry=geometry,
    )


def runtime_with_intents():
    intents: list[tuple[str, dict]] = []
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig("find chair", "chair"),
        submit_intent=lambda name, parameters: intents.append((name, dict(parameters))),
    )
    return runtime, intents


def test_lavira_config_exposes_dual_local_agent_endpoints() -> None:
    config = LaviraPlannerConfig("find chair", "chair")

    assert config.task_type == "object_nav"
    assert config.la_model == "Qwen3.5-27B-Q4_K_M"
    assert config.va_model == "Qwen3.5-27B-Q4_K_M"
    assert config.la_base_url.endswith(":8000/v1")
    assert config.va_base_url.endswith(":8001/v1")
    assert not hasattr(config, "qwenvl_model")
    assert not hasattr(config, "vision_backend")
    assert not hasattr(config, "model")
    assert not hasattr(config, "debug")
    assert not hasattr(config, "output_root")


def test_result_goal_maps_camera_right_to_negative_base_y() -> None:
    assert result_to_goal(nav_result()) == pytest.approx((1.73205, -1.0))


def test_typed_start_starts_one_generation_and_rejects_duplicate() -> None:
    runtime, _ = runtime_with_intents()
    assert runtime.start_navigation(1)
    assert runtime.phase == "nav"
    assert not runtime.start_navigation(1)


def test_worker_result_publishes_goal_not_timed_velocity_sequence() -> None:
    runtime, intents = runtime_with_intents()
    runtime.start_navigation(1)
    runtime.results.put(WorkerResult(1, nav_result(), None))
    runtime.tick()
    name, parameters = intents[-1]
    assert name == "navigation_goal"
    assert parameters["goal_base"] == pytest.approx((1.73205, -1.0))
    assert "segments" not in parameters
    assert "duration_s" not in parameters


def test_retired_single_cycle_result_is_still_accepted_without_timing_publish() -> None:
    runtime, _ = runtime_with_intents()
    runtime.start_navigation(1)
    timing = {"camera_rgbd": 0.1, "api_inference": 1.5, "total": 1.7}
    result = nav_result()
    result.geometry["timing_s"] = timing
    runtime.results.put(WorkerResult(1, result, None))

    assert runtime.tick() is None


def test_space_invalidates_generation_and_late_worker_result() -> None:
    runtime, intents = runtime_with_intents()
    runtime.start_navigation(1)
    runtime.cancel(2, "operator_stop")
    runtime.results.put(WorkerResult(1, nav_result(), None))
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
    stale = WorkerResult(1, nav_result(), None)
    current = WorkerResult(3, nav_result(angle_deg=15.0), None)
    runtime.results.put_nowait(stale)

    runtime.publish_worker_result(current)

    assert runtime.results.get_nowait() is current


def test_current_segment_status_wakes_agent_but_does_not_end_generation() -> None:
    runtime, _ = runtime_with_intents()
    runtime.start_navigation(1)
    assert not runtime.accept_status({"type": "sonic_navigation_status", "generation": 0, "state": "reached"})
    assert runtime.phase == "nav"
    assert runtime.accept_status(
        {
            "type": "sonic_navigation_status",
            "generation": 1,
            "segment_id": 0,
            "state": "reached",
        }
    )
    assert runtime.phase == "nav"


def test_worker_releases_da_after_rgbd_capture() -> None:
    intents: list[tuple[str, dict]] = []
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig("find chair", "chair"),
        submit_intent=lambda name, parameters: intents.append((name, dict(parameters))),
    )

    class Runner:
        def run_once(self, *, rgbd_capture_complete) -> ObjectNavResult:
            rgbd_capture_complete()
            return nav_result()

        def close(self) -> None:
            pass

    runner = Runner()

    def factory() -> Runner:
        return runner

    worker = threading.Thread(target=run_inference_worker, args=(factory, runtime))
    worker.start()
    assert runtime.start_navigation(1)
    result = runtime.results.get(timeout=1.0)
    runtime.shutdown()
    worker.join(timeout=1.0)

    assert result.result is not None
    assert result.error is None
    assert ("lavira_rgbd_captured", {"generation": 1}) in intents


def test_space_wakes_segment_wait_and_invalidates_late_status() -> None:
    runtime, _ = runtime_with_intents()
    runtime.start_navigation(1)
    errors: list[Exception] = []

    def wait() -> None:
        try:
            runtime.wait_status(1, 3, 10.0)
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
