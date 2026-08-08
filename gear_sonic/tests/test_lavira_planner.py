from __future__ import annotations

import threading

import pytest

from gear_sonic.scripts.lavira_planner import (
    LaviraPlannerConfig,
    LaviraPlannerRuntime,
    WorkerResult,
    run_inference_worker,
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


def runtime_with_intents():
    intents: list[tuple[str, dict]] = []
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig("find chair", "chair"),
        submit_intent=lambda name, parameters: intents.append((name, dict(parameters))),
    )
    return runtime, intents


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
    runtime.tick(2.0)
    name, parameters = intents[-1]
    assert name == "navigation_goal"
    assert parameters["goal_base"] == pytest.approx((1.73205, -1.0))
    assert "segments" not in parameters
    assert "duration_s" not in parameters


def test_space_invalidates_generation_and_late_worker_result() -> None:
    runtime, intents = runtime_with_intents()
    runtime.start_navigation(1)
    runtime.cancel(2, "operator_stop")
    runtime.results.put(WorkerResult(1, nav_result(), None))
    runtime.tick(2.0)
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


def test_current_generation_status_returns_to_listen_but_stale_status_is_ignored() -> None:
    runtime, _ = runtime_with_intents()
    runtime.start_navigation(1)
    assert not runtime.accept_status({"type": "sonic_navigation_status", "generation": 0, "state": "reached"})
    assert runtime.phase == "nav"
    assert runtime.accept_status({"type": "sonic_navigation_status", "generation": 1, "state": "reached"})
    assert runtime.phase == "listen_wasd"


def test_warmup_failure_does_not_kill_inference_worker() -> None:
    intents: list[tuple[str, dict]] = []
    logs: list[str] = []
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig("find chair", "chair"),
        submit_intent=lambda name, parameters: intents.append((name, dict(parameters))),
        logger=logs.append,
    )

    class Runner:
        def __init__(self, *, fail_warmup: bool) -> None:
            self.fail_warmup = fail_warmup

        def warmup(self) -> None:
            if self.fail_warmup:
                raise RuntimeError("missing API key")

        def run_once(self) -> ObjectNavResult:
            return nav_result()

        def close(self) -> None:
            pass

    attempts = 0

    def factory() -> Runner:
        nonlocal attempts
        attempts += 1
        return Runner(fail_warmup=attempts == 1)

    worker = threading.Thread(target=run_inference_worker, args=(factory, runtime))
    worker.start()
    assert runtime.start_navigation(1)
    result = runtime.results.get(timeout=1.0)
    runtime.shutdown()
    worker.join(timeout=1.0)

    assert result.result is not None
    assert result.error is None
    assert attempts == 2
    assert any("warmup failed: missing API key" in message for message in logs)
