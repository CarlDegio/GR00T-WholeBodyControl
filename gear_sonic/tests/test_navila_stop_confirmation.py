"""NaVILA observes only after the executor has published the current stop."""

from dataclasses import replace
import threading
import time
from types import SimpleNamespace

import pytest

from gear_sonic.experiments.recording import read_events
from gear_sonic.runtime.gateway.services.control import ControlGatewayRuntime
from gear_sonic.runtime.protocol.navigation import NavigationRuntimeStatus
from gear_sonic.tests import test_experiments_runtime as helpers


fake_navila = helpers.fake_navila


def stopped_status(controller, **overrides):
    generation, skill, segment, _, _ = controller.step
    return replace(NavigationRuntimeStatus(
        generation=generation, skill_id=skill, segment_id=segment,
        timestamp=controller.step_stop_requested[1], mode="stop", source="stop",
        requested_velocity=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0), reason="stopped",
    ), **overrides)


def waiting_for_stop(monkeypatch, tmp_path):
    controller, clock, sent = helpers.supervisor(monkeypatch, tmp_path, "navila_basepose_vla")
    controller.begin_step(dict(
        generation=controller.active[0], skill_id=1, segment_id=1,
        velocity=(0.0, 0.0, 0.3), duration_s=1.0,
    ))
    clock[0] = 1.0
    controller.tick()
    return controller, clock, sent


@pytest.mark.parametrize("changes", [
    {"generation": 0}, {"skill_id": 0}, {"segment_id": 0},
    {"timestamp": 0.0},
    {"mode": "manual_velocity", "reason": "depth_hard_stop"},
    {"velocity": (0.0, 0.0, 0.3)},  # Turning already has zero translation.
    {"velocity": (0.2, 0.0, 0.0)},
    {"requested_velocity": (0.0, 0.0, 0.3)},
])
def test_stale_or_nonzero_output_cannot_complete_step(monkeypatch, tmp_path, changes):
    controller, _, sent = waiting_for_stop(monkeypatch, tmp_path)
    assert not controller.accept_executor_status(stopped_status(controller, **changes))
    assert controller.step is not None and controller.gateway.navigation.mode == "lavira_nav"
    assert not any(isinstance(item, tuple) and item[0] == "navigation_status" for item in sent)


def test_sent_stop_is_confirmed_even_with_safety_reason(monkeypatch, tmp_path):
    controller, _, sent = waiting_for_stop(monkeypatch, tmp_path)
    status = stopped_status(controller, reason="radar_timeout")
    runtime = object.__new__(ControlGatewayRuntime)
    runtime.experiment = controller
    runtime.executor_status_sub = SimpleNamespace(recv=lambda: status.to_json().encode())
    runtime.show_event = lambda *a, **k: pytest.fail("valid executor report rejected")
    runtime._receive_executor_status()
    assert controller.step is None and controller.step_stop_requested is None
    assert sent[-1][1]["executor_zero_confirmed"] is True
    assert sent[-1][1]["velocity"] == [0.0, 0.0, 0.0]
    assert not controller.accept_executor_status(status)
    event = next(e for e in read_events(controller.recorder.path) if e["type"] == "navila_stop_confirmed")
    assert event["executor_status"]["velocity"] == {"vx": 0.0, "vy": 0.0, "wz": 0.0}


def test_missing_stop_confirmation_retries_then_cancels(monkeypatch, tmp_path):
    controller, clock, sent = waiting_for_stop(monkeypatch, tmp_path)
    status = stopped_status(controller)
    clock[0] = 1.1
    controller.tick()
    assert len([item for item in sent if isinstance(item, tuple) and item[0] == "stop"]) == 2
    clock[0] = 1.0 + controller.config["navila"]["stop_ack_timeout_s"]
    assert not controller.accept_executor_status(status)
    controller.tick()
    assert controller.active is None and controller.step is None
    assert sent[-1].agent_event == "cancel_navigation"
    assert sent[-1].reason == "navila_stop_ack_timeout"
    assert not controller.accept_executor_status(status)


def test_early_zero_cannot_finish_an_action_still_executing(monkeypatch, tmp_path):
    controller, _, sent = waiting_for_stop(monkeypatch, tmp_path)
    old_status = stopped_status(controller)
    assert controller.accept_executor_status(old_status)
    controller.begin_step(dict(
        generation=controller.active[0], skill_id=1, segment_id=2,
        velocity=(0.4, 0.0, 0.0), duration_s=1.0,
    ))
    assert not controller.accept_executor_status(replace(old_status, segment_id=2, timestamp=time.time()))
    assert controller.step is not None


def test_agent_waits_for_executor_zero_before_next_frame(monkeypatch, tmp_path, fake_navila):
    controller, clock, _ = helpers.supervisor(monkeypatch, tmp_path, "navila_basepose_vla")
    agent, camera, _, _ = helpers.make_agent(monkeypatch, tmp_path, "navila_basepose_vla")
    waiting, released, cancelled = (threading.Event() for _ in range(3))
    received, results, errors = [], [], []
    camera.current_sonic_yaw = lambda: pytest.fail("NaVILA stop must not read IMU")
    camera.current_pose = lambda: pytest.fail("NaVILA stop must not read odometry")
    agent.cancelled = lambda generation: cancelled.is_set()
    agent._before_nav_handoff = lambda *args: None
    agent.submit_intent = lambda name, parameters: controller.begin_step(parameters)

    def dispatch(name, parameters):
        assert name == "navigation_status"
        received.append(parameters)
        released.set()

    def wait_status(*args):
        waiting.set()
        assert released.wait(3)
        return received[0] if received else {}

    controller.gateway.dispatch_navigation_event = dispatch
    agent.wait_status = wait_status

    def execute():
        try:
            results.append(agent._move_to(controller.active[0], 1, {}, "navigate"))
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=execute, daemon=True)
    worker.start()
    try:
        assert waiting.wait(2)
        clock[0] = 1.25
        controller.tick()
        assert not released.is_set()
        assert agent.navila.sequence == 1 and len(camera.capture_cutoffs) == 1
        assert not controller.accept_executor_status(stopped_status(controller, velocity=(0.0, 0.0, 0.3)))
        assert len(camera.capture_cutoffs) == 1
        confirmed_ns = time.time_ns()
        assert controller.accept_executor_status(stopped_status(controller))
        worker.join(3)
        assert not worker.is_alive() and not errors and len(results) == 1
        assert agent.navila.sequence == 2
        assert len(camera.capture_cutoffs) == 2 and camera.capture_cutoffs[1] >= confirmed_ns
    finally:
        cancelled.set()
        released.set()
        worker.join(3)
