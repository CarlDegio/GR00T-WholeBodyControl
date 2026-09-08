"""Exercise geometric startup and manual VLA recovery without robot connections."""

import threading
from types import SimpleNamespace

import pytest

from gear_sonic.experiments import base_pose
from gear_sonic.runtime.gateway.control import NavigationControlState
from gear_sonic.utils.inference.vla.runtime import _VlaRuntimeState
from gear_sonic.utils.inference.vla.service import _build_experiment_inference_callbacks


class GeometricServiceHarness:
    def __init__(self, monkeypatch, *, manual=False, gap_s=0.5, rounds=1, perceive=False):
        self.now = 1.0
        self.rounds = 0
        self.sent = []
        self.workers = []
        self.timeouts = []
        self.nav = NavigationControlState()
        start = self.nav.handle_key("b" if manual else "n", now=self.now)
        parameters = dict(generation=start.generation, skill_id=0, segment_id=0)
        if not manual:
            parameters.update(skill_id=2, segment_id=11, target="explicit basket", yaw_align_target="explicit basket")
            self.nav.accept_base_pose_start(parameters)
        commands = iter([SimpleNamespace(name="start_base_pose", parameters=parameters)])
        geometry = dict(
            target="blue basket", distance_m=1.1, longitudinal_tolerance_m=0.1,
            angle_tolerance_deg=9.0, coarse_wz=0.3, vx=0.4, fine_wz=0.2,
            stable_frames=3, timeout_s=60.0,
        )
        self.config = SimpleNamespace(
            dual_head_camera_stream="ego_view", dual_head_depth_stream="ego_depth",
            sensor_gateway_request_timeout_ms=100, sensor_gateway_max_age_ms=1000,
            sensor_gateway_max_skew_ms=5, planner_hz=50, raw_camera_stale_s=0.2,
        )
        self.profile = SimpleNamespace(
            components={"experiment": {"geometric": geometry}}, endpoint_uri=lambda name: name,
        )

        def send(name, parameters):
            self.sent.append((name, parameters))
            if name == "base_pose_velocity" and parameters["generation"] == self.nav.generation:
                self.nav.accept_base_pose_velocity(parameters, now=self.now)

        def sleep(_duration):
            self.now += gap_s
            self.rounds += 1
            timeout = self.nav.tick(now=self.now)
            if timeout is not None:
                self.timeouts.append(timeout)
            if self.rounds >= rounds:
                raise KeyboardInterrupt

        class SingleFrameStop:
            # Run one synthetic camera frame when perception is enabled.
            def __init__(self):
                self.checks = 0

            def is_set(self):
                self.checks += 1
                return self.checks > 1

            def set(self):
                self.checks = 2

        def worker(*, target, args, **kwargs):
            self.workers.append(args)
            return SimpleNamespace(start=lambda: target(*args) if perceive else None)

        def close():
            pass

        monkeypatch.setattr(base_pose, "InferenceServiceContext", lambda *args: SimpleNamespace(close=close))
        monkeypatch.setattr(base_pose, "ControlGatewaySubscriber", lambda *a, **k: SimpleNamespace(
            read_command=lambda: next(commands, None), close=close,
        ))
        monkeypatch.setattr(base_pose, "ControlGatewayIntentClient", lambda *a, **k: SimpleNamespace(
            send=send, close=close,
        ))
        monkeypatch.setattr(base_pose, "SensorGatewayDualBasePoseCamera", lambda *a, **k: SimpleNamespace(
            begin_generation=lambda generation: None, close=close,
            capture_stream=lambda *a, **k: SimpleNamespace(rgb=None, timestamp=10.0),
        ))
        monkeypatch.setattr(base_pose, "threading", SimpleNamespace(
            Thread=worker, Lock=threading.Lock, Event=SingleFrameStop,
        ))
        monkeypatch.setattr(base_pose, "time", SimpleNamespace(monotonic=lambda: self.now, sleep=sleep))
        monkeypatch.setattr(base_pose, "head_calibration", lambda config: None)
        monkeypatch.setattr(base_pose, "tracker_for", lambda *a: SimpleNamespace(track=lambda rgb: []))
        monkeypatch.setattr(base_pose, "_resolve_target", lambda *a: (SimpleNamespace(track_id=1), None, None))
        monkeypatch.setattr(base_pose, "_observation", lambda *a, **k: SimpleNamespace(
            target=SimpleNamespace(forward_m=2.0, right_m=0.0),
        ))

    def run(self):
        base_pose.run_geometric_service(self.config, self.profile)


def test_geometric_waiting_for_first_target_does_not_acquire_motion_lease(monkeypatch):
    harness = GeometricServiceHarness(monkeypatch)
    harness.run()
    assert not harness.timeouts
    assert all(p["velocity"] == [0.0, 0.0, 0.0] for name, p in harness.sent)
    assert all(p["action"] == "hold" for name, p in harness.sent[:-1])
    assert harness.workers[0][1] == "explicit basket"


def test_manual_b_uses_experiment_geometry_target(monkeypatch):
    harness = GeometricServiceHarness(monkeypatch, manual=True)
    harness.run()
    assert harness.workers[0][1] == "blue basket"
    assert not harness.timeouts


def test_geometric_wait_is_still_bounded_by_alignment_timeout(monkeypatch):
    harness = GeometricServiceHarness(monkeypatch, gap_s=10.0, rounds=7)
    harness.run()
    assert not harness.timeouts
    statuses = [p for name, p in harness.sent if name == "base_pose_status"]
    assert len(statuses) == 1
    assert statuses[0]["reason"] == "geometric_timeout"
    assert harness.sent[-2][1]["action"] == "stop"


def test_geometric_actual_motion_keeps_velocity_watchdog(monkeypatch):
    harness = GeometricServiceHarness(monkeypatch, perceive=True)
    harness.run()
    assert any(p.get("velocity") == [0.4, 0.0, 0.0] for name, p in harness.sent)
    assert len(harness.timeouts) == 1
    assert harness.timeouts[0].reason == "base_pose_velocity_timeout"


def inference_callbacks(*, active):
    state = _VlaRuntimeState(
        task_active=active, task_generation=4, task_skill_id=2, inference_generation=8,
        cpp_loop_running=True, cpp_mode="POSE", pause_loop=False,
    )
    prepared, inferred, recorded = [], [], []

    def prepare(*, observation_callback):
        prepared.append("sensor observation")
        if observation_callback is not None:
            observation_callback("exact camera input", "exact robot input")
        return {"observation": "prepared"}, {}

    def infer(epoch, observation):
        inferred.append((epoch, observation))
        return {"motion_token": "policy action"}, {}

    callbacks = _build_experiment_inference_callbacks(
        state, prepare_observation=prepare, run_inference=infer,
        first_observation=SimpleNamespace(capture=lambda *args: recorded.append(args)),
    )
    return state, callbacks, prepared, inferred, recorded


@pytest.mark.parametrize("active", [False, True])
def test_experiment_profile_allows_manual_and_agent_vla(active):
    state, (prepare, infer), prepared, inferred, recorded = inference_callbacks(active=active)
    observation = prepare(state.inference_generation)
    assert observation is not None
    result = infer(state.inference_generation, observation[0])
    assert result[0]["motion_token"] == "policy action"
    assert prepared == ["sensor observation"]
    assert len(inferred) == 1
    expected = [((4, 2), "exact camera input", "exact robot input")] if active else []
    assert recorded == expected


@pytest.mark.parametrize("active", [False, True])
def test_cancelled_epoch_cannot_prepare_or_infer_after_manual_recovery(active):
    state, (prepare, infer), prepared, inferred, recorded = inference_callbacks(active=active)
    old_epoch = state.inference_generation
    observation = prepare(old_epoch)
    assert observation is not None
    state.inference_generation += 1
    state.task_active = False
    assert infer(old_epoch, observation[0]) is None
    assert prepare(old_epoch) is None
    assert not inferred and not recorded

    fresh = prepare(state.inference_generation)
    assert fresh is not None
    assert infer(state.inference_generation, fresh[0]) is not None
    assert len(prepared) == 2 and len(inferred) == 1
    assert not recorded
