"""Nearfield keys select an isolated, recorded ALIGN/VLA trial without navigation."""

import copy
import threading
import time
from types import SimpleNamespace, MethodType

import av
import numpy as np
import pytest
import yaml

from gear_sonic.experiments.agent import ExperimentAgent
from gear_sonic.experiments.config import CONFIGS, resolve
from gear_sonic.experiments.demo_video import DemoSession
from gear_sonic.experiments.recording import read_events
from gear_sonic.experiments.results import trial_rows
from gear_sonic.experiments.supervisor import ExperimentSupervisor
from gear_sonic.runtime.gateway.control import ControlGatewayCore, NavigationControlState, OperatorConsoleRouter
from gear_sonic.runtime.gateway.services.control import ControlGatewayRuntime
from gear_sonic.tests import test_lavira_agent as helpers
from gear_sonic.tests.test_demo_recording import jpeg, metadata
from gear_sonic.utils.inference.lavira import service
from gear_sonic.utils.operator.cv_viewer import NavigationViewerState


def profile(tmp_path, mission="t1_bottle_to_basket.yaml"):
    raw, missing = resolve(CONFIGS / "full_vln.yaml", mission_path=CONFIGS.parent / "missions" / mission)
    assert not missing
    raw["components"]["experiment"].update(run_dir=str(tmp_path), session_id="near-test", demo=True)
    path = tmp_path / "runtime.yaml"
    path.write_text(yaml.safe_dump(raw))
    return SimpleNamespace(components=raw["components"]), path


def gateway(monkeypatch, p):
    events = []
    g = SimpleNamespace(profile=p, navigation=NavigationControlState(), pending_vla_completion=None,
        navigation_pub=SimpleNamespace(send_string=lambda _: None),
        dispatch_navigation_event=lambda name, params: events.append((name, params)),
        show_event=lambda *a, **kw: None)
    g.publish_navigation_action = MethodType(ControlGatewayRuntime.publish_navigation_action, g)
    g.experiment = ExperimentSupervisor(g)
    g.experiment.recorder.write("session", experiment=p.components["experiment"])
    monkeypatch.setattr("gear_sonic.experiments.snapshots.record_pose_async", lambda *a, **kw: None)
    return g, events


def press(g, key):
    command = OperatorConsoleRouter().accept_line(key, core=ControlGatewayCore()).command
    assert command.name == "navigation_key"
    ControlGatewayRuntime._handle_navigation_key(g, command)


@pytest.mark.parametrize("filename,key", [("t1_bottle_to_basket.yaml", "m"), ("t2_bag_and_box.yaml", "m"),
    ("t3_bowl_to_bin.yaml", "m"), ("bag_then_bin.yaml", "m"), ("bag_then_bin.yaml", "t")])
def test_real_factory_runs_one_alignment_and_operation_with_matching_backend(monkeypatch, tmp_path, filename, key):
    p, path = profile(tmp_path, filename)
    original = copy.deepcopy(p.components["experiment"])
    base, camera, client, intents, _ = helpers.build_agent([], alignment_groundings=[helpers.alignment_grounding()],
        postchecks=[helpers.ready_to_manipulate(), helpers.ready_to_manipulate(), helpers.task_complete()])
    capture = camera.capture_rgb
    camera.capture_rgb = lambda *, camera_stream="chest_view", timeout_ms=None: capture(camera_stream=camera_stream)
    runtime = SimpleNamespace(submit_intent=base.submit_intent, wait_status=base.wait_status,
        is_cancelled=lambda _: False, pop_failure=lambda *a: None,
        report_agent_event=lambda *a, **kw: None, report_todo=lambda *a: None)
    monkeypatch.setattr(service, "SensorGatewayRGBDCamera", lambda *a, **kw: camera)
    monkeypatch.setattr(service, "LaViRAClient", lambda **kw: client)
    agent = service._agent(service.load_lavira_config(str(path)), "unused", runtime, nearfield=True, nearfield_key=key)
    assert type(agent) is ExperimentAgent
    agent.manipulation_window_seconds = 0
    agent.sleep = lambda _: None
    result = agent.run(1)
    assert result.reason == "manipulation_completed", result
    assert not client.la_calls
    assert [name for name, _ in intents if name.startswith("navigation")] == []
    assert len([1 for name, _ in intents if name == "start_base_pose"]) == 1
    start, = [params for name, params in intents if name == "start_vla_task"]
    skill = original["skills"][agent.experiment["nearfield_skill"]]
    align, = [params for name, params in intents if name == "start_base_pose"]
    expected_distances = ({"raw_head_target_distance_m": 1.25, "raw_chest_target_distance_m": 0.9,
                           "raw_head_target_confidence": 0.3, "raw_chest_target_confidence": 0.3,
                           "raw_stable_frames": 3, "raw_lateral_tolerance_m": 0.12}
                          if filename == "bag_then_bin.yaml" and key == "t" else
                          {"raw_head_target_confidence": 0.1, "raw_chest_target_confidence": 0.1}
                          if filename == "bag_then_bin.yaml" and key == "m" else {})
    assert align.get("base_pose", {}) == expected_distances
    assert start["skill_key"] == skill["backend"]
    assert start["handoff_context"] == skill["vla_trained_prompt"]
    assert not start.get("continue_mission")
    assert client.alignment_grounding_calls[0]["manipulation_prompt"] == skill["vla_prompt"]
    expected = agent.experiment["nearfield_expected_postcondition"]
    assert all(call["mission"] == skill["vla_prompt"] for call in client.postcheck_calls)
    assert all(call["expected_postcondition"] == expected
               for call in client.postcheck_calls if call.get("skill") == "MANIPULATE")
    stop, = [params for name, params in intents if name == "stop_vla_task"]
    assert stop["exit_policy"] == skill["exit"]
    assert start["timeout_seconds"] == skill.get("timeout_seconds", 180)
    assert start["on_timeout"] == "fail"
    assert agent.experiment["demo"] and agent.client.client.demo_recorder is not None
    assert p.components["experiment"] == original
    normal = service._agent(service.load_lavira_config(str(path)), "unused", runtime)
    from gear_sonic.experiments.flexible import FlexibleExperimentAgent
    assert type(normal) is FlexibleExperimentAgent
    assert normal.mission == original["mission"]


def test_m_restart_n_and_manual_result_keep_trial_modes_separate(monkeypatch, tmp_path):
    p, _ = profile(tmp_path)
    g, dispatched = gateway(monkeypatch, p)
    press(g, "m")
    assert dispatched[-1][1]["nearfield"] is True
    first = g.experiment.active[0]
    press(g, "m")
    second = g.experiment.active[0]
    assert second > first
    g.experiment.confirm_result(second, success=True, completion_time_s=2.0,
        completed_monotonic_ns=time.monotonic_ns(), completed_wall_time_ns=time.time_ns())
    # Mirror the operator result's release before launching the next full run.
    g.navigation.handle_key(" ", now=time.monotonic())
    press(g, "n")
    assert "nearfield" not in dispatched[-1][1]
    rows = trial_rows(read_events(g.experiment.recorder.path))
    assert [r["method"] for r in rows] == ["near_dual", "near_dual", "full_vln"]
    assert [r["entry_stage"] for r in rows] == ["alignment", "alignment", "navigation"]
    assert rows[0]["terminal_reason"] == "restarted_with_m"
    assert rows[1]["success"] is True and rows[1]["nav_success"] is None
    assert g.experiment.config["flexible_mission"] is True
    assert all(r["task_id"] == "T1" for r in rows)


def test_unconfigured_t_rejected_before_cancelling_active_full_trial(monkeypatch, tmp_path):
    p, _ = profile(tmp_path)
    g, dispatched = gateway(monkeypatch, p)
    press(g, "n")
    active = g.experiment.active
    before = list(dispatched)
    with pytest.raises(ValueError, match="No nearfield operation"):
        press(g, "t")
    assert g.experiment.active == active and dispatched == before


def test_t4_keys_select_distinct_operations_and_n_restores_whole_mission(monkeypatch, tmp_path):
    p, _ = profile(tmp_path, "bag_then_bin.yaml")
    g, dispatched = gateway(monkeypatch, p)
    for key, skill, milestone in [("m", "pick_bag", "stable_bag_grasp"),
                                  ("t", "step_bin", "bag_in_bin_released")]:
        press(g, key)
        assert dispatched[-1][1]["nearfield_key"] == key
        assert g.experiment.config["nearfield_skill"] == skill
        assert g.experiment.config["task"]["milestones"] == [milestone]
    press(g, "n")
    assert "nearfield_key" not in dispatched[-1][1]
    rows = trial_rows(read_events(g.experiment.recorder.path))
    assert [r["task_id"] for r in rows] == ["T4"] * 3
    assert [r["nearfield_skill"] for r in rows] == ["pick_bag", "step_bin", None]
    assert [r["start_key"] for r in rows] == ["m", "t", None]
    assert [r["terminal_reason"] for r in rows[:2]] == ["restarted_with_t", "restarted_with_n"]
    assert len(g.experiment.config["skills"]) == 2 and g.experiment.config["flexible_mission"]


@pytest.mark.parametrize("key", ["m", "t"])
def test_worker_receives_nearfield_option_and_cancellation_discards_it(tmp_path, key):
    p, path = profile(tmp_path, "bag_then_bin.yaml")
    runtime = service.LaviraPlannerRuntime(service.load_lavira_config(str(path)), submit_intent=lambda *a: None)
    runtime.experiment = p.components["experiment"]
    runtime.start_navigation(1, nearfield=True, nearfield_key=key)
    runtime.cancel(2, "operator_cancel")
    assert runtime.requests.empty()
    called = []
    completed = threading.Event()

    def factory(**kwargs):
        called.append(kwargs)
        return SimpleNamespace(run=lambda generation: None, camera=SimpleNamespace(close=completed.set))

    worker = threading.Thread(target=service.run_agent_worker, args=(factory, runtime))
    worker.start()
    try:
        assert runtime.start_navigation(3, nearfield=True, nearfield_key=key)
        assert completed.wait(2)
        assert called == [{"nearfield": True, "nearfield_key": key}]
    finally:
        runtime.shutdown()
        worker.join(2)
    assert not worker.is_alive()


@pytest.mark.parametrize("filename,key", [("t1_bottle_to_basket.yaml", "m"),
    ("bag_then_bin.yaml", "m"), ("bag_then_bin.yaml", "t")])
def test_nearfield_trial_exports_both_cameras_with_mask_and_line(monkeypatch, tmp_path, filename, key):
    p, _ = profile(tmp_path, filename)
    g, _ = gateway(monkeypatch, p)
    session = DemoSession(p)
    try:
        press(g, key)
        gen = g.experiment.active[0]
        state = NavigationViewerState(active=True, generation=gen, camera_stream="camera/ego_view",
            target_camera_stream="camera/ego_view", yaw_align_geometry_camera_stream="camera/chest_view",
            target_bbox_xyxy=(5, 40, 45, 85), target_lateral_anchor_px=(25, 65),
            yaw_align_edge_endpoints_px=((55, 60), (115, 60)), overlay_image_size=(128, 96),
            completed_yaw_align_target_mask_row_spans=tuple((row, 60, 110) for row in range(65, 90)))
        session.poll(state)
        assert session.take is not None
        start = session.take.start["monotonic_ns"]
        session.take.overlay(state, start)
        for name in ("head", "chest"):
            session.take.add(name, jpeg(), metadata(1), wall_ns=time.time_ns(), mono_ns=start)
        with monkeypatch.context() as clock:
            clock.setattr("gear_sonic.experiments.recording.time.monotonic_ns", lambda: start + 300_000_000)
            g.experiment.finish(gen, "reached", "manipulation_completed")
        session.poll(state)
    finally:
        session.close()
    video = tmp_path / f"demo/near-test-g{gen}/video"
    frames = {}
    for name in ("head", "chest"):
        with av.open(str(video / f"{name}.mp4")) as container:
            frames[name] = list(container.decode(video=0))[1].to_ndarray(format="bgr24")
    assert frames["head"][60, 5, 1] > 130
    assert frames["chest"][78, 85, 0] > 75
    assert np.linalg.norm(frames["chest"][60, 85].astype(float) - 45) > 80
    assert any(e["type"] == "demo_video_saved" for e in read_events(g.experiment.recorder.path))
