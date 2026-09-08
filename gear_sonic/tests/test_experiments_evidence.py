"""Protocol, first-observation provenance, annotations, and known table results."""

from __future__ import annotations

import json
import queue
import threading
import time
from types import SimpleNamespace

import cv2
import msgpack
import numpy as np
import pytest

from gear_sonic.experiments.navila import NavilaClient, parse_action
from gear_sonic.experiments.recording import Recorder, read_events
from gear_sonic.experiments.results import export, main, rate, summarize, trial_rows
from gear_sonic.experiments.snapshots import FirstObservationRecorder
from gear_sonic.tests.test_experiments_runtime import experiment


def test_navila_msgpack_wire_reset_history_and_sequence(monkeypatch, tmp_path):
    """An in-memory REP transport decodes the actual MessagePack wire bytes."""
    import gear_sonic.experiments.navila as module

    requests, server_history = [], []

    class Socket:
        def setsockopt(self, *args):
            pass

        def connect(self, endpoint):
            assert endpoint == "tcp://127.0.0.1:30000"

        def close(self):
            pass

        def send(self, data):
            self.request = msgpack.unpackb(data, raw=False)
            requests.append(self.request)

        def recv(self):
            r = self.request
            assert r["api_token"] == "fake-test-token"
            if r["endpoint"] == "reset":
                server_history.clear()
            if r["endpoint"] != "get_action":
                return msgpack.packb({"status": "ok"}, use_bin_type=True)
            d = r["data"]
            assert d["type"] == "navila_camera_frame" and d["version"] == 1
            assert d["instruction"] == "test route"
            assert cv2.imdecode(np.frombuffer(d["image_jpeg"], np.uint8), cv2.IMREAD_COLOR).shape == (8, 8, 3)
            server_history.append(d["sequence"])
            return msgpack.packb(
                dict(type="navila_action", version=1, sequence=d["sequence"], text="Move forward 25 cm."),
                use_bin_type=True,
            )

    monkeypatch.setattr(
        module.zmq, "Context", SimpleNamespace(instance=lambda: SimpleNamespace(socket=lambda *_: Socket()))
    )
    monkeypatch.setenv("NAVILA_API_TOKEN", "fake-test-token")
    client = NavilaClient(experiment(tmp_path)["navila"])
    image = np.zeros((8, 8, 3), np.uint8)
    client.reset()
    assert client.action(image, "test route")[1] == pytest.approx(0.25 / 0.3)
    client.action(image, "test route")
    assert server_history == [1, 2]
    client.reset()
    client.action(image, "test route")
    assert server_history == [1]
    assert [r["endpoint"] for r in requests].count("reset") == 2
    with pytest.raises(ValueError, match="sequence"):
        parse_action(dict(type="navila_action", version=1, sequence=99, text="stop"), 1, client.config)
    with pytest.raises(ValueError, match="Unsupported"):
        parse_action(dict(type="navila_action", version=1, sequence=1, text="move backward 2m"), 1, client.config)


def test_navila_real_zmq_req_rep_timeout_and_error(monkeypatch, tmp_path):
    """Exercise actual ZMQ sockets over inproc; no network or real service."""
    import zmq

    import gear_sonic.experiments.navila as module

    context = zmq.Context()
    server = context.socket(zmq.REP)
    server.setsockopt(zmq.LINGER, 0)
    server.bind("inproc://navila-test")

    def serve():
        for response in ({"status": "ok"}, {"status": "ok"}, {"error": "fake model error"}):
            request = msgpack.unpackb(server.recv(), raw=False)
            assert request["endpoint"] in {"ping", "reset", "get_action"}
            server.send(msgpack.packb(response, use_bin_type=True))

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()

    class Socket:
        def __init__(self):
            self.socket = context.socket(zmq.REQ)

        def __getattr__(self, name):
            return getattr(self.socket, name)

        def connect(self, endpoint):
            self.socket.connect("inproc://navila-test")

    monkeypatch.setattr(
        module.zmq, "Context", SimpleNamespace(instance=lambda: SimpleNamespace(socket=lambda *_: Socket()))
    )
    config = experiment(tmp_path)["navila"]
    config["timeout_s"] = 0.05
    try:
        client = NavilaClient(config)
        client.reset()
        with pytest.raises(RuntimeError, match="fake model error"):
            client.action(np.zeros((8, 8, 3), np.uint8), "route")
        worker.join(1.0)
        with pytest.raises(zmq.Again):
            client.call("ping")
    finally:
        server.close()
        context.term()


def test_head_camera_buffer_never_subscribes_to_chest():
    from gear_sonic.tests.test_base_pose_dual_yolo_adapter import _FakeDualGatewayClient
    from gear_sonic.utils.inference.base_pose.sensor import SensorGatewayDualBasePoseCamera

    clients = {}

    def factory(stream):
        assert stream == "ego_view"
        clients[stream] = _FakeDualGatewayClient(stream)
        return clients[stream]

    camera = SensorGatewayDualBasePoseCamera(
        "inproc://unused",
        stream_depths={"ego_view": "camera/ego_view_depth"},
        allow_single=True,
        client_factory=factory,
        timeout_ms=100,
        poll_hz=200.0,
    )
    try:
        clients["ego_view"].publish(1_000_000_000)
        snap = camera.capture_stream("ego_view")
        assert snap.depth_aligned_to == "ego_view" and set(camera._buffers) == {"ego_view"}
    finally:
        camera.close()


def test_snapshot_saves_first_exact_four_rgb_and_rejects_later_depth(monkeypatch, tmp_path):
    exp = experiment(tmp_path)
    requests = []

    class Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read_snapshot(self, req):
            requests.append(req)
            stream = req.streams[0]
            if stream == "ros/odometry":
                raise RuntimeError("pose unavailable")
            stamp = req.anchor_timestamp_ns + (6_000_000 if "chest" in stream else 0)
            frame = SimpleNamespace(
                source_timestamp_ns=stamp,
                attributes={"camera_info": {"depth_scale_m": 0.001}, "depth_source": "head hardware"},
            )
            return SimpleNamespace(
                snapshot=SimpleNamespace(frames={stream: frame}), arrays={stream: np.ones((8, 8), np.uint16)}
            )

    monkeypatch.setattr("gear_sonic.experiments.snapshots.SensorGatewayClient", Client)
    profile = SimpleNamespace(
        components={"experiment": exp}, endpoint_uri=lambda _: "inproc://fake", component=lambda _: {}
    )
    names = ["ego_view", "chest_view", "left_wrist", "right_wrist"]
    camera = dict(
        images={n: f"exact first JPEG {n}".encode() for n in names},
        timestamps={n: 1.0 for n in names},
        camera_info={},
    )
    recorder = FirstObservationRecorder(profile)
    recorder.capture((1, 2), camera, {"base_quat": [1.0, 0, 0, 0]})
    recorder.capture((1, 2), dict(camera, images={n: b"later" for n in names}), {})
    recorder.close()
    events = read_events(recorder.recorder.path)
    assert len([e for e in events if e["type"] == "vla_first_inference"]) == 1
    event = next(e for e in events if e["type"] == "vla_snapshot")
    from pathlib import Path

    metadata = Path(event["path"])
    saved = json.loads(metadata.read_text())
    assert event["rgb_count"] == 4 and event["depth_count"] == 1 and event["pose_valid"] is False
    for name in names:
        assert (metadata.parent / saved["cameras"][name]["rgb"]).read_bytes() == camera["images"][name]
    assert "6.000 ms" in saved["cameras"]["chest_view"]["depth_missing_reason"]
    assert saved["odometry_missing_reason"] == "pose unavailable"
    assert all(r.anchor_timestamp_ns == 1_000_000_000 for r in requests)


def test_offline_role_masks_use_saved_image_and_keep_geometry_unknown(monkeypatch, tmp_path):
    from gear_sonic.experiments import offline
    from gear_sonic.tests.test_lavira_agent import alignment_grounding
    from gear_sonic.utils.inference.base_pose.servo import TrackedInstance

    mask = np.zeros((8, 8), bool)
    mask[1:4, 2:6] = True
    instance = TrackedInstance(9, 0, 0.9, (2.0, 1.0, 6.0, 4.0), mask)
    image = np.ones((8, 8, 3), np.uint8)

    class Tracker:
        def track(self, rgb):
            assert rgb is image
            return [instance]

    monkeypatch.setattr(offline, "tracker_for", lambda *a: Tracker())
    result = offline.segment_roles(
        None, image, alignment_grounding(target="box", yaw_align_target="box"), tmp_path
    )
    assert result["roles_detected"] and result["detection_geometry_available"] is None
    assert result["geometry_missing_reason"] == "no_synchronized_depth"
    for key in ("target", "yaw_align_target"):
        assert np.array_equal(cv2.imread(str(tmp_path / result[key]["mask"]), 0) > 0, mask)
        assert result[key]["track_id"] == 9


def test_offline_visual_client_reuses_local_va_key_without_la(monkeypatch, tmp_path):
    from gear_sonic.experiments import offline
    from gear_sonic.runtime.profile import load_runtime_profile

    (tmp_path / ".env.local").write_text('export LAVIRA_VA_API_KEY="fake-key"\n')
    monkeypatch.setattr(offline, "ROOT", tmp_path / "gear_sonic")
    monkeypatch.delenv("LAVIRA_VA_API_KEY", raising=False)
    captured = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(offline, "LaViRAClient", factory)
    client = offline.visual_client(load_runtime_profile())
    assert captured["va_api_key"] == "fake-key" and captured["la_client"] is not None
    assert client.save_request_context is False


def log_events(tmp_path, task="T1", successes=(True,), *, method="full_vln", completions=None):
    exp = experiment(tmp_path, method)
    exp["task_id"] = task
    recorder = Recorder(SimpleNamespace(components={"experiment": exp}))
    events = [recorder.write("session", experiment=exp)]
    for i, success in enumerate(successes):
        gen = i + 1
        start = recorder.write("trial_start", generation=gen)
        events.append(start)
        events.append(recorder.write("runtime", generation=gen, code="SKILL_STARTED", skill="MANIPULATE"))
        events.append(recorder.write("vla_first_action", generation=gen, skill_id=1))
        events.append(
            recorder.write("trial_end", generation=gen, state="completion_candidate", reason="VA_result")
        )
        if success is not None:
            values = dict(success=success, progress=2 if success else 0)
            if success and (completions is None or completions[i] is not None):
                values["completion_time_s"] = 40.0 if completions is None else completions[i]
            events.append(recorder.write("annotation", generation=gen, values=values))
    return events


def test_known_four_task_macro_failure_time_and_pending_labels(tmp_path):
    logs = [
        log_events(tmp_path / f"T{i}", task=f"T{i}", successes=outcomes)
        for i, outcomes in enumerate([(True,), (False, False, False), (True, False), (True, True)], 1)
    ]
    report = summarize(logs)
    method = report["methods"]["full_vln"]
    assert method["mean_sr"] == pytest.approx(0.625)  # Not pooled 5/8.
    assert method["mean_progress"] == pytest.approx(0.625)
    assert method["mean_time_s"] == pytest.approx((40 + 600 + 320 + 40) / 4)
    assert method["tasks"]["T2"]["conditional_manipulation_sr"]["k"] == 0
    assert method["tasks"]["T2"]["conditional_manipulation_sr"]["n"] == 3
    export(report, tmp_path / "tables")
    assert "62.5%" in (tmp_path / "tables/tables.md").read_text()
    pending = summarize([log_events(tmp_path / "pending", successes=(True, None), completions=(None, None))])
    t = pending["methods"]["full_vln"]["tasks"]["T1"]
    assert t["sr"]["value"] == 1 and t["sr"]["missing"] == 1 and t["time_s"] is None
    assert pending["methods"]["full_vln"]["mean_sr"] is None


def test_actual_action_required_late_actions_and_interventions_do_not_inflate_sr(tmp_path):
    events = log_events(tmp_path, successes=(True,))
    action = next(e for e in events if e["type"] == "vla_first_action")
    end = next(e for e in events if e["type"] == "trial_end")
    action["monotonic_ns"] = end["monotonic_ns"] + 1
    row = trial_rows(events)[0]
    assert not row["vla_started"] and row["handoff_entered"]
    trial = next(e for e in events if e["type"] == "trial_start")
    events.append(
        dict(type="intervention", trial_id=trial["trial_id"], monotonic_ns=trial["monotonic_ns"] + 10_000_000_000)
    )
    row = trial_rows(events)[0]
    assert row["physical_success"] is True and row["success"] is False and row["time_s"] == 600
    events[-1]["monotonic_ns"] = trial["monotonic_ns"] + 41_000_000_000
    assert trial_rows(events)[0]["success"] is True
    end["reason"] = "operator_closed_log_offline"
    row = trial_rows(events)[0]
    assert row["actual_runtime_s"] is None and not row["complete_log"]


def test_gate_truth_denominators_are_independent_and_ignore_late_decisions(tmp_path):
    exp = experiment(tmp_path, "gate_nav_off")
    recorder = Recorder(SimpleNamespace(components={"experiment": exp}))
    recorder.write("session", experiment=exp)
    recorder.write("trial_start", generation=1)
    known = recorder.write("gate", generation=1, gate="nav", applied="block")
    recorder.write("gate", generation=1, gate="nav", applied="allow")  # Missing truth.
    recorder.write("gate", generation=1, gate="align", applied="block")  # Other gate not under test.
    recorder.write("gate_label", gate_event_id=known["event_id"], truth=True)
    recorder.write("trial_end", generation=1, state="failed", reason="total_timeout")
    recorder.write("gate", generation=1, gate="nav", applied="allow")  # Late reply.
    metric = summarize([read_events(recorder.path)])["gates"]["gate_nav_off:nav"]
    assert metric["false_block"]["k"] == 1 and metric["false_block"]["n"] == 1
    assert metric["false_block"]["missing"] == 1 and metric["unlabeled"] == 1
    assert metric["false_allow"]["n"] == 0 and metric["events"] == 2


def test_semantic_labels_and_cli_validation(tmp_path):
    exp = experiment(tmp_path, "semantic_roles")
    recorder = Recorder(SimpleNamespace(components={"experiment": exp}))
    recorder.write("session", experiment=exp)
    for i in range(1, 5):
        recorder.write("semantic_result", sample_id=f"s{i}", task_id=f"T{i}")
        main(
            [
                "semantic",
                str(recorder.path),
                "--sample",
                f"s{i}",
                "--position",
                "true",
                "--yaw",
                "true",
                "--joint",
                "true",
                "--usable",
                "false" if i == 4 else "true",
            ]
        )
    report = summarize([read_events(recorder.path)])
    assert report["semantic_macro"]["joint"] == 1 and report["semantic_macro"]["usable"] == 0.75
    with pytest.raises(SystemExit):
        main(
            [
                "semantic",
                str(recorder.path),
                "--sample",
                "s1",
                "--position",
                "false",
                "--yaw",
                "true",
                "--joint",
                "true",
                "--usable",
                "true",
            ]
        )
    assert rate([])["value"] is None and rate([None])["missing"] == 1


def test_partial_semantic_labels_preserve_unknown_geometry(tmp_path):
    exp = experiment(tmp_path, "semantic_roles")
    recorder = Recorder(SimpleNamespace(components={"experiment": exp}))
    recorder.write("session", experiment=exp)
    recorder.write("semantic_result", sample_id="partial", task_id="T1")
    main(["semantic", str(recorder.path), "--sample", "partial", "--position", "true"])
    main(["semantic", str(recorder.path), "--sample", "partial", "--yaw", "true", "--joint", "true"])
    metrics = summarize([read_events(recorder.path)])["semantic"]["T1"]
    assert metrics["position"]["value"] == 1 and metrics["joint"]["value"] == 1
    assert metrics["usable"]["value"] is None and metrics["usable"]["missing"] == 1


def test_blocked_old_agent_does_not_delay_new_generation(monkeypatch, tmp_path):
    from gear_sonic.utils.inference.lavira.service import (
        LaviraPlannerRuntime,
        load_lavira_config,
        run_agent_worker,
    )

    runtime = LaviraPlannerRuntime(load_lavira_config(), submit_intent=lambda *a: None)
    runtime.experiment = {"enabled": True}
    entered, release, next_started = threading.Event(), threading.Event(), threading.Event()

    def factory():
        class Agent:
            camera = SimpleNamespace(close=lambda: None)

            def run(self, generation):
                if generation == 1:
                    entered.set()
                    release.wait(2.0)
                else:
                    next_started.set()
                return SimpleNamespace(state="reached", reason="test")

        return Agent()

    worker = threading.Thread(target=run_agent_worker, args=(factory, runtime), daemon=True)
    worker.start()
    try:
        runtime.start_navigation(1)
        assert entered.wait(1.0)
        runtime.cancel(2, "cancel")
        runtime.start_navigation(3)
        assert next_started.wait(1.0)
        release.set()
        deadline = time.monotonic() + 1
        while runtime.results.empty() and time.monotonic() < deadline:
            time.sleep(0.001)
        assert runtime.results.get(timeout=1).generation == 3
    finally:
        release.set()
        runtime.shutdown()
        worker.join(1.0)


def test_late_policy_worker_exception_cannot_fail_new_trial():
    from gear_sonic.utils.inference.vla.service import _inference_worker_loop

    requests, results = queue.Queue(), queue.Queue()
    requests.put(1)
    stop, busy = threading.Event(), threading.Event()
    failures = []

    def fail(_):
        stop.set()
        raise RuntimeError("old response")

    _inference_worker_loop(
        requests,
        results,
        stop,
        busy,
        lambda: ({}, {}),
        fail,
        failure_callback=failures.append,
        generation_is_current=lambda epoch: epoch == 2,
    )
    assert not failures and results.empty()


@pytest.mark.parametrize("cpp_mode", ["POSE", "PLANNER"])
def test_success_cancellation_returns_to_planner_without_stopping_cpp(cpp_mode):
    from gear_sonic.runtime.gateway.control import ControlGatewayCore
    from gear_sonic.utils.inference.vla.runtime import _VlaCommandHandler, _VlaRuntimeState

    state = _VlaRuntimeState(
        cpp_loop_running=True, cpp_mode=cpp_mode, pause_loop=False,
        task_generation=1, task_skill_id=3, task_active=True,
    )
    handler = object.__new__(_VlaCommandHandler)
    handler.state = state
    invalidations, cpp_commands = [], []
    handler.invalidate_inference = invalidations.append

    def send_cpp_control_command(**parameters):
        cpp_commands.append(parameters)
        state.cpp_loop_running = parameters["start"]
        state.cpp_mode = "PLANNER" if parameters["planner"] else "POSE"

    handler.send_cpp_control_command = send_cpp_control_command
    command = ControlGatewayCore().accept_command(
        "cancel_navigation", parameters=dict(generation=2, reason="operator_success"),
    ).command

    handler._handle_task_command(command)

    assert state.cpp_loop_running and state.cpp_mode == "PLANNER"
    assert not state.task_active and state.pause_loop
    assert state.task_generation == 2 and invalidations
    assert cpp_commands == ([dict(start=True, planner=True)] if cpp_mode == "POSE" else [])
    late_start = ControlGatewayCore().accept_command(
        "start_vla_task", parameters=dict(generation=1, skill_id=3),
    ).command
    handler._handle_task_command(late_start)
    assert not state.task_active and state.cpp_mode == "PLANNER"


def test_vla_runtime_start_resume_and_restart_keep_trained_prompt(monkeypatch):
    from gear_sonic.utils.inference.vla import runtime

    state = runtime._VlaRuntimeState()
    prompts = ["initial"]

    def noop(*a, **k):
        pass

    monkeypatch.setattr(runtime, "_current_vla_safety_reason", lambda *a: "clear")
    handler = runtime._VlaCommandHandler(
        state,
        control_listener=None,
        language_prompt_ref=prompts,
        inference_failures=queue.Queue(),
        inference_failed_event=threading.Event(),
        vla_safety_gate=None,
        vla_safety_monitor=None,
        task_status_intent=None,
        policy=SimpleNamespace(ping=lambda **kw: True),
        record_event=noop,
        invalidate_inference=noop,
        publish_initial_pose=lambda: True,
        send_cpp_control_command=lambda **kw: True,
        activate_vla_metrics=noop,
        fail_active_task=noop,
        publish_task_status=noop,
        trained_prompt="TRAINED preserved string",
    )

    def command(name, generation, **p):
        handler._handle_task_command(
            SimpleNamespace(
                name=name,
                parameters=dict(
                    generation=generation,
                    skill_id=1,
                    handoff_context="SEMANTIC must not enter policy",
                    task="SEMANTIC",
                    **p,
                ),
            )
        )

    command("start_vla_task", 1)
    assert prompts[0] == "TRAINED preserved string"
    command("resume_vla_task", 1, window_id=1)
    assert prompts[0] == "TRAINED preserved string"
    command("cancel_navigation", 2)
    command("start_vla_task", 3)
    assert prompts[0] == "TRAINED preserved string" and state.task_generation == 3
