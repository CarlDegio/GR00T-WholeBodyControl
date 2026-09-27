"""Exercise experiment methods through the real LA, gateway and VLA loop."""

from types import SimpleNamespace

import pytest
import yaml

from gear_sonic.experiments.agent import RecordedCamera, RecordedVisualClient
from gear_sonic.experiments.config import CONFIGS, launch, resolve
from gear_sonic.experiments.flexible import FlexibleExperimentAgent
from gear_sonic.experiments.recording import Recorder, read_events
from gear_sonic.experiments.results import trial_rows
from gear_sonic.experiments.supervisor import ExperimentSupervisor
from gear_sonic.tests import test_flexible_mission as helpers
from gear_sonic.utils.inference.lavira import service

METHODS = ("full_vln", "nav_direct_vla", "geometric_vla", "baselinelavira", "unilm_nav")
MISSION = "gear_sonic/config/missions/bag_then_bin.yaml"


def harness(monkeypatch, tmp_path, method, actions):
    monkeypatch.setattr(helpers, "FlexibleMissionAgent", FlexibleExperimentAgent)
    h = helpers.flexible_harness(monkeypatch, actions)
    raw, missing = resolve(CONFIGS / f"{method}.yaml", mission_path=MISSION)
    assert not missing
    exp = raw["components"]["experiment"]
    exp.update(run_dir=str(tmp_path), session_id="flex-test")
    agent = h.agent
    agent.experiment = exp
    agent.recorder = Recorder(SimpleNamespace(components={"experiment": exp}))
    agent.recorder.write("session", experiment=exp)
    agent.camera = RecordedCamera(h.camera)
    agent.client = RecordedVisualClient(h.client, agent)
    agent.navila = None
    agent.head_only = False
    agent.skills["bag"]["geometry_target"] = "white bag"
    agent.skills["bin"]["geometry_target"] = "green bin"
    h.gateway.profile = SimpleNamespace(components={"experiment": exp})
    h.gateway.experiment = ExperimentSupervisor(h.gateway, clock=lambda: h.now[0])
    monkeypatch.setattr("gear_sonic.experiments.snapshots.record_pose_async", lambda *a, **k: None)
    h.gateway.experiment.start(1)
    report = agent.report_event

    def record(level, code, message, **fields):
        report(level, code, message, **fields)
        agent.recorder.runtime("lavira", code, **fields)

    agent.report_event = record
    start = agent._start_manipulation

    def started(generation, skill_id):
        start(generation, skill_id)
        agent.recorder.write("vla_first_action", generation=generation, skill_id=skill_id)
        h.gateway.experiment.first_action(dict(generation=generation, skill_id=skill_id))

    agent._start_manipulation = started
    return h


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("between", [[], [("MOVE_TO", "trash can")]])
def test_two_operations_keep_la_order_models_prompts_and_trial(monkeypatch, tmp_path, method, between):
    actions = [("ALIGN", "bag"), ("MANIPULATE", "bag"), *between,
               ("ALIGN", "bin"), ("MANIPULATE", "bin"), ("COMPLETE", "bin")]
    h = harness(monkeypatch, tmp_path, method, actions)
    result = h.agent.run(1)
    assert result.reason == "mission_completed", result
    assert h.selected == ["pill", "rubbish"]
    assert h.actual_prompts == ["collect the pill bottle", "trained pedal prompt "]
    assert len(h.client.la_calls) == len(actions)
    assert all(call["mission"] == h.agent.mission for call in h.client.la_calls)
    assert [call["mission"] for call in h.client.postcheck_calls if call.get("skill") == "MANIPULATE"] == [
        "grasp and hold bag", "step on pedal",
    ]
    base_starts = [p for n, p in h.intents if n == "start_base_pose"]
    assert len(base_starts) == (0 if method == "nav_direct_vla" else 2)
    if method == "geometric_vla":
        assert [p["target"] for p in base_starts] == ["white bag", "green bin"]
    if method not in {"full_vln", "baselinelavira"}:
        assert not h.client.alignment_grounding_calls
    if method in {"nav_direct_vla", "unilm_nav"}:
        assert not any(call.get("skill") == "ALIGN" for call in h.client.postcheck_calls)
    if method == "unilm_nav":
        assert [p["task"] for p in base_starts] == ["grasp and hold bag", "step on pedal"]
        assert all(p["operation_id"] for p in base_starts)
        assert base_starts[0]["observation_source"] == "current"
        assert base_starts[1]["observation_source"] == ("navigation" if between else "current")
    events = read_events(h.agent.recorder.path)
    operations = [e for e in events if e["type"] == "vla_operation"]
    assert [e["logical_skill"] for e in operations] == ["bag", "bin"]
    assert len({e["alignment_skill_id"] for e in operations}) == 2
    assert h.gateway.experiment.active and h.gateway.experiment.vla_deadline is None
    assert not any(e["type"] in {"trial_end", "annotation"} for e in events)
    h.gateway.experiment.finish(1, result.state, result.reason)
    events = read_events(h.agent.recorder.path)
    assert len([e for e in events if e["type"] == "trial_end"]) == 1
    row = trial_rows(events)[0]
    assert row["vla_activation_count"] == 2 and row["align_retries"] == row["nav_retries"] == 0
    assert row["align_attempts"] == (0 if method == "nav_direct_vla" else 2)
    assert row["success"] is None and row["completion_time_s"] is None


@pytest.mark.parametrize("reason", ["geometric_timeout", "geometric_target_lost"])
def test_geometric_fallback_keeps_evidence_and_runs_both_skills(monkeypatch, tmp_path, reason):
    h = harness(monkeypatch, tmp_path, "geometric_vla", [
        ("ALIGN", "bag"), ("MANIPULATE", "bag"), ("ALIGN", "bin"),
        ("MANIPULATE", "bin"), ("COMPLETE", "bin"),
    ])
    wait = h.agent.wait_status

    def fallback(generation, skill_id, segment_id, timeout):
        if h.intents[-1][0] != "start_base_pose":
            return wait(generation, skill_id, segment_id, timeout)
        status = dict(generation=generation, skill_id=skill_id, segment_id=segment_id,
                      state="failed", reason=reason)
        assert h.gateway.navigation.accept_status(status, owner="base_pose", agent_final=False)
        return status

    h.agent.wait_status = fallback
    h.client.postchecks = iter([helpers.postcheck("NOT_SATISFIED")] * 4)
    result = h.agent.run(1)
    assert result.reason == "mission_completed", result
    assert h.selected == ["pill", "rubbish"]
    gates = [e for e in read_events(h.agent.recorder.path) if e["type"] == "gate" and e["gate"] == "align"]
    assert len(gates) == 2
    assert all(e["applied"] == "allow" and not e["controller_aligned"] and not e["va_recommendation"] for e in gates)


@pytest.mark.parametrize("method", METHODS)
def test_operation_timeout_continues_but_trial_total_budget_remains(monkeypatch, tmp_path, method):
    h = harness(monkeypatch, tmp_path, method, [
        ("ALIGN", "bag"), ("MANIPULATE", "bag"), ("MOVE_TO", "trash can"),
        ("ALIGN", "bin"), ("MANIPULATE", "bin"), ("COMPLETE", "bin"),
    ])
    h.first_verdict = lambda: helpers.postcheck("NOT_SATISFIED")
    result = h.agent.run(1)
    assert result.reason == "mission_completed_with_timeouts", result
    assert h.selected == ["pill", "rubbish"]
    assert h.gateway.experiment.vla_deadline is None
    aborts = []
    h.gateway.experiment.abort = lambda *args: aborts.append(args)
    h.now[0] = h.gateway.experiment.active[1]
    h.gateway.experiment.tick()
    assert aborts == [("failed", "total_timeout")]


@pytest.mark.parametrize("state,reason", [("failed", "unilm_timeout"), ("stopped", "operator_cancelled"),
                                        ("failed", "aligned"), ("reached", "invalid_pose")])
def test_unilm_requires_real_alignment_success_before_vla(monkeypatch, tmp_path, state, reason):
    h = harness(monkeypatch, tmp_path, "unilm_nav", [("ALIGN", "bag"), ("MANIPULATE", "bag")])
    wait = h.agent.wait_status

    def failed(generation, skill_id, segment_id, timeout):
        if h.intents[-1][0] != "start_base_pose":
            return wait(generation, skill_id, segment_id, timeout)
        status = dict(generation=generation, skill_id=skill_id, segment_id=segment_id,
                      state=state, reason=reason)
        assert h.gateway.navigation.accept_status(status, owner="base_pose", agent_final=False)
        return status

    h.agent.wait_status = failed
    result = h.agent.run(1)
    assert result.reason == f"unilm_failed:{reason}"
    assert not h.selected and not any(n == "start_vla_task" for n, _ in h.intents)
    assert h.gateway.navigation.alignment_ticket is None


@pytest.mark.parametrize("method", METHODS)
def test_dry_run_and_service_select_flexible_experiment(monkeypatch, tmp_path, capsys, method):
    assert launch(method, ["--mission", MISSION, "--dry-run", "--output", str(tmp_path / "runs")]) == 0
    assert yaml.safe_load(capsys.readouterr().out)["ready"]
    assert not (tmp_path / "runs").exists()
    raw, _ = resolve(CONFIGS / f"{method}.yaml", mission_path=MISSION)
    path = tmp_path / "runtime.yaml"
    path.write_text(yaml.safe_dump(raw))
    h = helpers.mission_harness(monkeypatch)
    monkeypatch.setattr(service, "SensorGatewayRGBDCamera", lambda *a, **k: h.camera)
    monkeypatch.setattr(service, "LaViRAClient", lambda **k: h.client)
    agent = service._agent(service.load_lavira_config(str(path)), "unused", h.mailbox)
    assert isinstance(agent, FlexibleExperimentAgent)
    assert agent.experiment["method"] == method
    assert agent.skills["step_bin"]["vla_trained_prompt"].endswith(". ")


@pytest.mark.parametrize("method", METHODS)
def test_new_navigation_invalidates_previous_operation_preparation(monkeypatch, tmp_path, method):
    h = harness(monkeypatch, tmp_path, method, [
        ("ALIGN", "bag"), ("MOVE_TO", "table"), ("MANIPULATE", "bag"),
    ])
    result = h.agent.run(1)
    assert result.state == "failed"
    assert not h.selected


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("failure", ["cancelled", "expired", "safety"])
def test_baselines_preserve_cancellation_freshness_and_safety(monkeypatch, tmp_path, method, failure):
    h = harness(monkeypatch, tmp_path, method, [
        ("ALIGN", "bag"), ("MANIPULATE", "bag"), ("COMPLETE", "bag"),
    ])
    original = h.client.language_action

    def interrupt(**kwargs):
        result = original(**kwargs)
        if result["skill"] == "MANIPULATE":
            if failure == "cancelled":
                h.agent.cancelled = lambda generation: True
            elif failure == "expired":
                h.now[0] += h.agent.operation_alignment_max_age_seconds + 1
            else:
                h.agent.readiness = lambda *args: (False, "safety_blocked")
        return result

    h.client.language_action = interrupt
    result = h.agent.run(1)
    assert result.state in {"failed", "stopped"}, result
    assert not h.selected


@pytest.mark.parametrize("method", METHODS)
def test_repeated_same_skill_needs_fresh_preparation(monkeypatch, tmp_path, method):
    h = harness(monkeypatch, tmp_path, method, [
        ("ALIGN", "bag"), ("MANIPULATE", "bag"), ("ALIGN", "bag"),
        ("MANIPULATE", "bag"), ("COMPLETE", "bag"),
    ])
    result = h.agent.run(1)
    assert result.reason == "mission_completed", result
    operations = [e for e in read_events(h.agent.recorder.path) if e["type"] == "vla_operation"]
    assert len({e["operation_id"] for e in operations}) == 2
    assert h.selected == ["pill", "pill"]


def test_geometric_requires_per_skill_target_and_rejects_smoke(tmp_path):
    from pathlib import Path
    from gear_sonic.runtime.profile import load_runtime_profile
    from gear_sonic.utils.inference.lavira.mission import validate_mission_profile

    overlay = yaml.safe_load(Path(MISSION).read_text())
    del overlay["components"]["lavira"]["skills"]["step_bin"]["geometry_target"]
    path = tmp_path / "mission.yaml"
    path.write_text(yaml.safe_dump(overlay))
    raw, _ = resolve(CONFIGS / "geometric_vla.yaml", mission_path=path)
    profile = tmp_path / "profile.yaml"
    profile.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="geometry_target"):
        validate_mission_profile(load_runtime_profile(profile))
    raw, _ = resolve(CONFIGS / "full_vln.yaml", mission_path=MISSION)
    raw["components"]["lavira"]["smoke_assume_vla_success"] = True
    profile.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="smoke"):
        validate_mission_profile(load_runtime_profile(profile))


@pytest.mark.parametrize("mode", ["dual", "none", "geometric", "unilm_nav"])
def test_real_la_request_describes_selected_alignment_method(tmp_path, mode):
    import json
    from gear_sonic.utils.inference.lavira.agent import LaViRAClient

    requests = []
    value = helpers.decision("ALIGN")
    value.update(skill_args={"skill_key": "bag", "task": "pick bag"}, updated_todo_list="- [ ] pick bag")

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(value)))])

    cloud = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client = LaViRAClient(la_client=cloud, va_client=cloud, request_context_dir=tmp_path,
                         la_base_url="https://example.invalid", va_base_url="https://example.invalid")
    result = client.language_action(
        mission="whole mission", navigation_mode="vln", global_target=None, current_step=1,
        todo_list="", scan_views=(), move_to_views=(), available_skills={"bag": {"description": "pick bag"}},
        alignment_mode=mode,
    )
    assert result["skill_args"]["skill_key"] == "bag" and len(requests) == 1
    content = json.dumps(requests[0]["messages"])
    assert {"dual": "BasePose executes alignment", "none": "skips all physical alignment",
            "geometric": "simple geometric approach/centering", "unilm_nav": "VLM-driven pose alignment"}[mode] in content


@pytest.mark.parametrize("method", METHODS)
def test_offline_masks_use_current_operation_not_previous_roles(monkeypatch, tmp_path, method):
    import json
    import cv2
    import numpy as np
    from gear_sonic.experiments import offline
    from gear_sonic.runtime.profile import load_runtime_profile

    raw, _ = resolve(CONFIGS / f"{method}.yaml", mission_path=MISSION)
    raw["components"]["experiment"].update(run_dir=str(tmp_path), session_id="masks")
    path = tmp_path / "runtime.yaml"
    path.write_text(yaml.safe_dump(raw))
    recorder = Recorder(load_runtime_profile(path))
    recorder.write("trial_start", generation=1)
    for skill_id, name in ((2, "bag"), (5, "bin")):
        recorder.write("check", generation=1, skill_id=skill_id - 1, kind="alignment_grounding",
                       result=helpers.alignment_grounding(target=name, yaw_align_target=name))
        # A later unrelated check must never supply this operation's roles.
        recorder.write("check", generation=1, skill_id=99, kind="alignment_grounding",
                       result=helpers.alignment_grounding(target="wrong", yaw_align_target="wrong"))
        recorder.write("vla_operation", generation=1, skill_id=skill_id,
                       alignment_skill_id=skill_id - 1, geometry_target=name, vla_prompt=name)
        recorder.write("vla_first_inference", generation=1, skill_id=skill_id)
        folder = tmp_path / str(skill_id)
        folder.mkdir()
        cv2.imwrite(str(folder / "rgb.jpg"), np.zeros((4, 4, 3), np.uint8))
        metadata = folder / "snapshot.json"
        metadata.write_text(json.dumps({
            "cameras": {stream: {"rgb": "rgb.jpg"} for stream in ("ego_view", "chest_view")},
            "mount_calibration": {},
        }))
        recorder.write("vla_snapshot", generation=1, skill_id=skill_id, path=str(metadata))
    recorder.write("trial_end", generation=1)
    targets = []
    monkeypatch.setattr(offline, "segment_roles", lambda config, image, roles, *args, **kwargs:
                        targets.append(roles["target"]["name"]) or {})
    monkeypatch.setattr(offline, "visual_client", lambda profile: SimpleNamespace(
        alignment_grounding=lambda **kwargs: helpers.alignment_grounding(
            target=kwargs["manipulation_prompt"], yaw_align_target=kwargs["manipulation_prompt"])))
    # Direct VLA has no control grounding; require independent offline selection.
    if method == "nav_direct_vla":
        events = [e for e in read_events(recorder.path) if e["type"] != "check"]
        recorder.path.write_text("".join(json.dumps(e) + "\n" for e in events))
    offline.enrich_snapshots(path)
    assert targets == ["bag", "bag", "bin", "bin"]
    assert len([e for e in read_events(recorder.path) if e["type"] == "mask_enrichment"]) == 2
