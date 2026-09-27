"""Experiment configuration and launch checks without robot/model processes."""

from types import SimpleNamespace

import pytest
import yaml

from gear_sonic.experiments.config import CONFIGS, launch, read_yaml, resolve
from gear_sonic.experiments.recording import read_events
from gear_sonic.runtime.profile import load_runtime_profile
from gear_sonic.scripts import launch_inference
from gear_sonic.utils.inference.iplanner.config import IPlannerServerConfig, navigation_backend
from gear_sonic.utils.inference.iplanner.control import load_navigation_config


def custom_spec(tmp_path):
    spec = read_yaml(CONFIGS / "baselinelavira.yaml")
    for name in ("runtime_profile", "tasks_file", "cases_file"):
        spec[name] = str((CONFIGS / spec[name]).resolve())
    spec["runtime_overlays"] = [str((CONFIGS / p).resolve()) for p in spec["runtime_overlays"]]
    return spec


@pytest.mark.parametrize("task", ["T1", "T2", "T3", "T4"])
@pytest.mark.parametrize("case", ["visible_01", "invisible_01"])
@pytest.mark.parametrize("mission", [None, "gear_sonic/config/missions/bag_then_bin.yaml"])
def test_baseline_reads_the_same_task_case_and_shared_components_as_full_vln(task, case, mission):
    baseline, missing = resolve(CONFIGS / "baselinelavira.yaml", task, case, mission_path=mission)
    full, expected_missing = resolve(CONFIGS / "full_vln.yaml", task, case, mission_path=mission)
    assert missing == expected_missing
    assert baseline["endpoints"] == full["endpoints"]
    assert baseline["ros_topics"] == full["ros_topics"]
    for name, component in full["components"].items():
        if name != "experiment":
            assert baseline["components"][name] == component
    exp = baseline["components"]["experiment"]
    for name in ("task", "condition", "gates", "alignment", "entry_stage", "navigation_backend"):
        assert exp[name] == full["components"]["experiment"][name]
    assert exp["method"] == "baselinelavira"
    if mission:
        assert exp["flexible_mission"] and exp["skills"] == full["components"]["experiment"]["skills"]
        assert baseline["components"]["lavira"]["mission"] == exp["mission"]
    else:
        assert baseline["components"]["lavira"]["mission"] == (
            exp["condition"]["navigation_instructions"][task] or "REQUIRED"
        )
    assert exp["models"]["navigation_policy"] == "iplanner"
    assert "navdp_checkpoint" not in exp["models"]
    assert baseline["profile"] == f"icra_baselinelavira_{task}_{case}"


def test_relative_overlays_then_experiment_overrides_apply_before_task_prompts(tmp_path):
    spec = custom_spec(tmp_path)
    overlay = tmp_path / "local.yaml"
    overlay.write_text(yaml.safe_dump({"components": {
        "iplanner_navigation": {"target_speed": 0.25},
        "lavira": {"mission": "must be replaced by the selected case"},
    }}))
    spec["runtime_overlays"].append(overlay.name)
    spec["runtime_overrides"] = {"components": {"iplanner_navigation": {"target_speed": 0.28}}}
    path = tmp_path / "method.yaml"
    path.write_text(yaml.safe_dump(spec))
    runtime, _ = resolve(path, "T1", "invisible_01")
    assert runtime["components"]["iplanner_navigation"]["target_speed"] == 0.28
    assert runtime["components"]["lavira"]["mission"] == (
        read_yaml(CONFIGS / "cases_main.yaml")["cases"]["invisible_01"]["navigation_instructions"]["T1"] or "REQUIRED"
    )


def test_missing_weights_are_reported_and_block_before_results_or_processes(tmp_path, monkeypatch, capsys):
    missing = tmp_path / "absent.pth"
    monkeypatch.setattr(IPlannerServerConfig, "required_assets", lambda self: ((missing, "iPlanner checkpoint"),))
    monkeypatch.setattr("gear_sonic.experiments.config.subprocess.run",
                        lambda *a, **k: pytest.fail("preflight invoked a process"))
    monkeypatch.setattr(launch_inference, "main", lambda *a: pytest.fail("preflight launched the robot"))
    args = ["--task", "T1", "--case", "visible_01", "--output", str(tmp_path / "runs")]
    assert launch("baselinelavira", [*args, "--dry-run"]) == 0
    output = yaml.safe_load(capsys.readouterr().out)
    assert not output["ready"]
    assert f"iPlanner checkpoint: {missing}" in output["missing"]
    with pytest.raises(SystemExit) as error:
        launch("baselinelavira", args)
    assert error.value.code == 2
    assert not (tmp_path / "runs").exists()


def test_launch_freezes_baseline_profile_and_model_metadata(tmp_path, monkeypatch):
    spec = custom_spec(tmp_path)
    root = tmp_path / "source" / "iplanner"
    root.mkdir(parents=True)
    # Placeholder assets only exercise preflight existence checks. The real
    # launcher is replaced below; no weights or model code are executed.
    for path in (root / "iplanner_server.py", root / "weights.pth", root / "model.yaml"):
        path.touch()
    spec["runtime_overrides"] = {"components": {"iplanner": {
        "root": str(root), "checkpoint": "weights.pth", "model_config": "model.yaml",
    }}}
    config = tmp_path / "baseline.yaml"
    config.write_text(yaml.safe_dump(spec))
    calls = []
    monkeypatch.setattr(launch_inference, "main", calls.append)
    monkeypatch.setattr("gear_sonic.experiments.config.subprocess.run",
                        lambda args, **kw: SimpleNamespace(
                            stdout="test-commit" if args[:2] == ["git", "rev-parse"] else "",
                            returncode=0,
                        ))
    output = tmp_path / "runs"
    assert launch("baselinelavira", ["--config", str(config), "--task", "T1", "--case", "visible_01",
                                    "--output", str(output)]) == 0
    assert len(calls) == 1
    snapshots = list(output.glob("baselinelavira_T1_visible_01_*/runtime.yaml"))
    assert len(snapshots) == 1
    profile = load_runtime_profile(snapshots[0])
    assert navigation_backend(profile) == "iplanner"
    assert load_navigation_config(profile).lookahead_dist == 0.5
    exp = profile.components["experiment"]
    assert exp["models"]["iplanner_checkpoint"] == str(root / "weights.pth")
    assert profile.components["vla"]["prompt"] == exp["task"]["vla_trained_prompt"]
    assert profile.components["lavira"]["manipulation_prompt"] == exp["task"]["vla_prompt"]
    command = launch_inference.build_navdp_server_command(calls[0])
    assert "gear_sonic.utils.inference.iplanner.server" in command
    assert str(root / "weights.pth") in command
    assert "eval.src.policy_server" not in command
    worker = launch_inference.build_navdp_planner_command(calls[0], CONFIGS.parents[2])
    assert str(snapshots[0]) in worker
    session = read_events(snapshots[0].parent / "events.jsonl")[0]
    assert session["experiment"]["method"] == "baselinelavira"
    assert session["experiment"]["config_sha256"] == exp["config_sha256"]


def test_invalid_navigation_speed_is_rejected_in_dry_run(tmp_path):
    spec = custom_spec(tmp_path)
    spec["runtime_overrides"] = {"components": {"iplanner_navigation": {"target_speed": -0.1}}}
    config = tmp_path / "baseline.yaml"
    config.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError, match="target_speed"):
        launch("baselinelavira", ["--config", str(config), "--dry-run"])
