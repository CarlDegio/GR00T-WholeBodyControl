"""Named missions retain task identity, trained inputs and historical labels."""

import pytest
import yaml

from gear_sonic.experiments.config import CONFIGS, launch, resolve
from gear_sonic.experiments.recording import Recorder, read_events
from gear_sonic.experiments.results import trial_rows

MISSIONS = CONFIGS.parent / "missions"
CASES = [
    ("T1", "t1_bottle_to_basket.yaml", "collect the pill bottle", 0.9, 0.1),
    ("T2", "t2_bag_and_box.yaml", "collect the bag and put it into box, then carry the box", 1.15, 0.1),
    ("T3", "t3_bowl_to_bin.yaml", "collect the pill bottle", 0.9, 0.08),
]


@pytest.mark.parametrize("task,filename,trained,distance,lateral", CASES)
def test_single_operation_missions_validate_and_keep_recorded_inputs(capsys, task, filename, trained, distance, lateral):
    path = MISSIONS / filename
    raw, missing = resolve(CONFIGS / "full_vln.yaml", "T4", mission_path=path)
    c = raw["components"]
    assert not missing
    assert c["experiment"]["task_id"] == task
    assert c["lavira"]["flexible_mission"] is True
    assert len(c["lavira"]["skills"]) == len(c["vla"]["backends"]) == 1
    skill = next(iter(c["lavira"]["skills"].values()))
    backend = c["vla"]["backends"][skill["backend"]]
    assert skill["vla_trained_prompt"] == backend["prompt"] == trained
    assert skill["vla_prompt"] in c["lavira"]["mission"]
    assert backend["port"] == 29999
    assert skill["timeout_seconds"] == 180 and skill["on_timeout"] == "fail"
    assert c["base_pose"]["raw_chest_target_distance_m"] == distance
    assert c["base_pose"]["raw_lateral_tolerance_m"] == lateral
    assert c["base_pose"]["raw_max_lateral_speed_m_s"] == 0.4
    assert launch("full_vln", ["--mission", str(path), "--demo", "--dry-run"]) == 0
    output = yaml.safe_load(capsys.readouterr().out)
    assert output["ready"] and output["experiment"]["demo"]
    assert output["experiment"]["task_id"] == task


@pytest.mark.parametrize("method", ["full_vln", "nav_direct_vla", "geometric_vla", "baselinelavira", "unilm_nav"])
def test_bag_mission_owns_t4_even_with_old_task_argument(method):
    raw, missing = resolve(CONFIGS / f"{method}.yaml", "T1", mission_path=MISSIONS / "bag_then_bin.yaml")
    assert not missing
    assert raw["components"]["experiment"]["task_id"] == "T4"
    assert raw["profile"].startswith(f"icra_{method}_T4_")
    assert raw["components"]["experiment"]["task"]["milestones"] == ["stable_bag_grasp", "bag_in_bin_released"]


def test_launch_directory_and_frozen_session_use_mission_identity(monkeypatch, tmp_path):
    # Stop before the launcher can contact hardware; inspect its actual output.
    class LaunchReached(Exception):
        pass

    def stop(_):
        raise LaunchReached

    monkeypatch.setattr("gear_sonic.scripts.launch_inference.main", stop)
    with pytest.raises(LaunchReached):
        launch("full_vln", ["--mission", str(MISSIONS / "bag_then_bin.yaml"), "--task", "T1", "--output", str(tmp_path)])
    run, = tmp_path.iterdir()
    assert run.name.startswith("full_vln_T4_")
    assert yaml.safe_load((run / "runtime.yaml").read_text())["components"]["experiment"]["task_id"] == "T4"
    assert read_events(run / "events.jsonl")[0]["experiment"]["task_id"] == "T4"


def test_task_correction_preserves_outcomes_condition_times_and_progress(tmp_path):
    raw, _ = resolve(CONFIGS / "full_vln.yaml", "T1")
    recorder = Recorder(path=tmp_path / "events.jsonl")
    recorder.write("session", experiment=raw["components"]["experiment"])
    recorder.write("trial_start", trial_id="trial")
    recorder.write("annotation", trial_id="trial", values={"success": True, "completion_time_s": 25.0,
                   "progress": 2, "progress_denominator": 3, "condition": {"case_id": "invisible_02"}})
    before, = trial_rows(read_events(recorder.path))
    recorder.write("annotation", trial_id="trial", values={"task_id": "T4"})
    after, = trial_rows(read_events(recorder.path))
    assert after == {**before, "task_id": "T4"}
    recorder.write("annotation", trial_id="trial", values={"task_id": "T2"})
    assert trial_rows(read_events(recorder.path))[0]["task_id"] == "T2"


def test_invalid_mission_identity_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("components: {mission_metadata: {task_id: T5}}\n")
    with pytest.raises(ValueError, match="Invalid mission task_id"):
        resolve(CONFIGS / "full_vln.yaml", mission_path=path)
