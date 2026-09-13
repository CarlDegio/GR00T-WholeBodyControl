"""NaVILA uses configured roles and never constructs or calls LA/VA clients."""

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from gear_sonic.experiments import config, offline
from gear_sonic.experiments.agent import DisabledLanguageVisualClient
from gear_sonic.experiments.recording import Recorder, read_events
from gear_sonic.tests.test_experiments_runtime import experiment
from gear_sonic.utils.inference.lavira.agent import LaViRAAgentError
from gear_sonic.utils.inference.lavira import service


def test_missing_case_targets_block_only_navila_launch(monkeypatch):
    read = config.read_yaml

    def without_targets(path):
        data = read(path)
        if path.name == "cases_main.yaml":
            data["cases"]["visible_01"]["navila_alignment"] = dict(target=None, yaw_align_target=None)
        return data

    monkeypatch.setattr(config, "read_yaml", without_targets)
    payload, missing = config.resolve(config.CONFIGS / "navila_basepose_vla.yaml")
    assert "visible_01.navila_alignment.target" in missing
    assert "visible_01.navila_alignment.yaw_align_target" in missing
    assert not any(payload["components"]["experiment"]["gates"].values())
    _, other_missing = config.resolve(config.CONFIGS / "full_vln.yaml")
    assert not any("navila_alignment" in name for name in other_missing)


def test_selected_case_targets_are_compiled_without_changing_other_cases(monkeypatch):
    read = config.read_yaml

    def with_targets(path):
        data = read(path)
        if path.name == "cases_main.yaml":
            data["cases"]["visible_01"]["navila_alignment"] = dict(target=None, yaw_align_target=None)
            data["cases"]["visible_05"]["navila_alignment"] = dict(
                target="  blue basket ", yaw_align_target=" desk ",
            )
        return data

    monkeypatch.setattr(config, "read_yaml", with_targets)
    payload, missing = config.resolve(config.CONFIGS / "navila_basepose_vla.yaml", case_id="visible_05")
    assert not missing
    exp = payload["components"]["experiment"]
    assert exp["condition"]["navila_alignment"] == dict(target="blue basket", yaw_align_target="desk")
    assert exp["models"]["la"] is None and exp["models"]["va"] is None
    _, missing = config.resolve(config.CONFIGS / "navila_basepose_vla.yaml", case_id="visible_01")
    assert "visible_01.navila_alignment.target" in missing


@pytest.mark.parametrize("gate", ["nav", "align", "completion"])
def test_navila_rejects_reenabled_visual_gates(monkeypatch, gate):
    read = config.read_yaml

    def with_gate(path):
        data = read(path)
        if path.name == "navila_basepose_vla.yaml":
            data["gates"][gate] = True
        return data

    monkeypatch.setattr(config, "read_yaml", with_gate)
    with pytest.raises(ValueError, match="without LA/VA"):
        config.resolve(config.CONFIGS / "navila_basepose_vla.yaml")


def test_navila_service_needs_no_la_va_client_or_credentials(monkeypatch, tmp_path):
    exp = experiment(tmp_path, "navila_basepose_vla")
    profile = SimpleNamespace(components={"experiment": exp})
    monkeypatch.setattr("gear_sonic.runtime.profile.load_runtime_profile", lambda *a, **k: profile)
    monkeypatch.setattr(service, "SensorGatewayRGBDCamera", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(service, "LaViRAClient", lambda **kwargs: pytest.fail("NaVILA constructed an LA/VA client"))
    for name in ("LAVIRA_LA_API_KEY", "LAVIRA_VA_API_KEY", "DASHSCOPE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    cfg = service.LaviraPlannerConfig(mission="find basket", alignment_prompt="unused", global_target="")
    runtime = service.LaviraPlannerRuntime(cfg, submit_intent=lambda *args: None)
    agent = service._agent(cfg, "inproc://unused", runtime)
    assert isinstance(agent.client.client, DisabledLanguageVisualClient)
    assert agent.navila_targets == dict(target="blue basket", yaw_align_target="desk")
    for method in ("language_action", "grounding", "alignment_grounding", "postcheck"):
        with pytest.raises(LaViRAAgentError, match="disabled"):
            getattr(agent.client.client, method)()
    assert not (tmp_path / "lavira_requests").exists()


@pytest.mark.parametrize("missing", [False, True])
def test_navila_offline_masks_never_fall_back_to_va(monkeypatch, tmp_path, missing):
    exp = experiment(tmp_path, "navila_basepose_vla")
    if missing:
        exp["condition"]["navila_alignment"]["yaw_align_target"] = ""
    profile = SimpleNamespace(components={"experiment": exp})
    recording = Recorder(profile)
    recording.write("trial_start", generation=1)
    recording.write("vla_first_inference", generation=1, skill_id=3)
    metadata = tmp_path / "snapshot.json"
    image_path = tmp_path / "rgb.png"
    assert cv2.imwrite(str(image_path), np.zeros((8, 8, 3), np.uint8))
    metadata.write_text(json.dumps(dict(
        cameras={name: {"rgb": image_path.name} for name in ("ego_view", "chest_view")},
        mount_calibration={},
    )))
    recording.write("vla_snapshot", generation=1, skill_id=3, path=str(metadata))
    recording.write("trial_end", generation=1, state="completion_candidate")
    monkeypatch.setattr(offline, "load_runtime_profile", lambda *args: profile)
    monkeypatch.setattr(offline, "load_base_pose_config", lambda *args: SimpleNamespace())
    monkeypatch.setattr(offline, "visual_client", lambda *args: pytest.fail("NaVILA offline called VA"))
    roles_seen = []

    def segment(_config, _image, roles, *_args, **_kwargs):
        roles_seen.append(roles)
        return {"roles": roles}

    monkeypatch.setattr(offline, "segment_roles", segment)
    offline.enrich_snapshots(tmp_path / "runtime.yaml")
    events = read_events(recording.path)
    if missing:
        assert events[-1]["type"] == "mask_enrichment_failed" and not roles_seen
    else:
        assert events[-1]["type"] == "mask_enrichment"
        assert events[-1]["role_source"] == "case_navila_alignment"
        assert len(roles_seen) == 2
        assert roles_seen[0]["target"]["name"] == "blue basket"
        assert roles_seen[0]["yaw_align_target"]["name"] == "desk"
