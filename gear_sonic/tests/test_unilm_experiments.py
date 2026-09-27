"""UniLM replaces alignment while retaining the shared mission runtime."""

from pathlib import Path

import pytest
import yaml

from gear_sonic.experiments.config import CONFIGS, resolve
from gear_sonic.scripts.launch_inference import build_base_pose_agent_command, load_inference_launch_config


@pytest.mark.parametrize("mission", [None, "gear_sonic/config/missions/bag_then_bin.yaml"])
def test_unilm_preserves_navigation_agent_vla_and_uses_chest_alignment(tmp_path, mission):
    full, missing = resolve(CONFIGS / "full_vln.yaml", mission_path=mission)
    unilm, actual_missing = resolve(CONFIGS / "unilm_nav.yaml", mission_path=mission)
    assert missing == actual_missing
    assert unilm["endpoints"] == full["endpoints"]
    for name, component in full["components"].items():
        if name != "experiment":
            assert unilm["components"][name] == component
    config = unilm["components"]["unilm_nav"]
    assert config["camera"] == "chest" and config["camera_height_m"] == 1.0
    assert unilm["components"]["experiment"]["models"]["unilm_nav"]["camera"] == "chest"
    path = tmp_path / "runtime.yaml"
    path.write_text(yaml.safe_dump(unilm))
    command = build_base_pose_agent_command(load_inference_launch_config(path), Path.cwd())
    assert ". ./.env.local" in command
    assert "gear_sonic.utils.inference.base_pose.agent" in command
    assert f"--profile {path}" in command
