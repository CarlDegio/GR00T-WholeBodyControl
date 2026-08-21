from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "launch_filter_mujoco_tmux.sh"


def test_launcher_starts_relay_mujoco_and_official_cpp_deploy():
    source = SCRIPT.read_text()

    assert "python -m gear_sonic.utils.planner_control.velocity_relay" in source
    assert "python -m gear_sonic.utils.mujoco_sim.service" in source
    assert "just run g1_deploy_onnx_ref" in source
    assert "--input-type zmq_manager" in source
    assert "gear_sonic.utils.teleop.pico_manager" not in source
    assert "while [[ ! -e '$READY_FILE' ]]" in source
