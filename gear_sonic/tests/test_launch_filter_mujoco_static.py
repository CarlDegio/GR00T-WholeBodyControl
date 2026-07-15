from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "launch_filter_mujoco_tmux.sh"


def test_launcher_starts_relay_mujoco_and_official_cpp_deploy():
    source = SCRIPT.read_text()

    assert "filter_velocity_relay.py" in source
    assert "run_sim_loop.py" in source
    assert "just run g1_deploy_onnx_ref" in source
    assert "--input-type zmq_manager" in source
    assert "pico_manager_thread_server.py" not in source
    assert "while [[ ! -e '$READY_FILE' ]]" in source
