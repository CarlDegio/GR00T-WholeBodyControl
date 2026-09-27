"""Deployment checks use fake processes/listeners and never activate the robot."""

from contextlib import nullcontext
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

from gear_sonic.scripts import launch_inference as launch
from gear_sonic.scripts.launcher import TmuxSession


def test_shared_cpp_dependencies_are_explicit_and_valid():
    root = Path(__file__).resolve().parents[2]
    config = launch.load_inference_launch_config()
    settings = launch._runtime_profile(config).component("deploy")
    launch.validate_deploy_dependencies(settings, root / "gear_sonic_deploy")
    command = launch.build_deploy_command(config, root)
    assert f"export onnxruntime_ROOT={settings['onnxruntime_root']}" in command
    assert f"export LD_LIBRARY_PATH={settings['onnxruntime_root']}/lib" in command
    assert f"--planner {settings['planner']}" in command
    assert f"--motion-data {settings['motion_data']}" in command


def test_missing_cpp_assets_fail_before_services_start(tmp_path):
    with pytest.raises(FileNotFoundError) as error:
        launch.validate_deploy_dependencies(
            {"planner": "", "motion_data": "", "onnxruntime_root": str(tmp_path / "sdk")},
            tmp_path,
        )
    assert "SONIC planner model" in str(error.value)
    assert "SONIC reference motions" in str(error.value)
    assert "onnxruntime_cxx_api.h" in str(error.value)


def test_lfs_pointer_cannot_pass_as_a_planner_model(tmp_path):
    (tmp_path / "planner.onnx").write_text("version https://git-lfs.github.com/spec/v1\n")
    (tmp_path / "joint_pos.csv").write_text("0,0\n")
    with pytest.raises(FileNotFoundError, match="Git LFS pointer"):
        launch.validate_deploy_dependencies({"planner": "planner.onnx", "motion_data": "."}, tmp_path)


def test_deploy_output_survives_failure_and_exit_status_is_preserved(tmp_path):
    log = tmp_path / "deploy output.log"
    command = shlex.join(
        [
            sys.executable,
            "-c",
            "import sys; print('build output'); print('missing SDK', file=sys.stderr); sys.exit(17)",
        ]
    )
    result = subprocess.run(
        ["/bin/bash", "-c", launch.build_deploy_pane_command(command, log)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 17
    assert "build output" in log.read_text()
    assert "missing SDK" in log.read_text()


@pytest.mark.parametrize("returncode,output,alive", [(0, "0\n", True), (0, "1\n", False), (1, "", False)])
def test_pane_health_checks_the_exact_pane(returncode, output, alive):
    commands = []

    def runner(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=returncode, stdout=output)

    session = TmuxSession("test", runner=runner)
    assert session.pane_alive("%4") is alive
    assert commands == [["tmux", "display-message", "-p", "-t", "%4", "#{pane_dead}"]]


def test_cpp_wait_fails_on_process_exit_before_polling_a_port(monkeypatch, tmp_path):
    monkeypatch.setattr(launch._TMUX, "pane_alive", lambda pane: False)
    monkeypatch.setattr(launch.socket, "create_connection", lambda *a, **k: pytest.fail("process already exited"))
    with pytest.raises(RuntimeError, match=r"C\+\+ deploy exited during startup"):
        launch.wait_for_deploy(launch.load_inference_launch_config(), "%4", tmp_path / "deploy.log")


def test_cpp_wait_checks_output_listener_without_sending_control(monkeypatch, tmp_path):
    connections = []
    monkeypatch.setattr(launch._TMUX, "pane_alive", lambda pane: True)
    monkeypatch.setattr(launch._TMUX, "send", lambda *a, **k: pytest.fail("readiness must not activate control"))

    def connect(address, **kwargs):
        connections.append(address)
        return nullcontext()

    monkeypatch.setattr(launch.socket, "create_connection", connect)
    config = launch.load_inference_launch_config()
    launch.wait_for_deploy(config, "%4", tmp_path / "deploy.log")
    endpoint = launch._runtime_profile(config).endpoint("cpp_state")
    assert connections == [(endpoint.host, endpoint.port)]


def test_cpp_wait_has_a_bounded_startup_timeout(monkeypatch, tmp_path):
    clock = iter([0.0, 601.0])
    monkeypatch.setattr(launch.time, "monotonic", lambda: next(clock))
    with pytest.raises(RuntimeError, match="did not initialize within"):
        launch.wait_for_deploy(launch.load_inference_launch_config(), "%4", tmp_path / "deploy.log")
