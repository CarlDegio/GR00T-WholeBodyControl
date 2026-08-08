from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


sys.modules.setdefault("tyro", types.ModuleType("tyro"))

from gear_sonic.runtime.config import load_runtime_profile
from gear_sonic.runtime.endpoints import get_endpoint
from gear_sonic.scripts.launch_inference import (
    InferenceLaunchConfig,
    _dotenv_has_nonempty_value,
    _parse_pane_ids,
    build_fastlio_command,
    build_control_gateway_command,
    build_operator_console_command,
    build_operator_interface_command,
    build_livox_command,
    build_lingbot_command,
    build_planner_input_command,
    build_navdp_planner_command,
    build_navdp_server_command,
    build_sensor_gateway_command,
    build_sensor_shadow_command,
    build_vla_inference_command,
    load_inference_launch_config,
    run_readiness_gate,
)


def test_dotenv_key_check_does_not_require_loading_secret(tmp_path: Path) -> None:
    env_file = tmp_path / ".env.local"
    env_file.write_text(
        "# local secrets\nexport DASHSCOPE_API_KEY='secret-value'\nEMPTY=\n",
        encoding="utf-8",
    )

    assert _dotenv_has_nonempty_value(env_file, "DASHSCOPE_API_KEY")
    assert not _dotenv_has_nonempty_value(env_file, "EMPTY")
    assert not _dotenv_has_nonempty_value(env_file, "MISSING")


def test_parse_pane_ids_returns_stable_ids_in_visual_index_order() -> None:
    output = "2 %8\n0 %3\n1 %5\n5 %13\n4 %11\n3 %9\n"

    assert _parse_pane_ids(output) == ["%3", "%5", "%8", "%9", "%11", "%13"]


def test_parse_pane_ids_rejects_incomplete_layout() -> None:
    with pytest.raises(RuntimeError, match="expected 6 tmux panes, found 5"):
        _parse_pane_ids("0 %1\n1 %2\n2 %3\n3 %4\n4 %5\n")


def test_parse_pane_ids_supports_gateways_in_the_inference_window() -> None:
    output = "\n".join(f"{index} %{index + 1}" for index in range(9))

    assert _parse_pane_ids(output, 9) == [f"%{index + 1}" for index in range(9)]


def test_vla_action_horizon_defaults_to_fifty() -> None:
    assert InferenceLaunchConfig().action_horizon == 50


def test_yaml_contains_every_launch_parameter() -> None:
    loaded = load_inference_launch_config()

    assert loaded.prompt.startswith("Approach the tabletop")
    assert loaded.lavira_mission == "blue basket"
    assert loaded.lavira_global_target == "blue basket"
    assert loaded.lavira_vision_backend == "qwenvl"


def test_launcher_defaults_match_current_endpoint_inventory() -> None:
    config = InferenceLaunchConfig()

    assert config.policy_port == get_endpoint("policy_server").port
    assert config.camera_port == get_endpoint("camera_server").port
    assert config.keyboard_planner_port == get_endpoint("navigation_command").port
    assert config.navdp_output_port == get_endpoint("planner_relay").port
    assert config.lavira_depth_port == get_endpoint("lingbot_depth").port
    assert config.navdp_port == get_endpoint("xnavdp_http").port
    assert config.sensor_gateway_port == get_endpoint("sensor_gateway_metadata").port
    assert config.vla_timing_port == get_endpoint("vla_timing_ingress").port
    assert config.control_gateway_intent_port == get_endpoint("control_gateway_intent").port
    assert config.control_gateway_status_port == get_endpoint("control_gateway_status").port
    assert config.control_gateway_dispatch_port == get_endpoint("control_gateway_dispatch").port
    assert config.keyboard_zmq_port == get_endpoint("operator_keyboard_legacy").port
    assert config.sensor_gateway_visualization_port == get_endpoint(
        "sensor_gateway_visualization_ingress"
    ).port


def test_control_gateway_replaces_inline_keyboard_and_preserves_ports() -> None:
    command = build_control_gateway_command(
        InferenceLaunchConfig(),
        Path("/workspace/sonic"),
    )

    assert "run_control_gateway.py" in command
    assert "--legacy-port 5580" in command
    assert "--intent-port 5561" in command
    assert "--dispatch-port 5565" in command
    assert "--status-port 5562" in command
    assert "--navigation-port 5558" in command
    assert "--navigation-status-port 5559" in command

    console = build_operator_console_command(
        InferenceLaunchConfig(),
        Path("/workspace/sonic"),
    )
    assert "run_operator_console.py" in console
    assert "--port 5561" in console
    operator = build_operator_interface_command(
        InferenceLaunchConfig(),
        Path("/workspace/sonic"),
    )
    assert "run_operator_console.py" in operator
    assert "run_operator_cv_viewer.py" not in operator


def test_launcher_defaults_match_unified_runtime_profile() -> None:
    config = load_inference_launch_config()
    profile = load_runtime_profile()

    launcher = profile.component("launcher")
    vla = profile.component("vla")
    assert launcher["sim"] is config.sim
    assert launcher["keyboard_planner"] is config.keyboard_planner
    assert launcher["planner_input"] == config.planner_input
    assert launcher["data_exporter"] is config.data_exporter
    assert launcher["sensor_gateway"] is config.sensor_gateway
    assert launcher["sensor_shadow"] is config.sensor_shadow
    assert launcher["control_gateway"] is config.control_gateway
    assert launcher["opencv_viewer"] is config.opencv_viewer
    assert launcher["lidar_ready_timeout_s"] == config.lidar_ready_timeout
    assert launcher["navigation_ready_timeout_s"] == config.navigation_ready_timeout
    assert vla["embodiment_tag"] == config.embodiment_tag
    assert vla["prompt"] == config.prompt
    assert vla["action_publish_hz"] == config.action_publish_rate
    assert vla["action_horizon"] == config.action_horizon
    assert vla["control_input"] == config.vla_control_input
    assert vla["sensor_input"] == config.vla_sensor_input
    assert vla["sensor_gateway_poll_hz"] == config.vla_sensor_gateway_poll_hz
    assert profile.component("navdp")["sensor_input"] == config.navdp_sensor_input


def test_navdp_stack_commands_use_ros_topics_and_official_xnavdp_server() -> None:
    config = InferenceLaunchConfig()
    planner = build_navdp_planner_command(config, Path("/workspace/sonic"))
    server = build_navdp_server_command(config)
    livox = build_livox_command(config)
    fastlio = build_fastlio_command(config)

    assert "navdp_planner.py" in planner
    assert "--sensor-input gateway" in planner
    assert "--sensor-gateway-endpoint tcp://127.0.0.1:5560" in planner
    assert "--no-visualize" in planner
    assert "--visualization-gateway-endpoint tcp://127.0.0.1:5566" in planner
    assert "--sensor-gateway-endpoint tcp://127.0.0.1:5560" in planner
    assert "--navdp-request-timeout-s 10.0" in planner
    assert "source /opt/ros/humble/setup.bash" in planner
    assert config.navdp_root == "/home/user/Project/NavDP/baselines/x-navdp"
    assert config.navdp_checkpoint.endswith("/x-navdp_posttrain.ckpt")
    assert "python -m eval.src.policy_server" in server
    assert "--embodiment humanoid" in server
    assert "--real" in server
    assert "--no-visualization" in server
    assert f"--checkpoint {config.navdp_checkpoint}" in server
    assert "msg_MID360_launch.py" in livox
    assert "mapping.launch.py" in fastlio
    assert "rviz:=false" in fastlio
    assert "mid360_reasan_open3d.py" not in " ".join((planner, server, livox, fastlio))


def test_lingbot_uses_control_gateway_for_pose_planner_gpu_gating() -> None:
    config = InferenceLaunchConfig(planner_input="lavira")
    planner_command = build_planner_input_command(
        config,
        Path("/workspace/sonic"),
    )
    command = build_lingbot_command(
        config,
        Path("/workspace/sonic"),
    )

    assert "run_lingbot_depth_viewer.py" in command
    assert "run_lingbot_depth_viewer.py" not in planner_command
    assert "sonic_lingbot_ready" in planner_command
    assert "--sensor-gateway-endpoint tcp://127.0.0.1:5560" in command
    assert "run_lingbot_depth_viewer.py --camera-host" not in command
    assert "--control-gateway-endpoint tcp://127.0.0.1:5565" in command


def test_lavira_uses_only_control_and_sensor_gateways() -> None:
    command = build_planner_input_command(
        InferenceLaunchConfig(planner_input="lavira"),
        Path("/workspace/sonic"),
    )

    assert "--sensor-gateway-endpoint tcp://127.0.0.1:5560" in command
    assert "--control-gateway-endpoint tcp://127.0.0.1:5565" in command
    assert "--control-gateway-intent-endpoint tcp://127.0.0.1:5561" in command
    assert "--host" not in command
    assert "--port 5558" not in command
    assert "--status-port 5559" not in command
    assert "5564" not in command


def test_runtime_sidecars_are_read_only_and_navdp_uses_gateway_by_default() -> None:
    config = InferenceLaunchConfig(camera_host="192.168.123.164")
    root = Path("/workspace/sonic")
    gateway = build_sensor_gateway_command(config, root)
    shadow = build_sensor_shadow_command(config, root)
    planner = build_navdp_planner_command(config, root)

    assert "run_sensor_gateway.py" in gateway
    assert "run_operator_cv_viewer.py" in gateway
    assert "/tmp/sonic_opencv_viewer.log" in gateway
    assert "run_lingbot_depth_viewer.py" in gateway
    assert "/tmp/sonic_lingbot.log" in gateway
    assert "msg_MID360_launch.py" in gateway
    assert "mapping.launch.py" in gateway
    assert "/tmp/sonic_livox_driver.log" in gateway
    assert "/tmp/sonic_fastlio.log" in gateway
    assert "run_sensor_gateway_shadow.py" in shadow
    assert "--camera-host 192.168.123.164" in gateway
    assert "--camera-host 192.168.123.164" in shadow
    assert "--rpc-port 5560" in gateway
    assert "--gateway-port 5560" in shadow
    assert "PYTHONPATH=/workspace/sonic:$PYTHONPATH" in gateway
    assert "PYTHONPATH=/workspace/sonic:$PYTHONPATH" in shadow
    assert "cpp_command" not in gateway + shadow
    assert "5556" not in gateway + shadow
    assert "--sensor-input gateway" in planner
    assert "--sensor-gateway-endpoint tcp://127.0.0.1:5560" in planner
    assert "run_sensor_gateway" not in planner


def test_vla_gateway_input_is_explicit_and_preserves_control_endpoints() -> None:
    root = Path("/workspace/sonic")
    gateway = build_vla_inference_command(InferenceLaunchConfig(), root)
    legacy = build_vla_inference_command(
        InferenceLaunchConfig(
            vla_sensor_input="legacy",
            vla_control_input="legacy",
        ),
        root,
    )

    assert "--sensor-input legacy" in legacy
    assert "--control-input legacy" in legacy
    assert "--sensor-input gateway" in gateway
    assert "--control-input gateway" in gateway
    assert "--control-gateway-endpoint tcp://127.0.0.1:5565" in gateway
    assert "--sensor-gateway-endpoint tcp://127.0.0.1:5560" in gateway
    assert "--sensor-gateway-poll-hz 50.0" in gateway
    assert "--planner-relay-zmq-port 5563" in gateway
    assert "--action-zmq-port" not in gateway


def test_navdp_gateway_input_is_explicit_and_keeps_the_same_output_contract() -> None:
    config = InferenceLaunchConfig(navdp_sensor_input="gateway")

    planner = build_navdp_planner_command(config, Path("/workspace/sonic"))

    assert "--sensor-input gateway" in planner
    assert "--sensor-gateway-endpoint tcp://127.0.0.1:5560" in planner
    assert "--navdp-server http://127.0.0.1:19999" in planner
    assert "--output-endpoint" not in planner


def test_launcher_runs_readiness_gate_synchronously(monkeypatch) -> None:
    commands: list[list[str]] = []

    class Result:
        returncode = 0

    def run(command, **_kwargs):
        commands.append(command)
        return Result()

    monkeypatch.setattr("gear_sonic.scripts.launch_inference.subprocess.run", run)
    config = InferenceLaunchConfig(lidar_ready_timeout=45.0)

    assert run_readiness_gate(config, Path("/workspace/sonic"), "lidar")
    shell = commands[0][-1]
    assert "navdp_readiness_gate.py --stage lidar" in shell
    assert "--timeout 45.0" in shell


def test_gateway_navdp_readiness_requires_the_gateway_rpc(monkeypatch) -> None:
    commands: list[list[str]] = []

    class Result:
        returncode = 0

    def run(command, **_kwargs):
        commands.append(command)
        return Result()

    monkeypatch.setattr("gear_sonic.scripts.launch_inference.subprocess.run", run)
    config = InferenceLaunchConfig(navdp_sensor_input="gateway")

    assert run_readiness_gate(config, Path("/workspace/sonic"), "navigation")
    shell = commands[0][-1]
    assert "--require-sensor-gateway" in shell
    assert "--sensor-gateway-port 5560" in shell
