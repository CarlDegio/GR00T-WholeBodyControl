from __future__ import annotations

import shlex
import subprocess
import sys
import types
from pathlib import Path

import pytest


sys.modules.setdefault("tyro", types.ModuleType("tyro"))

from gear_sonic.runtime.config import load_runtime_profile
from gear_sonic.runtime.endpoints import get_endpoint
from gear_sonic.scripts.launch_inference import (
    InferenceLaunchConfig,
    _check_prerequisites,
    _clear_stale_fastlio_processes,
    _clear_stale_navdp_processes,
    _dotenv_has_nonempty_value,
    _inference_pane_count,
    _parse_pane_ids,
    build_base_pose_agent_command,
    build_fastlio_command,
    build_fastlio_supervisor_command,
    build_slam_debug_command,
    build_control_gateway_command,
    build_data_exporter_command,
    build_deploy_command,
    build_operator_console_command,
    build_operator_interface_command,
    build_livox_command,
    build_depth_anything_command,
    build_planner_input_command,
    build_planner_velocity_executor_command,
    build_navdp_planner_command,
    build_navdp_planner_pane_command,
    build_navdp_server_background_command,
    build_navdp_server_command,
    build_sensor_gateway_command,
    build_vla_inference_command,
    default_launch_config_path,
    load_inference_launch_config,
    resolve_deploy_policy,
    run_readiness_gate,
)


def test_startup_clears_only_stale_fastlio_processes(monkeypatch) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(
        "gear_sonic.scripts.launch_inference.subprocess.run",
        lambda command, **_kwargs: commands.append(command),
    )
    monkeypatch.setattr("gear_sonic.scripts.launch_inference.time.sleep", lambda _s: None)

    _clear_stale_fastlio_processes()

    assert [command[:3] for command in commands] == [
        ["pkill", "-TERM", "-f"],
        ["pkill", "-TERM", "-f"],
        ["pkill", "-KILL", "-f"],
        ["pkill", "-KILL", "-f"],
    ]
    assert all("fast_lio" in command[-1] or "fastlio_mapping" in command[-1] for command in commands)


def test_startup_clears_only_stale_navdp_processes(monkeypatch) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(
        "gear_sonic.scripts.launch_inference.subprocess.run",
        lambda command, **_kwargs: commands.append(command),
    )
    monkeypatch.setattr("gear_sonic.scripts.launch_inference.time.sleep", lambda _s: None)

    _clear_stale_navdp_processes()

    assert [command[:3] for command in commands] == [
        ["pkill", "-TERM", "-f"],
        ["pkill", "-TERM", "-f"],
        ["pkill", "-KILL", "-f"],
        ["pkill", "-KILL", "-f"],
    ]
    assert all(
        "navdp_planner" in command[-1] or "eval\\.src\\.policy_server" in command[-1]
        for command in commands
    )


def _write_deploy_policy_files(
    deploy_dir: Path,
    checkpoint: str,
    obs_config: str,
) -> None:
    prefix = deploy_dir / checkpoint
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_name(prefix.name + "_encoder.onnx").write_bytes(b"encoder")
    prefix.with_name(prefix.name + "_decoder.onnx").write_bytes(b"decoder")
    config_path = deploy_dir / obs_config
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("observations: []\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("variant", "checkpoint", "obs_config"),
    [
        ("default", "policy/release/model", "policy/release/observation_config.yaml"),
        (
            "low_latency",
            "policy/low_latency/model",
            "policy/low_latency/observation_config.yaml",
        ),
        (
            "sonic_v1_1",
            "policy/sonic_v1_1/model",
            "policy/sonic_v1_1/observation_config.yaml",
        ),
    ],
)
def test_deploy_policy_presets_keep_checkpoint_and_observation_config_paired(
    tmp_path: Path,
    variant: str,
    checkpoint: str,
    obs_config: str,
) -> None:
    _write_deploy_policy_files(tmp_path, checkpoint, obs_config)

    resolved = resolve_deploy_policy(
        InferenceLaunchConfig(deploy_policy_variant=variant),
        tmp_path,
    )

    assert resolved == (checkpoint, obs_config)


def test_deploy_policy_allows_paired_custom_override(tmp_path: Path) -> None:
    checkpoint = "policy/custom/controller"
    obs_config = "policy/custom/observations.yaml"
    _write_deploy_policy_files(tmp_path, checkpoint, obs_config)

    resolved = resolve_deploy_policy(
        InferenceLaunchConfig(
            deploy_policy_variant="sonic_v1_1",
            deploy_checkpoint=checkpoint,
            deploy_obs_config=obs_config,
        ),
        tmp_path,
    )

    assert resolved == (checkpoint, obs_config)


def test_deploy_policy_expands_custom_tilde_paths_for_the_launch_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home with spaces"
    checkpoint = home / "policy" / "custom" / "controller"
    obs_config = home / "policy" / "custom" / "observations.yaml"
    monkeypatch.setenv("HOME", str(home))
    _write_deploy_policy_files(
        tmp_path,
        str(checkpoint),
        str(obs_config),
    )

    command = build_deploy_command(
        InferenceLaunchConfig(
            deploy_checkpoint="~/policy/custom/controller",
            deploy_obs_config="~/policy/custom/observations.yaml",
        ),
        tmp_path,
    )

    assert f"--cp {shlex.quote(str(checkpoint))}" in command
    assert f"--obs-config {shlex.quote(str(obs_config))}" in command


def test_deploy_policy_rejects_partial_custom_override(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be set together"):
        resolve_deploy_policy(
            InferenceLaunchConfig(deploy_checkpoint="policy/custom/model"),
            tmp_path,
        )


def test_deploy_policy_reports_missing_v1_1_files_and_download_command(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_deploy_policy(
            InferenceLaunchConfig(deploy_policy_variant="sonic_v1_1"),
            tmp_path,
        )

    message = str(exc_info.value)
    assert "model_encoder.onnx" in message
    assert "model_decoder.onnx" in message
    assert "observation_config.yaml" in message
    assert "python download_from_hf.py --sonic-v1-1" in message


def test_build_deploy_command_uses_resolved_v1_1_policy(tmp_path: Path) -> None:
    deploy_dir = tmp_path / "gear_sonic_deploy"
    checkpoint = "policy/sonic_v1_1/model"
    obs_config = "policy/sonic_v1_1/observation_config.yaml"
    _write_deploy_policy_files(deploy_dir, checkpoint, obs_config)

    command = build_deploy_command(
        InferenceLaunchConfig(deploy_policy_variant="sonic_v1_1", sim=True),
        tmp_path,
    )

    assert f"cd {deploy_dir}" in command
    assert "--cp policy/sonic_v1_1/model" in command
    assert "--obs-config policy/sonic_v1_1/observation_config.yaml" in command
    assert command.endswith("sim")


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
    output = "2 %8\n0 %3\n1 %5\n4 %11\n3 %9\n"

    assert _parse_pane_ids(output) == ["%3", "%5", "%8", "%9", "%11"]


def test_parse_pane_ids_rejects_incomplete_layout() -> None:
    with pytest.raises(RuntimeError, match="expected 5 tmux panes, found 4"):
        _parse_pane_ids("0 %1\n1 %2\n2 %3\n3 %4\n")


def test_parse_pane_ids_supports_gateways_in_the_inference_window() -> None:
    output = "\n".join(f"{index} %{index + 1}" for index in range(8))

    assert _parse_pane_ids(output, 8) == [f"%{index + 1}" for index in range(8)]


def test_configured_inference_layout_uses_nine_panes() -> None:
    assert _inference_pane_count(load_inference_launch_config()) == 9


def test_vla_action_horizon_defaults_to_fifty() -> None:
    assert InferenceLaunchConfig().action_horizon == 50


def test_gateways_are_mandatory_launcher_components() -> None:
    config = InferenceLaunchConfig()
    profile = load_runtime_profile()

    assert not hasattr(config, "sensor_gateway")
    assert not hasattr(config, "control_gateway")
    assert "sensor_gateway" not in profile.component("launcher")
    assert "control_gateway" not in profile.component("launcher")


def test_yaml_contains_every_launch_parameter() -> None:
    loaded = load_inference_launch_config()

    assert loaded.deploy_policy_variant == "sonic_v1_1"
    assert loaded.prompt.startswith("Move in front of the table")
    assert loaded.lavira_mission == "blue basket"
    assert loaded.lavira_global_target == "blue basket"
    assert loaded.lavira_qwenvl_model == "qwen3-vl-32b-instruct"
    assert loaded.base_pose_enabled is True
    assert loaded.base_pose_task == "align to the blue basket"
    assert loaded.base_pose_surface_prompt == "desk"
    assert loaded.base_pose_dual_chest_depth_stream == "camera/chest_view_depth"
    assert loaded.base_pose_raw_min_linear_speed_m_s == pytest.approx(0.4)
    assert loaded.base_pose_raw_max_lateral_speed_m_s == pytest.approx(0.4)
    assert loaded.base_pose_mode == "dual_raw_yoloe_servo"
    assert not hasattr(loaded, "base_pose_vision_backend")
    assert not hasattr(loaded, "base_pose_model")
    assert not hasattr(loaded, "base_pose_codex_fast")
    assert loaded.base_pose_qwenvl_model == "qwen3-vl-plus"
    assert loaded.slam_debug is False


def test_schema_v1_yaml_without_policy_variant_uses_default(
    tmp_path: Path,
) -> None:
    current_text = default_launch_config_path().read_text(encoding="utf-8")
    legacy_text = "\n".join(
        line
        for line in current_text.splitlines()
        if not line.lstrip().startswith("deploy_policy_variant:")
    )
    legacy_path = tmp_path / "legacy_launch_inference.yaml"
    legacy_path.write_text(legacy_text + "\n", encoding="utf-8")

    loaded = load_inference_launch_config(legacy_path)

    assert loaded.deploy_policy_variant == "default"


def test_launcher_defaults_match_current_endpoint_inventory() -> None:
    config = InferenceLaunchConfig()

    assert config.policy_port == get_endpoint("policy_server").port
    assert config.camera_port == get_endpoint("camera_server").port
    assert config.keyboard_planner_port == get_endpoint("navigation_command").port
    assert config.depth_anything_port == get_endpoint("depth_anything").port
    assert config.sensor_gateway_port == get_endpoint("sensor_gateway_metadata").port
    assert config.vla_timing_port == get_endpoint("vla_timing_ingress").port
    assert config.control_gateway_intent_port == get_endpoint("control_gateway_intent").port
    assert config.control_gateway_status_port == get_endpoint("control_gateway_status").port
    assert config.control_gateway_dispatch_port == get_endpoint("control_gateway_dispatch").port
    assert config.sensor_gateway_visualization_port == get_endpoint(
        "sensor_gateway_visualization_ingress"
    ).port


def test_control_gateway_uses_only_typed_gateway_ports() -> None:
    command = build_control_gateway_command(
        InferenceLaunchConfig(),
        Path("/workspace/sonic"),
    )

    assert "run_control_gateway.py" in command
    assert "5580" not in command
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


def test_data_exporter_uses_runtime_profile_gateways_only() -> None:
    command = build_data_exporter_command(
        InferenceLaunchConfig(
            task_prompt="collect a demo",
            dataset_name="session one",
            record_chest_camera=True,
        ),
        Path("/workspace/sonic"),
    )

    assert "run_data_exporter.py" in command
    assert "--profile" in command
    assert "--task-prompt 'collect a demo'" in command
    assert "--dataset-name 'session one'" in command
    assert "--record-chest-camera" in command
    assert "--camera-host" not in command
    assert "--camera-port" not in command
    assert "--state-zmq" not in command
    assert "--data-collection-frequency" not in command


def test_launcher_defaults_match_unified_runtime_profile() -> None:
    config = load_inference_launch_config()
    profile = load_runtime_profile()

    launcher = profile.component("launcher")
    vla = profile.component("vla")
    assert launcher["sim"] is config.sim
    assert launcher["keyboard_planner"] is config.keyboard_planner
    assert launcher["planner_input"] == config.planner_input
    assert launcher["data_exporter"] is config.data_exporter
    assert launcher["opencv_viewer"] is config.opencv_viewer
    assert launcher["lidar_ready_timeout_s"] == config.lidar_ready_timeout
    assert launcher["navigation_ready_timeout_s"] == config.navigation_ready_timeout
    assert vla["embodiment_tag"] == config.embodiment_tag
    assert vla["prompt"] == config.prompt
    assert vla["action_publish_hz"] == config.action_publish_rate
    assert vla["action_horizon"] == config.action_horizon
    assert vla["sensor_gateway_poll_hz"] == config.vla_sensor_gateway_poll_hz
    assert profile.component("navdp")["sensor_gateway_poll_hz"] == 20.0


def test_navdp_stack_commands_use_ros_topics_and_official_xnavdp_server() -> None:
    config = InferenceLaunchConfig()
    planner = build_navdp_planner_command(config, Path("/workspace/sonic"))
    planner_pane = build_navdp_planner_pane_command(
        config, Path("/workspace/sonic")
    )
    server = build_navdp_server_command(config)
    background_server = build_navdp_server_background_command(config)
    livox = build_livox_command(config)
    fastlio = build_fastlio_command(config)
    fastlio_supervisor = build_fastlio_supervisor_command(config)

    assert "navdp_planner.py" in planner
    assert "--sensor-input" not in planner
    assert "--sensor-gateway-endpoint tcp://127.0.0.1:5560" in planner
    assert "--no-visualize" in planner
    assert "--visualization-gateway-endpoint tcp://127.0.0.1:5566" in planner
    assert "--sensor-gateway-endpoint tcp://127.0.0.1:5560" in planner
    assert "--navdp-request-timeout-s 10.0" in planner
    assert "--control-hz 20.0" in planner
    assert "--mpc-hz 10.0" in planner
    assert "--goal-tolerance-m 0.5" in planner
    assert "--radar-timeout-s" not in planner
    assert "--output-endpoint 'tcp://*:5568'" in planner
    assert "source /opt/ros/humble/setup.bash" in planner
    assert config.navdp_root == "/home/user/Project/NavDP/baselines/x-navdp"
    assert config.navdp_checkpoint.endswith("/x-navdp_posttrain.ckpt")
    assert "python -m eval.src.policy_server" in server
    assert "--embodiment humanoid" in server
    assert "--real" in server
    assert "--no-visualization" in server
    assert "PYTHONUNBUFFERED=1" in server
    assert "NAVDP_SERVER_PID=$!" in background_server
    assert "[NavDP server:stderr]" in background_server
    assert "logs are routed to this pane" in background_server
    assert "trap _stop_navdp_server EXIT HUP" in background_server
    assert "navdp_planner.py" in planner_pane
    assert "_stop_navdp_server" in planner_pane
    assert 'exit "${NAVDP_PLANNER_STATUS}"' in planner_pane
    subprocess.run(
        ["bash", "-n"],
        input=f"{background_server}\n{planner_pane}\n",
        text=True,
        check=True,
    )
    assert f"--checkpoint {config.navdp_checkpoint}" in server
    assert "msg_MID360_launch.py" in livox
    assert "mapping.launch.py" in fastlio
    assert "rviz:=false" in fastlio
    assert "run_fastlio_supervisor.py" in fastlio_supervisor
    assert "--config-file mid360.yaml" in fastlio_supervisor
    assert "--control-gateway-endpoint tcp://127.0.0.1:5561" in fastlio_supervisor
    assert "mid360_reasan_open3d.py" not in " ".join((planner, server, livox, fastlio))


def test_depth_anything_is_an_independent_rgb_only_shared_service() -> None:
    config = InferenceLaunchConfig(planner_input="lavira")
    planner_command = build_planner_input_command(
        config,
        Path("/workspace/sonic"),
    )
    command = build_depth_anything_command(
        config,
        Path("/workspace/sonic"),
    )

    assert "run_depth_anything.py" in command
    assert "run_depth_anything.py" not in planner_command
    assert "sonic_depth_anything_ready" in planner_command
    assert "--ready-file /tmp/sonic_depth_anything_ready" in command
    assert "$ready_file" not in command
    assert "--sensor-gateway-endpoint tcp://127.0.0.1:5560" in command
    assert "--control-gateway-endpoint tcp://127.0.0.1:5565" in command
    assert "--encoder vitb" in command
    assert "--input-size 644" in command
    assert "--inference-hz 10.5" in command


def test_lavira_uses_only_control_and_sensor_gateways() -> None:
    command = build_planner_input_command(
        InferenceLaunchConfig(planner_input="lavira"),
        Path("/workspace/sonic"),
    )

    assert "--sensor-gateway-endpoint tcp://127.0.0.1:5560" in command
    assert "--control-gateway-endpoint tcp://127.0.0.1:5565" in command
    assert "--control-gateway-intent-endpoint tcp://127.0.0.1:5561" in command
    assert "--qwenvl-model qwen3-vl-32b-instruct" in command
    assert "--qwenvl-timeout-seconds 180.0" in command
    assert "--vision-backend" not in command
    assert "--model gpt-" not in command
    assert "codex" not in command.lower()
    assert "--host" not in command
    assert "--port 5558" not in command
    assert "--status-port 5559" not in command
    assert "5564" not in command


def test_base_pose_agent_uses_gateway_arbitration_and_fixed_task() -> None:
    config = InferenceLaunchConfig(
        base_pose_enabled=True,
        base_pose_task="align with the medicine bottle and basket",
        base_pose_surface_prompt="work bench",
    )
    command = build_base_pose_agent_command(config, Path("/workspace/sonic"))

    assert "gear_sonic/scripts/base_pose_agent.py" in command
    assert "--task 'align with the medicine bottle and basket'" in command
    assert "--surface-prompt 'work bench'" in command
    assert "--mode dual_raw_yoloe_servo" in command
    assert "--dual-head-camera-stream ego_view" in command
    assert "--dual-head-depth-stream camera/ego_view_depth" in command
    assert "--dual-chest-camera-stream chest_view" in command
    assert "--dual-chest-depth-stream camera/chest_view_depth" in command
    assert "--sensor-gateway-endpoint tcp://127.0.0.1:5560" in command
    assert "--control-gateway-endpoint tcp://127.0.0.1:5565" in command
    assert "--control-gateway-intent-endpoint tcp://127.0.0.1:5561" in command
    assert ". ./.env.local" in command
    assert "--qwenvl-model qwen3-vl-plus" in command
    assert "--qwenvl-timeout-seconds 600.0" in command
    assert "--dual-rgbd-buffer-size 8" in command
    assert "--dual-rgbd-poll-hz 60.0" in command
    assert "--dual-head-reacquire-frames 1" in command
    assert "--dual-head-release-missing-frames 3" in command
    assert "--raw-chest-handoff-distance-m" not in command
    assert "--raw-chest-fallback-forward-tolerance-m" not in command
    assert "--raw-chest-fallback-lateral-tolerance-m" not in command
    assert "--raw-head-target-distance-m 1.0" in command
    assert "--raw-chest-target-distance-m 0.8" in command
    assert "--vision-backend" not in command
    assert "codex" not in command.lower()
    assert "--port 5558" not in command


def test_base_pose_qwenvl_command_loads_local_key_without_persisting_by_default() -> None:
    command = build_base_pose_agent_command(
        InferenceLaunchConfig(
            base_pose_enabled=True,
            base_pose_task="align",
        ),
        Path("/workspace/sonic"),
    )

    assert ". ./.env.local" in command
    assert "--vision-backend" not in command
    assert "codex" not in command.lower()
    assert "--qwenvl-thinking-budget 500" in command
    assert "--persist-diagnostics" not in command


def test_base_pose_yoloe_command_uses_aligned_raw_depth_and_local_model() -> None:
    config = InferenceLaunchConfig(
        base_pose_enabled=True,
        base_pose_task="align",
        base_pose_mode="raw_yoloe_servo",
    )
    command = build_base_pose_agent_command(config, Path("/workspace/sonic"))
    executor = build_planner_velocity_executor_command(
        config,
        Path("/workspace/sonic"),
    )

    assert "--mode raw_yoloe_servo" in command
    assert "--depth-stream camera/ego_view_depth" in command
    assert "--raw-yoloe-model-path tools/yoloe26m/weights/yoloe-26m-seg.pt" in command
    assert "--raw-head-target-distance-m 1.0" in command
    assert "--raw-chest-target-distance-m 0.8" in command
    assert "--raw-orientation-telemetry-source tcp://127.0.0.1:5569" in command
    assert "--orientation-output-endpoint 'tcp://*:5569'" in executor


def test_orientation_telemetry_is_enabled_for_dual_raw_yoloe_mode() -> None:
    executor = build_planner_velocity_executor_command(
        InferenceLaunchConfig(
            base_pose_enabled=True,
            base_pose_mode="dual_raw_yoloe_servo",
        ),
        Path("/workspace/sonic"),
    )

    assert "--orientation-output-endpoint 'tcp://*:5569'" in executor


def test_runtime_sidecars_are_read_only_and_navdp_uses_gateway_by_default() -> None:
    config = InferenceLaunchConfig(camera_host="192.168.123.164")
    root = Path("/workspace/sonic")
    gateway = build_sensor_gateway_command(config, root)
    planner = build_navdp_planner_command(config, root)
    executor = build_planner_velocity_executor_command(config, root)

    assert "run_sensor_gateway.py" in gateway
    assert "run_operator_cv_viewer.py" in gateway
    assert "--control-gateway-endpoint tcp://127.0.0.1:5565" in gateway
    assert "--navigation-runtime-status-endpoint tcp://127.0.0.1:5570" in gateway
    assert "/tmp/sonic_opencv_viewer.log" in gateway
    assert "run_depth_anything.py" in gateway
    assert "/tmp/sonic_depth_anything.log" in gateway
    assert "msg_MID360_launch.py" in gateway
    assert "run_fastlio_supervisor.py" in gateway
    assert "--control-gateway-endpoint tcp://127.0.0.1:5561" in gateway
    assert "/tmp/sonic_livox_driver.log" in gateway
    assert "/tmp/sonic_fastlio.log" in gateway
    assert "ros2 bag record" not in gateway
    assert "--camera-host 192.168.123.164" in gateway
    assert "--rpc-port 5560" in gateway
    assert "PYTHONPATH=/workspace/sonic:$PYTHONPATH" in gateway
    assert "cpp_command" not in gateway
    assert "5556" not in gateway
    assert "--sensor-input" not in planner
    assert "--sensor-gateway-endpoint tcp://127.0.0.1:5560" in planner
    assert "run_sensor_gateway" not in planner
    assert "--output-endpoint 'tcp://*:5568'" in planner
    assert "planner_velocity_executor.py" in executor
    assert "--navdp-velocity-endpoint tcp://127.0.0.1:5568" in executor
    assert "--output-endpoint 'tcp://*:5563'" in executor
    assert "--runtime-status-endpoint 'tcp://*:5570'" in executor


def test_base_pose_uses_raw_chest_depth_without_starting_depth_anything() -> None:
    command = build_sensor_gateway_command(
        InferenceLaunchConfig(
            planner_input="keyboard",
            base_pose_enabled=True,
            base_pose_task="align",
        ),
        Path("/workspace/sonic"),
    )

    assert "run_depth_anything.py" not in command
    assert "/tmp/sonic_depth_anything.log" not in command
    assert "--no-enable-depth-anything" in command


def test_slam_debug_records_raw_inputs_and_fastlio_outputs_per_run() -> None:
    config = InferenceLaunchConfig(slam_debug=True)
    root = Path("/workspace/sonic")

    recorder = build_slam_debug_command(config)
    gateway = build_sensor_gateway_command(config, root)

    assert 'ros2 bag record -o "$slam_debug_dir/rosbag"' in recorder
    assert "/livox/lidar" in recorder
    assert "/livox/imu" in recorder
    assert "/Odometry_loc" in recorder
    assert "/cloud_registered_1" in recorder
    assert "outputs/slam_debug/$(date +%Y%m%d_%H%M%S_%N)" in gateway
    assert "[SLAM debug] recording to $slam_debug_dir" in gateway
    assert '>"$slam_debug_dir/livox_driver.log"' in gateway
    assert '>"$slam_debug_dir/fastlio.log"' in gateway
    assert '>"$slam_debug_dir/rosbag.log"' in gateway
    assert "/tmp/sonic_livox_driver.log" not in gateway
    assert "/tmp/sonic_fastlio.log" not in gateway


def test_vla_uses_only_gateway_inputs_and_preserves_control_endpoints() -> None:
    root = Path("/workspace/sonic")
    gateway = build_vla_inference_command(InferenceLaunchConfig(), root)

    assert "--sensor-input" not in gateway
    assert "--control-input" not in gateway
    assert "--camera-host" not in gateway
    assert "--camera-port" not in gateway
    assert "--control-gateway-endpoint tcp://127.0.0.1:5565" in gateway
    assert "--sensor-gateway-endpoint tcp://127.0.0.1:5560" in gateway
    assert "--sensor-gateway-poll-hz 50.0" in gateway
    assert "--planner-relay-zmq-port 5563" in gateway
    assert "--action-zmq-port" not in gateway


def test_navdp_is_a_pure_velocity_producer_for_the_shared_executor() -> None:
    config = InferenceLaunchConfig()

    planner = build_navdp_planner_command(config, Path("/workspace/sonic"))

    assert "--sensor-input" not in planner
    assert "--camera-host" not in planner
    assert "--camera-port" not in planner
    assert "--sensor-gateway-endpoint tcp://127.0.0.1:5560" in planner
    assert "--navdp-server http://127.0.0.1:19999" in planner
    assert "--output-endpoint 'tcp://*:5568'" in planner
    assert "--radar-timeout-s" not in planner


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
    config = InferenceLaunchConfig()

    assert run_readiness_gate(config, Path("/workspace/sonic"), "navigation")
    shell = commands[0][-1]
    assert "--require-sensor-gateway" in shell
    assert "--sensor-gateway-port 5560" in shell
