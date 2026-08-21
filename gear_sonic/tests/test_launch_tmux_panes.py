from __future__ import annotations

import shlex
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml


sys.modules.setdefault("tyro", types.ModuleType("tyro"))

from gear_sonic.runtime.profile import load_runtime_profile
from gear_sonic.utils.inference.base_pose.agent import load_base_pose_config
from gear_sonic.utils.inference.lavira.service import load_lavira_config
from gear_sonic.utils.inference.vla.service import load_inference_config
from gear_sonic.scripts.launch_inference import (
    InferenceLaunchConfig,
    _check_prerequisites,
    _clear_stale_fastlio_processes,
    _clear_stale_navdp_processes,
    _create_tmux_session,
    _dotenv_has_nonempty_value,
    _worker_pane_names,
    build_base_pose_agent_command,
    build_fastlio_supervisor_command,
    build_slam_debug_command,
    build_control_gateway_command,
    build_data_exporter_command,
    build_deploy_command,
    build_operator_console_command,
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
        "gear_sonic\\.utils\\.inference\\.navdp\\.service" in command[-1]
        or "eval\\.src\\.policy_server" in command[-1]
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


def _profile_config(
    tmp_path: Path,
    *,
    components: dict[str, dict] | None = None,
) -> InferenceLaunchConfig:
    payload = yaml.safe_load(default_launch_config_path().read_text(encoding="utf-8"))
    for name, values in (components or {}).items():
        payload["components"][name].update(values)
    profile_path = tmp_path / "runtime.yaml"
    profile_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return load_inference_launch_config(profile_path)


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
        {
            "policy_variant": variant,
            "checkpoint": "",
            "obs_config": "",
        },
        tmp_path,
    )

    assert resolved == (checkpoint, obs_config)


def test_deploy_policy_allows_paired_custom_override(tmp_path: Path) -> None:
    checkpoint = "policy/custom/controller"
    obs_config = "policy/custom/observations.yaml"
    _write_deploy_policy_files(tmp_path, checkpoint, obs_config)

    resolved = resolve_deploy_policy(
        {
            "policy_variant": "sonic_v1_1",
            "checkpoint": checkpoint,
            "obs_config": obs_config,
        },
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

    config = _profile_config(
        tmp_path,
        components={
            "deploy": {
                "checkpoint": "~/policy/custom/controller",
                "obs_config": "~/policy/custom/observations.yaml",
            }
        },
    )
    command = build_deploy_command(config, tmp_path)

    assert f"--cp {shlex.quote(str(checkpoint))}" in command
    assert f"--obs-config {shlex.quote(str(obs_config))}" in command


def test_deploy_policy_rejects_partial_custom_override(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be set together"):
        resolve_deploy_policy(
            {
                "policy_variant": "default",
                "checkpoint": "policy/custom/model",
                "obs_config": "",
            },
            tmp_path,
        )


def test_deploy_policy_reports_missing_v1_1_files_and_download_command(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_deploy_policy(
            {
                "policy_variant": "sonic_v1_1",
                "checkpoint": "",
                "obs_config": "",
            },
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

    config = _profile_config(
        tmp_path,
        components={
            "launcher": {"sim": True},
            "deploy": {"policy_variant": "sonic_v1_1"},
        },
    )
    command = build_deploy_command(config, tmp_path)

    assert f"cd {deploy_dir}" in command
    assert "./deploy.sh --yes " in command
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


def test_worker_layout_omits_disabled_navigation_components() -> None:
    assert _worker_pane_names(InferenceLaunchConfig(keyboard_planner=False)) == (
        "deploy",
        "vla",
    )


def test_worker_layout_omits_only_base_pose_when_disabled() -> None:
    assert _worker_pane_names(
        InferenceLaunchConfig(keyboard_planner=True, base_pose_enabled=False)
    ) == (
        "deploy",
        "vla",
        "planner_input",
        "navdp",
        "planner_executor",
    )


def test_tmux_session_builds_stacked_overview_and_tiled_workers(monkeypatch) -> None:
    tmux_commands: list[tuple[str, ...]] = []
    split_calls: list[tuple[str, tuple[str, ...]]] = []
    pane_ids = iter(("%performance", "%deploy"))
    split_ids = iter(
        (
            "%events",
            "%control",
            "%vla",
            "%planner_input",
            "%navdp",
            "%planner_executor",
            "%base_pose",
        )
    )

    def fake_tmux(*args: str, output: bool = False) -> str:
        del output
        tmux_commands.append(args)
        return ""

    def fake_split(
        target: str,
        _shell: tuple[str, ...],
        *split_args: str,
    ) -> str:
        split_calls.append((target, split_args))
        return next(split_ids)

    monkeypatch.setattr("gear_sonic.scripts.launch_inference._tmux", fake_tmux)
    monkeypatch.setattr(
        "gear_sonic.scripts.launch_inference._pane_id",
        lambda _target: next(pane_ids),
    )
    monkeypatch.setattr("gear_sonic.scripts.launch_inference._split_pane", fake_split)
    monkeypatch.setattr("gear_sonic.scripts.launch_inference.time.sleep", lambda _s: None)

    panes = _create_tmux_session(load_inference_launch_config())

    assert panes == {
        "performance": "%performance",
        "control": "%control",
        "events": "%events",
        "deploy": "%deploy",
        "vla": "%vla",
        "planner_input": "%planner_input",
        "navdp": "%navdp",
        "planner_executor": "%planner_executor",
        "base_pose": "%base_pose",
    }
    assert split_calls[:2] == [
        ("%performance", ("-h", "-p", "58")),
        ("%performance", ("-v", "-p", "42")),
    ]
    assert split_calls[2:] == [
        ("sonic_inference:workers", ("-h",)),
    ] * 5
    assert ("select-layout", "-t", "sonic_inference:workers", "tiled") in tmux_commands
    assert tmux_commands[-2:] == [
        ("select-window", "-t", "sonic_inference:overview"),
        ("select-pane", "-t", "%control"),
    ]


def test_vla_action_horizon_defaults_to_fifty() -> None:
    assert load_inference_config().action_horizon == 50


def test_gateways_are_mandatory_launcher_components() -> None:
    config = InferenceLaunchConfig()
    profile = load_runtime_profile()

    assert not hasattr(config, "sensor_gateway")
    assert not hasattr(config, "control_gateway")
    assert "sensor_gateway" not in profile.component("launcher")
    assert "control_gateway" not in profile.component("launcher")


def test_yaml_contains_every_launch_parameter() -> None:
    loaded = load_inference_launch_config()
    vla = load_inference_config()
    lavira = load_lavira_config()
    base_pose = load_base_pose_config()
    profile = load_runtime_profile()

    assert profile.component("deploy")["policy_variant"] == "sonic_v1_1"
    assert vla.prompt.startswith("Move in front of the table")
    assert lavira.mission == "blue basket"
    assert lavira.global_target == "blue basket"
    assert lavira.qwenvl_model == "qwen3-vl-32b-instruct"
    assert loaded.base_pose_enabled is True
    assert base_pose.task == "align to the blue basket"
    assert base_pose.target_prompt == "bluebasket"
    assert base_pose.surface_prompt == "desk"
    assert base_pose.dual_chest_depth_stream == "camera/chest_view_depth"
    assert base_pose.raw_min_linear_speed_m_s == pytest.approx(0.35)
    assert base_pose.raw_max_lateral_speed_m_s == pytest.approx(0.4)
    assert not hasattr(loaded, "base_pose_mode")
    assert not hasattr(loaded, "base_pose_vision_backend")
    assert not hasattr(loaded, "base_pose_model")
    assert not hasattr(loaded, "base_pose_codex_fast")
    assert not hasattr(loaded, "base_pose_qwenvl_model")
    assert not hasattr(loaded, "base_pose_dual_qwenvl_fallback_model")
    assert loaded.slam_debug is False


def test_runtime_profile_has_no_duplicate_launch_inference_section() -> None:
    payload = yaml.safe_load(default_launch_config_path().read_text(encoding="utf-8"))

    assert "launch_inference" not in payload


def test_launcher_keeps_endpoint_addresses_out_of_cli_config() -> None:
    config = InferenceLaunchConfig()

    for field in (
        "deploy_zmq_host",
        "policy_host",
        "policy_port",
        "camera_host",
        "camera_port",
        "sensor_gateway_port",
        "sensor_gateway_visualization_port",
        "vla_timing_port",
        "control_gateway_intent_port",
        "control_gateway_status_port",
        "control_gateway_dispatch_port",
        "keyboard_planner_host",
        "keyboard_planner_port",
        "depth_anything_port",
    ):
        assert not hasattr(config, field)


def test_launcher_forwards_selected_runtime_profile_without_copying_endpoints(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load(default_launch_config_path().read_text(encoding="utf-8"))
    payload["endpoints"]["policy_server"] = {"host": "10.20.30.40", "port": 31001}
    payload["endpoints"]["sensor_gateway_metadata"] = {
        "host": "127.0.0.1",
        "port": 31002,
    }
    payload["endpoints"]["control_gateway_dispatch"] = {
        "host": "127.0.0.1",
        "port": 31003,
    }
    payload["endpoints"]["planner_relay"] = {
        "host": "127.0.0.1",
        "port": 31004,
    }
    profile_path = tmp_path / "runtime.yaml"
    profile_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    config = load_inference_launch_config(profile_path)

    vla = build_vla_inference_command(config, Path("/workspace/sonic"))
    gateway = build_sensor_gateway_command(config, Path("/workspace/sonic"))
    control = build_control_gateway_command(config, Path("/workspace/sonic"))

    assert f"--profile {profile_path}" in vla
    assert "--host" not in vla
    assert "--sensor-gateway-endpoint" not in vla
    assert "--control-gateway-endpoint" not in vla
    assert "--planner-relay-zmq-port" not in vla
    assert f"--profile {profile_path}" in gateway
    assert f"--profile {profile_path}" in control


def test_control_gateway_uses_only_typed_gateway_ports() -> None:
    command = build_control_gateway_command(
        InferenceLaunchConfig(),
        Path("/workspace/sonic"),
    )

    assert "python -m gear_sonic.runtime.gateway.services.control" in command
    assert "5580" not in command
    assert "--profile" in command
    assert "--intent-port" not in command
    assert "--dispatch-port" not in command
    assert "--status-port" not in command
    assert "5562" not in command

    console = build_operator_console_command(
        InferenceLaunchConfig(),
        Path("/workspace/sonic"),
    )
    assert "python -m gear_sonic.utils.operator.console" in console
    assert "--profile" in console
    assert "--port" not in console


def test_data_exporter_uses_runtime_profile_gateways_only(tmp_path: Path) -> None:
    config = _profile_config(
        tmp_path,
        components={
            "data_exporter": {
                "task_prompt": "collect a demo",
                "dataset_name": "session one",
                "record_chest_camera": True,
            }
        },
    )
    command = build_data_exporter_command(
        config,
        Path("/workspace/sonic"),
    )

    assert "-m gear_sonic.utils.data_collection.service" in command
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
    assert launcher["lidar_ready_timeout_s"] == config.lidar_ready_timeout_s
    assert launcher["navigation_ready_timeout_s"] == config.navigation_ready_timeout_s
    assert set(config.__dict__) == {
        "sim",
        "opencv_viewer",
        "keyboard_planner",
        "planner_input",
        "depth_anything_ready_timeout_s",
        "base_pose_enabled",
        "slam_debug",
        "lidar_ready_timeout_s",
        "navigation_ready_timeout_s",
        "data_exporter",
        "config",
    }
    assert vla["action_horizon"] == load_inference_config().action_horizon
    assert profile.component("navdp")["sensor_gateway_poll_hz"] == 20.0


def test_navdp_stack_commands_use_ros_topics_and_official_xnavdp_server() -> None:
    config = InferenceLaunchConfig()
    planner = build_navdp_planner_command(config, Path("/workspace/sonic"))
    planner_pane = build_navdp_planner_pane_command(
        config, Path("/workspace/sonic")
    )
    server = build_navdp_server_command(config)
    background_server = build_navdp_server_background_command(config)
    fastlio_supervisor = build_fastlio_supervisor_command(config)

    assert "-m gear_sonic.utils.inference.navdp.service" in planner
    assert "--sensor-input" not in planner
    assert f"--profile {default_launch_config_path()}" in planner
    assert "--visualization-gateway-endpoint" not in planner
    assert "--sensor-gateway-endpoint" not in planner
    assert "--navdp-request-timeout-s" not in planner
    assert "--control-hz" not in planner
    assert "--mpc-hz" not in planner
    assert "--goal-tolerance-m" not in planner
    assert "--radar-timeout-s" not in planner
    assert "--output-endpoint" not in planner
    assert "source /opt/ros/humble/setup.bash" in planner
    navdp = load_runtime_profile().component("navdp")
    assert navdp["root"] == "/home/user/Project/NavDP/baselines/x-navdp"
    assert str(navdp["checkpoint"]).endswith("/x-navdp_posttrain.ckpt")
    assert "python -m eval.src.policy_server" in server
    assert "--embodiment humanoid" in server
    assert "--real" in server
    assert "--no-visualization" in server
    assert "PYTHONUNBUFFERED=1" in server
    assert "NAVDP_SERVER_PID=$!" in background_server
    assert "[NavDP server:stderr]" in background_server
    assert "logs are routed to this pane" in background_server
    assert "trap _stop_navdp_server EXIT HUP" in background_server
    assert "-m gear_sonic.utils.inference.navdp.service" in planner_pane
    assert "_stop_navdp_server" in planner_pane
    assert 'exit "${NAVDP_PLANNER_STATUS}"' in planner_pane
    subprocess.run(
        ["bash", "-n"],
        input=f"{background_server}\n{planner_pane}\n",
        text=True,
        check=True,
    )
    assert f"--checkpoint {navdp['checkpoint']}" in server
    assert "-m gear_sonic.utils.inference.navdp.slam_supervisor" in fastlio_supervisor
    assert "--config-file mid360.yaml" in fastlio_supervisor
    assert f"--profile {default_launch_config_path()}" in fastlio_supervisor
    assert "--control-gateway-endpoint" not in fastlio_supervisor
    assert "mid360_reasan_open3d.py" not in " ".join((planner, server))


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

    assert "-m gear_sonic.utils.inference.lavira.depth_service" in command
    assert "-m gear_sonic.utils.inference.lavira.depth_service" not in planner_command
    assert "sonic_depth_anything_ready" in planner_command
    assert "--ready-file /tmp/sonic_depth_anything_ready" in command
    assert "$ready_file" not in command
    assert f"--profile {default_launch_config_path()}" in command
    assert "--sensor-gateway-endpoint" not in command
    assert "--control-gateway-endpoint" not in command
    assert "--encoder" not in command
    assert "--input-size" not in command
    assert "--inference-hz" not in command


def test_lavira_uses_only_control_and_sensor_gateways() -> None:
    command = build_planner_input_command(
        InferenceLaunchConfig(planner_input="lavira"),
        Path("/workspace/sonic"),
    )

    assert f"--profile {default_launch_config_path()}" in command
    assert "--sensor-gateway-endpoint" not in command
    assert "--control-gateway-endpoint" not in command
    assert "--control-gateway-intent-endpoint" not in command
    assert "--qwenvl-model" not in command
    assert "--qwenvl-timeout-seconds" not in command
    assert load_lavira_config().qwenvl_model == "qwen3-vl-32b-instruct"
    assert load_lavira_config().qwenvl_timeout_seconds == 180.0
    assert "--vision-backend" not in command
    assert "--model gpt-" not in command
    assert "codex" not in command.lower()
    assert "--host" not in command
    assert "--port 5558" not in command
    assert "--status-port 5559" not in command
    assert "5564" not in command


def test_base_pose_agent_uses_gateway_arbitration_and_fixed_task() -> None:
    config = InferenceLaunchConfig(base_pose_enabled=True)
    command = build_base_pose_agent_command(config, Path("/workspace/sonic"))
    worker = load_base_pose_config()

    assert "-m gear_sonic.utils.inference.base_pose.agent" in command
    assert "--task" not in command
    assert "--target-prompt" not in command
    assert "--surface-prompt" not in command
    assert "--mode" not in command
    assert "--dual-head-camera-stream" not in command
    assert worker.dual_head_depth_stream == "camera/ego_view_depth"
    assert worker.dual_chest_depth_stream == "camera/chest_view_depth"
    assert f"--profile {default_launch_config_path()}" in command
    assert "--sensor-gateway-endpoint" not in command
    assert "--control-gateway-endpoint" not in command
    assert "--control-gateway-intent-endpoint" not in command
    assert ". ./.env.local" not in command
    assert "--qwenvl" not in command
    assert "dual-qwenvl" not in command
    assert "--dual-rgbd-buffer-size" not in command
    assert worker.dual_rgbd_buffer_size == 8
    assert worker.dual_rgbd_poll_hz == 60.0
    assert "--raw-chest-handoff-distance-m" not in command
    assert "--raw-chest-fallback-forward-tolerance-m" not in command
    assert "--raw-chest-fallback-lateral-tolerance-m" not in command
    assert "--raw-head-target-distance-m" not in command
    assert worker.raw_head_target_distance_m == pytest.approx(0.9)
    assert worker.raw_chest_target_distance_m == pytest.approx(0.7)
    assert worker.raw_post_stop_sample_frames == 30
    assert "--vision-backend" not in command
    assert "codex" not in command.lower()
    assert "--port 5558" not in command


def test_base_pose_command_has_no_remote_vision_model_arguments() -> None:
    command = build_base_pose_agent_command(
        InferenceLaunchConfig(base_pose_enabled=True),
        Path("/workspace/sonic"),
    )

    assert ". ./.env.local" not in command
    assert "--vision-backend" not in command
    assert "codex" not in command.lower()
    assert "qwen" not in command.lower()
    assert "--persist-diagnostics" not in command


def test_base_pose_yoloe_command_uses_dual_raw_depth_and_local_model() -> None:
    config = InferenceLaunchConfig(base_pose_enabled=True)
    command = build_base_pose_agent_command(config, Path("/workspace/sonic"))
    executor = build_planner_velocity_executor_command(
        config,
        Path("/workspace/sonic"),
    )

    assert "--mode" not in command
    assert "--dual-head-depth-stream" not in command
    assert "--dual-chest-depth-stream" not in command
    assert "--raw-yoloe-model-path" not in command
    assert "--raw-orientation-telemetry-source" not in command
    assert "--publish-orientation" in executor
    assert "--orientation-output-endpoint" not in executor


def test_orientation_telemetry_is_enabled_for_base_pose() -> None:
    executor = build_planner_velocity_executor_command(
        InferenceLaunchConfig(
            base_pose_enabled=True,
        ),
        Path("/workspace/sonic"),
    )

    assert "--publish-orientation" in executor
    assert "--orientation-output-endpoint" not in executor


def test_runtime_sidecars_are_read_only_and_navdp_uses_gateway_by_default() -> None:
    config = InferenceLaunchConfig()
    root = Path("/workspace/sonic")
    gateway = build_sensor_gateway_command(config, root)
    planner = build_navdp_planner_command(config, root)
    executor = build_planner_velocity_executor_command(config, root)

    assert "python -m gear_sonic.runtime.gateway.services.sensor" in gateway
    assert "python -m gear_sonic.utils.operator.cv_viewer" in gateway
    assert "--control-gateway-endpoint" not in gateway
    assert "--navigation-runtime-status-endpoint" not in gateway
    assert "/tmp/sonic_opencv_viewer.log" in gateway
    assert "-m gear_sonic.utils.inference.lavira.depth_service" in gateway
    assert "/tmp/sonic_depth_anything.log" not in gateway
    assert "msg_MID360_launch.py" in gateway
    assert "-m gear_sonic.utils.inference.navdp.slam_supervisor" in gateway
    assert "--control-gateway-endpoint" not in gateway
    assert "/tmp/sonic_livox_driver.log" in gateway
    assert "/tmp/sonic_fastlio.log" in gateway
    assert "ros2 bag record" not in gateway
    assert f"--profile {default_launch_config_path()}" in gateway
    assert "--camera-host" not in gateway
    assert "--rpc-port" not in gateway
    assert "PYTHONPATH=/workspace/sonic:$PYTHONPATH" in gateway
    assert "cpp_command" not in gateway
    assert "5556" not in gateway
    assert "--sensor-input" not in planner
    assert f"--profile {default_launch_config_path()}" in planner
    assert "--sensor-gateway-endpoint" not in planner
    assert "runtime_gateway.sensor_service" not in planner
    assert "--output-endpoint" not in planner
    assert "python -m gear_sonic.utils.planner_control.executor_service" in executor
    assert f"--profile {default_launch_config_path()}" in executor
    assert "--navdp-velocity-endpoint" not in executor
    assert "--output-endpoint" not in executor
    assert "--runtime-status-endpoint" not in executor


def test_base_pose_uses_raw_chest_depth_without_starting_depth_anything() -> None:
    command = build_sensor_gateway_command(
        InferenceLaunchConfig(
            planner_input="keyboard",
            base_pose_enabled=True,
        ),
        Path("/workspace/sonic"),
    )

    assert "-m gear_sonic.utils.inference.lavira.depth_service" not in command
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
    assert "--control-gateway-endpoint" not in gateway
    assert "--sensor-gateway-endpoint" not in gateway
    assert "--sensor-gateway-poll-hz" not in gateway
    assert "--planner-relay-zmq-port" not in gateway
    assert "--action-zmq-port" not in gateway


def test_navdp_is_a_pure_velocity_producer_for_the_shared_executor() -> None:
    config = InferenceLaunchConfig()

    planner = build_navdp_planner_command(config, Path("/workspace/sonic"))

    assert "--sensor-input" not in planner
    assert "--camera-host" not in planner
    assert "--camera-port" not in planner
    assert f"--profile {default_launch_config_path()}" in planner
    assert "--sensor-gateway-endpoint" not in planner
    assert "--navdp-server" not in planner
    assert "--output-endpoint" not in planner
    assert "--radar-timeout-s" not in planner


def test_launcher_runs_readiness_gate_synchronously(monkeypatch) -> None:
    commands: list[list[str]] = []

    class Result:
        returncode = 0

    def run(command, **_kwargs):
        commands.append(command)
        return Result()

    monkeypatch.setattr("gear_sonic.scripts.launch_inference.subprocess.run", run)
    config = InferenceLaunchConfig(lidar_ready_timeout_s=45.0)

    assert run_readiness_gate(config, Path("/workspace/sonic"), "lidar")
    shell = commands[0][-1]
    assert "-m gear_sonic.utils.inference.navdp.readiness --stage lidar" in shell
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
    assert f"--profile {default_launch_config_path()}" in shell
    assert "--sensor-gateway-port" not in shell
