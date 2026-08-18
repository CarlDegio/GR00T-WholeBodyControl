"""All-in-one tmux launcher for SONIC VLA inference.

The inference window contains the core deploy, operator, VLA, navigation, and
gateway panes, plus an optional Base-Pose agent pane. Simulation and data
collection use optional additional windows.

Prerequisites:
    - tmux installed (sudo apt install tmux)
    - Virtual environments set up:
        bash install_scripts/install_inference.sh     -> .venv_inference
        bash install_scripts/install_data_collection.sh -> .venv_data_collection (optional, for recording)
    - gear_sonic_deploy built (see docs)
    - OpenPI policy server running separately

Usage (from repo root — no venv activation needed):
    python gear_sonic/scripts/launch_inference.py                        # real robot
    python gear_sonic/scripts/launch_inference.py --sim                  # MuJoCo sim
    python gear_sonic/scripts/launch_inference.py --no-data-exporter     # no recording pane
"""

import argparse
from dataclasses import dataclass, fields
from pathlib import Path
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Literal, get_args, get_origin, get_type_hints


def _bootstrap_venv():
    """Re-exec with the inference Python if tyro is not available."""
    try:
        import tyro  # noqa: F401
        return
    except ImportError:
        pass

    repo_root = Path(__file__).resolve().parent.parent.parent
    venv_python = repo_root / ".venv_inference" / "bin" / "python"
    if not venv_python.exists():
        print(
            "ERROR: tyro is not installed and .venv_inference not found.\n"
            "  Run: bash install_scripts/install_inference.sh"
        )
        sys.exit(1)

    print(f"Re-launching with {venv_python} ...")
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)


_bootstrap_venv()

import tyro
import yaml

from gear_sonic.runtime.config import load_runtime_profile  # noqa: E402


def default_launch_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "launch_inference.yaml"


def _get_local_ip() -> str:
    """Best-effort detection of the PC's LAN IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown"


@dataclass
class InferenceLaunchConfig:
    """CLI config for the all-in-one VLA inference tmux launcher."""

    # Deployment mode
    sim: bool = False
    """Run against MuJoCo sim instead of real robot."""

    # C++ deploy options
    deploy_input_type: str = "zmq_manager"
    """Input type for the C++ deploy."""

    deploy_zmq_host: str = "localhost"
    """ZMQ host for the C++ deploy to listen on."""

    deploy_policy_variant: Literal["default", "low_latency", "sonic_v1_1"] = "default"
    """Bundled deploy checkpoint and observation-config preset."""

    deploy_checkpoint: str = ""
    """Checkpoint path for deploy.sh. Leave empty for default."""

    deploy_obs_config: str = ""
    """Observation config file for deploy.sh. Leave empty for default."""

    deploy_planner: str = ""
    """Planner model path for deploy.sh. Leave empty for default."""

    deploy_motion_data: str = ""
    """Motion data path for deploy.sh. Leave empty for default."""

    deploy_output_type: str = ""
    """Output type for deploy.sh. Leave empty for default."""

    # VLA inference options
    policy_host: str = "localhost"
    """OpenPI policy server host."""

    policy_port: int = 29999
    """OpenPI policy server port."""

    embodiment_tag: str = "unitree_g1_sonic"
    """Embodiment tag for policy inference."""

    prompt: str = "demo"
    """Language prompt for inference."""

    action_publish_rate: int = 50
    """Rate at which individual actions are published to the C++ control loop (Hz)."""

    action_horizon: int = 50
    """Action horizon of the VLA policy."""

    # Camera
    camera_host: str = "localhost"
    """Camera server host."""

    camera_port: int = 5555
    """Camera server port."""

    opencv_viewer: bool = True
    """Start the standalone OpenCV visualization process under the SensorGateway pane."""

    sensor_gateway_port: int = 5560
    """Local SensorGateway Snapshot and health REP port."""

    sensor_gateway_visualization_port: int = 5566
    """Rendered-panel ingress owned by SensorGateway."""

    vla_timing_port: int = 5567
    """Best-effort VLA timing ingress owned by SensorGateway."""

    control_gateway_intent_port: int = 5561
    """Structured operator-intent mirror published by ControlGateway."""

    control_gateway_status_port: int = 5562
    """ControlGateway acknowledgement/status stream."""

    control_gateway_dispatch_port: int = 5565
    """Validated structured commands published to migrated consumers."""

    vla_sensor_gateway_poll_hz: float = 50.0
    """Background Gateway cache update rate used by VLA."""

    vla_sensor_gateway_request_timeout_ms: int = 100
    """VLA local SensorGateway metadata request deadline."""

    vla_sensor_gateway_max_age_ms: float = 1000.0
    """Maximum accepted age for VLA Gateway frames and caches."""

    vla_sensor_gateway_max_skew_ms: float = 5.0
    """Maximum receive-time skew among VLA's four RGB streams."""

    keyboard_planner: bool = True
    """Start planner input, rule-based safety, and MID-360 sidecars."""

    keyboard_planner_port: int = 5558
    """Keyboard planner sidecar port."""

    keyboard_planner_publish_rate: int = 20
    """Keyboard planner publish rate (Hz)."""

    keyboard_planner_host: str = "localhost"
    """Keyboard planner sidecar host."""

    planner_input: Literal["keyboard", "lavira"] = "lavira"
    """Navigation keyboard and semantic target source used in pane 3."""

    lavira_mission: str = ""
    """Mission sent to LaViRA when --planner-input lavira is selected."""

    lavira_global_target: str = ""
    """Global target used by LaViRA when --planner-input lavira is selected."""

    lavira_model: str = "gpt-5.6-luna"
    """Codex CLI vision model used by LaViRA."""

    lavira_vision_backend: Literal["codex", "qwenvl"] = "codex"
    """Vision provider used by LaViRA target recognition."""

    lavira_qwenvl_model: str = "qwen3-vl-32b-instruct"
    """DashScope Qwen-VL model used when the backend is qwenvl."""

    lavira_qwenvl_base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    """DashScope OpenAI-compatible endpoint; key comes from DASHSCOPE_API_KEY."""

    lavira_warmup: bool = True
    """Run one discarded Luna request in the background at startup."""

    lavira_debug: bool = False
    """Enable LaViRA debug logging."""

    lavira_host: str = "*"
    """LaViRA REASAN publisher bind host."""

    lavira_control_input: Literal["tty", "gateway"] = "gateway"
    """LaViRA navigation keys come from its own pane or ControlGateway."""

    lavira_camera_timeout_ms: int = 15000
    """LaViRA pure LingBot RGB-D receive timeout (ms)."""

    lavira_depth_port: int = 5564
    """Local pure LingBot completed RGB-D stream consumed by LaViRA."""

    lingbot_ready_timeout: float = 180.0
    """Maximum time to wait for the independent LingBot pane to publish its first frame."""

    lavira_codex_timeout_seconds: float = 180.0
    """LaViRA Codex policy timeout (s)."""

    lavira_min_confidence: float = 0.6
    """Minimum Codex target confidence accepted by LaViRA."""

    lavira_output_root: str = "outputs/object_nav"
    """Directory where LaViRA stores ObjectNav diagnostics."""

    base_pose_enabled: bool = False
    """Start the independently triggered Base-Pose navigation agent."""

    base_pose_task: str = ""
    """Fixed manipulation task used whenever B starts Base-Pose alignment."""

    base_pose_mode: Literal[
        "raw_yoloe_servo",
        "dual_raw_yoloe_servo",
    ] = "dual_raw_yoloe_servo"
    """Agent-near raw-depth YOLOE mode."""

    base_pose_vision_backend: Literal["codex", "qwenvl"] = "codex"
    base_pose_model: str = "gpt-5.6-sol"
    base_pose_qwenvl_model: str = "qwen3-vl-plus"
    base_pose_qwenvl_base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    base_pose_qwenvl_thinking_budget: int = 500
    base_pose_reasoning_effort: str = "xhigh"
    base_pose_codex_fast: bool = True
    base_pose_codex_timeout_seconds: float = 600.0
    base_pose_camera_stream: str = "ego_view"
    base_pose_yoloe_depth_stream: str = "camera/ego_view_depth"
    base_pose_camera_intrinsics_path: str = "gear_sonic/config/camera_intrinsics.json"
    base_pose_camera_pitch_deg: float = -38.0
    base_pose_camera_roll_deg: float = 0.0
    base_pose_camera_yaw_deg: float = 0.0
    base_pose_camera_forward_offset_m: float = 0.0
    base_pose_camera_lateral_offset_m: float = 0.0
    base_pose_dual_head_camera_stream: str = "ego_view"
    base_pose_dual_head_depth_stream: str = "camera/ego_view_depth"
    base_pose_dual_chest_camera_stream: str = "chest_view"
    base_pose_dual_chest_depth_stream: str = "camera/chest_view_depth"
    base_pose_dual_chest_camera_pitch_deg: float = -3.0
    base_pose_dual_chest_camera_roll_deg: float = 0.0
    base_pose_dual_chest_camera_yaw_deg: float = 0.0
    base_pose_dual_chest_camera_forward_offset_m: float = 0.0
    base_pose_dual_chest_camera_lateral_offset_m: float = 0.0
    base_pose_dual_match_tolerance_frames: int = 30
    base_pose_dual_initialization_grace_s: float = 30.0
    base_pose_dual_qwenvl_fallback_model: str = "qwen3-vl-8b-instruct"
    base_pose_camera_timeout_ms: int = 15000
    base_pose_planner_hz: float = 20.0
    base_pose_output_root: str = "outputs/base_pose_adjustment"
    base_pose_yoloe_model_path: str = "tools/yoloe26m/weights/yoloe-26m-seg.pt"
    base_pose_yoloe_device: str = "0"
    base_pose_yoloe_confidence: float = 0.25
    base_pose_yoloe_imgsz: int = 640
    base_pose_raw_reference_update_interval_frames: int = 5
    base_pose_raw_reference_update_min_confidence: float = 0.35
    base_pose_raw_reference_update_min_iou: float = 0.50
    base_pose_raw_servo_hz: float = 10.0
    base_pose_raw_head_target_distance_m: float = 0.90
    base_pose_raw_chest_target_distance_m: float = 0.80
    base_pose_raw_forward_tolerance_m: float = 0.10
    base_pose_raw_lateral_tolerance_m: float = 0.10
    base_pose_raw_min_linear_speed_m_s: float = 0.40
    base_pose_raw_max_lateral_speed_m_s: float = 0.40
    base_pose_raw_min_yaw_speed_rad_s: float = 0.10
    base_pose_raw_yaw_tolerance_deg: float = 8.0
    base_pose_raw_yaw_coarse_speed_rad_s: float = 0.30
    base_pose_raw_yaw_trim_speed_rad_s: float = 0.20
    base_pose_raw_forward_recenter_yaw_speed_rad_s: float = 0.30
    base_pose_raw_horizontal_guard_fraction: float = 0.25
    base_pose_raw_horizontal_recovery_fraction: float = 0.30
    base_pose_raw_camera_stale_s: float = 0.40
    base_pose_raw_max_run_s: float = 180.0
    base_pose_raw_post_stop_sample_s: float = 3.0
    base_pose_raw_allow_missing_table: Literal[0, 1] = 0

    navdp_root: str = "/home/user/Project/NavDP/baselines/x-navdp"
    navdp_checkpoint: str = (
        "/home/user/Project/NavDP/baselines/x-navdp/checkpoints/x-navdp_posttrain.ckpt"
    )
    fastlio_workspace: str = "/home/user/Project/fastlio_humanoid_ws"
    livox_sdk_lib: str = "/home/user/Project/livox_sdk2_install/lib"
    fastlio_config: str = "mid360.yaml"
    slam_debug: bool = False
    """Record raw LiDAR/IMU and FAST-LIO outputs for each real-robot run."""

    lidar_ready_timeout: float = 30.0
    navigation_ready_timeout: float = 60.0
    # Data exporter (optional recording during inference)
    data_exporter: bool = True
    """Start the data exporter pane for recording during inference."""

    record_chest_camera: bool = False
    """Record chest camera stream (chest_view) in the dataset."""

    task_prompt: str = ""
    """Task prompt for the data exporter. Defaults to the inference prompt if empty."""

    dataset_name: str = ""
    """Dataset name for the data exporter. Leave empty to auto-generate."""

    config: str = str(default_launch_config_path())
    """YAML file containing the complete launch_inference configuration."""


def _validated_launch_values(raw_values: dict[str, Any]) -> dict[str, Any]:
    config_fields = {
        item.name: item for item in fields(InferenceLaunchConfig) if item.name != "config"
    }
    unknown = set(raw_values) - set(config_fields)
    missing = set(config_fields) - set(raw_values)
    if unknown:
        raise ValueError(
            "unknown launch_inference YAML fields: " + ", ".join(sorted(unknown))
        )
    if missing:
        raise ValueError(
            "missing launch_inference YAML fields: " + ", ".join(sorted(missing))
        )

    annotations = get_type_hints(InferenceLaunchConfig)
    values: dict[str, Any] = {}
    for name, value in raw_values.items():
        annotation = annotations[name]
        origin = get_origin(annotation)
        if origin is Literal:
            allowed = get_args(annotation)
            if value not in allowed:
                raise ValueError(
                    f"launch_inference.{name} must be one of {allowed}, got {value!r}"
                )
            values[name] = value
        elif annotation is bool:
            if not isinstance(value, bool):
                raise ValueError(f"launch_inference.{name} must be a boolean")
            values[name] = value
        elif annotation is int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"launch_inference.{name} must be an integer")
            values[name] = value
        elif annotation is float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"launch_inference.{name} must be a number")
            values[name] = float(value)
        elif annotation is str:
            if not isinstance(value, str):
                raise ValueError(f"launch_inference.{name} must be a string")
            values[name] = value
        else:
            raise TypeError(f"unsupported launch configuration type for {name}: {annotation}")
    return values


def load_inference_launch_config(path: str | Path | None = None) -> InferenceLaunchConfig:
    config_path = default_launch_config_path() if path is None else Path(path).expanduser()
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid launch YAML {config_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"launch YAML must contain an object: {config_path}")
    if payload.get("schema") != "sonic.runtime_profile" or payload.get("version") != 1:
        raise ValueError("launch YAML must use sonic.runtime_profile version 1")
    raw_values = payload.get("launch_inference")
    if not isinstance(raw_values, dict):
        raise ValueError("launch YAML must contain a launch_inference object")
    migrated_values = dict(raw_values)
    migrated_values.setdefault("deploy_policy_variant", "default")
    values = _validated_launch_values(migrated_values)
    return InferenceLaunchConfig(config=str(config_path), **values)


def parse_inference_launch_config(
    args: list[str] | None = None,
) -> InferenceLaunchConfig:
    argv = list(sys.argv[1:] if args is None else args)
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument(
        "--config",
        default=str(default_launch_config_path()),
    )
    bootstrap_args, _ = bootstrap.parse_known_args(argv)
    yaml_defaults = load_inference_launch_config(bootstrap_args.config)
    return tyro.cli(InferenceLaunchConfig, args=argv, default=yaml_defaults)


SESSION_NAME = "sonic_inference"
LINGBOT_READY_FILE = Path("/tmp/sonic_lingbot_ready")


def _runtime_profile(config: InferenceLaunchConfig):
    return load_runtime_profile(config.config)


DEPLOY_POLICY_PRESETS = {
    "default": (
        "policy/release/model",
        "policy/release/observation_config.yaml",
        "python download_from_hf.py",
    ),
    "low_latency": (
        "policy/low_latency/model",
        "policy/low_latency/observation_config.yaml",
        "python download_from_hf.py --low-latency",
    ),
    "sonic_v1_1": (
        "policy/sonic_v1_1/model",
        "policy/sonic_v1_1/observation_config.yaml",
        "python download_from_hf.py --sonic-v1-1",
    ),
}


def _deploy_asset_path(deploy_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else deploy_dir / path


def _normalized_deploy_argument(value: str) -> str:
    path = Path(value).expanduser()
    return str(path) if path.is_absolute() else value


def resolve_deploy_policy(
    config: InferenceLaunchConfig,
    deploy_dir: Path,
) -> tuple[str, str]:
    """Resolve and validate a paired deploy checkpoint and observation config."""
    has_checkpoint = bool(config.deploy_checkpoint.strip())
    has_obs_config = bool(config.deploy_obs_config.strip())
    if has_checkpoint != has_obs_config:
        raise ValueError(
            "deploy_checkpoint and deploy_obs_config must be set together"
        )

    if has_checkpoint:
        checkpoint = _normalized_deploy_argument(config.deploy_checkpoint.strip())
        obs_config = _normalized_deploy_argument(config.deploy_obs_config.strip())
        download_hint = "provide the matching custom deployment files"
    else:
        checkpoint, obs_config, download_hint = DEPLOY_POLICY_PRESETS[
            config.deploy_policy_variant
        ]

    required = (
        _deploy_asset_path(deploy_dir, f"{checkpoint}_encoder.onnx"),
        _deploy_asset_path(deploy_dir, f"{checkpoint}_decoder.onnx"),
        _deploy_asset_path(deploy_dir, obs_config),
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "deploy policy files are missing:\n"
            f"{formatted}\n"
            f"Download them with: {download_hint}"
        )
    return checkpoint, obs_config


def build_deploy_command(config: InferenceLaunchConfig, repo_root: Path) -> str:
    """Build the C++ deploy command from a validated policy selection."""
    deploy_dir = repo_root / "gear_sonic_deploy"
    checkpoint, obs_config = resolve_deploy_policy(config, deploy_dir)
    deploy_mode = "sim" if config.sim else "real"
    command = (
        f"cd {shlex.quote(str(deploy_dir))} && "
        f"./deploy.sh "
        f"--input-type {shlex.quote(config.deploy_input_type)} "
        f"--zmq-host {shlex.quote(config.deploy_zmq_host)} "
        f"--hand-type dex1 "
        f"--dex1-kp 1.0 "
        f"--dex1-kd 0.05 "
        f"--cp {shlex.quote(checkpoint)} "
        f"--obs-config {shlex.quote(obs_config)} "
    )
    if config.deploy_planner:
        command += f"--planner {shlex.quote(config.deploy_planner)} "
    if config.deploy_motion_data:
        command += f"--motion-data {shlex.quote(config.deploy_motion_data)} "
    if config.deploy_output_type:
        command += f"--output-type {shlex.quote(config.deploy_output_type)} "
    return command + deploy_mode


def build_vla_inference_command(
    config: InferenceLaunchConfig,
    repo_root: Path,
) -> str:
    """Build VLA command with an explicit, manually selected sensor source."""
    planner_relay_port = _runtime_profile(config).endpoint("planner_relay").port
    return (
        f"cd {shlex.quote(str(repo_root))} && "
        "source .venv_inference/bin/activate && "
        "python gear_sonic/scripts/run_vla_inference.py "
        f"--host {shlex.quote(config.policy_host)} "
        f"--port {config.policy_port} "
        f"--embodiment-tag {shlex.quote(config.embodiment_tag)} "
        f"--prompt {shlex.quote(config.prompt)} "
        f"--action-publish-rate {config.action_publish_rate} "
        f"--action-horizon {config.action_horizon} "
        f"--sensor-gateway-endpoint tcp://127.0.0.1:{config.sensor_gateway_port} "
        f"--sensor-gateway-poll-hz {config.vla_sensor_gateway_poll_hz} "
        f"--sensor-gateway-request-timeout-ms "
        f"{config.vla_sensor_gateway_request_timeout_ms} "
        f"--sensor-gateway-max-age-ms {config.vla_sensor_gateway_max_age_ms} "
        f"--sensor-gateway-max-skew-ms {config.vla_sensor_gateway_max_skew_ms} "
        f"--timing-endpoint tcp://127.0.0.1:{config.vla_timing_port} "
        f"--control-gateway-endpoint tcp://127.0.0.1:"
        f"{config.control_gateway_dispatch_port} "
        "--planner-relay-zmq-host localhost "
        f"--planner-relay-zmq-port {planner_relay_port}"
    )


def build_control_gateway_command(
    config: InferenceLaunchConfig,
    repo_root: Path,
) -> str:
    """Build the typed ControlGateway router command."""
    navigation_status_port = _runtime_profile(config).endpoint("navigation_status").port

    return (
        f"cd {shlex.quote(str(repo_root))} && "
        "source .venv_inference/bin/activate && "
        "python gear_sonic/scripts/run_control_gateway.py "
        f"--intent-port {config.control_gateway_intent_port} "
        f"--dispatch-port {config.control_gateway_dispatch_port} "
        f"--status-port {config.control_gateway_status_port} "
        f"--navigation-port {config.keyboard_planner_port} "
        f"--navigation-status-port {navigation_status_port}"
    )


def build_data_exporter_command(
    config: InferenceLaunchConfig,
    repo_root: Path,
) -> str:
    """Build DataExporter with runtime-profile Gateway inputs only."""
    task_prompt = config.task_prompt or config.prompt
    command = (
        f"cd {shlex.quote(str(repo_root))} && "
        "source .venv_data_collection/bin/activate && "
        "python gear_sonic/scripts/run_data_exporter.py "
        f"--profile {shlex.quote(config.config)} "
        f"--task-prompt {shlex.quote(task_prompt)}"
    )
    if config.dataset_name:
        command += f" --dataset-name {shlex.quote(config.dataset_name)}"
    if config.record_chest_camera:
        command += " --record-chest-camera"
    return command


def build_operator_console_command(
    config: InferenceLaunchConfig,
    repo_root: Path,
) -> str:
    return (
        f"cd {shlex.quote(str(repo_root))} && "
        "source .venv_inference/bin/activate && "
        "python gear_sonic/scripts/run_operator_console.py "
        f"--host 127.0.0.1 --port {config.control_gateway_intent_port}"
    )


def build_operator_interface_command(
    config: InferenceLaunchConfig,
    repo_root: Path,
) -> str:
    return build_operator_console_command(config, repo_root)


def build_planner_input_command(config: InferenceLaunchConfig, repo_root: Path) -> str:
    """Build the pane 3 command for the selected REASEN command source."""
    if config.planner_input == "keyboard":
        return (
            f"cd {repo_root} && "
            f"source .venv_teleop/bin/activate && "
            f"python gear_sonic/scripts/keyboard_planner_thread_server.py "
            f"--port {config.keyboard_planner_port} "
            f"--hz {config.keyboard_planner_publish_rate} "
            f"--host {config.keyboard_planner_host} "
        )

    debug = "--debug " if config.lavira_debug else ""
    warmup = "" if config.lavira_warmup else "--no-warmup "
    vision_backend = ""
    local_env = ""
    if config.lavira_vision_backend == "qwenvl":
        local_env = "set -a; [ ! -f .env.local ] || . ./.env.local; set +a; "
        vision_backend = (
            "--vision-backend qwenvl "
            f"--qwenvl-model {shlex.quote(config.lavira_qwenvl_model)} "
            f"--qwenvl-base-url {shlex.quote(config.lavira_qwenvl_base_url)} "
        )
    quoted_root = shlex.quote(str(repo_root))
    ready_file = shlex.quote(str(LINGBOT_READY_FILE))
    return (
        f"cd {quoted_root} && "
        f"{local_env}"
        f"timeout {config.lingbot_ready_timeout}s sh -c "
        f"'while [ ! -f {ready_file} ]; do sleep 0.2; done' || "
        "{ echo '[LaViRA] LingBot did not become ready' >&2; exit 1; }; "
        f".venv_inference/bin/python gear_sonic/scripts/lavira_planner.py "
        f"--mission {shlex.quote(config.lavira_mission)} "
        f"--global-target {shlex.quote(config.lavira_global_target)} "
        f"--model {shlex.quote(config.lavira_model)} "
        f"{vision_backend}{debug}{warmup}"
        f"--camera-timeout-ms {config.lavira_camera_timeout_ms} "
        f"--sensor-gateway-endpoint tcp://127.0.0.1:{config.sensor_gateway_port} "
        f"--sensor-gateway-request-timeout-ms "
        f"{config.vla_sensor_gateway_request_timeout_ms} "
        f"--sensor-gateway-max-age-ms {config.vla_sensor_gateway_max_age_ms} "
        f"--sensor-gateway-max-skew-ms {config.vla_sensor_gateway_max_skew_ms} "
        f"--codex-timeout-seconds {config.lavira_codex_timeout_seconds} "
        f"--min-confidence {config.lavira_min_confidence} "
        "--control-gateway-endpoint "
        f"tcp://127.0.0.1:{config.control_gateway_dispatch_port} "
        "--control-gateway-intent-endpoint "
        f"tcp://127.0.0.1:{config.control_gateway_intent_port} "
        f"--output-root {shlex.quote(config.lavira_output_root)}"
    )


def build_base_pose_agent_command(
    config: InferenceLaunchConfig, repo_root: Path
) -> str:
    """Build the independent ControlGateway-backed Base-Pose agent."""

    quoted_root = shlex.quote(str(repo_root))
    local_env = ""
    backend = f"--vision-backend {config.base_pose_vision_backend} "
    if config.base_pose_vision_backend == "qwenvl":
        local_env = "set -a; [ ! -f .env.local ] || . ./.env.local; set +a; "
        backend += (
            f"--qwenvl-model {shlex.quote(config.base_pose_qwenvl_model)} "
            f"--qwenvl-base-url {shlex.quote(config.base_pose_qwenvl_base_url)} "
            f"--qwenvl-thinking-budget {config.base_pose_qwenvl_thinking_budget} "
        )
    codex_fast = "" if config.base_pose_codex_fast else "--no-codex-fast "
    dual = ""
    if config.base_pose_mode == "dual_raw_yoloe_servo":
        dual = (
            f"--dual-head-camera-stream "
            f"{shlex.quote(config.base_pose_dual_head_camera_stream)} "
            f"--dual-head-depth-stream "
            f"{shlex.quote(config.base_pose_dual_head_depth_stream)} "
            f"--dual-chest-camera-stream "
            f"{shlex.quote(config.base_pose_dual_chest_camera_stream)} "
            f"--dual-chest-depth-stream "
            f"{shlex.quote(config.base_pose_dual_chest_depth_stream)} "
            f"--dual-chest-camera-pitch-deg "
            f"{config.base_pose_dual_chest_camera_pitch_deg} "
            f"--dual-chest-camera-roll-deg "
            f"{config.base_pose_dual_chest_camera_roll_deg} "
            f"--dual-chest-camera-yaw-deg "
            f"{config.base_pose_dual_chest_camera_yaw_deg} "
            f"--dual-chest-camera-forward-offset-m "
            f"{config.base_pose_dual_chest_camera_forward_offset_m} "
            f"--dual-chest-camera-lateral-offset-m "
            f"{config.base_pose_dual_chest_camera_lateral_offset_m} "
            f"--dual-match-tolerance-frames "
            f"{config.base_pose_dual_match_tolerance_frames} "
            f"--dual-initialization-grace-s "
            f"{config.base_pose_dual_initialization_grace_s} "
            f"--dual-qwenvl-fallback-model "
            f"{shlex.quote(config.base_pose_dual_qwenvl_fallback_model)} "
        )
    yoloe = (
        f"--camera-intrinsics-path {shlex.quote(config.base_pose_camera_intrinsics_path)} "
        f"--camera-pitch-deg {config.base_pose_camera_pitch_deg} "
        f"--camera-roll-deg {config.base_pose_camera_roll_deg} "
        f"--camera-yaw-deg {config.base_pose_camera_yaw_deg} "
        f"--camera-forward-offset-m {config.base_pose_camera_forward_offset_m} "
        f"--camera-lateral-offset-m {config.base_pose_camera_lateral_offset_m} "
        f"{dual}"
        f"--raw-yoloe-model-path {shlex.quote(config.base_pose_yoloe_model_path)} "
        f"--raw-yoloe-device {shlex.quote(config.base_pose_yoloe_device)} "
        f"--raw-yoloe-confidence {config.base_pose_yoloe_confidence} "
        f"--raw-yoloe-imgsz {config.base_pose_yoloe_imgsz} "
        f"--raw-reference-update-interval-frames "
        f"{config.base_pose_raw_reference_update_interval_frames} "
        f"--raw-reference-update-min-confidence "
        f"{config.base_pose_raw_reference_update_min_confidence} "
        f"--raw-reference-update-min-iou "
        f"{config.base_pose_raw_reference_update_min_iou} "
        f"--raw-servo-hz {config.base_pose_raw_servo_hz} "
        f"--raw-head-target-distance-m {config.base_pose_raw_head_target_distance_m} "
        f"--raw-chest-target-distance-m "
        f"{config.base_pose_raw_chest_target_distance_m} "
        f"--raw-forward-tolerance-m {config.base_pose_raw_forward_tolerance_m} "
        f"--raw-lateral-tolerance-m {config.base_pose_raw_lateral_tolerance_m} "
        f"--raw-min-linear-speed-m-s {config.base_pose_raw_min_linear_speed_m_s} "
        f"--raw-max-lateral-speed-m-s {config.base_pose_raw_max_lateral_speed_m_s} "
        f"--raw-min-yaw-speed-rad-s {config.base_pose_raw_min_yaw_speed_rad_s} "
        f"--raw-yaw-tolerance-deg {config.base_pose_raw_yaw_tolerance_deg} "
        f"--raw-yaw-coarse-speed-rad-s {config.base_pose_raw_yaw_coarse_speed_rad_s} "
        f"--raw-yaw-trim-speed-rad-s {config.base_pose_raw_yaw_trim_speed_rad_s} "
        f"--raw-forward-recenter-yaw-speed-rad-s "
        f"{config.base_pose_raw_forward_recenter_yaw_speed_rad_s} "
        f"--raw-horizontal-guard-fraction "
        f"{config.base_pose_raw_horizontal_guard_fraction} "
        f"--raw-horizontal-recovery-fraction "
        f"{config.base_pose_raw_horizontal_recovery_fraction} "
        f"--raw-camera-stale-s {config.base_pose_raw_camera_stale_s} "
        f"--raw-max-run-s {config.base_pose_raw_max_run_s} "
        f"--raw-post-stop-sample-s {config.base_pose_raw_post_stop_sample_s} "
        f"--raw-allow-missing-table {config.base_pose_raw_allow_missing_table} "
        "--raw-orientation-telemetry-source "
        f"{shlex.quote(_runtime_profile(config).endpoint_uri('orientation_telemetry'))} "
    )
    return (
        f"cd {quoted_root} && {local_env}"
        ".venv_inference/bin/python gear_sonic/scripts/base_pose_agent.py "
        f"--task {shlex.quote(config.base_pose_task)} "
        f"--mode {config.base_pose_mode} {backend}"
        f"--model {shlex.quote(config.base_pose_model)} "
        f"--reasoning-effort {shlex.quote(config.base_pose_reasoning_effort)} "
        f"{codex_fast}"
        f"--codex-timeout-seconds {config.base_pose_codex_timeout_seconds} "
        f"--camera-stream {shlex.quote(config.base_pose_camera_stream)} "
        f"--depth-stream {shlex.quote(config.base_pose_yoloe_depth_stream)} "
        f"--camera-timeout-ms {config.base_pose_camera_timeout_ms} "
        f"--sensor-gateway-endpoint tcp://127.0.0.1:{config.sensor_gateway_port} "
        f"--sensor-gateway-request-timeout-ms {config.vla_sensor_gateway_request_timeout_ms} "
        f"--sensor-gateway-max-age-ms {config.vla_sensor_gateway_max_age_ms} "
        f"--sensor-gateway-max-skew-ms {config.vla_sensor_gateway_max_skew_ms} "
        f"--control-gateway-endpoint tcp://127.0.0.1:{config.control_gateway_dispatch_port} "
        f"--control-gateway-intent-endpoint tcp://127.0.0.1:{config.control_gateway_intent_port} "
        f"--planner-hz {config.base_pose_planner_hz} "
        f"{yoloe}"
        f"--output-root {shlex.quote(config.base_pose_output_root)}"
    )


def build_lingbot_command(config: InferenceLaunchConfig, repo_root: Path) -> str:
    """Build the independent background LingBot depth-completion process."""
    quoted_root = shlex.quote(str(repo_root))
    ready_file = shlex.quote(str(LINGBOT_READY_FILE))
    return (
        f"PYTHONPATH={quoted_root} {quoted_root}/.venv_lingbot_depth/bin/python "
        "gear_sonic/scripts/run_lingbot_depth_viewer.py "
        f"--sensor-gateway-endpoint tcp://127.0.0.1:{config.sensor_gateway_port} "
        f"--publish-port {config.lavira_depth_port} --ready-file {ready_file} "
        "--no-visualize --visualization-gateway-endpoint "
        f"tcp://127.0.0.1:{config.sensor_gateway_visualization_port} "
        "--control-gateway-endpoint tcp://127.0.0.1:"
        f"{config.control_gateway_dispatch_port}"
    )


def build_navdp_planner_command(config: InferenceLaunchConfig, repo_root: Path) -> str:
    profile = _runtime_profile(config)
    settings = profile.component("navdp")
    recording = (
        "--record-actorray "
        f"--actorray-output-dir {shlex.quote(str(settings['actorray_output_dir']))} "
        f"--actorray-record-fps {settings['actorray_record_fps']} "
        if settings["record_actorray"]
        else ""
    )
    navigation_status = profile.endpoint("navigation_status")
    navdp_velocity = profile.endpoint("navdp_velocity")
    return (
        "unset COLCON_CURRENT_PREFIX AMENT_PREFIX_PATH CMAKE_PREFIX_PATH; "
        "source /opt/ros/humble/setup.bash && "
        f"source {shlex.quote(config.fastlio_workspace)}/install/setup.bash && "
        f"cd {shlex.quote(str(repo_root))} && source .venv_teleop/bin/activate && "
        "python gear_sonic/scripts/navdp_planner.py "
        f"--command-endpoint {shlex.quote(profile.endpoint_uri('navigation_command'))} "
        f"--status-endpoint {shlex.quote(f'tcp://*:{navigation_status.port}')} "
        f"--output-endpoint {shlex.quote(f'tcp://*:{navdp_velocity.port}')} "
        f"--sensor-gateway-endpoint "
        f"{shlex.quote(profile.endpoint_uri('sensor_gateway_metadata'))} "
        f"--navdp-server {shlex.quote(profile.endpoint_uri('xnavdp_http'))} "
        f"--sensor-gateway-poll-hz {settings['sensor_gateway_poll_hz']} "
        f"--sensor-gateway-request-timeout-ms "
        f"{settings['sensor_gateway_request_timeout_ms']} "
        f"--sensor-gateway-max-age-ms {settings['sensor_gateway_max_age_ms']} "
        f"--sensor-gateway-max-skew-ms {settings['sensor_gateway_max_skew_ms']} "
        f"--control-hz {settings['control_hz']} "
        f"--mpc-hz {settings['mpc_hz']} "
        f"--mpc-result-timeout-s {settings['mpc_result_timeout_s']} "
        f"--heading-preview-s {settings['heading_preview_s']} "
        f"--goal-tolerance-m {settings['goal_tolerance_m']} "
        f"--navdp-stop-threshold {settings['stop_threshold']} "
        f"--odom-timeout-s {settings['odometry_timeout_s']} "
        f"--trajectory-timeout-s {settings['trajectory_timeout_s']} "
        f"--navdp-request-timeout-s {settings['request_timeout_s']} "
        f"--no-visualize --visualization-gateway-endpoint "
        f"{shlex.quote(profile.endpoint_uri('sensor_gateway_visualization_ingress'))} "
        f"{recording}"
    )


def build_planner_velocity_executor_command(
    config: InferenceLaunchConfig, repo_root: Path
) -> str:
    profile = _runtime_profile(config)
    settings = profile.component("planner_executor")
    planner_relay = profile.endpoint("planner_relay")
    orientation_telemetry = profile.endpoint("orientation_telemetry")
    navigation_runtime_status = profile.endpoint("navigation_runtime_status")
    orientation_output = (
        "--orientation-output-endpoint "
        f"{shlex.quote(f'tcp://*:{orientation_telemetry.port}')} "
        if config.base_pose_enabled
        and config.base_pose_mode
        in {"raw_yoloe_servo", "dual_raw_yoloe_servo"}
        else ""
    )
    return (
        f"cd {shlex.quote(str(repo_root))} && source .venv_teleop/bin/activate && "
        "python gear_sonic/scripts/planner_velocity_executor.py "
        f"--command-endpoint {shlex.quote(profile.endpoint_uri('navigation_command'))} "
        f"--navdp-velocity-endpoint {shlex.quote(profile.endpoint_uri('navdp_velocity'))} "
        f"--output-endpoint {shlex.quote(f'tcp://*:{planner_relay.port}')} "
        "--runtime-status-endpoint "
        f"{shlex.quote(f'tcp://*:{navigation_runtime_status.port}')} "
        f"--sensor-gateway-endpoint {shlex.quote(profile.endpoint_uri('sensor_gateway_metadata'))} "
        f"--control-hz {settings['control_hz']} "
        f"--manual-velocity-timeout-s {settings['manual_velocity_timeout_s']} "
        f"--navdp-velocity-timeout-s {settings['navdp_velocity_timeout_s']} "
        f"--radar-timeout-s {settings['radar_timeout_s']} "
        f"--sensor-gateway-poll-hz {settings['sensor_gateway_poll_hz']} "
        f"--sensor-gateway-request-timeout-ms {settings['sensor_gateway_request_timeout_ms']} "
        f"--sensor-gateway-max-age-ms {settings['sensor_gateway_max_age_ms']} "
        f"{orientation_output}"
    )


def build_navdp_server_command(config: InferenceLaunchConfig) -> str:
    navdp_port = _runtime_profile(config).endpoint("xnavdp_http").port
    return (
        f"cd {shlex.quote(config.navdp_root)} && "
        "conda run --no-capture-output -n navdp python -m eval.src.policy_server "
        f"--port {navdp_port} --embodiment humanoid "
        f"--checkpoint {shlex.quote(config.navdp_checkpoint)} "
        "--device cuda:0 --real --no-visualization"
    )


def build_livox_command(config: InferenceLaunchConfig) -> str:
    return (
        "unset COLCON_CURRENT_PREFIX AMENT_PREFIX_PATH CMAKE_PREFIX_PATH; "
        "source /opt/ros/humble/setup.bash && "
        f"export LD_LIBRARY_PATH={shlex.quote(config.livox_sdk_lib)}:$LD_LIBRARY_PATH && "
        f"source {shlex.quote(config.fastlio_workspace)}/install/setup.bash && "
        "ros2 launch livox_ros_driver2 msg_MID360_launch.py"
    )


def build_fastlio_command(config: InferenceLaunchConfig) -> str:
    return (
        "unset COLCON_CURRENT_PREFIX AMENT_PREFIX_PATH CMAKE_PREFIX_PATH; "
        "source /opt/ros/humble/setup.bash && "
        f"export LD_LIBRARY_PATH={shlex.quote(config.livox_sdk_lib)}:$LD_LIBRARY_PATH && "
        f"source {shlex.quote(config.fastlio_workspace)}/install/setup.bash && "
        f"ros2 launch fast_lio mapping.launch.py config_file:={shlex.quote(config.fastlio_config)} rviz:=false"
    )


def build_fastlio_supervisor_command(config: InferenceLaunchConfig) -> str:
    return (
        "python gear_sonic/scripts/run_fastlio_supervisor.py "
        f"--profile {shlex.quote(config.config)} "
        f"--config-file {shlex.quote(config.fastlio_config)} "
        "--control-gateway-endpoint tcp://127.0.0.1:"
        f"{config.control_gateway_intent_port}"
    )


def build_slam_debug_command(config: InferenceLaunchConfig) -> str:
    """Record the raw inputs and outputs needed to replay a FAST-LIO failure."""
    profile = _runtime_profile(config)
    topic_roles = ("lidar", "lidar_imu", "odometry", "registered_cloud")
    topics = " ".join(shlex.quote(profile.ros_topics[role]) for role in topic_roles)
    return f'ros2 bag record -o "$slam_debug_dir/rosbag" {topics}'


def build_sensor_gateway_command(config: InferenceLaunchConfig, repo_root: Path) -> str:
    profile = _runtime_profile(config)
    ros_mode = "" if config.keyboard_planner and not config.sim else "--no-enable-ros "
    setup = (
        "unset COLCON_CURRENT_PREFIX AMENT_PREFIX_PATH CMAKE_PREFIX_PATH; "
        "source /opt/ros/humble/setup.bash && "
        f"export LD_LIBRARY_PATH={shlex.quote(config.livox_sdk_lib)}:$LD_LIBRARY_PATH && "
        f"source {shlex.quote(config.fastlio_workspace)}/install/setup.bash && "
        f"cd {shlex.quote(str(repo_root))} && source .venv_teleop/bin/activate && "
        f"export PYTHONPATH={shlex.quote(str(repo_root))}:$PYTHONPATH; "
    )
    gateway = (
        "python gear_sonic/scripts/run_sensor_gateway.py "
        f"--camera-host {shlex.quote(config.camera_host)} "
        f"--camera-port {config.camera_port} "
        f"--rpc-port {config.sensor_gateway_port} "
        f"--visualization-port {config.sensor_gateway_visualization_port} "
        f"--vla-timing-port {config.vla_timing_port} "
        f"{ros_mode}"
    )
    ros_stack_enabled = config.keyboard_planner and not config.sim
    slam_debug_enabled = config.slam_debug and ros_stack_enabled
    slam_debug_setup = ""
    livox_log = "/tmp/sonic_livox_driver.log"
    fastlio_log = "/tmp/sonic_fastlio.log"
    if slam_debug_enabled:
        debug_root = shlex.quote(str(repo_root / "outputs" / "slam_debug"))
        slam_debug_setup = (
            f"slam_debug_dir={debug_root}/$(date +%Y%m%d_%H%M%S_%N); "
            'mkdir -p "$slam_debug_dir"; '
            'echo "[SLAM debug] recording to $slam_debug_dir"; '
        )
        livox_log = '"$slam_debug_dir/livox_driver.log"'
        fastlio_log = '"$slam_debug_dir/fastlio.log"'

    background_commands: list[tuple[str, str, str]] = []
    if ros_stack_enabled:
        if slam_debug_enabled:
            background_commands.append(
                (
                    "slam_debug_pid",
                    build_slam_debug_command(config),
                    '"$slam_debug_dir/rosbag.log"',
                )
            )
        background_commands.extend(
            (
                (
                    "livox_pid",
                    "ros2 launch livox_ros_driver2 msg_MID360_launch.py",
                    livox_log,
                ),
                (
                    "fastlio_pid",
                    build_fastlio_supervisor_command(config),
                    fastlio_log,
                ),
            )
        )
    if config.opencv_viewer:
        background_commands.append(
            (
                "viewer_pid",
                "python gear_sonic/scripts/run_operator_cv_viewer.py "
                f"--sensor-gateway-endpoint tcp://127.0.0.1:{config.sensor_gateway_port} "
                "--control-gateway-endpoint "
                f"tcp://127.0.0.1:{config.control_gateway_dispatch_port} "
                "--navigation-runtime-status-endpoint "
                f"{shlex.quote(profile.endpoint_uri('navigation_runtime_status'))}",
                "/tmp/sonic_opencv_viewer.log",
            )
        )
    if config.keyboard_planner and config.planner_input == "lavira":
        background_commands.append(
            (
                "lingbot_pid",
                build_lingbot_command(config, repo_root),
                "/tmp/sonic_lingbot.log",
            )
        )
    if not background_commands:
        return setup + gateway

    ready_file_setup = (
        f"rm -f {shlex.quote(str(LINGBOT_READY_FILE))}; "
        if config.keyboard_planner and config.planner_input == "lavira"
        else ""
    )
    launch_background = "".join(
        f"{command} >{log_path} 2>&1 & {pid_name}=$!; "
        for pid_name, command, log_path in background_commands
    )
    pid_names = " ".join(f"${pid_name}" for pid_name, _, _ in background_commands)
    return (
        setup
        + slam_debug_setup
        + ready_file_setup
        + launch_background
        + gateway
        + f"; gateway_status=$?; kill {pid_names} 2>/dev/null; "
        + f"wait {pid_names} 2>/dev/null; "
        + f"rm -f {shlex.quote(str(LINGBOT_READY_FILE))}; (exit $gateway_status)"
    )


def run_readiness_gate(
    config: InferenceLaunchConfig, repo_root: Path, stage: Literal["lidar", "navigation"]
) -> bool:
    navdp_endpoint = _runtime_profile(config).endpoint("xnavdp_http")
    timeout = (
        config.lidar_ready_timeout if stage == "lidar" else config.navigation_ready_timeout
    )
    gateway_requirement = (
        f"--require-sensor-gateway --sensor-gateway-port {config.sensor_gateway_port} "
        if stage == "navigation"
        else ""
    )
    command = (
        "unset COLCON_CURRENT_PREFIX AMENT_PREFIX_PATH CMAKE_PREFIX_PATH; "
        "source /opt/ros/humble/setup.bash && "
        f"source {shlex.quote(config.fastlio_workspace)}/install/setup.bash && "
        f"cd {shlex.quote(str(repo_root))} && source .venv_teleop/bin/activate && "
        f"python gear_sonic/scripts/navdp_readiness_gate.py --stage {stage} "
        f"--timeout {timeout} "
        f"--camera-host {shlex.quote(config.camera_host)} --camera-port {config.camera_port} "
        f"--navdp-host {shlex.quote(navdp_endpoint.host)} "
        f"--navdp-port {navdp_endpoint.port} "
        f"{gateway_requirement}"
    )
    return subprocess.run(
        ["/usr/bin/bash", "--noprofile", "--norc", "-c", command]
    ).returncode == 0


def _dotenv_has_nonempty_value(path: Path, name: str) -> bool:
    """Check a dotenv key without loading or exposing its secret value."""
    if not path.is_file():
        return False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, separator, value = line.partition("=")
        if separator and key.strip() == name and value.strip().strip("'\""):
            return True
    return False


def _check_prerequisites(config: InferenceLaunchConfig):
    """Verify that required tools and venvs exist."""
    errors = []

    if not shutil.which("tmux"):
        errors.append("tmux is not installed. Install with: sudo apt install tmux")

    repo_root = Path(__file__).resolve().parent.parent.parent

    if not (repo_root / ".venv_inference" / "bin" / "activate").exists():
        errors.append(
            ".venv_inference not found. Run: bash install_scripts/install_inference.sh"
        )
    if not (repo_root / ".venv_teleop" / "bin" / "activate").exists():
        errors.append(".venv_teleop not found. Run: bash install_scripts/install_pico.sh")

    if config.planner_input == "lavira":
        if not config.lavira_mission.strip():
            errors.append("--lavira-mission is required when --planner-input lavira")
        if not config.lavira_global_target.strip():
            errors.append(
                "--lavira-global-target is required when --planner-input lavira"
            )
        if (
            config.lavira_vision_backend == "qwenvl"
            and not os.environ.get("DASHSCOPE_API_KEY", "").strip()
            and not _dotenv_has_nonempty_value(repo_root / ".env.local", "DASHSCOPE_API_KEY")
        ):
            errors.append(
                "Qwen-VL requires DASHSCOPE_API_KEY in the launcher environment "
                "or the gitignored .env.local file"
            )
        for path, label in (
            (
                Path(config.navdp_root) / "eval" / "src" / "policy_server.py",
                "X-NavDP server",
            ),
            (Path(config.navdp_checkpoint), "NavDP checkpoint"),
            (Path(config.fastlio_workspace) / "install" / "setup.bash", "FAST-LIO workspace"),
            (Path("/opt/ros/humble/setup.bash"), "ROS2 Humble"),
        ):
            if not path.exists():
                errors.append(f"{label} not found: {path}")
        try:
            with socket.create_connection((config.camera_host, config.camera_port), timeout=1.0):
                pass
        except OSError:
            errors.append(
                f"robot camera is not reachable at {config.camera_host}:{config.camera_port}"
            )

    if config.base_pose_enabled:
        if not config.base_pose_task.strip():
            errors.append("--base-pose-task is required when Base-Pose is enabled")
        if config.base_pose_planner_hz <= 0.0:
            errors.append("--base-pose-planner-hz must be positive")
        if config.base_pose_mode == "raw_yoloe_servo":
            expected_depth = f"camera/{config.base_pose_camera_stream}_depth"
            if config.base_pose_yoloe_depth_stream != expected_depth:
                errors.append(
                    "raw_yoloe_servo requires the raw depth stream aligned to its RGB: "
                    f"--base-pose-yoloe-depth-stream {expected_depth}"
                )
        elif config.base_pose_mode == "dual_raw_yoloe_servo":
            head = config.base_pose_dual_head_camera_stream
            chest = config.base_pose_dual_chest_camera_stream
            if not head or not chest or head == chest:
                errors.append("dual Base-Pose camera streams must be distinct")
            expected_head_depth = f"camera/{head}_depth"
            expected_chest_depth = f"camera/{chest}_depth"
            if config.base_pose_dual_head_depth_stream != expected_head_depth:
                errors.append(
                    "dual head raw depth must match its RGB: "
                    f"--base-pose-dual-head-depth-stream {expected_head_depth}"
                )
            if config.base_pose_dual_chest_depth_stream != expected_chest_depth:
                errors.append(
                    "dual chest raw depth must match its RGB: "
                    f"--base-pose-dual-chest-depth-stream {expected_chest_depth}"
                )
            if config.base_pose_dual_match_tolerance_frames <= 0:
                errors.append("dual match tolerance must be positive")
            if config.base_pose_dual_initialization_grace_s <= 0.0:
                errors.append("dual initialization grace must be positive")
            if config.base_pose_raw_reference_update_interval_frames <= 0:
                errors.append(
                    "dual Base-Pose reference update interval must be positive"
                )
        else:
            errors.append(f"unsupported Base-Pose mode: {config.base_pose_mode}")
        for value, label in (
            (config.base_pose_yoloe_confidence, "YOLOE confidence"),
            (config.base_pose_raw_servo_hz, "raw servo frequency"),
            (config.base_pose_raw_head_target_distance_m, "head target distance"),
            (config.base_pose_raw_chest_target_distance_m, "chest target distance"),
            (config.base_pose_raw_forward_tolerance_m, "forward tolerance"),
            (config.base_pose_raw_lateral_tolerance_m, "lateral tolerance"),
            (config.base_pose_raw_min_linear_speed_m_s, "minimum linear speed"),
            (config.base_pose_raw_max_lateral_speed_m_s, "maximum lateral speed"),
            (config.base_pose_raw_min_yaw_speed_rad_s, "minimum yaw speed"),
            (config.base_pose_raw_yaw_tolerance_deg, "yaw tolerance"),
            (config.base_pose_raw_yaw_coarse_speed_rad_s, "coarse yaw speed"),
            (config.base_pose_raw_yaw_trim_speed_rad_s, "trim yaw speed"),
            (
                config.base_pose_raw_forward_recenter_yaw_speed_rad_s,
                "forward recenter yaw speed",
            ),
            (config.base_pose_raw_camera_stale_s, "camera stale timeout"),
            (config.base_pose_raw_max_run_s, "maximum run time"),
        ):
            if value <= 0.0:
                errors.append(f"Base-Pose {label} must be positive")
        if not 0.0 < config.base_pose_yoloe_confidence <= 1.0:
            errors.append("Base-Pose YOLOE confidence must be in (0, 1]")
        if config.base_pose_yoloe_imgsz <= 0:
            errors.append("Base-Pose YOLOE image size must be positive")
        if config.base_pose_raw_reference_update_interval_frames < 0:
            errors.append("Base-Pose reference update interval cannot be negative")
        if not (
            0.0
            <= config.base_pose_raw_reference_update_min_confidence
            <= 1.0
            and 0.0 <= config.base_pose_raw_reference_update_min_iou <= 1.0
        ):
            errors.append("Base-Pose reference confidence and IoU must be in [0, 1]")
        if not (
            0.0 < config.base_pose_raw_horizontal_guard_fraction
            < config.base_pose_raw_horizontal_recovery_fraction
            < 0.5
        ):
            errors.append(
                "Base-Pose horizontal fractions must satisfy 0 < guard < recovery < 0.5"
            )
        for relative_path, label in (
            (config.base_pose_yoloe_model_path, "YOLOE model"),
            (config.base_pose_camera_intrinsics_path, "camera intrinsics"),
        ):
            path = Path(relative_path)
            if not path.is_absolute():
                path = repo_root / path
            if not path.is_file():
                errors.append(f"Base-Pose {label} not found: {path}")
        if (
            config.base_pose_vision_backend == "qwenvl"
            and not os.environ.get("DASHSCOPE_API_KEY", "").strip()
            and not _dotenv_has_nonempty_value(
                repo_root / ".env.local", "DASHSCOPE_API_KEY"
            )
        ):
            errors.append(
                "Base-Pose Qwen-VL requires DASHSCOPE_API_KEY in the launcher "
                "environment or the gitignored .env.local file"
            )

    deploy_dir = repo_root / "gear_sonic_deploy"
    if not (deploy_dir / "deploy.sh").exists():
        errors.append(
            f"gear_sonic_deploy/deploy.sh not found at {deploy_dir}. "
            "Ensure the deploy directory is set up."
        )
    else:
        try:
            resolve_deploy_policy(config, deploy_dir)
        except (ValueError, FileNotFoundError) as exc:
            errors.append(str(exc))

    if config.data_exporter:
        if not (repo_root / ".venv_data_collection" / "bin" / "activate").exists():
            errors.append(
                ".venv_data_collection not found (needed for data exporter). Run: "
                "bash install_scripts/install_data_collection.sh"
            )

    if config.sim and not (repo_root / ".venv_sim" / "bin" / "activate").exists():
        errors.append(
            ".venv_sim not found. Set up the simulation venv first."
        )

    if errors:
        print("ERROR: Prerequisites not met:\n")
        for e in errors:
            print(f"  - {e}")
        print()
        sys.exit(1)


def _kill_existing_session():
    subprocess.run(
        ["tmux", "kill-session", "-t", SESSION_NAME],
        capture_output=True,
    )


def _clear_stale_fastlio_processes() -> None:
    """Remove orphaned FAST-LIO launch and mapping processes from older runs."""
    patterns = (
        r"(^|/)ros2 launch fast_lio mapping\.launch\.py( |$)",
        r"(^|/)fastlio_mapping( |$)",
    )
    for pattern in patterns:
        subprocess.run(["pkill", "-TERM", "-f", pattern], capture_output=True)
    time.sleep(0.5)
    for pattern in patterns:
        subprocess.run(["pkill", "-KILL", "-f", pattern], capture_output=True)


def _parse_pane_ids(output: str, expected_count: int = 6) -> list[str]:
    indexed = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) == 2:
            indexed[int(fields[0])] = fields[1]
    if len(indexed) != expected_count or sorted(indexed) != list(range(expected_count)):
        raise RuntimeError(
            f"expected {expected_count} tmux panes, found {len(indexed)}"
        )
    return [indexed[index] for index in range(expected_count)]


def _create_tmux_session(pane_count: int = 6) -> list[str]:
    if pane_count < 6:
        raise ValueError("inference pane_count cannot be smaller than 6")
    bash = shutil.which("bash") or "/bin/bash"
    subprocess.run(
        [
            "tmux", "new-session", "-d", "-x", "240", "-y", "60",
            "-s", SESSION_NAME, bash, "--noprofile", "--norc",
        ],
        check=True,
    )
    subprocess.run(
        ["tmux", "set-option", "-t", SESSION_NAME, "-g", "mouse", "on"],
    )
    subprocess.run(
        [
            "tmux", "set-option", "-t", f"{SESSION_NAME}:0", "-w",
            "remain-on-exit", "on",
        ],
        check=True,
    )
    subprocess.run(
        ["tmux", "bind-key", "-T", "root", "C-\\", "kill-session"],
    )
    subprocess.run(
        ["tmux", "rename-window", "-t", f"{SESSION_NAME}:0", "inference"],
    )

    # Build three rows first, then distribute all inference and Gateway panes
    # over those rows.  Keeping one tmux window avoids a hidden runtime window.
    top_pane = subprocess.run(
        ["tmux", "display-message", "-p", "-t", f"{SESSION_NAME}:0.0", "#{pane_id}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    bottom_pane = subprocess.run(
        [
            "tmux", "split-window", "-v", "-t", top_pane, "-P", "-F", "#{pane_id}",
            bash, "--noprofile", "--norc",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    middle_pane = subprocess.run(
        [
            "tmux", "split-window", "-v", "-t", top_pane, "-P", "-F", "#{pane_id}",
            bash, "--noprofile", "--norc",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    row_panes = (top_pane, middle_pane, bottom_pane)
    columns_per_row = [pane_count // 3] * 3
    for index in range(pane_count % 3):
        columns_per_row[index] += 1
    for row_pane, column_count in zip(row_panes, columns_per_row, strict=True):
        for _ in range(column_count - 1):
            subprocess.run(
                [
                    "tmux", "split-window", "-h", "-t", row_pane,
                    bash, "--noprofile", "--norc",
                ],
                check=True,
            )
    subprocess.run(["tmux", "select-layout", "-t", f"{SESSION_NAME}:0", "tiled"], check=True)

    pane_output = subprocess.run(
        [
            "tmux", "list-panes", "-t", f"{SESSION_NAME}:0",
            "-F", "#{pane_index} #{pane_id}",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    pane_ids = _parse_pane_ids(pane_output, pane_count)
    time.sleep(5)
    return pane_ids


def _send_to_pane(pane_id: str, cmd: str, wait: float = 1.0):
    subprocess.run(
        ["tmux", "send-keys", "-t", pane_id, cmd, "C-m"],
        check=True,
    )
    time.sleep(wait)


def _check_pane_alive(pane_id: str) -> bool:
    result = subprocess.run(
        ["tmux", "list-panes", "-t", pane_id, "-F", "#{pane_dead}"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() != "1"


def main(config: InferenceLaunchConfig):
    repo_root = Path(__file__).resolve().parent.parent.parent

    _check_prerequisites(config)
    _kill_existing_session()
    _clear_stale_fastlio_processes()

    exporter_prompt = config.task_prompt if config.task_prompt else config.prompt

    print("=" * 60)
    print("  SONIC VLA Inference Launcher")
    print("=" * 60)
    print(f"  Launch config:   {config.config}")
    print(f"  Mode:            {'Simulation' if config.sim else 'Real Robot'}")
    print(f"  PolicyServer:    {config.policy_host}:{config.policy_port}")
    print(f"  Embodiment:      {config.embodiment_tag}")
    print(f"  Prompt:          {config.prompt}")
    print(f"  Action rate:     {config.action_publish_rate} Hz")
    print(f"  Action horizon:  {config.action_horizon}")
    print(f"  Camera:          {config.camera_host}:{config.camera_port}")
    print(f"  Data exporter:   {'Yes' if config.data_exporter else 'No'}")
    if config.data_exporter:
        print(f"    Task prompt:   {exporter_prompt}")
        print(f"    Chest camera:  {'Yes' if config.record_chest_camera else 'No'}")
    print(f"  PC IP:           {_get_local_ip()}")
    print("=" * 60)

    base_pose_runtime_enabled = config.keyboard_planner and config.base_pose_enabled
    runtime_pane_count = 2 + int(config.keyboard_planner) + int(base_pose_runtime_enabled)
    pane_ids = _create_tmux_session(6 + runtime_pane_count)
    print(f"Created tmux session: {SESSION_NAME}")

    # --- Window 1 (sim only): MuJoCo Simulator ---
    if config.sim:
        subprocess.run(
            ["tmux", "new-window", "-t", SESSION_NAME, "-n", "sim"],
        )
        sim_cmd = (
            f"cd {repo_root} && "
            f"source .venv_sim/bin/activate && "
            f"python gear_sonic/scripts/run_sim_loop.py "
            f"--enable-image-publish --enable-offscreen "
            f"--camera-port {config.camera_port}"
        )
        sim_target = f"{SESSION_NAME}:sim"
        subprocess.run(
            ["tmux", "send-keys", "-t", sim_target, sim_cmd, "C-m"],
        )
        print("Starting MuJoCo simulator (window: sim)...")
        time.sleep(3.0)

        subprocess.run(
            ["tmux", "select-window", "-t", f"{SESSION_NAME}:inference"],
        )

    # --- Pane 0 (top-left): C++ Deploy ---
    deploy_cmd = build_deploy_command(config, repo_root)

    print("Starting C++ deploy (pane 0)...")
    _send_to_pane(pane_ids[0], deploy_cmd, wait=3.0)

    if not _check_pane_alive(pane_ids[0]):
        print("WARNING: C++ deploy pane may have failed to start.")

    # Start the two mandatory Gateway boundaries before their clients.
    runtime_panes = pane_ids[6:]
    runtime_index = 0
    print(
        "Starting SensorGateway with background ROS/OpenCV/LingBot services "
        f"(pane {6 + runtime_index})..."
    )
    _send_to_pane(
        runtime_panes[runtime_index],
        build_sensor_gateway_command(config, repo_root),
        wait=1.0,
    )
    if config.slam_debug and config.keyboard_planner and not config.sim:
        print("SLAM/IMU debug recording: outputs/slam_debug/<launch timestamp>/")
    runtime_index += 1
    print(f"Starting ControlGateway router (pane {6 + runtime_index})...")
    _send_to_pane(
        runtime_panes[runtime_index],
        build_control_gateway_command(config, repo_root),
        wait=1.0,
    )
    if config.keyboard_planner:
        runtime_index += 1
        print(f"Starting shared Planner velocity executor (pane {6 + runtime_index})...")
        _send_to_pane(
            runtime_panes[runtime_index],
            build_planner_velocity_executor_command(config, repo_root),
            wait=1.0,
        )
    if base_pose_runtime_enabled:
        runtime_index += 1
        print(f"Starting Base-Pose agent (pane {6 + runtime_index})...")
        _send_to_pane(
            runtime_panes[runtime_index],
            build_base_pose_agent_command(config, repo_root),
            wait=1.0,
        )

    # --- Pane 2: VLA Inference ---
    inference_cmd = build_vla_inference_command(config, repo_root)

    print("Starting VLA inference (pane 2)...")
    _send_to_pane(pane_ids[2], inference_cmd, wait=1.0)

    # --- Pane 1: standalone operator CLI ---
    print("Starting standalone operator CLI (pane 1)...")
    _send_to_pane(
        pane_ids[1],
        build_operator_interface_command(config, repo_root),
        wait=2.0,
    )

    # --- Panes 3-5: semantic target, NavDP planner, and NavDP server ---
    if config.keyboard_planner:
        planner_input_cmd = build_planner_input_command(config, repo_root)
        commands = [
            (3, "LaViRA semantic planner", planner_input_cmd),
            (4, "NavDP planner", build_navdp_planner_command(config, repo_root)),
            (5, "NavDP server", build_navdp_server_command(config)),
        ]
        # Strictly serialized startup: the launcher does not dispatch a later
        # stage until real data has passed the previous readiness gate.
        pane, label, command = commands[2]
        print(f"Starting {label} (pane {pane})...")
        _send_to_pane(pane_ids[pane], command, wait=1.0)
        print("Waiting for real MID-360 LiDAR and IMU samples...")
        if not run_readiness_gate(config, repo_root, "lidar"):
            raise RuntimeError(
                "MID-360 did not become ready; FAST-LIO and navigation were not started"
            )

        print("Waiting for FAST-LIO odometry, registered cloud, camera, and NavDP server...")
        if not run_readiness_gate(config, repo_root, "navigation"):
            raise RuntimeError(
                "navigation prerequisites did not become ready; LaViRA and NavDP planner were not started"
            )

        for pane, label, command in (commands[0], commands[1]):
            print(f"Starting {label} (pane {pane})...")
            _send_to_pane(pane_ids[pane], command, wait=1.0)

    if config.data_exporter:
        subprocess.run(
            ["tmux", "new-window", "-t", SESSION_NAME, "-n", "data_exporter"],
        )
        exporter_cmd = build_data_exporter_command(config, repo_root)
        subprocess.run(
            ["tmux", "send-keys", "-t", f"{SESSION_NAME}:data_exporter", exporter_cmd, "C-m"],
        )
        print("Starting data exporter (window: data_exporter)...")
        time.sleep(2.0)
        subprocess.run(
            ["tmux", "select-window", "-t", f"{SESSION_NAME}:inference"],
        )


    print()
    print("=" * 60)
    print("  All components launched!")
    print()
    print(f"  tmux session: {SESSION_NAME}")
    print()
    if config.sim:
        print("  Window 'sim':")
        print("    MuJoCo Simulator (.venv_sim)")
        print()
    print("  Window 'inference':")
    print("    Pane 0: C++ Deploy")
    print("    Pane 1: SONIC Operator CLI")
    print("    Pane 2: VLA Inference")
    print("    Pane 3: LaViRA Semantic + LISTEN_WASD")
    print("    Pane 4: NavDP Velocity Producer")
    print("    Pane 5: NavDP Server")
    runtime_label_index = 6
    print(f"    Pane {runtime_label_index}: Read-only SensorGateway")
    runtime_label_index += 1
    print(f"    Pane {runtime_label_index}: ControlGateway Router")
    runtime_label_index += 1
    if config.keyboard_planner:
        print(f"    Pane {runtime_label_index}: Shared Planner Velocity Executor + Safety")
        runtime_label_index += 1
    if config.keyboard_planner and config.base_pose_enabled:
        print(f"    Pane {runtime_label_index}: Base-Pose Agent")
    if config.data_exporter:
        print("    Window 'data_exporter':")
        print("      Data Exporter (.venv_data_collection)")
        print()
    print()
    print("  ** deploy.sh (pane 0) is waiting for confirmation --")
    print("     click on pane 0 and press Enter to proceed **")
    print()
    print("  Planner workflow:")
    print("    1. In pane 1: k (start) -> o (PLANNER mode)")
    if config.planner_input == "lavira":
        if config.base_pose_enabled:
            print("    2. In pane 1: N starts AgentNav; B starts Base-Pose; Space cancels")
        else:
            print("    2. In pane 1: N starts AgentNav; Space cancels and stops")
    else:
        print("    2. In pane 3: W/S/A/D/Q/E for safety-guarded locomotion")
    print("    3. In pane 1: i (POSE mode)")
    print("  Keyboard controls (type in pane 1):")
    print("    p        - Pause / resume inference")
    print("    k        - Start / stop C++ control loop")
    print("    i        - Send initial pose")
    print("    [        - Toggle left hand open/closed (initial pose)")
    print("    ]        - Toggle right hand open/closed (initial pose)")
    print("    t <text> - Change inference prompt")
    if config.data_exporter:
        print("    c        - Start recording episode")
        print("    e        - Stop recording (success)")
        print("    f        - Stop recording (failure)")
    print()
    print("  Navigation:")
    print("    Ctrl+b, arrow keys  - Switch between panes")
    if config.sim or config.data_exporter:
        print("    Ctrl+b, n / p       - Next / previous window")
    print("    Ctrl+b, d           - Detach from session")
    print("    Ctrl+\\              - Kill entire session")
    print("=" * 60)

    try:
        subprocess.run(["tmux", "attach", "-t", SESSION_NAME])
    except KeyboardInterrupt:
        pass

    result = subprocess.run(
        ["tmux", "has-session", "-t", SESSION_NAME],
        capture_output=True,
    )
    if result.returncode == 0:
        print(f"\nSession '{SESSION_NAME}' is still running.")
        print(f"  Reattach:  tmux attach -t {SESSION_NAME}")
        print(f"  Kill:      tmux kill-session -t {SESSION_NAME}")


def _signal_handler(_sig, _frame):
    print("\nShutdown requested...")
    subprocess.run(
        ["tmux", "kill-session", "-t", SESSION_NAME],
        capture_output=True,
    )
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _signal_handler)
    config = parse_inference_launch_config()
    main(config)
