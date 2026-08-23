"""All-in-one tmux launcher for SONIC VLA inference.

The overview window contains performance, keyboard control, and event panes.
The workers window keeps deploy, VLA, navigation, and Base-Pose processes in
independent panes with their original TTYs and scrollback. The NavDP policy
server runs in the background of the NavDP planner pane and shares its log
output. Simulation and data collection use optional additional windows.

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
from dataclasses import dataclass
from pathlib import Path
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Literal


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
from gear_sonic.runtime.profile import (  # noqa: E402
    load_component_config,
    load_runtime_profile,
)


def default_launch_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "launch_inference.yaml"


@dataclass
class InferenceLaunchConfig:
    """CLI config for the all-in-one VLA inference tmux launcher."""

    # Deployment mode
    sim: bool = False
    """Run against MuJoCo sim instead of real robot."""

    opencv_viewer: bool = True
    """Start the standalone OpenCV visualization process under the SensorGateway pane."""

    keyboard_planner: bool = True
    """Start planner input, rule-based safety, and MID-360 sidecars."""

    planner_input: Literal["keyboard", "lavira"] = "lavira"
    """Navigation keyboard and semantic target source used in the worker window."""

    depth_anything_ready_timeout_s: float = 180.0
    """Maximum time to wait for the metric model to become ready."""

    base_pose_enabled: bool = False
    """Start the independently triggered Base-Pose navigation agent."""

    slam_debug: bool = False
    """Record raw LiDAR/IMU and FAST-LIO outputs for each real-robot run."""

    lidar_ready_timeout_s: float = 30.0
    navigation_ready_timeout_s: float = 60.0
    # Data exporter (optional recording during inference)
    data_exporter: bool = True
    """Start the data exporter pane for recording during inference."""

    config: str = str(default_launch_config_path())
    """YAML file containing the complete launch_inference configuration."""


def load_inference_launch_config(path: str | Path | None = None) -> InferenceLaunchConfig:
    return load_component_config(
        InferenceLaunchConfig,
        "launcher",
        default_launch_config_path() if path is None else path,
    )


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
DEPTH_ANYTHING_READY_FILE = Path("/tmp/sonic_depth_anything_ready")
PANE_TITLES = {
    "performance": "PERFORMANCE · SensorGateway",
    "control": "CONTROL · Operator CLI",
    "events": "EVENTS · ControlGateway",
    "deploy": "WORKER · C++ Deploy",
    "vla": "WORKER · VLA Inference",
    "planner_input": "WORKER · LaViRA / Planner Input",
    "navdp": "WORKER · NavDP + Server",
    "planner_executor": "WORKER · Planner Executor",
    "base_pose": "WORKER · Base-Pose",
}


def _runtime_profile(config: InferenceLaunchConfig):
    return load_runtime_profile(config.config)


def _component(config: InferenceLaunchConfig, name: str):
    return _runtime_profile(config).component(name)


def _depth_anything_required(config: InferenceLaunchConfig) -> bool:
    """Return whether the active navigation input still consumes DA depth."""
    return bool(config.keyboard_planner and config.planner_input == "lavira")


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
    settings: Any,
    deploy_dir: Path,
) -> tuple[str, str]:
    """Resolve and validate a paired deploy checkpoint and observation config."""
    has_checkpoint = bool(str(settings["checkpoint"]).strip())
    has_obs_config = bool(str(settings["obs_config"]).strip())
    if has_checkpoint != has_obs_config:
        raise ValueError(
            "deploy_checkpoint and deploy_obs_config must be set together"
        )

    if has_checkpoint:
        checkpoint = _normalized_deploy_argument(str(settings["checkpoint"]).strip())
        obs_config = _normalized_deploy_argument(str(settings["obs_config"]).strip())
        download_hint = "provide the matching custom deployment files"
    else:
        checkpoint, obs_config, download_hint = DEPLOY_POLICY_PRESETS[
            str(settings["policy_variant"])
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
    settings = _component(config, "deploy")
    checkpoint, obs_config = resolve_deploy_policy(settings, deploy_dir)
    command_endpoint = _runtime_profile(config).endpoint("cpp_command")
    deploy_mode = "sim" if config.sim else "real"
    command = (
        f"cd {shlex.quote(str(deploy_dir))} && "
        f"./deploy.sh "
        f"--yes "
        f"--input-type {shlex.quote(str(settings['input_type']))} "
        f"--zmq-host {shlex.quote(command_endpoint.host)} "
        f"--hand-type dex1 "
        f"--dex1-kp 1.0 "
        f"--dex1-kd 0.05 "
        f"--cp {shlex.quote(checkpoint)} "
        f"--obs-config {shlex.quote(obs_config)} "
    )
    if settings["planner"]:
        command += f"--planner {shlex.quote(str(settings['planner']))} "
    if settings["motion_data"]:
        command += f"--motion-data {shlex.quote(str(settings['motion_data']))} "
    if settings["output_type"]:
        command += f"--output-type {shlex.quote(str(settings['output_type']))} "
    return command + deploy_mode


def build_vla_inference_command(
    config: InferenceLaunchConfig,
    repo_root: Path,
) -> str:
    """Build VLA command using the selected runtime profile."""
    return (
        f"cd {shlex.quote(str(repo_root))} && "
        "source .venv_inference/bin/activate && "
        "python -m gear_sonic.utils.inference.vla.service "
        f"--profile {shlex.quote(config.config)}"
    )


def build_control_gateway_command(
    config: InferenceLaunchConfig,
    repo_root: Path,
) -> str:
    """Build the typed ControlGateway router command."""
    return (
        f"cd {shlex.quote(str(repo_root))} && "
        "source .venv_inference/bin/activate && "
        "python -m gear_sonic.runtime.gateway.services.control "
        f"--profile {shlex.quote(config.config)}"
    )


def build_data_exporter_command(
    config: InferenceLaunchConfig,
    repo_root: Path,
) -> str:
    """Build DataExporter with runtime-profile Gateway inputs only."""
    settings = _component(config, "data_exporter")
    task_prompt = str(settings["task_prompt"]) or str(_component(config, "vla")["prompt"])
    command = (
        f"cd {shlex.quote(str(repo_root))} && "
        "source .venv_data_collection/bin/activate && "
        "python -m gear_sonic.utils.data_collection.service "
        f"--profile {shlex.quote(config.config)} "
        f"--task-prompt {shlex.quote(task_prompt)}"
    )
    if settings["dataset_name"]:
        command += f" --dataset-name {shlex.quote(str(settings['dataset_name']))}"
    if settings["record_chest_camera"]:
        command += " --record-chest-camera"
    return command


def build_operator_console_command(
    config: InferenceLaunchConfig,
    repo_root: Path,
) -> str:
    return (
        f"cd {shlex.quote(str(repo_root))} && "
        "source .venv_inference/bin/activate && "
        "python -m gear_sonic.utils.operator.console "
        f"--profile {shlex.quote(config.config)}"
    )


def build_planner_input_command(config: InferenceLaunchConfig, repo_root: Path) -> str:
    """Build the worker command for the selected REASEN command source."""
    if config.planner_input == "keyboard":
        return (
            f"cd {repo_root} && "
            f"source .venv_teleop/bin/activate && "
            f"python -m gear_sonic.utils.planner_control.keyboard "
            f"--profile {shlex.quote(config.config)}"
        )

    local_env = "set -a; [ ! -f .env.local ] || . ./.env.local; set +a; "
    quoted_root = shlex.quote(str(repo_root))
    ready_file = shlex.quote(str(DEPTH_ANYTHING_READY_FILE))
    return (
        f"cd {quoted_root} && "
        f"{local_env}"
        f"timeout {config.depth_anything_ready_timeout_s}s sh -c "
        f"'while [ ! -f {ready_file} ]; do sleep 0.2; done' || "
        "{ echo '[LaViRA] Depth Anything did not become ready' >&2; exit 1; }; "
        f".venv_inference/bin/python -m gear_sonic.utils.inference.lavira.service "
        f"--profile {shlex.quote(config.config)}"
    )


def build_base_pose_agent_command(
    config: InferenceLaunchConfig, repo_root: Path
) -> str:
    """Build the independent ControlGateway-backed Base-Pose agent."""

    return (
        f"cd {shlex.quote(str(repo_root))} && "
        ".venv_inference/bin/python -m gear_sonic.utils.inference.base_pose.agent "
        f"--profile {shlex.quote(config.config)}"
    )


def build_depth_anything_command(
    config: InferenceLaunchConfig, repo_root: Path
) -> str:
    """Build the shared RGB-only metric chest-depth process."""
    quoted_root = shlex.quote(str(repo_root))
    ready_file = shlex.quote(str(DEPTH_ANYTHING_READY_FILE))
    return (
        f"PYTHONPATH={quoted_root} {quoted_root}/.venv_depth_anything/bin/python "
        "-m gear_sonic.utils.inference.lavira.depth_service "
        f"--profile {shlex.quote(config.config)} --ready-file {ready_file}"
    )


def build_navdp_planner_command(config: InferenceLaunchConfig, repo_root: Path) -> str:
    profile = _runtime_profile(config)
    fastlio = profile.component("fastlio")
    return (
        "unset COLCON_CURRENT_PREFIX AMENT_PREFIX_PATH CMAKE_PREFIX_PATH; "
        "source /opt/ros/humble/setup.bash && "
        f"source {shlex.quote(str(fastlio['workspace']))}/install/setup.bash && "
        f"cd {shlex.quote(str(repo_root))} && source .venv_teleop/bin/activate && "
        "python -m gear_sonic.utils.inference.navdp.service "
        f"--profile {shlex.quote(config.config)}"
    )


def build_planner_velocity_executor_command(
    config: InferenceLaunchConfig, repo_root: Path
) -> str:
    orientation_output = (
        "--publish-orientation "
        if config.base_pose_enabled
        else ""
    )
    return (
        f"cd {shlex.quote(str(repo_root))} && source .venv_teleop/bin/activate && "
        "python -m gear_sonic.utils.planner_control.executor_service "
        f"--profile {shlex.quote(config.config)} "
        f"{orientation_output}"
    )


def build_navdp_server_command(config: InferenceLaunchConfig) -> str:
    profile = _runtime_profile(config)
    navdp_port = profile.endpoint("xnavdp_http").port
    settings = profile.component("navdp")
    return (
        f"cd {shlex.quote(str(settings['root']))} && "
        "PYTHONUNBUFFERED=1 conda run --no-capture-output -n navdp "
        "python -m eval.src.policy_server "
        f"--port {navdp_port} --embodiment humanoid "
        f"--checkpoint {shlex.quote(str(settings['checkpoint']))} "
        "--device cuda:0 --real --no-visualization"
    )


def build_navdp_server_background_command(config: InferenceLaunchConfig) -> str:
    """Start NavDP server as a managed background job in the planner pane."""

    server = build_navdp_server_command(config)
    return (
        "_stop_navdp_server() { "
        "if [ -n \"${NAVDP_SERVER_PID:-}\" ]; then "
        "kill -- \"-${NAVDP_SERVER_PID}\" 2>/dev/null || "
        "kill \"${NAVDP_SERVER_PID}\" 2>/dev/null || true; "
        "wait \"${NAVDP_SERVER_PID}\" 2>/dev/null || true; "
        "unset NAVDP_SERVER_PID; "
        "fi; "
        "}; "
        "trap _stop_navdp_server EXIT HUP; "
        f"{server} "
        "> >(sed -u 's/^/[NavDP server] /') "
        "2> >(sed -u 's/^/[NavDP server:stderr] /' >&2) & "
        "NAVDP_SERVER_PID=$!; "
        "echo \"[NavDP server] started in background as PID "
        "${NAVDP_SERVER_PID}; logs are routed to this pane\""
    )


def build_navdp_planner_pane_command(
    config: InferenceLaunchConfig, repo_root: Path
) -> str:
    """Run the planner in front, then stop its pane-local NavDP server."""

    planner = build_navdp_planner_command(config, repo_root)
    return (
        f"{planner}; "
        "NAVDP_PLANNER_STATUS=$?; "
        "_stop_navdp_server; "
        "trap - EXIT HUP; "
        "exit \"${NAVDP_PLANNER_STATUS}\""
    )


def build_fastlio_supervisor_command(config: InferenceLaunchConfig) -> str:
    settings = _component(config, "fastlio")
    return (
        "python -m gear_sonic.utils.inference.navdp.slam_supervisor "
        f"--profile {shlex.quote(config.config)} "
        f"--config-file {shlex.quote(str(settings['config']))}"
    )


def build_slam_debug_command(config: InferenceLaunchConfig) -> str:
    """Record the raw inputs and outputs needed to replay a FAST-LIO failure."""
    profile = _runtime_profile(config)
    topic_roles = ("lidar", "lidar_imu", "odometry", "registered_cloud")
    topics = " ".join(shlex.quote(profile.ros_topics[role]) for role in topic_roles)
    return f'ros2 bag record -o "$slam_debug_dir/rosbag" {topics}'


def build_sensor_gateway_command(config: InferenceLaunchConfig, repo_root: Path) -> str:
    fastlio = _component(config, "fastlio")
    ros_mode = "" if config.keyboard_planner and not config.sim else "--no-enable-ros "
    setup = (
        "unset COLCON_CURRENT_PREFIX AMENT_PREFIX_PATH CMAKE_PREFIX_PATH; "
        "source /opt/ros/humble/setup.bash && "
        f"export LD_LIBRARY_PATH={shlex.quote(str(fastlio['livox_sdk_lib']))}:$LD_LIBRARY_PATH && "
        f"source {shlex.quote(str(fastlio['workspace']))}/install/setup.bash && "
        f"cd {shlex.quote(str(repo_root))} && source .venv_teleop/bin/activate && "
        f"export PYTHONPATH={shlex.quote(str(repo_root))}:$PYTHONPATH; "
    )
    depth_anything_flag = (
        "--enable-depth-anything "
        if _depth_anything_required(config)
        else "--no-enable-depth-anything "
    )
    gateway = (
        "python -m gear_sonic.runtime.gateway.services.sensor "
        f"--profile {shlex.quote(config.config)} "
        f"{depth_anything_flag}"
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
                "python -m gear_sonic.utils.operator.cv_viewer "
                f"--profile {shlex.quote(config.config)}",
                "/tmp/sonic_opencv_viewer.log",
            )
        )
    depth_anything_required = _depth_anything_required(config)
    if depth_anything_required:
        background_commands.append(
            (
                "depth_anything_pid",
                build_depth_anything_command(config, repo_root),
                "/dev/null",
            )
        )
    if not background_commands:
        return setup + gateway

    ready_file_setup = (
        f"rm -f {shlex.quote(str(DEPTH_ANYTHING_READY_FILE))}; "
        if depth_anything_required
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
        + f"rm -f {shlex.quote(str(DEPTH_ANYTHING_READY_FILE))}; "
        + "(exit $gateway_status)"
    )


def run_readiness_gate(
    config: InferenceLaunchConfig, repo_root: Path, stage: Literal["lidar", "navigation"]
) -> bool:
    timeout = (
        config.lidar_ready_timeout_s
        if stage == "lidar"
        else config.navigation_ready_timeout_s
    )
    fastlio = _component(config, "fastlio")
    gateway_requirement = "--require-sensor-gateway " if stage == "navigation" else ""
    command = (
        "unset COLCON_CURRENT_PREFIX AMENT_PREFIX_PATH CMAKE_PREFIX_PATH; "
        "source /opt/ros/humble/setup.bash && "
        f"source {shlex.quote(str(fastlio['workspace']))}/install/setup.bash && "
        f"cd {shlex.quote(str(repo_root))} && source .venv_teleop/bin/activate && "
        f"python -m gear_sonic.utils.inference.navdp.readiness --stage {stage} "
        f"--profile {shlex.quote(config.config)} "
        f"--timeout {timeout} "
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


def _lavira_api_key_errors(env_file: Path) -> tuple[str, ...]:
    """Report role keys that cannot resolve from process or dotenv settings."""

    def available(*names: str) -> bool:
        return any(
            bool(os.environ.get(name, "").strip())
            or _dotenv_has_nonempty_value(env_file, name)
            for name in names
        )

    errors = []
    if not available("LAVIRA_LA_API_KEY", "DASHSCOPE_API_KEY"):
        errors.append(
            "LaViRA LA API key is missing; set LAVIRA_LA_API_KEY or "
            "DASHSCOPE_API_KEY in the environment or .env.local"
        )
    if not available("LAVIRA_VA_API_KEY", "DASHSCOPE_API_KEY"):
        errors.append(
            "LaViRA VA API key is missing; set LAVIRA_VA_API_KEY or "
            "DASHSCOPE_API_KEY in the environment or .env.local"
        )
    return tuple(errors)


def _check_prerequisites(config: InferenceLaunchConfig):
    """Verify that required tools and venvs exist."""
    errors = []
    profile = _runtime_profile(config)

    if not shutil.which("tmux"):
        errors.append("tmux is not installed. Install with: sudo apt install tmux")

    repo_root = Path(__file__).resolve().parent.parent.parent

    if not (repo_root / ".venv_inference" / "bin" / "activate").exists():
        errors.append(
            ".venv_inference not found. Run: bash install_scripts/install_inference.sh"
        )
    if not (repo_root / ".venv_teleop" / "bin" / "activate").exists():
        errors.append(".venv_teleop not found. Run: bash install_scripts/install_pico.sh")

    depth_anything_required = _depth_anything_required(config)
    if depth_anything_required:
        depth_settings = _runtime_profile(config).component("depth_anything")
        for path, label in (
            (Path(str(depth_settings["root"])), "Depth Anything metric source"),
            (Path(str(depth_settings["checkpoint"])), "Depth Anything Base checkpoint"),
            (
                repo_root / ".venv_depth_anything" / "bin" / "python",
                "Depth Anything Python",
            ),
        ):
            if not path.exists():
                errors.append(f"{label} not found: {path}")

    if config.planner_input == "lavira":
        errors.extend(_lavira_api_key_errors(repo_root / ".env.local"))
        lavira = profile.component("lavira")
        navdp = profile.component("navdp")
        fastlio = profile.component("fastlio")
        if not str(lavira["mission"]).strip():
            errors.append("components.lavira.mission is required")
        navigation_mode = str(lavira["navigation_mode"])
        if navigation_mode not in {"vln", "object_nav"}:
            errors.append(
                "components.lavira.navigation_mode must be vln or object_nav"
            )
        for endpoint_name in ("la_base_url", "va_base_url"):
            if not str(lavira[endpoint_name]).strip():
                errors.append(f"components.lavira.{endpoint_name} is required")
        for path, label in (
            (
                Path(str(navdp["root"])) / "eval" / "src" / "policy_server.py",
                "X-NavDP server",
            ),
            (Path(str(navdp["checkpoint"])), "NavDP checkpoint"),
            (
                Path(str(fastlio["workspace"])) / "install" / "setup.bash",
                "FAST-LIO workspace",
            ),
            (Path("/opt/ros/humble/setup.bash"), "ROS2 Humble"),
        ):
            if not path.exists():
                errors.append(f"{label} not found: {path}")
        camera_endpoint = profile.endpoint("camera_server")
        try:
            with socket.create_connection(
                (camera_endpoint.host, camera_endpoint.port), timeout=1.0
            ):
                pass
        except OSError:
            errors.append(
                "robot camera is not reachable at "
                f"{camera_endpoint.host}:{camera_endpoint.port}"
            )

    if config.base_pose_enabled:
        base_pose = profile.component("base_pose")
        for key in ("task", "target_prompt", "surface_prompt"):
            if not str(base_pose[key]).strip():
                errors.append(f"components.base_pose.{key} is required")
        for relative_path, label in (
            (base_pose["raw_yoloe_model_path"], "YOLOE model"),
            (base_pose["camera_intrinsics_path"], "camera intrinsics"),
        ):
            path = Path(relative_path)
            if not path.is_absolute():
                path = repo_root / path
            if not path.is_file():
                errors.append(f"Base-Pose {label} not found: {path}")
        inference_python = repo_root / ".venv_inference" / "bin" / "python"
        if inference_python.is_file():
            clip_check = subprocess.run(
                [str(inference_python), "-c", "import clip"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if clip_check.returncode != 0:
                errors.append(
                    "Base-Pose YOLOE text recovery requires the pinned "
                    "Ultralytics CLIP fork. Run: bash tools/yoloe26m/setup.sh"
                )
        mobileclip_path = repo_root / "mobileclip2_b.ts"
        if not mobileclip_path.is_file():
            errors.append(
                "Base-Pose MobileCLIP2 text encoder not found: "
                f"{mobileclip_path}. Run: bash tools/yoloe26m/setup.sh"
            )

    deploy_dir = repo_root / "gear_sonic_deploy"
    if not (deploy_dir / "deploy.sh").exists():
        errors.append(
            f"gear_sonic_deploy/deploy.sh not found at {deploy_dir}. "
            "Ensure the deploy directory is set up."
        )
    else:
        try:
            resolve_deploy_policy(profile.component("deploy"), deploy_dir)
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


def _clear_stale_navdp_processes() -> None:
    """Remove NavDP workers that survived an earlier tmux session.

    A dead pane can leave both the planner's ZMQ publishers and the policy
    server alive with a deleted pseudo-terminal.  Besides occupying the fixed
    ports, that planner keeps publishing its last in-memory SLAM map forever.
    Match only the two repository-owned entry points so unrelated Python and
    conda jobs are left untouched.
    """
    patterns = (
        r"(^|/)(python|python3) -m gear_sonic\.utils\.inference\.navdp\.service( |$)",
        r"(^|/)(python|python3) -m eval\.src\.policy_server( |$)",
        r"(^|/)(python|python3) ([^ ]*/)?gear_sonic/scripts/navdp_planner\.py( |$)",
    )
    for pattern in patterns:
        subprocess.run(["pkill", "-TERM", "-f", pattern], capture_output=True)
    time.sleep(0.5)
    for pattern in patterns:
        subprocess.run(["pkill", "-KILL", "-f", pattern], capture_output=True)


def _worker_pane_names(config: InferenceLaunchConfig) -> tuple[str, ...]:
    names = ["deploy", "vla"]
    if config.keyboard_planner:
        names.extend(("planner_input", "navdp", "planner_executor"))
        if config.base_pose_enabled:
            names.append("base_pose")
    return tuple(names)


def _tmux(*args: str, output: bool = False) -> str:
    result = subprocess.run(
        ["tmux", *args],
        check=True,
        capture_output=output,
        text=output,
    )
    return result.stdout.strip() if output else ""


def _pane_id(target: str) -> str:
    return _tmux(
        "display-message",
        "-p",
        "-t",
        target,
        "#{pane_id}",
        output=True,
    )


def _split_pane(
    target: str,
    shell: tuple[str, ...],
    *split_args: str,
) -> str:
    return _tmux(
        "split-window",
        *split_args,
        "-t",
        target,
        "-P",
        "-F",
        "#{pane_id}",
        *shell,
        output=True,
    )


def _create_tmux_session(config: InferenceLaunchConfig) -> dict[str, str]:
    bash = shutil.which("bash") or "/bin/bash"
    shell = (bash, "--noprofile", "--norc")
    _tmux(
        "new-session",
        "-d",
        "-x",
        "240",
        "-y",
        "60",
        "-s",
        SESSION_NAME,
        "-n",
        "overview",
        *shell,
    )
    _tmux("set-option", "-t", SESSION_NAME, "mouse", "on")
    _tmux("bind-key", "-T", "root", "C-\\", "kill-session")

    # Overview: compact performance/control panes on the left and a full-height
    # event stream on the right.  The left side is kept wide enough for the
    # existing SensorGateway dashboard at the launcher's 240-column baseline.
    performance = _pane_id(f"{SESSION_NAME}:overview.0")
    events = _split_pane(performance, shell, "-h", "-p", "58")
    control = _split_pane(performance, shell, "-v", "-p", "42")
    panes = {
        "performance": performance,
        "control": control,
        "events": events,
    }

    worker_names = _worker_pane_names(config)
    _tmux(
        "new-window",
        "-d",
        "-t",
        SESSION_NAME,
        "-n",
        "workers",
        *shell,
    )
    panes[worker_names[0]] = _pane_id(f"{SESSION_NAME}:workers.0")
    for name in worker_names[1:]:
        panes[name] = _split_pane(f"{SESSION_NAME}:workers", shell, "-h")
    _tmux("select-layout", "-t", f"{SESSION_NAME}:workers", "tiled")

    for window in ("overview", "workers"):
        target = f"{SESSION_NAME}:{window}"
        _tmux("set-option", "-w", "-t", target, "remain-on-exit", "on")
        _tmux("set-option", "-w", "-t", target, "pane-border-status", "top")
        _tmux(
            "set-option",
            "-w",
            "-t",
            target,
            "pane-border-format",
            " #[fg=colour45,bold]#{pane_title}#[default] ",
        )
    for name, pane in panes.items():
        _tmux("select-pane", "-t", pane, "-T", PANE_TITLES[name])

    _tmux("select-window", "-t", f"{SESSION_NAME}:overview")
    _tmux("select-pane", "-t", panes["control"])
    time.sleep(5)
    return panes


def _send_to_pane(pane_id: str, cmd: str, wait: float = 1.0):
    _tmux("send-keys", "-t", pane_id, cmd, "C-m")
    time.sleep(wait)


def _start_in_pane(
    panes: dict[str, str],
    name: str,
    label: str,
    command: str,
    *,
    wait: float = 1.0,
    required: bool = False,
) -> None:
    print(f"Starting {label} ({name})...")
    _send_to_pane(panes[name], command, wait=wait)
    if required and not _check_pane_alive(panes[name]):
        raise RuntimeError(
            f"{label} exited during startup; inspect "
            "outputs/logs/inference/navdp.log for the fatal error"
        )


def _check_pane_alive(pane_id: str) -> bool:
    result = subprocess.run(
        ["tmux", "list-panes", "-t", pane_id, "-F", "#{pane_dead}"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() != "1"


def main(config: InferenceLaunchConfig):
    repo_root = Path(__file__).resolve().parent.parent.parent
    profile = _runtime_profile(config)

    _check_prerequisites(config)
    _kill_existing_session()
    _clear_stale_fastlio_processes()
    _clear_stale_navdp_processes()

    print(
        f"Launching {SESSION_NAME} "
        f"({'simulation' if config.sim else 'real robot'}) from {config.config}"
    )
    camera_endpoint = profile.endpoint("camera_server")

    base_pose_runtime_enabled = config.keyboard_planner and config.base_pose_enabled
    panes = _create_tmux_session(config)
    print(f"Created tmux session: {SESSION_NAME}")

    # --- Optional window: MuJoCo Simulator ---
    if config.sim:
        _tmux("new-window", "-t", SESSION_NAME, "-n", "sim")
        sim_cmd = (
            f"cd {repo_root} && "
            f"source .venv_sim/bin/activate && "
            f"python -m gear_sonic.utils.mujoco_sim.service "
            f"--enable-image-publish --enable-offscreen "
            f"--camera-port {camera_endpoint.port}"
        )
        sim_target = f"{SESSION_NAME}:sim"
        print("Starting MuJoCo simulator (window: sim)...")
        _send_to_pane(sim_target, sim_cmd, wait=3.0)

        _tmux("select-window", "-t", f"{SESSION_NAME}:overview")

    # Workers retain independent TTYs and scrollback in the second window.
    deploy_cmd = build_deploy_command(config, repo_root)
    _start_in_pane(panes, "deploy", "C++ deploy", deploy_cmd, wait=3.0)
    if not _check_pane_alive(panes["deploy"]):
        print("WARNING: C++ deploy pane may have failed to start.")

    # The overview window directly hosts the performance, control, and event
    # processes; start both Gateway boundaries before their clients.
    _start_in_pane(
        panes,
        "performance",
        "SensorGateway with background ROS/OpenCV/Depth Anything services",
        build_sensor_gateway_command(config, repo_root),
        wait=1.0,
    )
    if config.slam_debug and config.keyboard_planner and not config.sim:
        print("SLAM/IMU debug recording: outputs/slam_debug/<launch timestamp>/")
    _start_in_pane(
        panes,
        "events",
        "ControlGateway router",
        build_control_gateway_command(config, repo_root),
        wait=1.0,
    )
    if config.keyboard_planner:
        _start_in_pane(
            panes,
            "planner_executor",
            "shared Planner velocity executor",
            build_planner_velocity_executor_command(config, repo_root),
            wait=1.0,
        )
    if base_pose_runtime_enabled:
        _start_in_pane(
            panes,
            "base_pose",
            "Base-Pose agent",
            build_base_pose_agent_command(config, repo_root),
            wait=1.0,
        )

    inference_cmd = build_vla_inference_command(config, repo_root)
    _start_in_pane(panes, "vla", "VLA inference", inference_cmd, wait=1.0)

    _start_in_pane(
        panes,
        "control",
        "standalone operator CLI",
        build_operator_console_command(config, repo_root),
        wait=2.0,
    )

    if config.keyboard_planner:
        planner_input_cmd = build_planner_input_command(config, repo_root)
        commands = [
            ("planner_input", "LaViRA semantic planner", planner_input_cmd),
            (
                "navdp",
                "NavDP planner",
                build_navdp_planner_pane_command(config, repo_root),
            ),
        ]
        # Strictly serialized startup: the launcher does not dispatch a later
        # stage until real data has passed the previous readiness gate.
        print("Starting NavDP server in background (navdp)...")
        _send_to_pane(
            panes["navdp"],
            build_navdp_server_background_command(config),
            wait=1.0,
        )
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

        for name, label, command in commands:
            _start_in_pane(
                panes,
                name,
                label,
                command,
                required=name == "navdp",
            )

    if config.data_exporter:
        _tmux("new-window", "-t", SESSION_NAME, "-n", "data_exporter")
        exporter_cmd = build_data_exporter_command(config, repo_root)
        print("Starting data exporter (window: data_exporter)...")
        _send_to_pane(
            f"{SESSION_NAME}:data_exporter",
            exporter_cmd,
            wait=2.0,
        )
        _tmux("select-window", "-t", f"{SESSION_NAME}:overview")

    print(f"All components launched; attaching to tmux session {SESSION_NAME}")

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
