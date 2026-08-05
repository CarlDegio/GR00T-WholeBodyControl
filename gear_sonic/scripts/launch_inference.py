"""
All-in-one tmux launcher for SONIC VLA inference.

Starts the inference stack in a single tmux session:

    Window 0 — inference (4 panes):
    ┌───────────────────────┬───────────────────────┐
    │ Pane 0: C++ Deploy    │ Pane 1: VLA Inference │
    │ (gear_sonic_deploy)   │ (.venv_inference)     │
    ├───────────────────────┼───────────────────────┤
    │ Pane 2: Keyboard Pub  │ Pane 3: Keyboard Planner │
    │ (.venv_inference)     │ (.venv_inference)│
    └───────────────────────┴───────────────────────┘

    Window 1 — keyboard planner (only when --keyboard-planner is passed):
    ┌─────────────────────────────────────────────────┐
    │ Keyboard Planner (keyboard_planner_thread_server.py)│
    │ (.venv_inference)                                 │
    └─────────────────────────────────────────────────┘

    Window 2 — sim  (only when --sim is passed):
    ┌─────────────────────────────────────────────────┐
    │ MuJoCo Simulator (run_sim_loop.py)              │
    │ (.venv_sim)                                     │
    └─────────────────────────────────────────────────┘

Prerequisites:
    - tmux installed (sudo apt install tmux)
    - Virtual environments set up:
        bash install_scripts/install_inference.sh     -> .venv_inference
        bash install_scripts/install_data_collection.sh -> .venv_data_collection (optional, for recording)
    - gear_sonic_deploy built (see docs)
    - Isaac-GR00T PolicyServer running separately

Usage (from repo root — no venv activation needed):
    python gear_sonic/scripts/launch_inference.py                        # real robot
    python gear_sonic/scripts/launch_inference.py --sim                  # MuJoCo sim
    python gear_sonic/scripts/launch_inference.py --no-data-exporter     # no recording pane
"""

from dataclasses import dataclass
from pathlib import Path
import os
import shlex
import shutil
import signal
import socket
import base64
import subprocess
import sys
import textwrap
import time
from typing import Literal


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
    """Isaac-GR00T PolicyServer host."""

    policy_port: int = 5550
    """Isaac-GR00T PolicyServer port."""

    embodiment_tag: str = "unitree_g1_sonic"
    """Embodiment tag for policy inference."""

    prompt: str = "demo"
    """Language prompt for inference."""

    action_publish_rate: int = 50
    """Rate at which individual actions are published to the C++ control loop (Hz)."""

    action_horizon: int = 70
    """Action horizon of the VLA policy."""

    # Camera
    camera_host: str = "localhost"
    """Camera server host."""

    camera_port: int = 5555
    """Camera server port."""

    keyboard_planner: bool = True
    """Start planner input, rule-based safety, and MID-360 sidecars."""

    keyboard_planner_port: int = 5558
    """Keyboard planner sidecar port."""

    keyboard_planner_publish_rate: int = 20
    """Keyboard planner publish rate (Hz)."""

    keyboard_planner_host: str = "localhost"
    """Keyboard planner sidecar host."""

    planner_input: Literal["keyboard", "lavira", "base_pose"] = "keyboard"
    """SONIC planner command source used in pane 3."""

    base_pose_mode: Literal["rgb", "rgbd", "rgb_depth_query"] = "rgb"
    """Head-vision input experiment used by the base-pose planner."""

    base_pose_task: str = ""
    """Manipulation task for base-pose adjustment; defaults to ``prompt``."""

    base_pose_model: str = "gpt-5.6-sol"
    """Codex CLI vision model used by base-pose adjustment."""

    base_pose_reasoning_effort: str = "max"
    """Reasoning effort passed to the base-pose model."""

    base_pose_camera_stream: str = "ego_view"
    """Head RGB stream; aligned depth is read from ``<stream>_depth``."""

    base_pose_camera_height_m: float = 1.2
    """Head camera optical-center height above the ground (m)."""

    base_pose_camera_pitch_deg: float = -47.6
    """Head camera optical-axis pitch under the positive-upward convention."""

    base_pose_vertical_fov_deg: float = 55.2
    """Full head-camera vertical field of view."""

    base_pose_camera_forward_offset_m: float = 0.0
    """Configured camera optical-center forward offset from the base."""

    base_pose_camera_lateral_offset_m: float = 0.0
    """Configured camera optical-center left/right offset from the base."""

    base_pose_depth_port: int = 5564
    """Local pure LingBot RGB-D stream used by the two depth modes."""

    base_pose_camera_timeout_ms: int = 15000
    """Timeout for a fresh base-pose camera snapshot."""

    base_pose_codex_timeout_seconds: float = 600.0
    """Timeout for each base-pose Codex inference call."""

    base_pose_rotation_speed: float = 0.4
    """Fixed SONIC yaw speed used to execute model rotations (rad/s)."""

    base_pose_translation_speed: float = 0.3
    """Fixed SONIC forward/backward speed used for model translations (m/s)."""

    base_pose_transition_pause: float = 0.5
    """Planner IDLE duration between model-authored motion steps (s)."""

    base_pose_final_stop_count: int = 3
    """Number of zero-motion publications after completion or cancellation."""

    base_pose_depth_visual_max_m: float = 3.0
    """Maximum range in the attached fixed-scale color depth image."""

    base_pose_output_root: str = "outputs/base_pose_adjustment"
    """Root for RGB/depth/prompts/results and execution diagnostics."""

    base_pose_lingbot_model: str = "robbyant/lingbot-depth-pretrain-vitl-14-v0.5"
    """LingBot-Depth checkpoint used by base-pose depth modes."""

    base_pose_lingbot_root: str = "/home/user/Project/lingbot-depth"
    """LingBot-Depth checkout on the deployment machine."""

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

    lavira_qwenvl_base_url: str = (
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    )
    """DashScope OpenAI-compatible endpoint; key comes from DASHSCOPE_API_KEY."""

    lavira_warmup: bool = True
    """Run one discarded Luna request in the background at startup."""

    lavira_debug: bool = False
    """Enable LaViRA debug logging."""

    lavira_host: str = "*"
    """LaViRA REASAN publisher bind host."""

    lavira_planner_hz: float = 20.0
    """LaViRA velocity publication rate (Hz)."""

    lavira_transition_pause: float = 0.5
    """LaViRA stop duration between rotation and translation (s)."""

    lavira_final_stop_count: int = 3
    """Number of final zero-velocity messages published by LaViRA."""

    lavira_max_speed: float = 0.5
    """LaViRA maximum translation speed (m/s)."""

    lavira_max_duration: float = 30.0
    """LaViRA maximum command duration (s)."""

    lavira_max_abs_yaw: float = 3.141592653589793
    """LaViRA maximum relative yaw (rad)."""

    lavira_camera_timeout_ms: int = 15000
    """LaViRA pure LingBot RGB-D receive timeout (ms)."""

    lavira_depth_port: int = 5564
    """Local pure LingBot completed RGB-D stream consumed by LaViRA."""

    lavira_codex_timeout_seconds: float = 180.0
    """LaViRA Codex policy timeout (s)."""

    lavira_min_confidence: float = 0.6
    """Minimum Codex target confidence accepted by LaViRA."""

    lavira_rotation_speed: float = 0.4
    """LaViRA automatic rotation speed (rad/s)."""

    lavira_forward_speed: float = 0.3
    """LaViRA automatic forward speed (m/s)."""

    lavira_target_standoff_distance: float = 0.0
    """LaViRA target standoff distance (m)."""

    lavira_max_direct_travel: float = 8.0
    """LaViRA maximum automatic direct travel distance (m)."""

    lavira_output_root: str = "outputs/object_nav"
    """Directory where LaViRA stores ObjectNav diagnostics."""

    reasan_ray_port: int = 5562
    """MID-360 ActorRay publisher port."""

    reasan_avoidance: bool = True
    """Run the MID-360 rule-based forward protective stop."""

    reasan_planner_port: int = 5563
    """Filtered SONIC planner publisher port consumed by VLA inference."""

    reasan_radar_interface: str = "enx6c1ff7bed314"
    """Network interface connected to the G1 MID-360."""

    # Data exporter (optional recording during inference)
    data_exporter: bool = True
    """Start the data exporter pane for recording during inference."""

    data_exporter_frequency: int = 50
    """Data collection frequency (Hz) for the data exporter."""

    record_chest_camera: bool = False
    """Record chest camera stream (chest_view) in the dataset."""

    task_prompt: str = ""
    """Task prompt for the data exporter. Defaults to the inference prompt if empty."""

    dataset_name: str = ""
    """Dataset name for the data exporter. Leave empty to auto-generate."""


SESSION_NAME = "sonic_inference"


def uses_reasan_avoidance(config: InferenceLaunchConfig) -> bool:
    """Base-pose adjustment intentionally bypasses MID-360/REASAN."""
    return config.reasan_avoidance and config.planner_input != "base_pose"


def _base_pose_planner_command(config: InferenceLaunchConfig, repo_root: Path) -> str:
    task = config.base_pose_task.strip() or config.prompt.strip()
    quoted_root = shlex.quote(str(repo_root))
    planner = (
        ".venv_inference/bin/python gear_sonic/scripts/base_pose_planner.py "
        f"--task {shlex.quote(task)} "
        f"--mode {shlex.quote(config.base_pose_mode)} "
        f"--model {shlex.quote(config.base_pose_model)} "
        f"--reasoning-effort {shlex.quote(config.base_pose_reasoning_effort)} "
        f"--host {shlex.quote(config.keyboard_planner_host)} "
        f"--port {config.keyboard_planner_port} "
        f"--planner-hz {config.keyboard_planner_publish_rate} "
        f"--transition-pause {config.base_pose_transition_pause} "
        f"--final-stop-count {config.base_pose_final_stop_count} "
        f"--rotation-speed {config.base_pose_rotation_speed} "
        f"--translation-speed {config.base_pose_translation_speed} "
        f"--camera-timeout-ms {config.base_pose_camera_timeout_ms} "
        f"--camera-stream {shlex.quote(config.base_pose_camera_stream)} "
        f"--camera-height-m {config.base_pose_camera_height_m} "
        f"--camera-pitch-deg {config.base_pose_camera_pitch_deg} "
        f"--vertical-fov-deg {config.base_pose_vertical_fov_deg} "
        f"--camera-forward-offset-m {config.base_pose_camera_forward_offset_m} "
        f"--camera-lateral-offset-m {config.base_pose_camera_lateral_offset_m} "
        f"--depth-visual-max-m {config.base_pose_depth_visual_max_m} "
        f"--codex-timeout-seconds {config.base_pose_codex_timeout_seconds} "
        f"--output-root {shlex.quote(config.base_pose_output_root)}"
    )
    if config.base_pose_mode == "rgb":
        return (
            f"cd {quoted_root} && "
            f"{planner} --camera-host {shlex.quote(config.camera_host)} "
            f"--camera-port {config.camera_port}"
        )

    return (
        f"cd {quoted_root} && "
        f"ready_file=/tmp/sonic_base_pose_lingbot_ready_$$; rm -f $ready_file; "
        f"PYTHONPATH={quoted_root} .venv_lingbot_depth/bin/python "
        f"gear_sonic/scripts/run_lingbot_depth_viewer.py "
        f"--camera-host {shlex.quote(config.camera_host)} "
        f"--camera-port {config.camera_port} "
        f"--stream-name {shlex.quote(config.base_pose_camera_stream)} "
        f"--publish-port {config.base_pose_depth_port} "
        f"--ready-file $ready_file "
        f"--model {shlex.quote(config.base_pose_lingbot_model)} "
        f"--lingbot-root {shlex.quote(config.base_pose_lingbot_root)} & "
        f"viewer_pid=$!; "
        f"trap 'kill $viewer_pid 2>/dev/null; rm -f $ready_file' EXIT; "
        f"while [ ! -f $ready_file ]; do "
        f"kill -0 $viewer_pid 2>/dev/null || {{ wait $viewer_pid; exit 1; }}; "
        f"sleep 0.2; done; "
        f"{planner} --camera-host 127.0.0.1 "
        f"--camera-port {config.base_pose_depth_port}"
    )


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
    if config.planner_input == "base_pose":
        return _base_pose_planner_command(config, repo_root)

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
    quoted_camera_host = shlex.quote(config.camera_host)
    return (
        f"cd {quoted_root} && "
        f"{local_env}"
        f"ready_file=/tmp/sonic_lingbot_ready_$$; rm -f $ready_file; "
        f"PYTHONPATH={quoted_root} .venv_lingbot_depth/bin/python "
        f"gear_sonic/scripts/run_lingbot_depth_viewer.py "
        f"--camera-host {quoted_camera_host} --camera-port {config.camera_port} "
        f"--publish-port {config.lavira_depth_port} --ready-file $ready_file & "
        f"viewer_pid=$!; "
        f"trap 'kill $viewer_pid 2>/dev/null; rm -f $ready_file' EXIT; "
        f"while [ ! -f $ready_file ]; do "
        f"kill -0 $viewer_pid 2>/dev/null || {{ wait $viewer_pid; exit 1; }}; "
        f"sleep 0.2; done; "
        f".venv_inference/bin/python gear_sonic/scripts/lavira_planner.py "
        f"--mission {shlex.quote(config.lavira_mission)} "
        f"--global-target {shlex.quote(config.lavira_global_target)} "
        f"--model {shlex.quote(config.lavira_model)} "
        f"{vision_backend}{debug}{warmup}--host {shlex.quote(config.lavira_host)} "
        f"--port {config.keyboard_planner_port} "
        f"--planner-hz {config.lavira_planner_hz} "
        f"--transition-pause {config.lavira_transition_pause} "
        f"--final-stop-count {config.lavira_final_stop_count} "
        f"--max-speed {config.lavira_max_speed} "
        f"--max-duration {config.lavira_max_duration} "
        f"--max-abs-yaw {config.lavira_max_abs_yaw} "
        f"--camera-host 127.0.0.1 "
        f"--camera-port {config.lavira_depth_port} "
        f"--camera-timeout-ms {config.lavira_camera_timeout_ms} "
        f"--codex-timeout-seconds {config.lavira_codex_timeout_seconds} "
        f"--min-confidence {config.lavira_min_confidence} "
        f"--rotation-speed {config.lavira_rotation_speed} "
        f"--forward-speed {config.lavira_forward_speed} "
        f"--target-standoff-distance {config.lavira_target_standoff_distance} "
        f"--max-direct-travel {config.lavira_max_direct_travel} "
        f"--output-root {shlex.quote(config.lavira_output_root)}"
    )


def build_reasan_planner_command(config: InferenceLaunchConfig, repo_root: Path) -> str:
    """Build rule-based safety or a direct planner-input-to-SONIC relay."""
    if config.planner_input == "base_pose":
        return (
            f"cd {shlex.quote(str(repo_root))} && "
            f"source .venv_teleop/bin/activate && "
            f"python gear_sonic/scripts/lavira_sonic_relay.py "
            f"--source tcp://127.0.0.1:{config.keyboard_planner_port} "
            f"--output 'tcp://*:{config.reasan_planner_port}' --hz 20"
        )
    common = (
        f"cd {repo_root} && "
        f"source .venv_teleop/bin/activate && "
        f"python gear_sonic/scripts/reasan_planner.py "
        f"--keyboard-endpoint tcp://127.0.0.1:{config.keyboard_planner_port} "
        f"--output-endpoint 'tcp://*:{config.reasan_planner_port}' "
    )
    if not config.reasan_avoidance:
        return (
            f"cd {repo_root} && "
            f"source .venv_teleop/bin/activate && "
            f"python gear_sonic/scripts/lavira_sonic_relay.py "
            f"--source tcp://127.0.0.1:{config.keyboard_planner_port} "
            f"--output 'tcp://*:{config.reasan_planner_port}' --hz 20"
        )
    return (
        common
        + f"--ray-endpoint tcp://127.0.0.1:{config.reasan_ray_port} "
        + f"--camera-host {shlex.quote(config.camera_host)} "
        + f"--camera-port {config.camera_port}"
    )


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
        errors.append(
            ".venv_teleop not found. Run: bash install_scripts/install_pico.sh"
        )

    if config.planner_input == "lavira":
        if not config.lavira_mission.strip():
            errors.append("--lavira-mission is required when --planner-input lavira")
        if not config.lavira_global_target.strip():
            errors.append(
                "--lavira-global-target is required when --planner-input lavira"
            )
    if config.planner_input == "base_pose":
        if not config.keyboard_planner:
            errors.append(
                "--keyboard-planner is required for --planner-input base_pose"
            )
        if not (config.base_pose_task.strip() or config.prompt.strip()):
            errors.append(
                "--base-pose-task or --prompt is required when --planner-input base_pose"
            )
        if (
            config.base_pose_mode != "rgb"
            and not (repo_root / ".venv_lingbot_depth" / "bin" / "python").exists()
        ):
            errors.append(
                ".venv_lingbot_depth not found; it is required by base-pose "
                "rgbd and rgb_depth_query modes"
            )

    deploy_dir = repo_root / "gear_sonic_deploy"
    if not (deploy_dir / "deploy.sh").exists():
        errors.append(
            f"gear_sonic_deploy/deploy.sh not found at {deploy_dir}. "
            "Ensure the deploy directory is set up."
        )

    if config.data_exporter:
        if not (repo_root / ".venv_data_collection" / "bin" / "activate").exists():
            errors.append(
                ".venv_data_collection not found (needed for data exporter). Run: "
                "bash install_scripts/install_data_collection.sh"
            )

    if config.sim and not (repo_root / ".venv_sim" / "bin" / "activate").exists():
        errors.append(".venv_sim not found. Set up the simulation venv first.")

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


def _parse_pane_ids(output: str) -> list[str]:
    indexed = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) == 2:
            indexed[int(fields[0])] = fields[1]
    if len(indexed) != 6 or sorted(indexed) != list(range(6)):
        raise RuntimeError(f"expected 6 tmux panes, found {len(indexed)}")
    return [indexed[index] for index in range(6)]


def _create_tmux_session() -> list[str]:
    bash = shutil.which("bash") or "/bin/bash"
    subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-x",
            "240",
            "-y",
            "60",
            "-s",
            SESSION_NAME,
            bash,
            "--noprofile",
            "--norc",
        ],
        check=True,
    )
    subprocess.run(
        ["tmux", "set-option", "-t", SESSION_NAME, "-g", "mouse", "on"],
    )
    subprocess.run(
        [
            "tmux",
            "set-option",
            "-t",
            f"{SESSION_NAME}:0",
            "-w",
            "remain-on-exit",
            "on",
        ],
        check=True,
    )
    subprocess.run(
        ["tmux", "bind-key", "-T", "root", "C-\\", "kill-session"],
    )
    subprocess.run(
        ["tmux", "rename-window", "-t", f"{SESSION_NAME}:0", "inference"],
    )

    # Build two rows first, then split each row into three columns. This also
    # works when the detached tmux server initially reports a short terminal.
    top_pane = subprocess.run(
        ["tmux", "display-message", "-p", "-t", f"{SESSION_NAME}:0.0", "#{pane_id}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    bottom_pane = subprocess.run(
        [
            "tmux",
            "split-window",
            "-v",
            "-t",
            top_pane,
            "-P",
            "-F",
            "#{pane_id}",
            bash,
            "--noprofile",
            "--norc",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    for row_pane in (top_pane, bottom_pane):
        for _ in range(2):
            subprocess.run(
                [
                    "tmux",
                    "split-window",
                    "-h",
                    "-t",
                    row_pane,
                    bash,
                    "--noprofile",
                    "--norc",
                ],
                check=True,
            )
    subprocess.run(
        ["tmux", "select-layout", "-t", f"{SESSION_NAME}:0", "tiled"], check=True
    )

    pane_output = subprocess.run(
        [
            "tmux",
            "list-panes",
            "-t",
            f"{SESSION_NAME}:0",
            "-F",
            "#{pane_index} #{pane_id}",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    pane_ids = _parse_pane_ids(pane_output)
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

    exporter_prompt = config.task_prompt if config.task_prompt else config.prompt

    print("=" * 60)
    print("  SONIC VLA Inference Launcher")
    print("=" * 60)
    print(f"  Mode:            {'Simulation' if config.sim else 'Real Robot'}")
    print(f"  PolicyServer:    {config.policy_host}:{config.policy_port}")
    print(f"  Embodiment:      {config.embodiment_tag}")
    print(f"  Prompt:          {config.prompt}")
    print(f"  Action rate:     {config.action_publish_rate} Hz")
    print(f"  Action horizon:  {config.action_horizon}")
    print(f"  Camera:          {config.camera_host}:{config.camera_port}")
    print(f"  Data exporter:   {'Yes' if config.data_exporter else 'No'}")
    if config.data_exporter:
        print(f"    DC frequency:  {config.data_exporter_frequency} Hz")
        print(f"    Task prompt:   {exporter_prompt}")
        print(f"    Chest camera:  {'Yes' if config.record_chest_camera else 'No'}")
    print(f"  PC IP:           {_get_local_ip()}")
    print("=" * 60)

    pane_ids = _create_tmux_session()
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
    deploy_mode = "sim" if config.sim else "real"
    deploy_cmd = (
        f"cd {repo_root / 'gear_sonic_deploy'} && "
        f"./deploy.sh "
        f"--input-type {config.deploy_input_type} "
        f"--zmq-host {config.deploy_zmq_host} "
        f"--hand-type dex1 "
        f"--dex1-kp 1.0 "
        f"--dex1-kd 0.05 "
    )
    if config.deploy_checkpoint:
        deploy_cmd += f"--cp {config.deploy_checkpoint} "
    if config.deploy_obs_config:
        deploy_cmd += f"--obs-config {config.deploy_obs_config} "
    if config.deploy_planner:
        deploy_cmd += f"--planner {config.deploy_planner} "
    if config.deploy_motion_data:
        deploy_cmd += f"--motion-data {config.deploy_motion_data} "
    if config.deploy_output_type:
        deploy_cmd += f"--output-type {config.deploy_output_type} "
    deploy_cmd += deploy_mode

    print("Starting C++ deploy (pane 0)...")
    _send_to_pane(pane_ids[0], deploy_cmd, wait=3.0)

    if not _check_pane_alive(pane_ids[0]):
        print("WARNING: C++ deploy pane may have failed to start.")

    # --- Pane 1 (top-right): VLA Inference ---
    inference_cmd = (
        f"cd {repo_root} && "
        f"source .venv_inference/bin/activate && "
        f"python gear_sonic/scripts/run_vla_inference.py "
        f"--host {config.policy_host} "
        f"--port {config.policy_port} "
        f"--embodiment-tag {config.embodiment_tag} "
        f"--prompt '{config.prompt}' "
        f"--action-publish-rate {config.action_publish_rate} "
        f"--action-horizon {config.action_horizon} "
        f"--camera-host {config.camera_host} "
        f"--camera-port {config.camera_port} "
        f"--planner-relay-zmq-host localhost "
        f"--planner-relay-zmq-port {config.reasan_planner_port}"
    )
    print("Starting VLA inference (pane 1)...")
    _send_to_pane(pane_ids[2], inference_cmd, wait=1.0)

    # --- Pane 2 (bottom-left): Keyboard Publisher ---
    keyboard_script = textwrap.dedent(
        """\
        import zmq, time
        ctx = zmq.Context()
        pub = ctx.socket(zmq.PUB)
        pub.bind('tcp://localhost:5580')
        time.sleep(0.5)
        print('Keyboard publisher ready. Keys: p=pause, k=start/stop, i=pose mode, o=planner mode, [/]=toggle hands, t=prompt')
        while True:
            key = input()
            if key.startswith('t '):
                pub.send_string('prompt:' + key[2:])
                print('Sent prompt: ' + key[2:])
            else:
                pub.send_string(key)
                print('Sent: ' + key)
    """
    )
    encoded = base64.b64encode(keyboard_script.encode()).decode()
    keyboard_cmd = (
        f"cd {repo_root} && "
        f"source .venv_teleop/bin/activate && "
        f"python -c \"import base64;exec(base64.b64decode('{encoded}'))\""
    )

    print("Starting keyboard publisher (pane 2)...")
    _send_to_pane(pane_ids[1], keyboard_cmd, wait=2.0)

    # --- Panes 3-5: planner input, rule-based safety, and MID-360 ---
    if config.keyboard_planner:
        reasan_keyboard_cmd = build_planner_input_command(config, repo_root)
        reasan_planner_cmd = build_reasan_planner_command(config, repo_root)
        radar_cmd = (
            f"cd {repo_root} && "
            f"source .venv_teleop/bin/activate && "
            f"python tools/mid360_reasan_open3d.py "
            f"--interface {config.reasan_radar_interface} "
            f"--min-range 0.3 --filter-ground "
            f"--ray-median-window 5 "
            f"--zmq-endpoint 'tcp://*:{config.reasan_ray_port}'"
        )
        if config.planner_input == "lavira":
            print("Starting LaViRA planner (pane 3)...")
        elif config.planner_input == "base_pose":
            print("Starting GPT base-pose adjustment planner (pane 3)...")
        else:
            print("Starting REASEN keyboard (pane 3)...")
        _send_to_pane(pane_ids[3], reasan_keyboard_cmd, wait=1.0)
        if uses_reasan_avoidance(config):
            print("Starting REASEN rule-based safety planner (pane 4)...")
        elif config.planner_input == "base_pose":
            print("Starting stateless base-pose-to-SONIC relay (pane 4)...")
        else:
            print("Starting direct LaViRA-to-SONIC relay (pane 4)...")
        _send_to_pane(pane_ids[4], reasan_planner_cmd, wait=1.0)
        if uses_reasan_avoidance(config):
            print("Starting MID-360 ActorRay publisher (pane 5)...")
            _send_to_pane(pane_ids[5], radar_cmd, wait=2.0)
        elif config.planner_input == "base_pose":
            print("Base-pose direct relay selected; MID-360 pane left idle.")
        else:
            print("REASEN avoidance disabled; MID-360 pane left idle.")

    if config.data_exporter:
        subprocess.run(
            ["tmux", "new-window", "-t", SESSION_NAME, "-n", "data_exporter"],
        )
        exporter_cmd = (
            f"cd {repo_root} && "
            f"source .venv_data_collection/bin/activate && "
            f"python gear_sonic/scripts/run_data_exporter.py "
            f"--task-prompt '{exporter_prompt}' "
            f"--data-collection-frequency {config.data_exporter_frequency} "
            f"--camera-host {config.camera_host} "
            f"--camera-port {config.camera_port}"
        )
        if config.dataset_name:
            exporter_cmd += f" --dataset-name '{config.dataset_name}'"
        if config.record_chest_camera:
            exporter_cmd += " --record-chest-camera"
        subprocess.run(
            [
                "tmux",
                "send-keys",
                "-t",
                f"{SESSION_NAME}:data_exporter",
                exporter_cmd,
                "C-m",
            ],
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
    print("    Pane 1: SONIC Keyboard Publisher")
    print("    Pane 2: VLA Inference")
    if config.planner_input == "lavira":
        print("    Pane 3: LaViRA AgentNav Planner")
    elif config.planner_input == "base_pose":
        print(f"    Pane 3: GPT Base Pose Planner ({config.base_pose_mode})")
    else:
        print("    Pane 3: REASEN Keyboard")
    if uses_reasan_avoidance(config):
        print("    Pane 4: REASEN Rule-Based Safety Planner")
        print("    Pane 5: MID-360 ActorRay/IMU")
    elif config.planner_input == "base_pose":
        print("    Pane 4: Stateless Base-Pose-to-SONIC Relay")
        print("    Pane 5: Idle (base_pose does not use MID-360/REASAN)")
    else:
        print("    Pane 4: Direct LaViRA-to-SONIC Relay")
        print("    Pane 5: Idle (REASEN/MID-360 disabled)")
    if config.data_exporter:
        print("    Window 'data_exporter':")
        print("      Data Exporter (.venv_data_collection)")
        print()
    print()
    print("  ** deploy.sh (pane 0) is waiting for confirmation --")
    print("     click on pane 0 and press Enter to proceed **")
    print()
    print("  Planner workflow:")
    if config.planner_input == "base_pose":
        print("    1. In pane 1: k starts the C++ loop in PLANNER mode")
        print("    2. In pane 3: N plans; Space cancels+stops; X exits source")
        print("    3. Press N again only after IDLE and only from a fresh observation")
    elif config.planner_input == "lavira":
        print("    1. In pane 1: k (start) -> o (PLANNER mode)")
        print("    2. In pane 3: N starts AgentNav; Space cancels and stops")
        print("    3. In pane 1: i (POSE mode)")
    else:
        print("    1. In pane 1: k (start) -> o (PLANNER mode)")
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
    config = tyro.cli(InferenceLaunchConfig)
    main(config)
