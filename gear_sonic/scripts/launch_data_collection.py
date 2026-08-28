"""
All-in-one tmux launcher for SONIC data collection.

Starts the full data collection stack in a single tmux session:

    Window 0 — data_collection (4 panes):
    ┌───────────────────────┬───────────────────────┐
    │ Pane 0: C++ Deploy    │ Pane 1: Data Exporter │
    │ (gear_sonic_deploy)   │ (.venv_data_collection)│
    ├───────────────────────┼───────────────────────┤
    │ Pane 2: PICO Teleop   │ Pane 3: Camera Viewer │
    │ (.venv_teleop)        │ (.venv_data_collection)│
    └───────────────────────┴───────────────────────┘

    Window 1 — sim  (only when --sim is passed):
    ┌─────────────────────────────────────────────────┐
    │ MuJoCo Simulator service                        │
    │ (.venv_sim)                                     │
    └─────────────────────────────────────────────────┘

    Window — gateways (3 panes):
    ┌────────────────────────┬────────────────────────┐
    │ SensorGateway          │ ControlGateway         │
    │ camera/state/config    │ typed recording input  │
    ├────────────────────────┴────────────────────────┤
    │ PICO Video Bridge (native USBOnly, supervised) │
    └─────────────────────────────────────────────────┘

Prerequisites:
    - tmux installed (sudo apt install tmux)
    - Virtual environments set up:
        bash install_scripts/install_pico.sh          -> .venv_teleop
        bash install_scripts/install_data_collection.sh -> .venv_data_collection
    - gear_sonic_deploy built (see docs)
    - For sim: .venv_sim must exist (see install instructions)

Usage (from repo root — no venv activation needed):
    python gear_sonic/scripts/launch_data_collection.py              # real robot (default)
    python gear_sonic/scripts/launch_data_collection.py --sim        # MuJoCo sim
    python gear_sonic/scripts/launch_data_collection.py --no-camera-viewer  # skip viewer
    python gear_sonic/scripts/launch_data_collection.py --no-pico-video  # skip PICO video
"""

import argparse
from dataclasses import dataclass, fields
import importlib
from pathlib import Path
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Any, get_type_hints

if __package__:
    from gear_sonic.scripts.launcher import TmuxSession, bootstrap_venv
else:
    from launcher import TmuxSession, bootstrap_venv


def _launcher_dependencies_available(import_module=importlib.import_module) -> bool:
    try:
        import_module("tyro")
        import_module("yaml")
    except ImportError:
        return False
    return True


def _bootstrap_venv():
    """Re-exec with the data-collection Python if launcher deps are unavailable."""
    bootstrap_venv(
        _launcher_dependencies_available(),
        repo_root=Path(__file__).resolve().parent.parent.parent,
        venv_name=".venv_data_collection",
        missing_message=(
            "ERROR: tyro/PyYAML unavailable and .venv_data_collection not found.\n"
            "  Run: bash install_scripts/install_data_collection.sh"
        ),
    )


_bootstrap_venv()

import tyro
import yaml


def default_data_collection_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "launch_data_collection.yaml"


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
class DataCollectionLaunchConfig:
    """CLI config for the all-in-one data collection tmux launcher."""

    # Deployment mode
    sim: bool = False
    """Run against MuJoCo sim (deploy.sh sim) instead of real robot."""

    # C++ deploy options
    deploy_input_type: str = "zmq_manager"
    """Input type for the C++ deploy (zmq_manager, keyboard, etc.)."""

    deploy_zmq_host: str = "localhost"
    """ZMQ host for the C++ deploy to listen on."""

    deploy_checkpoint: str = ""
    """Checkpoint path for deploy.sh (e.g., 'policy/checkpoints/my_model/model_step_100000').
    Leave empty to use the deploy.sh default."""

    deploy_obs_config: str = ""
    """Observation config file for deploy.sh. Leave empty for default."""

    deploy_planner: str = ""
    """Planner model path for deploy.sh. Leave empty for default."""

    deploy_motion_data: str = ""
    """Motion data path for deploy.sh. Leave empty for default."""

    deploy_output_type: str = ""
    """Output type for deploy.sh. Leave empty for default."""

    # PICO teleop options
    pico_manager: bool = True
    """Run pico_manager_thread_server with --manager flag."""

    pico_vis_vr3pt: bool = False
    """Enable VR 3-point visualization on the teleop streamer."""

    pico_vis_smpl: bool = False
    """Enable SMPL visualization on the teleop streamer."""

    pico_waist_tracking: bool = False
    """Enable waist tracking on the teleop streamer."""

    pico_video: bool = True
    """Keep the SensorGateway-to-PICO USBOnly video bridge running."""

    # Data exporter options
    task_prompt: str = "demo"
    """Language task prompt for the data exporter."""

    dataset_name: str = ""
    """Dataset name for the data exporter. Leave empty to auto-generate from timestamp."""

    runtime_profile: str = str(
        Path(__file__).resolve().parent.parent / "config" / "launch_inference.yaml"
    )
    """Runtime profile used by SensorGateway, ControlGateway, and DataExporter."""

    record_wrist_cameras: bool = False
    """Record wrist camera streams (left_wrist, right_wrist) in the dataset."""

    record_chest_camera: bool = False
    """Record chest camera stream (chest_view) in the dataset."""

    text_to_speech: bool = True
    """Enable voice feedback via espeak (data exporter)."""

    # Camera viewer
    camera_viewer: bool = True
    """Start the camera viewer pane."""

    camera_host: str = "localhost"
    """Camera server host used by SensorGateway."""

    camera_port: int = 5555
    """Camera server port used by SensorGateway and the simulator publisher."""

    config: str = str(default_data_collection_config_path())
    """YAML file containing the data collection launcher configuration."""


def _validated_data_collection_values(raw_values: dict[str, Any]) -> dict[str, Any]:
    config_fields = {
        item.name: item
        for item in fields(DataCollectionLaunchConfig)
        if item.name != "config"
    }
    unknown = set(raw_values) - set(config_fields)
    missing = set(config_fields) - set(raw_values)
    if unknown:
        raise ValueError(
            "unknown launch_data_collection YAML fields: "
            + ", ".join(sorted(unknown))
        )
    if missing:
        raise ValueError(
            "missing launch_data_collection YAML fields: "
            + ", ".join(sorted(missing))
        )

    annotations = get_type_hints(DataCollectionLaunchConfig)
    values: dict[str, Any] = {}
    for name, value in raw_values.items():
        annotation = annotations[name]
        if annotation is bool:
            if not isinstance(value, bool):
                raise ValueError(f"launch_data_collection.{name} must be a boolean")
            values[name] = value
        elif annotation is int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"launch_data_collection.{name} must be an integer")
            values[name] = value
        elif annotation is float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"launch_data_collection.{name} must be a number")
            values[name] = float(value)
        elif annotation is str:
            if not isinstance(value, str):
                raise ValueError(f"launch_data_collection.{name} must be a string")
            values[name] = value
        else:
            raise TypeError(
                f"unsupported data collection launch type for {name}: {annotation}"
            )
    return values


def load_data_collection_launch_config(
    path: str | Path | None = None,
) -> DataCollectionLaunchConfig:
    config_path = (
        default_data_collection_config_path()
        if path is None
        else Path(path).expanduser()
    )
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
    except OSError as exc:
        raise ValueError(f"cannot read launch YAML {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid launch YAML {config_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"launch YAML must contain an object: {config_path}")
    root_fields = {"schema", "version", "profile", "launch_data_collection"}
    unknown_root_fields = set(payload) - root_fields
    if unknown_root_fields:
        raise ValueError(
            "unknown data collection YAML root fields: "
            + ", ".join(sorted(unknown_root_fields))
        )
    if (
        payload.get("schema") != "sonic.data_collection_launch"
        or type(payload.get("version")) is not int
        or payload.get("version") != 1
    ):
        raise ValueError(
            "launch YAML must use sonic.data_collection_launch version 1"
        )
    profile = payload.get("profile")
    if not isinstance(profile, str) or not profile.strip():
        raise ValueError("launch YAML profile must be a non-empty string")
    raw_values = payload.get("launch_data_collection")
    if not isinstance(raw_values, dict):
        raise ValueError("launch YAML must contain a launch_data_collection object")
    values = _validated_data_collection_values(raw_values)
    return DataCollectionLaunchConfig(config=str(config_path), **values)


def parse_data_collection_launch_config(
    args: list[str] | None = None,
) -> DataCollectionLaunchConfig:
    argv = list(sys.argv[1:] if args is None else args)
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument(
        "--config",
        default=str(default_data_collection_config_path()),
    )
    bootstrap_args, _ = bootstrap.parse_known_args(argv)
    yaml_defaults = load_data_collection_launch_config(bootstrap_args.config)
    return tyro.cli(DataCollectionLaunchConfig, args=argv, default=yaml_defaults)


SESSION_NAME = "sonic_data_collection"


def _check_prerequisites(sim: bool = False, pico_video: bool = True):
    """Verify that required tools and venvs exist."""
    errors = []

    if not shutil.which("tmux"):
        errors.append("tmux is not installed. Install with: sudo apt install tmux")

    if pico_video and not shutil.which("ffmpeg"):
        errors.append("ffmpeg is required for PICO video encoding")

    repo_root = Path(__file__).resolve().parent.parent.parent

    if not (repo_root / ".venv_teleop" / "bin" / "activate").exists():
        errors.append(
            ".venv_teleop not found. Run: bash install_scripts/install_pico.sh"
        )

    if not (repo_root / ".venv_data_collection" / "bin" / "activate").exists():
        errors.append(
            ".venv_data_collection not found. Run: "
            "bash install_scripts/install_data_collection.sh"
        )

    deploy_dir = repo_root / "gear_sonic_deploy"
    if not (deploy_dir / "deploy.sh").exists():
        errors.append(
            f"gear_sonic_deploy/deploy.sh not found at {deploy_dir}. "
            "Ensure the deploy directory is set up."
        )

    if sim and not (repo_root / ".venv_sim" / "bin" / "activate").exists():
        errors.append(
            ".venv_sim not found. Set up the simulation venv first "
            "(see install instructions)."
        )

    if errors:
        print("ERROR: Prerequisites not met:\n")
        for e in errors:
            print(f"  - {e}")
        print()
        sys.exit(1)


def _kill_existing_session():
    """Kill any existing tmux session with our name."""
    _tmux_session().kill()


def _tmux_session() -> TmuxSession:
    return TmuxSession(
        SESSION_NAME,
        runner=subprocess.run,
        sleeper=time.sleep,
    )


def _create_tmux_session():
    """Create a 4-pane tmux layout."""
    tmux = _tmux_session()
    # Create detached session
    tmux.command("new-session", "-d", "-s", SESSION_NAME)

    # Enable mouse support (click panes, scroll, resize)
    tmux.command(
        "set-option", "-t", SESSION_NAME, "-g", "mouse", "on", check=False,
    )

    # Bind Ctrl+\ to kill the entire session (no prefix needed)
    tmux.command(
        "bind-key", "-T", "root", "C-\\", "kill-session", check=False,
    )

    # Rename default window
    tmux.command(
        "rename-window", "-t", f"{SESSION_NAME}:0", "data_collection", check=False,
    )

    # Split into 4 panes:
    #   0 | 1
    #   -----
    #   2 | 3

    # Split horizontally: pane 0 (left) and pane 1 (right)
    tmux.command(
        "split-window", "-t", f"{SESSION_NAME}:0", "-h", check=False,
    )

    # Split left pane vertically: pane 0 (top-left) and pane 2 (bottom-left)
    tmux.command(
        "split-window", "-t", f"{SESSION_NAME}:0.0", "-v", check=False,
    )

    # Split right pane vertically: pane 1 becomes top-right, new pane 3 bottom-right
    tmux.command(
        "split-window", "-t", f"{SESSION_NAME}:0.2", "-v", check=False,
    )

    # Let all pane shells finish initialization (.bashrc, conda, etc.)
    time.sleep(5)


def _send_to_pane(pane_index: int, cmd: str, wait: float = 1.0):
    """Send a command string to a tmux pane."""
    target = f"{SESSION_NAME}:0.{pane_index}"
    _tmux_session().send(target, cmd, wait=wait, check=False)


def _check_pane_alive(pane_index: int) -> bool:
    """Check if a tmux pane's process is still running."""
    target = f"{SESSION_NAME}:0.{pane_index}"
    return _tmux_session().pane_alive(target)


def build_camera_viewer_command(
    config: DataCollectionLaunchConfig,
    repo_root: Path,
) -> str:
    """Build the Gateway-only camera viewer command for its tmux pane."""
    return (
        f"cd {shlex.quote(str(repo_root))} && "
        "source .venv_data_collection/bin/activate && "
        "python -m gear_sonic.utils.operator.camera_viewer "
        f"--profile {shlex.quote(config.runtime_profile)}"
    )


def build_sensor_gateway_command(
    config: DataCollectionLaunchConfig,
    repo_root: Path,
) -> str:
    """Build SensorGateway using endpoints from the unified runtime profile."""
    return (
        f"cd {shlex.quote(str(repo_root))} && "
        "source .venv_teleop/bin/activate && "
        "python -m gear_sonic.runtime.gateway.services.sensor "
        f"--profile {shlex.quote(config.runtime_profile)} "
        "--no-enable-depth-anything --no-enable-ros "
        "--no-enable-visualization --no-enable-vla-timing"
    )


def build_pico_video_command(
    config: DataCollectionLaunchConfig,
    repo_root: Path,
) -> str:
    """Build the supervised native USBOnly PICO video command."""
    return (
        f"cd {shlex.quote(str(repo_root))} && "
        "source .venv_teleop/bin/activate && "
        "python -m gear_sonic.utils.pico_video.service "
        f"--profile {shlex.quote(config.runtime_profile)} "
        "--encoder h264_nvenc --stay-alive"
    )


def main(config: DataCollectionLaunchConfig):
    repo_root = Path(__file__).resolve().parent.parent.parent

    _check_prerequisites(sim=config.sim, pico_video=config.pico_video)
    _kill_existing_session()

    print("=" * 60)
    print("  SONIC Data Collection Launcher")
    print("=" * 60)
    print(f"  Mode:            {'Simulation' if config.sim else 'Real Robot'}")
    print(f"  Task prompt:     {config.task_prompt}")
    print(f"  Dataset name:    {config.dataset_name or '(auto)'}")
    print(f"  Deploy input:    {config.deploy_input_type}")
    if config.deploy_checkpoint:
        print(f"  Checkpoint:      {config.deploy_checkpoint}")
    print(f"  Camera:          {config.camera_host}:{config.camera_port}")
    print(f"  Camera viewer:   {'Yes' if config.camera_viewer else 'No'}")
    print(f"  PICO video:      {'USBOnly (supervised)' if config.pico_video else 'No'}")
    print(f"  Wrist cameras:   {'Yes' if config.record_wrist_cameras else 'No'}")
    print(f"  Chest camera:    {'Yes' if config.record_chest_camera else 'No'}")
    print(f"  Text-to-speech:  {'Yes' if config.text_to_speech else 'No'}")
    print(f"  PICO vis:        vr3pt={config.pico_vis_vr3pt} smpl={config.pico_vis_smpl}")
    print(f"  PC IP (for PICO): {_get_local_ip()}")
    print("=" * 60)

    _create_tmux_session()
    print(f"Created tmux session: {SESSION_NAME}")

    # Gateway window: DataExporter has no direct camera/state/control sockets.
    subprocess.run(
        ["tmux", "new-window", "-t", SESSION_NAME, "-n", "gateways"],
        check=True,
    )
    subprocess.run(
        ["tmux", "split-window", "-h", "-t", f"{SESSION_NAME}:gateways"],
        check=True,
    )
    if config.pico_video:
        subprocess.run(
            [
                "tmux",
                "split-window",
                "-v",
                "-t",
                f"{SESSION_NAME}:gateways.1",
            ],
            check=True,
        )
        subprocess.run(
            ["tmux", "select-layout", "-t", f"{SESSION_NAME}:gateways", "tiled"],
            check=True,
        )
    profile_arg = shlex.quote(config.runtime_profile)
    sensor_gateway_cmd = build_sensor_gateway_command(config, repo_root)
    control_gateway_cmd = (
        f"cd {shlex.quote(str(repo_root))} && "
        "source .venv_teleop/bin/activate && "
        "python -m gear_sonic.runtime.gateway.services.control "
        f"--profile {profile_arg}"
    )
    pico_video_cmd = (
        build_pico_video_command(config, repo_root) if config.pico_video else ""
    )
    subprocess.run(
        [
            "tmux",
            "send-keys",
            "-t",
            f"{SESSION_NAME}:gateways.0",
            sensor_gateway_cmd,
            "C-m",
        ],
        check=True,
    )
    subprocess.run(
        [
            "tmux",
            "send-keys",
            "-t",
            f"{SESSION_NAME}:gateways.1",
            control_gateway_cmd,
            "C-m",
        ],
        check=True,
    )
    if config.pico_video:
        subprocess.run(
            [
                "tmux",
                "send-keys",
                "-t",
                f"{SESSION_NAME}:gateways.2",
                pico_video_cmd,
                "C-m",
            ],
            check=True,
        )
    gateway_components = "SensorGateway, ControlGateway"
    if config.pico_video:
        gateway_components += ", and supervised PICO video"
    print(f"Starting {gateway_components} (window: gateways)...")
    time.sleep(2.0)
    subprocess.run(
        ["tmux", "select-window", "-t", f"{SESSION_NAME}:data_collection"],
        check=True,
    )

    # --- Window 1 (sim only): MuJoCo Simulator ---
    if config.sim:
        subprocess.run(
            ["tmux", "new-window", "-t", SESSION_NAME, "-n", "sim"],
        )
        sim_cmd = (
            f"cd {repo_root} && "
            f"source .venv_sim/bin/activate && "
            f"python -m gear_sonic.utils.mujoco_sim.service "
            f"--enable-image-publish --enable-offscreen "
            f"--camera-port {config.camera_port}"
        )
        sim_target = f"{SESSION_NAME}:sim"
        subprocess.run(
            ["tmux", "send-keys", "-t", sim_target, sim_cmd, "C-m"],
        )
        print("Starting MuJoCo simulator (window: sim)...")
        time.sleep(3.0)

        # Switch back to the data_collection window for the remaining panes
        subprocess.run(
            ["tmux", "select-window", "-t", f"{SESSION_NAME}:data_collection"],
        )

    # --- Pane 0 (top-left): C++ Deploy ---
    deploy_mode = "sim" if config.sim else "real"
    deploy_cmd = (
        f"cd {repo_root / 'gear_sonic_deploy'} && "
        f"./deploy.sh "
        f"--yes "
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
    _send_to_pane(0, deploy_cmd, wait=3.0)

    if not _check_pane_alive(0):
        print("WARNING: C++ deploy pane may have failed to start.")

    # --- Pane 2 (bottom-left): PICO Teleop Streamer ---
    pico_cmd = (
        f"cd {repo_root} && "
        f"source .venv_teleop/bin/activate && "
        f"python -m gear_sonic.utils.teleop.pico_manager"
    )
    if config.pico_manager:
        pico_cmd += " --manager"
    if config.pico_vis_vr3pt:
        pico_cmd += " --vis_vr3pt"
    if config.pico_vis_smpl:
        pico_cmd += " --vis_smpl"
    if config.pico_waist_tracking:
        pico_cmd += " --waist_tracking"

    print("Starting PICO teleop streamer (pane 2)...")
    _send_to_pane(1, pico_cmd, wait=2.0)

    # --- Pane 3 (bottom-right): Camera Viewer ---
    if config.camera_viewer:
        viewer_cmd = build_camera_viewer_command(config, repo_root)
        print("Starting camera viewer (pane 3)...")
        _send_to_pane(3, viewer_cmd, wait=2.0)

    # --- Pane 1 (top-right): Data Exporter ---
    exporter_cmd = (
        f"cd {shlex.quote(str(repo_root))} && "
        "source .venv_data_collection/bin/activate && "
        "python -m gear_sonic.utils.data_collection.service "
        f"--profile {profile_arg} "
        f"--task-prompt {shlex.quote(config.task_prompt)}"
    )
    if config.dataset_name:
        exporter_cmd += f" --dataset-name {shlex.quote(config.dataset_name)}"
    if config.record_wrist_cameras:
        exporter_cmd += " --record-wrist-cameras"
    if config.record_chest_camera:
        exporter_cmd += " --record-chest-camera"
    if not config.text_to_speech:
        exporter_cmd += " --no-text-to-speech"

    print("Starting data exporter (pane 1)...")
    _send_to_pane(2, exporter_cmd, wait=1.0)

    # Select the data exporter pane so the user lands there for interactive input
    subprocess.run(
        ["tmux", "select-pane", "-t", f"{SESSION_NAME}:0.2"],
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
    print("  Window 'data_collection':")
    print("    Pane 0 (top-left):     C++ Deploy")
    print("    Pane 1 (bottom-left):  PICO Teleop")
    print("    Pane 2 (top-right):    Data Exporter  <-- you are here")
    if config.camera_viewer:
        print("    Pane 3 (bottom-right): Camera Viewer")
    gateway_summary = "SensorGateway | ControlGateway"
    if config.pico_video:
        gateway_summary += " | PICO Video (USBOnly, supervised)"
    print(f"  Window 'gateways': {gateway_summary}")
    print()
    print("  Controls:")
    print("    Ctrl+b, arrow keys  - Switch between panes")
    if config.sim:
        print("    Ctrl+b, n / p       - Next / previous window")
    print("    Ctrl+b, d           - Detach from session")
    print("    Ctrl+\\              - Kill entire session")
    print("=" * 60)

    # Attach to the session
    # After detach/exit, offer cleanup
    if _tmux_session().attach():
        print(f"\nSession '{SESSION_NAME}' is still running.")
        print(f"  Reattach:  tmux attach -t {SESSION_NAME}")
        print(f"  Kill:      tmux kill-session -t {SESSION_NAME}")


def _signal_handler(sig, frame):
    print("\nShutdown requested...")
    _tmux_session().kill()
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _signal_handler)
    try:
        config = parse_data_collection_launch_config()
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    main(config)
