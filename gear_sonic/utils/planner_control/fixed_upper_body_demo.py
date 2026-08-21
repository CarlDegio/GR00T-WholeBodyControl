#!/usr/bin/env python3
"""
Sim2sim demo: fixed upper-body pose via ZMQ planner.

Edit UPPER_BODY_RAD below (17 DOF, deploy ZMQ order, radians).
Order matches policy_parameters.hpp upper_body_joint_isaaclab_order_in_mujoco_index:
  waist_yaw, waist_roll, waist_pitch,
  L/R shoulder_pitch, L/R shoulder_roll, L/R shoulder_yaw,
  L/R elbow, L/R wrist_roll, L/R wrist_pitch, L/R wrist_yaw

Usage:
  python -m gear_sonic.utils.planner_control.fixed_upper_body_demo
  python -m gear_sonic.utils.planner_control.fixed_upper_body_demo --publisher
  python -m gear_sonic.utils.planner_control.fixed_upper_body_demo --skip-sim --no-attach
"""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_ROOT = REPO_ROOT / "gear_sonic_deploy"
SESSION_NAME = "fixed_ub_planner_demo"

ZMQ_PORT = 5556
PLANNER_HZ = 50.0
DEPLOY_DEBUG_PORT = 5557

# ---------------------------------------------------------------------------
# Upper-body presets (17 DOF, ZMQ planner order — IsaacLab L/R interleaved)
#
# Sources in repo:
#   SONIC_STAND  — gear_sonic_deploy/.../policy_parameters.hpp default_angles[12:29]
#   VLA_INITIAL  — gear_sonic/utils/inference/vla_initial_upper_body_recorded.json
#                  (symmetrized pose from LATENT_INITIAL_MOTION_TOKEN / VLA service)
#   ZERO         — all zeros (MuJoCo sim wbc yaml also uses 0 for upper body)
#
# Joint order: waist_yaw, waist_roll, waist_pitch,
#   L/R shoulder_pitch, L/R shoulder_roll, L/R shoulder_yaw,
#   L/R elbow, L/R wrist_roll, L/R wrist_pitch, L/R wrist_yaw
# ---------------------------------------------------------------------------

SONIC_STAND_UPPER_BODY_RAD: list[float] = [
    0.0, 0.0, 0.0,
    0.0, 0.0,  # shoulder_pitch
    0.0, 0.0, # shoulder_roll
    0.0, 0.0, # shoulder_yaw
    0.0, 0.0, 
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
]

VLA_INITIAL_UPPER_BODY_RAD: list[float] = [
    0.0, 0.0, 0.14,
    0.0, 0.0,
    0.22, -0.22,
    -0.43, 0.43,
    1.0, 1.0,
    0.0, 0.0,
    0.06, 0.06,
    0.0, 0.0,
]

ZERO_UPPER_BODY_RAD: list[float] = [0.0] * 17

# Active pose — change this line to switch preset:
#UPPER_BODY_RAD: list[float] = VLA_INITIAL_UPPER_BODY_RAD
UPPER_BODY_RAD = SONIC_STAND_UPPER_BODY_RAD
# UPPER_BODY_RAD = ZERO_UPPER_BODY_RAD

def _bootstrap_venv() -> None:
    preferred = REPO_ROOT / ".venv_teleop" / "bin" / "python"
    if preferred.is_file() and Path(sys.executable).resolve() != preferred.resolve():
        print(f"Re-launching with {preferred} ...")
        os.execv(str(preferred), [str(preferred)] + sys.argv)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        import tyro  # noqa: F401
    except ImportError:
        print("ERROR: tyro missing. Run: bash install_scripts/install_pico.sh")
        sys.exit(1)


_bootstrap_venv()

import tyro  # noqa: E402
import zmq  # noqa: E402

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (  # noqa: E402
    build_command_message,
    build_planner_message,
)


@dataclass
class Config:
    publisher: bool = False
    """Run ZMQ publisher only (tmux worker pane)."""
    zmq_port: int = ZMQ_PORT
    planner_hz: float = PLANNER_HZ
    deploy_debug_port: int = DEPLOY_DEBUG_PORT
    skip_sim: bool = False
    auto_drop_sim: bool = True
    sim_drop_delay_sec: float = 15.0
    sim_startup_delay_sec: float = 5.0
    attach: bool = True


def run_publisher(config: Config) -> None:
    pose = list(UPPER_BODY_RAD)
    vel = [0.0] * 17
    print("[publisher] upper_body_position:", ", ".join(f"{v:.4f}" for v in pose))

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://*:{config.zmq_port}")
    time.sleep(0.5)

    period = 1.0 / config.planner_hz
    start_sent = False
    tick = 0

    def send_planner() -> None:
        pub.send(
            build_planner_message(
                0,
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                speed=-1.0,
                height=-1.0,
                upper_body_position=pose,
                upper_body_velocity=vel,
            )
        )

    try:
        while True:
            send_planner()
            if not start_sent:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(0.01)
                    if sock.connect_ex(("127.0.0.1", config.deploy_debug_port)) == 0:
                        for _ in range(10):
                            send_planner()
                            time.sleep(0.02)
                        pub.send(build_command_message(start=True, stop=False, planner=True))
                        start_sent = True
                        print("[publisher] deploy ready, sent start")
            tick += 1
            if tick == 1 or tick % int(config.planner_hz) == 0:
                print(f"[publisher] tick={tick}")
            time.sleep(period)
    except KeyboardInterrupt:
        print("[publisher] stopped")
    finally:
        pub.close()
        ctx.term()


def _check_prerequisites(skip_sim: bool) -> None:
    errors: list[str] = []
    if not shutil.which("tmux"):
        errors.append("tmux not installed")
    if not (REPO_ROOT / ".venv_teleop" / "bin" / "activate").exists():
        errors.append(".venv_teleop missing — bash install_scripts/install_pico.sh")
    if not skip_sim and not (REPO_ROOT / ".venv_sim" / "bin" / "activate").exists():
        errors.append(".venv_sim missing — bash install_scripts/install_mujoco_sim.sh")
    deploy_bin = DEPLOY_ROOT / "target" / "release" / "g1_deploy_onnx_ref"
    if not deploy_bin.is_file():
        errors.append(f"deploy binary missing: {deploy_bin}")
    if errors:
        sys.exit("Prerequisites failed:\n  - " + "\n  - ".join(errors))


def _deploy_cmd(port: int) -> str:
    return (
        f"cd {DEPLOY_ROOT} && source scripts/setup_env.sh && "
        f"just run g1_deploy_onnx_ref lo "
        f"policy/release/model_decoder.onnx reference/example/ "
        f"--obs-config policy/release/observation_config.yaml "
        f"--encoder-file policy/release/model_encoder.onnx "
        f"--planner-file planner/target_vel/V2/planner_sonic.onnx "
        f"--input-type zmq_manager --output-type all "
        f"--zmq-host localhost --zmq-port {port} --disable-crc-check"
    )


def run_launcher(config: Config) -> None:
    _check_prerequisites(config.skip_sim)
    subprocess.run(["tmux", "kill-session", "-t", SESSION_NAME], capture_output=True)

    module = "gear_sonic.utils.planner_control.fixed_upper_body_demo"
    pub_cmd = (
        f"cd {REPO_ROOT} && source .venv_teleop/bin/activate && "
        f"python -m {module} --publisher "
        f"--zmq-port {config.zmq_port} --planner-hz {config.planner_hz}"
    )

    subprocess.run(["tmux", "new-session", "-d", "-s", SESSION_NAME], check=True)
    subprocess.run(["tmux", "rename-window", "-t", f"{SESSION_NAME}:0", "demo"], check=True)
    subprocess.run(["tmux", "split-window", "-t", f"{SESSION_NAME}:demo", "-h"], check=True)
    time.sleep(0.5)

    if not config.skip_sim:
        subprocess.run(["tmux", "new-window", "-t", SESSION_NAME, "-n", "sim"], check=True)
        sim_cmd = (
            f"cd {REPO_ROOT} && source .venv_sim/bin/activate && "
            f"python -m gear_sonic.utils.mujoco_sim.service"
        )
        subprocess.run(["tmux", "send-keys", "-t", f"{SESSION_NAME}:sim", sim_cmd, "C-m"], check=True)
        if config.auto_drop_sim:
            def drop() -> None:
                time.sleep(config.sim_drop_delay_sec)
                subprocess.run(
                    ["tmux", "send-keys", "-t", f"{SESSION_NAME}:sim", "9"],
                    check=True,
                )
            threading.Thread(target=drop, daemon=True).start()
        time.sleep(config.sim_startup_delay_sec)

    subprocess.run(["tmux", "send-keys", "-t", f"{SESSION_NAME}:demo.1", pub_cmd, "C-m"], check=True)
    time.sleep(1.5)
    subprocess.run(["tmux", "send-keys", "-t", f"{SESSION_NAME}:demo.0", _deploy_cmd(config.zmq_port), "C-m"], check=True)

    print(f"Launched. Edit UPPER_BODY_RAD in {script}")
    print(f"  tmux attach -t {SESSION_NAME}")

    if config.attach:
        try:
            subprocess.run(["tmux", "attach", "-t", SESSION_NAME])
        except KeyboardInterrupt:
            pass


def main(config: Config) -> None:
    if config.publisher:
        run_publisher(config)
    else:
        run_launcher(config)


if __name__ == "__main__":
    def _sigint(_s, _f) -> None:
        subprocess.run(["tmux", "kill-session", "-t", SESSION_NAME], capture_output=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint)
    main(tyro.cli(Config))
