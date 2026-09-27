#!/usr/bin/env python3
"""
Sim2sim demo: random velocities with the planner default upper body held.

Starts from the current POSE initial upper-body target (17 DOF, radians)
and holds each hand at its keyboard-selected open or closed target.
Order matches policy_parameters.hpp upper_body_joint_isaaclab_order_in_mujoco_index:
  waist_yaw, waist_roll, waist_pitch,
  L/R shoulder_pitch, L/R shoulder_roll, L/R shoulder_yaw,
  L/R elbow, L/R wrist_roll, L/R wrist_pitch, L/R wrist_yaw

Usage:
  python -m gear_sonic.utils.planner_control.fixed_upper_body_demo
  python -m gear_sonic.utils.planner_control.fixed_upper_body_demo --publisher
  python -m gear_sonic.utils.planner_control.fixed_upper_body_demo --no-random-motion
  python -m gear_sonic.utils.planner_control.fixed_upper_body_demo --skip-sim --no-attach
"""

from __future__ import annotations

import os
import json
import math
import random
import select
import re
import shlex
import shutil
import socket
import subprocess
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEPLOY_ROOT = REPO_ROOT / "gear_sonic_deploy"
SESSION_NAME = "fixed_ub_planner_demo"

ZMQ_PORT = 15556
PLANNER_HZ = 50.0
DEPLOY_DEBUG_PORT = 5557

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

from gear_sonic.utils.inference.vla.poses import (
    VLA_INITIAL_UPPER_BODY_RAD as POSE_INITIAL_UPPER_BODY_RAD,
    SONIC_STAND_UPPER_BODY_RAD,
)

# Share the preset with POSE entry so the two modes cannot drift apart.
UPPER_BODY_RAD = list(POSE_INITIAL_UPPER_BODY_RAD)

import tyro  # noqa: E402
import zmq  # noqa: E402
import yaml

from gear_sonic.runtime.protocol.cpp_control import (  # noqa: E402
    build_command_message,
    build_planner_message,
)
from gear_sonic.utils.teleop.solver.hand.g1_gripper_ik_solver import (
    G1GripperInverseKinematicsSolver,
)


@dataclass
class Config:
    publisher: bool = False
    """Run ZMQ publisher only (tmux worker pane)."""
    zmq_port: int = ZMQ_PORT
    planner_hz: float = PLANNER_HZ
    deploy_debug_port: int = DEPLOY_DEBUG_PORT
    skip_sim: bool = False
    keyboard: bool = True
    """Enable terminal pause/resume and hand controls (or movement in manual mode)."""
    speed: float = 0.3
    """Walking speed in m/s."""
    turn_speed: float = 0.4
    """Turning speed in rad/s."""
    initial_yaw: float = 0.0
    """Planner facing in radians; preserves heading when restarting the publisher."""
    random_motion: bool = True
    """Automatically vary body-frame velocity with the default upper body held."""
    velocity_interval: float = 3.0
    pose_transition: float = 2.0
    velocity_transition: float = 0.6
    base_pose_profile: str = "gear_sonic/config/launch_inference.yaml"
    """Read velocity magnitudes from components.base_pose in this runtime profile."""
    seed: int = 42
    initial_state: str = ""
    """Optional saved planner fields for preserving targets across publisher restart."""
    key_timeout_sec: float = 0.7
    """Return to standing when movement key repeats stop."""
    auto_drop_sim: bool = False
    sim_drop_delay_sec: float = 15.0
    sim_startup_delay_sec: float = 5.0
    attach: bool = True


@dataclass
class KeyboardMotion:
    yaw: float = 0.0
    key: str = " "
    last_key_time: float = -math.inf

    def update(self, key: str, now: float) -> None:
        if key in "wasdqe ":
            self.key = key
            self.last_key_time = now

    def command(self, now: float, dt: float, config: Config):
        key = self.key if now - self.last_key_time < config.key_timeout_sec else " "
        self.yaw += ({"q": 1.0, "e": -1.0}.get(key, 0.0)
                     * config.turn_speed * dt)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        x, y = {"w": (1, 0), "s": (-1, 0), "a": (0, 1), "d": (0, -1)}.get(key, (0, 0))
        return (1 if key in "wasdqe" else 0,
                [c * x - s * y, s * x + c * y, 0.0],
                [c, s, 0.0], config.speed if x or y else 0.0)


class RandomMotion:
    """Seeded velocity sampling and smooth entry into the default upper-body pose."""

    def __init__(self, config: Config, pose: list[float], yaw: float):
        profile = Path(config.base_pose_profile)
        if not profile.is_absolute():
            profile = REPO_ROOT / profile
        base_pose = yaml.safe_load(profile.read_text())["components"]["base_pose"]
        forward = float(base_pose["raw_min_linear_speed_m_s"])
        lateral = float(base_pose["raw_max_lateral_speed_m_s"])
        yaw_coarse = float(base_pose["raw_yaw_coarse_speed_rad_s"])
        yaw_trim = float(base_pose["raw_yaw_trim_speed_rad_s"])
        for name in ("velocity_interval", "pose_transition", "velocity_transition"):
            if not math.isfinite(getattr(config, name)) or getattr(config, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not all(math.isfinite(v) and v > 0 for v in (forward, lateral, yaw_coarse, yaw_trim)):
            raise ValueError("BasePose velocity magnitudes must be positive and finite")
        self.velocity_choices = [(0.0, 0.0, 0.0), (forward, 0.0, 0.0), (-forward, 0.0, 0.0),
                                 (0.0, lateral, 0.0), (0.0, -lateral, 0.0),
                                 (0.0, 0.0, yaw_coarse), (0.0, 0.0, -yaw_coarse),
                                 (0.0, 0.0, yaw_trim), (0.0, 0.0, -yaw_trim)]
        if config.velocity_transition > config.velocity_interval:
            raise ValueError("Transition durations must not exceed their sampling intervals")
        self.config = config
        self.rng = random.Random(config.seed)
        self.yaw = yaw
        self.elapsed = 0.0
        self.velocity = [0.0] * 3
        self.velocity_from = list(self.velocity)
        self.velocity_to = list(self.velocity)
        self.velocity_since = 0.0
        self.next_velocity = 0.0
        self.pose = list(pose)
        self.pose_from = list(pose)
        self.pose_to = list(SONIC_STAND_UPPER_BODY_RAD)
        self.pose_since = 0.0

    @staticmethod
    def blend(start, target, elapsed, duration):
        t = min(1.0, max(0.0, elapsed / duration))
        alpha = t * t * (3.0 - 2.0 * t)
        rate = 6.0 * t * (1.0 - t) / duration
        return ([a + alpha * (b - a) for a, b in zip(start, target)],
                [rate * (b - a) for a, b in zip(start, target)])

    def step(self, dt: float):
        c = self.config
        events = []
        if self.elapsed >= self.next_velocity:
            self.velocity_from = list(self.velocity)
            choices = [v for v in self.velocity_choices if list(v) != self.velocity_to]
            self.velocity_to = list(self.rng.choice(choices))
            self.velocity_since = self.elapsed
            self.next_velocity = self.elapsed + c.velocity_interval
            events.append(f"velocity target vx/vy/wz: {self.velocity_to}")
        self.elapsed += dt
        self.velocity, _ = self.blend(self.velocity_from, self.velocity_to,
                                     self.elapsed - self.velocity_since, c.velocity_transition)
        self.pose, pose_velocity = self.blend(self.pose_from, self.pose_to,
                                            self.elapsed - self.pose_since, c.pose_transition)
        vx, vy, wz = self.velocity
        self.yaw = math.remainder(self.yaw + wz * dt, 2 * math.pi)
        cosine, sine = math.cos(self.yaw), math.sin(self.yaw)
        speed = math.hypot(vx, vy)
        movement = ([0.0] * 3 if speed < 1e-8 else
                    [(cosine * vx - sine * vy) / speed,
                     (sine * vx + cosine * vy) / speed, 0.0])
        return (1, movement, [cosine, sine, 0.0], speed), list(self.pose), pose_velocity, events


def run_publisher(config: Config) -> None:
    if config.planner_hz <= 0 or config.key_timeout_sec <= 0:
        raise ValueError("planner_hz and key_timeout_sec must be positive")
    if config.keyboard and not sys.stdin.isatty():
        raise ValueError("Keyboard control requires a terminal (or use --no-keyboard)")
    pose = list(UPPER_BODY_RAD)
    vel = [0.0] * 17
    initial_state = json.loads(Path(config.initial_state).read_text()) if config.initial_state else {}
    if initial_state:
        pose = list(initial_state["upper_body_position"])
        facing = initial_state["facing"]
        config.initial_yaw = math.atan2(facing[1], facing[0])
    print("[publisher] upper_body_position:", ", ".join(f"{v:.4f}" for v in pose))

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://127.0.0.1:{config.zmq_port}")
    time.sleep(0.5)

    period = 1.0 / config.planner_hz
    start_sent = False
    tick = 0
    motion = KeyboardMotion(yaw=config.initial_yaw)
    hand_targets = {side: [0.0] * 7 for side in ("left", "right")}
    for side in hand_targets:
        if side + "_hand_joints" in initial_state:
            hand_targets[side] = list(initial_state[side + "_hand_joints"])
    closed_targets = {
        side: G1GripperInverseKinematicsSolver(side=side)._get_middle_close_q_desired().tolist()
        for side in hand_targets
    }
    terminal_state = None
    automatic = RandomMotion(config, pose, config.initial_yaw) if config.random_motion else None
    paused = False
    last_publish = time.monotonic()

    def send_planner() -> None:
        nonlocal pose, vel, last_publish
        now = time.monotonic()
        dt = min(max(now - last_publish, 0.0), 0.1)
        last_publish = now
        if automatic is not None and start_sent and not paused:
            command, pose, vel, events = automatic.step(dt)
            for event in events:
                print(f"[random] {event}", flush=True)
            mode, movement, facing, speed = command
            motion.yaw = automatic.yaw
        elif automatic is not None:
            mode, movement, facing, speed = 0, [0.0] * 3, [math.cos(motion.yaw), math.sin(motion.yaw), 0.0], 0.0
            vel = [0.0] * 17
        else:
            mode, movement, facing, speed = motion.command(time.monotonic(), period, config)
        pub.send(
            build_planner_message(
                mode,
                movement,
                facing,
                speed=speed,
                height=-1.0,
                upper_body_position=pose,
                upper_body_velocity=vel,
                left_hand_position=hand_targets["left"],
                right_hand_position=hand_targets["right"],
            )
        )

    try:
        if config.keyboard:
            terminal_state = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
            print("J/K left close/open | U/I right close/open | C/V both close/open (held until changed)")
            if automatic is not None:
                print("AUTO RANDOM enabled | Space pause (hold current pose) | R resume | 9 release sim support | X exit", flush=True)
                print(f"[upper-body] smooth entry to planner default in {config.pose_transition}s, then HOLD", flush=True)
            else:
                print("W/S forward/back | A/D strafe | Q/E turn | Space stand | 9 release sim support | X exit")
                print("Hold movement keys to move; releasing returns to standing after the key timeout.")
        while True:
            if config.keyboard:
                while select.select([sys.stdin], [], [], 0)[0]:
                    key = os.read(sys.stdin.fileno(), 1).decode(errors="ignore").lower()
                    if key == "x":
                        return
                    if key in ("j", "k", "u", "i", "c", "v"):
                        sides = ("left", "right") if key in "cv" else (("left",) if key in "jk" else ("right",))
                        closed = key in "juc"
                        for side in sides:
                            hand_targets[side] = list(closed_targets[side]) if closed else [0.0] * 7
                        print(f"[hands] {', '.join(sides)} -> {'closed' if closed else 'open'}", flush=True)
                        continue
                    if key == "9" and start_sent:
                        subprocess.run(["tmux", "send-keys", "-t", f"{SESSION_NAME}:sim", "9"], check=True)
                    elif automatic is not None:
                        if key == " ":
                            paused = True
                            print("[random] paused", flush=True)
                        elif key == "r":
                            paused = False
                            print("[random] resumed", flush=True)
                    elif start_sent:
                        motion.update(key, time.monotonic())
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
            if not config.keyboard and (tick == 1 or tick % max(1, int(config.planner_hz)) == 0):
                print(f"[publisher] tick={tick}")
            time.sleep(period)
    except KeyboardInterrupt:
        print("[publisher] stopped")
    finally:
        paused = True
        motion.update(" ", time.monotonic())
        for _ in range(5):
            send_planner()
            time.sleep(period)
        if terminal_state is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, terminal_state)
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
    library_dirs = [
        "/opt/ros/humble/lib",
        "/home/user/Project/.tools/onnxruntime/lib",
        "/usr/local/cuda-13.0/lib64",
    ]
    library_path = ":".join(p for p in library_dirs if Path(p).is_dir())
    driver_info = Path("/proc/driver/nvidia/version")
    if driver_info.is_file():
        version = re.search(r"Kernel Module\s+(\d+\.\d+\.\d+)", driver_info.read_text())
        if version:
            driver_lib = REPO_ROOT.parent / ".tools" / f"nvidia-{version[1]}" / "usr/lib/x86_64-linux-gnu"
            if (driver_lib / "libcuda.so.1").is_file():
                library_path = f"{driver_lib}:{library_path}"
    if os.environ.get("LD_LIBRARY_PATH"):
        library_path += ":" + os.environ["LD_LIBRARY_PATH"]
    return (
        f"cd {shlex.quote(str(DEPLOY_ROOT))} && "
        f"LD_LIBRARY_PATH={shlex.quote(library_path)} "
        f"./target/release/g1_deploy_onnx_ref lo "
        f"policy/release/model_decoder.onnx reference/example/ "
        f"--obs-config policy/release/observation_config.yaml "
        f"--encoder-file policy/release/model_encoder.onnx "
        f"--planner-file planner/target_vel/V2/planner_sonic.onnx "
        f"--input-type zmq_manager --output-type zmq "
        f"--zmq-host localhost --zmq-port {port} --disable-crc-check"
    )


def run_launcher(config: Config) -> None:
    _check_prerequisites(config.skip_sim)
    subprocess.run(["tmux", "kill-session", "-t", SESSION_NAME], capture_output=True)

    module = "gear_sonic.utils.planner_control.fixed_upper_body_demo"
    pub_cmd = (
        f"cd {REPO_ROOT} && source .venv_teleop/bin/activate && "
        f"python -m {module} --publisher "
        f"--zmq-port {config.zmq_port} --planner-hz {config.planner_hz} "
        f"--deploy-debug-port {config.deploy_debug_port} "
        f"--speed {config.speed} --turn-speed {config.turn_speed} "
        f"--initial-yaw {config.initial_yaw} "
        f"{'--random-motion' if config.random_motion else '--no-random-motion'} "
        f"--velocity-interval {config.velocity_interval} "
        f"--pose-transition {config.pose_transition} --velocity-transition {config.velocity_transition} "
        f"--base-pose-profile {shlex.quote(config.base_pose_profile)} --seed {config.seed} "
        f"--initial-state {shlex.quote(config.initial_state)} "
        f"--key-timeout-sec {config.key_timeout_sec} "
        f"{'--keyboard' if config.keyboard else '--no-keyboard'}"
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
            threading.Thread(target=drop, daemon=False).start()
        time.sleep(config.sim_startup_delay_sec)

    subprocess.run(["tmux", "send-keys", "-t", f"{SESSION_NAME}:demo.1", pub_cmd, "C-m"], check=True)
    time.sleep(1.5)
    subprocess.run(["tmux", "send-keys", "-t", f"{SESSION_NAME}:demo.0", _deploy_cmd(config.zmq_port), "C-m"], check=True)

    print(f"Launched {'random velocity' if config.random_motion else 'keyboard'} demo with fixed default upper body.")
    print(f"  tmux attach -t {SESSION_NAME}")
    subprocess.run(["tmux", "select-window", "-t", f"{SESSION_NAME}:demo"], check=True)
    subprocess.run(["tmux", "select-pane", "-t", f"{SESSION_NAME}:demo.1"], check=True)

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
    main(tyro.cli(Config))
