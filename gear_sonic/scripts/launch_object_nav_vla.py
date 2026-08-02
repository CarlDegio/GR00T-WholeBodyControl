"""Launch the N-triggered ObjectNav -> VLA workflow in tmux."""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
from typing import Any


def _bootstrap_venv() -> None:
    try:
        import tyro  # noqa: F401
    except ImportError:
        repo_root = Path(__file__).resolve().parents[2]
        python = repo_root / ".venv_inference" / "bin" / "python"
        if not python.is_file():
            raise RuntimeError(".venv_inference is required; run install_scripts/install_inference.sh")
        os.execv(str(python), [str(python), *sys.argv])


_bootstrap_venv()

import tyro
import zmq

from gear_sonic.utils.inference.object_nav import (
    ObjectNavConfig,
    ObjectNavRunner,
    SonicPlannerRequestError,
    send_object_nav_commands,
)


@dataclass
class ObjectNavVlaLaunchConfig:
    mission: str
    global_target: str

    navigation_mode: str = "repeat-until-stop"
    """repeat-until-stop or once."""

    vla_prompt: str = ""
    max_navigation_iterations: int = 0
    replan_delay_seconds: float = 0.1
    startup_timeout_seconds: float = 300.0

    camera_host: str = "localhost"
    camera_port: int = 5555
    camera_timeout_ms: int = 3000
    output_root: str = "outputs/object_nav"
    codex_timeout_seconds: float = 180.0
    min_confidence: float = 0.6
    safe_distance: float = 0.0
    max_direct_travel: float = 8.0

    planner_json_host: str = "127.0.0.1"
    planner_json_port: int = 5559
    planner_control_host: str = "127.0.0.1"
    planner_control_port: int = 5560
    planner_timeout_ms: int = 70000
    action_host: str = "*"
    action_port: int = 5556
    keyboard_host: str = "localhost"
    keyboard_port: int = 5580

    policy_host: str = "localhost"
    policy_port: int = 5550
    embodiment_tag: str = "unitree_g1_sonic"
    action_publish_rate: int = 50
    action_horizon: int = 70

    deploy_input_type: str = "zmq_manager"
    deploy_zmq_host: str = "localhost"
    deploy_checkpoint: str = ""
    deploy_obs_config: str = ""
    deploy_planner: str = ""
    deploy_motion_data: str = ""
    deploy_output_type: str = ""

    session_name: str = "sonic_object_nav_vla"
    replace_session: bool = False
    attach: bool = True

    worker_config_b64: str = ""
    """Internal serialized worker configuration; do not set manually."""


def _validate_config(config: ObjectNavVlaLaunchConfig) -> None:
    if config.navigation_mode not in {"repeat-until-stop", "once"}:
        raise ValueError("navigation_mode must be repeat-until-stop or once")
    if not config.mission.strip() or not config.global_target.strip():
        raise ValueError("mission and global_target are required")
    if config.max_navigation_iterations < 0:
        raise ValueError("max_navigation_iterations must be non-negative")
    if config.replan_delay_seconds < 0 or config.startup_timeout_seconds <= 0:
        raise ValueError("replan delay must be non-negative and startup timeout positive")


def planner_control_request(
    host: str,
    port: int,
    operation: str,
    *,
    timeout_ms: int = 1000,
    context: Any | None = None,
) -> dict[str, Any]:
    zmq_context = context if context is not None else zmq.Context.instance()
    socket = zmq_context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    try:
        socket.connect(f"tcp://{host}:{int(port)}")
        socket.send_json({"op": operation})
        if not socket.poll(timeout_ms):
            raise TimeoutError(f"planner control {operation} timed out")
        reply = socket.recv_json()
        if not isinstance(reply, dict):
            raise RuntimeError("planner control returned a malformed reply")
        return reply
    finally:
        socket.close()


def _wait_for_n(config: ObjectNavVlaLaunchConfig) -> None:
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(f"tcp://{config.keyboard_host}:{config.keyboard_port}")
    print("[Workflow] Waiting for N to start ObjectNav", flush=True)
    try:
        while True:
            if not socket.poll(250):
                continue
            key = socket.recv_string().strip().lower()
            if key == "n":
                print("[Workflow] N received; arming ObjectNav", flush=True)
                return
    finally:
        socket.close()
        context.term()


def _wait_for_action_subscribers(config: ObjectNavVlaLaunchConfig) -> None:
    deadline = time.monotonic() + config.startup_timeout_seconds
    last_message = "planner control endpoint is not ready"
    while time.monotonic() < deadline:
        try:
            reply = planner_control_request(
                config.planner_control_host,
                config.planner_control_port,
                "status",
                timeout_ms=500,
            )
        except Exception as exc:
            last_message = str(exc)
        else:
            if reply.get("action_ready"):
                return
            last_message = "C++ command/planner subscribers are not ready"
        time.sleep(0.25)
    raise TimeoutError(f"startup timed out: {last_message}")


def _start_planner(config: ObjectNavVlaLaunchConfig) -> None:
    reply = planner_control_request(
        config.planner_control_host,
        config.planner_control_port,
        "start",
        timeout_ms=2000,
    )
    if reply.get("status") != "ready" or not reply.get("planner_ready"):
        raise RuntimeError(f"failed to enter PLANNER mode: {reply}")


def _planner_command(config: ObjectNavVlaLaunchConfig, repo_root: Path) -> list[str]:
    python = repo_root / ".venv_inference" / "bin" / "python"
    return [
        str(python),
        "gear_sonic/scripts/controlled_uni_lavira_planner_thread_server.py",
        "--json-host",
        config.planner_json_host,
        "--json-port",
        str(config.planner_json_port),
        "--control-host",
        config.planner_control_host,
        "--control-port",
        str(config.planner_control_port),
        "--action-host",
        config.action_host,
        "--action-port",
        str(config.action_port),
        "--keyboard-host",
        config.keyboard_host,
        "--keyboard-port",
        str(config.keyboard_port),
    ]


def _vla_command(config: ObjectNavVlaLaunchConfig, repo_root: Path) -> str:
    python = repo_root / ".venv_inference" / "bin" / "python"
    prompt = config.vla_prompt.strip() or config.mission
    argv = [
        str(python),
        "gear_sonic/scripts/run_vla_inference.py",
        "--host",
        config.policy_host,
        "--port",
        str(config.policy_port),
        "--embodiment-tag",
        config.embodiment_tag,
        "--prompt",
        prompt,
        "--action-publish-rate",
        str(config.action_publish_rate),
        "--action-horizon",
        str(config.action_horizon),
        "--camera-host",
        config.camera_host,
        "--camera-port",
        str(config.camera_port),
        "--action-zmq-port",
        str(config.action_port),
        "--keyboard-zmq-host",
        config.keyboard_host,
        "--keyboard-zmq-port",
        str(config.keyboard_port),
        "--assume-cpp-planner-running",
    ]
    return f"cd {shlex.quote(str(repo_root))} && {shlex.join(argv)}"


def _start_vla_pane(config: ObjectNavVlaLaunchConfig, repo_root: Path) -> None:
    target = f"={config.session_name}:workflow.3"
    subprocess.run(
        ["tmux", "send-keys", "-t", target, _vla_command(config, repo_root), "C-m"],
        check=True,
    )
    print("[Workflow] VLA process started; C++ remains in PLANNER", flush=True)
    print("[Workflow] Use the keyboard pane: i -> p", flush=True)


def _safe_release_planner(
    config: ObjectNavVlaLaunchConfig, planner_process: subprocess.Popen[Any]
) -> bool:
    """Release the ObjectNav publisher while preserving running PLANNER mode."""
    try:
        reply = planner_control_request(
            config.planner_control_host,
            config.planner_control_port,
            "handoff",
            timeout_ms=2000,
        )
        if reply.get("status") != "handoff" or not reply.get("planner_ready"):
            raise RuntimeError(f"unexpected safe-release reply: {reply}")
    except Exception as exc:
        print(f"[Workflow] planner safe release failed: {exc}", file=sys.stderr)
        if planner_process.poll() is None:
            planner_process.send_signal(signal.SIGINT)
    try:
        planner_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        print("[Workflow] planner did not release the action port", file=sys.stderr)
        return False
    return planner_process.returncode == 0


def run_workflow(config: ObjectNavVlaLaunchConfig) -> int:
    """Worker process hosted in the ObjectNav tmux pane."""
    _validate_config(config)
    repo_root = Path(__file__).resolve().parents[2]
    planner_process = subprocess.Popen(_planner_command(config, repo_root), cwd=repo_root)
    runner: ObjectNavRunner | None = None
    handoff = False
    try:
        _wait_for_n(config)
        _wait_for_action_subscribers(config)
        _start_planner(config)
        runner = ObjectNavRunner(
            ObjectNavConfig(
                mission=config.mission,
                global_target=config.global_target,
                camera_host=config.camera_host,
                camera_port=config.camera_port,
                camera_timeout_ms=config.camera_timeout_ms,
                codex_timeout_seconds=config.codex_timeout_seconds,
                min_confidence=config.min_confidence,
                safe_distance=config.safe_distance,
                max_direct_travel=config.max_direct_travel,
                output_root=config.output_root,
            )
        )
        iteration = 0
        while True:
            iteration += 1
            if (
                config.max_navigation_iterations > 0
                and iteration > config.max_navigation_iterations
            ):
                raise RuntimeError("ObjectNav reached max_navigation_iterations without STOP")
            print(f"[Workflow] ObjectNav iteration {iteration}", flush=True)
            result = runner.run_once(iteration=iteration)
            if result.outcome == "STOP":
                print("[Workflow] Codex returned STOP", flush=True)
                handoff = True
                break
            if result.outcome != "NAVIGATE":
                raise RuntimeError(
                    f"ObjectNav ended fail-closed: {result.outcome}: {result.error or result.geometry}"
                )
            try:
                reply = send_object_nav_commands(
                    result.commands,
                    host=config.planner_json_host,
                    port=config.planner_json_port,
                    timeout_ms=config.planner_timeout_ms,
                )
            except SonicPlannerRequestError as exc:
                raise RuntimeError(str(exc)) from exc
            print(
                f"[Workflow] motion completed; heading={reply.get('heading_rad', 'unknown')}",
                flush=True,
            )
            if config.navigation_mode == "once":
                handoff = True
                break
            time.sleep(config.replan_delay_seconds)
    except (KeyboardInterrupt, Exception) as exc:
        print(f"[Workflow] stopped without VLA handoff: {exc}", file=sys.stderr, flush=True)
        handoff = False
    finally:
        if runner is not None:
            runner.close()
        planner_released = _safe_release_planner(config, planner_process)

    if handoff and planner_released:
        _start_vla_pane(config, repo_root)
        return 0
    print("[Workflow] VLA was not started", file=sys.stderr, flush=True)
    return 1


def _tmux_target(config: ObjectNavVlaLaunchConfig, pane: int) -> str:
    return f"={config.session_name}:workflow.{pane}"


def _send_to_pane(config: ObjectNavVlaLaunchConfig, pane: int, command: str) -> None:
    subprocess.run(
        ["tmux", "send-keys", "-t", _tmux_target(config, pane), command, "C-m"],
        check=True,
    )


def _deploy_command(config: ObjectNavVlaLaunchConfig, repo_root: Path) -> str:
    argv = [
        "./deploy.sh",
        "--input-type",
        config.deploy_input_type,
        "--zmq-host",
        config.deploy_zmq_host,
        "--hand-type",
        "dex1",
        "--dex1-kp",
        "1.0",
        "--dex1-kd",
        "0.05",
    ]
    optional = (
        ("--cp", config.deploy_checkpoint),
        ("--obs-config", config.deploy_obs_config),
        ("--planner", config.deploy_planner),
        ("--motion-data", config.deploy_motion_data),
        ("--output-type", config.deploy_output_type),
    )
    for flag, value in optional:
        if value:
            argv.extend((flag, value))
    argv.append("real")
    deploy_dir = repo_root / "gear_sonic_deploy"
    return f"cd {shlex.quote(str(deploy_dir))} && {shlex.join(argv)}"


def _create_tmux_session(config: ObjectNavVlaLaunchConfig) -> None:
    subprocess.run(["tmux", "new-session", "-d", "-s", config.session_name], check=True)
    subprocess.run(
        ["tmux", "rename-window", "-t", f"={config.session_name}:0", "workflow"],
        check=True,
    )
    subprocess.run(["tmux", "split-window", "-t", _tmux_target(config, 0), "-h"], check=True)
    subprocess.run(["tmux", "split-window", "-t", _tmux_target(config, 0), "-v"], check=True)
    subprocess.run(["tmux", "split-window", "-t", _tmux_target(config, 2), "-v"], check=True)
    subprocess.run(["tmux", "set-option", "-t", f"={config.session_name}", "-g", "mouse", "on"], check=True)
    for pane, title in enumerate(("deploy", "keyboard", "object_nav", "vla")):
        subprocess.run(
            ["tmux", "select-pane", "-t", _tmux_target(config, pane), "-T", title],
            check=True,
        )


def launch(config: ObjectNavVlaLaunchConfig) -> int:
    _validate_config(config)
    repo_root = Path(__file__).resolve().parents[2]
    python = repo_root / ".venv_inference" / "bin" / "python"
    if not shutil.which("tmux"):
        raise RuntimeError("tmux is not installed")
    if not python.is_file():
        raise RuntimeError(".venv_inference is missing")

    existing = subprocess.run(
        ["tmux", "has-session", "-t", f"={config.session_name}"],
        capture_output=True,
    ).returncode == 0
    if existing and not config.replace_session:
        raise RuntimeError(
            f"tmux session {config.session_name!r} already exists; attach or use --replace-session"
        )
    if existing:
        subprocess.run(["tmux", "kill-session", "-t", f"={config.session_name}"], check=True)

    _create_tmux_session(config)
    _send_to_pane(config, 0, _deploy_command(config, repo_root))
    keyboard_argv = [
        str(python),
        "gear_sonic/scripts/keyboard_command_publisher.py",
        "--host",
        "*",
        "--port",
        str(config.keyboard_port),
    ]
    _send_to_pane(
        config,
        1,
        f"cd {shlex.quote(str(repo_root))} && {shlex.join(keyboard_argv)}",
    )

    serialized = base64.urlsafe_b64encode(json.dumps(asdict(config)).encode()).decode()
    worker_argv = [
        str(python),
        "gear_sonic/scripts/launch_object_nav_vla.py",
        "--mission",
        config.mission,
        "--global-target",
        config.global_target,
        "--worker-config-b64",
        serialized,
    ]
    _send_to_pane(
        config,
        2,
        f"cd {shlex.quote(str(repo_root))} && {shlex.join(worker_argv)}",
    )
    _send_to_pane(
        config,
        3,
        "printf '\\n[VLA] Waiting for ObjectNav handoff. After start use keyboard: i -> p\\n'",
    )
    subprocess.run(["tmux", "select-pane", "-t", _tmux_target(config, 1)], check=True)
    print(f"Started tmux session {config.session_name}")
    print("Confirm C++ deploy, then type N in the keyboard pane to start ObjectNav.")
    if config.attach:
        subprocess.run(["tmux", "attach", "-t", f"={config.session_name}"])
    return 0


def main(config: ObjectNavVlaLaunchConfig) -> int:
    if config.worker_config_b64:
        decoded = json.loads(base64.urlsafe_b64decode(config.worker_config_b64).decode())
        decoded["worker_config_b64"] = ""
        return run_workflow(ObjectNavVlaLaunchConfig(**decoded))
    return launch(config)


if __name__ == "__main__":
    raise SystemExit(main(tyro.cli(ObjectNavVlaLaunchConfig)))
