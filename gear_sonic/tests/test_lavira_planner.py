"""Behavioral tests for the single-cycle LaViRA-to-REASAN command source."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import math
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import types

import pytest


# launch_inference bootstraps its production virtual environment when tyro is
# unavailable. The command builder is pure, so a minimal tyro module keeps its
# import side-effect free in this software-only test suite.
sys.modules.setdefault("tyro", types.ModuleType("tyro"))

from gear_sonic.scripts.launch_inference import (
    InferenceLaunchConfig,
    _check_prerequisites,
    build_planner_input_command,
    build_reasan_planner_command,
)

from gear_sonic.scripts.lavira_planner import (
    CommandValidationError,
    LaviraPlannerConfig,
    LaviraPlannerController,
    LaviraPlannerRuntime,
    ObjectNavBatch,
    PlannerBusyError,
    VelocityCommand,
    WorkerResult,
    _PlannerTermination,
    _TerminationControl,
    _install_termination_handlers,
    build_reasan_velocity_message,
    run_inference_worker,
    run_planner_loop,
    validate_object_nav_batch,
)
from gear_sonic.utils.inference.object_nav import ObjectNavResult


def test_launch_keyboard_input_preserves_existing_pane_command() -> None:
    repo_root = Path("/workspace/sonic")

    command = build_planner_input_command(InferenceLaunchConfig(), repo_root)

    assert command == (
        "cd /workspace/sonic && "
        "source .venv_teleop/bin/activate && "
        "python gear_sonic/scripts/keyboard_planner_thread_server.py "
        "--port 5558 --hz 20 --host localhost "
    )


def test_launch_lavira_input_quotes_mission_and_passes_agentnav_values() -> None:
    repo_root = Path("/workspace/sonic")
    config = InferenceLaunchConfig(
        planner_input="lavira",
        lavira_mission="approach O'Reilly's chair",
        lavira_global_target="red chair; echo unsafe",
        lavira_model="gpt-5.6-luna",
        lavira_debug=True,
        lavira_host="*",
        lavira_planner_hz=19.5,
        lavira_transition_pause=0.75,
        lavira_final_stop_count=4,
        lavira_max_speed=0.45,
        lavira_max_duration=29.0,
        lavira_max_abs_yaw=3.0,
        camera_host="camera host",
        camera_port=5555,
        lavira_camera_timeout_ms=2500,
        lavira_codex_timeout_seconds=90.0,
        lavira_min_confidence=0.7,
        lavira_rotation_speed=0.35,
        lavira_forward_speed=0.25,
        lavira_target_standoff_distance=0.2,
        lavira_max_direct_travel=7.0,
        lavira_output_root="outputs/nav runs",
    )
    command = build_planner_input_command(config, repo_root)

    assert command == (
        "cd /workspace/sonic && "
        "ready_file=/tmp/sonic_lingbot_ready_$$; rm -f $ready_file; "
        "PYTHONPATH=/workspace/sonic .venv_lingbot_depth/bin/python "
        "gear_sonic/scripts/run_lingbot_depth_viewer.py "
        "--camera-host 'camera host' --camera-port 5555 --publish-port 5564 "
        "--ready-file $ready_file & viewer_pid=$!; "
        "trap 'kill $viewer_pid 2>/dev/null; rm -f $ready_file' EXIT; "
        "while [ ! -f $ready_file ]; do kill -0 $viewer_pid 2>/dev/null || "
        "{ wait $viewer_pid; exit 1; }; sleep 0.2; done; "
        ".venv_inference/bin/python gear_sonic/scripts/lavira_planner.py "
        "--mission 'approach O'\"'\"'Reilly'\"'\"'s chair' "
        "--global-target 'red chair; echo unsafe' "
        "--model gpt-5.6-luna "
        "--debug --host '*' --port 5558 "
        "--planner-hz 19.5 --transition-pause 0.75 --final-stop-count 4 "
        "--max-speed 0.45 --max-duration 29.0 --max-abs-yaw 3.0 "
        "--camera-host 127.0.0.1 --camera-port 5564 "
        "--camera-timeout-ms 2500 --codex-timeout-seconds 90.0 "
        "--min-confidence 0.7 --rotation-speed 0.35 --forward-speed 0.25 "
        "--target-standoff-distance 0.2 --max-direct-travel 7.0 "
        "--output-root 'outputs/nav runs'"
    )


def test_launch_can_disable_lavira_warmup() -> None:
    command = build_planner_input_command(
        InferenceLaunchConfig(
            planner_input="lavira",
            lavira_mission="find chair",
            lavira_global_target="chair",
            lavira_warmup=False,
        ),
        Path("/workspace/sonic"),
    )

    assert "--no-warmup" in command


def test_launch_lavira_can_select_qwenvl_backend() -> None:
    command = build_planner_input_command(
        InferenceLaunchConfig(
            planner_input="lavira",
            lavira_mission="find chair",
            lavira_global_target="chair",
            lavira_vision_backend="qwenvl",
        ),
        Path("/workspace/sonic"),
    )

    assert "--vision-backend qwenvl" in command
    assert ". ./.env.local" in command
    assert "--qwenvl-model qwen3-vl-32b-instruct" in command
    assert (
        "--qwenvl-base-url "
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    ) in command


def test_launch_lavira_starts_depth_viewer_in_same_pane_shell() -> None:
    command = build_planner_input_command(
        InferenceLaunchConfig(
            planner_input="lavira",
            lavira_mission="find basket",
            lavira_global_target="basket",
            camera_host="192.168.123.164",
            camera_port=5555,
        ),
        Path("/workspace/sonic"),
    )

    assert ".venv_lingbot_depth/bin/python" in command
    assert "PYTHONPATH=/workspace/sonic .venv_lingbot_depth/bin/python" in command
    assert "gear_sonic/scripts/run_lingbot_depth_viewer.py" in command
    assert "--camera-host 192.168.123.164 --camera-port 5555" in command
    assert "--publish-port 5564" in command
    assert "ready_file=/tmp/sonic_lingbot_ready_$$" in command
    assert "--ready-file $ready_file" in command
    assert "while [ ! -f $ready_file ]" in command
    assert "kill -0 $viewer_pid" in command
    assert "lavira_planner.py" in command
    assert "--camera-host 127.0.0.1 --camera-port 5564" in command
    assert "--camera-timeout-ms 15000" in command
    assert "viewer_pid=$!" in command
    assert "trap 'kill $viewer_pid 2>/dev/null; rm -f $ready_file' EXIT" in command


def test_launch_uses_long_timeout_for_low_rate_lingbot_frames() -> None:
    assert InferenceLaunchConfig().lavira_camera_timeout_ms == 15000


def test_disabled_reasan_uses_standalone_lavira_sonic_relay() -> None:
    config = InferenceLaunchConfig(reasan_avoidance=False)

    command = build_reasan_planner_command(config, Path("/workspace/sonic"))

    assert "python gear_sonic/scripts/lavira_sonic_relay.py" in command
    assert "reasan_planner.py" not in command
    assert "--filter" not in command
    assert "--ray-endpoint" not in command


def test_reasan_safety_subscribes_to_raw_chest_depth_camera() -> None:
    config = InferenceLaunchConfig(
        reasan_avoidance=True,
        camera_host="192.168.123.164",
        camera_port=5555,
    )

    command = build_reasan_planner_command(config, Path("/workspace/sonic"))

    assert "--camera-host 192.168.123.164" in command
    assert "--camera-port 5555" in command


def test_launch_requires_lavira_mission_and_target_only_when_selected(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("gear_sonic.scripts.launch_inference.shutil.which", lambda _: "/tmux")
    monkeypatch.setattr(
        "gear_sonic.scripts.launch_inference.Path.exists", lambda _: True
    )
    monkeypatch.setattr(
        "gear_sonic.scripts.launch_inference.Path.is_file", lambda _: True
    )

    _check_prerequisites(InferenceLaunchConfig(data_exporter=False))

    with pytest.raises(SystemExit):
        _check_prerequisites(
            InferenceLaunchConfig(planner_input="lavira", data_exporter=False)
        )

    assert capsys.readouterr().out == (
        "ERROR: Prerequisites not met:\n\n"
        "  - --lavira-mission is required when --planner-input lavira\n"
        "  - --lavira-global-target is required when --planner-input lavira\n"
        "\n"
    )


def commands(
    *,
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.4, 1.0),
    translation: tuple[float, float, float, float] = (0.3, 0.0, 0.0, 2.0),
) -> dict[str, object]:
    return {
        "commands": [
            dict(zip(("vx", "vy", "wz", "duration"), rotation)),
            dict(zip(("vx", "vy", "wz", "duration"), translation)),
        ]
    }


def result(
    outcome: str = "NAVIGATE", payload: dict[str, object] | None = None
) -> ObjectNavResult:
    return ObjectNavResult(
        outcome=outcome,
        policy={},
        commands=commands() if payload is None else payload,
        geometry={},
        output_dir="/tmp/object-nav",
        error=None,
    )


def decoded_messages(messages: list[str]) -> list[dict[str, object]]:
    return [json.loads(message) for message in messages]


def assert_stop_messages(messages: list[str], count: int = 3) -> None:
    decoded = decoded_messages(messages[-count:])
    assert [message["action"] for message in decoded] == ["stop"] * count
    assert [message["velocity"] for message in decoded] == [
        {"vx": 0.0, "vy": 0.0, "wz": 0.0}
    ] * count


def test_velocity_command_is_immutable_and_exposes_velocity_tuple() -> None:
    command = VelocityCommand(0.3, -0.1, 0.2, 0.5)

    assert command.velocity == (0.3, -0.1, 0.2)
    with pytest.raises(FrozenInstanceError):
        command.vx = 1.0  # type: ignore[misc]


def test_adapter_builds_existing_reasan_velocity_protocol() -> None:
    message = json.loads(
        build_reasan_velocity_message(
            VelocityCommand(0.0, 0.0, -0.4, 1.25), action="turn_right"
        )
    )

    assert message["type"] == "navila_reasan_velocity_command"
    assert message["version"] == 1
    assert message["status"] == "ok"
    assert message["action"] == "turn_right"
    assert message["duration_s"] == 1.25
    assert message["velocity"] == {"vx": 0.0, "vy": 0.0, "wz": -0.4}
    assert message["segments"] == [
        {"duration_s": 1.25, "vx": 0.0, "vy": 0.0, "wz": -0.4}
    ]


def test_adapter_missing_tyro_fallback_is_hermetic_and_cleans_up() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = """
import builtins
import json
import sys

from gear_sonic.scripts.lavira_planner import (
    VelocityCommand,
    build_reasan_velocity_message,
)

real_import = builtins.__import__

def import_without_tyro(name, *args, **kwargs):
    if name == "tyro" and "tyro" not in sys.modules:
        raise ModuleNotFoundError("No module named 'tyro'", name="tyro")
    return real_import(name, *args, **kwargs)

builtins.__import__ = import_without_tyro
message = json.loads(build_reasan_velocity_message(
    VelocityCommand(0.0, 0.0, 0.0, 0.05), action="stop"
))
print(json.dumps({"action": message["action"], "tyro_present": "tyro" in sys.modules}))
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {
        "action": "stop",
        "tyro_present": False,
    }


def test_validation_accepts_exact_rotation_then_translation_boundaries() -> None:
    batch = validate_object_nav_batch(
        commands(
            rotation=(0.0, 0.0, math.pi / 30.0, 30.0),
            translation=(0.3, 0.4, 0.0, 0.05),
        )
    )

    assert batch == ObjectNavBatch(
        rotation=VelocityCommand(0.0, 0.0, math.pi / 30.0, 30.0),
        translation=VelocityCommand(0.3, 0.4, 0.0, 0.05),
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "exactly two"),
        ({"commands": [{}, {}, {}]}, "exactly two"),
        (commands(rotation=(0.001, 0.0, 0.4, 1.0)), "pure rotation"),
        (commands(translation=(0.3, 0.0, 0.001, 2.0)), "pure translation"),
        (commands(rotation=(0.0, 0.0, float("nan"), 1.0)), "finite"),
        (commands(rotation=(0.0, 0.0, 0.4, 0.049)), "publish period"),
        (commands(translation=(0.3, 0.0, 0.0, 30.001)), "duration exceeds"),
        (commands(translation=(0.5, 0.001, 0.0, 2.0)), "speed exceeds"),
        (commands(rotation=(0.0, 0.0, 0.4, math.pi / 0.4 + 0.001)), "yaw exceeds"),
    ],
)
def test_validation_rejects_malformed_or_unsafe_batches(
    payload: object, message: str
) -> None:
    with pytest.raises(CommandValidationError, match=message):
        validate_object_nav_batch(payload)


@pytest.mark.parametrize(
    "payload",
    [
        commands(rotation=(0.0, 0.0, 0.4, -0.1)),
        commands(translation=(0.3, 0.0, 0.0, -0.1)),
        commands(rotation=(0.0, 0.0, 0.4, True)),
        {"commands": [
            {"vx": 0.0, "vy": 0.0, "wz": 0.4},
            {"vx": 0.3, "vy": 0.0, "wz": 0.0, "duration": 1.0},
        ]},
    ],
)
def test_validation_rejects_invalid_command_fields(payload: object) -> None:
    with pytest.raises(CommandValidationError):
        validate_object_nav_batch(payload)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_speed": 0.0}, "limits"),
        ({"max_duration": float("inf")}, "limits"),
        ({"max_abs_yaw": True}, "limits"),
        ({"min_positive_duration": -0.1}, "min_positive_duration"),
    ],
)
def test_validation_rejects_invalid_safety_limits(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_object_nav_batch(commands(), **kwargs)  # type: ignore[arg-type]


def test_controller_obeys_rotation_pause_translation_deadlines() -> None:
    controller = LaviraPlannerController(transition_pause=0.5)
    controller.start(result(), now=10.0)

    assert controller.phase == "rotating"
    assert controller.step(10.0).velocity == (0.0, 0.0, 0.4)
    assert controller.step(10.999).velocity == (0.0, 0.0, 0.4)
    assert controller.step(11.0).velocity == (0.0, 0.0, 0.0)
    assert controller.phase == "transition_pause"
    assert controller.step(11.499).velocity == (0.0, 0.0, 0.0)
    assert controller.step(11.5).velocity == (0.3, 0.0, 0.0)
    assert controller.phase == "translating"
    assert controller.step(13.499).velocity == (0.3, 0.0, 0.0)
    assert controller.step(13.5).velocity == (0.0, 0.0, 0.0)
    assert controller.phase == "idle"
    assert controller.failure_reason is None


def test_delayed_rotation_transition_starts_full_pause_from_observation() -> None:
    controller = LaviraPlannerController(transition_pause=0.5)
    controller.start(result(), now=10.0)

    assert controller.step(11.4).velocity == (0.0, 0.0, 0.0)
    assert controller.phase == "transition_pause"
    assert controller.step(11.899).velocity == (0.0, 0.0, 0.0)
    assert controller.step(11.9).velocity == (0.3, 0.0, 0.0)


def test_controller_finishes_directly_in_idle() -> None:
    controller = LaviraPlannerController()
    controller.start(
        result(
            payload=commands(
                rotation=(0.0, 0.0, 0.0, 0.0),
                translation=(0.0, 0.0, 0.0, 0.0),
            )
        ),
        now=2.0,
    )

    assert controller.step(2.5).velocity == (0.0, 0.0, 0.0)
    assert controller.phase == "idle"


def test_controller_skips_zero_duration_motion_phases_but_keeps_pause() -> None:
    controller = LaviraPlannerController(transition_pause=0.5)
    controller.start(
        result(payload=commands(rotation=(0.0, 0.0, 0.0, 0.0))), now=4.0
    )

    assert controller.phase == "transition_pause"
    assert controller.step(4.0).velocity == (0.0, 0.0, 0.0)
    assert controller.step(4.5).velocity == (0.3, 0.0, 0.0)


@pytest.mark.parametrize("outcome", ["STOP", "FAILED", "REJECTED", "UNKNOWN"])
def test_non_navigation_results_fail_closed_directly_to_idle(outcome: str) -> None:
    controller = LaviraPlannerController()
    controller.start(result(outcome), now=1.0)

    assert controller.phase == "idle"
    assert controller.step(1.0).velocity == (0.0, 0.0, 0.0)


def test_model_stop_is_successful_idle_not_a_failure() -> None:
    controller = LaviraPlannerController()
    controller.start(result("STOP"), now=1.0)

    assert controller.phase == "idle"
    assert controller.failure_reason is None


def test_invalid_navigation_result_fails_closed_without_motion() -> None:
    controller = LaviraPlannerController()
    controller.start(
        result(payload=commands(translation=(0.3, 0.0, 0.2, 1.0))), now=1.0
    )

    assert controller.phase == "idle"
    assert controller.failure_reason is not None
    assert controller.step(1.0).velocity == (0.0, 0.0, 0.0)


def test_controller_rejects_busy_start_and_nonfinite_clock() -> None:
    controller = LaviraPlannerController()
    controller.start(result(), now=1.0)

    with pytest.raises(PlannerBusyError):
        controller.start(result(), now=2.0)
    with pytest.raises(ValueError, match="finite"):
        controller.step(float("nan"))


def test_controller_rejects_zero_stop_message_duration() -> None:
    with pytest.raises(ValueError, match="min_positive_duration"):
        LaviraPlannerController(min_positive_duration=0.0)


def test_config_preserves_agentnav_and_keyboard_defaults() -> None:
    config = LaviraPlannerConfig(mission="find chair", global_target="chair")

    assert config.port == 5558
    assert config.planner_hz == 20.0
    assert config.transition_pause == 0.5
    assert config.max_speed == 0.5
    assert config.max_duration == 30.0
    assert config.max_abs_yaw == math.pi
    assert config.camera_timeout_ms == 3000
    assert config.codex_timeout_seconds == 180.0
    assert config.min_confidence == 0.6
    assert config.rotation_speed == 0.4
    assert config.forward_speed == 0.3
    assert config.target_standoff_distance == 0.0
    assert config.max_direct_travel == 8.0


def test_runtime_uses_one_slot_queues_and_rejects_repeated_n() -> None:
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(mission="find chair", global_target="chair"),
        publish=lambda _message: None,
        sleep=lambda _duration: None,
    )

    assert runtime.request_queue.maxsize == 1
    assert runtime.result_queue.maxsize == 1
    assert runtime.handle_key("n", now=1.0) == "started"
    assert runtime.phase == "nav"
    assert runtime.generation == 1
    assert runtime.handle_key("N", now=1.1) == "busy"
    assert runtime.generation == 1
    assert runtime.request_queue.get_nowait() == 1


def test_runtime_starts_in_listen_wasd_and_republishes_held_key_at_20_hz() -> None:
    published: list[str] = []
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(mission="find chair", global_target="chair"),
        publish=published.append,
        sleep=lambda _duration: None,
    )

    assert runtime.phase == "listen_wasd"
    assert runtime.handle_key("w", now=1.0) == "manual"
    assert published == []
    assert runtime.publish_due(1.0).velocity == (0.3, 0.0, 0.0)
    assert runtime.publish_due(1.049) is None
    assert runtime.publish_due(1.05).velocity == (0.3, 0.0, 0.0)


def test_listen_wasd_releases_to_stop_after_keyboard_hold_window() -> None:
    published: list[str] = []
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(mission="find chair", global_target="chair"),
        publish=published.append,
        sleep=lambda _duration: None,
    )

    runtime.handle_key("w", now=2.0)
    assert runtime.publish_due(2.50).velocity == (0.3, 0.0, 0.0)
    assert runtime.publish_due(2.55).velocity == (0.0, 0.0, 0.0)
    assert runtime.phase == "listen_wasd"


@pytest.mark.parametrize("key", ["w", "s", "a", "d", "q", "e", "n"])
def test_nav_rejects_every_key_except_space_and_exit(key: str) -> None:
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(mission="find chair", global_target="chair"),
        publish=lambda _message: None,
        sleep=lambda _duration: None,
    )
    assert runtime.handle_key("n", now=1.0) == "started"

    assert runtime.handle_key(key, now=1.1) in {"ignored", "busy"}
    assert runtime.phase == "nav"
    assert runtime.generation == 1


def test_space_is_the_only_motion_key_that_returns_nav_to_listen_wasd() -> None:
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(mission="find chair", global_target="chair"),
        publish=lambda _message: None,
        sleep=lambda _duration: None,
    )
    runtime.handle_key("n", now=1.0)

    assert runtime.handle_key(" ", now=1.1) == "cancelled"
    assert runtime.phase == "listen_wasd"


def test_cancel_increments_generation_publishes_stops_and_discards_late_result() -> None:
    published: list[str] = []
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(mission="find chair", global_target="chair"),
        publish=published.append,
        sleep=lambda _duration: None,
    )
    assert runtime.handle_key("n", now=1.0) == "started"

    assert runtime.handle_key(" ", now=1.1) == "cancelled"
    assert runtime.generation == 2
    assert runtime.phase == "listen_wasd"
    assert decoded_messages(published)[-1]["action"] == "stop"

    assert runtime.accept_worker_result(WorkerResult(1, result(), None), now=2.0) is False
    assert runtime.phase == "listen_wasd"
    assert runtime.publish_due(2.0).velocity == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("key", ["w", "s", "a", "d", "q", "e"])
def test_listen_wasd_manual_keys_reuse_keyboard_defaults(key: str) -> None:
    published: list[str] = []
    logs: list[str] = []
    config = LaviraPlannerConfig(mission="find chair", global_target="chair")
    runtime = LaviraPlannerRuntime(
        config,
        publish=published.append,
        sleep=lambda _duration: None,
        logger=logs.append,
    )
    assert runtime.handle_key(key.upper(), now=1.1) == "manual"
    assert runtime.generation == 0
    assert runtime.phase == "listen_wasd"
    assert "[LaViRA] STATE FINAL_STOP" not in logs
    runtime.publish_due(1.1)
    expected_action, expected_velocity = {
        "w": ("forward", (0.3, 0.0, 0.0)),
        "s": ("backward", (-0.3, 0.0, 0.0)),
        "a": ("move_left", (0.0, 0.15, 0.0)),
        "d": ("move_right", (0.0, -0.15, 0.0)),
        "q": ("turn_left", (0.0, 0.0, 0.5)),
        "e": ("turn_right", (0.0, 0.0, -0.5)),
    }[key]
    manual = json.loads(published[-1])
    assert manual["action"] == expected_action
    assert manual["velocity"] == dict(zip(("vx", "vy", "wz"), expected_velocity))
    assert manual["duration_s"] == 0.05


def test_manual_key_is_published_by_nonblocking_20_hz_loop() -> None:
    events: list[tuple[str, object]] = []

    def publish(message: str) -> None:
        events.append(("publish", json.loads(message)["action"]))

    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(mission="find chair", global_target="chair"),
        publish=publish,
        sleep=lambda duration: events.append(("sleep", duration)),
    )

    assert runtime.handle_key("w", now=1.0) == "manual"
    assert events == []
    runtime.publish_due(1.0)
    assert events == [("publish", "forward")]


def test_termination_gate_suppresses_pending_manual_nonzero() -> None:
    published: list[str] = []
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(mission="find chair", global_target="chair"),
        publish=published.append,
        sleep=lambda _duration: None,
    )

    runtime.handle_key("w", now=1.0)
    assert runtime.publish_due(1.0, running=lambda: False) is None

    assert all(message["action"] == "stop" for message in decoded_messages(published))


def test_shutdown_spaces_all_repeated_stop_publications() -> None:
    events: list[tuple[str, object]] = []
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(mission="find chair", global_target="chair"),
        publish=lambda message: events.append(
            ("publish", json.loads(message)["action"])
        ),
        sleep=lambda duration: events.append(("sleep", duration)),
    )

    runtime.shutdown()

    assert events == [
        ("publish", "stop"),
        ("sleep", 0.05),
        ("publish", "stop"),
        ("sleep", 0.05),
        ("publish", "stop"),
        ("sleep", 0.05),
    ]


def test_runtime_logs_each_state_transition_once() -> None:
    logs: list[str] = []
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(mission="find chair", global_target="chair"),
        publish=lambda _message: None,
        sleep=lambda _duration: None,
        logger=logs.append,
    )

    assert runtime.handle_key("n", now=0.0) == "started"
    assert runtime.accept_worker_result(WorkerResult(1, result(), None), now=0.0)
    for now in (0.0, 0.5, 1.0, 1.25, 1.5, 2.0, 3.5, 3.55, 3.60):
        runtime.publish_due(now)

    assert logs == [
        "[LaViRA] STATE LISTEN_WASD",
        "[LaViRA] STATE NAV",
        "[LaViRA] STATE LISTEN_WASD",
    ]


def test_runtime_logs_busy_rejection_without_duplicate_state() -> None:
    logs: list[str] = []
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(mission="find chair", global_target="chair"),
        publish=lambda _message: None,
        sleep=lambda _duration: None,
        logger=logs.append,
    )

    assert runtime.handle_key("n", now=0.0) == "started"
    assert runtime.handle_key("n", now=0.1) == "busy"

    assert logs == [
        "[LaViRA] STATE LISTEN_WASD",
        "[LaViRA] STATE NAV",
        "[LaViRA] BUSY navigation request rejected",
    ]


def test_runtime_logs_cancellation_and_worker_failure_reasons() -> None:
    logs: list[str] = []
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(mission="find chair", global_target="chair"),
        publish=lambda _message: None,
        sleep=lambda _duration: None,
        logger=logs.append,
    )
    runtime.handle_key("n", now=0.0)
    assert runtime.accept_worker_result(
        WorkerResult(1, None, "camera timeout"), now=0.1
    )
    runtime.handle_key(" ", now=0.2)

    assert "[LaViRA] FAILURE camera timeout" in logs
    assert runtime.phase == "listen_wasd"


def test_runtime_logs_segmented_object_nav_latency() -> None:
    logs: list[str] = []
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(mission="find chair", global_target="chair"),
        publish=lambda _message: None,
        sleep=lambda _duration: None,
        logger=logs.append,
    )
    runtime.handle_key("n", now=0.0)
    nav_result = ObjectNavResult(
        outcome="STOP",
        policy={},
        commands={"commands": []},
        geometry={
            "timing_s": {
                "camera_rgbd": 0.25,
                "image_io": 0.10,
                "auth_check": 0.20,
                "api_inference": 7.31,
                "postprocess": 0.06,
                "total": 7.92,
            }
        },
        output_dir="/tmp/object-nav",
    )

    runtime.accept_worker_result(WorkerResult(1, nav_result, None), now=8.0)

    assert (
        "[LaViRA] latency total=7.920s api=7.310s auth=0.200s "
        "camera=0.250s io=0.100s post=0.060s"
    ) in logs


def test_exit_key_cancels_and_publishes_repeated_stop() -> None:
    published: list[str] = []
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(mission="find chair", global_target="chair"),
        publish=published.append,
        sleep=lambda _duration: None,
    )
    runtime.handle_key("n", now=1.0)

    assert runtime.handle_key("x", now=1.1) == "exit"
    assert runtime.generation == 2
    assert_stop_messages(published)


def test_current_worker_result_starts_motion_and_stale_or_error_results_do_not() -> None:
    published: list[str] = []
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(mission="find chair", global_target="chair"),
        publish=published.append,
        sleep=lambda _duration: None,
    )
    runtime.handle_key("n", now=10.0)
    assert runtime.request_queue.get_nowait() == 1

    assert runtime.accept_worker_result(WorkerResult(0, result(), None), now=10.0) is False
    assert runtime.phase == "nav"
    assert runtime.accept_worker_result(WorkerResult(1, result(), None), now=10.0) is True
    assert runtime.phase == "nav"

    runtime.handle_key(" ", now=10.1)
    runtime.handle_key("n", now=10.2)
    assert runtime.accept_worker_result(
        WorkerResult(3, None, "camera timeout"), now=10.3
    ) is True
    assert runtime.phase == "listen_wasd"
    assert runtime.publish_due(10.3).velocity == (0.0, 0.0, 0.0)
    assert_stop_messages(published)


def test_poll_worker_results_drains_queue_and_disregards_stale_generation() -> None:
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(mission="find chair", global_target="chair"),
        publish=lambda _message: None,
        sleep=lambda _duration: None,
    )
    runtime.handle_key("n", now=1.0)
    runtime.result_queue.put_nowait(WorkerResult(0, result(), None))

    assert runtime.poll_worker_results(now=2.0) == 0
    assert runtime.phase == "nav"
    assert runtime.result_queue.empty()


class RequestSequence:
    def __init__(self, *items: int | None):
        self.items = iter(items)

    def get(self) -> int | None:
        return next(self.items)


class FakeRunner:
    def __init__(self, *, error: Exception | None = None):
        self.error = error
        self.closed = False

    def run_once(self) -> ObjectNavResult:
        if self.error is not None:
            raise self.error
        return result()

    def close(self) -> None:
        self.closed = True


class WarmupRunner(FakeRunner):
    def __init__(self, *, warmup_error: str | None = None):
        super().__init__()
        self.warmup_error = warmup_error
        self.warmup_count = 0
        self.run_count = 0

    def warmup(self) -> ObjectNavResult:
        self.warmup_count += 1
        return ObjectNavResult(
            outcome="FAILED" if self.warmup_error else "STOP",
            policy={},
            commands={"commands": []},
            geometry={"timing_s": {"total": 4.25}},
            output_dir="/tmp/warmup",
            error=self.warmup_error,
        )

    def run_once(self) -> ObjectNavResult:
        self.run_count += 1
        return result()


@pytest.mark.parametrize("warmup_error", [None, "temporary API failure"])
def test_worker_warms_up_once_without_publishing_result_or_blocking_requests(
    warmup_error: str | None,
) -> None:
    runner = WarmupRunner(warmup_error=warmup_error)
    results: queue.Queue[WorkerResult] = queue.Queue(maxsize=1)
    logs: list[str] = []

    run_inference_worker(
        lambda: runner,
        RequestSequence(7, None),  # type: ignore[arg-type]
        results,
        warmup=True,
        logger=logs.append,
    )

    assert runner.warmup_count == 1
    assert runner.run_count == 1
    assert results.get_nowait().generation == 7
    assert any("warmup" in message.lower() for message in logs)


def test_worker_tags_result_replaces_full_slot_and_closes_runner() -> None:
    runner = FakeRunner()
    results: queue.Queue[WorkerResult] = queue.Queue(maxsize=1)
    results.put_nowait(WorkerResult(2, result("FAILED"), None))

    run_inference_worker(
        lambda: runner,
        RequestSequence(7, None),  # type: ignore[arg-type]
        results,
    )

    item = results.get_nowait()
    assert item.generation == 7
    assert item.result == result()
    assert item.error is None
    assert runner.closed is True


def test_worker_converts_inference_exception_to_generation_tagged_error() -> None:
    runner = FakeRunner(error=RuntimeError("camera unavailable"))
    results: queue.Queue[WorkerResult] = queue.Queue(maxsize=1)

    run_inference_worker(
        lambda: runner,
        RequestSequence(4, None),  # type: ignore[arg-type]
        results,
    )

    assert results.get_nowait() == WorkerResult(4, None, "camera unavailable")
    assert runner.closed is True


def test_shutdown_replaces_full_request_and_worker_does_not_start_it() -> None:
    stop_event = threading.Event()
    requests: queue.Queue[int | None] = queue.Queue(maxsize=1)
    results: queue.Queue[WorkerResult] = queue.Queue(maxsize=1)
    factory_calls: list[str] = []
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(mission="find chair", global_target="chair"),
        publish=lambda _message: None,
        sleep=lambda _duration: None,
        request_queue=requests,
        worker_stop_event=stop_event,
    )
    assert runtime.handle_key("n", now=0.0) == "started"

    runtime.shutdown("test_teardown")

    assert stop_event.is_set()
    assert requests.get_nowait() is None
    requests.put_nowait(None)

    def runner_factory() -> FakeRunner:
        factory_calls.append("started")
        return FakeRunner()

    worker = threading.Thread(
        target=run_inference_worker,
        args=(runner_factory, requests, results, stop_event),
    )
    worker.start()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert factory_calls == []
    assert results.empty()


def test_publish_due_uses_20_hz_cadence_and_command_action_names() -> None:
    published: list[str] = []
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(mission="find chair", global_target="chair"),
        publish=published.append,
        sleep=lambda _duration: None,
    )
    runtime.handle_key("n", now=10.0)
    runtime.accept_worker_result(WorkerResult(1, result(), None), now=10.0)

    assert runtime.publish_due(10.0).velocity == (0.0, 0.0, 0.4)
    assert runtime.publish_due(10.049) is None
    assert runtime.publish_due(10.05).velocity == (0.0, 0.0, 0.4)
    assert [message["action"] for message in decoded_messages(published)[-2:]] == [
        "turn_left",
        "turn_left",
    ]


def test_runtime_maps_negative_yaw_translation_and_stop_actions() -> None:
    published: list[str] = []
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(mission="find chair", global_target="chair"),
        publish=published.append,
        sleep=lambda _duration: None,
    )
    runtime.handle_key("n", now=0.0)
    runtime.accept_worker_result(
        WorkerResult(
            1,
            result(
                payload=commands(
                    rotation=(0.0, 0.0, -0.4, 0.05),
                    translation=(0.3, 0.0, 0.0, 0.05),
                )
            ),
            None,
        ),
        now=0.0,
    )

    for now in (0.0, 0.05, 0.55, 0.60, 0.65, 0.70):
        runtime.publish_due(now)

    assert [message["action"] for message in decoded_messages(published)[1:5]] == [
        "turn_right",
        "stop",
        "move_forward",
        "stop",
    ]
    assert runtime.phase == "listen_wasd"


class RunningSequence:
    def __init__(self, *values: bool):
        self.values = iter(values)
        self.last = values[-1]

    def __call__(self) -> bool:
        self.last = next(self.values, self.last)
        return self.last


@pytest.mark.parametrize(
    ("running_values", "result_was_accepted"),
    [
        ((True, False), False),
        ((True, True, False), True),
        ((True, True, True, False), True),
    ],
)
def test_termination_gate_prevents_nonzero_after_request(
    running_values: tuple[bool, ...], result_was_accepted: bool
) -> None:
    published: list[str] = []
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(mission="find chair", global_target="chair"),
        publish=published.append,
        sleep=lambda _duration: None,
    )
    runtime.handle_key("n", now=1.0)
    runtime.result_queue.put_nowait(WorkerResult(1, result(), None))

    run_planner_loop(
        runtime,
        read_key=lambda: None,
        monotonic=lambda: 1.0,
        sleep=lambda _duration: None,
        running=RunningSequence(*running_values),
    )

    assert all(message["action"] == "stop" for message in decoded_messages(published))
    assert runtime.result_queue.empty() is result_was_accepted


def test_keyboard_cancel_wins_when_current_worker_result_is_already_ready() -> None:
    published: list[str] = []
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(mission="find chair", global_target="chair"),
        publish=published.append,
        sleep=lambda _duration: None,
    )
    runtime.handle_key("n", now=1.0)
    runtime.result_queue.put_nowait(WorkerResult(1, result(), None))

    run_planner_loop(
        runtime,
        read_key=iter([" "]).__next__,
        monotonic=lambda: 1.0,
        sleep=lambda _duration: None,
        running=RunningSequence(True, True, True, False),
    )

    assert runtime.result_queue.empty()
    assert all(message["action"] == "stop" for message in decoded_messages(published))


class SignalingResultQueue(queue.Queue[WorkerResult]):
    def __init__(self, handler: object):
        super().__init__(maxsize=1)
        self.handler = handler

    def get_nowait(self) -> WorkerResult:
        item = super().get_nowait()
        self.handler(signal.SIGTERM, None)  # type: ignore[operator]
        return item


def test_signal_after_result_dequeue_aborts_before_controller_acceptance() -> None:
    published: list[str] = []
    termination = _TerminationControl()
    results = SignalingResultQueue(termination.request)
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(mission="find chair", global_target="chair"),
        publish=published.append,
        sleep=lambda _duration: None,
        result_queue=results,
    )
    runtime.handle_key("n", now=1.0)
    results.put_nowait(WorkerResult(1, result(), None))

    run_planner_loop(
        runtime,
        read_key=lambda: None,
        monotonic=lambda: 1.0,
        sleep=lambda _duration: None,
        running=termination.running,
    )

    assert results.empty()
    assert all(message["action"] == "stop" for message in decoded_messages(published))


def test_signal_inside_nonzero_publisher_aborts_before_recording_motion() -> None:
    published: list[str] = []
    termination = _TerminationControl()

    def publish(message: str) -> None:
        if json.loads(message)["action"] != "stop":
            termination.request(signal.SIGTERM, None)
        published.append(message)

    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(mission="find chair", global_target="chair"),
        publish=publish,
        sleep=lambda _duration: None,
    )
    runtime.handle_key("n", now=1.0)
    runtime.result_queue.put_nowait(WorkerResult(1, result(), None))

    run_planner_loop(
        runtime,
        read_key=lambda: None,
        monotonic=lambda: 1.0,
        sleep=lambda _duration: None,
        running=termination.running,
    )

    assert all(message["action"] == "stop" for message in decoded_messages(published))


def test_sighup_during_active_motion_reaches_repeated_stop_cleanup() -> None:
    published: list[str] = []
    termination = _TerminationControl()

    def publish(message: str) -> None:
        if json.loads(message)["action"] != "stop":
            termination.request(signal.SIGHUP, None)
        published.append(message)

    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(mission="find chair", global_target="chair"),
        publish=publish,
        sleep=lambda _duration: None,
    )
    runtime.handle_key("n", now=1.0)
    runtime.result_queue.put_nowait(WorkerResult(1, result(), None))

    run_planner_loop(
        runtime,
        read_key=lambda: None,
        monotonic=lambda: 1.0,
        sleep=lambda _duration: None,
        running=termination.running,
    )

    assert_stop_messages(published)


def test_repeated_sighup_during_cleanup_does_not_interrupt_stop_cadence() -> None:
    published: list[str] = []
    termination = _TerminationControl()
    with pytest.raises(_PlannerTermination):
        termination.request(signal.SIGHUP, None)

    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(mission="find chair", global_target="chair"),
        publish=published.append,
        sleep=lambda _duration: termination.request(signal.SIGHUP, None),
    )

    runtime.shutdown("sighup")

    assert_stop_messages(published)


def test_signal_handlers_are_injectable_and_raise_private_termination() -> None:
    registered: dict[int, object] = {}
    termination = _TerminationControl()

    _install_termination_handlers(
        termination,
        register=lambda signum, handler: registered.update({signum: handler}),
    )

    assert set(registered) == {signal.SIGHUP, signal.SIGINT, signal.SIGTERM}
    with pytest.raises(_PlannerTermination):
        registered[signal.SIGINT](signal.SIGINT, None)  # type: ignore[operator]
    assert termination.running() is False


@pytest.mark.parametrize(
    "exit_mode", ["key", "signal", "exception", "keyboard_interrupt"]
)
def test_event_loop_repeats_stop_on_exit_signal_and_exception(exit_mode: str) -> None:
    published: list[str] = []
    runtime = LaviraPlannerRuntime(
        LaviraPlannerConfig(mission="find chair", global_target="chair"),
        publish=published.append,
        sleep=lambda _duration: None,
    )
    if exit_mode == "key":
        keys = iter(["x"])
        read_key = lambda: next(keys)
        running = lambda: True
    elif exit_mode == "signal":
        read_key = lambda: None
        running = lambda: False
    elif exit_mode == "exception":
        def read_key() -> str:
            raise RuntimeError("terminal failed")

        running = lambda: True
    else:
        def read_key() -> str:
            raise KeyboardInterrupt

        running = lambda: True

    if exit_mode == "exception":
        with pytest.raises(RuntimeError, match="terminal failed"):
            run_planner_loop(
                runtime,
                read_key=read_key,
                monotonic=lambda: 1.0,
                sleep=lambda _duration: None,
                running=running,
            )
    else:
        run_planner_loop(
            runtime,
            read_key=read_key,
            monotonic=lambda: 1.0,
            sleep=lambda _duration: None,
            running=running,
        )

    assert_stop_messages(published)
