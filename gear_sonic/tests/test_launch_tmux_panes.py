from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest


sys.modules.setdefault("tyro", types.ModuleType("tyro"))

from gear_sonic.scripts import launch_inference
from gear_sonic.scripts.launch_inference import (
    InferenceLaunchConfig,
    _parse_pane_ids,
)


def test_parse_pane_ids_returns_stable_ids_in_visual_index_order() -> None:
    output = "2 %8\n0 %3\n1 %5\n5 %13\n4 %11\n3 %9\n"

    assert _parse_pane_ids(output) == ["%3", "%5", "%8", "%9", "%11", "%13"]


def test_parse_pane_ids_rejects_incomplete_layout() -> None:
    with pytest.raises(RuntimeError, match="expected 6 tmux panes, found 5"):
        _parse_pane_ids("0 %1\n1 %2\n2 %3\n3 %4\n4 %5\n")


def test_raw_yoloe_launches_key_triggered_base_keyboard_in_pane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, float]] = []

    def record_send(pane_id: str, command: str, wait: float = 1.0) -> None:
        calls.append((pane_id, command, wait))

    monkeypatch.setattr(launch_inference, "_send_to_pane", record_send)
    config = InferenceLaunchConfig(
        planner_input="base_pose",
        base_pose_mode="raw_yoloe_servo",
        base_pose_manual_keyboard_port=6002,
    )

    launch_inference._launch_base_pose_manual_keyboard_pane(
        config, Path("/workspace/sonic"), "%13"
    )

    assert calls == [
        (
            "%13",
            "cd /workspace/sonic && source .venv_teleop/bin/activate && "
            "python gear_sonic/scripts/keyboard_planner_thread_server.py "
            "--port 6002 --hz 20 --host localhost",
            1.0,
        )
    ]


def test_non_raw_base_pose_does_not_launch_base_keyboard_in_pane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, float]] = []
    monkeypatch.setattr(
        launch_inference,
        "_send_to_pane",
        lambda pane_id, command, wait=1.0: calls.append((pane_id, command, wait)),
    )

    launch_inference._launch_base_pose_manual_keyboard_pane(
        InferenceLaunchConfig(planner_input="base_pose", base_pose_mode="rgb"),
        Path("/workspace/sonic"),
        "%13",
    )

    assert calls == []


def test_main_routes_raw_yoloe_keyboard_to_visible_pane_five(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pane_ids = ["%0", "%1", "%2", "%3", "%4", "%5"]
    pane_calls: list[tuple[str, str, float]] = []
    subprocess_calls: list[list[str]] = []

    monkeypatch.setattr(launch_inference, "_check_prerequisites", lambda _config: None)
    monkeypatch.setattr(launch_inference, "_kill_existing_session", lambda: None)
    monkeypatch.setattr(launch_inference, "_create_tmux_session", lambda: pane_ids)
    monkeypatch.setattr(launch_inference, "_check_pane_alive", lambda _pane: True)
    monkeypatch.setattr(launch_inference, "_get_local_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(
        launch_inference,
        "_send_to_pane",
        lambda pane_id, command, wait=1.0: pane_calls.append(
            (pane_id, command, wait)
        ),
    )

    def record_run(command: list[str], **_kwargs: object) -> types.SimpleNamespace:
        subprocess_calls.append(command)
        return types.SimpleNamespace(returncode=1)

    monkeypatch.setattr(launch_inference.subprocess, "run", record_run)

    launch_inference.main(
        InferenceLaunchConfig(
            planner_input="base_pose",
            base_pose_mode="raw_yoloe_servo",
            data_exporter=False,
        )
    )

    keyboard_calls = [
        call
        for call in pane_calls
        if "keyboard_planner_thread_server.py" in call[1]
    ]
    assert keyboard_calls == [
        (
            "%5",
            launch_inference.build_base_pose_manual_keyboard_command(
                InferenceLaunchConfig(
                    planner_input="base_pose",
                    base_pose_mode="raw_yoloe_servo",
                    data_exporter=False,
                ),
                Path(launch_inference.__file__).resolve().parents[2],
            ),
            1.0,
        )
    ]
    assert not any(
        "base_keyboard" in argument
        for command in subprocess_calls
        for argument in command
    )
