from __future__ import annotations

import sys
import types

import pytest
from pathlib import Path


sys.modules.setdefault("tyro", types.ModuleType("tyro"))

from gear_sonic.scripts.launch_inference import (
    InferenceLaunchConfig,
    _parse_pane_ids,
    build_fastlio_command,
    build_livox_command,
    build_navdp_planner_command,
    build_navdp_server_command,
    run_readiness_gate,
)


def test_parse_pane_ids_returns_stable_ids_in_visual_index_order() -> None:
    output = "2 %8\n0 %3\n1 %5\n5 %13\n4 %11\n3 %9\n8 %19\n7 %17\n6 %15\n"

    assert _parse_pane_ids(output) == ["%3", "%5", "%8", "%9", "%11", "%13", "%15", "%17", "%19"]


def test_parse_pane_ids_rejects_incomplete_layout() -> None:
    with pytest.raises(RuntimeError, match="expected 9 tmux panes, found 8"):
        _parse_pane_ids("0 %1\n1 %2\n2 %3\n3 %4\n4 %5\n5 %6\n6 %7\n7 %8\n")


def test_navdp_stack_commands_use_ros_topics_and_unchanged_external_server() -> None:
    config = InferenceLaunchConfig()
    planner = build_navdp_planner_command(config, Path("/workspace/sonic"))
    server = build_navdp_server_command(config)
    livox = build_livox_command(config)
    fastlio = build_fastlio_command(config)

    assert "navdp_planner.py" in planner
    assert "source /opt/ros/humble/setup.bash" in planner
    assert "navdp_server.py" in server
    assert "--checkpoint /home/user/Downloads/navdp-cross-modal.ckpt" in server
    assert "msg_MID360_launch.py" in livox
    assert "mapping.launch.py" in fastlio
    assert "rviz:=false" in fastlio
    assert "mid360_reasan_open3d.py" not in " ".join((planner, server, livox, fastlio))


def test_launcher_runs_readiness_gate_synchronously(monkeypatch) -> None:
    commands: list[list[str]] = []

    class Result:
        returncode = 0

    def run(command, **_kwargs):
        commands.append(command)
        return Result()

    monkeypatch.setattr("gear_sonic.scripts.launch_inference.subprocess.run", run)
    config = InferenceLaunchConfig(lidar_ready_timeout=45.0)

    assert run_readiness_gate(config, Path("/workspace/sonic"), "lidar")
    shell = commands[0][-1]
    assert "navdp_readiness_gate.py --stage lidar" in shell
    assert "--timeout 45.0" in shell
