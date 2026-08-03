"""Regression coverage for the LaViRA-to-REASAN velocity-command boundary."""

from __future__ import annotations

import sys

import pytest

from gear_sonic.scripts.lavira_planner import (
    VelocityCommand,
    build_reasan_velocity_message,
)
from gear_sonic.scripts.reasan_planner import decode_velocity_command, parse_args


def test_reasan_default_endpoints_preserve_lavira_input_and_planner_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REASAN must receive LaViRA on 5558 and publish filtered output on 5563."""
    monkeypatch.setattr(sys, "argv", ["reasan_planner.py"])

    args = parse_args()

    assert args.keyboard_endpoint == "tcp://127.0.0.1:5558"
    assert args.output_endpoint == "tcp://*:5563"


def test_lavira_messages_decode_for_turn_translation_and_stop() -> None:
    """Keep LaViRA output compatible with REASAN's unchanged command decoder."""
    rotation = decode_velocity_command(
        build_reasan_velocity_message(
            VelocityCommand(0.0, 0.0, 0.4, 1.0), action="turn_left"
        )
    )
    translation = decode_velocity_command(
        build_reasan_velocity_message(
            VelocityCommand(0.3, 0.0, 0.0, 1.0), action="move_forward"
        )
    )
    stop = decode_velocity_command(
        build_reasan_velocity_message(
            VelocityCommand(0.0, 0.0, 0.0, 1.0), action="stop"
        )
    )

    assert rotation["duration"] == pytest.approx(1.0)
    assert translation["duration"] == pytest.approx(1.0)
    assert stop["duration"] == pytest.approx(1.0)
    assert rotation["velocity"].tolist() == pytest.approx([0.0, 0.0, 0.4])
    assert translation["velocity"].tolist() == pytest.approx([0.3, 0.0, 0.0])
    assert stop["velocity"].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert abs(float(rotation["velocity"][0])) <= 1.0e-6
    assert abs(float(rotation["velocity"][1])) <= 1.0e-6
    assert abs(float(rotation["velocity"][2])) > 1.0e-6
    assert abs(float(translation["velocity"][0])) > 1.0e-6

    rotation_bypasses_filter = bool(
        abs(float(rotation["velocity"][0])) <= 1.0e-6
        and abs(float(rotation["velocity"][1])) <= 1.0e-6
        and abs(float(rotation["velocity"][2])) > 1.0e-6
    )
    translation_bypasses_filter = bool(
        abs(float(translation["velocity"][0])) <= 1.0e-6
        and abs(float(translation["velocity"][1])) <= 1.0e-6
        and abs(float(translation["velocity"][2])) > 1.0e-6
    )

    assert rotation_bypasses_filter
    assert not translation_bypasses_filter
