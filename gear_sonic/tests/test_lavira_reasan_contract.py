"""Regression coverage for the LaViRA-to-REASAN velocity-command boundary."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from gear_sonic.scripts.lavira_planner import (
    VelocityCommand,
    build_reasan_velocity_message,
)
from gear_sonic.scripts.lavira_sonic_relay import (
    LatestCommand,
    decode_velocity_command as decode_direct_velocity_command,
)
from gear_sonic.scripts.reasan_planner import (
    apply_rule_based_safety,
    decode_actor_ray,
    decode_velocity_command,
    parse_args,
)
from tools.mid360_reasan_open3d import (
    build_actor_ray_message,
    parser as mid360_parser,
)


def test_reasan_default_endpoints_preserve_lavira_input_and_planner_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Safety guard must receive LaViRA on 5558 and publish on 5563."""
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


def test_direct_relay_decodes_lavira_command_without_reasan() -> None:
    decoded = decode_direct_velocity_command(
        build_reasan_velocity_message(
            VelocityCommand(0.3, -0.1, 0.2, 1.0), action="move"
        )
    )

    assert decoded["velocity"].tolist() == pytest.approx([0.3, -0.1, 0.2])
    assert decoded["duration"] == pytest.approx(1.0)


def test_direct_relay_zeros_stale_command() -> None:
    latest = LatestCommand()
    latest.update(
        {"velocity": np.array([0.3, 0.0, 0.0], dtype=np.float32), "duration": 2.0},
        now=10.0,
    )

    assert latest.velocity(10.5, timeout=0.7).tolist() == pytest.approx([0.3, 0.0, 0.0])
    assert latest.velocity(10.8, timeout=0.7).tolist() == pytest.approx([0.0, 0.0, 0.0])


def actor_ray(*, angle_deg: float | None = None, distance_m: float = 3.0) -> str:
    rays = np.ones(180, dtype=np.float32)
    if angle_deg is not None:
        index = int(round((angle_deg + 179.0) / 2.0))
        rays[index] = distance_m / 3.0
    return json.dumps({
        "type": "reasan_actor_ray",
        "version": 1,
        "sequence": 1,
        "source": "direct_median",
        "angle_min_deg": -179.0,
        "angle_increment_deg": 2.0,
        "range_max_m": 3.0,
        "normalized": rays.tolist(),
    })


def test_actor_ray_decoder_does_not_require_imu_for_static_rule_guard() -> None:
    decoded = decode_actor_ray(actor_ray(angle_deg=0.0, distance_m=0.4))

    assert decoded["distances"][90] == pytest.approx(0.4)
    assert decoded["angles_deg"][90] == pytest.approx(1.0)


def test_rule_guard_stops_forward_command_for_obstacle_in_front_90_degree_sector() -> None:
    ray = decode_actor_ray(actor_ray(angle_deg=1.0, distance_m=0.49))
    command = np.array([0.3, 0.1, 0.4], dtype=np.float32)

    safe = apply_rule_based_safety(command, ray)

    assert safe.tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_rule_guard_passes_forward_command_when_obstacle_is_outside_front_sector() -> None:
    ray = decode_actor_ray(actor_ray(angle_deg=47.0, distance_m=0.3))
    command = np.array([0.3, 0.1, 0.4], dtype=np.float32)

    safe = apply_rule_based_safety(command, ray)

    assert safe.tolist() == pytest.approx(command.tolist())


@pytest.mark.parametrize(
    "command",
    [
        [0.0, 0.0, 0.8],
        [-0.3, 0.0, 0.0],
        [0.0, 0.15, 0.0],
    ],
)
def test_rule_guard_never_checks_turning_backward_or_lateral_motion(
    command: list[float],
) -> None:
    ray = decode_actor_ray(actor_ray(angle_deg=1.0, distance_m=0.2))

    safe = apply_rule_based_safety(np.asarray(command, dtype=np.float32), ray)

    assert safe.tolist() == pytest.approx(command)


def test_mid360_cli_has_no_estimator_or_ray_source_modes() -> None:
    destinations = {action.dest for action in mid360_parser()._actions}

    assert "estimator" not in destinations
    assert "ray_source" not in destinations
    assert "imu_topic" not in destinations


def test_mid360_actor_ray_message_is_direct_and_contains_no_model_features() -> None:
    message = build_actor_ray_message(
        np.ones(180, dtype=np.float32), sequence=7, max_range=3.0
    )

    assert message["source"] == "direct_median"
    assert "imu_valid" not in message
    assert "projected_gravity" not in message
    assert "angular_velocity" not in message
