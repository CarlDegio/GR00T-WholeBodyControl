"""Regression coverage for the LaViRA-to-REASAN velocity-command boundary."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from gear_sonic.scripts import lavira_sonic_relay as direct_relay
from gear_sonic.scripts.lavira_planner import (
    VelocityCommand,
    build_reasan_velocity_message,
)
from gear_sonic.scripts.lavira_sonic_relay import (
    LatestCommand,
    decode_velocity_command as decode_direct_velocity_command,
    parse_args as parse_direct_relay_args,
)
from gear_sonic.scripts.reasan_planner import (
    apply_rule_based_safety,
    raw_depth_requires_stop,
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
    assert args.camera_host == "127.0.0.1"
    assert args.camera_port == 5555
    assert args.depth_stop_distance == pytest.approx(0.30)
    assert args.depth_stop_min_area_pixels == 2000


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


@pytest.mark.parametrize(
    ("argv", "expected_limit"),
    [
        (["lavira_sonic_relay.py"], 0.40),
        (["lavira_sonic_relay.py", "--max-lateral-speed-m-s", "0.11"], 0.11),
    ],
)
def test_direct_relay_cli_exposes_configurable_lateral_limit(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    expected_limit: float,
) -> None:
    monkeypatch.setattr(sys, "argv", argv)

    args = parse_direct_relay_args()

    assert args.max_lateral_speed_m_s == pytest.approx(expected_limit)


def test_direct_relay_manual_source_is_optional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["lavira_sonic_relay.py"])

    args = parse_direct_relay_args()

    assert args.manual_source == ""


def _latest(velocity: list[float], *, received_at: float) -> LatestCommand:
    latest = LatestCommand()
    latest.update(
        {
            "velocity": np.asarray(velocity, dtype=np.float32),
            "duration": 2.0,
        },
        received_at,
    )
    return latest


def test_direct_relay_fresh_manual_command_overrides_automatic() -> None:
    automatic = _latest([0.3, 0.0, 0.0], received_at=10.0)
    manual = _latest([0.0, 0.0, 0.0], received_at=10.4)

    selected = direct_relay.select_velocity(
        automatic, manual, now=10.5, timeout=0.7
    )

    assert selected.tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_direct_relay_stale_manual_falls_back_to_automatic() -> None:
    automatic = _latest([0.3, 0.0, 0.0], received_at=10.4)
    manual = _latest([0.0, 0.0, 0.5], received_at=9.0)

    selected = direct_relay.select_velocity(
        automatic, manual, now=10.5, timeout=0.7
    )

    assert selected.tolist() == pytest.approx([0.3, 0.0, 0.0])


def test_direct_relay_decodes_lavira_command_without_reasan() -> None:
    decoded = decode_direct_velocity_command(
        build_reasan_velocity_message(
            VelocityCommand(0.2, -0.16, 0.2, 1.0), action="move"
        )
    )

    assert decoded["velocity"].tolist() == pytest.approx([0.2, -0.16, 0.2])
    assert decoded["duration"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("requested_vy", "expected_vy"),
    [(0.60, 0.40), (-0.60, -0.40)],
)
def test_direct_relay_caps_lateral_velocity_at_configured_default(
    requested_vy: float, expected_vy: float
) -> None:
    decoded = decode_direct_velocity_command(
        build_reasan_velocity_message(
            VelocityCommand(0.0, requested_vy, 0.0, 1.0), action="move"
        )
    )

    assert decoded["velocity"].tolist() == pytest.approx(
        [0.0, expected_vy, 0.0]
    )


def test_direct_relay_scales_both_linear_axes_to_preserve_ratio() -> None:
    decoded = decode_direct_velocity_command(
        build_reasan_velocity_message(
            VelocityCommand(0.30, 0.50, 0.50, 1.0), action="move"
        )
    )

    vx, vy, wz = map(float, decoded["velocity"])
    assert (vx, vy, wz) == pytest.approx((0.24, 0.40, 0.50))
    assert vx / vy == pytest.approx(0.30 / 0.50)
    assert np.hypot(vx, vy) < np.hypot(0.30, 0.50)


def test_direct_relay_accepts_configurable_lateral_limit() -> None:
    decoded = decode_direct_velocity_command(
        build_reasan_velocity_message(
            VelocityCommand(0.30, -0.20, 0.0, 1.0), action="move"
        ),
        max_lateral_speed_m_s=0.10,
    )

    assert decoded["velocity"].tolist() == pytest.approx([0.15, -0.10, 0.0])


def test_direct_relay_float32_output_never_exceeds_configured_lateral_limit() -> None:
    limit = 0.14
    decoded = decode_direct_velocity_command(
        build_reasan_velocity_message(
            VelocityCommand(0.30, 0.20, 0.0, 1.0), action="move"
        ),
        max_lateral_speed_m_s=limit,
    )

    vx, vy, _ = map(float, decoded["velocity"])
    assert vy <= limit
    assert vx / vy == pytest.approx(0.30 / 0.20)


def test_direct_relay_uses_most_restrictive_linear_scale_for_both_axes() -> None:
    decoded = decode_direct_velocity_command(
        build_reasan_velocity_message(
            VelocityCommand(2.0, 0.40, 0.0, 1.0), action="move"
        )
    )

    vx, vy, _ = map(float, decoded["velocity"])
    assert (vx, vy) == pytest.approx((1.00, 0.20))
    assert vx / vy == pytest.approx(2.0 / 0.40)


@pytest.mark.parametrize(
    ("requested_vx", "expected_velocity"),
    [(2.0, [1.0, 0.05, 0.0]), (-1.0, [-0.5, 0.05, 0.0])],
)
def test_direct_relay_scales_vy_when_vx_hits_transport_limit(
    requested_vx: float, expected_velocity: list[float]
) -> None:
    decoded = decode_direct_velocity_command(
        build_reasan_velocity_message(
            VelocityCommand(requested_vx, 0.10, 0.0, 1.0), action="move"
        )
    )

    assert decoded["velocity"].tolist() == pytest.approx(expected_velocity)


@pytest.mark.parametrize("limit", [0.0, -0.1, float("nan"), float("inf")])
def test_direct_relay_rejects_invalid_lateral_limit(limit: float) -> None:
    with pytest.raises(ValueError, match="lateral speed"):
        decode_direct_velocity_command(
            build_reasan_velocity_message(
                VelocityCommand(0.0, 0.1, 0.0, 1.0), action="move"
            ),
            max_lateral_speed_m_s=limit,
        )


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


def test_raw_depth_stops_when_connected_near_area_exceeds_two_thousand_pixels() -> None:
    depth_mm = np.full((50, 50), 1000, dtype=np.uint16)
    depth_mm.flat[:2001] = 299

    assert raw_depth_requires_stop(depth_mm, depth_scale_m=0.001)


def test_raw_depth_does_not_stop_at_exactly_two_thousand_or_for_invalid_pixels() -> None:
    depth_mm = np.full((50, 50), 1000, dtype=np.uint16)
    depth_mm.flat[:2000] = 299

    assert not raw_depth_requires_stop(depth_mm, depth_scale_m=0.001)

    depth_mm.flat[:2100] = 0
    assert not raw_depth_requires_stop(depth_mm, depth_scale_m=0.001)


def test_raw_depth_does_not_combine_scattered_near_pixels_across_the_image() -> None:
    depth_mm = np.full((100, 100), 1000, dtype=np.uint16)
    depth_mm[::2, ::2] = 299

    assert not raw_depth_requires_stop(depth_mm, depth_scale_m=0.001)


def test_camera_stop_overrides_motion_independently_of_actor_ray() -> None:
    ray = decode_actor_ray(actor_ray(angle_deg=90.0, distance_m=3.0))
    command = np.array([0.0, 0.15, 0.4], dtype=np.float32)

    safe = apply_rule_based_safety(command, ray, camera_stop=True)

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
