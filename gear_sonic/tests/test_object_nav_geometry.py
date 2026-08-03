"""Behavioral tests for pure ObjectNav RGB-D geometry."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gear_sonic.utils.inference.object_nav_geometry import (
    ObjectNavGeometryError,
    build_object_nav_commands_from_frames,
    measure_object_nav_target,
)


def navigate_policy(bbox_2d: list[float]) -> dict[str, object]:
    return {"action": "NAVIGATE", "bbox_2d": bbox_2d}


def depth_frame(depth_mm: float = 2000.0) -> np.ndarray:
    return np.full((101, 101), depth_mm, dtype=np.float32)


def test_measures_centered_target_without_lateral_offset() -> None:
    measurement = measure_object_nav_target(
        navigate_policy([450, 450, 550, 550]), depth_frame(), 100.0, 50.0
    )

    assert measurement["bbox_center_pixel"] == [50, 50]
    assert measurement["angle_deg"] == pytest.approx(0.0)
    assert measurement["goal_x"] == pytest.approx(2.0)
    assert measurement["goal_y"] == pytest.approx(0.0)
    assert measurement["range"] == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("bbox_2d", "expected_angle_sign", "expected_goal_y_sign"),
    [([50, 450, 150, 550], 1, 1), ([850, 450, 950, 550], -1, -1)],
)
def test_measures_target_direction_on_each_side_of_image(
    bbox_2d: list[float], expected_angle_sign: int, expected_goal_y_sign: int
) -> None:
    measurement = measure_object_nav_target(
        navigate_policy(bbox_2d), depth_frame(), 100.0, 50.0
    )

    assert math.copysign(1, measurement["angle_rad"]) == expected_angle_sign
    assert math.copysign(1, measurement["goal_y"]) == expected_goal_y_sign


def test_suppresses_rotation_below_two_degrees() -> None:
    frames = [(depth_frame(), 1000.0, 50.0)] * 5
    commands, geometry = build_object_nav_commands_from_frames(
        navigate_policy([480, 450, 500, 550]), frames
    )

    assert abs(geometry["angle_deg"]) < 2.0
    assert commands["commands"][0] == {
        "vx": 0.0,
        "vy": 0.0,
        "wz": 0.0,
        "duration": 0.0,
    }


def test_builds_commands_from_three_valid_measurements() -> None:
    frames = [
        (depth_frame(1000.0), 100.0, 50.0),
        (depth_frame(2000.0), 100.0, 50.0),
        (depth_frame(3000.0), 100.0, 50.0),
        (np.zeros((101, 101), dtype=np.float32), 100.0, 50.0),
        (np.zeros((101, 101), dtype=np.float32), 100.0, 50.0),
    ]

    commands, geometry = build_object_nav_commands_from_frames(
        navigate_policy([450, 450, 550, 550]), frames
    )

    assert geometry["valid_frame_count"] == 3
    assert [frame["status"] for frame in geometry["frame_measurements"]] == [
        "valid",
        "valid",
        "valid",
        "invalid",
        "invalid",
    ]
    assert geometry["mean_range"] == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("policy", "depth_mm", "fx", "cx", "message"),
    [
        ({"action": "STOP", "bbox_2d": None}, depth_frame(), 100.0, 50.0, "NAVIGATE"),
        (navigate_policy([500, 500, 500, 600]), depth_frame(), 100.0, 50.0, "ordering"),
        (navigate_policy([450, 450, 550, 550]), np.zeros((101, 101)), 100.0, 50.0, "valid aligned depth"),
        (navigate_policy([450, 450, 550, 550]), depth_frame(), 0.0, 50.0, "fx"),
    ],
)
def test_rejects_invalid_bbox_depth_or_intrinsics(
    policy: dict[str, object], depth_mm: np.ndarray, fx: float, cx: float, message: str
) -> None:
    with pytest.raises(ObjectNavGeometryError, match=message):
        measure_object_nav_target(policy, depth_mm, fx, cx)


def test_zero_target_standoff_preserves_full_direct_travel() -> None:
    frames = [(np.full((5, 5), 2000, dtype=np.float32), 100.0, 2.0)] * 5

    commands, geometry = build_object_nav_commands_from_frames(
        navigate_policy([450, 450, 550, 550]), frames
    )

    assert commands["commands"][0] == {
        "vx": 0.0,
        "vy": 0.0,
        "wz": 0.0,
        "duration": 0.0,
    }
    assert commands["commands"][1]["vx"] == 0.3
    assert commands["commands"][1]["duration"] == pytest.approx(
        geometry["travel"] / 0.3, abs=1e-3
    )
    assert geometry["travel"] == 2.0


def test_rejects_direct_travel_above_eight_metres() -> None:
    frames = [(depth_frame(8100.0), 100.0, 50.0)] * 5

    with pytest.raises(ObjectNavGeometryError, match="exceeds"):
        build_object_nav_commands_from_frames(
            navigate_policy([450, 450, 550, 550]), frames
        )
