from __future__ import annotations

import numpy as np
import pytest

from gear_sonic.utils.planner_control import NavigationRuntimeStatus
from gear_sonic.runtime.protocol import MessageMetadata, OperatorCommand
from gear_sonic.utils.operator.cv_viewer import (
    ACTOR_RAY_STREAM,
    BASE_POSE_ACTIVE_COLOR,
    CHEST_RGB_STREAM,
    HEAD_RGB_STREAM,
    LEFT_WRIST_RGB_STREAM,
    RIGHT_WRIST_RGB_STREAM,
    SLAM_2D_STREAM,
    NavigationViewerState,
    compose_visualization_canvas,
    draw_base_pose_overlays,
    gateway_frame_to_bgr,
    navigation_status_text,
    parse_base_pose_viewer_overlay,
)


def _command(
    name: str,
    parameters: dict[str, object],
    *,
    sequence: int,
) -> OperatorCommand:
    return OperatorCommand(
        metadata=MessageMetadata(
            source="control_gateway_navigation",
            sequence=sequence,
            timestamp_ns=100 + sequence,
            ttl_ms=1000,
        ),
        command_id=f"viewer-{sequence}",
        name=name,
        parameters=parameters,
    )


def test_composer_places_four_panels_above_two_aligned_body_cameras() -> None:
    frames = {
        ACTOR_RAY_STREAM: np.full((124, 100, 3), (10, 20, 30), dtype=np.uint8),
        SLAM_2D_STREAM: np.full((124, 100, 3), (40, 50, 60), dtype=np.uint8),
        HEAD_RGB_STREAM: np.full((48, 64, 3), (101, 112, 123), dtype=np.uint8),
        CHEST_RGB_STREAM: np.full((48, 64, 3), (11, 22, 33), dtype=np.uint8),
        LEFT_WRIST_RGB_STREAM: np.full((48, 64, 3), (44, 55, 66), dtype=np.uint8),
        RIGHT_WRIST_RGB_STREAM: np.full((48, 64, 3), (77, 88, 99), dtype=np.uint8),
    }

    canvas = compose_visualization_canvas(frames, width=400, height=300)

    assert canvas.shape == (300, 400, 3)
    np.testing.assert_array_equal(canvas[80, 50], (10, 20, 30))
    np.testing.assert_array_equal(canvas[80, 150], (40, 50, 60))
    np.testing.assert_array_equal(canvas[80, 250], (44, 55, 66))
    np.testing.assert_array_equal(canvas[80, 350], (77, 88, 99))
    np.testing.assert_array_equal(canvas[220, 100], (101, 112, 123))
    np.testing.assert_array_equal(canvas[220, 300], (11, 22, 33))


def test_composer_leaves_missing_views_black() -> None:
    canvas = compose_visualization_canvas({}, width=400, height=300)

    assert not np.any(canvas)


def test_base_pose_status_selects_camera_and_clears_terminal_overlay() -> None:
    state = NavigationViewerState()

    assert state.accept_control(
        _command("start_base_pose", {"generation": 4}, sequence=0)
    )
    assert state.active
    assert state.accept_control(
        _command(
            "base_pose_runtime_status",
            {
                "generation": 4,
                "state": "motion",
                "camera_stream": "ego_view",
                "action": "turn_left",
                "velocity": [0.0, 0.0, 0.2],
            },
            sequence=1,
        )
    )

    assert state.camera_stream == HEAD_RGB_STREAM
    assert state.is_active_camera(HEAD_RGB_STREAM)
    assert not state.is_active_camera(CHEST_RGB_STREAM)

    assert state.accept_control(
        _command(
            "base_pose_runtime_status",
            {
                "generation": 4,
                "state": "reached",
                "reason": "aligned",
                "velocity": [0.0, 0.0, 0.0],
            },
            sequence=2,
        )
    )
    assert not state.active
    assert state.camera_stream is None


def test_base_pose_status_draws_target_desk_and_table_on_active_camera() -> None:
    state = NavigationViewerState()
    state.accept_control(
        _command("start_base_pose", {"generation": 4}, sequence=0)
    )
    overlay = {
        "target_bbox_xyxy": [10.0, 10.0, 30.0, 30.0],
        "target_lateral_anchor_px": [20.0, 20.0],
        "table_edge_endpoints_px": [[5.0, 35.0], [55.0, 35.0]],
        "desk_mask_row_spans": [[32, 4, 60], [33, 4, 60]],
        "image_size": [64, 48],
    }
    assert state.accept_control(
        _command(
            "base_pose_runtime_status",
            {
                "generation": 4,
                "state": "motion",
                "camera_stream": "ego_view",
                "viewer_overlay": overlay,
            },
            sequence=1,
        )
    )
    source = np.zeros((48, 64, 3), dtype=np.uint8)

    head = draw_base_pose_overlays(source.copy(), HEAD_RGB_STREAM, state)
    chest = draw_base_pose_overlays(source.copy(), CHEST_RGB_STREAM, state)

    assert np.any(head)
    assert not np.any(chest)
    np.testing.assert_array_equal(head[10, 10], (0, 255, 0))
    np.testing.assert_array_equal(head[35, 40], (0, 0, 255))
    assert state.desk_mask_row_spans == ((32, 4, 60), (33, 4, 60))

    state.accept_control(
        _command(
            "base_pose_runtime_status",
            {"generation": 4, "state": "reached"},
            sequence=2,
        )
    )
    assert state.target_bbox_xyxy is None


def test_base_pose_routes_target_and_table_to_their_source_cameras() -> None:
    state = NavigationViewerState()
    state.accept_control(
        _command("start_base_pose", {"generation": 4}, sequence=0)
    )
    assert state.accept_control(
        _command(
            "base_pose_runtime_status",
            {
                "generation": 4,
                "state": "motion",
                # The red mode border remains on the head-stage window.
                "camera_stream": "ego_view",
                "viewer_overlay": {
                    "target_camera_stream": "chest_view",
                    "table_camera_stream": "ego_view",
                    "target_bbox_xyxy": [10.0, 10.0, 30.0, 30.0],
                    "table_edge_endpoints_px": [
                        [5.0, 35.0],
                        [55.0, 35.0],
                    ],
                    "desk_mask_row_spans": [[32, 4, 60]],
                    "image_size": [64, 48],
                },
            },
            sequence=1,
        )
    )
    source = np.zeros((48, 64, 3), dtype=np.uint8)

    head = draw_base_pose_overlays(source.copy(), HEAD_RGB_STREAM, state)
    chest = draw_base_pose_overlays(source.copy(), CHEST_RGB_STREAM, state)

    assert state.is_active_camera(HEAD_RGB_STREAM)
    assert not state.is_active_camera(CHEST_RGB_STREAM)
    np.testing.assert_array_equal(head[35, 40], (0, 0, 255))
    assert not np.any(head[10, 10])
    np.testing.assert_array_equal(chest[10, 10], (0, 255, 0))
    assert not np.any(chest[35, 40])


@pytest.mark.parametrize(
    ("overlay", "expected_pixel"),
    (
        ({"target_bbox_xyxy": [10.0, 10.0, 30.0, 30.0]}, (10, 10)),
        (
            {"table_edge_endpoints_px": [[5.0, 35.0], [55.0, 35.0]]},
            (35, 40),
        ),
        (
            {
                "desk_mask_row_spans": [[32, 4, 60]],
                "image_size": [64, 48],
            },
            (32, 20),
        ),
    ),
)
def test_base_pose_draws_each_available_overlay_independently(
    overlay: dict[str, object],
    expected_pixel: tuple[int, int],
) -> None:
    state = NavigationViewerState()
    state.accept_control(
        _command("start_base_pose", {"generation": 4}, sequence=0)
    )
    assert state.accept_control(
        _command(
            "base_pose_runtime_status",
            {
                "generation": 4,
                "state": "motion",
                "camera_stream": "ego_view",
                "viewer_overlay": overlay,
            },
            sequence=1,
        )
    )

    rendered = draw_base_pose_overlays(
        np.zeros((48, 64, 3), dtype=np.uint8),
        HEAD_RGB_STREAM,
        state,
    )

    assert np.any(rendered)
    assert np.any(rendered[expected_pixel])


def test_base_pose_viewer_rejects_mask_span_outside_source_image() -> None:
    with pytest.raises(ValueError, match="outside the image"):
        parse_base_pose_viewer_overlay(
            {
                "viewer_overlay": {
                    "target_bbox_xyxy": [1.0, 1.0, 4.0, 3.0],
                    "desk_mask_row_spans": [[48, 0, 64]],
                    "image_size": [64, 48],
                }
            }
        )


def test_composer_highlights_only_the_active_base_pose_camera() -> None:
    state = NavigationViewerState(
        active=True,
        generation=2,
        camera_stream=HEAD_RGB_STREAM,
    )
    frames = {
        HEAD_RGB_STREAM: np.full((48, 64, 3), (1, 2, 3), dtype=np.uint8),
        CHEST_RGB_STREAM: np.full((48, 64, 3), (4, 5, 6), dtype=np.uint8),
    }

    canvas = compose_visualization_canvas(
        frames,
        width=400,
        height=300,
        base_pose=state,
    )

    np.testing.assert_array_equal(canvas[132, 1], BASE_POSE_ACTIVE_COLOR)
    assert not np.array_equal(canvas[132, 201], BASE_POSE_ACTIVE_COLOR)


def test_wasd_status_reports_key_and_final_safety_filtered_velocity() -> None:
    owner, text = navigation_status_text(
        NavigationRuntimeStatus(
            generation=2,
            timestamp=10.0,
            mode="manual_velocity",
            source="operator_console",
            requested_velocity=(0.3, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            reason="depth_hard_stop",
        )
    )

    assert owner == "wasd"
    assert "W" in text
    assert "REQ +0.30/+0.00/+0.00" in text
    assert "depth_hard_stop" in text


def test_navdp_runtime_status_uses_the_shared_velocity_formatter() -> None:
    owner, text = navigation_status_text(
        NavigationRuntimeStatus(
            generation=5,
            timestamp=10.0,
            mode="nav_goal",
            source="navdp",
            requested_velocity=(0.2, 0.0, -0.1),
            velocity=(0.2, 0.0, -0.1),
            reason="clear",
        )
    )

    assert owner == "navdp"
    assert "NAVDP G5" in text
    assert "OUT vx +0.20" in text


def test_non_right_wrist_camera_is_converted_to_bgr_without_rotation() -> None:
    rgb = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 0]],
        ],
        dtype=np.uint8,
    )

    bgr = gateway_frame_to_bgr(CHEST_RGB_STREAM, rgb)

    assert bgr is not None
    np.testing.assert_array_equal(
        bgr,
        np.array(
            [
                [[0, 0, 255], [0, 255, 0]],
                [[255, 0, 0], [0, 255, 255]],
            ],
            dtype=np.uint8,
        ),
    )


def test_right_wrist_camera_is_converted_to_bgr_and_rotated_180_degrees() -> None:
    rgb = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 0]],
        ],
        dtype=np.uint8,
    )

    bgr = gateway_frame_to_bgr(RIGHT_WRIST_RGB_STREAM, rgb)

    assert bgr is not None
    np.testing.assert_array_equal(
        bgr,
        np.array(
            [
                [[0, 255, 255], [255, 0, 0]],
                [[0, 255, 0], [0, 0, 255]],
            ],
            dtype=np.uint8,
        ),
    )
