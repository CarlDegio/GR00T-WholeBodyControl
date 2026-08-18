from __future__ import annotations

import numpy as np

from gear_sonic.runtime.contracts import MessageMetadata, OperatorCommand
from gear_sonic.scripts.run_operator_cv_viewer import (
    ACTOR_RAY_STREAM,
    BASE_POSE_ACTIVE_COLOR,
    CHEST_RGB_STREAM,
    HEAD_RGB_STREAM,
    LEFT_WRIST_RGB_STREAM,
    RIGHT_WRIST_RGB_STREAM,
    SLAM_2D_STREAM,
    BasePoseViewerState,
    compose_visualization_canvas,
    gateway_frame_to_bgr,
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
    np.testing.assert_array_equal(canvas[230, 100], (101, 112, 123))
    np.testing.assert_array_equal(canvas[230, 300], (11, 22, 33))


def test_composer_leaves_missing_views_black() -> None:
    canvas = compose_visualization_canvas({}, width=400, height=300)

    assert not np.any(canvas)


def test_base_pose_status_selects_camera_and_tracks_safe_velocity() -> None:
    state = BasePoseViewerState()

    assert state.accept(
        _command("start_base_pose", {"generation": 4}, sequence=0), now=1.0
    )
    assert state.active
    assert state.state == "inference"
    assert state.accept(
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
        ),
        now=1.1,
    )

    assert state.camera_stream == HEAD_RGB_STREAM
    assert state.is_active_camera(HEAD_RGB_STREAM)
    assert not state.is_active_camera(CHEST_RGB_STREAM)
    assert state.velocity == (0.0, 0.0, 0.2)
    assert "turn_left" in state.status_text()

    assert state.accept(
        _command(
            "base_pose_runtime_status",
            {
                "generation": 4,
                "state": "reached",
                "reason": "aligned",
                "velocity": [0.0, 0.0, 0.0],
            },
            sequence=2,
        ),
        now=1.2,
    )
    assert not state.active
    assert state.action == "stop"
    assert state.reason == "aligned"


def test_composer_highlights_only_the_active_base_pose_camera() -> None:
    state = BasePoseViewerState(
        active=True,
        generation=2,
        state="motion",
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

    np.testing.assert_array_equal(canvas[151, 1], BASE_POSE_ACTIVE_COLOR)
    assert not np.array_equal(canvas[151, 201], BASE_POSE_ACTIVE_COLOR)


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
