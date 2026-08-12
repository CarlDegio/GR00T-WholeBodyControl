from __future__ import annotations

import numpy as np

from gear_sonic.scripts.run_operator_cv_viewer import (
    CHEST_RGB_STREAM,
    HEAD_RGB_STREAM,
    HEAD_RGBD_STREAM,
    LEFT_WRIST_RGB_STREAM,
    LINGBOT_STREAM,
    NAVIGATION_STREAM,
    RIGHT_WRIST_RGB_STREAM,
    compose_visualization_canvas,
    gateway_frame_to_bgr,
)


def test_composer_preserves_all_seven_views_in_one_canvas() -> None:
    frames = {
        NAVIGATION_STREAM: np.full((30, 90, 3), (10, 20, 30), dtype=np.uint8),
        HEAD_RGBD_STREAM: np.full((26, 64, 3), (40, 50, 60), dtype=np.uint8),
        LINGBOT_STREAM: np.full((26, 64, 3), (70, 80, 90), dtype=np.uint8),
        HEAD_RGB_STREAM: np.full((48, 64, 3), (101, 112, 123), dtype=np.uint8),
        CHEST_RGB_STREAM: np.full((48, 64, 3), (11, 22, 33), dtype=np.uint8),
        LEFT_WRIST_RGB_STREAM: np.full((48, 64, 3), (44, 55, 66), dtype=np.uint8),
        RIGHT_WRIST_RGB_STREAM: np.full((48, 64, 3), (77, 88, 99), dtype=np.uint8),
    }

    canvas = compose_visualization_canvas(frames, width=400, height=300)

    assert canvas.shape == (300, 400, 3)
    assert np.any(np.all(canvas == (10, 20, 30), axis=2))
    assert np.any(np.all(canvas == (40, 50, 60), axis=2))
    assert np.any(np.all(canvas == (70, 80, 90), axis=2))
    assert np.any(np.all(canvas == (101, 112, 123), axis=2))
    assert np.any(np.all(canvas == (11, 22, 33), axis=2))
    assert np.any(np.all(canvas == (44, 55, 66), axis=2))
    assert np.any(np.all(canvas == (77, 88, 99), axis=2))


def test_composer_leaves_missing_views_black() -> None:
    canvas = compose_visualization_canvas({}, width=400, height=300)

    assert not np.any(canvas)


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
