from __future__ import annotations

from datetime import datetime, timezone

import cv2
import numpy as np
import pytest

from gear_sonic.utils.pico_video.frames import (
    FrameError,
    compose_ego_with_wrist_views,
    compose_mono_sbs,
    decode_jpeg_rgb,
    render_status_card,
    render_test_card,
)


def test_wrist_views_are_aspect_preserved_side_by_side_above_ego_view() -> None:
    ego = np.full((48, 64, 3), (255, 0, 0), dtype=np.uint8)
    left = np.full((48, 64, 3), (0, 255, 0), dtype=np.uint8)
    right = np.full((48, 64, 3), (0, 0, 255), dtype=np.uint8)

    output = compose_ego_with_wrist_views(ego, left, right)

    assert output.shape == (72, 64, 3)
    assert np.all(output[:24, :32] == (0, 255, 0))
    assert np.all(output[:24, 32:] == (0, 0, 255))
    assert np.all(output[24:] == (255, 0, 0))


def test_wrist_row_centers_views_with_different_aspect_ratios() -> None:
    ego = np.full((20, 40, 3), 30, dtype=np.uint8)
    left = np.full((20, 20, 3), 100, dtype=np.uint8)
    right = np.full((10, 20, 3), 200, dtype=np.uint8)

    output = compose_ego_with_wrist_views(ego, left, right)

    assert output.shape == (40, 40, 3)
    assert np.all(output[:20, :20] == 100)
    assert np.all(output[:5, 20:] == 0)
    assert np.all(output[5:15, 20:] == 200)
    assert np.all(output[15:20, 20:] == 0)
    assert np.all(output[20:] == 30)


def test_right_wrist_view_is_rotated_180_degrees_for_pico() -> None:
    ego = np.full((2, 4, 3), 10, dtype=np.uint8)
    left = np.full((2, 2, 3), 20, dtype=np.uint8)
    right = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)

    output = compose_ego_with_wrist_views(ego, left, right)

    np.testing.assert_array_equal(output[:2, :2], left)
    np.testing.assert_array_equal(output[:2, 2:], right[::-1, ::-1])
    np.testing.assert_array_equal(output[2:], ego)


def test_mono_sbs_has_identical_left_and_right_eyes() -> None:
    source = np.zeros((24, 32, 3), dtype=np.uint8)
    source[:, :16] = (255, 0, 0)

    output = compose_mono_sbs(source, eye_width=64, height=48)

    assert output.shape == (48, 128, 3)
    np.testing.assert_array_equal(output[:, :64], output[:, 64:])
    assert output[:, :64, 0].mean() > output[:, :64, 2].mean()


def test_mono_sbs_letterboxes_without_stretching_portrait_input() -> None:
    source = np.full((40, 20, 3), 255, dtype=np.uint8)

    output = compose_mono_sbs(source, eye_width=80, height=40)
    left = output[:, :80]

    assert np.all(left[:, :29] == 0)
    assert np.all(left[:, 30:50] == 255)
    assert np.all(left[:, 51:] == 0)


def test_jpeg_decoder_returns_rgb_pixels() -> None:
    bgr = np.zeros((8, 8, 3), dtype=np.uint8)
    bgr[:] = (0, 0, 255)
    ok, encoded = cv2.imencode(".jpg", bgr)
    assert ok

    rgb = decode_jpeg_rgb(encoded.tobytes())

    assert rgb.shape == (8, 8, 3)
    assert rgb[..., 0].mean() > 240
    assert rgb[..., 2].mean() < 15


def test_jpeg_decoder_rejects_invalid_bytes() -> None:
    with pytest.raises(FrameError, match="JPEG"):
        decode_jpeg_rgb(b"not-a-jpeg")


def test_test_card_changes_with_frame_number_and_marks_orientation() -> None:
    now = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)

    first = render_test_card(640, 480, frame_number=7, wall_time=now)
    second = render_test_card(640, 480, frame_number=8, wall_time=now)

    assert first.shape == (480, 640, 3)
    assert first.dtype == np.uint8
    assert not np.array_equal(first, second)
    assert tuple(first[12, 12]) == (255, 0, 0)
    assert tuple(first[12, -13]) == (0, 255, 0)
    assert tuple(first[-13, 12]) == (0, 0, 255)
    assert tuple(first[-13, -13]) == (255, 255, 0)


def test_status_card_is_visibly_red_and_contains_no_old_frame() -> None:
    card = render_status_card(
        1280,
        480,
        "SENSORGATEWAY STALE",
        wall_time=datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc),
    )

    assert card.shape == (480, 1280, 3)
    assert card.dtype == np.uint8
    assert card[..., 0].mean() > card[..., 1].mean() * 1.5
    assert card[..., 0].mean() > card[..., 2].mean() * 1.5


@pytest.mark.parametrize(
    ("width", "height"),
    [(0, 480), (640, 0), (-1, 480), (640, -1)],
)
def test_generated_cards_reject_nonpositive_dimensions(width: int, height: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        render_test_card(width, height, frame_number=0)
