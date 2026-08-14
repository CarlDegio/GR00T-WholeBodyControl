from __future__ import annotations

from datetime import datetime, timezone

import cv2
import numpy as np
import pytest

from gear_sonic.pico_video.frames import (
    FrameError,
    compose_mono_sbs,
    decode_jpeg_rgb,
    render_status_card,
    render_test_card,
)


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
