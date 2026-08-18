"""RGB frame decoding, composition, and diagnostic card rendering."""

from __future__ import annotations

from datetime import datetime, timezone

import cv2
import numpy as np


class FrameError(ValueError):
    """An input image cannot be converted into a PICO video frame."""


def _validate_dimensions(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("frame dimensions must be positive")


def _validate_rgb(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise FrameError("RGB frame must be a NumPy array")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise FrameError("RGB frame must have uint8 HxWx3 shape")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise FrameError("RGB frame dimensions must be positive")


def decode_jpeg_rgb(jpeg: bytes) -> np.ndarray:
    """Decode one complete JPEG byte string into contiguous RGB uint8 pixels."""

    if not isinstance(jpeg, bytes | bytearray | memoryview) or not jpeg:
        raise FrameError("JPEG payload must be non-empty bytes")
    encoded = np.frombuffer(jpeg, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise FrameError("JPEG payload could not be decoded as a color image")
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    """Resize one RGB image to an exact width without changing its aspect ratio."""

    if width <= 0:
        raise ValueError("resized image width must be positive")
    source_height, source_width = image.shape[:2]
    scale = width / source_width
    height = max(1, round(source_height * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)


def compose_ego_with_wrist_views(
    ego_view: np.ndarray,
    left_wrist: np.ndarray,
    right_wrist: np.ndarray,
) -> np.ndarray:
    """Place two smaller wrist views side-by-side above the full-width ego view.

    Both wrist images are aspect-fitted to half of the ego-view width.  The
    right-wrist camera is rotated 180 degrees for its physical mounting.  If
    the wrist cameras have different aspect ratios, the shorter image is
    centered vertically in a black top row.  No source image is stretched or
    cropped.
    """

    _validate_rgb(ego_view)
    _validate_rgb(left_wrist)
    _validate_rgb(right_wrist)
    ego_width = ego_view.shape[1]
    if ego_width < 2:
        raise FrameError("ego-view width must be at least two pixels")

    left_width = ego_width // 2
    right_width = ego_width - left_width
    left_resized = _resize_to_width(left_wrist, left_width)
    right_upright = cv2.rotate(right_wrist, cv2.ROTATE_180)
    right_resized = _resize_to_width(right_upright, right_width)
    wrist_row_height = max(left_resized.shape[0], right_resized.shape[0])
    wrist_row = np.zeros((wrist_row_height, ego_width, 3), dtype=np.uint8)

    left_y = (wrist_row_height - left_resized.shape[0]) // 2
    right_y = (wrist_row_height - right_resized.shape[0]) // 2
    wrist_row[
        left_y : left_y + left_resized.shape[0],
        :left_width,
    ] = left_resized
    wrist_row[
        right_y : right_y + right_resized.shape[0],
        left_width:,
    ] = right_resized
    return np.ascontiguousarray(np.concatenate((wrist_row, ego_view), axis=0))


def compose_mono_sbs(image: np.ndarray, *, eye_width: int, height: int) -> np.ndarray:
    """Aspect-fit one RGB image and copy it into identical left/right eyes."""

    _validate_rgb(image)
    _validate_dimensions(eye_width, height)
    source_height, source_width = image.shape[:2]
    scale = min(eye_width / source_width, height / source_height)
    resized_width = max(1, min(eye_width, round(source_width * scale)))
    resized_height = max(1, min(height, round(source_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=interpolation,
    )
    eye = np.zeros((height, eye_width, 3), dtype=np.uint8)
    x = (eye_width - resized_width) // 2
    y = (height - resized_height) // 2
    eye[y : y + resized_height, x : x + resized_width] = resized
    return np.ascontiguousarray(np.concatenate((eye, eye.copy()), axis=1))


def _put_centered_text(
    image: np.ndarray,
    text: str,
    y: int,
    *,
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(4, (image.shape[1] - text_width) // 2)
    baseline_y = max(text_height + 4, min(image.shape[0] - 4, y))
    cv2.putText(
        image,
        text,
        (x, baseline_y),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def render_test_card(
    width: int,
    height: int,
    *,
    frame_number: int,
    wall_time: datetime | None = None,
) -> np.ndarray:
    """Render a deterministic, visibly animated RGB camera test source."""

    _validate_dimensions(width, height)
    if frame_number < 0:
        raise ValueError("frame_number cannot be negative")
    timestamp = wall_time or datetime.now(timezone.utc)
    image = np.full((height, width, 3), (18, 24, 36), dtype=np.uint8)

    grid_color = (55, 70, 90)
    for x in range(max(1, width // 8), width, max(1, width // 8)):
        cv2.line(image, (x, 0), (x, height - 1), grid_color, 1)
    for y in range(max(1, height // 6), height, max(1, height // 6)):
        cv2.line(image, (0, y), (width - 1, y), grid_color, 1)

    marker_size = max(16, min(width, height) // 12)
    cv2.rectangle(image, (0, 0), (marker_size, marker_size), (255, 0, 0), -1)
    cv2.rectangle(
        image,
        (width - marker_size - 1, 0),
        (width - 1, marker_size),
        (0, 255, 0),
        -1,
    )
    cv2.rectangle(
        image,
        (0, height - marker_size - 1),
        (marker_size, height - 1),
        (0, 0, 255),
        -1,
    )
    cv2.rectangle(
        image,
        (width - marker_size - 1, height - marker_size - 1),
        (width - 1, height - 1),
        (255, 255, 0),
        -1,
    )

    font_scale = max(0.4, min(width / 640.0, height / 480.0))
    _put_centered_text(
        image,
        "TOP",
        max(30, height // 9),
        scale=font_scale,
        color=(255, 255, 255),
        thickness=2,
    )
    _put_centered_text(
        image,
        f"FRAME {frame_number:06d}",
        height // 2,
        scale=font_scale * 1.4,
        color=(255, 220, 40),
        thickness=2,
    )
    _put_centered_text(
        image,
        timestamp.isoformat(timespec="milliseconds"),
        height // 2 + max(32, height // 10),
        scale=font_scale * 0.65,
        color=(220, 230, 255),
        thickness=1,
    )
    cv2.putText(
        image,
        "LEFT",
        (max(4, marker_size // 3), height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    right_text_width = cv2.getTextSize(
        "RIGHT", cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
    )[0][0]
    cv2.putText(
        image,
        "RIGHT",
        (max(4, width - right_text_width - marker_size // 3), height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    _put_centered_text(
        image,
        "BOTTOM",
        height - max(12, height // 24),
        scale=font_scale,
        color=(255, 255, 255),
        thickness=2,
    )
    return image


def render_status_card(
    width: int,
    height: int,
    status: str,
    *,
    wall_time: datetime | None = None,
) -> np.ndarray:
    """Render a fresh red RGB card that cannot be mistaken for live video."""

    _validate_dimensions(width, height)
    if not status.strip():
        raise ValueError("status cannot be empty")
    timestamp = wall_time or datetime.now(timezone.utc)
    image = np.full((height, width, 3), (170, 12, 12), dtype=np.uint8)
    stripe_step = max(24, width // 16)
    for x in range(-height, width, stripe_step):
        cv2.line(image, (x, 0), (x + height, height - 1), (110, 8, 8), 5)
    scale = max(0.5, min(width / 1280.0, height / 480.0))
    _put_centered_text(
        image,
        "VIDEO SOURCE ERROR",
        height // 3,
        scale=scale * 1.4,
        color=(255, 255, 255),
        thickness=3,
    )
    _put_centered_text(
        image,
        status.strip().upper(),
        height // 2,
        scale=scale,
        color=(255, 240, 80),
        thickness=2,
    )
    _put_centered_text(
        image,
        timestamp.isoformat(timespec="seconds"),
        height * 2 // 3,
        scale=scale * 0.75,
        color=(255, 255, 255),
        thickness=2,
    )
    return image
