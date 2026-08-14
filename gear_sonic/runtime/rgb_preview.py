"""Best-effort OpenCV preview for GROOT RGB data collection."""

from __future__ import annotations

import base64
import math
import os
import sys
import threading
from typing import Any, Mapping

import cv2
import numpy as np


WINDOW_NAME = "GROOT RGB Preview"
_LABEL_HEIGHT = 32
_GRID_COLUMNS = 2


def _decode_bgr(image: bytes | str | np.ndarray, *, encoded: bool) -> np.ndarray:
    if encoded:
        if isinstance(image, str):
            payload = base64.b64decode(image, validate=True)
        elif isinstance(image, bytes):
            payload = image
        else:
            raise ValueError("encoded RGB preview frames must be bytes or base64 strings")
        decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("OpenCV could not decode an RGB preview frame")
        return decoded

    values = np.asarray(image)
    if values.ndim != 3 or values.shape[-1] != 3 or values.dtype != np.uint8:
        raise ValueError(
            f"decoded RGB preview frames must be HxWx3 uint8, got {values.shape} {values.dtype}"
        )
    return cv2.cvtColor(values, cv2.COLOR_RGB2BGR)


def _labelled_tile(name: str, image_bgr: np.ndarray, tile_size: tuple[int, int]) -> np.ndarray:
    tile_width, tile_height = tile_size
    if tile_width <= 0 or tile_height <= 0:
        raise ValueError("RGB preview tile dimensions must be positive")

    source_height, source_width = image_bgr.shape[:2]
    scale = min(tile_width / source_width, tile_height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    resized = cv2.resize(
        image_bgr,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )

    tile = np.zeros((tile_height + _LABEL_HEIGHT, tile_width, 3), dtype=np.uint8)
    tile[:_LABEL_HEIGHT] = (35, 35, 35)
    x_offset = (tile_width - resized_width) // 2
    y_offset = _LABEL_HEIGHT + (tile_height - resized_height) // 2
    tile[y_offset : y_offset + resized_height, x_offset : x_offset + resized_width] = resized
    cv2.putText(
        tile,
        name,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return tile


def build_preview_frame(
    images: Mapping[str, bytes | str | np.ndarray],
    *,
    encoded: bool,
    tile_size: tuple[int, int] = (320, 240),
) -> np.ndarray:
    """Decode and tile camera images in insertion order into one BGR frame."""
    if not images:
        raise ValueError("RGB preview requires at least one camera image")

    tiles = [
        _labelled_tile(name, _decode_bgr(image, encoded=encoded), tile_size)
        for name, image in images.items()
    ]
    columns = min(_GRID_COLUMNS, len(tiles))
    rows = math.ceil(len(tiles) / columns)
    tile_height, tile_width = tiles[0].shape[:2]
    preview = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        y = row * tile_height
        x = column * tile_width
        preview[y : y + tile_height, x : x + tile_width] = tile
    return preview


def _copy_images(
    images: Mapping[str, bytes | str | np.ndarray],
) -> dict[str, bytes | str | np.ndarray]:
    return {
        name: image.copy() if isinstance(image, np.ndarray) else image
        for name, image in images.items()
    }


class RgbPreviewWorker:
    """Render only the newest camera set without blocking collection."""

    def __init__(
        self,
        *,
        encoded: bool,
        window_name: str = WINDOW_NAME,
        refresh_hz: float = 30.0,
        gui: Any = cv2,
        display_available: bool | None = None,
    ) -> None:
        if refresh_hz <= 0.0:
            raise ValueError("RGB preview refresh_hz must be positive")
        self.encoded = bool(encoded)
        self.window_name = window_name
        self.refresh_hz = float(refresh_hz)
        self._gui = gui
        self._display_available = (
            bool(display_available)
            if display_available is not None
            else not (
                gui is cv2
                and sys.platform.startswith("linux")
                and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
            )
        )
        self._condition = threading.Condition()
        self._pending_images: dict[str, bytes | str | np.ndarray] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False

    @property
    def is_active(self) -> bool:
        return bool(self._thread is not None and self._thread.is_alive())

    def publish(self, images: Mapping[str, bytes | str | np.ndarray]) -> None:
        if self._closed or self._stop.is_set():
            return
        copied = _copy_images(images)
        with self._condition:
            self._pending_images = copied
            self._condition.notify()

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("RGB preview worker is closed")
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="groot-rgb-preview",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        period_s = 1.0 / self.refresh_hz
        try:
            if not self._display_available:
                raise RuntimeError("no graphical display available")
            while not self._stop.is_set():
                with self._condition:
                    if self._pending_images is None:
                        self._condition.wait(period_s)
                    images = self._pending_images
                    self._pending_images = None
                if images is not None:
                    self._gui.imshow(
                        self.window_name,
                        build_preview_frame(images, encoded=self.encoded),
                    )
                if images is not None and self._gui.waitKey(1) & 0xFF == ord("q"):
                    self._stop.set()
        except Exception as exc:
            print(f"[DataExporter] RGB preview disabled: {exc}", flush=True)
            self._stop.set()
        finally:
            if self._display_available:
                try:
                    self._gui.destroyWindow(self.window_name)
                except Exception:
                    pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 2.0 / self.refresh_hz))
            if self._thread.is_alive():
                raise RuntimeError("RGB preview worker did not stop")
