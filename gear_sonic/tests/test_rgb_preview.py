from __future__ import annotations

import base64
import threading
import time

import cv2
import numpy as np

from gear_sonic.runtime.gateway.rgb_preview import RgbPreviewWorker, build_preview_frame


class _FakeGui:
    def __init__(self, *, key: int = -1, display_error: Exception | None = None) -> None:
        self.key = key
        self.display_error = display_error
        self.frames: list[np.ndarray] = []
        self.destroyed: list[str] = []
        self.shown = threading.Event()

    def imshow(self, _window_name: str, frame: np.ndarray) -> None:
        if self.display_error is not None:
            raise self.display_error
        self.frames.append(frame.copy())
        self.shown.set()

    def waitKey(self, _delay_ms: int) -> int:
        return self.key

    def destroyWindow(self, window_name: str) -> None:
        self.destroyed.append(window_name)


def _jpeg_bytes(bgr: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", bgr)
    assert ok
    return encoded.tobytes()


def _wait_until_stopped(worker: RgbPreviewWorker) -> None:
    deadline = time.monotonic() + 1.0
    while worker.is_active and time.monotonic() < deadline:
        time.sleep(0.005)
    assert not worker.is_active


def test_build_preview_decodes_encoded_bytes_and_base64_into_one_grid() -> None:
    blue_bgr = np.full((40, 80, 3), (255, 0, 0), dtype=np.uint8)
    green_bgr = np.full((80, 40, 3), (0, 255, 0), dtype=np.uint8)

    preview = build_preview_frame(
        {
            "ego_view": _jpeg_bytes(blue_bgr),
            "left_wrist": base64.b64encode(_jpeg_bytes(green_bgr)).decode("ascii"),
        },
        encoded=True,
        tile_size=(160, 120),
    )

    assert preview.shape == (152, 320, 3)
    assert preview.dtype == np.uint8
    np.testing.assert_allclose(preview[92, 80], (255, 0, 0), atol=8)
    np.testing.assert_allclose(preview[92, 240], (0, 255, 0), atol=8)


def test_build_preview_converts_decoded_rgb_and_tiles_four_cameras() -> None:
    rgb = np.full((30, 50, 3), (255, 0, 0), dtype=np.uint8)

    preview = build_preview_frame(
        {name: rgb for name in ("ego_view", "left_wrist", "right_wrist", "chest_view")},
        encoded=False,
        tile_size=(100, 60),
    )

    assert preview.shape == (184, 200, 3)
    np.testing.assert_array_equal(preview[62, 50], (0, 0, 255))


def test_worker_displays_only_latest_frame_and_q_closes_preview() -> None:
    gui = _FakeGui(key=ord("q"))
    worker = RgbPreviewWorker(encoded=False, gui=gui, refresh_hz=120.0)
    first = np.full((20, 20, 3), (255, 0, 0), dtype=np.uint8)
    latest = np.full((20, 20, 3), (0, 255, 0), dtype=np.uint8)
    worker.publish({"ego_view": first})
    worker.publish({"ego_view": latest})

    worker.start()
    assert gui.shown.wait(1.0)
    _wait_until_stopped(worker)
    worker.close()

    assert len(gui.frames) == 1
    np.testing.assert_array_equal(gui.frames[0][152, 160], (0, 255, 0))
    assert gui.destroyed == ["GROOT RGB Preview"]


def test_worker_disables_preview_when_opencv_display_fails(capsys) -> None:
    gui = _FakeGui(display_error=RuntimeError("no display"))
    worker = RgbPreviewWorker(encoded=False, gui=gui, refresh_hz=120.0)
    worker.publish({"ego_view": np.zeros((20, 20, 3), dtype=np.uint8)})

    worker.start()
    _wait_until_stopped(worker)
    worker.publish({"ego_view": np.ones((20, 20, 3), dtype=np.uint8)})
    worker.close()

    assert "RGB preview disabled: no display" in capsys.readouterr().out
    assert gui.destroyed == ["GROOT RGB Preview"]


def test_worker_never_calls_opencv_gui_when_no_display_is_available(capsys) -> None:
    gui = _FakeGui()
    worker = RgbPreviewWorker(
        encoded=False,
        gui=gui,
        display_available=False,
        refresh_hz=120.0,
    )
    worker.publish({"ego_view": np.zeros((20, 20, 3), dtype=np.uint8)})

    worker.start()
    _wait_until_stopped(worker)
    worker.close()

    assert not gui.frames
    assert "RGB preview disabled: no graphical display available" in capsys.readouterr().out
