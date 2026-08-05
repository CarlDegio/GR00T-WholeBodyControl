from pathlib import Path

import cv2
import numpy as np
import pytest

from gear_sonic.scripts import save_first_ego_frame
from gear_sonic.scripts.save_first_ego_frame import (
    capture_first_ego_frame,
    extract_ego_rgb,
    save_rgb_png,
)


def test_extract_ego_rgb_accepts_only_valid_uint8_rgb():
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    assert extract_ego_rgb({"images": {"ego_view": image}}) is image
    assert extract_ego_rgb({"images": {"chest_view": image}}) is None

    with pytest.raises(ValueError, match="HxWx3"):
        extract_ego_rgb({"images": {"ego_view": np.zeros((2, 3), dtype=np.uint8)}})
    with pytest.raises(ValueError, match="uint8"):
        extract_ego_rgb({"images": {"ego_view": image.astype(np.float32)}})


def test_save_rgb_png_preserves_rgb_channels(tmp_path: Path):
    image = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
    output = tmp_path / "ego.png"

    save_rgb_png(image, output)

    decoded_bgr = cv2.imread(str(output), cv2.IMREAD_COLOR)
    decoded_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(decoded_rgb, image)

    with pytest.raises(FileExistsError, match="already exists"):
        save_rgb_png(image, output)


def test_capture_saves_first_available_ego_frame_and_closes_client(
    monkeypatch, tmp_path: Path
):
    first_rgb = np.full((2, 3, 3), [10, 20, 30], dtype=np.uint8)

    class FakeClient:
        instance = None

        def __init__(self, server_ip, port, decode_images):
            assert (server_ip, port, decode_images) == ("robot", 5555, True)
            self.messages = [None, {"images": {"ego_view": first_rgb}}]
            self.closed = False
            self.__class__.instance = self

        def read(self, blocking=False):
            assert blocking is False
            return self.messages.pop(0)

        def close(self):
            self.closed = True

    monkeypatch.setattr(save_first_ego_frame, "ComposedCameraClientSensor", FakeClient)
    monkeypatch.setattr(save_first_ego_frame.time, "sleep", lambda _seconds: None)
    output = tmp_path / "first.png"
    ready = tmp_path / "subscriber.ready"

    capture_first_ego_frame(
        camera_host="robot",
        camera_port=5555,
        output_path=output,
        timeout_sec=1.0,
        ready_file=ready,
    )

    assert ready.is_file()
    assert output.is_file()
    assert FakeClient.instance.closed is True
    decoded_rgb = cv2.cvtColor(cv2.imread(str(output)), cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(decoded_rgb, first_rgb)
