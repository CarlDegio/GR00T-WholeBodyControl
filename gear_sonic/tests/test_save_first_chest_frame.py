import os
import subprocess
from importlib import import_module
from pathlib import Path

import cv2
import numpy as np
import pytest


def test_extract_chest_rgb_accepts_only_valid_uint8_rgb():
    chest_frame = import_module("gear_sonic.scripts.save_first_chest_frame")
    image = np.zeros((2, 3, 3), dtype=np.uint8)

    assert chest_frame.extract_chest_rgb({"images": {"chest_view": image}}) is image
    assert chest_frame.extract_chest_rgb({"images": {"ego_view": image}}) is None

    with pytest.raises(ValueError, match="HxWx3"):
        chest_frame.extract_chest_rgb(
            {"images": {"chest_view": np.zeros((2, 3), dtype=np.uint8)}}
        )
    with pytest.raises(ValueError, match="uint8"):
        chest_frame.extract_chest_rgb(
            {"images": {"chest_view": image.astype(np.float32)}}
        )


def test_save_rgb_png_preserves_rgb_channels(tmp_path: Path):
    chest_frame = import_module("gear_sonic.scripts.save_first_chest_frame")
    image = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
    output = tmp_path / "chest.png"

    chest_frame.save_rgb_png(image, output)

    decoded_bgr = cv2.imread(str(output), cv2.IMREAD_COLOR)
    decoded_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(decoded_rgb, image)

    with pytest.raises(FileExistsError, match="already exists"):
        chest_frame.save_rgb_png(image, output)


def test_capture_saves_first_available_chest_frame_and_closes_client(
    monkeypatch, tmp_path: Path
):
    chest_frame = import_module("gear_sonic.scripts.save_first_chest_frame")
    first_rgb = np.full((2, 3, 3), [10, 20, 30], dtype=np.uint8)

    class FakeClient:
        instance = None

        def __init__(self, server_ip, port, decode_images):
            assert (server_ip, port, decode_images) == ("robot", 5555, True)
            self.messages = [None, {"images": {"chest_view": first_rgb}}]
            self.closed = False
            self.__class__.instance = self

        def read(self, blocking=False):
            assert blocking is False
            return self.messages.pop(0)

        def close(self):
            self.closed = True

    monkeypatch.setattr(chest_frame, "ComposedCameraClientSensor", FakeClient)
    monkeypatch.setattr(chest_frame.time, "sleep", lambda _seconds: None)
    output = tmp_path / "first.png"
    ready = tmp_path / "subscriber.ready"

    chest_frame.capture_first_chest_frame(
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


def test_parse_args_accepts_remote_camera_server(monkeypatch, tmp_path: Path):
    chest_frame = import_module("gear_sonic.scripts.save_first_chest_frame")
    output = tmp_path / "remote_chest.png"
    monkeypatch.setattr(
        "sys.argv",
        [
            "save_first_chest_frame",
            "--camera-host",
            "192.168.123.164",
            "--camera-port",
            "6000",
            "--output-path",
            str(output),
            "--timeout-sec",
            "30",
            "--overwrite",
        ],
    )

    args = chest_frame.parse_args()

    assert args.camera_host == "192.168.123.164"
    assert args.camera_port == 6000
    assert args.output_path == output
    assert args.timeout_sec == 30.0
    assert args.overwrite is True


def test_capture_closes_client_when_ready_file_creation_fails(
    monkeypatch, tmp_path: Path
):
    chest_frame = import_module("gear_sonic.scripts.save_first_chest_frame")

    class FakeClient:
        instance = None

        def __init__(self, server_ip, port, decode_images):
            assert (server_ip, port, decode_images) == ("robot", 5555, True)
            self.closed = False
            self.__class__.instance = self

        def close(self):
            self.closed = True

    monkeypatch.setattr(chest_frame, "ComposedCameraClientSensor", FakeClient)

    def fail_touch(_path):
        raise PermissionError("read-only directory")

    monkeypatch.setattr(Path, "touch", fail_touch)
    ready = tmp_path / "subscriber.ready"
    with pytest.raises(PermissionError, match="read-only directory"):
        chest_frame.capture_first_chest_frame(
            camera_host="robot",
            camera_port=5555,
            output_path=tmp_path / "unused.png",
            timeout_sec=1.0,
            ready_file=ready,
        )

    assert FakeClient.instance.closed is True


def test_save_rgb_png_rejects_non_png_suffix(tmp_path: Path):
    chest_frame = import_module("gear_sonic.scripts.save_first_chest_frame")
    image = np.zeros((2, 3, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="\\.png"):
        chest_frame.save_rgb_png(image, tmp_path / "chest.jpg")


def test_local_launcher_forwards_remote_camera_arguments(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    launcher = repo_root / "save_first_chest_frame.sh"
    output = tmp_path / "remote_chest.png"
    env = os.environ.copy()
    env["FIRST_CHEST_FRAME_PYTHON"] = "/bin/echo"

    result = subprocess.run(
        [
            str(launcher),
            "192.168.123.164",
            str(output),
            "6000",
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout.strip().split() == [
        "-m",
        "gear_sonic.scripts.save_first_chest_frame",
        "--camera-host",
        "192.168.123.164",
        "--camera-port",
        "6000",
        "--output-path",
        str(output),
    ]
