from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from gear_sonic.camera.calibration import (
    CameraCalibrationError,
    load_camera_intrinsics,
    persist_calibration_capture,
)
from gear_sonic.scripts.capture_camera_calibration import (
    capture_camera_calibration,
)


def complete_message() -> dict[str, object]:
    ego_rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    ego_rgb[..., 0] = 255
    chest_rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    chest_rgb[..., 1] = 255
    return {
        "timestamps": {
            "ego_view": 12.0,
            "ego_view_depth": 12.0,
            "chest_view": 12.5,
            "chest_view_depth": 12.5,
        },
        "images": {
            "ego_view": ego_rgb,
            "ego_view_depth": np.full((3, 4), 900, dtype=np.uint16),
            "chest_view": chest_rgb,
            "chest_view_depth": np.full((3, 4), 1100, dtype=np.uint16),
        },
        "camera_info": {
            "ego_view": {
                "fx": 500.0,
                "fy": 501.0,
                "cx": 1.5,
                "cy": 1.0,
                "width": 4,
                "height": 3,
                "distortion_model": "distortion.brown_conrady",
                "distortion_coeffs": [0.1, -0.2, 0.01, -0.02, 0.03],
                "distortion_coeff_order": ["k1", "k2", "p1", "p2", "k3"],
                "source_distortion_model": "distortion.brown_conrady",
                "source_distortion_coeffs": [0.1, -0.2, 0.01, -0.02, 0.03],
                "rgb_undistorted": True,
                "camera_type": "orbbec",
                "camera_serial": "head-serial",
                "color_image_dim": [4, 3],
                "depth_image_dim": [4, 3],
                "fps": 30,
                "depth_scale_m": 0.001,
                "depth_aligned_to": "ego_view",
            },
            "chest_view": {
                "fx": 510.0,
                "fy": 511.0,
                "cx": 1.4,
                "cy": 1.1,
                "width": 4,
                "height": 3,
                "distortion_model": "distortion.none",
                "distortion_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
                "distortion_coeff_order": ["k1", "k2", "p1", "p2", "k3"],
                "source_distortion_model": "distortion.none",
                "source_distortion_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
                "rgb_undistorted": False,
                "camera_type": "realsense",
                "camera_serial": "chest-serial",
                "color_image_dim": [4, 3],
                "depth_image_dim": [4, 3],
                "fps": 30,
                "depth_scale_m": 0.001,
                "depth_aligned_to": "chest_view",
            },
        },
    }


def test_persist_dual_camera_calibration_writes_active_backup_and_images(
    tmp_path: Path,
) -> None:
    active_path = tmp_path / "config" / "camera_intrinsics.json"
    backup_root = tmp_path / "outputs"

    backup_dir = persist_calibration_capture(
        complete_message(),
        active_path=active_path,
        backup_root=backup_root,
        capture_id="20260816_170000",
        robot_host="192.168.123.164",
    )

    assert backup_dir == backup_root / "20260816_170000"
    active = json.loads(active_path.read_text())
    backup = json.loads((backup_dir / "camera_intrinsics.json").read_text())
    assert active == backup
    assert active["format_version"] == 1
    assert active["robot_host"] == "192.168.123.164"
    assert active["streams"]["ego_view"]["camera_serial"] == "head-serial"
    assert active["streams"]["chest_view"]["camera_serial"] == "chest-serial"
    assert active["streams"]["ego_view"]["distortion_coeffs"] == [
        0.1,
        -0.2,
        0.01,
        -0.02,
        0.03,
    ]

    assert cv2.imread(str(backup_dir / "ego_view_rgb.png")).shape == (3, 4, 3)
    assert cv2.imread(
        str(backup_dir / "ego_view_depth_raw.png"), cv2.IMREAD_UNCHANGED
    ).dtype == np.uint16
    assert cv2.imread(str(backup_dir / "chest_view_rgb.png")).shape == (3, 4, 3)
    assert cv2.imread(
        str(backup_dir / "chest_view_depth_raw.png"), cv2.IMREAD_UNCHANGED
    ).dtype == np.uint16

    loaded = load_camera_intrinsics(active_path)
    assert loaded["ego_view"].fx == 500.0
    assert loaded["chest_view"].depth_aligned_to == "chest_view"
    assert loaded["ego_view"].distortion_coeffs == (
        0.1,
        -0.2,
        0.01,
        -0.02,
        0.03,
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value["images"].pop("chest_view_depth"), "chest_view_depth"),
        (
            lambda value: value["images"].__setitem__(
                "ego_view_depth", np.zeros((3, 4), dtype=np.float32)
            ),
            "uint16",
        ),
        (
            lambda value: value["camera_info"]["ego_view"].__setitem__(
                "depth_aligned_to", "chest_view"
            ),
            "aligned",
        ),
        (
            lambda value: value["camera_info"]["chest_view"].pop(
                "distortion_coeffs"
            ),
            "distortion_coeffs",
        ),
    ],
)
def test_invalid_capture_writes_nothing(
    tmp_path: Path, mutation, match: str
) -> None:
    message = complete_message()
    mutation(message)
    active_path = tmp_path / "camera_intrinsics.json"
    backup_root = tmp_path / "outputs"

    with pytest.raises(CameraCalibrationError, match=match):
        persist_calibration_capture(
            message,
            active_path=active_path,
            backup_root=backup_root,
            capture_id="invalid",
            robot_host="192.168.123.164",
        )

    assert not active_path.exists()
    assert not backup_root.exists()


def test_capture_backup_is_never_overwritten(tmp_path: Path) -> None:
    kwargs = {
        "active_path": tmp_path / "camera_intrinsics.json",
        "backup_root": tmp_path / "outputs",
        "capture_id": "same",
        "robot_host": "192.168.123.164",
    }
    persist_calibration_capture(complete_message(), **kwargs)

    with pytest.raises(FileExistsError, match="already exists"):
        persist_calibration_capture(complete_message(), **kwargs)



def test_one_shot_capture_connects_first_and_closes_client(tmp_path: Path) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.messages = [{"images": {}}, complete_message()]
            self.closed = False

        def read(self, blocking: bool = False):
            assert blocking is False
            return self.messages.pop(0)

        def close(self) -> None:
            self.closed = True

    client = FakeClient()
    ready_file = tmp_path / "subscriber_ready"

    backup_dir = capture_camera_calibration(
        camera_host="192.168.123.164",
        camera_port=5555,
        active_path=tmp_path / "active.json",
        backup_root=tmp_path / "outputs",
        timeout_sec=1.0,
        ready_file=ready_file,
        capture_id="one-shot",
        client_factory=lambda **_kwargs: client,
    )

    assert ready_file.exists()
    assert client.closed is True
    assert backup_dir == tmp_path / "outputs" / "one-shot"


def test_one_shot_capture_times_out_and_closes_client(tmp_path: Path) -> None:
    class EmptyClient:
        closed = False

        @staticmethod
        def read(blocking: bool = False):
            return None

        def close(self) -> None:
            self.closed = True

    client = EmptyClient()
    with pytest.raises(TimeoutError, match="complete dual-camera calibration"):
        capture_camera_calibration(
            camera_host="192.168.123.164",
            camera_port=5555,
            active_path=tmp_path / "active.json",
            backup_root=tmp_path / "outputs",
            timeout_sec=0.01,
            capture_id="timeout",
            client_factory=lambda **_kwargs: client,
        )
    assert client.closed is True
