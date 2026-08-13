import cv2
import numpy as np

from gear_sonic.camera.composed_camera import ComposedCameraConfig, ComposedCameraSensor
from gear_sonic.camera.sensor_server import ImageMessageSchema, ImageUtils
from gear_sonic.utils.mujoco_sim import sensor_server as mujoco_sensor_server


def test_schema_default_reaches_jpeg_encoder_as_quality_95(monkeypatch):
    """Catch a schema default that degrades host-encoded production RGB."""
    qualities = []

    def capture_quality(image, quality):
        qualities.append(quality)
        return "jpeg"

    monkeypatch.setattr(ImageUtils, "encode_image", capture_quality)

    ImageMessageSchema(
        timestamps={"ego_view": 1.0},
        images={"ego_view": np.zeros((8, 8, 3), np.uint8)},
    ).serialize()

    assert qualities == [95]


def test_composed_camera_default_reaches_jpeg_encoder_as_quality_95(monkeypatch):
    """Catch a composed-camera default that does not reach its JPEG encoder."""
    qualities = []

    def capture_quality(image, quality):
        qualities.append(quality)
        return "jpeg"

    monkeypatch.setattr(ImageUtils, "encode_image", capture_quality)
    composed = object.__new__(ComposedCameraSensor)
    composed.config = ComposedCameraConfig(server=False)
    composed._image_encoder_pool = None

    composed.serialize_message(
        {
            "ego_view": {
                "timestamps": {"ego_view": 1.0},
                "images": {"ego_view": np.zeros((8, 8, 3), np.uint8)},
                "camera_info": {},
            }
        }
    )

    assert qualities == [95]


def test_mujoco_encoder_passes_quality_95_to_opencv(monkeypatch):
    """Catch MuJoCo camera frames being encoded below the production quality."""
    imencode_calls = []

    def capture_imencode(extension, image, parameters):
        imencode_calls.append((extension, parameters))
        return True, np.array([0], dtype=np.uint8)

    monkeypatch.setattr(mujoco_sensor_server.cv2, "imencode", capture_imencode)

    mujoco_sensor_server.ImageUtils.encode_image(np.zeros((8, 8, 3), np.uint8))

    assert imencode_calls == [(".jpg", [int(cv2.IMWRITE_JPEG_QUALITY), 95])]
