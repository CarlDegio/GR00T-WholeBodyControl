import importlib.util
from pathlib import Path
import sys
import types

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


def test_oak_mjpeg_default_reaches_depthai_encoder_as_quality_95(monkeypatch):
    """Catch OAK MJPEG defaults that never reach the DepthAI encoder."""
    encoder_qualities = []

    class FakeOutput:
        def link(self, target):
            return None

        def createOutputQueue(self, **kwargs):
            return types.SimpleNamespace(tryGet=lambda: None)

    class FakeCamera:
        def build(self, socket):
            return self

        def requestOutput(self, *args, **kwargs):
            return FakeOutput()

    class FakeEncoder:
        input = object()
        out = FakeOutput()

        def setDefaultProfilePreset(self, fps, profile):
            return None

        def setQuality(self, quality):
            encoder_qualities.append(quality)

    class FakePipeline:
        def __init__(self, device):
            self.running = False

        def create(self, node):
            if node is fake_dai.node.Camera:
                return FakeCamera()
            return FakeEncoder()

        def start(self):
            self.running = True

        def isRunning(self):
            return self.running

        def stop(self):
            self.running = False

    class FakeDevice:
        @staticmethod
        def getAllAvailableDevices():
            return [object()]

        def __init__(self, *args, **kwargs):
            return None

        def getDeviceName(self):
            return "fake-oak"

        def getDeviceId(self):
            return "fake-id"

        def getConnectedCameras(self):
            return [fake_dai.CameraBoardSocket.CAM_A]

        def isPipelineRunning(self):
            return True

        def close(self):
            return None

    fake_dai = types.SimpleNamespace(
        Device=FakeDevice,
        Pipeline=FakePipeline,
        CameraBoardSocket=types.SimpleNamespace(CAM_A="CAM_A"),
        ImgFrame=types.SimpleNamespace(Type=types.SimpleNamespace(NV12="NV12")),
        VideoEncoderProperties=types.SimpleNamespace(
            Profile=types.SimpleNamespace(MJPEG="MJPEG")
        ),
        node=types.SimpleNamespace(Camera=object(), VideoEncoder=object()),
    )
    monkeypatch.setitem(sys.modules, "depthai", fake_dai)
    module_path = Path(__file__).parents[1] / "camera" / "drivers" / "oak.py"
    spec = importlib.util.spec_from_file_location("test_oak_driver", module_path)
    assert spec is not None and spec.loader is not None
    oak = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oak)
    monkeypatch.setattr(oak.time, "sleep", lambda _: None)

    config = oak.OAKConfig()
    config.use_mjpeg = True
    config.autofocus = True
    oak.OAKSensor(config=config)

    assert encoder_qualities == [95]
