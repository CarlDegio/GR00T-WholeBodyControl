import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gear_sonic.camera.sensor_server import ImageMessageSchema


def _fake_orbbec_module():
    fake_ob = types.ModuleType("pyorbbecsdk")
    fake_ob.OBSensorType = types.SimpleNamespace(
        COLOR_SENSOR="color_sensor", DEPTH_SENSOR="depth_sensor"
    )
    fake_ob.OBFormat = types.SimpleNamespace(RGB="rgb", Y16="y16")
    fake_ob.OBStreamType = types.SimpleNamespace(COLOR_STREAM="color_stream")
    fake_ob.OBFrameAggregateOutputMode = types.SimpleNamespace(
        FULL_FRAME_REQUIRE="full_frame_require"
    )

    class FakeDeviceInfo:
        def __init__(self, serial_number):
            self._serial_number = serial_number

        def get_name(self):
            return "Orbbec Gemini 345Lg"

        def get_serial_number(self):
            return self._serial_number

        def get_firmware_version(self):
            return "1.2.3"

        def get_vid(self):
            return 0x2BC5

        def get_pid(self):
            return 0x0813

    class FakeDevice:
        def __init__(self, serial_number):
            self.serial_number = serial_number

        def get_device_info(self):
            return FakeDeviceInfo(self.serial_number)

    class FakeDeviceList:
        serials = ["serial-b", "serial-a"]

        def get_count(self):
            return len(self.serials)

        def get_device_serial_number_by_index(self, index):
            reported_serials = getattr(fake_ob, "reported_serials", None)
            if reported_serials is not None:
                return reported_serials[index]
            return self.serials[index]

        def get_device_by_index(self, index):
            return FakeDevice(self.serials[index])

        def get_device_by_serial_number(self, serial_number):
            return FakeDevice(serial_number)

    class FakeContext:
        def query_devices(self):
            return FakeDeviceList()

    class FakeVideoProfile:
        def __init__(self, sensor_type, width, height, image_format, fps):
            self.sensor_type = sensor_type
            self.width = width
            self.height = height
            self.image_format = image_format
            self.fps = fps

        def get_intrinsic(self):
            assert self.sensor_type == fake_ob.OBSensorType.COLOR_SENSOR
            return types.SimpleNamespace(
                fx=501.0,
                fy=502.0,
                cx=1.0,
                cy=0.5,
                width=self.width,
                height=self.height,
            )
        def get_distortion(self):
            return types.SimpleNamespace(
                model="OBCameraDistortionModel.BROWN_CONRADY_K6",
                k1=0.2,
                k2=-0.09,
                k3=-0.004,
                k4=0.59,
                k5=-0.1,
                k6=-0.02,
                p1=-0.0003,
                p2=0.0001,
            )


    class FakeProfileList:
        def __init__(self, sensor_type):
            self.sensor_type = sensor_type

        def get_video_stream_profile(self, width, height, image_format, fps):
            expected_format = (
                fake_ob.OBFormat.RGB
                if self.sensor_type == fake_ob.OBSensorType.COLOR_SENSOR
                else fake_ob.OBFormat.Y16
            )
            assert image_format == expected_format
            return FakeVideoProfile(
                self.sensor_type, width, height, image_format, fps
            )

    class FakeConfig:
        instances = []

        def __init__(self):
            self.enabled_profiles = []
            self.aggregate_mode = None
            self.__class__.instances.append(self)

        def enable_stream(self, profile):
            self.enabled_profiles.append(profile)

        def set_frame_aggregate_output_mode(self, mode):
            self.aggregate_mode = mode

    class FakeColorFrame:
        def __init__(self, image):
            self.image = np.ascontiguousarray(image)

        def get_width(self):
            return self.image.shape[1]

        def get_height(self):
            return self.image.shape[0]

        def get_data(self):
            return self.image

        def get_format(self):
            return fake_ob.OBFormat.RGB
        def get_stream_profile(self):
            return FakeVideoProfile(
                fake_ob.OBSensorType.COLOR_SENSOR,
                self.image.shape[1],
                self.image.shape[0],
                fake_ob.OBFormat.RGB,
                30,
            )


    class FakeDepthFrame:
        def __init__(self, image, depth_scale_mm):
            self.image = np.ascontiguousarray(image)
            self.depth_scale_mm = depth_scale_mm

        def get_width(self):
            return self.image.shape[1]

        def get_height(self):
            return self.image.shape[0]

        def get_data(self):
            return self.image

        def get_depth_scale(self):
            return self.depth_scale_mm

    class FakeFrames:
        def __init__(self, color, depth):
            self.color = FakeColorFrame(color)
            self.depth = FakeDepthFrame(depth, depth_scale_mm=1.0)

        def get_color_frame(self):
            return self.color

        def get_depth_frame(self):
            return self.depth

    raw_frames = FakeFrames(
        np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8),
        np.array([[100, 101]], dtype=np.uint16),
    )
    aligned_frames = FakeFrames(
        np.array([[[11, 22, 33], [44, 55, 66]]], dtype=np.uint8),
        np.array([[200, 201]], dtype=np.uint16),
    )

    class FakePipeline:
        instances = []

        def __init__(self, device):
            self.device = device
            self.started_config = None
            self.stopped = False
            self.__class__.instances.append(self)

        def get_stream_profile_list(self, sensor_type):
            return FakeProfileList(sensor_type)

        def start(self, config):
            self.started_config = config

        def wait_for_frames(self, timeout_ms):
            assert timeout_ms == 1000
            return raw_frames

        def stop(self):
            self.stopped = True

    class FakeAlignResult:
        def as_frame_set(self):
            return aligned_frames

    class FakeAlignFilter:
        instances = []

        def __init__(self, align_to_stream):
            self.align_to_stream = align_to_stream
            self.processed_frames = []
            self.__class__.instances.append(self)

        def process(self, frames):
            self.processed_frames.append(frames)
            return FakeAlignResult()

    fake_ob.Context = FakeContext
    fake_ob.Config = FakeConfig
    fake_ob.Pipeline = FakePipeline
    fake_ob.AlignFilter = FakeAlignFilter
    return fake_ob, FakePipeline, FakeConfig, FakeAlignFilter, raw_frames


def _import_orbbec_driver(monkeypatch):
    fake = _fake_orbbec_module()
    monkeypatch.setitem(sys.modules, "pyorbbecsdk", fake[0])
    monkeypatch.delitem(
        sys.modules, "gear_sonic.camera.drivers.orbbec", raising=False
    )
    return importlib.import_module("gear_sonic.camera.drivers.orbbec"), fake


def test_orbbec_read_matches_realsense_rgbd_contract(monkeypatch):
    """Catch BGR output, raw depth publication, or incompatible calibration."""
    orbbec, (_, fake_pipeline, fake_config, fake_align, raw_frames) = (
        _import_orbbec_driver(monkeypatch)
    )
    monkeypatch.setattr(orbbec.time, "time", lambda: 123.5)
    config = orbbec.OrbbecConfig()
    config.color_image_dim = (2, 1)
    config.depth_image_dim = (2, 1)
    config.fps = 30
    config.enable_depth = True

    sensor = orbbec.OrbbecSensor(
        config=config, device_id="serial-b", mount_position="ego_view"
    )
    result = sensor.read()

    assert sensor.serial_number == "serial-b"
    assert fake_pipeline.instances[-1].device.serial_number == "serial-b"
    assert fake_config.instances[-1].aggregate_mode == "full_frame_require"
    assert fake_align.instances[-1].align_to_stream == "color_stream"
    assert fake_align.instances[-1].processed_frames == [raw_frames]
    assert result is not None
    assert set(result) == {"timestamps", "images", "camera_info"}
    assert result["timestamps"] == {
        "ego_view": 123.5,
        "ego_view_depth": 123.5,
    }
    np.testing.assert_array_equal(
        result["images"]["ego_view"],
        np.array([[[11, 22, 33], [44, 55, 66]]], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        result["images"]["ego_view_depth"],
        np.array([[200, 201]], dtype=np.uint16),
    )
    assert result["images"]["ego_view"].dtype == np.uint8
    assert result["images"]["ego_view_depth"].dtype == np.uint16
    info = result["camera_info"]["ego_view"]
    assert info["fx"] == 501.0
    assert info["fy"] == 502.0
    assert info["cx"] == 1.0
    assert info["cy"] == 0.5
    assert (info["width"], info["height"]) == (2, 1)
    assert info["depth_scale_m"] == 0.001
    assert info["depth_aligned_to"] == "ego_view"
    assert info["distortion_model"] == "OBCameraDistortionModel.BROWN_CONRADY_K6"
    assert info["distortion_coeffs"] == [0.2, -0.09, -0.0003, 0.0001, -0.004, 0.59, -0.1, -0.02]
    assert info["distortion_coeff_order"] == ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]
    assert info["source_distortion_model"] == info["distortion_model"]
    assert info["source_distortion_coeffs"] == info["distortion_coeffs"]
    assert info["rgb_undistorted"] is False
    assert info["camera_serial"] == "serial-b"
    assert info["color_image_dim"] == [2, 1]
    assert info["depth_image_dim"] == [2, 1]
    assert info["fps"] == 30

    decoded = ImageMessageSchema.deserialize(sensor.serialize(result))
    np.testing.assert_array_equal(
        decoded.images["ego_view_depth"], result["images"]["ego_view_depth"]
    )
    assert decoded.camera_info == result["camera_info"]

    sensor.close()
    assert fake_pipeline.instances[-1].stopped is True


def test_orbbec_device_index_is_sorted_by_serial(monkeypatch):
    """Catch USB enumeration order changing the selected camera."""
    orbbec, (_, fake_pipeline, _, _, _) = _import_orbbec_driver(monkeypatch)
    config = orbbec.OrbbecConfig()
    config.color_image_dim = (2, 1)
    config.enable_depth = False

    sensor = orbbec.OrbbecSensor(config=config, id=0)

    assert sensor.serial_number == "serial-a"
    assert fake_pipeline.instances[-1].device.serial_number == "serial-a"
    sensor.close()


def test_orbbec_serial_selection_falls_back_to_opened_device_info(monkeypatch):
    """Catch SDK device-list serial caches returning empty strings."""
    orbbec, (fake_ob, fake_pipeline, _, _, _) = _import_orbbec_driver(monkeypatch)
    fake_ob.reported_serials = ["", ""]
    config = orbbec.OrbbecConfig()
    config.color_image_dim = (2, 1)
    config.enable_depth = False

    sensor = orbbec.OrbbecSensor(config=config, device_id="serial-b")

    assert sensor.serial_number == "serial-b"
    assert fake_pipeline.instances[-1].device.serial_number == "serial-b"
    sensor.close()


def test_orbbec_unknown_serial_lists_available_devices(monkeypatch):
    """Catch silently falling back to the wrong camera for a bad serial."""
    orbbec, _ = _import_orbbec_driver(monkeypatch)

    with pytest.raises(
        ValueError,
        match=r"Orbbec device with serial 'missing' not found.*serial-a.*serial-b",
    ):
        orbbec.OrbbecSensor(device_id="missing")
