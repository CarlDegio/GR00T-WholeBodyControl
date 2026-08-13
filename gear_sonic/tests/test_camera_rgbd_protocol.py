import importlib
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gear_sonic.camera.sensor_server import ImageMessageSchema, ImageUtils


class _FakeFrame:
    def __init__(self, image):
        self.image = image

    def get_data(self):
        return self.image


class _FakeFrames:
    def __init__(self, color, depth):
        self.color = _FakeFrame(color)
        self.depth = _FakeFrame(depth)

    def get_color_frame(self):
        return self.color

    def get_depth_frame(self):
        return self.depth


def _fake_realsense_module():
    raw_frames = _FakeFrames(
        np.full((480, 640, 3), 10, dtype=np.uint8),
        np.full((480, 640), 100, dtype=np.uint16),
    )
    aligned_frames = _FakeFrames(
        np.full((480, 640, 3), 20, dtype=np.uint8),
        np.full((480, 640), 200, dtype=np.uint16),
    )

    class FakeDevice:
        def get_info(self, _info):
            return "fake-device"

        def first_depth_sensor(self):
            return types.SimpleNamespace(get_depth_scale=lambda: 0.001)

    class FakeProfile:
        def get_device(self):
            return FakeDevice()

        def get_stream(self, stream):
            assert stream == fake_rs.stream.color
            intrinsics = types.SimpleNamespace(
                fx=500.0, fy=501.0, ppx=320.0, ppy=240.0, width=640, height=480
            )
            return types.SimpleNamespace(
                as_video_stream_profile=lambda: types.SimpleNamespace(
                    get_intrinsics=lambda: intrinsics
                )
            )

    class FakePipeline:
        def start(self, _config):
            return FakeProfile()

        def wait_for_frames(self):
            return raw_frames

        def stop(self):
            pass

    class FakeConfig:
        def enable_device(self, _device_id):
            pass

        def enable_stream(self, *_args):
            pass

    class FakeAlign:
        instances = []

        def __init__(self, stream):
            self.stream = stream
            self.processed_frames = []
            self.__class__.instances.append(self)

        def process(self, frames):
            self.processed_frames.append(frames)
            return aligned_frames

    fake_rs = types.SimpleNamespace(
        context=lambda: types.SimpleNamespace(query_devices=lambda: [FakeDevice()]),
        pipeline=FakePipeline,
        config=FakeConfig,
        align=FakeAlign,
        stream=types.SimpleNamespace(color="color", depth="depth"),
        format=types.SimpleNamespace(rgb8="rgb8", z16="z16"),
        camera_info=types.SimpleNamespace(
            name="name", serial_number="serial_number", firmware_version="firmware_version"
        ),
    )
    return fake_rs, raw_frames, FakeAlign


def test_rgbd_schema_round_trip_preserves_uint16_and_camera_info():
    depth = np.array([[0, 1000], [2345, 65535]], dtype=np.uint16)
    schema = ImageMessageSchema(
        timestamps={"chest_view": 1.0, "chest_view_depth": 1.0},
        images={
            "chest_view": np.zeros((2, 2, 3), np.uint8),
            "chest_view_depth": depth,
        },
        camera_info={
            "chest_view": {
                "fx": 500.0,
                "fy": 501.0,
                "cx": 1.0,
                "cy": 1.0,
                "width": 2,
                "height": 2,
                "depth_scale_m": 0.001,
                "depth_aligned_to": "chest_view",
            }
        },
    )
    wire = schema.serialize()
    decoded = ImageMessageSchema.deserialize(wire)
    np.testing.assert_array_equal(decoded.images["chest_view_depth"], depth)
    assert decoded.images["chest_view_depth"].dtype == np.uint16
    assert decoded.camera_info == schema.camera_info
    assert wire["schema_version"] == 2


def test_schema_parallel_serialization_preserves_order_and_payload(monkeypatch):
    """Catch a fallback to serial encoding or a parallel wire-format change."""
    rgb_a = np.zeros((16, 24, 3), dtype=np.uint8)
    rgb_a[:, :8] = (240, 20, 10)
    rgb_b = np.full((16, 24, 3), (15, 120, 230), dtype=np.uint8)
    depth_a = np.arange(16 * 24, dtype=np.uint16).reshape(16, 24)
    depth_b = np.full((16, 24), 2345, dtype=np.uint16)
    schema = ImageMessageSchema(
        timestamps={
            "ego_view": 1.0,
            "ego_view_depth": 1.0,
            "chest_view": 2.0,
            "chest_view_depth": 2.0,
        },
        images={
            "ego_view": rgb_a,
            "ego_view_depth": depth_a,
            "chest_view": rgb_b,
            "chest_view_depth": depth_b,
        },
        camera_info={"chest_view": {"depth_scale_m": 0.001}},
    )
    serial_wire = schema.serialize()

    original_rgb_encoder = ImageUtils.encode_image
    original_depth_encoder = ImageUtils.encode_depth_image
    all_encoders_started = threading.Barrier(4)

    def encode_rgb_after_barrier(image, quality=80):
        all_encoders_started.wait(timeout=2.0)
        return original_rgb_encoder(image, quality=quality)

    def encode_depth_after_barrier(image):
        all_encoders_started.wait(timeout=2.0)
        return original_depth_encoder(image)

    monkeypatch.setattr(ImageUtils, "encode_image", encode_rgb_after_barrier)
    monkeypatch.setattr(ImageUtils, "encode_depth_image", encode_depth_after_barrier)

    with ThreadPoolExecutor(max_workers=4) as executor:
        parallel_wire = schema.serialize(executor=executor)

    assert list(parallel_wire["images"]) == list(schema.images)
    assert parallel_wire == serial_wire
    decoded = ImageMessageSchema.deserialize(parallel_wire)
    np.testing.assert_array_equal(decoded.images["ego_view_depth"], depth_a)
    np.testing.assert_array_equal(decoded.images["chest_view_depth"], depth_b)
    assert decoded.camera_info == schema.camera_info


def test_legacy_base64_jpeg_does_not_claim_policy_channel_compatibility() -> None:
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    image[:, :16] = (240, 20, 10)
    image[:, 16:32] = (15, 230, 25)
    image[:, 32:] = (5, 30, 220)

    wire = ImageMessageSchema(
        timestamps={"ego_view": 1.0},
        images={"ego_view": image},
    ).serialize()
    legacy_rgb = ImageMessageSchema.deserialize(wire).images["ego_view"]
    jpeg_bytes = __import__("base64").b64decode(wire["images"]["ego_view"])
    direct_bgr = __import__("cv2").imdecode(
        np.frombuffer(jpeg_bytes, dtype=np.uint8),
        __import__("cv2").IMREAD_COLOR,
    )
    direct_rgb = __import__("cv2").cvtColor(
        direct_bgr,
        __import__("cv2").COLOR_BGR2RGB,
    )

    assert not np.array_equal(direct_rgb, legacy_rgb)


def test_depth_encoder_rejects_wrong_dtype_and_shape():
    with pytest.raises(ValueError, match="2D uint16"):
        ImageUtils.encode_depth_image(np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="2D uint16"):
        ImageUtils.encode_depth_image(np.zeros((2, 2, 1), dtype=np.uint16))
    with pytest.raises(ValueError, match="2D uint16"):
        ImageUtils.encode_depth_image([[0, 1]])


def test_legacy_rgb_message_without_camera_info_still_decodes():
    decoded = ImageMessageSchema.deserialize({"timestamps": {}, "images": {}})
    assert decoded.camera_info == {}


def test_realsense_depth_uses_color_aligned_frames_and_publishes_calibration(monkeypatch):
    """Catch raw-depth publication or calibration missing from an RGB-D frame."""
    fake_rs, raw_frames, fake_align = _fake_realsense_module()
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake_rs)
    monkeypatch.delitem(sys.modules, "gear_sonic.camera.drivers.realsense", raising=False)
    realsense = importlib.import_module("gear_sonic.camera.drivers.realsense")

    config = realsense.RealSenseConfig()
    config.enable_depth = True
    sensor = realsense.RealSenseSensor(
        config=config, device_id="fake-device", mount_position="chest_view"
    )

    result = sensor.read()

    assert result is not None
    np.testing.assert_array_equal(
        result["images"]["chest_view_depth"], np.full((480, 640), 200, dtype=np.uint16)
    )
    assert fake_align.instances[0].stream == fake_rs.stream.color
    assert fake_align.instances[0].processed_frames == [raw_frames]
    assert result["camera_info"]["chest_view"] == {
        "fx": 500.0, "fy": 501.0, "cx": 320.0, "cy": 240.0,
        "width": 640, "height": 480,
        "depth_scale_m": 0.001,
        "depth_aligned_to": "chest_view",
    }


def test_composed_camera_enables_depth_only_for_chest_and_merges_camera_info(monkeypatch):
    """Catch depth enabled on non-chest cameras or metadata dropped by composition."""
    from gear_sonic.camera.composed_camera import ComposedCameraConfig, ComposedCameraSensor

    class FakeRealSenseConfig:
        created = []

        def __init__(self):
            self.fps = 30
            self.enable_depth = True
            self.__class__.created.append(self)

    class FakeRealSenseSensor:
        def __init__(self, **kwargs):
            self.config = kwargs["config"]

    fake_driver = types.ModuleType("gear_sonic.camera.drivers.realsense")
    fake_driver.RealSenseConfig = FakeRealSenseConfig
    fake_driver.RealSenseSensor = FakeRealSenseSensor
    monkeypatch.setitem(sys.modules, "gear_sonic.camera.drivers.realsense", fake_driver)

    composed = object.__new__(ComposedCameraSensor)
    composed.config = ComposedCameraConfig(realsense_enable_depth=True)
    composed._instantiate_camera("ego_view", "realsense")
    composed._instantiate_camera("chest_view", "realsense")

    assert [config.enable_depth for config in FakeRealSenseConfig.created] == [False, True]

    result = ImageMessageSchema.deserialize(
        composed.serialize_message(
            {
                "ego_view": {
                    "timestamps": {"ego_view": 1.0},
                    "images": {"ego_view": np.zeros((1, 1, 3), dtype=np.uint8)},
                    "camera_info": {},
                },
                "chest_view": {
                    "timestamps": {"chest_view": 1.0, "chest_view_depth": 1.0},
                    "images": {
                        "chest_view": np.zeros((1, 1, 3), dtype=np.uint8),
                        "chest_view_depth": np.ones((1, 1), dtype=np.uint16),
                    },
                    "camera_info": {"chest_view": {"fx": 500.0}},
                },
            }
        )
    ).asdict()
    assert result["camera_info"] == {"chest_view": {"fx": 500.0}}


def test_composed_camera_serializes_images_with_owned_executor(monkeypatch):
    """Catch composed serialization bypassing its parallel encoder pool."""
    from gear_sonic.camera.composed_camera import ComposedCameraConfig, ComposedCameraSensor

    original_rgb_encoder = ImageUtils.encode_image
    original_depth_encoder = ImageUtils.encode_depth_image
    all_encoders_started = threading.Barrier(4)

    def encode_rgb_after_barrier(image, quality=80):
        all_encoders_started.wait(timeout=2.0)
        return original_rgb_encoder(image, quality=quality)

    def encode_depth_after_barrier(image):
        all_encoders_started.wait(timeout=2.0)
        return original_depth_encoder(image)

    monkeypatch.setattr(ImageUtils, "encode_image", encode_rgb_after_barrier)
    monkeypatch.setattr(ImageUtils, "encode_depth_image", encode_depth_after_barrier)

    composed = object.__new__(ComposedCameraSensor)
    composed.config = ComposedCameraConfig(server=False)
    composed._image_encoder_pool = ThreadPoolExecutor(max_workers=4)
    message = {
        "ego_view": {
            "timestamps": {"ego_view": 1.0, "ego_view_depth": 1.0},
            "images": {
                "ego_view": np.zeros((8, 8, 3), dtype=np.uint8),
                "ego_view_depth": np.ones((8, 8), dtype=np.uint16),
            },
            "camera_info": {},
        },
        "chest_view": {
            "timestamps": {"chest_view": 2.0, "chest_view_depth": 2.0},
            "images": {
                "chest_view": np.zeros((8, 8, 3), dtype=np.uint8),
                "chest_view_depth": np.full((8, 8), 2, dtype=np.uint16),
            },
            "camera_info": {},
        },
    }
    try:
        wire = composed.serialize_message(message)
    finally:
        composed._image_encoder_pool.shutdown(wait=True, cancel_futures=True)

    assert list(wire["images"]) == [
        "ego_view",
        "ego_view_depth",
        "chest_view",
        "chest_view_depth",
    ]


def test_composed_camera_applies_software_jpeg_quality_to_rgb_payloads():
    """Catch a composed-camera quality option that never reaches OpenCV."""
    from gear_sonic.camera.composed_camera import ComposedCameraConfig, ComposedCameraSensor

    image = np.random.default_rng(0).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    message = {
        "ego_view": {
            "timestamps": {"ego_view": 1.0},
            "images": {"ego_view": image},
            "camera_info": {},
        }
    }

    payloads = {}
    for quality in (80, 95):
        composed = object.__new__(ComposedCameraSensor)
        composed.config = ComposedCameraConfig(jpeg_quality=quality, server=False)
        wire = composed.serialize_message(message)
        decoded = ImageMessageSchema.deserialize(wire).images["ego_view"]
        assert decoded.shape == image.shape
        payloads[quality] = wire["images"]["ego_view"]

    assert len(payloads[95]) > len(payloads[80])


@pytest.mark.parametrize("quality", [0, 101])
def test_composed_camera_rejects_invalid_software_jpeg_quality(quality):
    """Catch an invalid JPEG quality reaching OpenCV without validation."""
    from gear_sonic.camera.composed_camera import ComposedCameraConfig

    with pytest.raises(ValueError, match="jpeg_quality must be between 1 and 100"):
        ComposedCameraConfig(jpeg_quality=quality)


def test_composed_camera_close_shuts_down_encoder_pool():
    """Catch encoder threads surviving after the camera server closes."""
    from gear_sonic.camera.composed_camera import ComposedCameraConfig, ComposedCameraSensor

    composed = object.__new__(ComposedCameraSensor)
    composed.config = ComposedCameraConfig(ego_view_camera=None, server=False)
    composed.camera_queues = {}
    composed.camera_threads = {}
    composed.shutdown_events = {}
    composed._image_encoder_pool = ThreadPoolExecutor(max_workers=1)

    try:
        composed.close()
        with pytest.raises(RuntimeError, match="cannot schedule new futures"):
            composed._image_encoder_pool.submit(lambda: None)
    finally:
        composed._image_encoder_pool.shutdown(wait=True, cancel_futures=True)
