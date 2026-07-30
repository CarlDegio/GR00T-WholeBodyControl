"""Contract tests for lossless depth in the composed camera protocol."""

from __future__ import annotations

import base64
import importlib
import sys
import types
import unittest
from unittest import mock

import cv2
import numpy as np


if importlib.util.find_spec("msgpack") is None:
    sys.modules["msgpack"] = types.SimpleNamespace(
        packb=lambda value, **_kwargs: value,
        unpackb=lambda value, **_kwargs: value,
    )
if importlib.util.find_spec("msgpack_numpy") is None:
    sys.modules["msgpack_numpy"] = types.SimpleNamespace(
        decode=lambda value: value
    )
if importlib.util.find_spec("zmq") is None:
    sys.modules["zmq"] = types.SimpleNamespace(Again=RuntimeError)


class CameraRGBDProtocolTests(unittest.TestCase):
    def _schema_type(self):
        from gear_sonic.camera.sensor_server import ImageMessageSchema

        return ImageMessageSchema

    def test_depth_round_trips_losslessly_as_uint16_png(self):
        schema_type = self._schema_type()
        color_rgb = np.zeros((2, 3, 3), dtype=np.uint8)
        color_rgb[:, :, 0] = 255
        depth_raw = np.array(
            [[0, 1, 1000], [2000, 4096, 65535]],
            dtype=np.uint16,
        )
        camera_info = {
            "chest_view": {
                "fx": 610.0,
                "fy": 611.0,
                "cx": 1.0,
                "cy": 0.5,
                "width": 3,
                "height": 2,
                "depth_scale_m": 0.001,
                "depth_aligned_to": "chest_view",
            }
        }
        serialized = schema_type(
            timestamps={
                "chest_view": 100.0,
                "chest_view_depth": 100.0,
            },
            images={
                "chest_view": color_rgb,
                "chest_view_depth": depth_raw,
            },
            camera_info=camera_info,
        ).serialize()

        self.assertEqual(serialized["schema_version"], 2)
        self.assertEqual(serialized["camera_info"], camera_info)
        depth_png = base64.b64decode(
            serialized["images"]["chest_view_depth"]
        )
        decoded_png = cv2.imdecode(
            np.frombuffer(depth_png, dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
        np.testing.assert_array_equal(decoded_png, depth_raw)

        restored = schema_type.deserialize(serialized).asdict()
        np.testing.assert_array_equal(
            restored["images"]["chest_view_depth"],
            depth_raw,
        )
        self.assertEqual(restored["camera_info"], camera_info)

    def test_depth_rejects_non_uint16_input(self):
        schema_type = self._schema_type()

        with self.assertRaisesRegex(ValueError, "uint16"):
            schema_type(
                timestamps={"chest_view_depth": 100.0},
                images={
                    "chest_view_depth": np.ones(
                        (2, 3),
                        dtype=np.uint8,
                    )
                },
            ).serialize()

    def test_realsense_read_uses_aligned_frames_and_emits_calibration(self):
        fake_realsense = types.SimpleNamespace()
        with mock.patch.dict(
            sys.modules,
            {"pyrealsense2": fake_realsense},
        ):
            module = importlib.import_module(
                "gear_sonic.camera.drivers.realsense"
            )

        color_rgb = np.zeros((2, 3, 3), dtype=np.uint8)
        depth_raw = np.array(
            [[1000, 2000, 3000], [4000, 0, 5000]],
            dtype=np.uint16,
        )
        intrinsics = types.SimpleNamespace(
            fx=610.0,
            fy=611.0,
            ppx=1.0,
            ppy=0.5,
            width=3,
            height=2,
        )
        color_profile = types.SimpleNamespace(
            as_video_stream_profile=lambda: types.SimpleNamespace(
                intrinsics=intrinsics
            )
        )
        color_frame = types.SimpleNamespace(
            get_data=lambda: color_rgb,
            profile=color_profile,
        )
        depth_frame = types.SimpleNamespace(get_data=lambda: depth_raw)
        aligned_frames = types.SimpleNamespace(
            get_color_frame=lambda: color_frame,
            get_depth_frame=lambda: depth_frame,
        )
        raw_frames = object()
        aligner = mock.Mock()
        aligner.process.return_value = aligned_frames
        pipeline = mock.Mock()
        pipeline.wait_for_frames.return_value = raw_frames

        sensor = module.RealSenseSensor.__new__(module.RealSenseSensor)
        sensor.pipeline = pipeline
        sensor._align = aligner
        sensor._depth_scale_m = 0.001
        sensor._realsense_config = types.SimpleNamespace(enable_depth=True)
        sensor.mount_position = "chest_view"

        result = sensor.read()

        aligner.process.assert_called_once_with(raw_frames)
        np.testing.assert_array_equal(result["images"]["chest_view"], color_rgb)
        np.testing.assert_array_equal(
            result["images"]["chest_view_depth"],
            depth_raw,
        )
        self.assertEqual(
            result["camera_info"]["chest_view"],
            {
                "fx": 610.0,
                "fy": 611.0,
                "cx": 1.0,
                "cy": 0.5,
                "width": 3,
                "height": 2,
                "depth_scale_m": 0.001,
                "depth_aligned_to": "chest_view",
            },
        )

    def test_composed_message_preserves_camera_info(self):
        schema_type = self._schema_type()
        from gear_sonic.camera.composed_camera import ComposedCameraSensor

        camera_info = {
            "chest_view": {
                "fx": 610.0,
                "fy": 611.0,
                "cx": 1.0,
                "cy": 0.5,
                "width": 3,
                "height": 2,
                "depth_scale_m": 0.001,
                "depth_aligned_to": "chest_view",
            }
        }
        message = {
            "chest_view": {
                "timestamps": {
                    "chest_view": 100.0,
                    "chest_view_depth": 100.0,
                },
                "images": {
                    "chest_view": np.zeros((2, 3, 3), dtype=np.uint8),
                    "chest_view_depth": np.ones((2, 3), dtype=np.uint16),
                },
                "camera_info": camera_info,
            }
        }
        composed = ComposedCameraSensor.__new__(ComposedCameraSensor)

        serialized = composed.serialize_message(message)

        self.assertIsInstance(schema_type, type)
        self.assertEqual(serialized["camera_info"], camera_info)

    def test_depth_flag_enables_only_chest_realsense(self):
        from gear_sonic.camera.composed_camera import (
            ComposedCameraConfig,
            ComposedCameraSensor,
        )

        class FakeRealSenseConfig:
            fps = 30
            enable_depth = False

        class FakeRealSenseSensor:
            def __init__(self, config, mount_position, device_id):
                self.enable_depth = config.enable_depth
                self.mount_position = mount_position
                self.device_id = device_id

        fake_driver = types.SimpleNamespace(
            RealSenseConfig=FakeRealSenseConfig,
            RealSenseSensor=FakeRealSenseSensor,
        )
        composed = ComposedCameraSensor.__new__(ComposedCameraSensor)
        composed.config = ComposedCameraConfig(
            realsense_enable_depth=True,
            run_as_server=False,
        )

        with mock.patch.dict(
            sys.modules,
            {"gear_sonic.camera.drivers.realsense": fake_driver},
        ):
            chest = composed._instantiate_camera(
                "chest_view",
                "realsense",
                "chest-serial",
            )
            wrist = composed._instantiate_camera(
                "left_wrist",
                "realsense",
                "wrist-serial",
            )

        self.assertTrue(chest.enable_depth)
        self.assertFalse(wrist.enable_depth)


if __name__ == "__main__":
    unittest.main()
