import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gear_sonic.camera.sensor_server import ImageMessageSchema, ImageUtils


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
