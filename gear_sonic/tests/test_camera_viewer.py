"""Compatibility tests for the RGB-only camera viewer."""

from __future__ import annotations

import sys
import types

import numpy as np


sys.modules.setdefault("tyro", types.ModuleType("tyro"))

from gear_sonic.camera.sensor_server import ImageMessageSchema
from gear_sonic.scripts.run_camera_viewer import _rgb_camera_names


def test_rgb_viewer_ignores_depth_from_schema_v2_message() -> None:
    message = ImageMessageSchema(
        timestamps={"chest_view": 1.0, "chest_view_depth": 1.0},
        images={
            "chest_view": np.zeros((2, 3, 3), dtype=np.uint8),
            "chest_view_depth": np.full((2, 3), 1000, dtype=np.uint16),
        },
        camera_info={
            "chest_view": {
                "fx": 500.0,
                "fy": 500.0,
                "cx": 1.0,
                "cy": 1.0,
                "width": 3,
                "height": 2,
                "depth_scale_m": 0.001,
                "depth_aligned_to": "chest_view",
            }
        },
    )
    decoded = ImageMessageSchema.deserialize(message.serialize())

    assert _rgb_camera_names(decoded.images) == ["chest_view"]
