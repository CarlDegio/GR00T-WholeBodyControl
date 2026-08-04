"""Compatibility tests for the RGB-only camera viewer."""

from __future__ import annotations

import sys
import types

import numpy as np


sys.modules.setdefault("tyro", types.ModuleType("tyro"))

from gear_sonic.camera.sensor_server import ImageMessageSchema
from gear_sonic.scripts.run_camera_viewer import _rgb_camera_names
from gear_sonic.scripts.run_depth_camera_viewer import colorize_depth


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


def test_colorize_depth_uses_fixed_range_and_marks_invalid_pixels() -> None:
    depth_mm = np.array([[0, 1000, 5000, 6000]], dtype=np.uint16)

    color, stats = colorize_depth(depth_mm, max_depth_m=5.0)

    assert color.shape == (1, 4, 3)
    assert color.dtype == np.uint8
    assert np.array_equal(color[0, 0], np.zeros(3, dtype=np.uint8))
    assert np.array_equal(color[0, 3], np.zeros(3, dtype=np.uint8))
    assert stats.valid_ratio == 0.5
    assert stats.min_depth_m == 1.0
    assert stats.median_depth_m == 3.0
