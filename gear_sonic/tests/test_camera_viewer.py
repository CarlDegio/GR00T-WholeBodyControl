"""Compatibility tests for the RGB-only camera viewer."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest


sys.modules.setdefault("tyro", types.ModuleType("tyro"))

from gear_sonic.camera.sensor_server import ImageMessageSchema
from gear_sonic.scripts.run_camera_viewer import (
    _rgb_camera_names,
    camera_label_color,
    draw_base_pose_overlays,
    format_velocity_label,
    parse_base_pose_viewer_status,
    select_rgb_camera_names,
)
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


def test_rgb_viewer_selects_requested_streams_in_requested_order() -> None:
    images = {
        "ego_view": np.zeros((2, 3, 3), dtype=np.uint8),
        "chest_view": np.zeros((2, 3, 3), dtype=np.uint8),
        "chest_view_depth": np.zeros((2, 3), dtype=np.uint16),
        "left_wrist": np.zeros((2, 3, 3), dtype=np.uint8),
    }

    assert select_rgb_camera_names(
        images, "chest_view,ego_view,missing"
    ) == ["chest_view", "ego_view"]


def test_base_pose_viewer_status_exposes_active_camera_and_velocity() -> None:
    status = parse_base_pose_viewer_status(
        b'{"type":"navila_reasan_velocity_command","source":"base_pose",'
        b'"camera_stream":"chest_view",'
        b'"velocity":{"vx":0.4,"vy":-0.2,"wz":0.1}}'
    )

    assert status.active_camera_stream == "chest_view"
    assert status.velocity == (0.4, -0.2, 0.1)
    assert status.target_bbox_xyxy is None
    assert status.target_lateral_anchor_px is None
    assert status.table_edge_endpoints_px is None
    assert status.desk_mask_row_spans is None
    assert status.overlay_image_size is None
    assert camera_label_color("chest_view", status.active_camera_stream) == (
        0,
        0,
        255,
    )
    assert camera_label_color("ego_view", status.active_camera_stream) == (
        0,
        255,
        0,
    )
    assert format_velocity_label(status.velocity) == (
        "CMD vx=+0.400  vy=-0.200  wz=+0.100"
    )


def test_base_pose_viewer_draws_scaled_target_box_and_table_edge() -> None:
    status = parse_base_pose_viewer_status(
        b'{"type":"navila_reasan_velocity_command","source":"base_pose",'
        b'"camera_stream":"chest_view",'
        b'"velocity":{"vx":0.0,"vy":0.0,"wz":0.0},'
        b'"viewer_overlay":{"target_bbox_xyxy":[40,20,160,80],'
        b'"target_lateral_anchor_px":[100,30],'
        b'"table_edge_endpoints_px":[[0,60],[200,60]],'
        b'"image_size":[200,100]}}'
    )
    active = np.zeros((50, 100, 3), dtype=np.uint8)
    inactive = np.zeros_like(active)

    assert status.target_bbox_xyxy == (40.0, 20.0, 160.0, 80.0)
    assert status.target_lateral_anchor_px == (100.0, 30.0)
    assert status.table_edge_endpoints_px == (
        (0.0, 60.0),
        (200.0, 60.0),
    )
    assert status.overlay_image_size == (200, 100)

    draw_base_pose_overlays(active, "chest_view", status)
    draw_base_pose_overlays(inactive, "ego_view", status)

    assert active[15, 50, 1] > 0
    assert active[15, 50, 0] == 0
    assert active[15, 50, 2] == 0
    assert not np.any(active[25, 50])
    assert np.array_equal(active[30, 50], np.array([0, 0, 255]))
    assert np.count_nonzero(active) > 0
    assert not np.any(inactive)


def test_base_pose_viewer_overlay_requires_valid_bbox_corner_order() -> None:
    with pytest.raises(ValueError, match="invalid corner order"):
        parse_base_pose_viewer_status(
            b'{"type":"navila_reasan_velocity_command","source":"base_pose",'
            b'"velocity":{"vx":0.0,"vy":0.0,"wz":0.0},'
            b'"viewer_overlay":{"target_bbox_xyxy":[20,10,5,30]}}'
        )


def test_base_pose_viewer_draws_processed_desk_mask_on_active_camera() -> None:
    spans = ",".join(f"[{row},2,12]" for row in range(2, 12))
    status = parse_base_pose_viewer_status(
        '{"type":"navila_reasan_velocity_command","source":"base_pose",'
        '"camera_stream":"chest_view",'
        '"velocity":{"vx":0.0,"vy":0.0,"wz":0.0},'
        '"viewer_overlay":{"target_bbox_xyxy":[15,15,19,19],'
        f'"desk_mask_row_spans":[{spans}],"image_size":[20,20]}}}}'
    )
    active = np.zeros((20, 20, 3), dtype=np.uint8)
    inactive = np.zeros_like(active)

    draw_base_pose_overlays(active, "chest_view", status)
    draw_base_pose_overlays(inactive, "ego_view", status)

    assert status.desk_mask_row_spans is not None
    assert active[7, 7, 0] > active[7, 7, 1] > 0
    assert active[7, 7, 2] == 0
    assert not np.any(inactive)


def test_base_pose_viewer_rejects_desk_mask_span_outside_image() -> None:
    with pytest.raises(ValueError, match="outside the image"):
        parse_base_pose_viewer_status(
            b'{"type":"navila_reasan_velocity_command",'
            b'"source":"base_pose",'
            b'"velocity":{"vx":0.0,"vy":0.0,"wz":0.0},'
            b'"viewer_overlay":{"target_bbox_xyxy":[1,1,2,2],'
            b'"desk_mask_row_spans":[[4,2,11]],'
            b'"image_size":[10,10]}}'
        )


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
