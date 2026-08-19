from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from gear_sonic.utils.inference import base_pose_visual_servo as raw_servo
from gear_sonic.utils.inference.base_pose import AlignedRGBDSnapshot
from gear_sonic.utils.inference.base_pose_visual_servo import (
    RawServoCalibration,
    RawServoObservation,
    ServoPhase,
    TableGeometry,
    TargetGeometry,
    VisualServoController,
    estimate_table_geometry,
)


def _observation(
    *,
    bbox: tuple[float, float, float, float],
    yaw: float = 0.0,
) -> RawServoObservation:
    return RawServoObservation(
        target=TargetGeometry(
            1.2,
            0.0,
            (1.2, 0.0, 0.5),
            300,
            0.9,
            1.2,
            lateral_anchor_px=(320.0, 200.0),
        ),
        table=TableGeometry(yaw, 1.0, 100, 0.01, (1.0, 0.0)),
        camera_timestamp=1.0,
        target_track_id=1,
        surface_track_id=2,
        target_bbox_xyxy=bbox,
        image_width=640,
        image_height=480,
    )


def test_table_mask_cleanup_keeps_largest_component_and_fills_holes() -> None:
    mask = np.zeros((30, 45), dtype=bool)
    mask[5:25, 5:25] = True
    mask[10:15, 10:15] = False
    mask[1:4, 35:38] = True

    cleaned = raw_servo._largest_filled_component(mask)

    assert cleaned.dtype == np.uint8
    assert int(np.count_nonzero(cleaned)) == 400
    assert np.all(cleaned[10:15, 10:15] == 1)
    assert np.all(cleaned[1:4, 35:38] == 0)


def test_desk_mask_row_spans_round_trip_binary_regions() -> None:
    mask = np.zeros((4, 7), dtype=np.uint8)
    mask[1, 1:4] = 1
    mask[2, 0:2] = 1
    mask[2, 5:7] = 1

    spans = raw_servo._binary_mask_row_spans(mask)

    assert spans == ((1, 1, 4), (2, 0, 2), (2, 5, 7))


def _snapshot(
    *,
    rgb: np.ndarray | None = None,
    depth_raw: np.ndarray | None = None,
) -> AlignedRGBDSnapshot:
    if rgb is None:
        rgb = np.zeros((200, 240, 3), dtype=np.uint8)
    if depth_raw is None:
        depth_raw = np.full(rgb.shape[:2], 1000, dtype=np.uint16)
    return AlignedRGBDSnapshot(
        rgb=rgb,
        depth_raw=depth_raw,
        fx=200.0,
        fy=200.0,
        cx=120.0,
        cy=100.0,
        depth_scale_m=0.001,
        depth_aligned_to="ego_view",
        depth_source=None,
        timestamp=1.0,
    )


def _calibration() -> RawServoCalibration:
    return RawServoCalibration(240, 200, 200.0, 200.0, 120.0, 100.0)


def _rgb_with_mask_contrast(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[np.asarray(mask) > 0] = 255
    return rgb


def test_rgb_pixel_lines_select_minimum_mean_of_twenty_depths() -> None:
    midpoint_outlier = raw_servo._PixelLineSegment(
        endpoint_a=np.array([20.0, 50.0]),
        endpoint_b=np.array([220.0, 50.0]),
        length=200.0,
    )
    consistently_near = raw_servo._PixelLineSegment(
        endpoint_a=np.array([20.0, 100.0]),
        endpoint_b=np.array([220.0, 100.0]),
        length=200.0,
    )
    depth_raw = np.full((200, 240), 2000, dtype=np.uint16)
    depth_raw[50, :] = 1500
    depth_raw[50, 110:131] = 500
    depth_raw[100, :] = 1000

    selected, sampled_pixels, sampled_depth_m = (
        raw_servo._select_nearest_pixel_segment(
            [midpoint_outlier, consistently_near],
            depth_raw=depth_raw,
            depth_scale_m=0.001,
        )
    )

    assert selected is consistently_near
    assert sampled_pixels.shape == (20, 2)
    assert np.mean(sampled_depth_m) == pytest.approx(1.0)


def test_rgb_pixel_line_depth_tie_prefers_longer_segment() -> None:
    short = raw_servo._PixelLineSegment(
        endpoint_a=np.array([40.0, 50.0]),
        endpoint_b=np.array([140.0, 50.0]),
        length=100.0,
    )
    long = raw_servo._PixelLineSegment(
        endpoint_a=np.array([20.0, 100.0]),
        endpoint_b=np.array([220.0, 100.0]),
        length=200.0,
    )

    selected, _, _ = raw_servo._select_nearest_pixel_segment(
        [short, long],
        depth_raw=np.full((200, 240), 1000, dtype=np.uint16),
        depth_scale_m=0.001,
    )

    assert selected is long


def test_rgb_pixel_line_requires_all_twenty_valid_depths() -> None:
    segment = raw_servo._PixelLineSegment(
        endpoint_a=np.array([20.0, 100.0]),
        endpoint_b=np.array([220.0, 100.0]),
        length=200.0,
    )
    depth_raw = np.full((200, 240), 1000, dtype=np.uint16)
    sampled = np.rint(
        np.linspace(segment.endpoint_a, segment.endpoint_b, 20)
    ).astype(int)
    depth_raw[sampled[5, 1], sampled[5, 0]] = 0

    with pytest.raises(ValueError, match="20 valid sampled depths"):
        raw_servo._select_nearest_pixel_segment(
            [segment],
            depth_raw=depth_raw,
            depth_scale_m=0.001,
        )


def test_rgb_edges_are_intersected_with_mask_and_exclusion() -> None:
    rgb = np.zeros((200, 240, 3), dtype=np.uint8)
    cv2.line(rgb, (20, 100), (220, 100), (255, 255, 255), 3)
    mask = np.zeros((200, 240), dtype=np.uint8)
    mask[70:130, 100:230] = 1
    exclusion = np.zeros_like(mask)
    exclusion[:, 140:161] = 1

    intersection = raw_servo._rgb_mask_edge_intersection(
        rgb,
        mask,
        exclusion_mask=exclusion,
    )

    assert np.count_nonzero(intersection) > 0
    assert not np.any(intersection[:, :100])
    assert not np.any(intersection[:, 140:161])


def test_hough_candidates_are_strictly_longer_than_45_pixels() -> None:
    rgb = np.zeros((200, 240, 3), dtype=np.uint8)
    cv2.line(rgb, (20, 100), (220, 100), (255, 255, 255), 3)
    mask = np.ones((200, 240), dtype=np.uint8)

    candidates = raw_servo._rgb_mask_line_segments(rgb, mask)

    assert candidates
    assert all(
        segment.length > raw_servo._TABLE_EDGE_MIN_LENGTH_PX
        for segment in candidates
    )


def test_table_geometry_uses_camera_plane_without_body_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_camera_to_body(_self, _points):
        raise AssertionError("table edge must remain in the camera frame")

    monkeypatch.setattr(
        RawServoCalibration,
        "camera_to_body",
        fail_camera_to_body,
    )
    mask = np.zeros((200, 240), dtype=np.uint8)
    mask[60:160, 20:220] = 1

    geometry = estimate_table_geometry(
        _snapshot(rgb=_rgb_with_mask_contrast(mask)),
        mask,
        _calibration(),
    )

    assert geometry.line_endpoints_px is not None
    assert geometry.line_length_m > 0.0


def test_table_geometry_rejects_rgb_lines_not_longer_than_45_pixels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    short_segment = raw_servo._PixelLineSegment(
        endpoint_a=np.array([100.0, 100.0]),
        endpoint_b=np.array([145.0, 100.0]),
        length=45.0,
    )
    monkeypatch.setattr(
        raw_servo,
        "_rgb_mask_line_segments",
        lambda *_args, **_kwargs: [short_segment],
    )
    mask = np.ones((200, 240), dtype=np.uint8)

    with pytest.raises(ValueError, match="longer than 45 px"):
        estimate_table_geometry(
            _snapshot(rgb=_rgb_with_mask_contrast(mask)),
            mask,
            _calibration(),
        )


def test_table_geometry_dilates_target_exclusion_by_twenty_pixels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, np.ndarray] = {}
    segment = raw_servo._PixelLineSegment(
        endpoint_a=np.array([20.0, 100.0]),
        endpoint_b=np.array([220.0, 100.0]),
        length=200.0,
    )

    def capture_segments(_rgb, _mask, *, exclusion_mask=None):
        assert exclusion_mask is not None
        captured["exclusion"] = exclusion_mask.copy()
        return [segment]

    monkeypatch.setattr(raw_servo, "_rgb_mask_line_segments", capture_segments)
    desk_mask = np.ones((200, 240), dtype=np.uint8)
    target_mask = np.zeros_like(desk_mask)
    target_mask[90:111, 110:131] = 1

    estimate_table_geometry(
        _snapshot(rgb=_rgb_with_mask_contrast(desk_mask)),
        desk_mask,
        _calibration(),
        target_mask=target_mask,
    )

    exclusion = captured["exclusion"]
    assert np.all(exclusion[90:111, 110:131] > 0)
    assert exclusion[100, 90] > 0
    assert exclusion[100, 150] > 0
    assert exclusion[100, 89] == 0
    assert exclusion[100, 151] == 0


def test_diagnostic_frame_carries_completed_desk_mask_by_value() -> None:
    desk_mask = np.zeros((200, 240), dtype=np.uint8)
    desk_mask[60:160, 20:220] = 1
    observation = replace(
        _observation(bbox=(80.0, 40.0, 160.0, 140.0)),
        desk_mask=desk_mask,
    )

    frame = raw_servo._diagnostic_frame(
        0,
        _snapshot(),
        None,
        None,
        observation,
        kind="observation",
    )
    desk_mask[:] = 0

    assert frame.completed_surface_mask is not None
    assert np.count_nonzero(frame.completed_surface_mask) == 20_000


def test_vertical_recenter_uses_only_configured_forward_speed() -> None:
    controller = VisualServoController(min_linear_speed_m_s=0.22)
    controller.reset(1.0, initial_phase=ServoPhase.VERTICAL_RECENTER)

    command = controller.update(
        _observation(bbox=(260.0, 0.0, 380.0, 20.0)),
        now=1.1,
    )

    assert controller.phase is ServoPhase.VERTICAL_RECENTER
    assert controller.vertical_recenter_started_at == pytest.approx(1.1)
    assert command.velocity == pytest.approx((0.22, 0.0, 0.0))


def test_vertical_recenter_exits_at_eighty_percent_box_bottom() -> None:
    controller = VisualServoController()
    controller.reset(1.0, initial_phase=ServoPhase.VERTICAL_RECENTER)
    controller.update(
        _observation(bbox=(260.0, 0.0, 380.0, 20.0)), now=1.1
    )

    command = controller.update(
        _observation(bbox=(260.0, 300.0, 380.0, 384.0)), now=1.2
    )

    assert command.velocity == (0.0, 0.0, 0.0)
    assert controller.phase is ServoPhase.YAW_ALIGN
    assert controller.vertical_recenter_stable_frames == 1


def test_vertical_recenter_exits_after_point_seven_seconds() -> None:
    controller = VisualServoController()
    controller.reset(1.0, initial_phase=ServoPhase.VERTICAL_RECENTER)
    controller.update(
        _observation(bbox=(260.0, 0.0, 380.0, 20.0)), now=1.1
    )

    assert not controller.stop_if_timed_out(now=1.799)
    assert controller.phase is ServoPhase.VERTICAL_RECENTER
    assert not controller.stop_if_timed_out(now=1.8)
    assert controller.phase is ServoPhase.YAW_ALIGN
    assert controller.vertical_recenter_elapsed_s == pytest.approx(0.7)


def test_normal_head_flow_does_not_use_vertical_target_position() -> None:
    controller = VisualServoController()
    controller.reset(1.0)

    command = controller.update(
        _observation(bbox=(260.0, 0.0, 380.0, 20.0), yaw=0.4),
        now=1.1,
    )

    assert controller.phase is ServoPhase.YAW_ALIGN
    assert command.vx == 0.0
    assert command.vy == 0.0
    assert command.wz > 0.0
