from __future__ import annotations

import json

import cv2
import numpy as np

from gear_sonic.utils.inference.base_pose_visual_servo_diagnostics import (
    AsyncFrameDiagnosticsWriter,
    DetectionFrameData,
    FrameDiagnosticsWriter,
)


def _frame(index: int) -> DetectionFrameData:
    return DetectionFrameData(
        frame_index=index,
        camera_timestamp=float(index),
        rgb=np.zeros((4, 6, 3), dtype=np.uint8),
        target_bbox_xyxy=(1.0, 1.0, 4.0, 3.0),
        target_track_id=1,
        target_confidence=0.9,
    )


def test_writer_serializes_nonfinite_controller_values_as_null(tmp_path) -> None:
    writer = FrameDiagnosticsWriter(tmp_path, review_stride=5)

    writer.write(
        _frame(1),
        controller_state={
            "phase": "vertical_recenter",
            "filtered_errors": [float("nan"), float("inf"), 0.0],
            "vertical_recenter_armed": True,
            "vertical_recenter_elapsed_s": float("nan"),
        },
        command={
            "vx": float("inf"),
            "vy": 0.0,
            "wz": 0.0,
            "duration_s": 0.15,
        },
    )

    record = json.loads((tmp_path / "raw_servo_frames.jsonl").read_text())
    assert record["controller"]["filtered_errors"] == [None, None, 0.0]
    assert record["controller"]["vertical_recenter_elapsed_s"] is None
    assert record["command"]["vx"] is None


def test_async_writer_failure_does_not_stop_later_frames(tmp_path) -> None:
    written: list[int] = []
    logs: list[str] = []

    class FailsFirstFrame:
        def write(self, frame, **_kwargs) -> None:
            if frame.frame_index == 0:
                raise OSError("sampled PNG failed")
            written.append(frame.frame_index)

    diagnostics = AsyncFrameDiagnosticsWriter(
        logger=logs.append,
        writer_factory=lambda _output_dir: FailsFirstFrame(),
    )
    for index in (0, 1):
        diagnostics.submit_frame(1, tmp_path, _frame(index))
        diagnostics.submit_decision(
            1,
            index,
            control_applied=False,
            controller_state=None,
            command=None,
        )
    diagnostics.close()

    assert written == [1]
    assert len(logs) == 1
    assert "continuing with later frames" in logs[0]


def test_writer_saves_completed_table_mask_losslessly(tmp_path) -> None:
    original = np.zeros((4, 6), dtype=np.uint8)
    original[1:3, 1:5] = 1
    original[2, 2:4] = 0
    completed = original.copy()
    completed[2, 2:4] = 1
    frame = DetectionFrameData(
        frame_index=5,
        camera_timestamp=5.0,
        rgb=np.zeros((4, 6, 3), dtype=np.uint8),
        surface_mask=original,
        completed_surface_mask=completed,
    )
    writer = FrameDiagnosticsWriter(tmp_path, review_stride=5)

    writer.write(
        frame,
        control_applied=False,
        controller_state=None,
        command=None,
    )

    record = json.loads((tmp_path / "raw_servo_frames.jsonl").read_text())
    relative = record["review_artifacts"]["table_completed_mask"]
    assert relative == "review_samples/masks/000005_table_completed.png"
    saved = cv2.imread(str(tmp_path / relative), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(saved, completed * 255)


def test_writer_saves_paired_rgb_depth_and_rgb_edges(tmp_path) -> None:
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[:, :, 0] = 17
    rgb[:, :, 1] = 83
    rgb[:, :, 2] = 149
    depth = np.arange(24, dtype=np.uint16).reshape(4, 6) * 37
    edges = np.zeros((4, 6), dtype=np.uint8)
    edges[2, 1:5] = 255
    frame = DetectionFrameData(
        frame_index=5,
        camera_timestamp=5.0,
        rgb=rgb,
        depth_raw=depth,
        depth_scale_m=0.001,
        table_rgb_edges=edges,
    )
    writer = FrameDiagnosticsWriter(tmp_path, review_stride=5)

    writer.write(
        frame,
        control_applied=False,
        controller_state=None,
        command=None,
    )

    record = json.loads((tmp_path / "raw_servo_frames.jsonl").read_text())
    artifacts = record["review_artifacts"]
    assert artifacts["raw_rgb"] == "review_samples/raw/000005.png"
    assert artifacts["raw_depth"] == "review_samples/depth/000005.png"
    assert artifacts["table_rgb_edges"] == (
        "review_samples/edges/000005_table_rgb.png"
    )
    assert artifacts["depth_scale_m"] == 0.001
    saved_rgb = cv2.imread(
        str(tmp_path / artifacts["raw_rgb"]), cv2.IMREAD_COLOR
    )
    saved_depth = cv2.imread(
        str(tmp_path / artifacts["raw_depth"]), cv2.IMREAD_UNCHANGED
    )
    saved_edges = cv2.imread(
        str(tmp_path / artifacts["table_rgb_edges"]), cv2.IMREAD_UNCHANGED
    )
    np.testing.assert_array_equal(saved_rgb, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.testing.assert_array_equal(saved_depth, depth)
    np.testing.assert_array_equal(saved_edges, edges)
