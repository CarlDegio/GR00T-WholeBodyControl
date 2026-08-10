"""Per-frame diagnostic artifacts for raw YOLOE visual servo."""

from __future__ import annotations

import json

import cv2
import numpy as np

from gear_sonic.utils.inference.base_pose_visual_servo_diagnostics import (
    DetectionFrameData,
    FrameDiagnosticsWriter,
)


def diagnostic_frame(frame_index: int, *, include_table: bool) -> DetectionFrameData:
    target_mask = np.zeros((48, 64), dtype=bool)
    target_mask[12:32, 24:44] = True
    table_mask = np.zeros((48, 64), dtype=bool)
    table_mask[4:42, 2:62] = True
    return DetectionFrameData(
        frame_index=frame_index,
        camera_timestamp=float(frame_index),
        rgb=np.full((48, 64, 3), (25, 50, 75), dtype=np.uint8),
        target_bbox_xyxy=(24.0, 12.0, 44.0, 32.0),
        target_mask=target_mask,
        target_track_id=11,
        target_confidence=0.91,
        surface_bbox_xyxy=(2.0, 4.0, 62.0, 42.0) if include_table else None,
        surface_mask=table_mask if include_table else None,
        surface_track_id=22 if include_table else None,
        surface_confidence=0.81 if include_table else None,
    )


def test_writer_records_complete_jsonl_and_annotated_jpeg(tmp_path) -> None:
    target_mask = np.zeros((48, 64), dtype=bool)
    target_mask[12:32, 24:44] = True
    table_mask = np.zeros((48, 64), dtype=bool)
    table_mask[4:42, 2:62] = True
    frame = DetectionFrameData(
        frame_index=7,
        camera_timestamp=12.5,
        rgb=np.zeros((48, 64, 3), dtype=np.uint8),
        target_bbox_xyxy=(24.0, 12.0, 44.0, 32.0),
        target_mask=target_mask,
        target_track_id=11,
        target_confidence=0.91,
        surface_bbox_xyxy=(2.0, 4.0, 62.0, 42.0),
        surface_mask=table_mask,
        surface_track_id=22,
        surface_confidence=0.81,
        target_geometry={
            "forward_m": 0.9,
            "right_m": 0.1,
            "valid_depth_pixels": 300,
            "valid_ratio": 0.75,
            "median_depth_m": 1.0,
        },
        table_geometry={
            "yaw_error_rad": 0.2,
            "line_length_m": 0.8,
            "inlier_count": 120,
            "residual_m": 0.006,
        },
        perception_kind="observation",
        perception_error=None,
    )
    writer = FrameDiagnosticsWriter(tmp_path)

    writer.write(
        frame,
        controller_state={
            "phase": "yaw_align",
            "resume_phase": None,
            "transition_reason": None,
            "filtered_errors": [0.3, 0.1, 0.18],
            "invalid_frames": 0,
            "stable_frames": 0,
            "yaw_stable_frames": 0,
            "recenter_stable_frames": 0,
        },
        command={"vx": 0.0, "vy": 0.0, "wz": 0.05, "duration_s": 0.15},
    )

    record = json.loads((tmp_path / "raw_servo_frames.jsonl").read_text())
    assert record["frame_index"] == 7
    assert record["controller"]["phase"] == "yaw_align"
    assert record["detections"]["target"]["track_id"] == 11
    assert record["geometry"]["target"]["valid_depth_pixels"] == 300
    assert record["command"]["wz"] == 0.05
    assert (tmp_path / "frames" / "000007.jpg").is_file()


def test_writer_keeps_null_fields_for_invalid_frame(tmp_path) -> None:
    writer = FrameDiagnosticsWriter(tmp_path)
    writer.write(
        DetectionFrameData(
            frame_index=0,
            camera_timestamp=1.0,
            rgb=np.zeros((48, 64, 3), dtype=np.uint8),
            perception_kind="invalid",
            perception_error="missing tracked target",
        ),
        controller_state={"phase": "recenter"},
        command={"vx": 0.0, "vy": 0.0, "wz": 0.0, "duration_s": 0.15},
    )

    record = json.loads((tmp_path / "raw_servo_frames.jsonl").read_text())
    assert record["detections"]["target"]["bbox_xyxy"] is None
    assert record["geometry"]["table"] is None
    assert record["geometry"]["table_error"] is None
    assert record["perception_error"] == "missing tracked target"


def test_writer_saves_lossless_review_artifacts_every_five_frames(tmp_path) -> None:
    writer = FrameDiagnosticsWriter(tmp_path)
    for frame_index in range(6):
        writer.write(
            diagnostic_frame(frame_index, include_table=True),
            controller_state={"phase": "yaw_align"},
            command={"vx": 0.0, "vy": 0.0, "wz": 0.0, "duration_s": 0.15},
        )

    rows = [
        json.loads(line)
        for line in (tmp_path / "raw_servo_frames.jsonl").read_text().splitlines()
    ]
    assert [row["review_artifacts"]["sampled"] for row in rows] == [
        True, False, False, False, False, True
    ]
    assert rows[1]["review_artifacts"] == {
        "sampled": False,
        "raw_rgb": None,
        "target_mask": None,
        "table_mask": None,
    }
    assert (tmp_path / "review_samples/raw/000000.png").is_file()
    assert (tmp_path / "review_samples/raw/000005.png").is_file()
    assert not (tmp_path / "review_samples/raw/000001.png").exists()
    restored_rgb = cv2.imread(str(tmp_path / "review_samples/raw/000005.png"))
    expected_rgb = diagnostic_frame(5, include_table=True).rgb
    np.testing.assert_array_equal(
        restored_rgb,
        cv2.cvtColor(expected_rgb, cv2.COLOR_RGB2BGR),
    )
    restored_mask = cv2.imread(
        str(tmp_path / "review_samples/masks/000005_target.png"),
        cv2.IMREAD_UNCHANGED,
    )
    expected = diagnostic_frame(5, include_table=True).target_mask
    assert expected is not None
    np.testing.assert_array_equal(restored_mask, expected.astype(np.uint8) * 255)


def test_sampled_frame_records_null_for_missing_table_mask(tmp_path) -> None:
    writer = FrameDiagnosticsWriter(tmp_path)
    writer.write(
        diagnostic_frame(5, include_table=False),
        controller_state={"phase": "recenter"},
        command={"vx": 0.0, "vy": -0.02, "wz": 0.0, "duration_s": 0.15},
    )
    row = json.loads((tmp_path / "raw_servo_frames.jsonl").read_text())
    review = row["review_artifacts"]
    assert review["raw_rgb"] == "review_samples/raw/000005.png"
    assert review["target_mask"] == "review_samples/masks/000005_target.png"
    assert review["table_mask"] is None
