"""Per-frame diagnostic artifacts for raw YOLOE visual servo."""

from __future__ import annotations

from dataclasses import replace
import json
import math
import threading

import cv2
import numpy as np

from gear_sonic.scripts.base_pose_planner import BasePosePlannerConfig
from gear_sonic.utils.inference.base_pose_visual_servo import (
    RawServoEvent,
    RawServoObservation,
    RawServoRuntime,
    TableGeometry,
    TargetGeometry,
)
from gear_sonic.utils.inference.base_pose_visual_servo_diagnostics import (
    AsyncFrameDiagnosticsWriter,
    DetectionFrameData,
    FrameDiagnosticsWriter,
)


def diagnostic_frame(frame_index: int, *, include_table: bool) -> DetectionFrameData:
    target_mask = np.zeros((48, 64), dtype=bool)
    target_mask[12:32, 24:44] = True
    table_mask = np.zeros((48, 64), dtype=bool)
    table_mask[4:42, 2:62] = True
    table_mask[10:15, 20:25] = False
    completed_table_mask = table_mask.copy()
    completed_table_mask[10:15, 20:25] = True
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
        completed_surface_mask=completed_table_mask if include_table else None,
        surface_track_id=22 if include_table else None,
        surface_confidence=0.81 if include_table else None,
    )


def orientation_diagnostics() -> dict[str, float]:
    return {
        "actual_yaw_rad": 0.5,
        "actual_heading_rad": 0.2,
        "heading_setpoint_rad": 0.25,
        "heading_lag_rad": 0.05,
        "state_age_s": 0.02,
        "telemetry_age_s": 0.01,
    }


def servo_observation() -> RawServoObservation:
    return RawServoObservation(
        camera_timestamp=0.0,
        target=TargetGeometry(
            forward_m=0.6,
            right_m=0.0,
            body_xyz_m=(0.6, 0.0, 0.8),
            valid_depth_pixels=1000,
            valid_ratio=1.0,
            median_depth_m=1.0,
        ),
        table=TableGeometry(
            yaw_error_rad=math.radians(20.0),
            line_length_m=0.7,
            inlier_count=200,
            residual_m=0.005,
            line_center_xy_m=(1.0, 0.0),
        ),
        target_track_id=11,
        surface_track_id=22,
        target_bbox_xyxy=(24.0, 12.0, 44.0, 32.0),
        image_width=64,
    )


def test_writer_records_complete_jsonl_without_online_annotation(tmp_path) -> None:
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
            "line_endpoints_xy_m": [[0.9, -0.3], [0.9, 0.3]],
            "line_endpoints_px": [[4.0, 24.0], [59.0, 24.0]],
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
        orientation=orientation_diagnostics(),
    )

    record = json.loads((tmp_path / "raw_servo_frames.jsonl").read_text())
    assert record["frame_index"] == 7
    assert record["controller"]["phase"] == "yaw_align"
    assert record["detections"]["target"]["track_id"] == 11
    assert record["geometry"]["target"]["valid_depth_pixels"] == 300
    assert record["geometry"]["table"]["line_endpoints_px"] == [
        [4.0, 24.0],
        [59.0, 24.0],
    ]
    assert record["command"]["wz"] == 0.05
    assert record["orientation"] == orientation_diagnostics()
    assert record["annotated_image"] is None
    assert not (tmp_path / "frames").exists()


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
    assert record["orientation"] is None


def test_writer_records_displaced_frame_without_fake_control_metadata(tmp_path) -> None:
    writer = FrameDiagnosticsWriter(tmp_path)

    writer.write(
        diagnostic_frame(1, include_table=True),
        control_applied=False,
        controller_state=None,
        command=None,
    )

    record = json.loads((tmp_path / "raw_servo_frames.jsonl").read_text())
    assert record["control_applied"] is False
    assert record["controller"] is None
    assert record["command"] is None
    assert record["detections"]["target"]["track_id"] == 11
    assert record["orientation"] is None


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
    assert all(row["annotated_image"] is None for row in rows)
    assert not (tmp_path / "frames").exists()
    assert [row["review_artifacts"]["sampled"] for row in rows] == [
        True, False, False, False, False, True
    ]
    assert rows[1]["review_artifacts"] == {
        "sampled": False,
        "raw_rgb": None,
        "target_mask": None,
        "table_mask": None,
        "table_completed_mask": None,
        "table_edge_overlay": None,
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
    completed_mask = cv2.imread(
        str(tmp_path / "review_samples/masks/000005_table_completed.png"),
        cv2.IMREAD_UNCHANGED,
    )
    expected_completed = diagnostic_frame(5, include_table=True).completed_surface_mask
    assert expected_completed is not None
    np.testing.assert_array_equal(
        completed_mask,
        expected_completed.astype(np.uint8) * 255,
    )


def test_writer_draws_recorded_table_edge_in_separate_mask_overlay(tmp_path) -> None:
    frame = replace(
        diagnostic_frame(5, include_table=True),
        table_geometry={
            "line_endpoints_xy_m": [[0.8, -0.2], [0.8, 0.2]],
            "line_endpoints_px": [[4.0, 24.0], [59.0, 24.0]],
        },
    )
    writer = AsyncFrameDiagnosticsWriter(logger=lambda _message: None)
    writer.submit_frame(1, tmp_path, frame)
    writer.submit_decision(
        1,
        5,
        control_applied=True,
        controller_state={"phase": "yaw_align"},
        command={"vx": 0.0, "vy": 0.0, "wz": 0.0, "duration_s": 0.15},
    )
    writer.close(drain=True)

    row = json.loads((tmp_path / "raw_servo_frames.jsonl").read_text())
    review = row["review_artifacts"]
    assert review["table_edge_overlay"] == (
        "review_samples/masks/000005_table_edge.png"
    )
    binary = cv2.imread(
        str(tmp_path / review["table_mask"]), cv2.IMREAD_UNCHANGED
    )
    overlay = cv2.imread(
        str(tmp_path / review["table_edge_overlay"]), cv2.IMREAD_COLOR
    )
    expected = frame.surface_mask
    assert expected is not None
    np.testing.assert_array_equal(binary, expected.astype(np.uint8) * 255)
    assert tuple(overlay[24, 32]) == (0, 0, 255)
    assert tuple(overlay[24, 4]) == (0, 255, 255)


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
    assert review["table_completed_mask"] is None
    assert review["table_edge_overlay"] is None


def test_async_writer_drains_rows_in_frame_order(tmp_path) -> None:
    writer = AsyncFrameDiagnosticsWriter(logger=lambda _message: None)
    output_dir = tmp_path / "run"

    writer.submit_frame(1, output_dir, diagnostic_frame(1, include_table=True))
    writer.submit_decision(
        1,
        1,
        control_applied=False,
        controller_state=None,
        command=None,
    )
    writer.submit_frame(1, output_dir, diagnostic_frame(0, include_table=True))
    writer.submit_decision(
        1,
        0,
        control_applied=True,
        controller_state={"phase": "yaw_align"},
        command={"vx": 0.0, "vy": 0.0, "wz": 0.05, "duration_s": 0.15},
        orientation=orientation_diagnostics(),
    )
    writer.close(drain=True)

    rows = [
        json.loads(line)
        for line in (output_dir / "raw_servo_frames.jsonl").read_text().splitlines()
    ]
    assert [row["frame_index"] for row in rows] == [0, 1]
    assert rows[0]["control_applied"] is True
    assert rows[0]["orientation"] == orientation_diagnostics()
    assert rows[1]["control_applied"] is False
    assert rows[1]["orientation"] is None


def test_async_writer_keeps_frames_and_masks_after_nonfinite_controller_values(
    tmp_path,
) -> None:
    warnings: list[str] = []
    writer = AsyncFrameDiagnosticsWriter(logger=warnings.append)
    output_dir = tmp_path / "run"

    for index in range(6):
        writer.submit_frame(
            1, output_dir, diagnostic_frame(index, include_table=True)
        )
        writer.submit_decision(
            1,
            index,
            control_applied=True,
            controller_state={
                "phase": "vertical_recenter",
                "filtered_errors": (
                    [float("inf"), float("-inf"), float("nan")]
                    if index == 0
                    else [0.1, 0.2, 0.3]
                ),
            },
            command={"vx": 0.3, "vy": 0.0, "wz": 0.0, "duration_s": 0.15},
        )
    writer.close(drain=True)

    rows = [
        json.loads(line)
        for line in (output_dir / "raw_servo_frames.jsonl").read_text().splitlines()
    ]
    assert [row["frame_index"] for row in rows] == list(range(6))
    assert rows[0]["controller"]["filtered_errors"] == [None, None, None]
    assert not warnings
    review_dir = output_dir / "review_samples"
    assert (review_dir / "raw" / "000000.png").exists()
    assert (review_dir / "raw" / "000005.png").exists()
    assert (review_dir / "masks" / "000005_target.png").exists()
    assert (review_dir / "masks" / "000005_table.png").exists()


def test_runtime_attaches_latest_orientation_to_applied_frame(tmp_path) -> None:
    provider_calls: list[float] = []

    def orientation_provider(now: float) -> dict[str, float]:
        provider_calls.append(now)
        return orientation_diagnostics()

    runtime = RawServoRuntime(
        BasePosePlannerConfig(task="align", output_root=str(tmp_path)),
        publish=lambda _message: None,
        logger=lambda _message: None,
        orientation_provider=orientation_provider,
    )
    assert runtime.handle_key("n", now=1.0) == "started"
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    frame = diagnostic_frame(0, include_table=True)
    runtime.diagnostics.submit_frame(1, output_dir, frame)

    assert runtime.accept_event(
        RawServoEvent(
            generation=1,
            kind="initialized",
            output_dir=str(output_dir),
            frame=frame,
            observation=servo_observation(),
        ),
        now=1.1,
    )
    runtime.flush_diagnostics()

    record = json.loads((output_dir / "raw_servo_frames.jsonl").read_text())
    assert provider_calls == [1.1]
    assert record["orientation"] == orientation_diagnostics()


def test_runtime_ignores_orientation_provider_failure(tmp_path) -> None:
    warnings: list[str] = []

    def failed_provider(_now: float) -> None:
        raise ValueError("bad telemetry")

    runtime = RawServoRuntime(
        BasePosePlannerConfig(task="align", output_root=str(tmp_path)),
        publish=lambda _message: None,
        logger=warnings.append,
        orientation_provider=failed_provider,
    )
    assert runtime.handle_key("n", now=1.0) == "started"
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    frame = diagnostic_frame(0, include_table=True)
    runtime.diagnostics.submit_frame(1, output_dir, frame)

    accepted = runtime.accept_event(
        RawServoEvent(
            generation=1,
            kind="initialized",
            output_dir=str(output_dir),
            frame=frame,
            observation=servo_observation(),
        ),
        now=1.1,
    )
    runtime.flush_diagnostics()

    record = json.loads((output_dir / "raw_servo_frames.jsonl").read_text())
    assert accepted
    assert runtime.phase == "aligning"
    assert record["orientation"] is None
    assert any("bad telemetry" in message for message in warnings)


def test_async_writer_submit_does_not_wait_for_blocked_disk(tmp_path) -> None:
    write_started = threading.Event()
    release_write = threading.Event()

    class BlockingWriter:
        def __init__(self, _output_dir) -> None:
            pass

        def write(self, *_args, **_kwargs) -> None:
            write_started.set()
            assert release_write.wait(1.0)

    writer = AsyncFrameDiagnosticsWriter(
        logger=lambda _message: None,
        writer_factory=BlockingWriter,
    )
    writer.submit_frame(1, tmp_path / "run", diagnostic_frame(0, include_table=True))
    writer.submit_decision(
        1,
        0,
        control_applied=False,
        controller_state=None,
        command=None,
    )
    assert write_started.wait(1.0)

    submit_finished = threading.Event()

    def submit_next() -> None:
        writer.submit_frame(
            1, tmp_path / "run", diagnostic_frame(1, include_table=True)
        )
        submit_finished.set()

    submitter = threading.Thread(target=submit_next)
    submitter.start()
    assert submit_finished.wait(0.2)
    release_write.set()
    submitter.join(timeout=1.0)
    writer.close(drain=True)


def test_async_writer_failure_does_not_stop_later_frames(tmp_path) -> None:
    warnings: list[str] = []

    class FailOnceWriter:
        def __init__(self, output_dir) -> None:
            self.writer = FrameDiagnosticsWriter(output_dir)
            self.failed = False

        def write(self, *args, **kwargs) -> None:
            if not self.failed:
                self.failed = True
                raise OSError("injected disk failure")
            self.writer.write(*args, **kwargs)

    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    good_dir = tmp_path / "good"

    def writer_factory(output_dir):
        if str(output_dir).endswith("bad"):
            return FailOnceWriter(output_dir)
        return FrameDiagnosticsWriter(output_dir)

    writer = AsyncFrameDiagnosticsWriter(
        logger=warnings.append,
        writer_factory=writer_factory,
    )
    for index in range(2):
        writer.submit_frame(
            1, tmp_path / "bad", diagnostic_frame(index, include_table=True)
        )
        writer.submit_decision(
            1,
            index,
            control_applied=False,
            controller_state=None,
            command=None,
        )
    writer.submit_frame(2, good_dir, diagnostic_frame(0, include_table=True))
    writer.submit_decision(
        2,
        0,
        control_applied=False,
        controller_state=None,
        command=None,
    )
    writer.close(drain=True)

    assert len(warnings) == 1
    assert "injected disk failure" in warnings[0]
    assert "continuing with later frames" in warnings[0]
    bad_row = json.loads((bad_dir / "raw_servo_frames.jsonl").read_text())
    good_row = json.loads((good_dir / "raw_servo_frames.jsonl").read_text())
    assert bad_row["frame_index"] == 1
    assert good_row["frame_index"] == 0


def test_sampled_png_failure_skips_frame_without_stopping_servo(
    tmp_path, monkeypatch
) -> None:
    def fail_encode_png(*_args, **_kwargs) -> bytes:
        raise OSError("injected review PNG failure")

    monkeypatch.setattr(
        FrameDiagnosticsWriter, "_encode_png", staticmethod(fail_encode_png)
    )
    messages: list[str] = []
    warnings: list[str] = []
    runtime = RawServoRuntime(
        BasePosePlannerConfig(task="align", output_root=str(tmp_path)),
        publish=messages.append,
        logger=warnings.append,
    )
    assert runtime.handle_key("n", now=1.0) == "started"
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    frame = diagnostic_frame(0, include_table=True)
    runtime.diagnostics.submit_frame(1, output_dir, frame)
    accepted = runtime.accept_event(
        RawServoEvent(
            generation=1,
            kind="initialized",
            output_dir=str(output_dir),
            frame=frame,
            observation=RawServoObservation(
                camera_timestamp=0.0,
                target=TargetGeometry(
                    forward_m=0.6,
                    right_m=0.0,
                    body_xyz_m=(0.6, 0.0, 0.8),
                    valid_depth_pixels=1000,
                    valid_ratio=1.0,
                    median_depth_m=1.0,
                ),
                table=TableGeometry(
                    yaw_error_rad=math.radians(20.0),
                    line_length_m=0.7,
                    inlier_count=200,
                    residual_m=0.005,
                    line_center_xy_m=(1.0, 0.0),
                ),
                target_track_id=11,
                surface_track_id=22,
                target_bbox_xyxy=(24.0, 12.0, 44.0, 32.0),
                image_width=64,
            ),
        ),
        now=1.1,
    )
    runtime.flush_diagnostics()

    assert accepted
    assert runtime.phase == "aligning"
    assert not runtime.controller.terminal
    assert len(messages) == 1
    diagnostic_warnings = [
        message for message in warnings if "diagnostic frame write failed" in message
    ]
    assert len(diagnostic_warnings) == 1
    assert "injected review PNG failure" in diagnostic_warnings[0]
    assert "continuing with later frames" in diagnostic_warnings[0]
