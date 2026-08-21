from __future__ import annotations

import json

from gear_sonic.utils.inference.base_pose.diagnostics import (
    AsyncFrameDiagnosticsWriter,
    DetectionFrameData,
    FrameDiagnosticsWriter,
)


def _frame(index: int) -> DetectionFrameData:
    return DetectionFrameData(
        frame_index=index,
        camera_timestamp=float(index),
        image_size=(6, 4),
        target_bbox_xyxy=(1.0, 1.0, 4.0, 3.0),
        target_track_id=1,
        target_confidence=0.9,
    )


def test_writer_serializes_nonfinite_controller_values_as_null(tmp_path) -> None:
    writer = FrameDiagnosticsWriter(tmp_path)
    writer.write(
        _frame(1),
        controller_state={
            "phase": "vertical_recenter",
            "filtered_errors": [float("nan"), float("inf"), 0.0],
            "vertical_recenter_armed": True,
            "vertical_recenter_elapsed_s": float("nan"),
        },
        command={"vx": float("inf"), "vy": 0.0, "wz": 0.0},
    )

    record = json.loads((tmp_path / "raw_servo_frames.jsonl").read_text())
    assert record["controller"]["filtered_errors"] == [None, None, 0.0]
    assert record["controller"]["vertical_recenter_elapsed_s"] is None
    assert record["command"]["vx"] is None


def test_writer_creates_only_jsonl(tmp_path) -> None:
    writer = FrameDiagnosticsWriter(tmp_path)
    writer.write(
        _frame(0),
        control_applied=False,
        controller_state=None,
        command=None,
    )

    record = json.loads((tmp_path / "raw_servo_frames.jsonl").read_text())
    assert {path.name for path in tmp_path.iterdir()} == {"raw_servo_frames.jsonl"}
    assert "review_artifacts" not in record
    assert "completion_capture" not in record
    assert "annotated_image" not in record


def test_async_writer_failure_does_not_stop_later_frames(tmp_path) -> None:
    written: list[int] = []
    logs: list[str] = []

    class FailsFirstFrame:
        def write(self, frame, **_kwargs) -> None:
            if frame.frame_index == 0:
                raise OSError("JSONL write failed")
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
