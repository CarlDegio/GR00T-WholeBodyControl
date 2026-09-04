from __future__ import annotations

from dataclasses import replace
import json

import cv2
import numpy as np

from gear_sonic.utils.inference.base_pose.diagnostics import (
    AsyncFrameDiagnosticsWriter,
    CameraImageData,
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


def test_writer_serializes_each_yaw_align_candidate_depth_stat(tmp_path) -> None:
    candidate = {
        "camera_stream": "ego_view",
        "line_endpoints_px": [[10.0, 20.0], [70.0, 22.0]],
        "line_length_px": 60.03,
        "valid_depth_samples": 4,
        "median_depth_m": 1.25,
        "passes_depth_filter": False,
        "selected": False,
    }
    writer = FrameDiagnosticsWriter(tmp_path)
    writer.write(
        replace(
            _frame(2),
            yaw_align_candidate_lines=(candidate,),
        ),
        control_applied=False,
        controller_state=None,
        command=None,
    )

    record = json.loads((tmp_path / "raw_servo_frames.jsonl").read_text())
    assert record["geometry"]["yaw_align_candidate_lines"] == [candidate]


def test_writer_saves_sampled_camera_rgb_with_mask_and_head_lines(
    tmp_path,
) -> None:
    rgb = np.zeros((40, 60, 3), dtype=np.uint8)
    mask = np.zeros((40, 60), dtype=np.uint8)
    mask[22:34, 20:40] = 1
    head = CameraImageData(
        camera_stream="ego_view",
        camera_timestamp=12.5,
        rgb=rgb,
        target_mask=mask,
        candidate_lines_px=(((5.0, 6.0), (54.0, 6.0)),),
        selected_line_px=((5.0, 12.0), (54.0, 12.0)),
    )
    chest = CameraImageData(
        camera_stream="chest_view",
        camera_timestamp=12.6,
        rgb=np.full_like(rgb, 32),
    )
    writer = FrameDiagnosticsWriter(tmp_path)
    writer.write(
        replace(_frame(5), camera_images=(head, chest)),
        control_applied=False,
        controller_state=None,
        command=None,
    )

    record = json.loads((tmp_path / "raw_servo_frames.jsonl").read_text())
    assert [item["camera_stream"] for item in record["image_artifacts"]] == [
        "ego_view",
        "chest_view",
    ]
    assert record["image_artifacts"][0]["target_mask"]
    assert record["image_artifacts"][0]["candidate_line_count"] == 1
    assert record["image_artifacts"][0]["selected_line"]

    head_path = tmp_path / "diagnostic_images/ego_view/frame_000005.jpg"
    chest_path = tmp_path / "diagnostic_images/chest_view/frame_000005.jpg"
    assert head_path.is_file()
    assert chest_path.is_file()
    annotated = cv2.cvtColor(cv2.imread(str(head_path)), cv2.COLOR_BGR2RGB)
    # Selected line is green and the target mask is a magenta overlay.
    assert int(annotated[12, 30, 1]) > int(annotated[12, 30, 0])
    assert int(annotated[28, 30, 0]) > int(annotated[28, 30, 1])
    assert int(annotated[28, 30, 2]) > int(annotated[28, 30, 1])


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


def test_async_writer_saves_initial_camera_images_without_control_frame(
    tmp_path,
) -> None:
    diagnostics = AsyncFrameDiagnosticsWriter()
    diagnostics.submit_camera_images(
        1,
        tmp_path,
        -1,
        (
            CameraImageData(
                camera_stream="ego_view",
                camera_timestamp=1.0,
                rgb=np.zeros((4, 6, 3), dtype=np.uint8),
            ),
            CameraImageData(
                camera_stream="chest_view",
                camera_timestamp=1.1,
                rgb=np.zeros((4, 6, 3), dtype=np.uint8),
            ),
        ),
    )
    diagnostics.close()

    assert (tmp_path / "diagnostic_images/ego_view/initial.jpg").is_file()
    assert (tmp_path / "diagnostic_images/chest_view/initial.jpg").is_file()
    assert not (tmp_path / "raw_servo_frames.jsonl").exists()
