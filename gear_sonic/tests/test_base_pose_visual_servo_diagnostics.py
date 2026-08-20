from __future__ import annotations

import json

import cv2
import numpy as np

from gear_sonic.utils.inference.base_pose_visual_servo_diagnostics import (
    AsyncFrameDiagnosticsWriter,
    CameraFrameData,
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


def test_writer_defaults_to_json_only_without_image_artifacts(tmp_path) -> None:
    writer = FrameDiagnosticsWriter(tmp_path, review_stride=1)

    writer.write(
        _frame(0),
        control_applied=False,
        controller_state=None,
        command=None,
    )

    record = json.loads((tmp_path / "raw_servo_frames.jsonl").read_text())
    assert not (tmp_path / "review_samples").exists()
    assert record["review_artifacts"] == {
        "sampled": False,
        "raw_rgb": None,
        "raw_depth": None,
        "depth_scale_m": None,
        "table_rgb_edges": None,
        "target_mask": None,
        "table_mask": None,
        "table_completed_mask": None,
        "table_edge_overlay": None,
    }


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
    writer = FrameDiagnosticsWriter(
        tmp_path, review_stride=5, save_images=True
    )

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
    writer = FrameDiagnosticsWriter(
        tmp_path, review_stride=5, save_images=True
    )

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


def test_completion_capture_saves_three_context_frames_and_all_post_stop_views(
    tmp_path,
) -> None:
    writer = FrameDiagnosticsWriter(tmp_path)
    table_mask = np.zeros((24, 32), dtype=np.uint8)
    table_mask[10:22, 2:30] = 1
    table_edges = np.zeros_like(table_mask)
    table_edges[14, 4:28] = 255
    geometry = {"line_endpoints_px": [[4.0, 14.0], [27.0, 14.0]]}

    def paired_frame(index: int) -> DetectionFrameData:
        chest_rgb = np.full((24, 32, 3), (10 + index), dtype=np.uint8)
        head_rgb = np.zeros((24, 32, 3), dtype=np.uint8)
        head_rgb[:, :, 0] = 120 + index
        return DetectionFrameData(
            frame_index=index,
            camera_timestamp=float(index),
            camera_stream="chest_view",
            rgb=chest_rgb,
            target_bbox_xyxy=(8.0, 4.0, 20.0, 12.0),
            surface_mask=table_mask,
            additional_camera_frames=(
                CameraFrameData(
                    camera_stream="ego_view",
                    camera_timestamp=float(index) + 0.01,
                    rgb=head_rgb,
                    target_bbox_xyxy=(5.0, 3.0, 18.0, 11.0),
                    surface_mask=table_mask,
                    completed_surface_mask=table_mask,
                    table_rgb_edges=table_edges,
                    table_geometry=geometry,
                ),
            ),
        )

    controller = {"phase": "post_stop_sampling"}
    command = {"vx": 0.0, "vy": 0.0, "wz": 0.0, "duration_s": 0.15}
    for index in range(6):
        writer.write(
            paired_frame(index),
            control_applied=index != 3,
            controller_state=None if index == 3 else controller,
            command=None if index == 3 else command,
            completion_capture=(
                "start" if index == 2 else "finish" if index == 4 else None
            ),
        )

    manifest = [
        json.loads(line)
        for line in (tmp_path / "completion_capture" / "manifest.jsonl")
        .read_text()
        .splitlines()
    ]
    completion = [item for item in manifest if item["stage"] == "completion"]
    post_stop = [item for item in manifest if item["stage"] == "post_stop"]
    assert len(completion) == 6
    assert {item["frame_index"] for item in completion} == {0, 1, 2}
    assert len(post_stop) == 4
    assert {item["frame_index"] for item in post_stop} == {3, 4}
    assert {item["camera_stream"] for item in manifest} == {
        "chest_view",
        "ego_view",
    }
    assert all((tmp_path / item["raw_rgb"]).is_file() for item in manifest)
    assert all((tmp_path / item["overlay_rgb"]).is_file() for item in manifest)
    assert not any(item["frame_index"] == 5 for item in manifest)

    head = next(
        item
        for item in post_stop
        if item["frame_index"] == 3 and item["camera_stream"] == "ego_view"
    )
    assert head["target_bbox_xyxy"] == [5.0, 3.0, 18.0, 11.0]
    assert head["table_edge_endpoints_px"] == [[4.0, 14.0], [27.0, 14.0]]
    assert (tmp_path / head["table_mask"]).is_file()
    assert (tmp_path / head["table_completed_mask"]).is_file()
    assert (tmp_path / head["table_rgb_edges"]).is_file()
    raw = cv2.imread(str(tmp_path / head["raw_rgb"]), cv2.IMREAD_COLOR)
    overlay = cv2.imread(str(tmp_path / head["overlay_rgb"]), cv2.IMREAD_COLOR)
    assert raw is not None and overlay is not None
    assert not np.array_equal(raw, overlay)


def test_async_writer_forwards_completion_capture_marker(tmp_path) -> None:
    markers: list[str | None] = []

    class RecordingWriter:
        def write(self, _frame, *, completion_capture=None, **_kwargs) -> None:
            markers.append(completion_capture)

    diagnostics = AsyncFrameDiagnosticsWriter(
        writer_factory=lambda _output_dir: RecordingWriter()
    )
    diagnostics.submit_frame(1, tmp_path, _frame(0))
    diagnostics.submit_decision(
        1,
        0,
        control_applied=False,
        controller_state=None,
        command=None,
        completion_capture="start",
    )
    diagnostics.close()

    assert markers == ["start"]
