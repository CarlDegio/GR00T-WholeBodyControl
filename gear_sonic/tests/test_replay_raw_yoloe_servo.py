from pathlib import Path
import json

import cv2
import numpy as np
import pytest

from gear_sonic.scripts.replay_raw_yoloe_servo import (
    main,
    render_review_samples,
)
from gear_sonic.utils.inference.base_pose_visual_servo_diagnostics import (
    DetectionFrameData,
    FrameDiagnosticsWriter,
)


def review_frame(frame_index: int, *, include_table: bool) -> DetectionFrameData:
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


def synthetic_sampled_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    writer = FrameDiagnosticsWriter(run_dir)
    for index in range(6):
        writer.write(
            review_frame(index, include_table=index == 0),
            controller_state={
                "phase": "yaw_align",
                "filtered_errors": [0.3, 0.1, 0.2],
            },
            command={"vx": 0.0, "vy": 0.0, "wz": 0.05, "duration_s": 0.15},
        )
    return run_dir


def test_renderer_restores_only_sampled_bbox_and_masks(tmp_path) -> None:
    run_dir = synthetic_sampled_run(tmp_path)

    rendered = render_review_samples(run_dir)

    assert [path.name for path in rendered] == ["000000.jpg", "000005.jpg"]
    first = cv2.imread(str(rendered[0]))
    raw = cv2.imread(str(run_dir / "review_samples/raw/000000.png"))
    assert first is not None
    assert raw is not None
    assert np.mean(np.abs(first[20, 30].astype(float) - raw[20, 30])) > 10.0
    assert int(first[12, 30, 1]) > int(first[12, 30, 0])


def test_renderer_accepts_sampled_frame_with_null_control_metadata(tmp_path) -> None:
    run_dir = tmp_path / "displaced"
    writer = FrameDiagnosticsWriter(run_dir)
    writer.write(
        review_frame(0, include_table=True),
        control_applied=False,
        controller_state=None,
        command=None,
    )

    rendered = render_review_samples(run_dir)

    assert [path.name for path in rendered] == ["000000.jpg"]
    assert cv2.imread(str(rendered[0])) is not None


def test_renderer_ignores_missing_legacy_annotated_jpegs(tmp_path) -> None:
    run_dir = synthetic_sampled_run(tmp_path)
    jsonl = run_dir / "raw_servo_frames.jsonl"
    rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
    for row in rows:
        if row["review_artifacts"]["sampled"]:
            row["annotated_image"] = f"frames/{row['frame_index']:06d}.jpg"
    jsonl.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    assert not (run_dir / "frames").exists()
    rendered = render_review_samples(run_dir)

    assert [path.name for path in rendered] == ["000000.jpg", "000005.jpg"]


def test_cli_renders_synthetic_run(tmp_path, capsys) -> None:
    run_dir = synthetic_sampled_run(tmp_path)
    output_dir = tmp_path / "cli-output"

    assert main([str(run_dir), "--output-dir", str(output_dir)]) == 0

    assert cv2.imread(str(output_dir / "000000.jpg")) is not None
    assert cv2.imread(str(output_dir / "000005.jpg")) is not None
    assert "rendered 2 review frames" in capsys.readouterr().out


def test_renderer_rejects_missing_referenced_raw_image(tmp_path) -> None:
    run_dir = synthetic_sampled_run(tmp_path)
    (run_dir / "review_samples/raw/000000.png").unlink()
    with pytest.raises(FileNotFoundError, match="review raw RGB"):
        render_review_samples(run_dir)


def test_renderer_rejects_mask_shape_mismatch(tmp_path) -> None:
    run_dir = synthetic_sampled_run(tmp_path)
    cv2.imwrite(
        str(run_dir / "review_samples/masks/000000_target.png"),
        np.zeros((2, 2), dtype=np.uint8),
    )
    with pytest.raises(ValueError, match="mask shape"):
        render_review_samples(run_dir)


def test_renderer_rejects_artifact_path_escape(tmp_path) -> None:
    run_dir = synthetic_sampled_run(tmp_path)
    jsonl = run_dir / "raw_servo_frames.jsonl"
    rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
    rows[0]["review_artifacts"]["raw_rgb"] = "../outside.png"
    jsonl.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    with pytest.raises(ValueError, match="escapes run directory"):
        render_review_samples(run_dir)


def test_renderer_rejects_absolute_artifact_path(tmp_path) -> None:
    run_dir = synthetic_sampled_run(tmp_path)
    jsonl = run_dir / "raw_servo_frames.jsonl"
    rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
    rows[0]["review_artifacts"]["raw_rgb"] = str(
        run_dir / "review_samples/raw/000000.png"
    )
    jsonl.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    with pytest.raises(ValueError, match="must be relative"):
        render_review_samples(run_dir)


def test_renderer_returns_empty_for_legacy_log(tmp_path) -> None:
    run_dir = tmp_path / "legacy"
    run_dir.mkdir()
    (run_dir / "raw_servo_frames.jsonl").write_text(
        json.dumps({"frame_index": 0}) + "\n"
    )

    assert render_review_samples(run_dir) == []
