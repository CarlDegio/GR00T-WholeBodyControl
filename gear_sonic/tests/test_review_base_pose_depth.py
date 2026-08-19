from __future__ import annotations

import base64
import json
import os

import cv2
import numpy as np
import pytest

from gear_sonic.scripts import review_base_pose_depth
from gear_sonic.scripts.review_base_pose_depth import (
    BrowserDepthReviewApp,
    DepthReviewViewer,
    auto_color_limits,
    colorize_depth,
    find_latest_run,
    load_review_samples,
    read_pixel,
)


def _write_png(path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


def _write_sample(run_dir, *, frame_index: int, scale: float = 0.001) -> None:
    stem = f"{frame_index:06d}"
    depth_relative = f"review_samples/depth/{stem}.png"
    rgb_relative = f"review_samples/raw/{stem}.png"
    _write_png(
        run_dir / depth_relative,
        np.array([[0, 500], [1250, 2000]], dtype=np.uint16),
    )
    _write_png(run_dir / rgb_relative, np.full((2, 2, 3), 80, dtype=np.uint8))
    record = {
        "frame_index": frame_index,
        "camera_timestamp": 1234.5,
        "camera_stream": "ego_view",
        "attempt_id": 3,
        "failover_stage": "origin_qwen",
        "perception_kind": "observation",
        "perception_error": None,
        "review_artifacts": {
            "raw_depth": depth_relative,
            "raw_rgb": rgb_relative,
            "depth_scale_m": scale,
        },
    }
    with (run_dir / "raw_servo_frames.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def test_latest_run_is_selected_by_frame_log_mtime(tmp_path) -> None:
    old_run = tmp_path / "dual_raw_yoloe_old"
    new_run = tmp_path / "dual_raw_yoloe_new"
    old_run.mkdir()
    new_run.mkdir()
    _write_sample(old_run, frame_index=5)
    _write_sample(new_run, frame_index=10)
    os.utime(old_run / "raw_servo_frames.jsonl", ns=(1_000, 1_000))
    os.utime(new_run / "raw_servo_frames.jsonl", ns=(2_000, 2_000))

    assert find_latest_run(tmp_path) == new_run.resolve()


def test_samples_take_depth_scale_and_metadata_from_log(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_sample(run_dir, frame_index=15, scale=0.0025)

    sample = load_review_samples(run_dir)[0]

    assert sample.frame_index == 15
    assert sample.depth_scale_m == 0.0025
    assert sample.camera_stream == "ego_view"
    assert sample.attempt_id == 3
    assert sample.failover_stage == "origin_qwen"
    assert sample.depth_path == (run_dir / "review_samples/depth/000015.png").resolve()


def test_saved_depth_without_logged_scale_is_rejected(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_sample(run_dir, frame_index=5)
    log_path = run_dir / "raw_servo_frames.jsonl"
    record = json.loads(log_path.read_text())
    record["review_artifacts"]["depth_scale_m"] = None
    log_path.write_text(json.dumps(record) + "\n")

    with pytest.raises(ValueError, match="no valid depth_scale_m"):
        load_review_samples(run_dir)


def test_incomplete_active_log_line_is_ignored(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_sample(run_dir, frame_index=20)
    log_path = run_dir / "raw_servo_frames.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write('{"frame_index": 21')

    with pytest.warns(RuntimeWarning, match="incomplete final line"):
        samples = load_review_samples(run_dir)

    assert [sample.frame_index for sample in samples] == [20]


def test_pixel_reading_and_color_limits_preserve_exact_depth() -> None:
    depth = np.array([[0, 500], [1250, 2000]], dtype=np.uint16)

    reading = read_pixel(depth, x=0, y=1, depth_scale_m=0.002)
    low, high = auto_color_limits(
        depth,
        depth_scale_m=0.002,
        lower_percentile=0,
        upper_percentile=100,
    )
    colors = colorize_depth(
        depth,
        depth_scale_m=0.002,
        min_depth_m=low,
        max_depth_m=high,
    )

    assert reading.raw == 1250
    assert reading.depth_m == pytest.approx(2.5)
    assert low == pytest.approx(1.0)
    assert high == pytest.approx(4.0)
    assert tuple(colors[0, 0]) == (24, 24, 24)
    assert not np.array_equal(colors[0, 1], colors[1, 1])


def test_viewer_renders_exact_value_grid_without_opening_window(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_sample(run_dir, frame_index=25)
    samples = load_review_samples(run_dir)
    viewer = DepthReviewViewer(
        run_dir.resolve(),
        samples,
        initial_index=0,
        zoom_radius=1,
        cell_size=40,
        lower_percentile=0,
        upper_percentile=100,
        min_depth_m=None,
        max_depth_m=None,
    )

    canvas = viewer.render()

    assert canvas.ndim == 3
    assert canvas.shape[2] == 3
    assert viewer.cursor == (1, 1)
    assert read_pixel(viewer.depth, x=1, y=1, depth_scale_m=0.001).depth_m == 2.0


def test_browser_payload_contains_lossless_little_endian_depth(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_sample(run_dir, frame_index=30, scale=0.002)
    app = BrowserDepthReviewApp(
        run_dir.resolve(),
        initial_frame_index=30,
        follow_latest=False,
        zoom_radius=4,
        cell_size=50,
        lower_percentile=0,
        upper_percentile=100,
        min_depth_m=None,
        max_depth_m=None,
    )

    manifest = app.manifest()
    payload = app.frame_payload(30)
    raw_bytes = base64.b64decode(payload["depth_u16_le_base64"])
    decoded = np.frombuffer(raw_bytes, dtype="<u2").reshape(2, 2)
    color = cv2.imdecode(
        np.frombuffer(base64.b64decode(payload["color_png_base64"]), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    np.testing.assert_array_equal(decoded, [[0, 500], [1250, 2000]])
    assert manifest["initial_frame_index"] == 30
    assert manifest["samples"][0]["depth_scale_m"] == 0.002
    assert payload["color_min_m"] == pytest.approx(1.0)
    assert payload["color_max_m"] == pytest.approx(4.0)
    assert color.shape == (2, 2, 3)


def test_cli_defaults_to_browser_ui_without_calling_highgui(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_sample(run_dir, frame_index=35)
    called = {}

    def fake_run_browser(app, *, host, port, open_browser) -> None:
        called.update(app=app, host=host, port=port, open_browser=open_browser)

    monkeypatch.setattr(review_base_pose_depth, "run_browser_viewer", fake_run_browser)
    review_base_pose_depth.main([str(run_dir), "--frame-index", "35"])

    assert isinstance(called["app"], BrowserDepthReviewApp)
    assert called["app"].initial_frame_index == 35
    assert called["host"] == "127.0.0.1"
    assert called["port"] == 8765
    assert called["open_browser"] is False
