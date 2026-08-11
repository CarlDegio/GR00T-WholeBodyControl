# YOLOE Review Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Save lossless raw RGB and detector masks every five processed YOLOE frames, then reconstruct bbox/mask overlays with a manually invoked offline script.

**Architecture:** Extend the existing `FrameDiagnosticsWriter` so sampled frame records atomically persist portable PNG review artifacts and reference them from `raw_servo_frames.jsonl`. Add a standalone renderer under `gear_sonic/scripts/` that reads only saved artifacts and metadata; it never imports or runs Codex, YOLOE, the camera service, or robot control.

**Tech Stack:** Python 3.10, dataclasses, pathlib, JSONL, NumPy, OpenCV, pytest, and the repository's atomic-write pattern.

## Global Constraints

- Sample exactly processed frame indices whose index modulo five is zero.
- Preserve the existing per-frame annotated JPEG and JSONL behavior for every accepted frame.
- Save unannotated RGB as lossless PNG and available masks as single-channel `0/255` PNG.
- Keep every artifact path relative to the request directory and always emit all `review_artifacts` keys.
- Complete sampled PNG writes before appending their JSONL record.
- Treat review-sample write failures as existing hard diagnostic failures and stop at zero velocity.
- Keep the offline renderer completely disconnected from real-robot launch and runtime imports.
- Do not invoke Codex/Qwen, YOLOE, a camera service, or robot control during tests or smoke verification.

---

### Task 1: Persist complete review samples every five frames

**Files:**
- Modify: `gear_sonic/utils/inference/base_pose_visual_servo_diagnostics.py`
- Modify: `gear_sonic/tests/test_base_pose_visual_servo_diagnostics.py`

**Interfaces:**
- Consumes: existing `DetectionFrameData` RGB, target/table masks, and `FrameDiagnosticsWriter.write` calls.
- Produces: `FrameDiagnosticsWriter(output_dir, review_stride=5)` and a `review_artifacts` JSON object with `sampled`, `raw_rgb`, `target_mask`, and `table_mask` fields.

- [ ] **Step 1: Write the failing stride and artifact test**

Add `import cv2` to the test module, then add a test that writes frames zero
through five with real boolean masks:

```python
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
```

Add a second test with a sampled frame whose table detection is absent:

```python
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
```

- [ ] **Step 2: Run the new diagnostics tests and verify RED**

Run:

```bash
source .venv_inference/bin/activate
pytest -q gear_sonic/tests/test_base_pose_visual_servo_diagnostics.py \
  -k 'review_artifacts or sampled_frame'
```

Expected: fail with a missing `review_artifacts` key and missing PNG files.

- [ ] **Step 3: Implement atomic sampled PNG persistence**

Extend the writer constructor and add focused helpers:

```python
class FrameDiagnosticsWriter:
    def __init__(self, output_dir: str | Path, *, review_stride: int = 5):
        if review_stride <= 0:
            raise ValueError("review_stride must be positive")
        self.output_dir = Path(output_dir).resolve()
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "raw_servo_frames.jsonl"
        self.review_stride = int(review_stride)
        self.review_raw_dir = self.output_dir / "review_samples" / "raw"
        self.review_masks_dir = self.output_dir / "review_samples" / "masks"

    @staticmethod
    def _encode_png(image: np.ndarray, *, name: str) -> bytes:
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise OSError(f"failed to encode {name}")
        return encoded.tobytes()

    def _write_review_artifacts(self, frame: DetectionFrameData) -> dict[str, Any]:
        result: dict[str, Any] = {
            "sampled": False,
            "raw_rgb": None,
            "target_mask": None,
            "table_mask": None,
        }
        if frame.frame_index % self.review_stride:
            return result
        self.review_raw_dir.mkdir(parents=True, exist_ok=True)
        self.review_masks_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{int(frame.frame_index):06d}"
        raw_relative = Path("review_samples") / "raw" / f"{stem}.png"
        raw_bgr = cv2.cvtColor(np.asarray(frame.rgb), cv2.COLOR_RGB2BGR)
        _atomic_write_bytes(
            self.output_dir / raw_relative,
            self._encode_png(raw_bgr, name=f"review RGB {stem}"),
        )
        result.update(sampled=True, raw_rgb=raw_relative.as_posix())
        for key, suffix, mask in (
            ("target_mask", "target", frame.target_mask),
            ("table_mask", "table", frame.surface_mask),
        ):
            if mask is None:
                continue
            if mask.shape != frame.rgb.shape[:2]:
                raise ValueError(f"{suffix} mask shape does not match RGB")
            relative = Path("review_samples") / "masks" / f"{stem}_{suffix}.png"
            binary = mask.astype(bool).astype(np.uint8) * 255
            _atomic_write_bytes(
                self.output_dir / relative,
                self._encode_png(binary, name=f"review {suffix} mask {stem}"),
            )
            result[key] = relative.as_posix()
        return result
```

In `write`, assign `review_artifacts = self._write_review_artifacts(frame)` before encoding the annotated JPEG, then include `"review_artifacts": review_artifacts` in `record` before appending JSONL.

- [ ] **Step 4: Run diagnostics tests and verify GREEN**

Run:

```bash
source .venv_inference/bin/activate
pytest -q gear_sonic/tests/test_base_pose_visual_servo_diagnostics.py
```

Expected: all diagnostics tests pass, including exact mask-pixel restoration.

- [ ] **Step 5: Commit the sampled-artifact unit**

```bash
git add \
  gear_sonic/utils/inference/base_pose_visual_servo_diagnostics.py \
  gear_sonic/tests/test_base_pose_visual_servo_diagnostics.py
git commit -m "feat: sample YOLOE review artifacts"
```

---

### Task 2: Reconstruct bbox and mask overlays offline

**Files:**
- Create: `gear_sonic/scripts/replay_raw_yoloe_servo.py`
- Create: `gear_sonic/tests/test_replay_raw_yoloe_servo.py`

**Interfaces:**
- Consumes: sampled `review_artifacts`, `detections`, `controller`, and `command` fields from `raw_servo_frames.jsonl`.
- Produces: `render_review_samples(run_dir: Path, output_dir: Path | None = None, *, alpha: float = 0.42) -> list[Path]` and `main(argv: Sequence[str] | None = None) -> int`, with positional `run_dir`, optional `--output-dir`, and optional `--alpha`.

- [ ] **Step 1: Write failing renderer tests**

Create a synthetic run through the real diagnostics writer, then assert only sampled frames are rendered:

```python
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


def test_cli_renders_synthetic_run(tmp_path, capsys) -> None:
    run_dir = synthetic_sampled_run(tmp_path)
    output_dir = tmp_path / "cli-output"

    assert main([str(run_dir), "--output-dir", str(output_dir)]) == 0

    assert cv2.imread(str(output_dir / "000000.jpg")) is not None
    assert cv2.imread(str(output_dir / "000005.jpg")) is not None
    assert "rendered 2 review frames" in capsys.readouterr().out
```

Add separate validation tests:

```python
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


def test_renderer_returns_empty_for_legacy_log(tmp_path) -> None:
    run_dir = tmp_path / "legacy"
    run_dir.mkdir()
    (run_dir / "raw_servo_frames.jsonl").write_text(
        json.dumps({"frame_index": 0}) + "\n"
    )

    assert render_review_samples(run_dir) == []
```

- [ ] **Step 2: Run renderer tests and verify RED**

Run:

```bash
source .venv_inference/bin/activate
pytest -q gear_sonic/tests/test_replay_raw_yoloe_servo.py
```

Expected: collection fails because `replay_raw_yoloe_servo` does not exist.

- [ ] **Step 3: Implement path-safe loading and rendering**

Create the renderer as a lightweight module with no imports from camera, inference,
planner, or control modules:

```python
"""Reconstruct sampled raw-YOLOE diagnostics without running robot services."""

from __future__ import annotations

from collections.abc import Sequence
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import cv2
import numpy as np


def _atomic_write_bytes(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}_",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def _resolve_artifact(run_dir: Path, relative: str, description: str) -> Path:
    candidate = (run_dir / relative).resolve()
    try:
        candidate.relative_to(run_dir)
    except ValueError as exc:
        raise ValueError(f"{description} escapes run directory") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"missing {description}: {candidate}")
    return candidate


def _load_mask(path: Path, image_shape: tuple[int, int], description: str) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise OSError(f"failed to decode {description}: {path}")
    if mask.shape != image_shape:
        raise ValueError(
            f"{description} mask shape {mask.shape} does not match RGB {image_shape}"
        )
    return mask > 0


def _blend(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    image[mask] = (
        image[mask].astype(np.float32) * (1.0 - alpha)
        + np.asarray(color, dtype=np.float32) * alpha
    ).astype(np.uint8)


def _draw_detection(
    image: np.ndarray,
    value: dict[str, Any],
    color: tuple[int, int, int],
    label: str,
) -> None:
    bbox = value.get("bbox_xyxy")
    if bbox is None:
        return
    x1, y1, x2, y2 = (int(round(float(item))) for item in bbox)
    confidence = value.get("confidence")
    suffix = "none" if confidence is None else f"{float(confidence):.3f}"
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        image,
        f"{label} {suffix}",
        (max(0, x1), max(14, y1 - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def render_review_samples(
    run_dir: Path,
    output_dir: Path | None = None,
    *,
    alpha: float = 0.42,
) -> list[Path]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be within [0, 1]")
    root = Path(run_dir).resolve()
    jsonl = root / "raw_servo_frames.jsonl"
    if not jsonl.is_file():
        raise FileNotFoundError(f"missing frame telemetry: {jsonl}")
    records = [
        json.loads(line)
        for line in jsonl.read_text().splitlines()
        if line.strip()
    ]
    sampled = [
        row
        for row in records
        if row.get("review_artifacts", {}).get("sampled") is True
    ]
    if not sampled:
        return []
    destination = (
        (root / "review_reconstructed").resolve()
        if output_dir is None
        else Path(output_dir).resolve()
    )
    destination.mkdir(parents=True, exist_ok=True)
    rendered: list[Path] = []
    for row in sampled:
        review = row["review_artifacts"]
        raw_value = review.get("raw_rgb")
        if not isinstance(raw_value, str):
            raise ValueError("sampled frame has no review raw RGB path")
        raw_path = _resolve_artifact(root, raw_value, "review raw RGB")
        image = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"failed to decode review raw RGB: {raw_path}")
        for key, color, description in (
            ("table_mask", (190, 70, 30), "table"),
            ("target_mask", (40, 190, 40), "target"),
        ):
            relative = review.get(key)
            if relative is None:
                continue
            if not isinstance(relative, str):
                raise ValueError(f"{description} mask path must be a string or null")
            mask_path = _resolve_artifact(root, relative, f"review {description} mask")
            mask = _load_mask(mask_path, image.shape[:2], description)
            _blend(image, mask, color, alpha)
        detections = row.get("detections", {})
        _draw_detection(
            image, detections.get("table", {}), (255, 130, 50), "table"
        )
        _draw_detection(
            image, detections.get("target", {}), (30, 255, 30), "target"
        )
        controller = row.get("controller", {})
        command = row.get("command", {})
        lines = (
            f"frame={int(row['frame_index'])} phase={controller.get('phase')}",
            f"errors={controller.get('filtered_errors')}",
            "cmd vx/vy/wz="
            f"{float(command.get('vx', 0.0)):+.3f}/"
            f"{float(command.get('vy', 0.0)):+.3f}/"
            f"{float(command.get('wz', 0.0)):+.3f}",
        )
        for index, line in enumerate(lines):
            cv2.putText(
                image,
                line,
                (6, 18 + index * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        path = destination / f"{int(row['frame_index']):06d}.jpg"
        ok, encoded = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92]
        )
        if not ok:
            raise OSError(
                f"failed to encode reconstructed frame {row['frame_index']}"
            )
        _atomic_write_bytes(path, encoded.tobytes())
        rendered.append(path)
    return rendered


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Reconstruct saved YOLOE bbox and mask overlays.",
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--alpha", type=float, default=0.42)
    args = parser.parse_args(argv)
    rendered = render_review_samples(
        args.run_dir,
        args.output_dir,
        alpha=args.alpha,
    )
    destination = (
        args.run_dir / "review_reconstructed"
        if args.output_dir is None
        else args.output_dir
    ).resolve()
    print(f"rendered {len(rendered)} review frames to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Use green for target masks/boxes and blue for table masks/boxes. The two
detection labels carry their confidences; the text panel carries frame index,
controller phase, filtered errors, and `vx/vy/wz`. Keep `argparse` inside
`main()`; the module guard makes imports side-effect-free.

- [ ] **Step 4: Run renderer tests and verify GREEN**

Run:

```bash
source .venv_inference/bin/activate
pytest -q gear_sonic/tests/test_replay_raw_yoloe_servo.py
```

Expected: all renderer success and validation tests pass.

- [ ] **Step 5: Commit the offline renderer unit**

```bash
git add \
  gear_sonic/scripts/replay_raw_yoloe_servo.py \
  gear_sonic/tests/test_replay_raw_yoloe_servo.py
git commit -m "feat: replay saved YOLOE masks offline"
```

---

### Task 3: Document and verify the complete review workflow

**Files:**
- Modify: `docs/base_pose_adjustment.md`
- Verify: `gear_sonic/utils/inference/base_pose_visual_servo_diagnostics.py`
- Verify: `gear_sonic/scripts/replay_raw_yoloe_servo.py`
- Verify: all `gear_sonic/tests`

**Interfaces:**
- Consumes: Task 1 artifact layout and Task 2 CLI.
- Produces: operator documentation and fresh end-to-end verification evidence.

- [ ] **Step 1: Document artifacts and the offline command**

Document this exact per-run layout:

```text
review_samples/
  raw/
    000000.png
    000005.png
  masks/
    000000_target.png
    000000_table.png
    000005_target.png
review_reconstructed/
  000000.jpg
  000005.jpg
```

State that a stride of five is about 2 Hz at the 10 Hz detector rate, and that
a 20-second 640x480 run is expected to add roughly 12--18 MB. Add this
placeholder-free command, which selects the most recently modified raw-YOLOE
run directory:

```bash
source .venv_inference/bin/activate
RUN_DIR="$(find outputs/base_pose_adjustment -maxdepth 1 -type d \
  -name 'raw_yoloe_*' -printf '%T@ %p\n' | sort -nr | sed -n '1s/^[^ ]* //p')"
test -n "$RUN_DIR"
python gear_sonic/scripts/replay_raw_yoloe_servo.py "$RUN_DIR"
```

State explicitly that the script is post-run only, writes reconstructed images
under `review_reconstructed/`, and never starts YOLOE or sends control commands.

- [ ] **Step 2: Run syntax and focused tests**

```bash
source .venv_inference/bin/activate
python -m py_compile \
  gear_sonic/utils/inference/base_pose_visual_servo_diagnostics.py \
  gear_sonic/scripts/replay_raw_yoloe_servo.py
pytest -q \
  gear_sonic/tests/test_base_pose_visual_servo_diagnostics.py \
  gear_sonic/tests/test_replay_raw_yoloe_servo.py \
  gear_sonic/tests/test_base_pose_visual_servo.py
```

Expected: all focused tests pass.

- [ ] **Step 3: Run full regression and whitespace checks**

```bash
source .venv_inference/bin/activate
pytest -q gear_sonic/tests
git diff --check
```

Expected: the complete test suite passes and `git diff --check` prints no errors.

- [ ] **Step 4: Run a no-model offline smoke replay**

Run the CLI-focused synthetic test, which creates frames zero through five via
`FrameDiagnosticsWriter`, invokes `main()` with an isolated output directory,
and decodes both reconstructed files:

```bash
source .venv_inference/bin/activate
pytest -q gear_sonic/tests/test_replay_raw_yoloe_servo.py::test_cli_renders_synthetic_run -v
```

The temporary run must be generated by test/helper data only; do not call a
model, camera, relay, or robot process.

- [ ] **Step 5: Commit documentation after verification**

```bash
git add docs/base_pose_adjustment.md
git commit -m "docs: explain offline YOLOE replay"
```
