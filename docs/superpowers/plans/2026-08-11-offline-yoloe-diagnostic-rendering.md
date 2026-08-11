# Offline YOLOE Diagnostic Rendering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop composing annotated YOLOE images in the live robot-control path while retaining exact source artifacts every five frames for manual post-run rendering.

**Architecture:** Keep `FrameDiagnosticsWriter` on the runtime boundary, but narrow it to JSONL telemetry plus lossless sampled RGB/mask source files. Reuse the existing `render_review_samples()` CLI as the only bbox/mask compositor; new JSONL records expose `annotated_image: null`, and legacy sampled runs remain readable.

**Tech Stack:** Python 3, OpenCV, NumPy, JSON Lines, pytest

## Global Constraints

- YOLOE tracking, bbox/mask inference, depth geometry, controller updates, and velocity publication remain online.
- The live path must not blend masks, draw boxes/guard lines/text, encode annotated JPEGs, or create `frames/`.
- Review sampling remains exactly every five processed camera frames; no new launch or configuration argument is added.
- Sampled source RGB remains lossless PNG, and sampled masks remain single-channel uint8 PNG containing only `0` and `255`.
- Every new frame record keeps the `annotated_image` key and sets it to JSON `null`.
- Offline output remains `<run_dir>/review_reconstructed/`, and the renderer must continue reading legacy sampled runs.
- Do not change camera-staleness timing, add a diagnostics background thread, or rerun YOLOE offline.
- Preserve all pre-existing worktree changes, especially the vertical-recenter fields already present in `base_pose_visual_servo_diagnostics.py` and the untracked raw-servo implementation/test files.

---

### Task 1: Remove online annotated-image composition

**Files:**
- Modify: `gear_sonic/tests/test_base_pose_visual_servo_diagnostics.py:38-167`
- Modify: `gear_sonic/tests/test_base_pose_visual_servo.py:402-439`
- Modify: `gear_sonic/utils/inference/base_pose_visual_servo_diagnostics.py:50-276`

**Interfaces:**
- Consumes: `DetectionFrameData` and `FrameDiagnosticsWriter.write(frame, *, controller_state, command) -> None`.
- Produces: one `raw_servo_frames.jsonl` row per accepted frame with `annotated_image: null`; sampled source paths continue under `review_artifacts`.
- Preserves: `_encode_png()`, `_write_review_artifacts()`, `_detection()`, `_controller()`, and `_command()` signatures.

- [ ] **Step 1: Write the failing source-only writer tests**

Rename `test_writer_records_complete_jsonl_and_annotated_jpeg` to
`test_writer_records_complete_jsonl_without_online_annotation`. Keep its
existing telemetry assertions and replace the annotated-JPEG assertion with:

```python
    assert record["annotated_image"] is None
    assert not (tmp_path / "frames").exists()
```

Extend `test_writer_saves_lossless_review_artifacts_every_five_frames` after
loading `rows`:

```python
    assert all(row["annotated_image"] is None for row in rows)
    assert not (tmp_path / "frames").exists()
```

Update `test_runtime_writes_each_initialized_frame_after_command_update` to
parse the telemetry row and assert source-only behavior:

```python
    record = json.loads((output_dir / "raw_servo_frames.jsonl").read_text())
    assert record["frame_index"] == 0
    assert record["controller"]["phase"] == "yaw_align"
    assert record["annotated_image"] is None
    assert not (output_dir / "frames").exists()
```

Do not change the sampled-PNG hard-stop test; it protects the retained online
source-artifact failure behavior.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
.venv_inference/bin/python -m pytest \
  gear_sonic/tests/test_base_pose_visual_servo_diagnostics.py::test_writer_records_complete_jsonl_without_online_annotation \
  gear_sonic/tests/test_base_pose_visual_servo_diagnostics.py::test_writer_saves_lossless_review_artifacts_every_five_frames \
  gear_sonic/tests/test_base_pose_visual_servo.py::test_runtime_writes_each_initialized_frame_after_command_update \
  -v
```

Expected: FAIL because current records contain `frames/<index>.jpg` and the
writer creates `frames/`.

- [ ] **Step 3: Implement the minimal source-only runtime writer**

Change the class responsibility and constructor to stop creating an annotated
frame directory:

```python
class FrameDiagnosticsWriter:
    """Write frame telemetry and sampled source artifacts for offline review."""

    def __init__(self, output_dir: str | Path, *, review_stride: int = 5):
        if review_stride <= 0:
            raise ValueError("review_stride must be positive")
        self.output_dir = Path(output_dir).resolve()
        self.jsonl_path = self.output_dir / "raw_servo_frames.jsonl"
        self.review_stride = int(review_stride)
        self.review_raw_dir = self.output_dir / "review_samples" / "raw"
        self.review_masks_dir = self.output_dir / "review_samples" / "masks"
```

Delete `_blend_mask()`, `_draw_box()`, and `_annotate()` from
`FrameDiagnosticsWriter`. In `write()`, retain metadata normalization and
sampled source writes, but remove all annotated-image composition/encoding:

```python
    def write(
        self,
        frame: DetectionFrameData,
        *,
        controller_state: Mapping[str, Any],
        command: Mapping[str, Any],
    ) -> None:
        controller = self._controller(controller_state)
        normalized_command = self._command(command)
        review_artifacts = self._write_review_artifacts(frame)
        record = {
            "frame_index": int(frame.frame_index),
            "camera_timestamp": float(frame.camera_timestamp),
            "perception_kind": frame.perception_kind,
            "perception_error": frame.perception_error,
            "annotated_image": None,
            "review_artifacts": review_artifacts,
```

Keep the existing `detections`, `geometry`, `controller`, `command`, and JSONL
append blocks unchanged. Keep `cv2` imported because sampled PNG encoding still
uses `cv2.imencode()` and `cv2.cvtColor()`.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```bash
.venv_inference/bin/python -m pytest \
  gear_sonic/tests/test_base_pose_visual_servo_diagnostics.py \
  gear_sonic/tests/test_base_pose_visual_servo.py::test_runtime_writes_each_initialized_frame_after_command_update \
  -v
```

Expected: all selected tests PASS, including exact mask restoration and the
sampled-PNG hard-stop test.

- [ ] **Step 5: Verify the live module contains no compositor code**

Run:

```bash
rg -n '_annotate|_blend_mask|_draw_box|IMWRITE_JPEG_QUALITY|frames/' \
  gear_sonic/utils/inference/base_pose_visual_servo_diagnostics.py
```

Expected: no matches and exit status `1`.

- [ ] **Step 6: Commit only isolatable Task 1 hunks**

`gear_sonic/utils/inference/base_pose_visual_servo_diagnostics.py` already has
uncommitted vertical-recenter metadata changes, and
`gear_sonic/tests/test_base_pose_visual_servo.py` is an existing untracked user
file. Preserve both. Stage the clean diagnostics test normally and stage only
the source-only writer hunks interactively:

```bash
git add gear_sonic/tests/test_base_pose_visual_servo_diagnostics.py
git add -p gear_sonic/utils/inference/base_pose_visual_servo_diagnostics.py
git diff --cached --check
git diff --cached --name-only
```

Select the class-docstring/constructor, compositor deletion, and
`annotated_image: None` hunks; skip the pre-existing
`vertical_resume_phase`/`vertical_recenter_stable_frames` hunk. Do not stage
`gear_sonic/tests/test_base_pose_visual_servo.py`. If Git cannot isolate those
hunks without including user changes, leave the production file unstaged and
do not create a partial Task 1 commit. Otherwise run:

```bash
git commit -m "perf: defer YOLOE diagnostic rendering"
```

---

### Task 2: Lock offline compatibility and update operator instructions

**Files:**
- Verify unchanged: `gear_sonic/scripts/replay_raw_yoloe_servo.py:100-216`
- Modify: `gear_sonic/tests/test_replay_raw_yoloe_servo.py:35-123`
- Modify: `docs/base_pose_adjustment.md:182-227`

**Interfaces:**
- Consumes: `render_review_samples(run_dir: Path, output_dir: Path | None = None, *, alpha: float = 0.42) -> list[Path]`.
- Produces: sampled JPEGs under `review_reconstructed/` without reading or requiring `annotated_image`.
- Preserves: existing CLI positional `run_dir`, `--output-dir`, and `--alpha` arguments.

- [ ] **Step 1: Add a legacy compatibility characterization test**

Add this test after `test_renderer_restores_only_sampled_bbox_and_masks`:

```python
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
```

This is a characterization test for already-supported legacy telemetry, not a
new renderer behavior. It is expected to pass once Task 1 stops creating
`frames/`, proving the offline script depends only on `review_artifacts`.

- [ ] **Step 2: Run new-format and legacy-format renderer tests**

Run:

```bash
.venv_inference/bin/python -m pytest \
  gear_sonic/tests/test_replay_raw_yoloe_servo.py::test_renderer_restores_only_sampled_bbox_and_masks \
  gear_sonic/tests/test_replay_raw_yoloe_servo.py::test_renderer_ignores_missing_legacy_annotated_jpegs \
  gear_sonic/tests/test_replay_raw_yoloe_servo.py::test_cli_renders_synthetic_run \
  -v
```

Expected: all three tests PASS without any production change to
`replay_raw_yoloe_servo.py`.

- [ ] **Step 3: Update the runtime/offline artifact documentation**

Replace the paragraph claiming one live annotated image per frame with:

```markdown
It writes one telemetry record per accepted camera frame to
`raw_servo_frames.jsonl`. New records set `annotated_image` to `null`; the live
runtime does not create `frames/` or compose bbox/mask overlays. The records
include detections, mask pixel counts, raw-depth geometry, controller phase,
filtered errors, transitions, commands, and sampled source-artifact paths.
Every fifth frame saves an unannotated RGB PNG and separate target/table mask
PNGs under `review_samples/`.
```

Keep the existing replay command and add this sentence immediately after it:

```markdown
Only this post-run command blends masks, draws bounding boxes and diagnostic
text, and encodes the reconstructed JPEGs.
```

- [ ] **Step 4: Verify documentation and the full offline renderer module**

Run:

```bash
rg -n 'annotated_image|does not create `frames/`|Only this post-run command' \
  docs/base_pose_adjustment.md
.venv_inference/bin/python -m pytest \
  gear_sonic/tests/test_replay_raw_yoloe_servo.py -v
```

Expected: documentation matches all three new phrases and every renderer test
passes.

- [ ] **Step 5: Commit the clean compatibility test without capturing user docs**

`docs/base_pose_adjustment.md` contains substantial pre-existing user changes.
Leave that file unstaged unless its new paragraph can be isolated without any
other hunk. Commit the clean tracked compatibility test separately:

```bash
git add gear_sonic/tests/test_replay_raw_yoloe_servo.py
git diff --cached --check
git commit -m "test: preserve offline YOLOE replay compatibility"
```

---

### Task 3: Verify the complete behavior change

**Files:**
- Verify: `gear_sonic/utils/inference/base_pose_visual_servo_diagnostics.py`
- Verify: `gear_sonic/scripts/replay_raw_yoloe_servo.py`
- Verify: `gear_sonic/tests/test_base_pose_visual_servo_diagnostics.py`
- Verify: `gear_sonic/tests/test_base_pose_visual_servo.py`
- Verify: `gear_sonic/tests/test_replay_raw_yoloe_servo.py`
- Verify: `docs/base_pose_adjustment.md`

**Interfaces:**
- Consumes: the source-only runtime writer and existing offline replay CLI.
- Produces: fresh test, syntax, source-scan, and whitespace evidence for handoff.

- [ ] **Step 1: Run the complete raw-servo regression set**

Run:

```bash
.venv_inference/bin/python -m pytest \
  gear_sonic/tests/test_base_pose_visual_servo_diagnostics.py \
  gear_sonic/tests/test_replay_raw_yoloe_servo.py \
  gear_sonic/tests/test_base_pose_visual_servo.py \
  -q
```

Expected: zero failures.

- [ ] **Step 2: Compile the changed Python modules**

Run:

```bash
.venv_inference/bin/python -m py_compile \
  gear_sonic/utils/inference/base_pose_visual_servo_diagnostics.py \
  gear_sonic/scripts/replay_raw_yoloe_servo.py \
  gear_sonic/tests/test_base_pose_visual_servo_diagnostics.py \
  gear_sonic/tests/test_replay_raw_yoloe_servo.py
```

Expected: exit status `0` with no output.

- [ ] **Step 3: Prove composition exists only in the offline script**

Run:

```bash
rg -n '_blend|_draw_detection|cv2\.rectangle|IMWRITE_JPEG_QUALITY' \
  gear_sonic/scripts/replay_raw_yoloe_servo.py
rg -n '_annotate|_blend_mask|_draw_box|IMWRITE_JPEG_QUALITY|frames/' \
  gear_sonic/utils/inference/base_pose_visual_servo_diagnostics.py
```

Expected: the first command finds offline composition/encoding; the second
finds nothing.

- [ ] **Step 4: Check whitespace and review the scoped diff**

Run:

```bash
git diff --check
git status --short
git diff -- \
  gear_sonic/utils/inference/base_pose_visual_servo_diagnostics.py \
  gear_sonic/tests/test_base_pose_visual_servo_diagnostics.py \
  gear_sonic/tests/test_replay_raw_yoloe_servo.py \
  docs/base_pose_adjustment.md
```

Expected: `git diff --check` exits `0`; the diff contains no unrelated edits
made by this task, while pre-existing user changes remain intact and clearly
distinguishable in the final handoff.
