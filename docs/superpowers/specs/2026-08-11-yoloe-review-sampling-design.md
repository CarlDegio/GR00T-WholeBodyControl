# YOLOE Review Sampling Design

## Goal

Add lightweight, lossless review samples to raw YOLOE base-servo runs so an
operator can reconstruct the detector's bounding boxes and segmentation masks
after a real-robot run without rerunning Codex, YOLOE, the camera service, or
the control stack.

## Scope

The runtime saves a complete review sample for processed frame indices
`0, 5, 10, ...`. Existing behavior remains unchanged for all processed frames:
each accepted frame still writes one `raw_servo_frames.jsonl` record and one
annotated image under `frames/`.

The new offline renderer is a manually invoked script. It is not imported,
started, or called by `launch_inference.py`, `base_pose_planner.py`, the camera
worker, or the SONIC relay.

## Runtime Artifact Contract

For each sampled frame, `FrameDiagnosticsWriter` writes lossless files below
the request directory:

```text
review_samples/
  raw/
    000000.png
    000005.png
  masks/
    000000_target.png
    000000_table.png
    000005_target.png
```

- Raw images are unannotated RGB camera frames encoded as ordinary PNG files.
- Masks are single-channel uint8 PNG files with values `0` and `255`.
- A target or table mask is omitted when that detection is absent.
- Sampling uses the camera-frame index, not wall-clock time.
- With the existing 10 Hz worker, a stride of five produces approximately
  2 Hz review samples.

Every JSONL record gains a `review_artifacts` object whose keys are always
present:

```json
{
  "sampled": true,
  "raw_rgb": "review_samples/raw/000005.png",
  "target_mask": "review_samples/masks/000005_target.png",
  "table_mask": null
}
```

For non-sampled frames, `sampled` is `false` and all three paths are `null`.
Paths are relative to the run directory so a run can be moved intact.

The review sampling stride defaults to exactly five. It is an internal
diagnostic constant for this implementation rather than a new real-robot
launch argument.

## Write Ordering and Failure Behavior

For a sampled frame, the writer atomically completes the raw PNG and available
mask PNGs before appending the JSONL record. Therefore a JSONL path never points
at an artifact that failed to finish writing.

Review-sample encoding or writing failures follow the existing per-frame
diagnostic fail-safe behavior: the runtime treats the error as hard, publishes
the normal zero-speed stop sequence, and terminates the active servo request.
No partially written file is exposed under its final name.

## Offline Renderer

Create `gear_sonic/scripts/replay_raw_yoloe_servo.py` with a library entry
point and CLI:

```python
def render_review_samples(
    run_dir: Path,
    output_dir: Path | None = None,
    *,
    alpha: float = 0.42,
) -> list[Path]:
    ...
```

```bash
source .venv_inference/bin/activate
python gear_sonic/scripts/replay_raw_yoloe_servo.py \
  outputs/base_pose_adjustment/raw_yoloe_<timestamp>_g<generation>
```

The default output directory is `<run_dir>/review_reconstructed/`. The script:

1. Reads `raw_servo_frames.jsonl` in frame order.
2. Selects records whose `review_artifacts.sampled` value is true.
3. Resolves the relative raw and mask paths inside the run directory.
4. Validates image and mask dimensions.
5. Alpha-blends the target mask in green and table mask in blue.
6. Draws the recorded target and table bounding boxes.
7. Adds frame index, confidence, phase, filtered errors, and selected command.
8. Writes one reconstructed JPEG per sampled frame with the original six-digit
   frame name.

The renderer never imports YOLOE and never performs inference. It raises a
clear exception for a missing JSONL file, missing referenced artifact,
malformed relative path, unreadable image, or shape mismatch. It accepts older
runs without `review_artifacts` and reports that no sampled frames are
available instead of fabricating overlays.

## Storage and Performance

At 640x480, a lossless raw PNG is currently around 300 KB and binary mask PNGs
are much smaller. A 20-second, 10 Hz run produces about 41 sampled frames and is
expected to add approximately 12--18 MB. PNG encoding occurs only at about
2 Hz; the existing per-frame annotated JPEG remains the dominant continuous
diagnostic path.

## Tests

Diagnostics tests will prove that:

- Frames `0` and `5` write raw RGB and available masks.
- Frames `1` through `4` do not write review artifacts.
- Missing table detections produce a sampled raw image and target mask with a
  null table-mask path.
- JSONL paths match files that exist and masks retain exact binary pixels.

Renderer tests will create a small synthetic run and prove that:

- Only sampled records produce reconstructed images.
- Target and table overlays alter the expected pixels and boxes are drawn.
- Missing optional masks are accepted.
- Missing referenced files and mismatched mask shapes fail with actionable
  errors.

The final verification includes focused diagnostics/renderer tests, the full
`gear_sonic/tests` suite, syntax compilation, whitespace checks, and one
offline replay against newly generated synthetic artifacts. No live camera,
Codex/Qwen request, YOLOE inference, or robot command is part of verification.
