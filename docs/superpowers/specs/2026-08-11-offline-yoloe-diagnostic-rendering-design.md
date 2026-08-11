# Offline YOLOE Diagnostic Rendering Design

## Goal

Remove bounding-box, mask-overlay, text-rendering, and annotated-JPEG work from
the live raw-YOLOE visual-servo control path. Preserve exact review data for
every fifth processed camera frame and generate the combined diagnostic images
only when an operator manually runs the offline renderer after the robot run.

## Scope

This change covers diagnostic image composition for raw-YOLOE base-servo runs.
YOLOE tracking, bounding-box and mask inference, depth geometry, controller
updates, and velocity publication remain online because they are required for
closed-loop control.

The runtime continues to save, for frame indices `0, 5, 10, ...`, an
unannotated RGB PNG and separate target/table mask PNG files. It also continues
to append one telemetry record per accepted frame to
`raw_servo_frames.jsonl`. The runtime no longer creates an annotated JPEG for
any frame.

This design supersedes the statements in
`2026-08-11-yoloe-review-sampling-design.md` that require an annotated image
under `frames/` for every accepted frame. The sampled source-artifact contract
and manually invoked offline renderer from that design remain in force.

Changing camera-staleness timing, moving sampled source-artifact writes to a
background thread, changing the sample stride, and rerunning YOLOE offline are
outside this change.

## Live Runtime Contract

`FrameDiagnosticsWriter` becomes a telemetry and sampled-source-artifact
writer. For every accepted frame it performs only the following work:

1. Normalize controller and command metadata.
2. For frame indices divisible by five, save the unannotated RGB image and the
   available binary target/table masks under `review_samples/`.
3. Append the detections, geometry, controller state, command, and sampled
   artifact paths to `raw_servo_frames.jsonl`.

The live path must not:

- alpha-blend a target or table mask into an RGB image;
- draw a target or table bounding box;
- draw guard lines, controller state, command text, or error text;
- encode or write an annotated JPEG;
- create a `frames/` directory for a new run.

The `annotated_image` key remains present in new JSONL records with the value
`null`. Keeping the key makes the schema transition explicit while allowing
consumers to distinguish new source-only records from legacy records whose
value is a relative JPEG path.

The sampled artifact layout remains:

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

Raw images remain unannotated camera RGB encoded losslessly as PNG. Masks
remain single-channel uint8 PNG files containing only `0` and `255`. Missing
detections continue to produce a null artifact path instead of an empty mask.

## Offline Rendering Contract

`gear_sonic/scripts/replay_raw_yoloe_servo.py` remains the sole component that
combines RGB, masks, bounding boxes, and controller text into review JPEGs. It
is manually invoked after a run:

```bash
.venv_inference/bin/python gear_sonic/scripts/replay_raw_yoloe_servo.py \
  outputs/base_pose_adjustment/raw_yoloe_<timestamp>_g<generation>
```

The default output remains `<run_dir>/review_reconstructed/`. The renderer
selects only records with `review_artifacts.sampled == true`, so output is
exactly one image for frame indices `0, 5, 10, ...` that completed their source
artifact writes. It uses the recorded masks and bounding boxes rather than
rerunning YOLOE, preserving the detections that actually drove the robot.

The renderer continues to accept legacy runs with sampled review artifacts and
non-null `annotated_image` paths. It does not depend on the legacy `frames/`
directory and does not overwrite source artifacts.

## Ordering and Failure Behavior

For a sampled frame, the runtime atomically completes the raw RGB and available
mask PNG files before appending the JSONL record that references them. A
failure to encode or save a sampled source artifact retains the existing hard
diagnostic failure behavior: the active servo request stops and publishes the
normal zero-speed stop sequence.

The offline renderer fails with an actionable exception for missing telemetry,
unsafe or missing artifact paths, unreadable source images, or mask/RGB shape
mismatches. It returns an empty output list for a legacy run without sampled
review artifacts.

## Compatibility and Documentation

- Existing run directories remain readable by the offline renderer.
- New run directories omit `frames/` and use `annotated_image: null`.
- `review_samples/`, `raw_servo_frames.jsonl`, and
  `review_reconstructed/` retain their existing names and meanings.
- Operator documentation states that annotated images appear only after the
  replay command is run.

No launch argument or configuration field is added. The review stride remains
the internal constant value five.

## Tests

Diagnostics tests will prove that:

- writing frames `0` through `5` does not create `frames/` or any annotated
  JPEG;
- every JSONL record contains `annotated_image: null`;
- only frames `0` and `5` write unannotated RGB and available mask PNG files;
- sampled mask PNGs retain exact binary pixels and absent table masks retain a
  null path;
- sampled-source write failures retain the hard-stop behavior at the runtime
  boundary.

Offline renderer tests will prove that:

- only sampled records generate reconstructed JPEGs;
- target/table masks, bounding boxes, and diagnostic text are composed by the
  offline script;
- new records with `annotated_image: null` and legacy records with a relative
  annotated-image path both render from `review_artifacts`;
- missing or unsafe source paths and shape mismatches fail clearly.

Final verification will run the focused diagnostics and replay tests, the
raw-YOLOE visual-servo test module, Python syntax compilation for changed
modules, and the repository whitespace check. Verification requires no live
camera, model inference, robot process, or network service.
