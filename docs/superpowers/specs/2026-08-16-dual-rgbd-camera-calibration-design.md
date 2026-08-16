# Dual RGB-D Camera Calibration Capture Design

## Goal

Capture and preserve the complete live intrinsics for the robot's head and
chest RealSense cameras once, use the saved calibration locally, and stop
including camera calibration metadata in normal robot-to-deployment streaming.

Normal streaming must contain aligned RGB-D for `ego_view` and `chest_view`,
and RGB only for `left_wrist_view` and `right_wrist_view`.

## Scope

This change covers:

- the clean `agent-near` checkout at
  `/home/unitree/GR00T-WholeBodyControl` on `192.168.123.164`;
- the local `agent-near` checkout used for base-pose YOLOE and AgentNav
  inference;
- a one-shot calibration capture workflow;
- local camera-calibration storage and loading;
- removal of raw-YOLOE's live-versus-configured intrinsic comparison; and
- a quantitative decision about whether lens distortion must be included in
  pixel deprojection.

This change does not introduce camera-to-robot extrinsic calibration. Existing
base-pose extrinsic settings remain unchanged.

## Camera Stream Contract

### Normal mode

The robot publishes these image keys on the existing camera ZMQ stream:

- `ego_view`: RGB;
- `ego_view_depth`: uint16 depth aligned to `ego_view`;
- `chest_view`: RGB;
- `chest_view_depth`: uint16 depth aligned to `chest_view`;
- `left_wrist_view`: RGB only; and
- `right_wrist_view`: RGB only.

Normal packets contain no `camera_info` calibration metadata. The existing
schema remains able to decode a missing or empty `camera_info` mapping for
compatibility.

### Calibration mode

An explicit robot-side calibration flag enables `camera_info` publication.
While the flag is enabled, every packet carries complete `ego_view` and
`chest_view` calibration until the local one-shot receiver has captured a
valid packet and the calibration server is stopped. Repetition during this
short session prevents ZMQ PUB/SUB slow-joiner loss from losing the only
calibration message.

Each RGB-D camera entry records:

- `fx`, `fy`, `cx`, and `cy` in pixels;
- color-image `width` and `height`;
- RealSense distortion model and every SDK-provided distortion coefficient;
- `depth_scale_m`;
- `depth_aligned_to`;
- camera serial number;
- configured color and depth dimensions; and
- configured frame rate.

The distortion coefficient order is stored together with the SDK distortion
model, rather than assuming every model uses Brown-Conrady ordering.

## Robot-Side Design

The composed-camera configuration will support more than one RealSense depth
mount. The standard `agent-near` camera launcher selects both `ego_view` and
`chest_view`; the wrist cameras remain RGB-only.

The RealSense driver continues to use `rs.align(rs.stream.color)` separately
for each RGB-D device. It extracts the color stream's complete SDK intrinsics
after the pipeline starts and annotates them with device and stream metadata.

The composed publisher receives an explicit `publish_camera_info` setting.
It merges and serializes `camera_info` only in calibration mode. Normal mode
omits the field contents without changing images or timestamps.

The final normal launcher leaves calibration publication disabled. A separate
calibration launcher or explicit flag enables it only for a deliberate
one-shot capture.

## Local Capture and Storage

A one-shot local receiver connects before the calibration camera server is
started. It waits for one packet satisfying all of these conditions:

- both RGB streams are present and decodable;
- both aligned uint16 depth streams are present with matching image shapes;
- both calibration entries contain all required finite intrinsic fields;
- both depth scales are finite and positive;
- each `depth_aligned_to` value names its corresponding RGB stream; and
- each distortion model has a complete numeric coefficient list.

After validation, the receiver atomically writes:

1. `gear_sonic/config/camera_intrinsics.json`, the active local calibration;
2. `outputs/camera_calibration/<timestamp>/camera_intrinsics.json`, an
   immutable raw copy of both complete camera entries;
3. lossless head and chest RGB PNG files; and
4. lossless head and chest raw uint16 depth PNG files.

The active JSON contains a format version, capture timestamp, robot address,
stream names, and the complete unmodified camera entries. Existing timestamped
backup directories are never overwritten.

## Local Runtime Consumption

A focused calibration loader validates the active JSON once at process startup
and returns calibration by stream name.

Raw base-pose YOLOE obtains `ego_view` width, height, `fx`, `fy`, `cx`, `cy`,
depth scale, and distortion metadata from this loader. Its former per-frame
comparison between live intrinsics and hard-coded expected intrinsics is
removed. The runtime still rejects missing depth, non-uint16 depth, RGB/depth
shape mismatch, unexpected configured resolution, invalid depth scale, and
depth not aligned to the requested RGB stream.

Chest RGB-D consumers obtain the `chest_view` calibration from the same file,
so normal robot packets do not need `camera_info`. Consumers that need only RGB
remain independent of the calibration file.

There is one canonical active calibration source. Static copies of the old
head intrinsics are removed from the raw-servo launch path so they cannot drift
from the saved file.

## Distortion Assessment

After capture, a local assessment samples a grid covering the full color image
and compares:

- rays produced by the existing pinhole formula; and
- rays corrected according to the captured RealSense distortion model and
  coefficients.

The report records the maximum and 95th-percentile pixel-equivalent deviation,
plus the corresponding lateral displacement at representative base-pose
working distances of 0.8 m and 1.5 m.

If the model is `none`, all coefficients are effectively zero, or the measured
deviation is negligible relative to segmentation and control tolerances, the
current pinhole deprojection remains. If the deviation is material, the local
deprojection path corrects pixel coordinates before converting them to 3-D.
The decision and measured values are saved beside the timestamped calibration
copy and reported to the user.

No distortion algorithm is selected before the actual model is known. In
particular, inverse Brown-Conrady data is not silently treated as forward
Brown-Conrady data.

## Failure Handling

- The one-shot receiver writes nothing if either camera or any required field
  is missing.
- Atomic replacement prevents a partial active calibration file.
- The timestamped raw backup is written before the active file is replaced.
- Normal local RGB-D startup fails with a clear calibration-file error when
  the required saved stream entry is unavailable.
- Calibration capture does not stop or replace an unrelated camera process;
  the camera port must be free before the calibration launcher is started.
- Robot-side edits preserve unrelated working-tree changes. The robot branch
  was confirmed clean before implementation begins.

## Testing and Verification

Tests will cover:

- selecting head and chest as simultaneous RealSense depth mounts;
- keeping both wrist cameras RGB-only;
- publishing complete calibration metadata only when explicitly enabled;
- preserving distortion model, coefficients, device metadata, and depth
  alignment through serialization;
- rejecting incomplete dual-camera calibration packets without writing files;
- atomically saving the active calibration and an immutable complete backup;
- decoding normal packets without `camera_info` by using saved calibration;
- removing only the raw-servo intrinsic-delta guard while retaining RGB-D
  validity checks; and
- distortion assessment for zero-distortion and nonzero-distortion fixtures.

Verification includes local unit tests, robot-side camera protocol tests,
Python compilation, one live dual-RGB-D calibration capture, inspection of the
saved JSON and PNG artifacts, and a final live normal-mode packet proving that
both RGB-D streams remain present while `camera_info` is absent.
