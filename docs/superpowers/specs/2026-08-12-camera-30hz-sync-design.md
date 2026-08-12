# G1 Camera 30 Hz and Host Synchronization Design

## Goal

Run the G1 composed camera server with four `640x480` RGB streams and two
color-aligned `uint16` depth streams at a sustained 30 Hz target, then keep the
G1 and host camera implementation synchronized without discarding the newer
host-side RGB-D fixes.

## Current Evidence

- The active G1 process publishes Orbbec ego RGB-D, RealSense chest RGB-D, and
  two RealSense wrist RGB streams.
- All four active devices negotiate USB 3 (`5000M` in the USB topology;
  librealsense reports USB 3.2 for the three active RealSense devices).
- A 12-second subscription measured every stream updating together at about
  `25.7 Hz`, with about `15.5 MB/s` of compressed ZMQ payload.
- One-frame measurement showed the server's equivalent serial work spends
  about `28 ms` encoding four JPEG images and two PNG depth images. Because all
  six timestamps slow down together, this common serial encoding path is the
  primary bottleneck rather than one camera independently dropping frames.

## Architecture

`ImageMessageSchema` remains the sole owner of wire encoding. It will accept an
optional executor when serializing images. Without an executor, behavior stays
serial and backward compatible. With an executor, each RGB JPEG or depth PNG
encode is submitted independently, and results are collected in the original
image order. The output types, keys, JPEG quality, PNG depth precision, Base64
representation, msgpack schema, and deserialization behavior do not change.

`ComposedCameraSensor` will own one persistent bounded `ThreadPoolExecutor` and
pass it to `ImageMessageSchema.serialize()`. Reusing the pool avoids per-frame
thread creation. `close()` will shut the pool down after camera workers stop.
OpenCV's native encoders can run on separate CPU cores while preserving the
existing per-camera acquisition threads.

## Compatibility and Failure Behavior

- Default `ImageMessageSchema.serialize()` remains serial for all existing
  callers.
- Encoded images preserve insertion order and the existing schema version.
- Encoder exceptions propagate to the server loop as they do today; no partial
  image message is sent.
- The camera server keeps the existing `640x480@30` stream profiles.
- RealSense depth stays chest-only; Orbbec depth stays controlled independently
  by `--orbbec-enable-depth`.
- No camera firmware, USB settings, image resolution, depth type, or downstream
  consumer changes are part of this work.

## G1 and Host Synchronization

The host's newer camera protocol files are the source baseline because they
already include stricter depth validation, improved deserialization, persistent
RealSense calibration, clean server shutdown, and robust Orbbec serial lookup.
The G1 device-specific launch configuration is merged into that baseline:

- Orbbec Gemini 345Lg ego serial: `CPMD464001G`
- RealSense chest serial: `408122070390`
- RealSense left wrist serial: `218622279421`
- RealSense right wrist serial: `352122270966`
- Both `--orbbec-enable-depth` and `--realsense-enable-depth`

After automated tests pass, only the reviewed camera files are deployed to the
G1. A backup of every replaced G1 file is retained. Following runtime
verification, the same final files are applied to the host working tree and
SHA-256 hashes are compared between the isolated worktree, G1, and host.

## Testing and Acceptance

Automated tests prove:

1. Parallel serialization produces the same decoded RGB/depth arrays, metadata,
   keys, and ordering as serial serialization.
2. Existing callers that omit an executor remain compatible.
3. The composed server uses the persistent executor and releases it on close.
4. Existing RGB-D protocol and camera tests remain green.

Runtime acceptance on the G1 uses a local ZMQ subscriber for at least 30
seconds and requires:

- composed publish rate at least `29.0 Hz`;
- unique timestamp update rate for each of `ego_view`, `ego_view_depth`,
  `chest_view`, `chest_view_depth`, `left_wrist`, and `right_wrist` at least
  `29.0 Hz`;
- no repeated camera reconnect or stale-frame warnings during the measurement;
- unchanged image shapes and depth `uint16` decoding.

If parallel encoding alone does not meet acceptance, the next measured fallback
is configurable low PNG compression for depth while retaining lossless
`uint16` pixels. It will only be added after a new failing performance test and
will trade network bandwidth for CPU time without reducing image quality.
