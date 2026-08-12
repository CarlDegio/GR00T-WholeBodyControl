# G1 Camera 30 Hz and Host Synchronization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sustain at least 29 Hz for all four RGB and two depth outputs on the G1 while preserving 640x480 image fidelity and synchronizing the verified camera implementation back to the host repository.

**Architecture:** Keep `ImageMessageSchema` responsible for the unchanged JPEG/PNG/Base64 wire representation, but allow an optional executor to encode independent images concurrently. `ComposedCameraSensor` owns and reuses a bounded executor, then shuts it down cleanly. The host's newer RGB-D implementation is the baseline, with the G1's Orbbec/RealSense launch configuration merged into it.

**Tech Stack:** Python 3.10, OpenCV, NumPy, `concurrent.futures.ThreadPoolExecutor`, msgpack/ZMQ, pytest, pyrealsense2, pyorbbecsdk.

## Global Constraints

- Preserve four `640x480@30` RGB streams and two color-aligned `uint16` depth streams.
- Preserve schema version 2, image keys, JPEG quality 80, lossless PNG depth pixels, Base64 representation, timestamps, and camera metadata.
- RealSense depth must remain chest-only; Orbbec depth remains independently controlled.
- Do not overwrite unrelated dirty host or G1 files.
- Runtime acceptance requires every stream and the composed publisher to sustain at least 29.0 Hz for 30 seconds.

---

### Task 1: Establish the synchronized camera baseline

**Files:**
- Modify: `gear_sonic/camera/composed_camera.py`
- Create: `gear_sonic/camera/drivers/orbbec.py`
- Modify: `start_camera_server.zsh`
- Test: `gear_sonic/tests/test_camera_rgbd_protocol.py`
- Test: `gear_sonic/tests/test_orbbec_driver.py`

**Interfaces:**
- Consumes: existing `ComposedCameraConfig`, `RealSenseSensor`, and the host's uncommitted but tested Orbbec driver.
- Produces: one composed launch configuration for Orbbec ego RGB-D, RealSense chest RGB-D, and two wrist RGB devices.

- [ ] **Step 1: Import the host's newer Orbbec integration into the isolated worktree**

Apply the existing host changes for `composed_camera.py`, `orbbec.py`, and their tests without changing behavior. Retain `orbbec_enable_depth: bool = False`, robust serial fallback, and `run_server_from_config()` cleanup.

- [ ] **Step 2: Merge the G1 device launch configuration**

Set `start_camera_server.zsh` to launch:

```zsh
--ego-view-camera orbbec --ego-view-device-id CPMD464001G
--chest-camera realsense --chest-device-id 408122070390
--left-wrist-camera realsense --left-wrist-device-id 218622279421
--right-wrist-camera realsense --right-wrist-device-id 352122270966
--orbbec-enable-depth
--realsense-enable-depth
```

- [ ] **Step 3: Run the synchronized baseline tests**

Run:

```bash
/home/user/Project/GR00T-WholeBodyControl/.venv_camera/bin/python -m pytest \
  gear_sonic/tests/test_camera_rgbd_protocol.py \
  gear_sonic/tests/test_orbbec_driver.py -q
```

Expected: all selected tests pass before encoding optimization.

- [ ] **Step 4: Commit the baseline**

```bash
git add gear_sonic/camera/composed_camera.py \
  gear_sonic/camera/drivers/orbbec.py \
  gear_sonic/tests/test_orbbec_driver.py \
  start_camera_server.zsh
git commit -m "feat: synchronize G1 RGB-D camera configuration"
```

### Task 2: Add executor-aware wire serialization using TDD

**Files:**
- Modify: `gear_sonic/tests/test_camera_rgbd_protocol.py`
- Modify: `gear_sonic/camera/sensor_server.py`

**Interfaces:**
- Consumes: `ImageMessageSchema.serialize()` and `ImageUtils.encode_image()` / `encode_depth_image()`.
- Produces: `ImageMessageSchema.serialize(executor: Executor | None = None) -> dict[str, Any]`.

- [ ] **Step 1: Write the failing parallel serialization test**

Add a recording executor that stores submitted callables and returns completed futures. Build a schema with two RGB images, two depth images, timestamps, and calibration. Assert that:

```python
wire = schema.serialize(executor=executor)
assert executor.submission_count == 4
assert list(wire["images"]) == list(schema.images)
decoded = ImageMessageSchema.deserialize(wire)
np.testing.assert_array_equal(decoded.images["chest_view_depth"], depth)
assert decoded.camera_info == schema.camera_info
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```bash
/home/user/Project/GR00T-WholeBodyControl/.venv_camera/bin/python -m pytest \
  gear_sonic/tests/test_camera_rgbd_protocol.py::test_schema_parallel_serialization_preserves_order_and_payload -q
```

Expected: FAIL because `serialize()` does not accept `executor`.

- [ ] **Step 3: Implement minimal executor-aware serialization**

Add a single-image helper that selects passthrough bytes, depth PNG, or RGB JPEG using the same rules as today. When `executor is None`, call it serially. Otherwise submit one task per image and retrieve future results in original key order. Do not change encoded values or schema fields.

- [ ] **Step 4: Verify GREEN and protocol regression tests**

Run:

```bash
/home/user/Project/GR00T-WholeBodyControl/.venv_camera/bin/python -m pytest \
  gear_sonic/tests/test_camera_rgbd_protocol.py gear_sonic/tests/test_camera_viewer.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit serializer support**

```bash
git add gear_sonic/camera/sensor_server.py gear_sonic/tests/test_camera_rgbd_protocol.py
git commit -m "perf: encode camera message images concurrently"
```

### Task 3: Reuse and close the composed-camera encoder pool using TDD

**Files:**
- Modify: `gear_sonic/tests/test_camera_rgbd_protocol.py`
- Modify: `gear_sonic/camera/composed_camera.py`

**Interfaces:**
- Consumes: `ImageMessageSchema.serialize(executor=...)` from Task 2.
- Produces: a persistent `ComposedCameraSensor._image_encoder_pool` used by `serialize_message()` and released by `close()`.

- [ ] **Step 1: Write the failing executor ownership test**

Construct `ComposedCameraSensor` without camera hardware, attach a recording executor, serialize a merged RGB-D message, and assert the executor receives one task per image. Call `close()` with empty worker/server state and assert executor shutdown is invoked exactly once.

- [ ] **Step 2: Run the new test and verify RED**

Run:

```bash
/home/user/Project/GR00T-WholeBodyControl/.venv_camera/bin/python -m pytest \
  gear_sonic/tests/test_camera_rgbd_protocol.py::test_composed_camera_reuses_and_closes_image_encoder_pool -q
```

Expected: FAIL because composed serialization does not pass an executor and `close()` does not shut it down.

- [ ] **Step 3: Implement persistent pool ownership**

Create a bounded `ThreadPoolExecutor` once in `ComposedCameraSensor.__init__`, with worker count limited to available CPUs and the maximum six encoded images. Pass it to `img_schema.serialize(executor=...)`. Shut it down with `wait=True` in `close()` after camera threads stop and before the server context is destroyed.

- [ ] **Step 4: Verify GREEN and all camera tests**

Run:

```bash
/home/user/Project/GR00T-WholeBodyControl/.venv_camera/bin/python -m pytest gear_sonic/tests -q
```

Expected: all camera-related tests pass with no new warnings.

- [ ] **Step 5: Commit composed pool ownership**

```bash
git add gear_sonic/camera/composed_camera.py gear_sonic/tests/test_camera_rgbd_protocol.py
git commit -m "perf: reuse parallel camera encoders"
```

### Task 4: Deploy safely and verify G1 runtime performance

**Files:**
- Deploy: `gear_sonic/camera/composed_camera.py`
- Deploy: `gear_sonic/camera/sensor_server.py`
- Deploy: `gear_sonic/camera/drivers/realsense.py`
- Deploy: `gear_sonic/camera/drivers/orbbec.py`
- Deploy: `start_camera_server.zsh`

**Interfaces:**
- Consumes: the tested files from Tasks 1-3.
- Produces: the active G1 camera server on port 5555 with the verified configuration.

- [ ] **Step 1: Back up replaced G1 files**

Create a timestamped directory under `/home/unitree/GR00T-WholeBodyControl/Log/camera_backup/` and copy only the five target files into it.

- [ ] **Step 2: Upload to temporary G1 paths and verify checksums**

Upload each file with a `.codex-new` suffix, compare SHA-256 against the isolated worktree, then atomically rename each temporary path onto the target.

- [ ] **Step 3: Restart only the composed camera service**

Terminate the currently identified `gear_sonic.camera.composed_camera` process, start `./start_camera_server.zsh`, and retain its log under `Log/`. Do not stop unrelated robot control or inference processes.

- [ ] **Step 4: Run the 30-second acceptance subscriber**

Measure compressed payload rate and unique timestamp rates for all six expected keys. Decode one message and assert:

```text
ego_view             (480, 640, 3)
ego_view_depth       (480, 640) uint16
chest_view           (480, 640, 3)
chest_view_depth     (480, 640) uint16
left_wrist           (480, 640, 3)
right_wrist          (480, 640, 3)
```

Expected: publisher and every key are at least 29.0 Hz over 30 seconds, with no reconnect/stale warnings in the new server log.

- [ ] **Step 5: If acceptance fails, gather evidence before fallback**

Record per-image encoding time, process CPU, ZMQ payload rate, and each timestamp rate. Only if depth PNG remains the measured bottleneck, add a separate failing test and configurable lossless low-compression PNG fallback.

### Task 5: Synchronize final G1 camera code to the host

**Files:**
- Synchronize the same five target files from Task 4 into `/home/user/Project/GR00T-WholeBodyControl`.

**Interfaces:**
- Consumes: the runtime-verified G1 files.
- Produces: checksum-identical G1, isolated worktree, and host camera implementations for the target files.

- [ ] **Step 1: Inspect host diffs immediately before synchronization**

Confirm no target file changed since the baseline comparison. Preserve unrelated dirty files and stop if any overlapping target has unexpected new edits.

- [ ] **Step 2: Apply only reviewed target-file changes**

Use patch-based updates for tracked files and preserve the robust untracked host Orbbec driver content. Do not copy G1 logs, virtual environments, caches, or unrelated files.

- [ ] **Step 3: Verify host tests**

Run:

```bash
./.venv_camera/bin/python -m pytest gear_sonic/tests -q
```

Expected: all camera-related tests pass.

- [ ] **Step 4: Compare final checksums**

Compute SHA-256 for all five target files on the isolated worktree, G1, and host. Expected: every corresponding hash is identical.

- [ ] **Step 5: Final completion audit**

Confirm the 30-second G1 evidence proves every stream is at least 29.0 Hz, image fidelity is unchanged, no active camera error is present, host tests pass, and all five file hashes match before declaring the goal complete.
