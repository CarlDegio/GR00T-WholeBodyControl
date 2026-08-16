# Dual RGB-D Camera Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stream aligned RGB-D from the robot head and chest cameras, capture both complete calibrations once, consume the saved values locally, and omit calibration metadata from normal robot packets.

**Architecture:** The robot camera server gains multi-mount depth selection and an explicit calibration-metadata mode. A local one-shot receiver validates and atomically stores a canonical dual-camera JSON plus immutable artifacts; inference receivers load this file instead of requiring `camera_info` in every packet. Raw YOLOE retains RGB-D validity checks but removes the live intrinsic-delta guard.

**Tech Stack:** Python 3.10+, pyrealsense2, NumPy, OpenCV, ZeroMQ, msgpack, pytest, zsh launchers.

## Global Constraints

- Target the clean robot `agent-near` branch at `/home/unitree/GR00T-WholeBodyControl` on `192.168.123.164`.
- Preserve `ego_view` and `chest_view` as aligned uint16 RGB-D; wrist streams remain RGB-only.
- Normal robot packets contain no calibration metadata.
- Preserve every SDK-provided distortion coefficient together with its distortion model.
- Store one canonical active calibration and one timestamped immutable raw backup.
- Do not change camera-to-robot extrinsics.
- Do not remove depth type, shape, scale, alignment, or resolution validation.

---

### Task 1: Robot dual-depth and calibration-mode protocol

**Files:**
- Modify locally and deploy equivalent patches to robot: `gear_sonic/camera/composed_camera.py`
- Modify locally and deploy equivalent patches to robot: `gear_sonic/camera/drivers/realsense.py`
- Modify locally and deploy equivalent patches to robot: `gear_sonic/tests/test_camera_rgbd_protocol.py`
- Modify locally and deploy equivalent patches to robot: `start_camera_server.zsh`
- Create locally and on robot: `start_camera_calibration_server.zsh`

**Interfaces:**
- Consumes: `ComposedCameraConfig.realsense_depth_mounts: tuple[str, ...]` and `publish_camera_info: bool`.
- Produces: packets with dual RGB-D always and `camera_info` only in explicit calibration mode.

- [ ] **Step 1: Write failing protocol tests**

Add tests that instantiate the composer with:

```python
ComposedCameraConfig(
    realsense_enable_depth=True,
    realsense_depth_mounts=("ego_view", "chest_view"),
    publish_camera_info=False,
)
```

Assert that both selected RealSense sensors receive `enable_depth=True`, wrist
sensors receive `False`, normal serialization returns an empty `camera_info`,
and calibration serialization preserves both complete camera entries.

Extend the fake RealSense intrinsics with a model and five coefficients. Assert
the emitted entry contains `distortion_model`, `distortion_coeffs`,
`camera_serial`, configured color/depth dimensions, FPS, depth scale, and
alignment target.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
.venv_inference/bin/python -m pytest -q gear_sonic/tests/test_camera_rgbd_protocol.py
```

Expected: failures because multi-mount selection, conditional metadata, and
distortion/device fields do not exist.

- [ ] **Step 3: Implement the minimum robot protocol**

Add validated multi-mount selection and conditional metadata merging. Preserve
the existing singular setting only as a compatibility fallback if existing
callers require it. Capture SDK intrinsics without interpreting coefficient
order:

```python
"distortion_model": str(intrinsics.model),
"distortion_coeffs": [float(value) for value in intrinsics.coeffs],
```

Add the selected serial and configured stream properties. Make the normal
launcher select `ego_view` and `chest_view` depth without enabling calibration
metadata. The calibration launcher uses the same streams with metadata enabled.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the same pytest command and require zero failures.

- [ ] **Step 5: Deploy the tested patch to the clean robot branch**

Transfer a generated patch, inspect it with `git diff --check`, apply it to the
robot checkout, and run the robot's camera protocol tests without starting
hardware.

### Task 2: Canonical calibration model and one-shot capture

**Files:**
- Create: `gear_sonic/camera/calibration.py`
- Create: `gear_sonic/scripts/capture_camera_calibration.py`
- Create: `gear_sonic/tests/test_camera_calibration.py`

**Interfaces:**
- Produces: `CameraIntrinsics.from_mapping(stream_name, value)`, `load_camera_intrinsics(path)`, and `capture_camera_calibration(...)`.
- Produces active file: `gear_sonic/config/camera_intrinsics.json`.

- [ ] **Step 1: Write failing validation and persistence tests**

Use a complete literal packet containing both RGB-D streams and two distinct
calibration entries. Assert that extraction rejects a missing stream, wrong
depth dtype, shape mismatch, invalid scale, wrong alignment, and malformed
distortion coefficients. Assert a valid capture writes four lossless PNGs, a
complete timestamped JSON backup, and then atomically replaces the active JSON.

- [ ] **Step 2: Run the new tests and verify RED**

```bash
.venv_inference/bin/python -m pytest -q gear_sonic/tests/test_camera_calibration.py
```

Expected: import failure because the calibration module and capture script do
not exist.

- [ ] **Step 3: Implement strict calibration parsing and atomic persistence**

The immutable calibration value exposes `fx`, `fy`, `cx`, `cy`, dimensions,
distortion model/coefficients, depth scale/alignment, serial, and capture
metadata. The receiver connects before the publisher, optionally touches a
ready file, and writes nothing until one packet passes every validation.

- [ ] **Step 4: Run the new tests and verify GREEN**

Run the focused test file and require zero failures.

### Task 3: Consume saved head and chest calibration locally

**Files:**
- Modify: `gear_sonic/utils/inference/base_pose.py`
- Modify: `gear_sonic/utils/inference/object_nav.py`
- Modify: `gear_sonic/scripts/base_pose_planner.py`
- Modify: `gear_sonic/scripts/launch_inference.py`
- Modify: `gear_sonic/tests/test_base_pose_adjustment.py`
- Modify: `gear_sonic/tests/test_object_nav_codex.py`

**Interfaces:**
- Consumes: `load_camera_intrinsics(path)[stream_name]`.
- Produces: RGB-D snapshots populated from saved calibration when packets omit `camera_info`.

- [ ] **Step 1: Write failing receiver tests**

Create packets containing images and timestamps but no `camera_info`. Pass a
literal saved head or chest calibration to the receiver and assert the decoded
snapshot contains the saved numeric values. Keep tests proving malformed RGB-D
still fails.

- [ ] **Step 2: Run focused receiver tests and verify RED**

```bash
.venv_inference/bin/python -m pytest -q \
  gear_sonic/tests/test_base_pose_adjustment.py \
  gear_sonic/tests/test_object_nav_codex.py
```

Expected: missing-`camera_info` errors.

- [ ] **Step 3: Implement startup-loaded calibration injection**

Load once per camera client construction, not per frame. Prefer explicitly
injected calibration in tests. Retain packet metadata as a legacy fallback for
LingBot-generated local streams, but normal remote operation must not require
it. Add a configurable active-calibration path to launch configuration.

- [ ] **Step 4: Run focused receiver tests and verify GREEN**

Run both test files and require zero failures.

### Task 4: Remove raw-YOLOE intrinsic delta guard

**Files:**
- Modify: `gear_sonic/utils/inference/base_pose_visual_servo.py`
- Modify: `gear_sonic/scripts/base_pose_planner.py`
- Modify: `gear_sonic/tests/test_base_pose_visual_servo.py`

**Interfaces:**
- Consumes: saved `ego_view` calibration.
- Produces: deprojection using saved head intrinsics with no per-frame delta comparison.

- [ ] **Step 1: Replace the intrinsic-mismatch test with retained safety tests**

Assert that a snapshot using injected saved intrinsics passes without a
live-versus-static comparison. Independently assert that missing depth,
unexpected resolution, nonaligned shape, and invalid scale still fail.

- [ ] **Step 2: Run the focused test and verify RED**

```bash
.venv_inference/bin/python -m pytest -q \
  gear_sonic/tests/test_base_pose_visual_servo.py \
  -k 'calibration or intrinsic or depth'
```

Expected: the old delta guard still raises.

- [ ] **Step 3: Remove static intrinsic fields and the delta comparison**

Build raw-servo calibration from the canonical head entry. Preserve the
pinhole values for deprojection, but delete `max_intrinsic_delta_px`, its launch
setting, and the tuple comparison in `validate_snapshot`.

- [ ] **Step 4: Run the full raw-servo test file and verify GREEN**

```bash
.venv_inference/bin/python -m pytest -q gear_sonic/tests/test_base_pose_visual_servo.py
```

Require zero failures.

### Task 5: Capture live calibration and assess distortion

**Files:**
- Generate: `gear_sonic/config/camera_intrinsics.json`
- Generate: `outputs/camera_calibration/<timestamp>/camera_intrinsics.json`
- Generate: four timestamped PNG artifacts and a distortion report.

**Interfaces:**
- Consumes: robot calibration-mode ZMQ packets.
- Produces: canonical calibration and a documented distortion decision.

- [ ] **Step 1: Start the local subscriber before the robot publisher**

Run the one-shot receiver against `192.168.123.164:5555` and wait for its ready
signal before starting the robot calibration launcher.

- [ ] **Step 2: Capture and inspect both complete entries**

Require both RGB-D images, both serials, all pinhole fields, both distortion
models and coefficient lists, depth scales, and alignment targets.

- [ ] **Step 3: Compute the distortion report**

Sample the image grid, use model-correct correction appropriate to the captured
model, and record max/p95 pixel-equivalent deviation plus lateral error at 0.8
m and 1.5 m. If material, add a failing correction test before implementing
distortion-aware deprojection; otherwise preserve the pinhole path and record
why.

- [ ] **Step 4: Stop calibration mode and start normal mode**

Confirm no unrelated process occupies port 5555 before either start. Stop only
the calibration process started by this task.

### Task 6: End-to-end verification and documentation

**Files:**
- Modify: `docs/base_pose_adjustment.md`
- Update generated distortion report if needed.

- [ ] **Step 1: Run local verification**

```bash
python3 -m py_compile \
  gear_sonic/camera/calibration.py \
  gear_sonic/scripts/capture_camera_calibration.py \
  gear_sonic/utils/inference/base_pose.py \
  gear_sonic/utils/inference/object_nav.py \
  gear_sonic/utils/inference/base_pose_visual_servo.py
.venv_inference/bin/python -m pytest -q \
  gear_sonic/tests/test_camera_rgbd_protocol.py \
  gear_sonic/tests/test_camera_calibration.py \
  gear_sonic/tests/test_base_pose_adjustment.py \
  gear_sonic/tests/test_object_nav_codex.py \
  gear_sonic/tests/test_base_pose_visual_servo.py
```

- [ ] **Step 2: Verify a live normal packet**

Decode one robot packet and assert its image keys contain both RGB-D pairs and
both wrist RGB streams, while `camera_info` is empty or absent.

- [ ] **Step 3: Verify repository state**

Run `git diff --check`, inspect local and robot diffs, and document the exact
commands needed for future one-shot recalibration.

- [ ] **Step 4: Commit intentional local changes**

Commit source, tests, active calibration, design/plan, and documentation while
leaving timestamped output artifacts untracked unless repository policy says
otherwise.
