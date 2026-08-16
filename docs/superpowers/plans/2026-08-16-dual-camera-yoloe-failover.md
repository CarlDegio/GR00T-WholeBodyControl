# Dual-Camera BasePose YOLOE Failover Implementation Plan

> **For agentic workers:** Execute this plan task-by-task with test-first red/green cycles. Keep the existing dirty-worktree changes intact and stage only files owned by this feature.

**Goal:** Add a `dual_raw_yoloe_servo` mode that grounds both head and chest RGB-D views, safely switches YOLOE between them with bounded retries, and holds zero velocity in Planner mode during every switch.

**Architecture:** A one-socket dual RGB-D reader returns independent per-stream results from each composed packet. A pure failover coordinator owns immutable initial references, five-frame latest references, and the `A -> B initial -> A latest` attempt sequence; a dedicated dual worker reuses the existing YOLOE tracker and raw-servo runtime event path. The existing `raw_yoloe_servo` worker remains unchanged.

**Tech Stack:** Python 3.10+, NumPy, OpenCV, ZeroMQ, msgpack, YOLOE/Ultralytics, pytest, tyro dataclass CLI.

## Global Constraints

- Preserve the existing `raw_yoloe_servo` behavior.
- Use one composed-camera subscription and decode `ego_view` and `chest_view` independently.
- Use saved per-stream intrinsics; chest pitch defaults to exactly `-3.0` degrees.
- Run both target/table grounding pairs concurrently at navigation start.
- Prefer head when both initial references validate.
- Update a coherent target-plus-table latest reference only every five frames by default.
- Give each YOLOE matching attempt 30 consecutive invalid frames.
- Hold zero velocity without leaving Planner mode throughout switching.
- Preserve one generation across switches and reject stale attempt events.
- End only after `A` fails, `B` initial-reference matching fails, and `A` latest-reference matching fails, unless a non-perception terminal condition occurs first.

---

### Task 1: One-socket dual RGB-D decoding

**Files:**
- Modify: `gear_sonic/utils/inference/base_pose.py`
- Modify: `gear_sonic/tests/test_base_pose_adjustment.py`

**Interfaces:**
- Produces: `DualRGBDCapture.snapshots: Mapping[str, AlignedRGBDSnapshot]` and `DualRGBDCapture.errors: Mapping[str, str]`.
- Produces: `DualAlignedRGBDCamera(host, port, stream_names, timeout_ms, calibration_path)` with `decode_payload`, `capture`, and `close`.

- [ ] **Step 1: Write failing decoder tests**

Add tests constructing one serialized `ImageMessageSchema` with both RGB-D pairs and saved calibration entries. Assert both snapshots decode from one payload. Corrupt only `chest_view_depth` and assert `ego_view` remains in `snapshots` while chest appears in `errors`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
.venv_inference/bin/python -m pytest -q gear_sonic/tests/test_base_pose_adjustment.py -k dual_rgbd
```

Expected: import failure because `DualAlignedRGBDCamera` and `DualRGBDCapture` do not exist.

- [ ] **Step 3: Implement the shared decoder and one socket**

Factor the existing per-stream payload validation into a private helper used by both camera classes. `DualAlignedRGBDCamera.decode_payload` must catch `BasePoseCameraError` separately for each stream and never let one stream remove a valid sibling result. `capture` reads and unpacks one ZMQ message, filters per-stream duplicate timestamps, and returns as soon as at least one requested stream is fresh.

- [ ] **Step 4: Run focused and existing camera tests**

```bash
.venv_inference/bin/python -m pytest -q gear_sonic/tests/test_base_pose_adjustment.py -k 'dual_rgbd or saved_calibration or dynamic_ego_rgbd'
```

Require all selected tests to pass.

### Task 2: Reference bank and failover coordinator

**Files:**
- Create: `gear_sonic/utils/inference/base_pose_dual_visual_servo.py`
- Create: `gear_sonic/tests/test_base_pose_dual_visual_servo.py`

**Interfaces:**
- Produces: `DualCameraReference(stream_name, rgb, target_prompt, target_bbox, table_bboxes, camera_timestamp, kind)`.
- Produces: `DualCameraAttempt(attempt_id, live_stream, reference, stage, origin_stream)`.
- Produces: `DualCameraFailoverCoordinator(stream_names, initial_references)` with `start`, `save_latest`, `mark_success`, and `advance_after_failure`.

- [ ] **Step 1: Write failing pure state-machine tests**

Cover head priority, only-one-initial-reference selection, alternate camera using its own initial reference, cross-camera fallback when it lacks one, return to origin latest reference, initial fallback when latest is absent, termination after the origin retry, and success resetting the cycle for repeated future switches.

- [ ] **Step 2: Verify RED**

```bash
.venv_inference/bin/python -m pytest -q gear_sonic/tests/test_base_pose_dual_visual_servo.py -k coordinator
```

Expected: module or symbol import failure.

- [ ] **Step 3: Implement immutable references and coordinator**

The coordinator must reject duplicate stream names, references with no table boxes, and references whose source image/boxes are incomplete. It selects references using this exact order:

```python
# active A failed
B live + (B.initial if present else A.initial)
# that attempt failed
A live + (A.latest if present else A.initial)
# that attempt failed
None
```

`mark_success` makes the successful live stream the origin of a new future cycle and clears the current fallback stage.

- [ ] **Step 4: Run coordinator tests and verify GREEN**

Run the focused command and require zero failures.

### Task 3: Dual grounding, calibration, and five-frame reference capture

**Files:**
- Modify: `gear_sonic/utils/inference/base_pose_dual_visual_servo.py`
- Modify: `gear_sonic/utils/inference/base_pose_visual_servo.py`
- Modify: `gear_sonic/tests/test_base_pose_dual_visual_servo.py`

**Interfaces:**
- Produces: `dual_calibrations_from_config(config) -> Mapping[str, RawServoCalibration]`.
- Produces: `ground_dual_raw_servo_references(config, captures, output_dir, client_factory) -> (references, errors)`.
- Produces: `LatestReferenceGate(interval_frames=5, ...)` that accepts only a same-frame target/table/geometry candidate.

- [ ] **Step 1: Write failing grounding and reference tests**

Use a four-party barrier in fake clients to prove four calls overlap. Assert independent per-camera failure, head selection when both succeed, correct use of saved head/chest intrinsics, exact chest pitch `-3.0`, pixel-to-normalized tracker-box conversion, and latest-reference updates only on frame indices divisible by five. Reject candidates with missing table or table geometry.

- [ ] **Step 2: Verify RED**

```bash
.venv_inference/bin/python -m pytest -q gear_sonic/tests/test_base_pose_dual_visual_servo.py -k 'ground or calibration or latest_reference'
```

- [ ] **Step 3: Implement the dual preparation helpers**

Allow `ground_raw_servo_references` to receive an explicit `RawServoCalibration` while retaining its current default. Run one existing two-request grounding operation per valid camera in a two-worker outer executor, giving four concurrent backend calls. Store per-camera prompts/results under separate subdirectories and write an eligibility summary.

At the existing five-frame cadence, save an in-memory owned RGB copy plus normalized target/table boxes only when target and table geometries are both valid and the existing confidence/visibility gate accepts the target.

- [ ] **Step 4: Run the full new test file**

Require all preparation tests to pass.

### Task 4: Dual worker and 30-frame attempt behavior

**Files:**
- Modify: `gear_sonic/utils/inference/base_pose_dual_visual_servo.py`
- Modify: `gear_sonic/tests/test_base_pose_dual_visual_servo.py`

**Interfaces:**
- Produces: `run_dual_raw_servo_worker(config, requests, events, gate, stop_event, ...)` compatible with `RawServoRuntime` queues.
- Emits: `switching`, `initialized`, `observation`, `invalid`, and final `error` `RawServoEvent` values with attempt metadata in `details`.

- [ ] **Step 1: Write failing worker tests with fake camera and tracker**

Assert that each attempt calls `tracker.start` once with the chosen reference, frames 1 through 29 emit soft invalid events, frame 30 emits `switching` and changes the live stream, a valid frame emits `initialized`, and the complete three-attempt failure sequence ends in one final error. Assert a successful alternate attempt resets the cycle and later failure switches back symmetrically.

- [ ] **Step 2: Verify RED**

```bash
.venv_inference/bin/python -m pytest -q gear_sonic/tests/test_base_pose_dual_visual_servo.py -k worker
```

- [ ] **Step 3: Implement the worker loop**

Maintain a generation-global monotonically increasing frame index and a monotonically increasing attempt ID. Install one reference at attempt start, then match up to 30 fresh live frames. A complete observation requires target, table, target depth, and table geometry. Soft invalid frames publish zero-capable invalid events for counts 1 through 29; count 30 advances the coordinator without sending a controller-terminal invalid event. Camera/protocol/YOLOE hard failures advance immediately. Final exhaustion emits `error` and cancels the generation.

- [ ] **Step 4: Verify worker GREEN**

Run the worker-focused selection and then the full new test file.

### Task 5: Runtime switching safety and diagnostic provenance

**Files:**
- Modify: `gear_sonic/utils/inference/base_pose_visual_servo.py`
- Modify: `gear_sonic/utils/inference/base_pose_visual_servo_diagnostics.py`
- Modify: `gear_sonic/tests/test_base_pose_visual_servo.py`
- Modify: `gear_sonic/tests/test_base_pose_dual_visual_servo.py`

**Interfaces:**
- `RawServoRuntime.accept_event` accepts `switching` and permits `initialized` from `switching` phase.
- `DetectionFrameData` gains optional `camera_stream`, `attempt_id`, `failover_stage`, `reference_source_stream`, and `reference_kind` fields.

- [ ] **Step 1: Write failing runtime safety tests**

Start a runtime, accept an initial observation, then a `switching` event. Assert immediate zero publication, phase `switching`, unchanged generation, active gate, reset controller phase, and repeated `publish_due` zero `hold` messages. Assert stale events from a previous attempt cannot resume motion and a current-attempt `initialized` event can.

- [ ] **Step 2: Verify RED**

```bash
.venv_inference/bin/python -m pytest -q gear_sonic/tests/test_base_pose_dual_visual_servo.py -k runtime
```

- [ ] **Step 3: Implement switching event handling and metadata**

Handle switching before normal observation logic, drain queued old observations, reset the controller, keep the generation gate active, and schedule the next zero heartbeat. Track the current attempt ID in runtime and reject lower IDs. Keep a separate navigation start time so controller resets cannot extend `raw_max_run_s`.

Serialize the five provenance fields at the top level of every diagnostic frame record and preserve defaults for the single-camera worker.

- [ ] **Step 4: Run runtime and existing diagnostic tests**

```bash
.venv_inference/bin/python -m pytest -q gear_sonic/tests/test_base_pose_dual_visual_servo.py gear_sonic/tests/test_base_pose_visual_servo.py -k 'runtime or diagnostic'
```

### Task 6: Mode configuration, launcher, and documentation

**Files:**
- Modify: `gear_sonic/utils/inference/base_pose.py`
- Modify: `gear_sonic/scripts/base_pose_planner.py`
- Modify: `gear_sonic/scripts/launch_inference.py`
- Modify: `gear_sonic/tests/test_base_pose_dual_visual_servo.py`
- Modify: `gear_sonic/tests/test_lavira_planner.py`
- Modify: `docs/base_pose_adjustment.md`

**Interfaces:**
- Adds mode string `dual_raw_yoloe_servo` to BasePose types, CLI dispatch, prerequisite selection, and launcher command construction.
- Adds dual head/chest stream and chest extrinsic flags; chest pitch default is `-3.0`.

- [ ] **Step 1: Write failing launch tests**

Assert the new mode uses direct RGB-D without LingBot, carries both stream names and chest pitch, starts the raw-servo process and existing manual Planner relay path, and leaves old raw mode commands unchanged.

- [ ] **Step 2: Verify RED**

```bash
.venv_inference/bin/python -m pytest -q gear_sonic/tests/test_base_pose_dual_visual_servo.py gear_sonic/tests/test_lavira_planner.py -k dual_raw_yoloe
```

- [ ] **Step 3: Wire the mode and document operation**

Route only `dual_raw_yoloe_servo` to `run_dual_raw_servo_worker`; route the original mode to its original worker. Extend the direct-camera and manual-keyboard conditions to include both raw modes. Document the command, reference priority, 30-frame stages, five-frame latest-reference cadence, chest calibration, and zero-velocity Planner-mode switching.

- [ ] **Step 4: Run launch tests and verify GREEN**

Run the focused command and require all selected tests to pass.

### Task 7: Full verification

**Files:**
- Verify all modified source, tests, plans, and documentation.

- [ ] **Step 1: Compile Python sources**

```bash
python3 -m py_compile \
  gear_sonic/utils/inference/base_pose.py \
  gear_sonic/utils/inference/base_pose_visual_servo.py \
  gear_sonic/utils/inference/base_pose_dual_visual_servo.py \
  gear_sonic/utils/inference/base_pose_visual_servo_diagnostics.py \
  gear_sonic/scripts/base_pose_planner.py \
  gear_sonic/scripts/launch_inference.py
```

- [ ] **Step 2: Run focused and regression suites**

```bash
.venv_inference/bin/python -m pytest -q \
  gear_sonic/tests/test_base_pose_dual_visual_servo.py \
  gear_sonic/tests/test_base_pose_visual_servo.py \
  gear_sonic/tests/test_base_pose_adjustment.py \
  gear_sonic/tests/test_lavira_planner.py \
  gear_sonic/tests/test_lavira_reasan_contract.py
```

- [ ] **Step 3: Check patch integrity**

Run `git diff --check`, inspect every feature-owned diff, and confirm unrelated dirty-worktree files were neither reverted nor staged.

- [ ] **Step 4: Report hardware boundary**

If both live cameras and the YOLOE GPU are unavailable, explicitly report that unit/integration tests passed but the live dual RGB-D navigation smoke test remains a hardware-side validation step.
