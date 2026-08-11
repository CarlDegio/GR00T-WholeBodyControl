# Visibility-Guarded Yaw Servo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add visibility-guarded yaw/recenter/translation phases and complete per-frame telemetry plus annotated images to the raw YOLOE base servo.

**Architecture:** Extend the deterministic controller with an explicit `ServoPhase` state machine and phase-dependent observation requirements. Carry primitive per-frame perception snapshots through the existing worker event queue, then let the runtime combine them with controller state and the final command in a focused diagnostics writer.

**Tech Stack:** Python 3.10, dataclasses/Enum, NumPy, OpenCV, JSONL, pytest, existing YOLOE/BoT-SORT worker and SONIC Planner relay.

## Global Constraints

- Initial phase is `YAW_ALIGN`; coarse yaw is capped at 0.25 rad/s.
- Target boxes crossing 8/92 percent image-edge guards immediately stop yaw and enter `RECENTER`.
- Three frames centered inside the 43/57 percent recovery band resume the interrupted yaw phase.
- Fine yaw is capped at 0.10 rad/s and locks after three frames within 4 degrees.
- `TRANSLATE_TARGET` requires only target tracking/depth and completes after five frames within 0.04 m forward and 0.03 m lateral error.
- A missing required observation emits zero velocity immediately and terminates on the fourth consecutive frame.
- Every processed frame writes one flushed JSONL record and one annotated JPEG.
- Preserve all existing speed, elapsed-time, cumulative-motion, camera-watchdog, track-ID, and operator-stop limits.

---

### Task 1: Visibility-guarded controller phases

**Files:**
- Modify: `gear_sonic/utils/inference/base_pose_visual_servo.py`
- Test: `gear_sonic/tests/test_base_pose_visual_servo.py`

**Interfaces:**
- Produces: `ServoPhase`, `VisualServoController.phase`, `VisualServoController.resume_phase`, `VisualServoController.table_required`, and phase-aware `update()`/`note_invalid()` behavior.
- Extends: `RawServoObservation` with `target_bbox_xyxy`; changes `table` and `surface_track_id` to optional values.

- [ ] **Step 1: Write failing controller tests**

Add focused tests that construct observations with explicit target boxes and prove coarse yaw, both edge guards, three-frame recenter recovery, fine-yaw cap/lock, target-only translation, five-frame completion, and immediate-zero/fourth-frame termination behavior.

```python
def test_right_guard_stops_yaw_and_enters_recenter():
    controller = VisualServoController()
    controller.reset(1.0)
    command = controller.update(
        observation(yaw=math.radians(20), bbox=(400, 100, 600, 300)), now=1.1
    )
    assert command.velocity == (0.0, 0.0, 0.0)
    assert controller.phase is ServoPhase.RECENTER
    assert controller.resume_phase is ServoPhase.YAW_ALIGN


def test_translate_phase_accepts_missing_table_and_finishes_in_five_frames():
    controller = VisualServoController()
    controller.reset(1.0)
    controller.phase = ServoPhase.TRANSLATE_TARGET
    for index in range(5):
        command = controller.update(
            observation(table=None, bbox=(260, 100, 380, 300)),
            now=1.1 + index * 0.1,
        )
    assert command.velocity == (0.0, 0.0, 0.0)
    assert controller.terminal_reason == "aligned"
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
./.venv_inference/bin/python -m pytest \
  gear_sonic/tests/test_base_pose_visual_servo.py \
  -k 'phase or guard or recenter or missing_table' -v
```

Expected: failures because `ServoPhase`, target boxes, and phase-dependent logic do not exist.

- [ ] **Step 3: Implement the minimal state machine**

Add `ServoPhase(str, Enum)` and implement the approved transition thresholds. Keep filtered geometry values and existing slew/limit integrators. Make `note_invalid()` assign a zero command on the first invalid frame while retaining its four-frame terminal threshold.

- [ ] **Step 4: Run focused and existing controller tests**

Run:

```bash
./.venv_inference/bin/python -m pytest \
  gear_sonic/tests/test_base_pose_visual_servo.py -v
```

Expected: all visual-servo controller tests pass.

### Task 2: Per-frame diagnostic contracts and rendering

**Files:**
- Create: `gear_sonic/utils/inference/base_pose_visual_servo_diagnostics.py`
- Create: `gear_sonic/tests/test_base_pose_visual_servo_diagnostics.py`

**Interfaces:**
- Produces immutable `DetectionFrameData` and `ServoFrameData` dataclasses.
- Produces `FrameDiagnosticsWriter(output_dir: Path)` with
  `write(frame, *, controller_state, command) -> tuple[Path, Path]`.
- Produces `detection_frame_data(instance) -> DetectionFrameData | None` in the worker-facing module or an equivalent primitive adapter without circular imports.

- [ ] **Step 1: Write failing diagnostics tests**

Test one valid and one invalid frame. Assert that two calls create exactly two JSONL objects and `frames/frame_000000.jpg` plus `frame_000001.jpg`; assert required keys remain present with JSON null values on the invalid frame; decode both JPEGs with OpenCV.

```python
def test_writer_records_every_valid_and_invalid_frame(tmp_path):
    writer = FrameDiagnosticsWriter(tmp_path)
    writer.write(valid_frame(0), controller_state=state(), command=ServoCommand(0, 0, .1))
    writer.write(invalid_frame(1), controller_state=state(), command=ServoCommand(0, 0, 0))
    rows = [json.loads(line) for line in (tmp_path / "frame_telemetry.jsonl").read_text().splitlines()]
    assert [row["frame_index"] for row in rows] == [0, 1]
    assert rows[1]["target"] is None
    assert cv2.imread(str(tmp_path / "frames/frame_000001.jpg")) is not None
```

- [ ] **Step 2: Run diagnostics tests and verify RED**

Run:

```bash
./.venv_inference/bin/python -m pytest \
  gear_sonic/tests/test_base_pose_visual_servo_diagnostics.py -v
```

Expected: import failure because the diagnostics module does not exist.

- [ ] **Step 3: Implement serialization and overlays**

Write one flushed line per call. Render RGB as BGR, alpha-blend target/table masks, draw detection boxes and center/guard/recovery lines, and add phase/error/depth/command text. Raise `OSError` if JSONL or JPEG output cannot be completed.

- [ ] **Step 4: Run diagnostics tests and verify GREEN**

Run the same test command and expect all diagnostics tests to pass.

### Task 3: Worker/runtime integration and phase-dependent perception

**Files:**
- Modify: `gear_sonic/utils/inference/base_pose_visual_servo.py`
- Modify: `gear_sonic/tests/test_base_pose_visual_servo.py`
- Modify: `gear_sonic/tests/test_base_pose_visual_servo_diagnostics.py`

**Interfaces:**
- Extends `RawServoEvent` with `frame: ServoFrameData | None`.
- Worker emits an observation with `table=None` when target geometry is valid but the table is absent or invalid.
- Runtime owns `FrameDiagnosticsWriter`, writes after command computation, and treats write failures as hard zero-speed terminal failures.

- [ ] **Step 1: Write failing integration tests**

Use fake camera/tracker inputs to prove the worker assigns monotonic frame indices to initialized, valid, target-only, and invalid frames. Feed those events to `RawServoRuntime` and assert one telemetry/JPEG pair per accepted event, table loss is allowed only in `TRANSLATE_TARGET`, and missing required observations immediately publish zero.

- [ ] **Step 2: Run integration tests and verify RED**

Run:

```bash
./.venv_inference/bin/python -m pytest \
  gear_sonic/tests/test_base_pose_visual_servo.py \
  gear_sonic/tests/test_base_pose_visual_servo_diagnostics.py \
  -k 'worker or runtime or frame' -v
```

Expected: failures because events do not carry frame diagnostics and the worker currently rejects every missing table.

- [ ] **Step 3: Implement worker and runtime handoff**

Compute target and table geometry independently. Preserve the expected IDs, masks, boxes, confidence, structured geometry error, RGB, and frame index in `ServoFrameData`. Size the runtime queue to at least `ceil(raw_servo_hz * raw_max_run_s) + 8`, then finalize each accepted frame after controller command calculation. Include phase and transition fields in `raw_servo_events.jsonl`.

- [ ] **Step 4: Make diagnostic failure fail safe**

Catch writer errors in `accept_event()`, set a hard controller failure, publish the normal three-message zero stop sequence, and record the failure through the existing logger when the JSONL itself is unavailable.

- [ ] **Step 5: Run both focused modules**

Run:

```bash
./.venv_inference/bin/python -m pytest \
  gear_sonic/tests/test_base_pose_visual_servo.py \
  gear_sonic/tests/test_base_pose_visual_servo_diagnostics.py -v
```

Expected: all tests pass.

### Task 4: Documentation and complete verification

**Files:**
- Modify: `docs/base_pose_adjustment.md`
- Verify: all files from Tasks 1-3.

**Interfaces:**
- Documents: phase order, exact thresholds, phase-dependent table requirement, immediate-zero loss behavior, and new artifacts.

- [ ] **Step 1: Update operator documentation**

Document `YAW_ALIGN -> RECENTER -> YAW_TRIM -> TRANSLATE_TARGET`, the repeatable visibility intervention, and the locations of `frame_telemetry.jsonl` and `frames/*.jpg`.

- [ ] **Step 2: Run syntax and focused verification**

```bash
./.venv_inference/bin/python -m py_compile \
  gear_sonic/utils/inference/base_pose_visual_servo.py \
  gear_sonic/utils/inference/base_pose_visual_servo_diagnostics.py
./.venv_inference/bin/python -m pytest \
  gear_sonic/tests/test_base_pose_visual_servo.py \
  gear_sonic/tests/test_base_pose_visual_servo_diagnostics.py -q
```

- [ ] **Step 3: Run the full regression suite**

```bash
./.venv_inference/bin/python -m pytest gear_sonic/tests -q
git diff --check
```

- [ ] **Step 4: Run a no-cloud saved-frame diagnostic smoke test**

Use the latest saved 640x480 RGB/depth frame with synthetic `TrackedInstance`
objects to produce one telemetry line and annotated JPEG. Verify both artifacts
are nonempty and that the JPEG decodes; do not invoke Codex/Qwen or a live robot.

- [ ] **Step 5: Inspect final status and diff**

Confirm only the intended controller, diagnostics, tests, and documentation are
newly changed by this plan, and preserve all pre-existing user modifications.
