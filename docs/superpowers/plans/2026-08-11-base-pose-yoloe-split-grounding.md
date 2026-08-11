# Base-Pose YOLOE Split Grounding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ground the task target and every visible table through two independent concurrent model requests before initializing the raw BasePose YOLOE servo.

**Architecture:** Keep the task-target contract local to the raw servo after removing all support-surface requirements. Reuse the single-target prompt, schema, and validator from `tools/yoloe26m/reference_pipeline.py` for the table branch, execute both branches concurrently against one saved RGB frame, and combine one target box with every table box in the two-class YOLOE visual prompt.

**Tech Stack:** Python 3, `concurrent.futures.ThreadPoolExecutor`, Codex/Qwen structured vision clients, NumPy, YOLOE visual prompts, pytest.

## Global Constraints

- The task-target prompt retains its current task, calibration, target selection, normalized bounding-box, safety, and no-motion content while removing support-surface/table requirements.
- The table prompt must equal `tools.yoloe26m.reference_pipeline.build_grounding_prompt("table")` byte for byte.
- Both requests use the same saved initial RGB image but separate configured client instances and run concurrently.
- Preserve every validated table box and assign each one YOLOE class ID `1`; the target box remains class ID `0`.
- Do not initialize YOLOE or publish motion if either grounding branch fails validation.
- Tests must not call live Codex/Qwen, start robot control, load YOLOE weights, or require a live camera.
- Preserve all pre-existing uncommitted work. Do not stage or commit overlapping implementation files unless the user separately requests it.

---

### Task 1: Split grounding contracts and support multi-box matching

**Files:**
- Modify: `gear_sonic/utils/inference/base_pose_visual_servo.py:40-360`
- Modify: `gear_sonic/utils/inference/base_pose_visual_servo.py:420-570`
- Test: `gear_sonic/tests/test_base_pose_visual_servo.py`

**Interfaces:**
- Consumes: `GROUNDING_SCHEMA`, `build_grounding_prompt("table")`, and `validate_grounding_response(value)` from `tools.yoloe26m.reference_pipeline`.
- Produces: target-only `RAW_SERVO_TARGET_SCHEMA` and `RawServoTargetSpec`; `build_raw_servo_table_prompt() -> str`; `validate_raw_servo_table(value) -> tuple[tuple[float, float, float, float], ...]`; multi-box `YoloePersistentTracker.start` and `select_initial_instance`.

- [ ] **Step 1: Write failing target/table contract tests**

Use target-only and table payload fixtures:

```python
def target_payload() -> dict[str, object]:
    return {
        "status": "READY",
        "primary_target": {
            "text_prompt": "blue plastic basket",
            "bbox_2d": [300.0, 250.0, 600.0, 700.0],
        },
        "manipulation_anchor": "basket opening",
        "selection_reason": "large stable task target",
        "confidence": 0.9,
        "limitations": "partially occluded",
    }


def table_payload() -> dict[str, object]:
    return {
        "status": "READY",
        "target": "table",
        "boxes": [
            {"bbox_2d": [25.0, 100.0, 475.0, 900.0], "confidence": 0.91},
            {"bbox_2d": [525.0, 120.0, 975.0, 880.0], "confidence": 0.84},
        ],
        "limitations": "",
    }
```

Assert the target prompt and schema omit `support_surface`, `operation platform`, and `table` while retaining `primary_target`. Assert `build_raw_servo_table_prompt() == build_grounding_prompt("table")`. Assert table validation returns both boxes in input order.

- [ ] **Step 2: Run tests to verify the old combined contract fails**

Run: `pytest -q gear_sonic/tests/test_base_pose_visual_servo.py -k 'target_prompt_and_schema or table_prompt_exactly or table_validator_preserves'`

Expected: failures because the target contract still contains `support_surface` and the table helpers are absent.

- [ ] **Step 3: Implement target-only and shared table contracts**

Import the shared primitives and add:

```python
def build_raw_servo_table_prompt() -> str:
    return build_grounding_prompt("table")


def validate_raw_servo_table(
    value: Mapping[str, Any],
) -> tuple[tuple[float, float, float, float], ...]:
    boxes = validate_grounding_response(value)
    return tuple((box.x1, box.y1, box.x2, box.y2) for box in boxes)
```

Remove `support_surface` from `TARGET_OUTPUT_KEYS`, `RAW_SERVO_TARGET_SCHEMA`, `RawServoTargetSpec`, and `validate_raw_servo_target`. Update `build_raw_servo_target_prompt` minimally: request exactly one primary target, keep all current target/calibration/safety content, make its box wording singular, and make `READY` depend only on the target.

- [ ] **Step 4: Write failing multi-table handoff tests**

Update the tracker test to pass:

```python
surface_bboxes=(
    (0.0, 100.0, 450.0, 900.0),
    (550.0, 100.0, 1000.0, 900.0),
)
```

Require visual prompt class IDs `[0, 1, 1]` and three pixel boxes. Add a selection test with two class-1 tracks where the selected track overlaps only the second table reference box.

- [ ] **Step 5: Run multi-box tests and verify singular interfaces fail**

Run: `pytest -q gear_sonic/tests/test_base_pose_visual_servo.py -k 'tracker_initializes or initial_table_selection'`

Expected: failure because `surface_bboxes` and `grounded_bboxes` are not accepted.

- [ ] **Step 6: Implement multi-box YOLOE handoff and matching**

In `YoloePersistentTracker.start`, reject an empty table-box sequence and build:

```python
normalized_boxes = (target_bbox, *surface_bboxes)
prompt_boxes = np.asarray(
    [normalized_bbox_to_pixels(box, width, height) for box in normalized_boxes],
    dtype=np.float32,
)
prompt_classes = np.asarray([0, *([1] * len(surface_bboxes))], dtype=np.int32)
```

Change `select_initial_instance` to accept `grounded_bboxes`, rank each candidate by its maximum IoU against any normalized reference box, use confidence as the tie-breaker, and preserve the `0.05` minimum maximum-IoU guard. Target callers pass a one-element tuple.

- [ ] **Step 7: Run the focused contract/handoff tests**

Run: `pytest -q gear_sonic/tests/test_base_pose_visual_servo.py -k 'prompt or schema or validator or tracker_initializes or initial_table_selection'`

Expected: all selected tests pass.

- [ ] **Step 8: Check the dirty-file checkpoint**

Run: `git diff --check -- gear_sonic/utils/inference/base_pose_visual_servo.py gear_sonic/tests/test_base_pose_visual_servo.py`

Expected: no whitespace errors. Leave both files unstaged because they contain pre-existing work.

---

### Task 2: Execute independent grounding clients concurrently

**Files:**
- Modify: `gear_sonic/utils/inference/base_pose_visual_servo.py:1060-1340`
- Test: `gear_sonic/tests/test_base_pose_visual_servo.py`

**Interfaces:**
- Consumes: target/table contracts and multi-box tracker interfaces from Task 1.
- Produces: `ground_raw_servo_references(config, rgb_path, output_dir, client_factory=None) -> tuple[RawServoTargetSpec, tuple[tuple[float, float, float, float], ...]]` and distinct target/table artifacts.

- [ ] **Step 1: Write a failing real-concurrency test**

Use a `threading.Barrier(2, timeout=1.0)` inside two fake clients. Call `ground_raw_servo_references` with a factory and assert:

```python
assert len(clients) == 2
assert clients[0] is not clients[1]
assert {Path(call["image_paths"][0]) for call in calls} == {rgb_path}
assert table_call["prompt"] == build_grounding_prompt("table")
assert table_call["schema"] is GROUNDING_SCHEMA
assert table_call["schema_filename"] == "table_schema.json"
assert len(table_bboxes) == 2
```

The barrier makes a sequential implementation fail.

- [ ] **Step 2: Run the concurrency test and verify the helper is absent**

Run: `pytest -q gear_sonic/tests/test_base_pose_visual_servo.py::test_grounding_requests_use_independent_clients_and_run_concurrently`

Expected: failure because `ground_raw_servo_references` is undefined.

- [ ] **Step 3: Implement the executor helper**

Create two separate client instances and submit both request functions to `ThreadPoolExecutor(max_workers=2, thread_name_prefix="raw-servo-grounding")` before awaiting either future. Both calls receive `image_paths=[rgb_path]` and `cwd=output_dir`.

The target branch writes `target_prompt.txt`, calls `RAW_SERVO_TARGET_SCHEMA` with `target_schema.json`, writes `target_result.json`, and validates the target. The table branch writes `table_prompt.txt`, calls the exact shared prompt and `GROUNDING_SCHEMA` with `table_schema.json`, writes `table_result.json`, and validates every table box.

- [ ] **Step 4: Write failing worker handoff and failure-short-circuit tests**

Adapt the existing fake client into a two-instance factory. Capture `tracker.start` arguments and require all table boxes in `surface_bboxes`. Require the YOLOE artifact class IDs to be `[0, 1, 1]` and both result files to exist.

In a second test, return `NOT_FOUND` from the table branch. Assert the worker emits an error event containing `grounding status is NOT_FOUND` and never calls `tracker_factory`.

- [ ] **Step 5: Run worker tests and verify the sequential combined flow fails**

Run: `pytest -q gear_sonic/tests/test_base_pose_visual_servo.py -k 'worker_keeps_target or grounding_failure_prevents_tracker'`

Expected: failures because the worker makes one combined request and supplies one table box.

- [ ] **Step 6: Integrate the concurrent helper into the worker**

Remove the cached single `client`. For each generation, call `ground_raw_servo_references` after saving the initial RGB/depth images, check the generation gate after both results validate, then start the tracker with one target box and all table boxes. Pass `(spec.target_bbox,)` to target selection and all table boxes to class-1 selection. Keep persistent IDs, observations, controller phases, and subsequent tracking unchanged.

Extend `tracking_ids.json` and initialized-event details with normalized table boxes while retaining `reference_boxes_xyxy` and `reference_class_ids` in `yoloe_reference_prompt.json`.

- [ ] **Step 7: Run the complete raw-servo test file**

Run: `pytest -q gear_sonic/tests/test_base_pose_visual_servo.py`

Expected: all tests pass with only fake clients/camera/tracker.

- [ ] **Step 8: Check the dirty-file checkpoint**

Run: `git diff --check -- gear_sonic/utils/inference/base_pose_visual_servo.py gear_sonic/tests/test_base_pose_visual_servo.py`

Expected: no whitespace errors; do not stage the overlapping files.

---

### Task 3: Update diagnostics documentation and verify regressions

**Files:**
- Modify: `docs/base_pose_adjustment.md:110-180`
- Test: `gear_sonic/tests/test_base_pose_visual_servo.py`
- Test: `gear_sonic/tests/test_base_pose_visual_servo_diagnostics.py`
- Test: `gear_sonic/tests/test_lavira_planner.py`

**Interfaces:**
- Consumes: two-branch artifact names and all-box YOLOE reference data from Tasks 1 and 2.
- Produces: accurate operator documentation and final verification evidence.

- [ ] **Step 1: Update raw YOLOE documentation**

State that two independent Codex or Qwen requests run concurrently against the same initial RGB. The task branch grounds only the primary target; the table branch uses the exact auto-reference single-target prompt with target `table` and preserves every table box. Document `target_prompt.txt`, `target_schema.json`, `target_result.json`, `table_prompt.txt`, `table_schema.json`, `table_result.json`, and the all-box `yoloe_reference_prompt.json`. State that either branch failing prevents tracking and motion.

- [ ] **Step 2: Run focused regression tests**

Run: `pytest -q gear_sonic/tests/test_base_pose_visual_servo.py gear_sonic/tests/test_base_pose_visual_servo_diagnostics.py gear_sonic/tests/test_lavira_planner.py`

Expected: all tests pass.

- [ ] **Step 3: Run syntax and formatting checks**

Run: `python3 -m py_compile gear_sonic/utils/inference/base_pose_visual_servo.py gear_sonic/tests/test_base_pose_visual_servo.py`

Run: `git diff --check -- gear_sonic/utils/inference/base_pose_visual_servo.py gear_sonic/tests/test_base_pose_visual_servo.py docs/base_pose_adjustment.md`

Expected: both commands exit zero and `git diff --check` prints nothing.

- [ ] **Step 4: Inspect final scope without staging user work**

Run: `git status --short` and inspect diffs for `gear_sonic/utils/inference/base_pose_visual_servo.py`, `gear_sonic/tests/test_base_pose_visual_servo.py`, and `docs/base_pose_adjustment.md`.

Expected: changes are limited to split grounding, all-table YOLOE handoff, tests, and documentation on top of the user's existing raw-servo work. Do not commit the overlapping dirty files.
