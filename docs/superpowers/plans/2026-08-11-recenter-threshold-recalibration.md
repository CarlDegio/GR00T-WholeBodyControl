# Recenter Threshold Recalibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recalibrate horizontal and vertical visibility recovery using bbox center x and visible bottom edge y2 so healthy g1/g3 first frames do not trigger.

**Architecture:** Keep the existing `VisualServoController` state machine and replace only its guard/recovery predicates and constants. Preserve transition ordering, commands, phase-resume state, diagnostics, and target-ID behavior.

**Tech Stack:** Python 3.10, pytest, NumPy, JSONL diagnostics.

## Global Constraints

- Horizontal entry: bbox center x outside 20%--80% image width.
- Horizontal recovery: bbox center x inside 25%--75% for three frames.
- Vertical entry: bbox bottom edge `y2 < 75 px`.
- Vertical recovery: bbox bottom edge `y2 >= 110 px` for three frames.
- Latest g1 and g3 initial bboxes must not enter either recenter state.
- Preserve transition-frame zero, `vx=+0.30` vertical recovery, vertical priority, resume phases, and all unrelated workspace changes.
- Do not commit because task files contain pre-existing untracked work.

---

### Task 1: Guard and recovery behavior

**Files:**
- Modify: `gear_sonic/tests/test_base_pose_visual_servo.py`
- Modify: `gear_sonic/utils/inference/base_pose_visual_servo.py`

**Interfaces:**
- Consumes: `RawServoObservation.target_bbox_xyxy`, `image_width`.
- Produces: `_visibility_guarded`, `_target_center_recovered`, `_vertical_visibility_guarded`, and `_target_vertical_recovered` with approved semantics.

- [ ] **Step 1: Write failing boundary and real-frame tests**

Add tests that use literal g1/g3 bboxes and exact 20%/80%, 25%/75%, 75 px, and 110 px boundaries. Update existing vertical and simultaneous-guard fixtures so they express the approved bottom-edge behavior.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
./.venv_inference/bin/python -m pytest gear_sonic/tests/test_base_pose_visual_servo.py -k 'latest_g1 or latest_g3 or guard_boundary or recovery_boundary' -q
```

Expected: the new real-frame and boundary tests fail under the old edge/center predicates.

- [ ] **Step 3: Implement minimal predicate changes**

Use horizontal center x with entry fractions 0.20/0.80 and recovery fractions 0.25/0.75. Use bbox `y2` with fixed pixel thresholds 75.0 and 110.0. Do not alter state transitions or command generation.

- [ ] **Step 4: Run the complete visual-servo module tests**

```bash
./.venv_inference/bin/python -m pytest \
  gear_sonic/tests/test_base_pose_visual_servo.py \
  gear_sonic/tests/test_base_pose_visual_servo_diagnostics.py -q
```

### Task 2: Documentation and regression verification

**Files:**
- Modify: `docs/base_pose_adjustment.md`

**Interfaces:**
- Documents the exact guard and recovery metrics used by the controller.

- [ ] **Step 1: Update operator documentation**

Replace the old outer-8%, 43%--57%, center-18%, and center-23% descriptions with center 20%/80%, recovery 25%--75%, bottom edge 75 px, and bottom edge 110 px.

- [ ] **Step 2: Run syntax, full regression, and diff checks**

```bash
./.venv_inference/bin/python -m py_compile \
  gear_sonic/utils/inference/base_pose_visual_servo.py \
  gear_sonic/tests/test_base_pose_visual_servo.py
./.venv_inference/bin/python -m pytest gear_sonic/tests -q
git diff --check
```

Expected: all tests pass and no unrelated files are modified by this task.
