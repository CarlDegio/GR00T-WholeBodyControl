# Raw Servo Tracking-Loss Grace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep base-pose raw visual-servo runs recoverable for 29 consecutive missing target or required-table frames and terminate on the 30th.

**Architecture:** Retain the controller's existing shared consecutive soft-invalid counter and zero-command safety behavior. Change its default terminal threshold and inclusive boundary; the worker, runtime, hard-failure handling, and phase-specific table requirement remain unchanged.

**Tech Stack:** Python 3, pytest, NumPy, the existing `VisualServoController` state machine.

## Global Constraints

- The threshold is exactly 30 visual tracking frames, not three wall-clock seconds.
- Frames 1 through 29 publish zero velocity without ending the active run; frame 30 uses the existing tracking-lost terminal path.
- A usable observation resets the loss streak and allows command generation to resume.
- Missing table geometry counts only in phases that require the table edge.
- Existing hard invalid events remain immediately terminal.
- Preserve all pre-existing user changes. Both implementation files are currently untracked as a whole, so do not commit or stage their unrelated contents.

---

### Task 1: Extend the visual tracking-loss grace window

**Files:**
- Modify: `gear_sonic/tests/test_base_pose_visual_servo.py:669`
- Modify: `gear_sonic/utils/inference/base_pose_visual_servo.py:1126-1145,1488-1494`

**Interfaces:**
- Consumes: `VisualServoController.update(observation, *, now) -> ServoCommand` and `VisualServoController.note_invalid(reason, *, hard, now, integrated=False) -> ServoCommand`.
- Produces: a default 30-frame soft tracking-loss threshold with inclusive termination on frame 30; no public API additions.

- [ ] **Step 1: Replace the old four-frame expectation with a failing 30-frame boundary test**

```python
def test_controller_zeros_through_twenty_nine_missing_frames_and_stops_on_thirtieth() -> None:
    controller = VisualServoController()
    controller.reset(1.0)
    controller.phase = ServoPhase.TRANSLATE_TARGET
    moving = controller.update(observation(forward=0.9), now=1.1)
    assert moving.vx > 0.0

    for index in range(29):
        stopped = controller.note_invalid(
            "target absent", hard=False, now=1.2 + 0.1 * index
        )
        assert stopped.velocity == (0.0, 0.0, 0.0)
        assert not controller.terminal

    stopped = controller.note_invalid("target absent", hard=False, now=4.1)
    assert stopped.velocity == (0.0, 0.0, 0.0)
    assert controller.terminal_reason == "tracking lost: target absent"
```

- [ ] **Step 2: Add failing recovery and required-table tests**

```python
def test_valid_target_resets_missing_frame_streak_and_resumes_motion() -> None:
    controller = VisualServoController()
    controller.reset(1.0)
    controller.phase = ServoPhase.TRANSLATE_TARGET
    controller.update(observation(forward=0.9), now=1.1)

    for index in range(3):
        controller.note_invalid(
            "missing tracked target", hard=False, now=1.2 + 0.1 * index
        )

    resumed = controller.update(observation(forward=0.9), now=1.5)
    assert resumed.vx > 0.0
    assert controller.invalid_frames == 0

    for index in range(29):
        held = controller.note_invalid(
            "missing tracked target", hard=False, now=1.6 + 0.1 * index
        )
        assert held.velocity == (0.0, 0.0, 0.0)
        assert not controller.terminal
    assert controller.invalid_frames == 29


def test_missing_required_table_holds_until_thirtieth_frame() -> None:
    controller = VisualServoController()
    controller.reset(1.0)

    for index in range(29):
        held = controller.update(
            observation(include_table=False), now=1.1 + 0.1 * index
        )
        assert held.velocity == (0.0, 0.0, 0.0)
        assert not controller.terminal

    stopped = controller.update(observation(include_table=False), now=4.0)
    assert stopped.velocity == (0.0, 0.0, 0.0)
    assert controller.terminal_reason == "tracking lost: missing table geometry"
```

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```bash
python -m pytest \
  gear_sonic/tests/test_base_pose_visual_servo.py::test_controller_zeros_through_twenty_nine_missing_frames_and_stops_on_thirtieth \
  gear_sonic/tests/test_base_pose_visual_servo.py::test_valid_target_resets_missing_frame_streak_and_resumes_motion \
  gear_sonic/tests/test_base_pose_visual_servo.py::test_missing_required_table_holds_until_thirtieth_frame \
  -q
```

Expected: FAIL because the current controller becomes terminal on the fourth consecutive soft invalid frame.

- [ ] **Step 4: Implement the minimal inclusive 30-frame threshold**

In `VisualServoController.__init__`, change:

```python
missing_tolerance_frames: int = 30,
```

In `VisualServoController.note_invalid`, change the terminal comparison to:

```python
if self.invalid_frames >= self.missing_tolerance_frames:
    return self._stop(f"tracking lost: {reason}")
```

- [ ] **Step 5: Run the focused tests and verify GREEN**

Run the exact pytest command from Step 3.

Expected: all three tests PASS.

- [ ] **Step 6: Run related regression tests**

Run:

```bash
python -m pytest gear_sonic/tests/test_base_pose_visual_servo.py -q
```

Expected: the complete raw visual-servo test module passes, including immediate hard-loss termination, first-invalid-frame zero publication, and translation without table geometry.

- [ ] **Step 7: Verify syntax and inspect the scoped diff**

Run:

```bash
python -m py_compile \
  gear_sonic/utils/inference/base_pose_visual_servo.py \
  gear_sonic/tests/test_base_pose_visual_servo.py
git diff --no-index /dev/null gear_sonic/utils/inference/base_pose_visual_servo.py
git diff --no-index /dev/null gear_sonic/tests/test_base_pose_visual_servo.py
```

Expected: compilation succeeds. Because both files predate this task but are untracked, use the recorded patch and exact edited line inspection to confirm only the planned threshold and tests were added; do not stage or commit the full files.
