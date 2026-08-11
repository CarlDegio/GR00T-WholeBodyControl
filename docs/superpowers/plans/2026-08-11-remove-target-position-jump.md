# Remove Base-Pose Target Position Jump Handling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every valid raw YOLOE target-position observation bypass `invalid_frames` so `target position jump` can never terminate base-pose control.

**Architecture:** Delete the controller's inter-frame displacement threshold and its previous-position state. Keep the existing EMA, phase logic, visibility guards, speed limits, and true perception-loss handling unchanged.

**Tech Stack:** Python 3, NumPy, pytest

## Global Constraints

- Do not introduce a replacement jump threshold, counter, hold, or diagnostic state.
- Keep `ema_alpha=0.35` and all existing motion and visibility limits unchanged.
- Keep the 30-frame missing-target and missing-table termination behavior unchanged.
- Do not modify perception, bbox tracking, target-reference updates, non-raw base-pose modes, relay logic, or runtime limits.

---

### Task 1: Process all valid target positions through the EMA

**Files:**
- Modify: `gear_sonic/tests/test_base_pose_visual_servo.py`
- Modify: `gear_sonic/utils/inference/base_pose_visual_servo.py:1185-1189,1307-1354`

**Interfaces:**
- Consumes: `VisualServoController.update(observation: RawServoObservation, *, now: float) -> ServoCommand`
- Produces: `_update_filter()` behavior that accepts every valid finite target geometry without incrementing `invalid_frames`

- [ ] **Step 1: Write the failing regression test**

Add next to the existing tracking-loss controller tests:

```python
def test_large_valid_target_position_change_resets_tracking_loss_streak() -> None:
    controller = VisualServoController()
    controller.reset(1.0)
    controller.phase = ServoPhase.TRANSLATE_TARGET
    controller.update(observation(forward=1.0), now=1.1)

    for index in range(29):
        controller.note_invalid(
            "missing tracked target", hard=False, now=1.2 + 0.1 * index
        )

    command = controller.update(observation(forward=1.30), now=4.1)

    assert command.velocity != (0.0, 0.0, 0.0)
    assert controller.invalid_frames == 0
    assert controller.terminal_reason is None
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
./.venv_inference/bin/python -m pytest -q \
  gear_sonic/tests/test_base_pose_visual_servo.py::test_large_valid_target_position_change_resets_tracking_loss_streak
```

Expected: FAIL because the `0.30 m` change is classified as `target position jump`, increments `invalid_frames` from 29 to 30, and terminates the controller.

- [ ] **Step 3: Remove target-position jump state and handling**

In `VisualServoController.reset()`, delete:

```python
self.last_raw_target: tuple[float, float] | None = None
```

At the start of `_update_filter()`, replace the previous-target comparison with only:

```python
raw_target = (observation.target.forward_m, observation.target.right_m)
```

In `update()`, replace the `try`/`except ValueError` wrapper with:

```python
forward_error, right_error, yaw_error = self._update_filter(observation)
```

- [ ] **Step 4: Verify the focused test and controller suite are GREEN**

Run:

```bash
./.venv_inference/bin/python -m pytest -q \
  gear_sonic/tests/test_base_pose_visual_servo.py::test_large_valid_target_position_change_resets_tracking_loss_streak
./.venv_inference/bin/python -m pytest -q \
  gear_sonic/tests/test_base_pose_visual_servo.py
```

Expected: both commands PASS; existing 30-frame tracking-loss tests remain green.

- [ ] **Step 5: Check the exact diff and commit the implementation**

Run:

```bash
git diff --check -- \
  gear_sonic/tests/test_base_pose_visual_servo.py \
  gear_sonic/utils/inference/base_pose_visual_servo.py
git diff -- \
  gear_sonic/tests/test_base_pose_visual_servo.py \
  gear_sonic/utils/inference/base_pose_visual_servo.py
git add \
  gear_sonic/tests/test_base_pose_visual_servo.py \
  gear_sonic/utils/inference/base_pose_visual_servo.py
git commit -m "fix: remove base pose target position jump handling"
```
