# Vertical Recenter and Target Track Reacquisition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the target visible at the top of the camera image with a forward-only, vertically prioritized recovery phase and continue when the same YOLOE target class receives a new BoT-SORT ID.

**Architecture:** Extend the deterministic visual-servo state machine with an explicit `VERTICAL_RECENTER` phase that preempts every active phase and returns through a saved vertical resume phase. Resolve target tracks class-first in the worker, adopt a highest-confidence replacement class-0 ID, and propagate takeover metadata through existing event and frame diagnostics.

**Tech Stack:** Python 3.10, dataclasses/Enum, NumPy, YOLOE/BoT-SORT, JSONL, pytest.

## Global Constraints

- Enter vertical recovery below 18 percent image-height center and recover at or beyond 23 percent for three frames.
- Stop on the transition frame; while unrecovered publish only `vx=+0.30 m/s` and keep `vy=wz=0`.
- Hold zero as soon as one frame crosses the recovery line while confirming three stable frames.
- Vertical recovery has priority over horizontal recovery and preserves the interrupted phase.
- Prefer the expected class-0 ID; otherwise adopt the highest-confidence class-0 ID unconditionally.
- A true target absence remains an immediate-zero soft miss and stops after the fourth consecutive miss.
- Preserve table-ID behavior, grounding prompts, confidence thresholds, watchdogs, and operator stops.
- Preserve all pre-existing workspace modifications. Do not create a Git commit because both task code files contain pre-existing untracked work.

---

### Task 1: Vertically prioritized recovery state

**Files:**
- Modify: `gear_sonic/utils/inference/base_pose_visual_servo.py`
- Test: `gear_sonic/tests/test_base_pose_visual_servo.py`

**Interfaces:**
- Extends `ServoPhase` with `VERTICAL_RECENTER`.
- Extends `RawServoObservation` with `image_height: int = 480`.
- Produces controller fields `vertical_resume_phase` and `vertical_recenter_stable_frames`.

- [ ] **Step 1: Write failing controller tests**

Add tests whose hand-derived 480-pixel thresholds prove transition zero, forward-only recovery, three stable frames, and vertical priority:

```python
def test_top_guard_stops_and_enters_vertical_recenter():
    controller = VisualServoController()
    controller.reset(1.0)
    command = controller.update(
        observation(yaw=math.radians(20), bbox=(340, 0, 600, 160)), now=1.1
    )
    assert command.velocity == (0.0, 0.0, 0.0)
    assert controller.phase is ServoPhase.VERTICAL_RECENTER
    assert controller.vertical_resume_phase is ServoPhase.YAW_ALIGN


def test_vertical_recenter_moves_only_forward_then_holds_for_three_frames():
    controller = VisualServoController()
    controller.reset(1.0)
    controller.update(observation(bbox=(260, 0, 380, 160)), now=1.1)
    assert controller.update(
        observation(bbox=(260, 0, 380, 180)), now=1.2
    ).velocity == pytest.approx((0.30, 0.0, 0.0))
    for index in range(3):
        command = controller.update(
            observation(bbox=(260, 0, 380, 240)), now=1.3 + index * 0.1
        )
        assert command.velocity == (0.0, 0.0, 0.0)
    assert controller.phase is ServoPhase.YAW_ALIGN
```

Add a simultaneous top/right guard test that recovers vertically first, then enters horizontal `RECENTER` with the original yaw phase still saved.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
./.venv_inference/bin/python -m pytest \
  gear_sonic/tests/test_base_pose_visual_servo.py \
  -k 'top_guard or vertical_recenter or vertical_priority' -v
```

Expected: failures because the phase, height field, and vertical controller behavior do not exist.

- [ ] **Step 3: Implement the minimal state-machine behavior**

Add normalized helpers:

```python
def _vertical_visibility_guarded(self, observation):
    center_y = 0.5 * (
        observation.target_bbox_xyxy[1] + observation.target_bbox_xyxy[3]
    )
    return center_y < self.vertical_guard_fraction * observation.image_height

def _target_vertical_recovered(self, observation):
    center_y = 0.5 * (
        observation.target_bbox_xyxy[1] + observation.target_bbox_xyxy[3]
    )
    return center_y >= self.vertical_recovery_fraction * observation.image_height
```

Use constants `0.18`, `0.23`, and three stable frames. Check the vertical guard before horizontal/yaw/translation branches. Preserve both vertical and horizontal resume state so vertical completion either returns to an interrupted horizontal recenter, enters one if the x guard is still active, or resumes the original phase.

- [ ] **Step 4: Run focused and full module tests**

```bash
./.venv_inference/bin/python -m pytest \
  gear_sonic/tests/test_base_pose_visual_servo.py -q
```

Expected: all visual-servo tests pass.

### Task 2: Same-class target-ID takeover and audit trail

**Files:**
- Modify: `gear_sonic/utils/inference/base_pose_visual_servo.py`
- Test: `gear_sonic/tests/test_base_pose_visual_servo.py`

**Interfaces:**
- Produces `_resolve_target(instances, expected_id, class_index=0) -> tuple[TrackedInstance | None, bool, bool]`, where booleans mean reacquired and hard class mismatch.
- Adds `event_details` to each runtime `control_update` record.

- [ ] **Step 1: Write failing resolver and worker tests**

Use real `TrackedInstance` values and literal expectations:

```python
def test_target_resolver_adopts_highest_confidence_same_class_new_id():
    low = tracked(track_id=7, class_index=0, confidence=0.60)
    high = tracked(track_id=8, class_index=0, confidence=0.95)
    chosen, reacquired, mismatch = raw_servo._resolve_target([low, high], 1)
    assert chosen is high
    assert reacquired
    assert not mismatch
```

Add coverage that the expected class-0 ID wins even at lower confidence, no class-0 instance remains a soft miss, and an expected ID under the wrong class with no target is a hard mismatch. Extend a fake worker tracker sequence from ID 1 to IDs 7/8 and assert the observation uses ID 8 plus details:

```python
assert followup.observation.target_track_id == 8
assert followup.details == {
    "surface_id_mismatch": False,
    "target_reacquired": True,
    "previous_target_id": 1,
    "target_track_id": 8,
}
```

- [ ] **Step 2: Run takeover tests and verify RED**

```bash
./.venv_inference/bin/python -m pytest \
  gear_sonic/tests/test_base_pose_visual_servo.py \
  -k 'resolver or reacquire or new_target_id' -v
```

Expected: failures because `_resolve_target` and takeover metadata do not exist.

- [ ] **Step 3: Implement class-first resolution and worker adoption**

Filter class-0 candidates, prefer `expected_id`, otherwise choose
`max(candidates, key=lambda item: item.confidence)`. Update `target_id` before
building the observation. Keep the previous hard mismatch only when no class-0
candidate exists but the expected ID is present under a different class.

Include the takeover fields in `RawServoEvent.details`. Add
`event_details=dict(event.details or {})` to runtime `control_update` JSONL
records so the handoff can be audited without changing event kinds.

- [ ] **Step 4: Run focused and module tests**

```bash
./.venv_inference/bin/python -m pytest \
  gear_sonic/tests/test_base_pose_visual_servo.py -q
```

Expected: all visual-servo tests pass.

### Task 3: Diagnostics, operator documentation, and regression verification

**Files:**
- Modify: `gear_sonic/utils/inference/base_pose_visual_servo.py`
- Modify: `docs/base_pose_adjustment.md`
- Verify: `gear_sonic/tests/test_base_pose_visual_servo_diagnostics.py`

**Interfaces:**
- Adds `vertical_resume_phase` and `vertical_recenter_stable_frames` to frame controller-state telemetry.
- Documents `VERTICAL_RECENTER` before horizontal `RECENTER` and same-class ID takeover.

- [ ] **Step 1: Extend diagnostics assertions and implementation**

Assert a vertical-recovery frame serializes phase, resume phase, and stable-frame count. Pass the two new controller fields through `_record_frame` without changing existing keys.

- [ ] **Step 2: Update operator documentation**

Document the 18-percent top guard, 23-percent three-frame recovery line,
forward-only 0.30 m/s command, vertical priority, and unconditional
highest-confidence class-0 ID takeover.

- [ ] **Step 3: Run syntax and focused verification**

```bash
./.venv_inference/bin/python -m py_compile \
  gear_sonic/utils/inference/base_pose_visual_servo.py
./.venv_inference/bin/python -m pytest \
  gear_sonic/tests/test_base_pose_visual_servo.py \
  gear_sonic/tests/test_base_pose_visual_servo_diagnostics.py -q
```

- [ ] **Step 4: Run complete regression and diff checks**

```bash
./.venv_inference/bin/python -m pytest gear_sonic/tests -q
git diff --check
git status --short
```

Expected: complete suite passes; only task-related files plus pre-existing user changes are present.
