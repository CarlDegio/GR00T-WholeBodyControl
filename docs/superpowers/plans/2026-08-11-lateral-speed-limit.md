# Lateral Speed Limit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise the final base-pose lateral command limit from 0.16 m/s to 0.22 m/s in both the visual controller and SONIC relay.

**Architecture:** The visual controller remains responsible for shaping proportional translation commands and applying the 0.30 m/s nonzero magnitude floor. It will apply a final lateral component bound after that floor, while the relay independently repeats the same bound at the transport boundary before converting velocity into SONIC direction and speed.

**Tech Stack:** Python 3.10, NumPy, pytest, ZMQ message builders.

## Global Constraints

- The final lateral component must remain within `[-0.22, 0.22]` m/s.
- Keep the existing `0.30 m/s` nonzero translation magnitude floor.
- Do not change forward/backward limits, yaw limits, controller phases, or collision-avoidance behavior.
- Preserve all pre-existing uncommitted work; do not commit the already-dirty production and test files as part of this implementation.

---

### Task 1: Bound visual-servo lateral output after minimum-speed scaling

**Files:**
- Modify: `gear_sonic/tests/test_base_pose_visual_servo.py`
- Modify: `gear_sonic/utils/inference/base_pose_visual_servo.py:752-1011`

**Interfaces:**
- Consumes: `VisualServoController.update(observation: RawServoObservation, *, now: float) -> ServoCommand`
- Produces: controller commands whose `vy` is always within `[-0.22, 0.22]` m/s.

- [ ] **Step 1: Write the failing controller test**

Add a test that places the controller directly in `ServoPhase.RECENTER`, supplies a large positive lateral error, and checks the observable command:

```python
def test_controller_caps_lateral_output_after_minimum_speed_scaling() -> None:
    controller = VisualServoController()
    controller.reset(1.0)
    controller.phase = ServoPhase.RECENTER

    command = controller.update(observation(right=1.0), now=1.1)

    assert command.vx == 0.0
    assert command.vy == pytest.approx(-0.22)
    assert command.wz == 0.0
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
source .venv_inference/bin/activate
pytest -q gear_sonic/tests/test_base_pose_visual_servo.py::test_controller_caps_lateral_output_after_minimum_speed_scaling
```

Expected: FAIL because the current controller returns `vy=-0.30` after minimum-speed scaling.

- [ ] **Step 3: Implement the visual-controller limit**

Add `self.max_lateral_speed_m_s = 0.22` beside the existing speed constants. Replace both proportional `desired_vy` literals with that property. After `_enforce_min_linear_speed(...)` in both `RECENTER` and `TRANSLATE_TARGET`, apply:

```python
vy = self._clip(vy, self.max_lateral_speed_m_s)
```

This preserves the existing radial floor computation but makes the emitted lateral component obey the final bound.

- [ ] **Step 4: Run the controller tests and verify GREEN**

Run:

```bash
source .venv_inference/bin/activate
pytest -q gear_sonic/tests/test_base_pose_visual_servo.py
```

Expected: all tests pass.

### Task 2: Raise and test the relay transport boundary

**Files:**
- Modify: `gear_sonic/tests/test_lavira_reasan_contract.py:90-105`
- Modify: `gear_sonic/scripts/lavira_sonic_relay.py:25-27`

**Interfaces:**
- Consumes: `decode_velocity_command(raw: bytes | str) -> dict[str, Any]`
- Produces: decoded relay velocity with `vy` clipped to `[-0.22, 0.22]` m/s.

- [ ] **Step 1: Write the failing relay boundary test**

Add a parameterized test using the real velocity-message builder:

```python
@pytest.mark.parametrize(
    ("requested_vy", "expected_vy"),
    [(0.40, 0.22), (-0.40, -0.22)],
)
def test_direct_relay_caps_lateral_velocity_at_point_two_two(
    requested_vy: float, expected_vy: float
) -> None:
    decoded = decode_direct_velocity_command(
        build_reasan_velocity_message(
            VelocityCommand(0.0, requested_vy, 0.0, 1.0), action="move"
        )
    )

    assert decoded["velocity"].tolist() == pytest.approx(
        [0.0, expected_vy, 0.0]
    )
```

- [ ] **Step 2: Run the focused relay test and verify RED**

Run:

```bash
source .venv_inference/bin/activate
pytest -q gear_sonic/tests/test_lavira_reasan_contract.py::test_direct_relay_caps_lateral_velocity_at_point_two_two
```

Expected: FAIL because the current relay clips both cases at `+/-0.16` m/s.

- [ ] **Step 3: Implement the relay limit**

Change only the lateral entries in the relay arrays:

```python
COMMAND_LOWER = np.array([-0.5, -0.22, -1.0], dtype=np.float32)
COMMAND_UPPER = np.array([1.0, 0.22, 1.0], dtype=np.float32)
```

- [ ] **Step 4: Run focused and regression tests**

Run:

```bash
source .venv_inference/bin/activate
pytest -q \
  gear_sonic/tests/test_base_pose_visual_servo.py \
  gear_sonic/tests/test_lavira_reasan_contract.py \
  gear_sonic/tests/test_base_pose_adjustment.py
```

Expected: all tests pass with no warnings or errors.

- [ ] **Step 5: Review the final diff**

Run:

```bash
git diff -- \
  gear_sonic/utils/inference/base_pose_visual_servo.py \
  gear_sonic/scripts/lavira_sonic_relay.py \
  gear_sonic/tests/test_base_pose_visual_servo.py \
  gear_sonic/tests/test_lavira_reasan_contract.py
```

Confirm that the new hunks only add the `0.22 m/s` behavior and its tests; leave the files uncommitted because they contained pre-existing user changes before this task.
