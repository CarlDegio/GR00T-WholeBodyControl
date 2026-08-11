# Base-Pose Orientation Telemetry and Safety Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enter horizontal recovery at 25%/75%, recover inside 30%--70%, cap coarse heading slew at 0.15 rad/s, and attach relay-authoritative actual yaw/heading telemetry to every applied raw visual-servo frame.

**Architecture:** Keep the visual controller state machine intact and change only its horizontal fractions and coarse-yaw cap. Add a small shared orientation-telemetry module, let `lavira_sonic_relay` publish its post-integration heading plus `g1_debug.base_quat` yaw on a dedicated conflated ZMQ stream, and let the raw base-pose runtime attach the latest sample to its existing asynchronous per-frame diagnostics.

**Tech Stack:** Python 3.10, NumPy, pyzmq, JSONL, pytest.

## Global Constraints

- Horizontal entry is strictly outside 25%--75% bbox-center width.
- Horizontal recovery is inclusively inside 30%--70% for three consecutive applied visual frames.
- Coarse `YAW_ALIGN` uses `|wz| <= 0.15 rad/s` and the existing 0.05 rad/s per-frame slew step.
- `YAW_TRIM` remains capped at 0.10 rad/s.
- Orientation telemetry is diagnostic only and must never pause, terminate, or alter a control command.
- `heading_setpoint_rad` comes from the relay after its actual integration step; do not approximate it in the visual process.
- Enabling telemetry must not freeze the upper body unless `--freeze-current-upper-body` is explicitly set.
- Expose symmetric guard/recovery margins as configurable fractions and require `0 < guard < recovery < 0.5`.
- Enable the telemetry publisher only for `planner_input="base_pose"` with `base_pose_mode="raw_yoloe_servo"`; keep other launch paths unchanged.
- Preserve all pre-existing workspace changes. Do not create commits because relevant production and test files already contain uncommitted user work.

---

### Task 1: Tune horizontal guard/recovery and coarse yaw

**Files:**
- Modify: `gear_sonic/tests/test_base_pose_visual_servo.py`
- Modify: `gear_sonic/scripts/base_pose_planner.py`
- Modify: `gear_sonic/utils/inference/base_pose_visual_servo.py:774-785,877-894,1056-1063`

**Interfaces:**
- Consumes: `RawServoObservation.target_bbox_xyxy`, `RawServoObservation.image_width`, and filtered yaw error.
- Produces: configurable `VisualServoController(..., horizontal_guard_fraction=..., horizontal_recovery_fraction=...)`, `_visibility_guarded(...)`, `_target_center_recovered(...)`, and `update(...)` with the approved defaults and cap.

- [ ] **Step 1: Change the boundary tests before production code**

Replace the existing 20%/80% and 25%--75% tests with literal 640 px fixtures:

```python
def test_horizontal_guard_uses_center_25_and_75_percent_boundaries() -> None:
    controller = VisualServoController()

    assert controller._visibility_guarded(
        observation(bbox=(0.0, 100.0, 319.0, 300.0))
    )
    assert not controller._visibility_guarded(
        observation(bbox=(0.0, 100.0, 320.0, 300.0))
    )
    assert not controller._visibility_guarded(
        observation(bbox=(320.0, 100.0, 640.0, 300.0))
    )
    assert controller._visibility_guarded(
        observation(bbox=(321.0, 100.0, 640.0, 300.0))
    )


def test_horizontal_recovery_uses_center_30_to_70_percent_boundaries() -> None:
    controller = VisualServoController()

    assert not controller._target_center_recovered(
        observation(bbox=(141.5, 100.0, 241.5, 300.0))
    )
    assert controller._target_center_recovered(
        observation(bbox=(142.0, 100.0, 242.0, 300.0))
    )
    assert controller._target_center_recovered(
        observation(bbox=(398.0, 100.0, 498.0, 300.0))
    )
    assert not controller._target_center_recovered(
        observation(bbox=(398.5, 100.0, 498.5, 300.0))
    )
```

Add a coarse-yaw saturation test that exercises the real update loop:

```python
def test_controller_coarse_yaw_caps_heading_slew_at_point_one_five() -> None:
    controller = VisualServoController()
    controller.reset(1.0)

    commands = [
        controller.update(
            observation(yaw=math.radians(20.0)), now=1.1 + index * 0.1
        )
        for index in range(4)
    ]

    assert [command.wz for command in commands] == pytest.approx(
        [0.05, 0.10, 0.15, 0.15]
    )
```

Add validation tests for `0 < guard < recovery < 0.5` and a runtime propagation
test using non-default values:

```python
def test_runtime_passes_configured_horizontal_intervals_to_controller(
    tmp_path,
) -> None:
    runtime = RawServoRuntime(
        BasePosePlannerConfig(
            task="align",
            output_root=str(tmp_path),
            raw_horizontal_guard_fraction=0.20,
            raw_horizontal_recovery_fraction=0.28,
        ),
        publish=lambda _message: None,
    )

    assert runtime.controller.guard_fraction == pytest.approx(0.20)
    assert runtime.controller.recovery_low_fraction == pytest.approx(0.28)
    assert runtime.controller.recovery_high_fraction == pytest.approx(0.72)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
./.venv_inference/bin/python -m pytest \
  gear_sonic/tests/test_base_pose_visual_servo.py \
  -k 'horizontal_guard_uses or horizontal_recovery_uses or coarse_yaw_caps' -q
```

Expected: the new boundary tests fail under 20%/80% and 25%--75%; the fourth coarse-yaw command is `0.20`, not `0.15`.

- [ ] **Step 3: Implement only the approved controller changes**

Add constructor parameters, validate their ordering, and derive symmetric
bounds:

```python
self.guard_fraction = float(horizontal_guard_fraction)
self.recovery_low_fraction = float(horizontal_recovery_fraction)
self.recovery_high_fraction = 1.0 - self.recovery_low_fraction
```

Add `raw_horizontal_guard_fraction: float = 0.25` and
`raw_horizontal_recovery_fraction: float = 0.30` to
`BasePosePlannerConfig`, then pass both values from `RawServoRuntime` into
`VisualServoController`.

and clamp coarse yaw before and after slew:

```python
desired_wz = self._clip(yaw_error, 0.15)
self.current = ServoCommand(
    0.0,
    0.0,
    self._clip(self._slew(self.current.wz, desired_wz, 0.05), 0.15),
    self.command_ttl_s,
)
```

Do not alter vertical guards, recenter velocities, trim limits, or transitions.

- [ ] **Step 4: Run the focused controller tests and verify GREEN**

Run the Step 2 command again. Expected: all selected tests pass.

---

### Task 2: Build and test the shared orientation telemetry contract

**Files:**
- Create: `gear_sonic/utils/teleop/sonic_orientation_telemetry.py`
- Create: `gear_sonic/tests/test_sonic_orientation_telemetry.py`

**Interfaces:**
- Produces: `OrientationTelemetrySample`, `OrientationTracker`, `quaternion_yaw_wxyz(...)`, `encode_orientation_telemetry(...)`, `decode_orientation_telemetry(...)`, and `LatestOrientationTelemetry`.
- Consumes later: relay state dictionaries with `base_quat`; raw runtime receives encoded telemetry strings.

- [ ] **Step 1: Write quaternion and validation tests**

Use hand-derived yaw fixtures and explicit invalid inputs:

```python
@pytest.mark.parametrize(
    ("quaternion", "expected_yaw"),
    [
        ([1.0, 0.0, 0.0, 0.0], 0.0),
        ([math.cos(0.25), 0.0, 0.0, math.sin(0.25)], 0.5),
        ([2.0 * math.cos(-0.2), 0.0, 0.0, 2.0 * math.sin(-0.2)], -0.4),
    ],
)
def test_quaternion_yaw_wxyz_normalizes_and_extracts_yaw(
    quaternion, expected_yaw
) -> None:
    assert quaternion_yaw_wxyz(quaternion) == pytest.approx(expected_yaw)


@pytest.mark.parametrize(
    "quaternion",
    [[], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [math.nan, 0.0, 0.0, 0.0]],
)
def test_quaternion_yaw_wxyz_rejects_invalid_values(quaternion) -> None:
    with pytest.raises(ValueError):
        quaternion_yaw_wxyz(quaternion)
```

- [ ] **Step 2: Write tracker behavior tests**

Test that a state sampled at yaw `0.4` while heading is zero establishes the origin, then yaw `0.5` with heading `0.15` reports literal values:

```python
def test_orientation_tracker_reports_post_integration_heading_and_wrapped_lag() -> None:
    tracker = OrientationTracker()
    tracker.update_state(
        {"base_quat": yaw_quaternion(0.4)},
        received_at_monotonic_s=10.0,
        heading_setpoint_rad=0.0,
    )
    tracker.update_state(
        {"base_quat": yaw_quaternion(0.5)},
        received_at_monotonic_s=10.1,
        heading_setpoint_rad=0.15,
    )

    sample = tracker.sample(now_monotonic_s=10.12, heading_setpoint_rad=0.15)

    assert sample.actual_yaw_rad == pytest.approx(0.5)
    assert sample.actual_heading_rad == pytest.approx(0.1)
    assert sample.heading_setpoint_rad == pytest.approx(0.15)
    assert sample.heading_lag_rad == pytest.approx(0.05)
    assert sample.state_age_s == pytest.approx(0.02)
```

Add separate tests proving:

- no state yields nullable actual fields without affecting heading;
- if heading becomes nonzero before the first valid state, later samples keep `actual_heading_rad` and `heading_lag_rad` null;
- a `+pi/-pi` crossing produces wrapped lag rather than a near-`2*pi` value;
- encode/decode rejects wrong type/version, malformed JSON, non-finite fields, and inconsistent nullable fields;
- `LatestOrientationTelemetry.diagnostics(now)` adds nonnegative `telemetry_age_s` and keeps only the newest decoded sample.

- [ ] **Step 3: Run the new module tests and verify RED**

Run:

```bash
./.venv_inference/bin/python -m pytest \
  gear_sonic/tests/test_sonic_orientation_telemetry.py -q
```

Expected: collection fails because the shared module does not exist.

- [ ] **Step 4: Implement the minimal shared module**

Use scalar-first yaw extraction:

```python
yaw = math.atan2(
    2.0 * (qw * qz + qx * qy),
    1.0 - 2.0 * (qy * qy + qz * qz),
)
```

Normalize quaternions first, wrap every angle with `math.remainder(value, 2*pi)`, require all non-null numeric fields to be finite, and encode with `allow_nan=False`. The wire object must contain:

```python
{
    "type": "sonic_orientation_telemetry",
    "version": 1,
    "emitted_at_monotonic_s": sample.emitted_at_monotonic_s,
    "actual_yaw_rad": sample.actual_yaw_rad,
    "actual_heading_rad": sample.actual_heading_rad,
    "heading_setpoint_rad": sample.heading_setpoint_rad,
    "heading_lag_rad": sample.heading_lag_rad,
    "state_age_s": sample.state_age_s,
}
```

- [ ] **Step 5: Run the module tests and verify GREEN**

Run the Step 3 command again. Expected: all orientation telemetry tests pass.

---

### Task 3: Publish relay-authoritative orientation telemetry

**Files:**
- Modify: `gear_sonic/scripts/lavira_sonic_relay.py`
- Modify: `gear_sonic/tests/test_base_pose_adjustment.py`
- Modify: `gear_sonic/tests/test_lavira_reasan_contract.py`

**Interfaces:**
- Consumes: `g1_debug.base_quat`, `PlannerState.heading`, and `OrientationTracker`.
- Produces: optional `--orientation-telemetry-output` PUB stream carrying `sonic_orientation_telemetry` messages after each planner integration tick.

- [ ] **Step 1: Write relay integration-unit tests**

Add tests that use real `PlannerState` and `OrientationTracker` to prove the telemetry sample is built after `PlannerState.message(...)` updates heading:

```python
def test_relay_orientation_sample_uses_post_integration_heading() -> None:
    planner = PlannerState()
    tracker = OrientationTracker()
    tracker.update_state(
        {"base_quat": [1.0, 0.0, 0.0, 0.0]},
        received_at_monotonic_s=10.0,
        heading_setpoint_rad=planner.heading,
    )

    planner.message(np.array([0.0, 0.0, 0.15], dtype=np.float32), 0.05)
    sample = tracker.sample(10.05, planner.heading)

    assert sample.heading_setpoint_rad == pytest.approx(0.0075)
    assert sample.heading_lag_rad == pytest.approx(0.0075)
```

Add a focused helper test proving telemetry-enabled state processing updates the orientation tracker but returns no `FrozenPlannerPose` when `freeze_current_upper_body=False`. The same helper must still extract a frozen pose when the flag is true.

Add an argument-default test proving telemetry output is disabled unless explicitly configured.

- [ ] **Step 2: Run the relay tests and verify RED**

Run:

```bash
./.venv_inference/bin/python -m pytest \
  gear_sonic/tests/test_base_pose_adjustment.py \
  gear_sonic/tests/test_lavira_reasan_contract.py \
  -k 'orientation or telemetry' -q
```

Expected: tests fail because the relay has no telemetry argument or state-processing helper.

- [ ] **Step 3: Add optional telemetry publication without changing planner control**

- Add `--orientation-telemetry-output`, defaulting to an empty string.
- Create the state subscriber when either upper-body freezing or orientation telemetry is enabled.
- Poll state continuously, update `OrientationTracker` when telemetry is enabled, and gate `extract_frozen_planner_pose(...)` only on `--freeze-current-upper-body`.
- On each planner tick, first call `planner.message(...)`, then publish the matching orientation telemetry sample.
- Configure the telemetry PUB socket with `LINGER=0` and close it in `finally`.
- Catch invalid `base_quat` as diagnostic loss; never suppress or modify the planner output.

- [ ] **Step 4: Run relay tests and verify GREEN**

Run the Step 2 command again. Expected: all selected relay tests pass.

---

### Task 4: Attach latest orientation telemetry to raw frame diagnostics

**Files:**
- Modify: `gear_sonic/utils/inference/base_pose_visual_servo_diagnostics.py`
- Modify: `gear_sonic/utils/inference/base_pose_visual_servo.py`
- Modify: `gear_sonic/scripts/base_pose_planner.py`
- Modify: `gear_sonic/tests/test_base_pose_visual_servo_diagnostics.py`
- Modify: `gear_sonic/tests/test_base_pose_visual_servo.py`

**Interfaces:**
- Consumes: `LatestOrientationTelemetry.diagnostics(now_monotonic_s)` through an injected `RawServoRuntime` provider.
- Produces: top-level `orientation` in `raw_servo_frames.jsonl`, or JSON `null` when unavailable.

- [ ] **Step 1: Write diagnostic writer tests first**

Extend the complete-record test to pass this literal mapping:

```python
orientation={
    "actual_yaw_rad": 1.2,
    "actual_heading_rad": 0.1,
    "heading_setpoint_rad": 0.15,
    "heading_lag_rad": 0.05,
    "state_age_s": 0.01,
    "telemetry_age_s": 0.02,
}
```

and assert exact normalized output. Extend the displaced-frame test to assert `record["orientation"] is None`.

Add an async-join test proving orientation survives `submit_decision(...)` and appears on the matching frame rather than a neighboring frame.

- [ ] **Step 2: Write runtime-provider tests first**

Construct `RawServoRuntime` with an injected provider returning the literal mapping above, accept a real initialized frame event, flush diagnostics, and assert that frame contains the mapping. Add a second test with a provider returning `None` and assert the applied frame records `orientation: null` while still producing the same command.

- [ ] **Step 3: Run diagnostic/runtime tests and verify RED**

Run:

```bash
./.venv_inference/bin/python -m pytest \
  gear_sonic/tests/test_base_pose_visual_servo_diagnostics.py \
  gear_sonic/tests/test_base_pose_visual_servo.py \
  -k 'orientation or complete_jsonl or displaced_frame' -q
```

Expected: calls fail because writers and runtime do not accept orientation data.

- [ ] **Step 4: Extend the asynchronous diagnostics contract**

- Add `orientation: Mapping[str, Any] | None` to `_ControlDecision`.
- Add optional `orientation=None` keyword parameters to `FrameDiagnosticsWriter.write(...)` and `AsyncFrameDiagnosticsWriter.submit_decision(...)` so existing callers remain compatible.
- Normalize the six approved fields and place them at top level in each JSONL record.
- Thread orientation through `_submit_diagnostic_decision(...)` only for applied runtime decisions; discarded worker frames remain `null`.
- Add an optional injected `orientation_provider: Callable[[float], Mapping[str, Any] | None]` to `RawServoRuntime` and call it once per accepted decision with that decision's monotonic time.
- Catch provider errors, warn without changing commands, and record `null`.

- [ ] **Step 5: Connect the real ZMQ subscriber in raw-servo startup**

Add `raw_orientation_telemetry_source: str = "tcp://127.0.0.1:5565"` to `BasePosePlannerConfig`. In `_raw_servo_main`, create a `zmq.SUB` socket with `SUBSCRIBE=b""`, `CONFLATE=1`, and `LINGER=0`; decode the newest packet into `LatestOrientationTelemetry`; provide its current diagnostics to `RawServoRuntime`; rate-limit malformed-message warnings to at most once per second; and close the socket in `finally`.

Do not create or connect this subscriber in non-raw base-pose modes.

- [ ] **Step 6: Run diagnostic/runtime tests and verify GREEN**

Run the Step 3 command again. Expected: all selected tests pass.

---

### Task 5: Wire launch endpoints and update operator documentation

**Files:**
- Modify: `gear_sonic/scripts/launch_inference.py`
- Modify: `gear_sonic/tests/test_lavira_planner.py`
- Modify: `gear_sonic/tests/test_base_pose_visual_servo.py`
- Modify: `docs/base_pose_adjustment.md`

**Interfaces:**
- Consumes: `InferenceLaunchConfig.base_pose_orientation_telemetry_port`.
- Produces: matching relay PUB and raw-servo SUB CLI arguments on the base-pose launch path.

- [ ] **Step 1: Write launch command tests first**

Extend raw-YOLOE base-pose command tests to assert:

```python
assert "--raw-orientation-telemetry-source tcp://127.0.0.1:5565" in planner_command
assert "--orientation-telemetry-output 'tcp://*:5565'" in relay_command
assert "--raw-horizontal-guard-fraction 0.25" in planner_command
assert "--raw-horizontal-recovery-fraction 0.3" in planner_command
```

Also assert RGB base-pose and standalone non-base-pose relay commands do not contain `--orientation-telemetry-output`.

- [ ] **Step 2: Run launch tests and verify RED**

Run:

```bash
./.venv_inference/bin/python -m pytest \
  gear_sonic/tests/test_lavira_planner.py \
  gear_sonic/tests/test_base_pose_visual_servo.py \
  -k 'base_pose and (command or relay or telemetry)' -q
```

Expected: endpoint assertions fail because launch wiring is absent.

- [ ] **Step 3: Implement matching endpoint wiring**

Add these fields to `InferenceLaunchConfig`:

```python
base_pose_orientation_telemetry_port: int = 5565
base_pose_raw_horizontal_guard_fraction: float = 0.25
base_pose_raw_horizontal_recovery_fraction: float = 0.30
```

Append both fraction arguments and the raw subscriber argument to
`build_planner_input_command(...)`; append the relay publisher argument only
when `planner_input == "base_pose"` and
`base_pose_mode == "raw_yoloe_servo"` in
`build_reasan_planner_command(...)`.

- [ ] **Step 4: Update operator documentation**

In `docs/base_pose_adjustment.md` document:

- horizontal entry at 25%/75%;
- three-frame recovery inside 30%--70%;
- the two launch configuration fields used to tune those symmetric intervals;
- coarse heading slew capped at 0.15 rad/s;
- the top-level orientation fields, their relative/absolute semantics, sample ages, and the fact that logging does not affect control.

- [ ] **Step 5: Run launch tests and verify GREEN**

Run the Step 2 command again. Expected: all selected launch tests pass.

---

### Task 6: Full verification and diff audit

**Files:**
- Verify all files changed by Tasks 1--5.

**Interfaces:**
- Produces: fresh evidence that the implementation compiles, focused behavior passes, and unrelated tests remain intact.

- [ ] **Step 1: Compile every changed Python module**

Run:

```bash
./.venv_inference/bin/python -m py_compile \
  gear_sonic/utils/teleop/sonic_orientation_telemetry.py \
  gear_sonic/scripts/lavira_sonic_relay.py \
  gear_sonic/scripts/base_pose_planner.py \
  gear_sonic/scripts/launch_inference.py \
  gear_sonic/utils/inference/base_pose_visual_servo.py \
  gear_sonic/utils/inference/base_pose_visual_servo_diagnostics.py
```

- [ ] **Step 2: Run all directly affected tests**

Run:

```bash
./.venv_inference/bin/python -m pytest -q \
  gear_sonic/tests/test_sonic_orientation_telemetry.py \
  gear_sonic/tests/test_base_pose_visual_servo.py \
  gear_sonic/tests/test_base_pose_visual_servo_diagnostics.py \
  gear_sonic/tests/test_base_pose_adjustment.py \
  gear_sonic/tests/test_lavira_reasan_contract.py \
  gear_sonic/tests/test_lavira_planner.py
```

- [ ] **Step 3: Run the complete repository test directory**

Run:

```bash
./.venv_inference/bin/python -m pytest gear_sonic/tests -q
```

- [ ] **Step 4: Audit formatting and scope**

Run:

```bash
git diff --check
git status --short
git diff -- \
  docs/base_pose_adjustment.md \
  gear_sonic/utils/teleop/sonic_orientation_telemetry.py \
  gear_sonic/scripts/lavira_sonic_relay.py \
  gear_sonic/scripts/base_pose_planner.py \
  gear_sonic/scripts/launch_inference.py \
  gear_sonic/utils/inference/base_pose_visual_servo.py \
  gear_sonic/utils/inference/base_pose_visual_servo_diagnostics.py \
  gear_sonic/tests/test_sonic_orientation_telemetry.py \
  gear_sonic/tests/test_base_pose_visual_servo.py \
  gear_sonic/tests/test_base_pose_visual_servo_diagnostics.py \
  gear_sonic/tests/test_base_pose_adjustment.py \
  gear_sonic/tests/test_lavira_reasan_contract.py \
  gear_sonic/tests/test_lavira_planner.py
```

Confirm every requested behavior has a corresponding red-green test, telemetry remains observational, no unrelated user changes were overwritten, and no commit was created.
