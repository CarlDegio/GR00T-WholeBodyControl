# Proportional Relay Lateral Limit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the base-pose controller's 0.30 m/s minimum nonzero translation while making the lateral limit configurable with a 0.16 m/s default and preserving the `vx:vy` ratio at the final relay boundary.

**Architecture:** One launch setting is forwarded to the raw visual-servo controller and the direct SONIC relay. The controller bounds its proportional lateral request, then applies the existing two-dimensional 0.30 m/s minimum without a post-floor component clip; the relay performs final transport limiting by multiplying both linear components by one scale factor whenever either linear component exceeds its bound.

**Tech Stack:** Python 3.10, dataclasses/tyro CLI configuration, NumPy, ZMQ command JSON, pytest.

## Global Constraints

- The configurable default maximum lateral speed is exactly `0.16 m/s`.
- Every nonzero upper-level controller translation is still raised to at least `0.30 m/s` before relay handling.
- Final relay limiting must preserve the ratio between `vx` and `vy`.
- Relay output after proportional limiting may have a total linear magnitude below `0.30 m/s`.
- Keep the fixed vertical-reacquisition command at `vx = 0.30 m/s`.
- Keep angular `wz` limiting independent from linear-vector scaling.
- Do not change controller phases, tracking-loss behavior, collision avoidance, or command timeouts.
- Preserve all pre-existing uncommitted work. Snapshot every dirty target file before editing and do not commit production/test files that already contain user changes.

---

### Task 1: Make the upper-level controller configurable without losing its 0.30 m/s floor

**Files:**
- Modify: `gear_sonic/tests/test_base_pose_visual_servo.py:565-600`
- Modify: `gear_sonic/utils/inference/base_pose_visual_servo.py:1132-1175`
- Modify: `gear_sonic/utils/inference/base_pose_visual_servo.py:1420-1427`
- Modify: `gear_sonic/utils/inference/base_pose_visual_servo.py:1470-1478`
- Modify: `gear_sonic/utils/inference/base_pose_visual_servo.py:2249-2258`
- Modify: `gear_sonic/scripts/base_pose_planner.py:76-88`

**Interfaces:**
- Consumes: `BasePosePlannerConfig.raw_max_lateral_speed_m_s: float`.
- Produces: `VisualServoController(..., max_lateral_speed_m_s: float = 0.16)` and raw commands whose nonzero linear magnitude is at least 0.30 m/s before the relay.

- [ ] **Step 1: Snapshot the dirty files**

Run:

```bash
cp -- gear_sonic/tests/test_base_pose_visual_servo.py /tmp/test_base_pose_visual_servo.py.before-proportional-relay
cp -- gear_sonic/utils/inference/base_pose_visual_servo.py /tmp/base_pose_visual_servo.py.before-proportional-relay
cp -- gear_sonic/scripts/base_pose_planner.py /tmp/base_pose_planner.py.before-proportional-relay
```

Expected: all three snapshots exist under `/tmp`; no repository file changes.

- [ ] **Step 2: Write failing controller tests**

Replace the old post-floor lateral-clip assertion and add configuration validation:

```python
def test_controller_minimum_speed_scaling_can_exceed_prefloor_lateral_limit() -> None:
    controller = VisualServoController()
    controller.reset(1.0)
    controller.phase = ServoPhase.RECENTER

    command = controller.update(observation(right=1.0), now=1.1)

    assert command.velocity == pytest.approx((0.0, -0.30, 0.0))
    assert math.hypot(command.vx, command.vy) >= 0.30


def test_controller_accepts_configurable_prefloor_lateral_limit() -> None:
    controller = VisualServoController(max_lateral_speed_m_s=0.12)
    controller.reset(1.0)
    controller.phase = ServoPhase.TRANSLATE_TARGET

    command = controller.update(observation(forward=1.2, right=1.0), now=1.1)

    assert math.hypot(command.vx, command.vy) == pytest.approx(0.30)
    assert command.vx / -command.vy == pytest.approx(0.20 / 0.12)


@pytest.mark.parametrize("limit", [0.0, -0.1, math.nan, math.inf])
def test_controller_rejects_invalid_lateral_speed_limit(limit: float) -> None:
    with pytest.raises(ValueError, match="lateral speed"):
        VisualServoController(max_lateral_speed_m_s=limit)
```

Update the recenter regression to expect the pre-relay command `vy=-0.30`.

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```bash
.venv_inference/bin/python -m pytest -q gear_sonic/tests/test_base_pose_visual_servo.py -k 'minimum_speed_scaling_can_exceed or configurable_prefloor or rejects_invalid_lateral_speed or recenter_uses_only_vy'
```

Expected: failures because the constructor does not accept the new argument and the current controller clips post-floor `vy` to 0.22 m/s.

- [ ] **Step 4: Implement the controller/config behavior**

Add the planner field:

```python
raw_max_lateral_speed_m_s: float = 0.16
```

Add the controller keyword argument and validation:

```python
max_lateral_speed_m_s: float = 0.16,
```

```python
lateral_speed_limit = float(max_lateral_speed_m_s)
if not math.isfinite(lateral_speed_limit) or lateral_speed_limit <= 0.0:
    raise ValueError("maximum lateral speed must be finite and positive")
self.min_linear_speed_m_s = 0.30
self.max_lateral_speed_m_s = lateral_speed_limit
```

Keep the existing `desired_vy = self._clip(..., self.max_lateral_speed_m_s)` calls, but remove both post-floor statements:

```python
vy = self._clip(vy, self.max_lateral_speed_m_s)
```

Pass the planner value when `RawServoRuntime` constructs the controller:

```python
max_lateral_speed_m_s=config.raw_max_lateral_speed_m_s,
```

- [ ] **Step 5: Run controller tests and verify GREEN**

Run:

```bash
.venv_inference/bin/python -m pytest -q gear_sonic/tests/test_base_pose_visual_servo.py
```

Expected: all tests pass, including the existing exact 0.30 m/s direction-preservation tests.

---

### Task 2: Apply ratio-preserving final transport bounds in the relay

**Files:**
- Modify: `gear_sonic/tests/test_lavira_reasan_contract.py:85-120`
- Modify: `gear_sonic/scripts/lavira_sonic_relay.py:27-68`
- Modify: `gear_sonic/scripts/lavira_sonic_relay.py:176-190`
- Modify: `gear_sonic/scripts/lavira_sonic_relay.py:270-278`

**Interfaces:**
- Consumes: `decode_velocity_command(raw: bytes | str, *, max_lateral_speed_m_s: float = 0.16)`.
- Produces: decoded `float32` velocity whose planar components retain their incoming ratio and satisfy `vx in [-0.5, 1.0]` plus `abs(vy) <= max_lateral_speed_m_s`.
- Produces: relay CLI option `--max-lateral-speed-m-s`, default `0.16`.

- [ ] **Step 1: Snapshot the dirty relay files**

Run:

```bash
cp -- gear_sonic/tests/test_lavira_reasan_contract.py /tmp/test_lavira_reasan_contract.py.before-proportional-relay
cp -- gear_sonic/scripts/lavira_sonic_relay.py /tmp/lavira_sonic_relay.py.before-proportional-relay
```

- [ ] **Step 2: Write failing relay tests**

Change the pure-lateral cases to `+/-0.16`, then add:

```python
def test_direct_relay_scales_both_linear_axes_to_preserve_ratio() -> None:
    decoded = decode_direct_velocity_command(
        build_reasan_velocity_message(
            VelocityCommand(0.30, 0.20, 0.0, 1.0), action="move"
        )
    )

    vx, vy, wz = map(float, decoded["velocity"])
    assert (vx, vy, wz) == pytest.approx((0.24, 0.16, 0.0))
    assert vx / vy == pytest.approx(0.30 / 0.20)
    assert np.hypot(vx, vy) < 0.30


def test_direct_relay_accepts_configurable_lateral_limit() -> None:
    decoded = decode_direct_velocity_command(
        build_reasan_velocity_message(
            VelocityCommand(0.30, -0.20, 0.0, 1.0), action="move"
        ),
        max_lateral_speed_m_s=0.10,
    )

    assert decoded["velocity"].tolist() == pytest.approx([0.15, -0.10, 0.0])


def test_direct_relay_scales_vy_when_vx_hits_transport_limit() -> None:
    decoded = decode_direct_velocity_command(
        build_reasan_velocity_message(
            VelocityCommand(2.0, 0.10, 0.0, 1.0), action="move"
        )
    )

    assert decoded["velocity"].tolist() == pytest.approx([1.0, 0.05, 0.0])


@pytest.mark.parametrize("limit", [0.0, -0.1, float("nan"), float("inf")])
def test_direct_relay_rejects_invalid_lateral_limit(limit: float) -> None:
    with pytest.raises(ValueError, match="lateral speed"):
        decode_direct_velocity_command(
            build_reasan_velocity_message(
                VelocityCommand(0.0, 0.1, 0.0, 1.0), action="move"
            ),
            max_lateral_speed_m_s=limit,
        )
```

- [ ] **Step 3: Run focused relay tests and verify RED**

Run:

```bash
.venv_teleop/bin/python -m pytest -q gear_sonic/tests/test_lavira_reasan_contract.py -k 'direct_relay'
```

Expected: the 0.16 expectation and mixed-axis ratio tests fail against component-wise 0.22 clipping; configurable calls raise `TypeError`.

- [ ] **Step 4: Implement uniform linear scaling and the CLI option**

Define:

```python
DEFAULT_MAX_LATERAL_SPEED_M_S = 0.16
COMMAND_LOWER = np.array([-0.5, -DEFAULT_MAX_LATERAL_SPEED_M_S, -1.0], dtype=np.float32)
COMMAND_UPPER = np.array([1.0, DEFAULT_MAX_LATERAL_SPEED_M_S, 1.0], dtype=np.float32)
```

Add a helper that validates the limit, computes one scale from the sign-appropriate `vx` bound and `abs(vy)` bound, multiplies both linear axes once, and clips only `wz` independently:

```python
def _bound_velocity_preserving_linear_ratio(
    velocity: np.ndarray, *, max_lateral_speed_m_s: float
) -> np.ndarray:
    limit = float(max_lateral_speed_m_s)
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("maximum lateral speed must be finite and positive")
    vx, vy, wz = map(float, velocity)
    scale = 1.0
    if vx > float(COMMAND_UPPER[0]):
        scale = min(scale, float(COMMAND_UPPER[0]) / vx)
    elif vx < float(COMMAND_LOWER[0]):
        scale = min(scale, float(COMMAND_LOWER[0]) / vx)
    if abs(vy) > limit:
        scale = min(scale, limit / abs(vy))
    return np.asarray(
        [vx * scale, vy * scale, np.clip(wz, COMMAND_LOWER[2], COMMAND_UPPER[2])],
        dtype=np.float32,
    )
```

Change the decoder signature to:

```python
def decode_velocity_command(
    raw: bytes | str,
    *,
    max_lateral_speed_m_s: float = DEFAULT_MAX_LATERAL_SPEED_M_S,
) -> dict[str, Any]:
```

Return the helper result instead of `np.clip(...)`. Add parser option:

```python
parser.add_argument(
    "--max-lateral-speed-m-s",
    type=float,
    default=DEFAULT_MAX_LATERAL_SPEED_M_S,
)
```

Validate it with `hz` and `timeout`, pass it into every `decode_velocity_command` call in `main`, and include it in the startup log.

- [ ] **Step 5: Run relay tests and verify GREEN**

Run:

```bash
.venv_teleop/bin/python -m pytest -q gear_sonic/tests/test_lavira_reasan_contract.py
```

Expected: all relay contract tests pass.

---

### Task 3: Propagate one launch value through planner and relay commands

**Files:**
- Modify: `gear_sonic/tests/test_lavira_planner.py:220-242`
- Modify: `gear_sonic/tests/test_base_pose_visual_servo.py:1618-1660`
- Modify: `gear_sonic/scripts/launch_inference.py:220-242`
- Modify: `gear_sonic/scripts/launch_inference.py:430-450`
- Modify: `gear_sonic/scripts/launch_inference.py:543-565`

**Interfaces:**
- Consumes: `InferenceLaunchConfig.base_pose_raw_max_lateral_speed_m_s: float = 0.16`.
- Produces: planner CLI `--raw-max-lateral-speed-m-s VALUE` and relay CLI `--max-lateral-speed-m-s VALUE` for `planner_input="base_pose"`.

- [ ] **Step 1: Snapshot dirty launcher files**

Run:

```bash
cp -- gear_sonic/tests/test_lavira_planner.py /tmp/test_lavira_planner.py.before-proportional-relay
cp -- gear_sonic/scripts/launch_inference.py /tmp/launch_inference.py.before-proportional-relay
```

- [ ] **Step 2: Write failing propagation tests**

Extend `test_raw_base_pose_launch_passes_safety_and_orientation_parameters`:

```python
base_pose_raw_max_lateral_speed_m_s=0.14,
```

```python
assert "--raw-max-lateral-speed-m-s 0.14" in planner
assert "--max-lateral-speed-m-s 0.14" in relay
```

Extend `test_raw_launch_default_standoff_is_point_eight_meters`:

```python
assert "--raw-max-lateral-speed-m-s 0.16" in command
```

- [ ] **Step 3: Run propagation tests and verify RED**

Run:

```bash
.venv_inference/bin/python -m pytest -q gear_sonic/tests/test_lavira_planner.py::test_raw_base_pose_launch_passes_safety_and_orientation_parameters gear_sonic/tests/test_base_pose_visual_servo.py::test_raw_launch_default_standoff_is_point_eight_meters
```

Expected: `InferenceLaunchConfig` rejects the new keyword and the default planner command lacks the option.

- [ ] **Step 4: Implement launch propagation**

Add:

```python
base_pose_raw_max_lateral_speed_m_s: float = 0.16
"""Maximum raw-servo lateral request and final relay lateral component (m/s)."""
```

Append to `_base_pose_planner_command`:

```python
f"--raw-max-lateral-speed-m-s "
f"{config.base_pose_raw_max_lateral_speed_m_s} "
```

Append to the base-pose branch of `build_reasan_planner_command`:

```python
f" --max-lateral-speed-m-s {config.base_pose_raw_max_lateral_speed_m_s}"
```

- [ ] **Step 5: Run launcher/controller/relay suites and verify GREEN**

Run:

```bash
.venv_inference/bin/python -m pytest -q gear_sonic/tests/test_lavira_planner.py gear_sonic/tests/test_base_pose_visual_servo.py gear_sonic/tests/test_lavira_reasan_contract.py
```

Expected: all tests pass.

---

### Task 4: Document final-stage semantics and verify the complete change

**Files:**
- Modify: `docs/base_pose_adjustment.md:148-190`

**Interfaces:**
- Documents: `base_pose_raw_max_lateral_speed_m_s`, controller minimum-speed scaling, and relay proportional scaling.

- [ ] **Step 1: Snapshot and update the operator documentation**

Run:

```bash
cp -- docs/base_pose_adjustment.md /tmp/base_pose_adjustment.md.before-proportional-relay
```

After the minimum-speed paragraph, document that the upper controller may emit `abs(vy) > 0.16` after its 0.30 m/s floor, and that the relay scales both axes with one factor to restore the final configured lateral limit. Include the `(0.30, 0.20) -> (0.24, 0.16)` example and state that final magnitude below 0.30 m/s is allowed.

- [ ] **Step 2: Run syntax validation**

Run:

```bash
python3 -m py_compile gear_sonic/utils/inference/base_pose_visual_servo.py gear_sonic/scripts/base_pose_planner.py gear_sonic/scripts/launch_inference.py gear_sonic/scripts/lavira_sonic_relay.py
```

Expected: exit status 0 with no output.

- [ ] **Step 3: Run the complete focused regression set**

Run:

```bash
.venv_inference/bin/python -m pytest -q gear_sonic/tests/test_base_pose_visual_servo.py gear_sonic/tests/test_lavira_reasan_contract.py gear_sonic/tests/test_lavira_planner.py gear_sonic/tests/test_base_pose_adjustment.py
```

Expected: all tests pass.

- [ ] **Step 4: Verify invariants and diff scope**

Run:

```bash
rg -n "0\.22|max_lateral_speed_m_s|raw_max_lateral_speed_m_s|max-lateral-speed-m-s" gear_sonic/utils/inference/base_pose_visual_servo.py gear_sonic/scripts/base_pose_planner.py gear_sonic/scripts/launch_inference.py gear_sonic/scripts/lavira_sonic_relay.py gear_sonic/tests/test_base_pose_visual_servo.py gear_sonic/tests/test_lavira_reasan_contract.py gear_sonic/tests/test_lavira_planner.py docs/base_pose_adjustment.md
git diff --check
git diff --stat
```

Expected: no speed-limit `0.22` remains in the target speed code/tests; unrelated horizontal-guard values may still be 0.22. Diff check reports no whitespace errors, and only intended hunks differ from the `/tmp` snapshots.

- [ ] **Step 5: Review changes against the saved snapshots**

Run one `diff -u` per target file against its `/tmp/*.before-proportional-relay` snapshot. Confirm every added/removed hunk implements configuration, retained 0.30 controller scaling, relay proportional scaling, tests, or documentation. Do not stage or commit these already-dirty production/test files.
