# Base-Pose Distance and Runtime Defaults Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make raw YOLOE BasePose alignment default to a 0.80 m target standoff and a 60-second maximum execution time at every default-producing layer.

**Architecture:** Keep the existing configuration flow: the all-in-one launcher emits the target-distance argument, `BasePosePlannerConfig` supplies direct-run defaults, and `RawServoRuntime` injects those values into `VisualServoController`. Synchronize constructor defaults as a safe library fallback while preserving all explicit overrides.

**Tech Stack:** Python 3.10 dataclasses, Tyro CLI configuration, pytest, Markdown documentation.

## Global Constraints

- Default target standoff is exactly `0.80 m`.
- Default maximum raw-servo execution time is exactly `60.0 s`.
- Explicit configuration and command-line overrides remain authoritative.
- Do not change controller gains, deadbands, speed limits, perception, phase transitions, or non-raw BasePose modes.
- Preserve all pre-existing uncommitted work and do not commit already-dirty production or test files.

---

### Task 1: Change and verify all runtime defaults

**Files:**
- Modify: `gear_sonic/tests/test_base_pose_visual_servo.py`
- Modify: `gear_sonic/scripts/base_pose_planner.py:55-90`
- Modify: `gear_sonic/scripts/launch_inference.py:220-230`
- Modify: `gear_sonic/utils/inference/base_pose_visual_servo.py:755-770`

**Interfaces:**
- Consumes: `InferenceLaunchConfig`, `BasePosePlannerConfig`, `build_planner_input_command`, `VisualServoController.update`, and `RawServoRuntime`.
- Produces: default launcher command containing `--raw-target-distance-m 0.8`, controller convergence around 0.80 m, and timeout at 60 seconds.

- [ ] **Step 1: Write failing behavior tests**

Change the shared aligned-observation fixture default to `forward=0.80`. Replace the old forty-second runtime test with:

```python
def test_raw_servo_defaults_use_point_eight_standoff_and_sixty_seconds(
    tmp_path: Path,
) -> None:
    runtime = RawServoRuntime(
        BasePosePlannerConfig(task="align", output_root=str(tmp_path)),
        publish=lambda _: None,
        logger=lambda _: None,
    )
    runtime.controller.reset(1.0)
    runtime.controller.phase = ServoPhase.TRANSLATE_TARGET

    for index in range(5):
        command = runtime.controller.update(
            observation(forward=0.80), now=1.1 + index * 0.1
        )
    assert command.velocity == (0.0, 0.0, 0.0)
    assert runtime.controller.terminal_reason == "aligned"

    runtime.controller.reset(1.0)
    runtime.controller.phase = ServoPhase.TRANSLATE_TARGET
    runtime.controller.update(observation(forward=1.0), now=60.9)
    assert not runtime.controller.terminal
    runtime.controller.update(observation(forward=1.0), now=61.0)
    assert runtime.controller.terminal_reason == "maximum run time reached"
```

Add a launcher data-flow test:

```python
def test_raw_launch_default_standoff_is_point_eight_meters() -> None:
    config = InferenceLaunchConfig(
        planner_input="base_pose",
        base_pose_mode="raw_yoloe_servo",
        base_pose_task="align",
    )

    command = build_planner_input_command(config, Path("/workspace/sonic"))

    assert "--raw-target-distance-m 0.8" in command
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
source .venv_inference/bin/activate
pytest -q \
  gear_sonic/tests/test_base_pose_visual_servo.py::test_raw_servo_defaults_use_point_eight_standoff_and_sixty_seconds \
  gear_sonic/tests/test_base_pose_visual_servo.py::test_raw_launch_default_standoff_is_point_eight_meters
```

Expected: FAIL because current defaults are `0.60 m` and `40.0 s`.

- [ ] **Step 3: Implement the synchronized defaults**

Apply these exact defaults:

```python
# gear_sonic/scripts/base_pose_planner.py
raw_target_distance_m: float = 0.80
raw_max_run_s: float = 60.0

# gear_sonic/scripts/launch_inference.py
base_pose_raw_target_distance_m: float = 0.80

# gear_sonic/utils/inference/base_pose_visual_servo.py
target_distance_m: float = 0.80
max_run_s: float = 60.0
```

Do not change the existing `RawServoRuntime` dependency injection because it already passes `config.raw_target_distance_m` and `config.raw_max_run_s` into the controller.

- [ ] **Step 4: Run the focused and controller tests and verify GREEN**

Run:

```bash
source .venv_inference/bin/activate
pytest -q gear_sonic/tests/test_base_pose_visual_servo.py
```

Expected: all tests pass; explicit `0.65 m` launcher override coverage remains green.

### Task 2: Update operator documentation and run regression verification

**Files:**
- Modify: `docs/base_pose_adjustment.md:60-75`
- Modify: `docs/base_pose_adjustment.md:135-150`

**Interfaces:**
- Consumes: the Task 1 defaults.
- Produces: launch and behavior documentation that states `0.80 m` and `60 s`.

- [ ] **Step 1: Update the launch example and behavior text**

Change the launch example to:

```bash
--base-pose-raw-target-distance-m 0.80
```

Change the `TRANSLATE_TARGET` description to state `0.80 m` standoff, and add one sentence stating that raw-servo execution stops after 60 seconds by default if alignment has not completed.

- [ ] **Step 2: Run regression and static verification**

Run:

```bash
source .venv_inference/bin/activate
pytest -q gear_sonic/tests
python -m py_compile \
  gear_sonic/scripts/base_pose_planner.py \
  gear_sonic/scripts/launch_inference.py \
  gear_sonic/utils/inference/base_pose_visual_servo.py
git diff --check
```

Expected: the complete `gear_sonic` test subsystem passes; syntax and diff checks exit zero. Existing third-party matplotlib/pyparsing deprecation warnings may remain unchanged.

- [ ] **Step 3: Review only relevant final values**

Run:

```bash
rg -n 'raw_target_distance_m|base_pose_raw_target_distance_m|target_distance_m: float|raw_max_run_s|max_run_s: float|0\.60 m|0\.80 m|60 seconds' \
  gear_sonic/scripts/base_pose_planner.py \
  gear_sonic/scripts/launch_inference.py \
  gear_sonic/utils/inference/base_pose_visual_servo.py \
  docs/base_pose_adjustment.md \
  gear_sonic/tests/test_base_pose_visual_servo.py
```

Confirm that default-producing entries use `0.80/60.0`, explicit non-default test fixtures remain unchanged, and unrelated numeric values are not rewritten.
