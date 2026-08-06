# Base-Pose Minimum Translation Prompt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Require every model-authored forward/backward command to be at least `0.3 m` and guide sub-`0.3 m` net corrections toward an indirect repositioning trajectory.

**Architecture:** Change only the text returned by `build_base_pose_prompt`. Add a real generated-prompt contract test so the lower bound, the indirect small-correction strategy, and the final consistency constraint cannot disappear independently.

**Tech Stack:** Python, NumPy fixture data, pytest-style contract test, `py_compile`.

## Global Constraints

- A translation value of exactly `0.3` meters is valid.
- No `MOVE_FORWARD` or `MOVE_BACKWARD` command in the prompt contract may be below `0.3` meters.
- For desired positional corrections below `0.3` meters, prefer a safe, geometrically appropriate backward/rotation/forward/final-rotation sequence instead of a sub-threshold translation.
- Do not change the JSON schema, runtime validator, execution speeds, model defaults, timeout, keyboard/relay behavior, or task injection.
- Preserve the current user-authored prompt rewrite and all unrelated dirty files.

---

### Task 1: Enforce the minimum translation in the generated BasePose prompt

**Files:**
- Modify: `gear_sonic/tests/test_base_pose_adjustment.py`
- Modify: `gear_sonic/utils/inference/base_pose.py:652-722`

**Interfaces:**
- Consumes: `build_base_pose_prompt(config: BasePoseConfig, snapshot: AlignedRGBDSnapshot) -> str`
- Produces: Generated prompt text containing all three approved minimum-translation rules.

- [ ] **Step 1: Write the failing generated-prompt contract test**

Import `build_base_pose_prompt` and add:

```python
def test_prompt_requires_point_three_meter_minimum_and_indirect_small_correction() -> None:
    snapshot = AlignedRGBDSnapshot(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth_raw=None,
        fx=1.0,
        fy=1.0,
        cx=0.5,
        cy=0.5,
        depth_scale_m=None,
        depth_aligned_to=None,
        depth_source=None,
        timestamp=1.0,
    )

    prompt = build_base_pose_prompt(BasePoseConfig(task="adjust pose"), snapshot)

    assert (
        "Every MOVE_FORWARD or MOVE_BACKWARD command must specify a distance "
        "greater than or equal to 0.3 meters."
    ) in prompt
    assert (
        "If the desired positional correction is less than 0.3 meters, do not "
        "output a translation below 0.3 meters."
    ) in prompt
    assert (
        "Every MOVE_FORWARD or MOVE_BACKWARD value must be greater than or "
        "equal to 0.3 meters."
    ) in prompt
```

- [ ] **Step 2: Run the focused test and verify RED**

Because the existing environments do not contain pytest, load the test file with a minimal local `pytest.mark.parametrize` stub and invoke only `test_prompt_requires_point_three_meter_minimum_and_indirect_small_correction`.

Expected: FAIL on the first missing prompt string.

- [ ] **Step 3: Add the two motion-planning rules**

Insert under `## Motion-planning rules`:

```text
* Every MOVE_FORWARD or MOVE_BACKWARD command must specify a distance greater than or equal to 0.3 meters.
* If the desired positional correction is less than 0.3 meters, do not output a translation below 0.3 meters. When geometrically appropriate and safe, prefer an indirect adjustment using backward movement, rotation, forward movement, and a final corrective rotation toward the manipulation target.
```

- [ ] **Step 4: Add the consistency constraint**

Insert under `## Consistency constraints`:

```text
* Every MOVE_FORWARD or MOVE_BACKWARD value must be greater than or equal to 0.3 meters.
```

- [ ] **Step 5: Run the focused test and verify GREEN**

Run the same software-only focused test invocation.

Expected: PASS.

- [ ] **Step 6: Run syntax and diff verification**

Run:

```bash
python3 -m py_compile gear_sonic/utils/inference/base_pose.py gear_sonic/tests/test_base_pose_adjustment.py
git diff --check
```

Expected: both commands exit `0`; no files outside the prompt, its focused test, and workflow documents are modified by this task.

- [ ] **Step 7: Commit the verified prompt contract**

Stage the final user-authored BasePose prompt and its focused test, leaving unrelated untracked files untouched, then commit:

```bash
git commit -m "feat: enforce base-pose translation prompt minimum"
```
