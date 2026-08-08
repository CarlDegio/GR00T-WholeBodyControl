# Base-Pose General Planner Prompt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make all three BasePose modes use the approved general manipulation-planning prompt, rename the final-plan anchor field to `alignment_anchor`, and ask the depth-query stage to cover the entire manipulation area whenever possible.

**Architecture:** Keep `build_base_pose_prompt` as the one common planning-prompt source for `rgb`, `rgbd`, and `rgb_depth_query`; retain the existing depth-specific evidence appended by the two depth modes. Change the singular final-plan field consistently across prompt, schema, validator, and tests, while leaving the separate plural depth-query field `manipulation_anchors` unchanged.

**Tech Stack:** Python 3, f-string prompt templates, JSON Schema dictionaries, pytest.

## Global Constraints

- Preserve runtime injection of the task array and the five existing planning camera parameters.
- Preserve the `rgbd` aligned depth-image section and `rgb_depth_query` numeric depth-evidence section.
- Use the approved common prompt text verbatim apart from dynamic task/camera values and Python f-string brace escaping.
- Rename only the singular final-plan field `manipulation_anchor`; retain depth-query `manipulation_anchors`.
- Add exactly this depth-query instruction: `Query the depth of the entire manipulation area whenever possible.`
- Preserve unrelated and pre-existing working-tree changes; review and stage hunks selectively.

---

### Task 1: Migrate the final-plan anchor contract

**Files:**
- Modify: `gear_sonic/tests/test_base_pose_adjustment.py`
- Modify: `gear_sonic/utils/inference/base_pose.py:39-87`

**Interfaces:**
- Consumes: `validate_base_pose_plan(plan: Any, ...) -> dict[str, Any]` and `BASE_POSE_OUTPUT_SCHEMA`.
- Produces: a final-plan `task_interpretation.alignment_anchor: str` contract used by all modes.

- [ ] **Step 1: Change the shared valid-plan fixture and add a rename regression test**

Change `plan()` to emit:

```python
"task_interpretation": {
    "primary_target": "cup",
    "secondary_targets": ["tray"],
    "alignment_anchor": "cup body and tray opening",
    "interaction_direction": "face the combined workspace",
    "selection_reason": "supports pickup and placement",
},
```

Add a focused test that proves the new field is accepted and the obsolete singular field is rejected:

```python
def test_final_plan_contract_uses_alignment_anchor() -> None:
    valid = plan()
    assert validate_base_pose_plan(valid)["task_interpretation"]["alignment_anchor"]

    obsolete = plan()
    interpretation = obsolete["task_interpretation"]
    interpretation["manipulation_anchor"] = interpretation.pop("alignment_anchor")
    with pytest.raises(BasePoseValidationError, match="task_interpretation"):
        validate_base_pose_plan(obsolete)
```

- [ ] **Step 2: Run the focused test and verify it fails before production changes**

Run:

```bash
pytest -q gear_sonic/tests/test_base_pose_adjustment.py::test_final_plan_contract_uses_alignment_anchor
```

Expected: FAIL because the production schema still requires `manipulation_anchor`.

- [ ] **Step 3: Rename the production contract**

In `gear_sonic/utils/inference/base_pose.py`, make both contract definitions use the same new key:

```python
TASK_INTERPRETATION_KEYS = {
    "primary_target",
    "secondary_targets",
    "alignment_anchor",
    "interaction_direction",
    "selection_reason",
}
```

and:

```python
"alignment_anchor": {"type": "string"},
```

Do not change `DEPTH_QUERY_SCHEMA` or `_validate_depth_queries`, where `manipulation_anchors` is intentionally plural.

- [ ] **Step 4: Run the focused contract test**

Run:

```bash
pytest -q gear_sonic/tests/test_base_pose_adjustment.py::test_final_plan_contract_uses_alignment_anchor
```

Expected: PASS.

### Task 2: Replace the common prompt and extend the depth-query prompt

**Files:**
- Modify: `gear_sonic/tests/test_base_pose_adjustment.py`
- Modify: `gear_sonic/utils/inference/base_pose.py:589-823`

**Interfaces:**
- Consumes: `BasePoseConfig`, `AlignedRGBDSnapshot`, `camera_parameters`, and existing depth-mode suffix assembly in `BasePoseRunner.run_once`.
- Produces: `build_base_pose_prompt(config, snapshot) -> str` with the approved common prompt and `build_depth_query_prompt(config, snapshot) -> str` with full-workspace depth guidance.

- [ ] **Step 1: Add generated-prompt contract coverage**

Import `build_depth_query_prompt` beside `build_base_pose_prompt`. Replace the old task-specific prompt assertions with a focused test using `BasePoseConfig(task="put cup in tray", camera_pitch_deg=-25.0)` and assert all of the following:

```python
assert "places the robot in a suitable pose for completing the manipulation task" in prompt
assert 'Task description:\n\n["put cup in tray"]' in prompt
assert '"camera_pitch_deg": -25.0' in prompt
assert "Identify the target manipulation area and the primary manipulation target." in prompt
assert "prefer a relatively large object that provides a stable and clear reference" in prompt
assert "## READY state" in prompt
assert '"status": "READY | ADJUST | UNSURE | UNSAFE"' in prompt
assert '"alignment_anchor": ""' in prompt
assert "alignment_anchor should describe the selected alignment reference" in prompt
assert "manipulation_anchor" not in prompt
assert "blue plastic basket" not in prompt
```

Keep assertions for the 0.3-meter translation minimum and 30-degree rotation minimum, updating their expected wording to the approved prompt exactly.

- [ ] **Step 2: Add depth-query selection prompt coverage**

Add:

```python
def test_depth_query_prompt_requests_entire_manipulation_area() -> None:
    prompt = build_depth_query_prompt(
        BasePoseConfig(task="put cup in tray", mode="rgb_depth_query"),
        snapshot(),
    )
    assert "Query the depth of the entire manipulation area whenever possible." in prompt
    assert "Return at most three tight bounding boxes" in prompt
```

- [ ] **Step 3: Update the two-mode final-prompt regression assertions**

In `test_depth_final_prompts_keep_current_rgb_rules_and_add_depth_evidence`, replace task-specific basket assertions with representative common rules from the approved prompt:

```python
for common_rule in (
    "Identify the target manipulation area and the primary manipulation target.",
    "## READY state",
    "Every `ROTATE_LEFT` or `ROTATE_RIGHT` command must specify an angle greater than or equal to 30 degrees.",
    "Apply status priority in the following order:",
    "`UNSAFE` > `UNSURE` > `READY` > `ADJUST`",
):
    assert common_rule in final_prompt
    assert final_prompt.index(common_rule) < depth_offset
```

- [ ] **Step 4: Run the prompt tests and verify the new expectations fail**

Run:

```bash
pytest -q \
  gear_sonic/tests/test_base_pose_adjustment.py::test_prompt_requires_point_three_meter_minimum_and_indirect_small_correction \
  gear_sonic/tests/test_base_pose_adjustment.py::test_depth_query_prompt_requests_entire_manipulation_area \
  gear_sonic/tests/test_base_pose_adjustment.py::test_depth_final_prompts_keep_current_rgb_rules_and_add_depth_evidence
```

Expected: FAIL on missing general prompt content and missing complete-manipulation-area depth instruction.

- [ ] **Step 5: Replace `build_base_pose_prompt` with the approved template**

Retain the existing dynamic setup:

```python
params = json.dumps(
    {
        "camera_height_m": config.camera_height_m,
        "camera_pitch_deg": config.camera_pitch_deg,
        "vertical_fov_deg": config.vertical_fov_deg,
        "pitch_reference": "robot_body_horizontal",
        "pitch_convention": "positive_upward",
    },
    ensure_ascii=False,
    indent=2,
)
task = json.dumps([config.task], ensure_ascii=False)
```

Replace the returned f-string body with the complete prompt approved in `docs/superpowers/specs/2026-08-07-base-pose-general-planner-prompt-design.md` and the originating user request. Preserve its Markdown headings, bullets, backticks, `READY` rules, status priority, field interpretation, and consistency constraints. Substitute only `{task}` and `{params}` at their input positions. Double every literal JSON brace in Python source (`{{` and `}}`) so the rendered prompt contains normal `{` and `}`. The JSON example must contain:

```json
"status": "READY | ADJUST | UNSURE | UNSAFE"
```

and:

```json
"alignment_anchor": ""
```

- [ ] **Step 6: Add the full-workspace depth instruction**

In `build_depth_query_prompt`, place this sentence immediately after the opening task/anchor interpretation sentence and before the bounding-box limit:

```text
Query the depth of the entire manipulation area whenever possible.
```

- [ ] **Step 7: Run the focused prompt tests**

Run the command from Step 4.

Expected: all selected tests PASS for both `rgbd` and `rgb_depth_query` parameter cases.

### Task 3: Verify all modes and stale-field cleanup

**Files:**
- Verify: `gear_sonic/utils/inference/base_pose.py`
- Verify: `gear_sonic/tests/test_base_pose_adjustment.py`

**Interfaces:**
- Consumes: the new prompt and final-plan contract from Tasks 1-2.
- Produces: evidence that all BasePose modes construct prompts, validate model output, and preserve their depth-specific evidence.

- [ ] **Step 1: Run the complete BasePose adjustment test module**

Run:

```bash
pytest -q gear_sonic/tests/test_base_pose_adjustment.py
```

Expected: PASS with no failures.

- [ ] **Step 2: Compile the modified Python files**

Run:

```bash
python3 -m py_compile \
  gear_sonic/utils/inference/base_pose.py \
  gear_sonic/tests/test_base_pose_adjustment.py
```

Expected: exit status 0 with no output.

- [ ] **Step 3: Check for stale singular contract references**

Run:

```bash
rg -n 'manipulation_anchor|alignment_anchor' \
  gear_sonic/utils/inference/base_pose.py \
  gear_sonic/tests/test_base_pose_adjustment.py
```

Expected: `alignment_anchor` appears in the final-plan schema, prompt, validator fixture, and tests. Any remaining `manipulation_anchors` references are plural and belong only to the depth-query selection schema/fixtures/validator; the only obsolete singular use is the intentional negative regression test.

- [ ] **Step 4: Review the final diff without disturbing pre-existing changes**

Run:

```bash
git diff -- \
  gear_sonic/utils/inference/base_pose.py \
  gear_sonic/tests/test_base_pose_adjustment.py
```

Confirm that the final diff preserves the existing f-string brace repair and depth-final-prompt regression work while replacing the obsolete task-specific prompt and updating its contract consistently. Do not stage or commit unrelated files.
