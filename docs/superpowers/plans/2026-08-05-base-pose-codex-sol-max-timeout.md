# Base-Pose Codex Sol/Max/Timeout Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task by task.

**Goal:** Make BasePose use the explicit local Codex subscription model `gpt-5.6-sol`, reasoning effort `max`, and a 600-second Codex subprocess timeout by default.

**Architecture:** Preserve the existing CLI override path while changing the same defaults at every construction boundary: the launch configuration, planner configuration, inference configuration, and direct structured-vision client. Lock the launch contract with a focused test and keep runtime behavior otherwise unchanged.

**Tech Stack:** Python dataclasses, argparse command construction, subprocess-based Codex CLI integration, pytest-style unit tests, Markdown documentation.

**Scope constraints:** Do not alter the BasePose prompt, task injection, camera parameters, proxy/authentication behavior, keyboard state machine, relay behavior, retries, model fallbacks, or live robot processes.

---

### Task 1: Lock the launch command defaults with a failing test

**Files:**
- Modify: `gear_sonic/tests/test_lavira_planner.py`
- Test: `gear_sonic/tests/test_lavira_planner.py`

**Step 1: Update the focused command-construction assertion**

In the existing BasePose planner command test, require the generated command to contain:

```python
assert command[command.index("--model") + 1] == "gpt-5.6-sol"
assert command[command.index("--reasoning-effort") + 1] == "max"
assert command[command.index("--codex-timeout-seconds") + 1] == "600.0"
```

Keep the external `--task` assertion unchanged.

**Step 2: Run the focused assertion and verify RED**

Run a small Python assertion script that imports `InferenceLaunchConfig` and `build_planner_input_command`, builds the default BasePose command, and checks the three expected argument values.

Expected: FAIL because the implementation still emits the prior model, effort, and timeout defaults.

### Task 2: Change the defaults at every BasePose construction boundary

**Files:**
- Modify: `gear_sonic/scripts/launch_inference.py`
- Modify: `gear_sonic/scripts/base_pose_planner.py`
- Modify: `gear_sonic/utils/inference/base_pose.py`

**Step 1: Update launcher defaults**

Set `InferenceLaunchConfig.base_pose_model` to `gpt-5.6-sol`, `base_pose_reasoning_effort` to `max`, and `base_pose_codex_timeout_seconds` to `600.0`.

**Step 2: Update planner defaults**

Set the matching fields in `BasePosePlannerConfig` to the same values.

**Step 3: Update inference/client defaults**

Set the matching fields in `BasePoseConfig` and the matching keyword defaults in `CodexStructuredVisionClient.__init__` to the same values.

**Step 4: Run focused assertions and verify GREEN**

Run software-only assertions covering all four default-bearing classes plus the generated launch command.

Expected: PASS with `gpt-5.6-sol`, `max`, and `600.0` consistently propagated.

### Task 3: Document the new explicit defaults

**Files:**
- Modify: `docs/base_pose_adjustment.md`

**Step 1: Update examples and default descriptions**

Replace stale BasePose model, effort, and timeout defaults with `gpt-5.6-sol`, `max`, and `600` seconds. Preserve the externally supplied task example and all unrelated documentation.

**Step 2: Check for stale values in BasePose default contexts**

Use `rg` across the four implementation/test files and the BasePose document, then distinguish intentional override examples from stale default documentation.

### Task 4: Verify and commit only the approved scope

**Files:**
- Verify: `gear_sonic/scripts/launch_inference.py`
- Verify: `gear_sonic/scripts/base_pose_planner.py`
- Verify: `gear_sonic/utils/inference/base_pose.py`
- Verify: `gear_sonic/tests/test_lavira_planner.py`
- Verify: `docs/base_pose_adjustment.md`

**Step 1: Run syntax validation**

Run:

```bash
python3 -m py_compile gear_sonic/scripts/launch_inference.py gear_sonic/scripts/base_pose_planner.py gear_sonic/utils/inference/base_pose.py gear_sonic/tests/test_lavira_planner.py
```

Expected: exit code 0.

**Step 2: Run focused software-only behavior checks**

Run the launch-command and direct-default assertion scripts again.

Expected: all assertions pass; no Codex request and no robot command is issued.

**Step 3: Review repository changes**

Run `git diff --check`, inspect the exact diff, and confirm prompt text, relay/keyboard behavior, proxy/authentication, and live processes are untouched.

**Step 4: Commit only this implementation**

Stage only the model/effort/timeout defaults, their test assertions, and their documentation. Commit with:

```bash
git commit -m "fix: use sol max defaults for base-pose"
```
