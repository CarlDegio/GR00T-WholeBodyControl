# Base-Pose Configurable Completion Tolerances Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the raw-servo's hard-coded 4 cm forward and 3 cm lateral completion thresholds with independently configurable 10 cm defaults.

**Architecture:** Add two float fields to the launcher and planner dataclasses, forward them through the generated command, and inject them into `VisualServoController`. Use the same validated values for translation deadbands and the existing five-frame completion counter.

**Tech Stack:** Python dataclasses, Tyro CLI generation, pytest

## Global Constraints

- Default forward completion tolerance is exactly `0.10 m`.
- Default lateral completion tolerance is exactly `0.10 m`.
- The two tolerances remain independently configurable.
- Preserve all unrelated uncommitted worktree changes.
- Do not change yaw, timeout, tracking-loss, or stable-frame-count behavior.

---

### Task 1: Controller tolerance API and behavior

**Files:**
- Modify: `gear_sonic/tests/test_base_pose_visual_servo.py`
- Modify: `gear_sonic/utils/inference/base_pose_visual_servo.py`

**Interfaces:**
- Consumes: `VisualServoController(forward_tolerance_m: float = 0.10, lateral_tolerance_m: float = 0.10)`
- Produces: validated `forward_tolerance_m` and `lateral_tolerance_m` attributes used by `TRANSLATE_TARGET`

- [ ] Add tests proving 9 cm errors complete with defaults, independently configured boundaries work, and non-positive/non-finite tolerances are rejected.
- [ ] Run the focused tests and confirm they fail because the constructor fields do not exist and the old thresholds reject 9 cm.
- [ ] Add the two validated constructor arguments and replace both hard-coded completion/deadband thresholds.
- [ ] Re-run the focused tests and confirm they pass.

### Task 2: Runtime and launcher propagation

**Files:**
- Modify: `gear_sonic/tests/test_base_pose_visual_servo.py`
- Modify: `gear_sonic/tests/test_lavira_planner.py`
- Modify: `gear_sonic/scripts/base_pose_planner.py`
- Modify: `gear_sonic/scripts/launch_inference.py`

**Interfaces:**
- Consumes: `BasePosePlannerConfig.raw_forward_tolerance_m` and `raw_lateral_tolerance_m`
- Produces: `InferenceLaunchConfig.base_pose_raw_forward_tolerance_m`, `base_pose_raw_lateral_tolerance_m`, and matching planner command flags

- [ ] Add tests for runtime injection and default/custom launcher command flags.
- [ ] Run the focused tests and confirm missing fields/flags cause the expected failures.
- [ ] Add dataclass fields, command forwarding, and runtime constructor arguments.
- [ ] Re-run focused tests and confirm they pass.

### Task 3: Documentation and regression verification

**Files:**
- Modify: `docs/base_pose_adjustment.md`

**Interfaces:**
- Consumes: the two launcher flags and their `0.10 m` defaults
- Produces: operator-facing configuration documentation

- [ ] Document the 10 cm defaults, five-frame rule, and both launcher arguments.
- [ ] Run raw-servo and launcher test modules.
- [ ] Run Python compilation checks for every modified Python file.
- [ ] Review the final diff to confirm unrelated worktree changes were preserved.
