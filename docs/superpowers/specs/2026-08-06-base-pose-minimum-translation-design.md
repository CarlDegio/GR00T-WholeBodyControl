# Base-Pose Minimum Translation Prompt Design

## Goal

Prevent the BasePose model from emitting forward or backward commands shorter than the robot's useful minimum translation while still allowing it to plan small net pose corrections through a safe indirect trajectory.

## Prompt changes

Add two rules to `Motion-planning rules`:

- Every `MOVE_FORWARD` and `MOVE_BACKWARD` command must specify a distance greater than or equal to `0.3` meters. A value of exactly `0.3` meters is valid.
- When the desired positional correction is less than `0.3` meters, the model must not emit a sub-threshold translation. When geometrically appropriate and safe, it should prefer an indirect sequence consisting of backward movement, rotation, forward movement, and a final corrective rotation toward the manipulation target.

Add one rule to `Consistency constraints`:

- Every `MOVE_FORWARD` and `MOVE_BACKWARD` value must be greater than or equal to `0.3` meters.

## Scope

Only the text produced by `build_base_pose_prompt` changes. The JSON schema, runtime validator, planner execution, motion speeds, model defaults, timeout, keyboard controls, relay behavior, and external task injection remain unchanged. The existing uncommitted prompt rewrite is preserved.

## Verification

- Build a real BasePose prompt and assert that all three approved rules appear in the generated text.
- Compile `gear_sonic/utils/inference/base_pose.py` with `py_compile`.
- Review the working diff to confirm that no existing prompt content or unrelated dirty files were changed beyond the three inserted rules.
