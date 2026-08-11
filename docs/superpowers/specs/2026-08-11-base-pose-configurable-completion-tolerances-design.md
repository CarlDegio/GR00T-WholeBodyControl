# Base-Pose Configurable Completion Tolerances Design

## Goal

Make the raw YOLOE base-pose servo's forward and lateral completion tolerances
independently configurable, with both defaults changed to `0.10 m`.

## Configuration and data flow

`InferenceLaunchConfig` exposes
`base_pose_raw_forward_tolerance_m` and
`base_pose_raw_lateral_tolerance_m`. The base-pose launcher forwards them as
`--raw-forward-tolerance-m` and `--raw-lateral-tolerance-m` to matching
`BasePosePlannerConfig` fields. `RawServoRuntime` passes both values into
`VisualServoController`.

Both values must be finite and strictly positive. The controller defaults are
also `0.10 m`, so direct controller and standalone planner use match the main
launcher defaults.

## Controller behavior

In `TRANSLATE_TARGET`, the filtered forward error is stable when its absolute
value is at most `forward_tolerance_m`, and the filtered lateral error is stable
when its absolute value is at most `lateral_tolerance_m`. Both must remain
stable for the existing five consecutive applied visual frames.

The same two tolerances define the corresponding velocity deadbands. This
prevents the controller from continuing to command motion inside the region it
already considers complete.

No yaw thresholds, stable-frame counts, timeout behavior, tracking-loss
behavior, or non-raw base-pose modes change.

## Verification

Tests cover the new `0.10 m` defaults, independent custom values, validation of
invalid values, runtime propagation, launcher command propagation, and the
five-frame completion boundary. The raw-servo tests and base-pose launcher
tests must pass after the change.
