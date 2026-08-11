# Base-Pose Distance and Runtime Defaults Design

## Goal

Change the raw YOLOE visual-servo defaults to a target standoff distance of
0.80 m and a maximum execution time of 60 seconds.

## Design

All default-producing layers will use the same values:

- `InferenceLaunchConfig.base_pose_raw_target_distance_m` will default to
  `0.80` so the all-in-one launcher passes the new standoff to BasePose.
- `BasePosePlannerConfig.raw_target_distance_m` will default to `0.80` and
  `BasePosePlannerConfig.raw_max_run_s` will default to `60.0` so direct
  `base_pose_planner.py` launches behave identically.
- `VisualServoController` constructor defaults will be `target_distance_m=0.80`
  and `max_run_s=60.0` so direct library use and tests do not silently retain
  the old behavior.

Explicit command-line overrides remain authoritative. No controller gains,
deadbands, speed limits, perception behavior, or phase transitions change.

## Runtime Behavior

The forward error remains `filtered_forward_m - target_distance_m`, so the
controller will stop translating after five stable frames within its existing
distance and lateral tolerances around 0.80 m. If alignment does not complete,
the controller will publish its existing stop sequence once elapsed execution
time reaches 60 seconds.

## Tests and Documentation

- Update the runtime-default test to assert a 60-second limit.
- Add or update default-config assertions for the 0.80 m launcher and planner
  values.
- Preserve tests that pass explicit target-distance overrides, proving CLI and
  configuration overrides still work.
- Update the BasePose launch example and behavior description from 0.60 m to
  0.80 m, and document the 60-second maximum execution time.

## Non-goals

- Changing the lateral speed limit, the 0.30 m/s minimum translation magnitude,
  or yaw behavior.
- Adding a proximity hard stop or changing table-edge collision handling.
- Changing non-raw BasePose modes or LaViRA navigation defaults.
