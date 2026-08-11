# Remove Base-Pose Target Position Jump Handling

## Goal

Ensure `raw_yoloe_servo` never treats a change in the estimated target
`(forward_m, right_m)` position as an invalid frame and can never terminate a
run because of `target position jump`.

## Design

Remove the controller's previous-target-position state and the fixed `0.15 m`
inter-frame jump check. Every valid target observation flows directly into the
existing position EMA, regardless of its distance from the previous estimate.

The existing safety and smoothing mechanisms remain unchanged:

- target positions use the existing EMA with `ema_alpha=0.35`;
- forward and lateral requests retain their deadbands and speed limits;
- horizontal and vertical visibility guards retain their current behavior;
- missing target or required-table observations still publish zero immediately
  and terminate only on the 30th consecutive soft-invalid frame;
- hard tracking mismatches, the 60-second deadline, and operator cancellation
  retain their current behavior.

No replacement jump threshold, jump counter, one-frame jump hold, or jump-only
diagnostic state is introduced.

## Data Flow

For every valid observation, `_update_filter()` constructs the raw target and
yaw sample, applies the existing EMA, and returns the filtered control errors.
Only perception failures passed through `note_invalid()` may increment
`invalid_frames`. A valid observation resets the tracking-loss streak through
the controller's existing phase logic.

## Tests

Add a controller regression test that supplies repeated valid observations
whose positions differ by more than `0.15 m`. The test must prove that the
controller continues processing them, keeps `invalid_frames` at zero, and does
not become terminal. Existing tracking-loss tests continue to prove that real
missing observations still terminate on the 30th consecutive soft-invalid
frame.

## Scope

Only the raw YOLOE base-pose controller and its focused tests change. Perception,
bbox tracking, target reference updates, non-raw base-pose modes, relay logic,
and runtime limits are out of scope.
