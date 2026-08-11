# Base-Pose Proportional Relay Lateral Limit Design

## Goal

Make the base-pose lateral speed limit configurable, with a default of
0.16 m/s, while preserving the existing 0.30 m/s minimum nonzero translation
behavior in the upper-level controller. The final LaViRA-to-SONIC relay must
enforce the lateral limit without changing the ratio between `vx` and `vy`.

## Configuration

A single base-pose configuration value will provide the maximum lateral speed
in meters per second. Its default is 0.16. The launcher will pass the value to
both the base-pose planner/controller path and the direct SONIC relay path so
that a command-line override changes both stages together.

The controller and relay will reject a non-finite or non-positive configured
limit rather than silently accepting an unsafe value.

## Controller Behavior

The visual-servo controller keeps its existing minimum-speed rule: every
nonzero translation request is scaled as a two-dimensional `(vx, vy)` vector
to a magnitude of at least 0.30 m/s. This upper-level operation preserves the
request's direction.

The configurable lateral value limits the controller's proportional lateral
request before minimum-speed scaling. The controller will not independently
clip `vy` after minimum-speed scaling, because doing so would change the
`vx:vy` ratio and would prevent the relay from seeing the scaled vector that it
must bound proportionally.

The existing fixed `vx = 0.30 m/s` vertical-reacquisition command is unchanged.

## Relay Behavior

The direct relay is the final lateral safety boundary. After decoding a valid
velocity vector, it applies these rules:

1. If `abs(vy)` is at or below the configured lateral limit, leave `vx` and
   `vy` unchanged, subject to the relay's existing forward/backward bounds.
2. If `abs(vy)` exceeds the lateral limit, compute
   `scale = lateral_limit / abs(vy)` and multiply both `vx` and `vy` by this
   same scale.
3. Apply any other required linear transport bound with uniform vector scaling
   as well, so no component-wise linear clipping can change direction.
4. Continue to clamp `wz` independently because angular velocity is not part
   of the planar translation ratio.

Because the relay operates after the controller's 0.30 m/s minimum-speed
scaling, proportional safety scaling may produce a final linear speed below
0.30 m/s. This is explicitly allowed.

For example, a relay input of `(vx=0.30, vy=0.20)` with the default limit is
scaled by `0.16 / 0.20 = 0.8` and becomes `(vx=0.24, vy=0.16)`. The direction
ratio remains `0.30:0.20` even though the final magnitude is lower than the
controller's input magnitude.

## Data Flow

1. The launcher supplies one lateral-limit value to the planner and relay.
2. Base-pose computes its proportional translation request.
3. Base-pose raises a nonzero request to the existing 0.30 m/s minimum while
   preserving the vector ratio.
4. The relay receives that command and proportionally scales the complete
   linear vector only when a final transport bound requires it.
5. SONIC receives the bounded `(vx, vy, wz)` command.

## Tests

- Controller tests verify that nonzero translation is still raised to at least
  0.30 m/s before relay handling.
- Controller tests verify that its lateral request limit defaults to 0.16 m/s
  and accepts an explicit override.
- Relay tests verify positive and negative lateral bounds at 0.16 m/s.
- A mixed-axis relay test verifies that `vx` and `vy` receive the same scale
  factor and that the final magnitude may be below 0.30 m/s.
- Launcher tests verify that one configured value is passed to both planner and
  relay commands.
- Existing base-pose and relay suites guard phase transitions, stop commands,
  yaw handling, and malformed-input behavior.

## Non-goals

- Removing or reducing the upper-level 0.30 m/s minimum translation behavior.
- Changing the fixed vertical-reacquisition forward command.
- Coupling angular velocity scaling to planar translation scaling.
- Changing base-pose phase transitions, tracking-loss behavior, or collision
  avoidance.
