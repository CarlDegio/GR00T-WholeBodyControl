# Base-Pose Lateral Speed Limit Design

## Goal

Raise the maximum commanded lateral speed from 0.16 m/s to 0.22 m/s while
keeping the visual-servo command log consistent with the command accepted by
the SONIC relay.

## Design

The visual-servo controller will use 0.22 m/s as its lateral component limit.
Because the controller subsequently raises every nonzero translation vector to
the 0.30 m/s minimum linear magnitude, it will clamp `vy` to +/-0.22 m/s again
after that scaling step. Zero translation remains zero, and the existing
minimum-speed behavior for the remaining vector is otherwise unchanged.

The direct LaViRA-to-SONIC relay will independently clamp `vy` to +/-0.22 m/s.
This is the final transport safety boundary and protects SONIC from other
velocity-command producers that request a larger lateral component. The relay's
existing `vx` and `wz` limits remain unchanged.

## Data Flow

1. Base-pose computes proportional `vx` and `vy` requests.
2. It applies the 0.30 m/s nonzero minimum translation magnitude.
3. It clamps the resulting lateral component to +/-0.22 m/s and records and
   publishes that bounded command.
4. The relay decodes the JSON command and clamps `vy` to +/-0.22 m/s again.
5. The relay converts the bounded vector into SONIC movement direction and
   speed.

## Tests

- A controller test will request pure lateral recentering and assert that its
  observable output is capped at 0.22 m/s rather than the pre-limit 0.30 m/s.
- Relay contract tests will send positive and negative lateral commands beyond
  the limit and assert that decoded output is capped at +/-0.22 m/s.
- Existing base-pose and relay suites will run to guard all other control
  behavior.

## Non-goals

- Changing the 0.30 m/s minimum translation magnitude.
- Changing forward/backward or yaw limits.
- Adding collision avoidance or changing controller phase transitions.
