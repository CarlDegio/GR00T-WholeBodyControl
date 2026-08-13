# Remove NavDP Radar Hard Stop

## Goal

Remove the NavDP radar-proximity hard-stop behavior because it is not enabled in
the deployed system and its presence in the code implies a safety behavior that
operators do not actually receive.

## Scope

The change removes only the MID-360 point-cloud proximity hard stop:

- Remove the function that turns a motion command into zero velocity when a
  radar point falls within the hard-stop distance and motion sector.
- Remove the function that aborts active navigation after that radar hard stop.
- Remove the control-loop state, status reason, diagnostics, exports, and unit
  tests that exist only for the radar proximity hard stop.

The change preserves:

- The `radar_timeout_s` freshness check and its `radar_timeout` stop reason.
- Radar ingestion, point processing, ActorRay generation, SLAM/localization,
  visualization, and recording.
- The depth-camera stop behavior: `depth_requires_stop` may still zero the
  outgoing velocity command.
- Existing odometry, trajectory, MPC freshness, explicit zero-action, and goal
  completion behavior.

## Control Flow

The NavDP loop will continue to compute a candidate velocity and apply freshness
checks. If the current depth frame requires a stop, the outgoing velocity will
be set to zero. Radar points will still be converted to ActorRay data for
recording and visualization, but they will no longer modify the velocity or
abort navigation based on proximity.

Navigation may still be stopped by an explicit zero NavDP/MPC action. The loop
will no longer emit `lidar_hard_stop` or `radar_hard_stop` for proximity events.
Fresh radar data remains mandatory because `radar_timeout_s` is unchanged.

## Code Boundaries

- `gear_sonic/navdp/control.py`: delete radar hard-stop helpers.
- `gear_sonic/scripts/navdp_planner.py`: remove their imports, exports, call
  sites, abort path, and proximity-only diagnostic state while retaining depth
  stop and radar timeout behavior.
- `gear_sonic/tests/test_navdp_planner.py`: remove obsolete helper tests and add
  regression coverage for the resulting public/control-loop contract.

No launch configuration or sensor-gateway schema changes are required.

## Verification

Automated verification will cover:

1. The removed radar hard-stop helpers are no longer exposed by the NavDP
   planner module.
2. The planner source has no radar/lidar proximity hard-stop status path.
3. `radar_timeout_s` remains in the runtime configuration and launch command.
4. Depth-camera stop behavior remains covered by its existing unit tests.
5. The targeted NavDP, runtime-config, and launcher test suites pass.

The completed branch will be left unmerged for end-to-end robot testing by the
user.
