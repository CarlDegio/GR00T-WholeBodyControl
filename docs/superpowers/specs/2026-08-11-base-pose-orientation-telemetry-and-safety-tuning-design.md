# Base-Pose Orientation Telemetry and Safety Tuning Design

## Goal

Make horizontal visibility recovery engage earlier, reduce coarse target-heading
slew, and record actual robot yaw beside every applied raw visual-servo frame so
future runs can distinguish commanded-facing lead from physical body motion.

## Control behavior

- Expose `raw_horizontal_guard_fraction` with default `0.25`. Enter
  horizontal `RECENTER` when the target bbox center is strictly below that
  fraction or strictly above its symmetric complement, producing the default
  25%--75% interval.
- Expose `raw_horizontal_recovery_fraction` with default `0.30`. Consider
  horizontal recovery valid when the center is inside the inclusive interval
  from that fraction through its symmetric complement, producing the default
  30%--70% band for three consecutive applied visual frames.
- Validate `0 < raw_horizontal_guard_fraction <
  raw_horizontal_recovery_fraction < 0.5` before motion starts.
- Keep the existing center-based metric. This change does not add an outer-edge
  or hard-clipping predicate.
- Limit coarse `YAW_ALIGN` output to `|wz| <= 0.15 rad/s` while retaining the
  existing `0.05 rad/s` per-visual-frame slew step.
- Keep `YAW_TRIM` at `|wz| <= 0.10 rad/s` and preserve all other state-machine,
  stop, invalid-frame, vertical-recenter, and translation behavior.

## Orientation telemetry architecture

The SONIC relay is the authoritative telemetry producer because it owns the
integrated planner heading. When orientation telemetry is enabled, the relay
also consumes the existing `g1_debug` state stream continuously and reads its
`base_quat` value in scalar-first `[qw, qx, qy, qz]` order.

The relay publishes a conflated local ZMQ telemetry stream at 20 Hz. The
base-pose process subscribes to the stream, retains only the latest sample, and
attaches that sample to the next applied raw visual-servo diagnostic frame.
The telemetry channel is observational only: missing, malformed, or stale
telemetry never changes controller commands or termination behavior.

The default endpoint is `tcp://127.0.0.1:5565` for the subscriber and
`tcp://*:5565` for the relay publisher. Launch configuration owns the port so
the relay and base-pose commands remain consistent. The publisher is enabled
only for `base_pose` with `raw_yoloe_servo`; other planner and base-pose modes
retain their existing relay behavior without state-telemetry overhead.

## Angle definitions

The relay converts a finite, nonzero quaternion to normalized yaw using the
standard scalar-first quaternion formula. It records:

- `actual_yaw_rad`: wrapped absolute yaw from `base_quat` in `[-pi, pi]`.
- `actual_heading_rad`: wrapped yaw relative to the first valid base yaw sampled
  after relay startup. The origin must be captured before the first nonzero yaw
  command; otherwise this field and `heading_lag_rad` remain unavailable rather
  than inventing an alignment.
- `heading_setpoint_rad`: the relay's exact post-integration
  `PlannerState.heading` in `[-pi, pi]`.
- `heading_lag_rad`: wrapped
  `heading_setpoint_rad - actual_heading_rad`, so a positive value means the
  planner-facing target is ahead of the measured body heading.
- `state_age_s`: monotonic age of the last valid `g1_debug` sample at telemetry
  publication time.
- `telemetry_age_s`: monotonic age of the relay telemetry sample when the visual
  frame decision is recorded.

The base-pose diagnostic record writes these values under a top-level
`orientation` object beside `controller` and `command`. Before the first valid
sample, or when validation fails, `orientation` is `null`. Once a valid sample
exists, each numeric field is either a finite JSON number or `null` when its
specific value is unavailable.

## Relay state handling

Continuous state sampling is independent of upper-body freezing. Enabling
orientation telemetry must not latch or alter the frozen planner pose unless
`--freeze-current-upper-body` was explicitly supplied. A single state sample
may update orientation telemetry and satisfy a requested pose latch, but the
two behaviors remain separately gated.

The relay continues sending planner messages even if state telemetry is absent.
Telemetry publication continues with unavailable actual-yaw fields so loss of
the state stream is visible without entering the control path.

## Diagnostics data flow

1. `lavira_sonic_relay` polls the latest `g1_debug` state and validates
   `base_quat`.
2. On each 20 Hz planner tick it integrates `wz`, sends the SONIC planner
   message, and publishes the corresponding orientation telemetry sample.
3. The raw base-pose runtime reads the latest conflated sample when applying a
   visual observation or invalid-frame decision.
4. `FrameDiagnosticsWriter` serializes the normalized `orientation` object into
   that frame's `raw_servo_frames.jsonl` record.

Telemetry timestamps are monotonic and used only to calculate ages. Camera and
relay samples are not claimed to be hardware-synchronized; their measured age
is preserved for offline interpretation.

## Error handling

- Reject malformed JSON, wrong message type/version, non-finite numbers,
  malformed quaternions, and zero-norm quaternions from diagnostic state.
- Log a rate-limited warning for malformed relay telemetry received by the
  base-pose process, then retain no value from that malformed message.
- Do not terminate or pause visual servo because orientation telemetry is
  absent, stale, or invalid.
- Never emit NaN or infinity into JSONL diagnostics.

## Testing

- Boundary tests prove the default horizontal entry immediately outside
  25%/75% and recovery inclusively at 30%/70%.
- Configuration tests prove non-default symmetric intervals propagate from
  launch configuration through `BasePosePlannerConfig` into the controller,
  and invalid/non-hysteretic fractions are rejected.
- A controller test proves coarse yaw saturates at `0.15 rad/s`; existing trim
  tests continue proving the `0.10 rad/s` limit.
- Quaternion tests cover identity, positive/negative yaw, normalization,
  malformed shape, non-finite values, and zero norm.
- Relay tests prove heading is reported after integration, lag uses wrapped
  subtraction, state loss leaves control output unchanged, and telemetry-only
  operation does not freeze upper-body targets.
- Diagnostic-writer and runtime tests prove the orientation object is attached
  to applied frames and `null` when unavailable.
- Launch-command tests prove both processes receive matching telemetry
  endpoints only for the base-pose path.

## Out of scope

- Enforcing a heading-lag safety bound or resetting facing on `RECENTER`.
- Changing lateral speed, minimum translation speed, vertical protection,
  tracking thresholds, or hard image-edge guards.
- Hardware timestamp synchronization or long-term IMU yaw-drift correction.
