# Visibility-Guarded Yaw Servo Design

## Goal

Prevent the task target from leaving the head-camera field of view while the
base aligns to a table edge, without approaching so early that the table edge
can no longer be observed. Record enough information for every camera frame to
reconstruct target tracking, table geometry, controller phase transitions, and
issued commands after a run.

## Evidence and constraints

The latest run initialized both YOLOE visual classes correctly and produced 34
valid observations at 10.01 Hz. The table yaw error moved from +20.65 degrees
to -6.38 degrees, but the basket right error grew from 0.137 m to 0.349 m. The
controller integrated an approximately +28 degree heading change while
translation was gated. Because the basket started about 9.6 degrees to the
camera's right, it moved beyond the right image boundary and tracking stopped
after four missing frames.

The design must preserve these existing constraints:

- Use YOLOE-26M visual prompts and persistent track IDs.
- Use the aligned raw RealSense depth and the configured -25 degree camera
  pitch.
- Keep `|vx| <= 0.20 m/s`, `|vy| <= 0.16 m/s`, and coarse
  `|wz| <= 0.25 rad/s`.
- Keep the 20 second, 0.75 m cumulative translation, and 90 degree cumulative
  yaw safety limits.
- Keep the operator `Space` stop behavior and Planner-mode zero-speed hold.
- Do not require the table to remain visible after final yaw alignment, because
  the near-distance view can exclude its usable edge.

## Selected approach

Use a deterministic phase state machine rather than blended simultaneous
control or model-predictive control. The explicit state machine makes every
motion decision auditable, requires no unknown camera-to-base translation or
robot dynamics model, and prevents the yaw loop from continuing when the
target approaches an image boundary.

The phases are `YAW_ALIGN`, `RECENTER`, `YAW_TRIM`, `TRANSLATE_TARGET`, and
`DONE`. A `resume_phase` field remembers whether `RECENTER` interrupted coarse
or fine yaw.

## Control phases

### YAW_ALIGN

- This is the initial phase.
- Require both the target and table track, valid target depth, and valid table
  line geometry.
- Command `vx = 0` and `vy = 0`.
- Apply the existing proportional yaw controller with `|wz| <= 0.25 rad/s` and
  the existing 0.05 rad/s-per-visual-frame slew limit.
- Enter `YAW_TRIM` when the filtered absolute table yaw error is at most 8
  degrees.
- Before issuing yaw, evaluate the target visibility guard. If the target box
  has `x1 < 0.08 * width` or `x2 > 0.92 * width`, set `wz = 0` immediately,
  remember `resume_phase = YAW_ALIGN`, and enter `RECENTER`.

### RECENTER

- Require the target track and valid target depth. The table observation is
  recorded when present but is not required while lateral recovery is active.
- Command `vx = 0` and `wz = 0`.
- Apply the existing lateral controller, `vy = clip(-0.6 * right_error,
  -0.16, 0.16)`, with the existing 0.02 m/s-per-visual-frame slew limit.
- Consider the target recovered when its box center is between `0.43 * width`
  and `0.57 * width` for three consecutive valid frames.
- Resume the remembered yaw phase after recovery when a valid table observation
  is available. If the target is centered but the table is unavailable, command
  zero velocity, start the table-loss counter, and terminate on its fourth
  consecutive missing frame. Table absence does not affect the counter while
  lateral recovery is still in progress.
- The visibility guard may interrupt yaw any number of times during a run.

### YAW_TRIM

- Require the same target and table observations as `YAW_ALIGN`.
- Command `vx = 0` and `vy = 0`.
- Apply proportional yaw control with the reduced limit `|wz| <= 0.10 rad/s`.
- Retain the same target visibility guard and use
  `resume_phase = YAW_TRIM` if recentering is needed.
- Lock yaw and enter `TRANSLATE_TARGET` after the filtered absolute yaw error is
  at most 4 degrees for three consecutive valid frames.

### TRANSLATE_TARGET

- Require only the target track and valid target-mask depth. A missing table is
  valid in this phase and must not increment a tracking-loss counter.
- Command `wz = 0`; do not resume table yaw control after this phase begins.
- Use the existing target distance and lateral loops with
  `|vx| <= 0.20 m/s`, `|vy| <= 0.16 m/s`, and existing slew limits.
- The default target distance remains 0.60 m.
- Enter `DONE` after `|forward_error| <= 0.04 m` and
  `|right_error| <= 0.03 m` for five consecutive valid frames.

### DONE

- Publish three zero-speed messages and return to the idle Planner hold state.
- A new operator request starts again in `YAW_ALIGN` with all counters reset.

## Loss and safety behavior

The current behavior holds the previous short-TTL motion for three invalid
frames. That is unsafe for a visibility guard because continued yaw can push an
already marginal target out of frame. Under the new design, any required target
or table observation becoming invalid immediately changes the command to zero.
The system waits stationary for up to three consecutive frames for
reacquisition and terminates on the fourth invalid frame.

Required observations are phase-dependent:

- `YAW_ALIGN` and `YAW_TRIM`: target, target depth, table, and table geometry.
- `RECENTER`: target and target depth; table is needed only before yaw resumes.
- `TRANSLATE_TARGET`: target and target depth only.

Track-ID class mismatches remain immediate hard failures. Camera staleness,
operator cancellation, elapsed-time limits, and cumulative-motion limits remain
immediate terminal stops.

## Per-frame telemetry

Create `frame_telemetry.jsonl` in each raw-servo request directory. Append and
flush exactly one JSON object for every camera frame processed after visual
prompt initialization, including invalid frames. Each record contains:

- `frame_index`, camera timestamp, wall-clock timestamp, controller phase,
  resume phase, and an optional phase-transition reason.
- Target presence, expected/observed track ID, class ID/name, confidence,
  pixel `xyxy` box, mask pixel count, and distance to each image boundary.
- Target forward/right position, 3-D body point, valid depth pixels, valid depth
  ratio, and median raw depth in meters, or a structured target error.
- Table presence, expected/observed track ID, confidence, pixel box, mask pixel
  count, yaw error, fitted-line length, inlier count, residual, and 3-D line
  center, or a structured table error.
- Raw and filtered forward/right/yaw errors, stability counters, invalid-frame
  counters, visibility-guard state, and cumulative translation/yaw integrals.
- Final `vx`, `vy`, `wz`, command TTL, and any stop reason.

Values that do not exist in a phase are represented by JSON `null`; fields are
not omitted. The writer flushes after every record so an emergency stop or
process failure preserves the last processed frame.

## Per-frame visualization

Save one JPEG per processed frame under `frames/frame_<six-digit-index>.jpg`.
The image contains:

- Target and table boxes, class names, confidence, and track IDs.
- Semi-transparent target and table masks with distinct colors.
- Image-center line, 8/92-percent guard lines, and 43/57-percent recovery lines.
- Controller phase, forward/right/yaw errors, valid-depth statistics, and the
  issued `vx`, `vy`, `wz` command.
- Invalid-observation and phase-transition messages when applicable.

The 640x480 JPEG write occurs after command computation. If telemetry or frame
writing fails, the worker emits a hard diagnostic failure and sends zero speed;
it does not continue an unaudited motion. A maximum 20-second run produces at
most about 200 images at the nominal 10 Hz rate.

## Component boundaries

`VisualServoController` owns phase transitions, counters, thresholds, command
generation, and safety stops. `RawServoObservation` carries the target box in
addition to target and table geometry so the controller can evaluate visibility
without depending on the tracker implementation.

A focused diagnostics module owns telemetry serialization and annotated-frame
rendering. The worker remains responsible for capturing frames, invoking the
tracker and geometry estimators, passing complete observations to the
controller, and writing one telemetry record for every success or failure.

## Verification

Software tests must prove:

- Coarse yaw commands no translation while the target is safely visible.
- Crossing either guard line immediately emits zero yaw and enters `RECENTER`.
- `RECENTER` commands only lateral motion and resumes the correct yaw phase
  after three centered frames.
- Fine yaw is capped at 0.10 rad/s and transitions after three stable frames.
- `TRANSLATE_TARGET` continues when the table is missing and completes after
  five stable target frames.
- Any required missing observation immediately emits zero velocity and the
  fourth consecutive miss terminates the run.
- Every processed valid or invalid frame produces one complete JSONL record and
  one annotated JPEG with monotonically increasing indices.
- Existing command bounds, cumulative limits, camera watchdog, and operator
  stops remain unchanged.

Run the focused visual-servo tests, the full `gear_sonic/tests` suite, syntax
checks, and `git diff --check`. A local no-cloud replay using saved camera data
must verify artifact creation and phase transitions before live-robot testing.

## Non-goals

- No learned or model-predictive motion controller.
- No change to Codex/Qwen grounding, YOLOE visual-prompt initialization, or
  SONIC Planner message semantics.
- No use of camera center-pixel depth.
- No requirement to recover a different physical instance after a track-ID
  switch.
