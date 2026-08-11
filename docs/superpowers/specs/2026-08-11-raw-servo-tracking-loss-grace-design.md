# Raw Servo Tracking-Loss Grace Design

## Goal

Keep an active `raw_yoloe_servo` base-pose run recoverable through short target
or required table-edge tracking gaps. The controller must hold zero velocity
while perception is unavailable and terminate only on the 30th consecutive
missing frame.

## Current behavior

`VisualServoController.note_invalid()` immediately replaces the active command
with zero velocity. Soft perception failures share one consecutive-invalid-frame
counter, and any usable observation resets that counter. The current default
tolerance permits three missing frames and terminates on the fourth.

The visual worker reports a missing tracked target as a soft invalid event. A
missing table produces an observation without table geometry; the controller
treats that as soft invalid only in phases where `table_required` is true.

## Required behavior

- Frames 1 through 29 of one uninterrupted soft tracking-loss sequence publish
  zero velocity and keep the current run active.
- Frame 30 terminates the run through the existing tracking-lost path and sends
  the existing final stop sequence.
- A usable observation before frame 30 resets the consecutive-loss count and
  allows the phase controller to resume producing motion commands.
- A newly started run begins with a zero loss count.
- Missing target data uses this policy in every active servo phase.
- Missing table geometry uses the same policy only while the current phase
  requires the table edge, such as yaw alignment and yaw trim. Translation
  phases continue to accept target-only observations.
- Existing hard failures, including track ID/class mismatch and worker/camera
  failures classified as hard, remain immediately terminal.
- The threshold is exactly 30 visual tracking frames. It is not converted to a
  wall-clock timeout. At the current default `raw_servo_hz=10.0`, this is
  approximately three seconds.

## Implementation

Keep the existing shared soft-invalid counter because it represents whether all
perception inputs required by the current controller phase are usable. Change
the controller's default soft-loss threshold to 30 frames and make the terminal
comparison inclusive so that the 30th consecutive invalid frame, rather than
the 31st, terminates the run.

Successful controller updates continue to clear the counter at the existing
phase-specific points. No worker-side retry loop or separate wall-clock timer is
needed: the worker continues emitting one invalid event per tracking frame, and
the runtime already publishes zero immediately when an invalid frame replaces a
non-zero command.

## Tests

Controller tests will prove that:

- target loss holds zero without termination for frames 1 through 29;
- target loss terminates with the existing reason on frame 30;
- a valid target observation after a short loss resets the counter and resumes
  a non-zero command, after which a new loss sequence starts at one;
- missing required table geometry follows the same 29-frame hold and 30th-frame
  termination boundary;
- missing table geometry remains accepted during target-only translation; and
- hard invalid events remain immediately terminal.

Runtime coverage will retain the existing assertion that the first invalid
frame is published as zero immediately.

## Scope

This change does not alter YOLOE association, reacquisition rules, hard-failure
classification, servo frequency, command TTL, maximum run time, or final stop
message format/count.
