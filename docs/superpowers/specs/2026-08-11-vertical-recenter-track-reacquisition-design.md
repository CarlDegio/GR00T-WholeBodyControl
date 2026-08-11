# Vertical Recenter and Target Track Reacquisition Design

## Goal

Prevent the task target from leaving through the top of the head-camera image,
give that recovery priority over the existing horizontal recenter mechanism,
and continue safely when BoT-SORT assigns the same YOLOE target class a new
track ID.

## Evidence and thresholds

The eight most recent raw-YOLOE runs with frame telemetry have initial target
box vertical centers at 260.3, 220.9, 204.5, 63.4, 84.6, 97.6, 111.6, and
105.5 pixels. Their median is 108.5 pixels, or 22.6 percent of a 480-pixel
image. Runs that lost the target through the top ended near 47, 26, and 23
pixels. Existing `TRANSLATE_TARGET` telemetry also shows that positive `vx`
moves a top-clipped target downward in the image.

Use normalized image-height thresholds so the behavior is resolution
independent:

- Enter vertical recovery when the target box center is below 18 percent of
  the image height.
- Consider the target vertically recovered after its center is at or below the
  normal viewing region, expressed in image coordinates as at least 23 percent
  of the image height, for three consecutive visual frames.
- During vertical recovery publish only `vx=+0.30 m/s`; hold `vy=wz=0`.

The 18/23 percent hysteresis prevents threshold chatter while restoring the
target close to the observed 22.6-percent initial median. There is no upper
recovery-band bound: once a target crosses 23 percent, forward motion stops
rather than continuing after an overshoot.

## Controller state machine

Add `ServoPhase.VERTICAL_RECENTER`. The vertical guard is evaluated before all
phase-specific behavior in every nonterminal phase, including the existing
horizontal `RECENTER` and `TRANSLATE_TARGET` phases.

Entering vertical recovery records the interrupted phase separately from the
existing horizontal `resume_phase`. The controller immediately returns a zero
command on the transition frame. On following frames it commands only forward
motion while the center remains above the 23-percent recovery line. As soon as
one frame reaches that line it holds zero while counting three consecutive
recovered frames, preventing forward overshoot during stability confirmation.

When vertical recovery completes:

1. If horizontal `RECENTER` was interrupted, resume it.
2. Otherwise, if the target still crosses the left/right visibility guard,
   enter horizontal `RECENTER` and arrange to resume the originally
   interrupted phase afterward.
3. Otherwise resume the interrupted yaw or target-translation phase directly.

This ordering makes vertical recovery dominant when both guards are active.
The table track is optional during vertical recovery, as it already is during
target translation and horizontal recovery.

`RawServoObservation` gains `image_height` alongside `image_width`. Per-frame
diagnostics expose the new phase, vertical resume phase, and vertical stable
frame count.

## Target track-ID reacquisition

BoT-SORT is configured without appearance ReID. Its 30-frame track buffer keeps
lost tracks available for matching but does not force a later detection to use
the old ID. Large clipping and box-shape changes can fail IoU/Kalman
association, causing a new track to be created after only a few missing frames.

Target resolution therefore becomes class-first:

1. Collect every current instance with target class index 0.
2. Prefer the current expected target ID when it is among those instances.
3. Otherwise select the highest-confidence class-0 instance unconditionally,
   update the worker's expected target ID, and continue the observation.
4. If no class-0 instance exists, preserve the existing immediate-zero soft
   miss behavior and terminate only after the fourth consecutive miss.
5. If the expected ID is present under a wrong class and no class-0 instance
   exists, retain the hard class-mismatch stop because no valid target exists.

Every successful takeover records `target_reacquired`, `previous_target_id`,
and `target_track_id` in the worker event details. Runtime control-update JSONL
records preserve those details, and frame telemetry naturally records the new
target track ID. Table-ID handling remains unchanged.

## Safety and failure behavior

- The first vertical or horizontal guard transition publishes zero before
  changing movement axes.
- Any missing target frame publishes zero immediately.
- Camera watchdog, operator stop, maximum run time, and diagnostic write
  failures retain their existing behavior.
- The change does not enable ReID, change YOLOE confidence thresholds, or alter
  target/table grounding prompts.
- Multiple class-0 detections are intentionally resolved by highest confidence,
  matching the operator-approved unconditional same-class takeover policy.

## Test strategy

Controller tests will prove vertical triggering, transition-frame zero,
forward-only recovery, three-frame hysteresis, vertical priority over a
simultaneous horizontal guard, and correct return to horizontal recenter or the
interrupted phase.

Worker tests will prove that the expected ID is preferred, a new class-0 ID is
adopted, the highest-confidence class-0 candidate wins, event details identify
the takeover, and a true absence remains a soft miss. Runtime tests will prove
that takeover details are persisted to `raw_servo_events.jsonl`.

The focused visual-servo tests and the complete `gear_sonic/tests` suite must
pass before completion.
