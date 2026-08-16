# Dual-Camera BasePose YOLOE Failover Design

## Goal

Add a `dual_raw_yoloe_servo` BasePose mode that consumes aligned RGB-D from
the head (`ego_view`) and chest (`chest_view`) cameras, prepares both cameras
at the start of one navigation request, and safely retries YOLOE across the
two views without leaving Planner mode. The existing single-head
`raw_yoloe_servo` mode remains unchanged.

## Scope

This change covers the BasePose launcher, camera receiver, raw YOLOE worker,
runtime state machine, diagnostics, documentation, and focused tests. It
reuses the dual RGB-D camera packet and saved intrinsics already present in the
repository.

It does not run both YOLOE trackers continuously, change the camera server
protocol, change the existing single-camera mode, or alter the latched upper
body and hand commands.

## Camera Contract and Calibration

One composed-camera packet is decoded once and yields independent per-stream
results for `ego_view` and `chest_view`. A malformed or missing stream is
recorded as that camera's failure and does not prevent the other stream from
being used.

Each live observation always uses the intrinsics and extrinsics of its active
camera, regardless of which camera supplied the YOLOE reference image:

- `ego_view` loads its entry from
  `gear_sonic/config/camera_intrinsics.json` and retains the existing head
  extrinsic defaults;
- `chest_view` loads its entry from the same file and uses a default pitch of
  `-3 degrees`; and
- the remaining chest extrinsic values are explicit dual-mode launch settings,
  so they can be calibrated without affecting the existing head-only mode.

Both streams require uint16 depth aligned to their own RGB image. Existing
shape, scale, timestamp, and calibration validation remains mandatory.

## Initial Grounding and Reference Bank

Pressing `n` starts one navigation generation and captures the two RGB-D views
from the same composed packet. Four grounding calls run concurrently through
the selected Codex or Qwen backend:

1. head target;
2. head table;
3. chest target; and
4. chest table.

Results are validated independently. A camera has an eligible initial
reference only when its target and table results both validate. Its immutable
`InitialReference` stores the source stream, RGB image, target prompt, target
box, every valid table box, camera timestamp, and grounding provenance.

If both cameras are eligible, `ego_view` is the initial active camera. If only
one is eligible, that camera is selected. If neither is eligible, the request
holds zero velocity and terminates without starting YOLOE motion.

Cross-camera visual prompting is permitted. A reference image and its boxes
always remain a coherent unit from one source image, while tracking and depth
geometry use the current live camera.

## Latest Reference Updates

Each camera has an independently stored `LatestReference`. The dual mode
attempts an update only at the existing rolling-reference cadence,
`raw_reference_update_interval_frames`, whose default is five frames.

An update candidate must contain, on the same RGB frame:

- a valid target track and target box;
- at least one valid table track and table box;
- valid target depth geometry;
- valid table-edge depth geometry; and
- the existing target confidence and image-boundary safety conditions.

The RGB image, target box, and table boxes are atomically replaced as one
reference snapshot. If any requirement fails, the previous reference remains
unchanged and diagnostics record the rejection. This extends the existing
five-frame target-reference mechanism; it does not increase the update rate.

## Runtime Architecture

The dual mode uses one camera connection, one YOLOE model/tracker, and one
controller. A small failover coordinator owns only camera/reference selection
and attempt state. It does not calculate motion.

Starting or switching an attempt performs these steps in order:

1. publish and continuously hold zero BasePose velocity;
2. retain the same navigation generation and Planner mode;
3. select a reference according to the failover stage;
4. clear YOLOE tracking state and install the selected visual prompts;
5. reset the visual-servo phase, filters, stable-frame counters, and
   per-attempt invalid-frame counter to `YAW_ALIGN`;
6. match up to 30 consecutive frames from the selected live camera; and
7. allow nonzero commands only after the first complete target, table, and
   depth observation succeeds.

Old-camera frames, events from a previous attempt ID, and frames captured
before the switch boundary cannot resume motion.

## Failover Cycle

Let `A` be the camera that was active when a perception attempt failed and `B`
be the other camera. One failover cycle is:

1. **Alternate-initial attempt:** use live input from `B`. Use `B`'s own
   `InitialReference` when it exists; otherwise use `A`'s
   `InitialReference` across cameras.
2. **Origin-latest retry:** if the first attempt cannot produce a complete
   observation, return to live input from `A`. Use `A`'s
   `LatestReference`, falling back to `A`'s `InitialReference` when no rolling
   update has yet been saved.
3. **Terminate:** if the origin-latest retry also fails, end this navigation
   generation at zero velocity.

The initial active camera's own YOLOE acquisition uses the same 30-frame
matching allowance. A successful first complete observation at any stage
makes that camera active, clears the failover cycle, and starts a fresh cycle
from that camera if a later failure occurs. This permits repeated switching
over a long navigation while still bounding each consecutive failure episode.

## Matching and Failure Semantics

Every YOLOE matching phase receives 30 consecutive invalid frames before it is
declared failed. This includes initial active-camera acquisition,
alternate-camera acquisition, origin-camera retry, and normal active tracking.
A complete target, table, target-depth, and table-depth observation resets the
matching counter.

Target, table, or geometric-depth loss is a soft invalid observation: command
zero immediately, tolerate frames 1 through 29, and advance the failover state
on frame 30. Unlike the single-camera mode's translation phase, the dual mode
requires valid table geometry in every control phase so that every successful
frame is suitable for coherent fallback reference capture.

Failures that make another frame attempt impossible advance the failover state
immediately. These include camera timeout, packet decode or protocol failure,
invalid calibration, YOLOE execution failure, and unrecoverable class/track
mismatch.

Normal alignment completion, the navigation-wide maximum runtime, `Space`
cancel, and `x` exit terminate directly and never trigger failover. Controller
motion state resets on every camera switch, but the navigation-wide runtime
begins at the original `n` press and is not extended by switching.

## Planner Safety

The relay and planner generation remain active throughout a switch. The
runtime publishes a zero-velocity `hold` immediately on failure and continues
zero heartbeats at the configured planner frequency while selecting a camera,
resetting YOLOE, and waiting for the first complete observation.

Switching never sends the Planner IDLE command, stops the C++ Planner loop, or
changes the latched 17-DoF upper-body and hand state. No nonzero command can be
published until an event from the current attempt ID passes freshness and
completeness checks.

## Configuration and Compatibility

`dual_raw_yoloe_servo` is added to the accepted BasePose modes and gets
dual-specific stream and extrinsic configuration. Its defaults are:

- head stream: `ego_view`;
- chest stream: `chest_view`;
- chest pitch: `-3 degrees`;
- reference update interval: five frames; and
- matching tolerance: 30 consecutive frames.

It reuses existing YOLOE model, confidence, image size, velocity bounds,
standoff, controller tolerances, camera endpoint, saved-intrinsics path, and
Planner relay settings. The original `raw_yoloe_servo` command construction and
runtime path stay behaviorally unchanged.

## Diagnostics

One navigation directory contains both cameras' initial RGB-D captures,
grounding prompts/results, initial reference eligibility, and failure reasons.
Every attempt records:

- active camera stream;
- attempt ID and failover stage;
- reference source stream and `initial` or `latest` kind;
- matching invalid-frame count;
- switch reason and zero-hold interval; and
- whether the first complete observation reset the cycle.

Per-frame diagnostic records add `camera_stream`, `attempt_id`,
`failover_stage`, `reference_source_stream`, and `reference_kind`. Every
five-frame reference candidate records whether the atomic reference update was
accepted or why it was rejected. Existing RGB/mask review sampling remains
bounded at its current cadence.

## Testing

Tests will cover:

- decoding both RGB-D streams from one packet while isolating a malformed
  stream;
- four concurrent grounding calls and head priority when both results are
  eligible;
- initial-reference selection with and without an eligible reference for the
  alternate camera;
- cross-camera reference use without mixing the active camera's calibration;
- atomic latest-reference updates only at the five-frame cadence;
- rejection of latest-reference candidates missing target, table, or either
  depth geometry;
- zero velocity for the first 29 invalid frames and a stage transition on the
  30th frame for every matching stage;
- the complete `A -> B initial -> A latest -> terminate` failure sequence;
- success at each stage clearing the cycle and allowing later repeated
  switching;
- stable generation and continuous Planner-mode zero hold across switches;
- stale and previous-attempt events being unable to resume motion;
- saved per-stream intrinsics and the chest `-3 degree` pitch;
- launch command wiring for `dual_raw_yoloe_servo`; and
- unchanged behavior of the existing `raw_yoloe_servo` tests.

Verification consists of focused unit tests, the full BasePose visual-servo
and launcher test suites, Python compilation, `git diff --check`, and a live or
recorded dual RGB-D smoke test when camera hardware is available.
