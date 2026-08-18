# G1 head-vision base-pose adjustment

`base_pose` is a SONIC planner command source independent of AgentNav. The model
authors the complete ordered sequence of rotations and forward/backward
translations; the runtime validates the sequence and converts each value to a
fixed-speed SONIC segment without changing the requested angle or distance.

## Camera server

Run the head RealSense as `ego_view` and align depth to its color stream:

```bash
./start_base_pose_camera_server.zsh
```

That script mirrors the existing multi-camera device configuration but selects
`ego_view` as the one RealSense mount that publishes aligned depth. The existing
`start_camera_server.zsh` remains chest-depth compatible for AgentNav.

The `rgb` experiment consumes only `ego_view`. The `raw_yoloe_servo`
experiment consumes `ego_view` plus its RealSense-aligned raw uint16 depth
directly from port `5555`; it does not start LingBot. The `rgbd` and
`rgb_depth_query` experiments require the deployment machine's
`.venv_lingbot_depth` environment and local LingBot-Depth model. They never
fall back to raw RealSense depth or RGB-only planning when enhanced depth is
unavailable.

The `dual_raw_yoloe_servo` experiment requires both `ego_view` and
`chest_view`, each with its aligned `<stream>_depth` image in the same composed
camera packet. Start the repository's dual RGB-D configuration with:

```bash
./start_camera_calibration_server.zsh
```

Despite its calibration-oriented name, this command continuously publishes
the dual RGB-D set needed by the servo. The runtime loads both saved intrinsic
records from `gear_sonic/config/camera_intrinsics.json`; it never substitutes
one camera's intrinsics for the other.

To start either camera configuration and save the first valid `ego_view` RGB
frame received after startup, use the wrapper below. It connects the ZMQ
subscriber before starting the camera process and saves a lossless PNG under
`outputs/camera_startup/` by default:

```bash
./start_camera_and_save_first_ego_frame.zsh base_pose
./start_camera_and_save_first_ego_frame.zsh original
```

An explicit output path can be supplied as the second argument:

```bash
./start_camera_and_save_first_ego_frame.zsh base_pose /tmp/ego_first.png
```

When the camera server runs on the robot and this repository runs on the
deployment machine, start the subscriber on the deployment machine first:

```bash
./start_camera_and_save_first_ego_frame.zsh remote 192.168.123.164
```

Then start `start_base_pose_camera_server.zsh` or `start_camera_server.zsh` on
the robot. The remote mode never launches a local camera process; it receives
from the robot's TCP port `5555`, saves the first received frame on the
deployment machine, and exits. If the robot camera is already publishing, it
saves the first frame received after the deployment-side subscriber connects.
The output path and a non-default port can be supplied as the third and fourth
arguments, respectively.

## Launch examples

Raw-depth YOLOE-26M closed-loop alignment (recommended for the new servo):

```bash
source .venv_inference/bin/activate
python gear_sonic/scripts/launch_inference.py \
  --planner-input base_pose \
  --base-pose-mode raw_yoloe_servo \
  --base-pose-task "align to the blue basket" \
  --base-pose-raw-head-target-distance-m 0.90 \
  --base-pose-raw-chest-target-distance-m 0.80 \
  --base-pose-raw-forward-tolerance-m 0.10 \
  --base-pose-raw-lateral-tolerance-m 0.10 \
  --base-pose-raw-yoloe-model-path \
    tools/yoloe26m/weights/yoloe-26m-seg.pt
```

Dual head/chest raw-depth YOLOE alignment:

```bash
source .venv_inference/bin/activate
python gear_sonic/scripts/launch_inference.py \
  --planner-input base_pose \
  --base-pose-mode dual_raw_yoloe_servo \
  --base-pose-task "align to the blue basket" \
  --base-pose-dual-head-camera-stream ego_view \
  --base-pose-dual-chest-camera-stream chest_view \
  --base-pose-dual-chest-camera-pitch-deg -3 \
  --base-pose-raw-head-target-distance-m 0.90 \
  --base-pose-raw-chest-target-distance-m 0.80 \
  --base-pose-dual-match-tolerance-frames 30 \
  --base-pose-dual-initialization-grace-s 30 \
  --base-pose-raw-max-run-s 180 \
  --base-pose-raw-yoloe-model-path \
    tools/yoloe26m/weights/yoloe-26m-seg.pt
```

The raw modes expect aligned RGB-D at the dimensions stored in
`gear_sonic/config/camera_intrinsics.json`. Head extrinsics default to height
`1.2 m`, pitch `-38°`; dual-mode chest extrinsics default to height `1.0 m`,
pitch `-3°`. Both default to zero roll/yaw and zero forward/lateral offset.
Override the corresponding `--base-pose-camera-*` or
`--base-pose-dual-chest-camera-*` arguments when measured extrinsics are
available.

RGB-only:

```bash
.venv_inference/bin/python gear_sonic/scripts/launch_inference.py \
  --planner-input base_pose \
  --base-pose-mode rgb \
  --base-pose-task "put the cup into the tray"
```

LingBot RGB-D or two-stage ROI depth lookup:

```bash
.venv_inference/bin/python gear_sonic/scripts/launch_inference.py \
  --planner-input base_pose \
  --base-pose-mode rgbd \
  --base-pose-task "put the cup into the tray"

.venv_inference/bin/python gear_sonic/scripts/launch_inference.py \
  --planner-input base_pose \
  --base-pose-mode rgb_depth_query \
  --base-pose-task "put the cup into the tray"
```

If `--base-pose-task` is empty, the launcher uses `--prompt`. Defaults are
`gpt-5.6-sol`, xhigh reasoning effort with Fast enabled, a `600 s` Codex
timeout, camera height
`1.2 m`, pitch `-25°`, and vertical FOV `43.077882°`. Camera intrinsics and
horizontal FOV come from the live frame.

## Raw YOLOE servo behavior

After `n`, two independent Codex or Qwen grounding requests run concurrently
against the same initial RGB image. The task request uses the current raw-servo
prompt to identify only the primary target. The desk request uses exactly the
single-target prompt from `tools/yoloe26m/auto_refer_detect.py` with its target
set to `desk`, and it preserves every visible desk box. Both results must
validate before YOLOE or robot motion starts.

That same RGB frame becomes YOLOE's `refer_image`; the boxes are converted to
original-image pixels and passed in `visual_prompts`, with the primary target
assigned class `0` and every desk box assigned class `1`. The resulting visual
embeddings, rather than text-only class embeddings, are installed in YOLOE-26M
before persistent BoT-SORT tracking begins at `10 Hz`. The primary box controls
target centering and the eroded target mask provides robust raw-depth range.
For the table, the per-column upper envelope is expressed as height above the
image bottom. Points below 85% of its 90th-percentile height are rejected and
interior gaps are linearly rebuilt. Retained runs no wider than 2% of the image
inside either outer 10% are removed without interpolation. All rebuilt envelope
pixels, including interpolated pixels, form the image-space RANSAC candidates.
The selected pixel line gates real contour depths in the 0.15--4.0 m range;
those points are deprojected and refitted in body XY. The final physical line
must be at least 0.35 m long and controls yaw perpendicular to the table edge.
Candidate ranking does not use distance from the robot.

The bounded controller publishes at `20 Hz` through the existing direct 5558
relay. Its visual state machine is:

1. `YAW_ALIGN`: rotate from the table edge estimate, with `vx=vy=0` and
   `|wz|<=0.30 rad/s`.
2. `RECENTER`: if the target-box horizontal center crosses the left/right
   25%/75% visibility guard, stop yaw immediately and use only `vy` until its
   center stays inside the 30--70% recovery band for three visual frames. Then
   resume the interrupted yaw phase.
3. `YAW_TRIM`: enter within the configured yaw tolerance, limit `|wz|` to
   `0.20 rad/s`, and lock yaw after the filtered error remains within that
   tolerance for three frames.
4. `TRANSLATE_TARGET`: hold `wz=0` and issue only one translation axis at a
   time. If the filtered lateral error is outside tolerance, use only `vy` to
   center the target; once lateral alignment is inside tolerance, use only
   `vx` to reach the active camera's standoff (`0.90 m` for head and `0.80 m`
   for chest by default). Finish after five stable frames with
   both errors inside their default `0.10 m` tolerances and the final yaw
   error inside the configured tolerance for all five consecutive frames.
   The table track is optional in this phase.

The target box's vertical image coordinate does not trigger a recovery state,
motion command, or reference-update pause.

Every nonzero closed-loop yaw command has magnitude at least
`base_pose_raw_min_yaw_speed_rad_s`, defaulting to `0.10 rad/s`. The launcher
forwards it as `--raw-min-yaw-speed-rad-s`. Commands reverse through zero so
neither direction emits a sub-minimum transient. Inside the accepted yaw band,
the controller holds zero while counting stable frames.

`base_pose_raw_yaw_tolerance_deg` is shared by `YAW_TRIM`,
`GLOBAL_YAW_ALIGN`, and `TRANSLATE_TARGET`, and defaults to `8.0` degrees. The
launcher forwards it to the planner as `--raw-yaw-tolerance-deg`.

The coarse and trim yaw limits are independently configurable as
`base_pose_raw_yaw_coarse_speed_rad_s` and
`base_pose_raw_yaw_trim_speed_rad_s`, defaulting to `0.30` and `0.20 rad/s`.
The launcher forwards them as `--raw-yaw-coarse-speed-rad-s` and
`--raw-yaw-trim-speed-rad-s`. Global yaw correction uses the trim limit.

Whenever translation is nonzero, the controller raises the active axis speed
to at least `base_pose_raw_min_linear_speed_m_s`, which defaults to
`0.40 m/s`; its raw YOLOE commands always satisfy `vx == 0` or `vy == 0`.
Zero translation and pure-yaw commands are unchanged. Before this
minimum-speed scaling, the proportional lateral request is limited by
`base_pose_raw_max_lateral_speed_m_s`, which also defaults to `0.40 m/s`.
The final relay restores that configured lateral bound for `vy`. The launcher
forwards `--raw-min-linear-speed-m-s` and
`--raw-max-lateral-speed-m-s` to the planner, and forwards
`--max-lateral-speed-m-s` to the relay.

During the chest-camera forward approach, crossing the horizontal visibility
guard no longer pauses longitudinal motion for a pure rotation. The controller
keeps the current distance-controlled `vx` and adds a fixed-magnitude `wz`
toward the basket, so the basket is brought back toward the image center while
the robot continues to approach. The yaw magnitude defaults to `0.30 rad/s`
and is configurable as
`base_pose_raw_forward_recenter_yaw_speed_rad_s`, forwarded to the planner as
`--raw-forward-recenter-yaw-speed-rad-s`. Only the sign changes with the
left/right image direction; head-camera recenter behavior is unchanged.

Any required invalid observation commands zero on that same frame. In
single-camera `raw_yoloe_servo`, the first 29 consecutive failures of the
required target/desk detection or depth geometry keep the current run active
at zero velocity. Desk is counted only while the current controller phase
requires it. On the 30th, YOLOE clears its tracking state and switches both
classes to pure text: `blue basket` for
class `0` and `desk` for class `1`. The controller remains in its current servo
phase and the missing-frame counter starts a second 30-frame window.

When both text classes are detected with valid target and desk geometry, their
boxes on that exact RGB frame immediately rebuild the normal two-class visual
prompt. Tracking and the every-fifth-frame target-reference updater then resume
from that visual reference. If the two-text mode instead misses either class or
valid geometry for 30 consecutive frames, the current navigation terminates.
The text and reconstructed visual prompt artifacts include the source frame in
their filenames for audit.

If the expected target ID disappears but any class-0 target remains, the
worker immediately adopts the highest-confidence class-0 ID and records the
old/new IDs in diagnostics. A true class mismatch with no class-0 target,
`Space`, or `180 s` of YOLOE detection time stops immediately and requires a
new `n`. The runtime budget begins when the first YOLOE attempt starts
detecting, so Codex/Qwen grounding does not consume it. A continuous
tracking gap over `0.4 s` instead enters a recoverable soft-stale hold: the
runtime keeps the generation active, publishes zero velocity, and resumes on
the first fresh valid observation. An observation already more than `0.4 s`
old when received cannot resume motion. Continuous observations use a
single-slot latest mailbox, so a newer waiting observation replaces the older
waiting one without interrupting an observation already being processed.
Accumulated translation and yaw remain diagnostic values but no longer
terminate a run. This mode still bypasses MID-360/REASAN, so the operator must
ensure a clear table-side motion area.

SONIC Planner does not consume angular velocity directly. The relay integrates
the servo's `wz` at `20 Hz` and sends the resulting target `facing` vector, so
`wz` is a bounded target-heading slew rate; actual body yaw rate is determined
by SONIC Planner and the whole-body controller.

### Dual-camera failover

In `dual_raw_yoloe_servo`, the initial head and chest RGB frames are grounded
at the same time. Each camera runs independent target and table requests, so
one failed view does not discard the other. A view is initially eligible only
when both target and desk boxes validate. Head is selected when both views are
eligible; otherwise the eligible view starts the workflow.

Each camera grounding runs in its own process. Once either view becomes
eligible, the other process gets at most 30 more seconds by default. If it does
not return in that grace period, its complete process group is terminated and
the eligible view starts YOLOE immediately. Configure this with
`base_pose_dual_initialization_grace_s`.

For each active YOLOE attempt, a valid frame requires the target, desk, target
depth geometry, and desk-edge depth geometry together. A successful complete
frame resets the failover cycle and permits later switching in either
direction. Every fifth complete frame is considered for an atomic
`LatestReference` update containing a copy of that same RGB image, target box,
and the desk box whose geometry validated. The two cameras keep independent
latest-reference gates.

The accepted frame indices are `5, 10, 15, ...`, exactly matching the saved
`review_samples/raw` cadence. An independent YOLOE encoder validates the new
target and desk visual embeddings off the servo thread. The active tracker
installs both embeddings atomically in visual mode. In `origin_text` it updates
only the visual desk embedding; `alternate_text` keeps both classes text-only
for that entire attempt. BoT-SORT state and controller state are kept, and a
successful installation changes the attempt's diagnostic `reference_kind` to
`latest`.

Each YOLOE detection stage tolerates 30 consecutive invalid frames. On the
30th, the runtime immediately publishes zero velocity, remains in Planner
mode, resets YOLOE and the controller, and advances without changing the
navigation generation. Camera A is the camera used by the attempt that just
failed; camera B is the other camera. The bounded order is:

1. Stay on A for `origin_text`: class `0` uses the target text embedding and
   class `1` uses A's visual desk embedding.
2. If that fails, stay on A for `origin_qwen`. Capture one new A frame and run
   the exact same target and desk grounding prompts used at initial setup with
   `qwen3-vl-8b-instruct`. The returned target and desk boxes become a new
   two-class YOLOE visual prompt, followed by another 30-frame detection window.
3. If the Qwen call fails, or the resulting YOLOE attempt fails, switch to B
   for `alternate_text`. Both the target and `desk` use YOLOE text embeddings;
   no visual desk box is installed during this stage.
4. If that fails, stay on B for `alternate_qwen`. Capture one new B frame,
   re-run the same Qwen grounding round, install its boxes as the two visual
   prompts, and allow the final 30-frame detection window.
5. If the Qwen call or YOLOE detection fails again, terminate the current
   navigation.

A Qwen API/validation failure advances immediately rather than consuming a
30-frame YOLOE window. The fallback model is configurable through
`base_pose_dual_qwenvl_fallback_model` / `--dual-qwenvl-fallback-model`; its
default is `qwen3-vl-8b-instruct`. Credentials continue to come from
`DASHSCOPE_API_KEY` in the environment or `.venv_inference/.env` and are never
written to run artifacts.

The same four-stage recovery cycle starts from whichever camera most recently
produced a successful complete frame. Old or future-attempt events cannot
resume motion. The navigation-wide timeout starts when the first YOLOE
detection attempt begins, defaults to 180 seconds, and is not extended by
camera switches.

The symmetric horizontal intervals are launch parameters. Set
`base_pose_raw_horizontal_guard_fraction` for entry and
`base_pose_raw_horizontal_recovery_fraction` for recovery; their defaults are
`0.25` and `0.30`. They are forwarded to the raw planner as
`--raw-horizontal-guard-fraction` and
`--raw-horizontal-recovery-fraction`. Values must satisfy
`0 < guard < recovery < 0.5`; for example `0.22`/`0.30` means enter outside
22%--78% and recover inside 30%--70%.

The translation completion tolerances are independently configurable through
`base_pose_raw_forward_tolerance_m` and
`base_pose_raw_lateral_tolerance_m`; both default to `0.10 m`. The launcher
forwards them as `--raw-forward-tolerance-m` and
`--raw-lateral-tolerance-m`. Each value must be finite and strictly positive.
The controller uses the configured values for both the velocity deadbands and
the five-consecutive-frame completion test.

The longitudinal arrival distance is configured independently for each camera:

- launcher: `base_pose_raw_head_target_distance_m` (default `0.90 m`) and
  `base_pose_raw_chest_target_distance_m` (default `0.80 m`);
- direct `base_pose_planner.py`: `--raw-head-target-distance-m` and
  `--raw-chest-target-distance-m`.

In dual-camera mode the runtime selects the corresponding value whenever the
active YOLOE camera changes. Single-camera `raw_yoloe_servo` uses the head value
by default; if its configured stream is the chest stream, it uses the chest
value.

After those five frames and the final yaw check pass, the runtime immediately
publishes zero-velocity `stop` messages and enters `post_stop_sampling`.
For `3.0 s` by default it keeps the active camera and YOLOE tracker running
at the normal detector rate, updates heading and visual errors, and never
issues a nonzero command or starts camera failover. Only after this window
does it write `finished` with `reason=aligned`. Configure the duration with
`base_pose_raw_post_stop_sample_s` / `--raw-post-stop-sample-s`; set it to
`0` to retain immediate completion.

## Operator state machine

- In pane 1, press `k`. The relay requests a fresh `g1_debug` sample, latches
  the measured 17-DoF upper body and both 7-DoF hands, and starts C++ in Planner
  mode only after the latch is ready. A second `k` stops the C++ loop.
- In pane 3, press `n` only while IDLE to capture a new frame and request a new
  plan.
- Press `Space` in pane 3 to cancel inference or motion, discard every remaining
  step, and keep publishing Planner IDLE with the latched upper body and hands.
  Press `n` for a new observation before moving again.
- Press `x` in pane 3 to exit the base-pose source. The direct relay continues
  to publish IDLE until the C++ loop is stopped with `k`.

The base-pose path deliberately bypasses MID-360/REASAN. It is intended for
table-side adjustment and relies on the model's visible-scene safety judgment
plus strict command bounds, not environmental obstacle avoidance.

Diagnostics for each request are written below
`outputs/base_pose_adjustment/<timestamp>/` (raw runs use
`raw_yoloe_<timestamp>_g<generation>/`; dual runs use
`dual_raw_yoloe_<timestamp>_g<generation>/`). Each raw run saves the exact task
request in `target_prompt.txt`, `target_schema.json`, and `target_result.json`,
and the independent table request in `table_prompt.txt`, `table_schema.json`,
and `table_result.json`. The pixel-space two-class handoff, including all table
reference boxes and their class IDs, is saved in `yoloe_reference_prompt.json`
for auditing. A dedicated diagnostic writer thread writes one record per
worker-produced camera frame to `raw_servo_frames.jsonl`; control processing
does not encode these images or wait for their file writes and `fsync` calls.
New records set `annotated_image` to `null`; the live runtime does not create
`frames/` or compose bbox/mask overlays. Applied frames include their exact
post-update controller state and command. A waiting observation displaced by a
newer frame, cancellation, or shutdown remains in the log with
`control_applied=false` and null controller/command metadata. Every fifth frame
still saves an unannotated RGB PNG and separate target/table mask PNGs under
`review_samples/`.
Successful table-edge estimates also record `line_endpoints_xy_m` and
`line_endpoints_px` under `geometry.table`. For sampled frames, the diagnostic
thread writes a separate `*_table_edge.png` image with a red selected edge and
yellow endpoints over the white table mask. The original binary `*_table.png`
remains unchanged for replay and machine processing; overlay rendering and PNG
I/O never run on the servo control path.


In raw mode, each applied frame also receives a top-level `orientation` object
from the relay's `g1_debug.base_quat` sample and its exact integrated SONIC
heading setpoint:

- `actual_yaw_rad`: wrapped absolute measured base yaw.
- `actual_heading_rad`: measured yaw relative to the relay-start yaw origin.
- `heading_setpoint_rad`: the relay's post-integration target heading.
- `heading_lag_rad`: wrapped `heading_setpoint_rad - actual_heading_rad`.
- `state_age_s`: age of the underlying `g1_debug` state at relay publication.
- `telemetry_age_s`: age of the relay telemetry when attached to the frame.

`orientation` is `null` before the first valid telemetry sample. If the relay
does not see a valid yaw before the first nonzero heading target,
`actual_heading_rad` and `heading_lag_rad` remain null rather than inventing a
relative origin. Missing or malformed telemetry only affects these diagnostic
fields; it never pauses, terminates, or changes visual-servo commands. The
launcher uses `base_pose_orientation_telemetry_port=5565` by default and only
opens this side channel for `base_pose_mode=raw_yoloe_servo` or
`base_pose_mode=dual_raw_yoloe_servo`.

Applied frame records also expose `visual_yaw_error_rad`,
`desired_heading_rad`, `heading_setpoint_error_rad`, `yaw_error_source`, and
`yaw_error_trusted` inside `controller`. During `post_stop_sampling`,
`post_stop_elapsed_s`, total/valid/invalid sample counts, and the configured
duration are updated on every frame. `raw_servo_events.jsonl` writes an
explicit `post_stop_sampling_started` event and copies the final counts into
the later `finished` event, so stopped-pose drift can be measured without
inferring the window from timestamps.

Dual runs also write `initial_reference_summary.json`, per-camera initial RGB-D
and grounding outputs, and the attempt provenance fields `camera_stream`,
`attempt_id`, `failover_stage`, `reference_source_stream`, and
`reference_kind` into per-frame diagnostics.

Normal shutdown stops motion and joins the perception worker before draining
and joining the diagnostic writer, then reports `diagnostics flushed`.
Therefore a normal completed run retains every produced frame even if some
were not used for control. A forced process kill or power loss can still lose
queued in-memory diagnostics. If diagnostic encoding or writing fails, the
runtime emits one warning and disables diagnostics for that generation without
terminating robot control. Global cancellation/failure events that occur before
a request directory is known are appended to
`outputs/base_pose_adjustment/runtime_events.jsonl`.

For both raw YOLOE modes, every successfully written
`raw_servo_events.jsonl` record is also printed as the exact same JSON line in
the Base Pose planner bash pane. When launched through `launch_inference.py`, a
second tmux window named `yoloe_log` also follows the current run and
automatically switches to the next `gN` run. Use `Ctrl+b, n` or `Ctrl+b, p` to
move between the inference and live-log windows.

After an idle `N` request is accepted, Base Pose opens one OpenCV window named
`Base Pose Cameras` containing only the configured head and chest RGB streams.
The viewer runs as a separate process with `.venv_teleop/bin/python`, whose
OpenCV build includes the Qt GUI backend; the planner itself continues to run in
`.venv_inference`. The existing `launch_inference` command does not change.
The viewer closes when navigation completes, is cancelled, or the planner exits;
a busy/repeated `N` never creates a duplicate process. Set
`base_pose_raw_live_camera_viewer=false` (or pass
`--no-raw-live-camera-viewer` directly to `base_pose_planner.py`) for headless
runs.

For lightweight offline review, raw YOLOE runs also sample review artifacts in
this per-run layout:

```text
review_samples/
  raw/
    000000.png
    000005.png
  masks/
    000000_target.png
    000000_table.png
    000005_target.png
review_reconstructed/
  000000.jpg
  000005.jpg
```

The default stride of five is about 2 Hz at the 10 Hz detector rate. A
20-second 640x480 run is expected to add roughly 12--18 MB. To replay the most
recently modified raw-YOLOE run after it has finished, use:

```bash
source .venv_inference/bin/activate
RUN_DIR="$(find outputs/base_pose_adjustment -maxdepth 1 -type d \
  -name 'raw_yoloe_*' -printf '%T@ %p\n' | sort -nr | sed -n '1s/^[^ ]* //p')"
test -n "$RUN_DIR"
python gear_sonic/scripts/replay_raw_yoloe_servo.py "$RUN_DIR"
```

This post-run-only script writes reconstructed images under
`review_reconstructed/`; it never starts YOLOE or sends control commands. Only
this post-run command blends masks, draws bounding boxes and diagnostic text,
and encodes the reconstructed JPEGs.
