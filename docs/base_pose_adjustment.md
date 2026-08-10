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

The `rgb` experiment consumes only `ego_view`. The `rgbd` and
`rgb_depth_query` experiments require the deployment machine's
`.venv_lingbot_depth` environment and local LingBot-Depth model. They never
fall back to raw RealSense depth or RGB-only planning when enhanced depth is
unavailable.

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
`gpt-5.6-sol`, max reasoning effort, a `600 s` Codex timeout, camera height
`1.2 m`, pitch `-47.6°`, and vertical FOV `55.2°`. Camera intrinsics and
horizontal FOV come from the live frame.

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
`outputs/base_pose_adjustment/<timestamp>/`. Global cancellation/failure events
that occur before a request directory is known are appended to
`outputs/base_pose_adjustment/runtime_events.jsonl`.

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
`review_reconstructed/`; it never starts YOLOE or sends control commands.
