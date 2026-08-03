# LaViRA Single-Cycle Planner over REASAN

## Context

The target branch is `replay_real` at `5dae650`. It already contains the
MID-360/IMU input, REASAN ONNX safety filter, and SONIC planner relay. It does
not contain the standalone `navila_planner.py` found on the `navila` branch.

This change adds LaViRA/AgentNav as a new command source. It does not replace
or modify the existing keyboard command source, REASAN filter, or VLA relay.

## Goals

- Start LaViRA in the existing `launch_inference.py` tmux layout.
- Trigger exactly one RGB-D AgentNav cycle from the pane 3 keyboard.
- Convert the resulting rotate-then-translate batch into the existing
  `navila_reasan_velocity_command` protocol on port 5558.
- Keep pure rotation on the existing REASAN `TURN-BYPASS` path.
- Pass translation through the existing REASAN ONNX filter before SONIC.
- Fail closed: camera, depth, Codex, geometry, validation, cancellation, and
  shutdown failures must end in repeated zero-velocity messages.
- Minimize the port by excluding the agent-nav sidecars, direct SONIC planner,
  continuous replanning, and ObjectNav-to-VLA handoff launcher.

## Non-goals

- Continuous autonomous replanning.
- Automatic VLA handoff after reaching a target.
- Importing or changing `navila_planner.py`.
- Modifying `reasan_planner.py`, `run_vla_inference.py`,
  `keyboard_planner_thread_server.py`, or `mid360_reasan_open3d.py`.
- Removing `TURN-BYPASS` or retraining the REASAN filter.
- Supporting backward, lateral, or combined translation-and-yaw AgentNav
  commands.

## Architecture

`launch_inference.py` selects one pane 3 command source:

```text
planner_input=keyboard -> keyboard_planner_thread_server.py
planner_input=lavira   -> lavira_planner.py
```

The LaViRA path is:

```text
chest RGB-D :5555
  -> lavira_planner.py (pane 3, N triggers one cycle)
  -> object_nav.py
  -> object_nav_geometry.py
  -> navila_reasan_velocity_command :5558
  -> reasan_planner.py (pane 4)
       rotation: TURN-BYPASS
       translation: REASAN ONNX filter + MID-360/IMU :5562
  -> filtered planner messages :5563
  -> run_vla_inference.py
  -> SONIC
```

LaViRA never binds the SONIC action port and never publishes SONIC planner
binary messages directly.

## Files

### New production files

`gear_sonic/scripts/lavira_planner.py`

- Owns pane 3 keyboard interaction and the single-cycle state machine.
- Runs AgentNav inference in a background worker so stop and exit keys remain
  responsive.
- Converts validated AgentNav commands through the existing
  `build_navila_message` helper without changing the keyboard module.
- Republishes the active velocity command at 20 Hz for its duration.
- Publishes repeated stop messages between phases and on every terminal path.

`gear_sonic/utils/inference/object_nav.py`

- Reads composed chest RGB-D frames and camera calibration.
- Collects five depth frames and sends the policy RGB frame to Codex.
- Validates the Codex policy response and returns one `ObjectNavResult`.
- Saves bounded diagnostic artifacts under `outputs/object_nav`.
- Excludes direct Sonic communication and continuous replanning.

`gear_sonic/utils/inference/object_nav_geometry.py`

- Validates normalized bounding boxes and aligned uint16 depth.
- Projects the target through camera intrinsics.
- Fuses valid depth measurements and creates exactly two commands: pure
  rotation followed by pure translation.
- Contains no camera, Codex, ZMQ, REASAN, or SONIC dependencies.

### Modified files

- `gear_sonic/camera/sensor_server.py`: add schema v2 `camera_info` and lossless
  uint16 PNG depth while retaining legacy RGB compatibility.
- `gear_sonic/camera/drivers/realsense.py`: align depth to color and publish
  intrinsics and depth scale.
- `gear_sonic/camera/composed_camera.py`: enable depth only for the chest
  RealSense and merge camera calibration.
- `start_camera_server.zsh`: pass `--realsense-enable-depth`.
- `gear_sonic/scripts/launch_inference.py`: add
  `planner_input: Literal["keyboard", "lavira"]`, LaViRA configuration, pane 3
  launch selection, and keyboard help.

### Tests

- `gear_sonic/tests/test_object_nav_geometry.py` for projection, depth fusion,
  command calculation, limits, and invalid input.
- `gear_sonic/tests/test_lavira_planner.py` for state transitions, protocol
  conversion, publication cadence, cancellation, stale worker results, and
  fail-closed stop behavior.
- `gear_sonic/tests/test_camera_rgbd_protocol.py` for uint16 depth,
  calibration, and legacy RGB compatibility.

## Keyboard controls

When `planner_input=lavira`:

- `n`: start one AgentNav cycle only while idle.
- `Space`: cancel inference or motion and publish stop.
- `x`: cancel, publish stop, and exit.
- `w`/`s`: cancel automation and send manual forward/backward.
- `a`/`d`: cancel automation and send manual left/right translation.
- `q`/`e`: cancel automation and send manual left/right yaw.
- Other keys are ignored. A second `n` while busy is rejected, not queued.

Manual mappings continue to use the current `replay_real` keyboard defaults.
They are not AgentNav parameters.

## State machine

```text
IDLE --N--> INFERENCING
INFERENCING --NAVIGATE--> ROTATING
ROTATING --> STOP_GAP --> FORWARD --> FINAL_STOP --> IDLE
```

Zero-duration phases are skipped. A `STOP` result goes directly to
`FINAL_STOP`. `FAILED`, `REJECTED`, invalid commands, user cancellation, and
shutdown also go to `FINAL_STOP` and never publish a new nonzero command.

Inference runs in a worker with a generation identifier. Cancellation bumps
the generation; a late worker result from an earlier generation is discarded.
The main loop retains control of keyboard input, timers, and ZMQ publication.

## AgentNav parameters

Automatic navigation values are copied from `origin/agent-nav` without safety
overrides:

| Parameter | Value |
| --- | ---: |
| `rotation_speed` | 0.4 rad/s |
| `forward_speed` | 0.3 m/s |
| `target_standoff_distance` | 0.0 m |
| `max_direct_travel` | 8.0 m |
| `min_rotation_angle` | 2 degrees |
| `depth_window_radius` | 3 pixels |
| `min_valid_depth` | 100 mm |
| `depth_frame_count` | 5 |
| `policy_frame_index` | 2 (zero-based) |
| `min_valid_frame_count` | 3 |
| `planner_hz` | 20 Hz |
| `transition_pause` | 0.5 s |
| `max_speed` | 0.5 m/s |
| `max_duration` | 30 s |
| `max_abs_yaw` | pi rad |
| `camera_timeout` | 3000 ms |
| `codex_timeout` | 180 s |
| `min_confidence` | 0.6 |

The upstream name `safe_distance` becomes `target_standoff_distance` to avoid
confusion with REASAN obstacle safety. Its formula and default remain unchanged:

```text
travel = max(0, measured_target_distance - target_standoff_distance)
```

Durations remain:

```text
rotation_duration = abs(angle_rad) / rotation_speed
forward_duration = travel / forward_speed
```

Validation follows agent-nav: exactly two commands, pure rotation then pure
translation, finite nonnegative duration, duration no greater than 30 seconds,
positive duration no shorter than one 20 Hz period, translation speed no
greater than 0.5 m/s, and absolute relative yaw no greater than pi. A direct
travel over 8 m is rejected rather than truncated.

## Publication and safety behavior

The LaViRA publisher repeats the active command at 20 Hz. This is necessary
because `reasan_planner.py` treats an upstream command as stale after at most
its configured command timeout. Phase duration is measured with a monotonic
clock, not inferred from message count.

The sequence is:

```text
rotation -> stop for 0.5 s -> translation -> repeated final stop -> IDLE
```

REASAN may reduce translation velocity. LaViRA does not extend the duration to
compensate, so filtering can only shorten the actual travel. Pure yaw retains
the existing `TURN-BYPASS` behavior.

On loss of the LaViRA publisher, REASAN's freshness timeout provides a second
zero-velocity boundary. Loss or invalidity of ActorRay/IMU continues to invoke
the existing REASAN safe-stop behavior.

`target_standoff_distance=0.0` means LaViRA does not reserve target standoff.
It does not replace REASAN's dynamic obstacle filtering.

## Launch configuration

`launch_inference.py` keeps `keyboard` as the default. The LaViRA selection
passes mission, global target, camera endpoint, port 5558 output, Codex timeout,
confidence, motion parameters, and output directory into pane 3. It uses the
inference environment because RGB-D/Codex dependencies are not guaranteed in
the teleoperation environment.

No pane, action port, REASAN port, or VLA relay port is added or changed.

## Verification and rollout

1. Add RGB-D schema tests and verify legacy RGB decoding.
2. Add geometry tests with synthetic aligned depth and calibration.
3. Test `ObjectNavRunner` with fake camera and Codex clients.
4. Test the LaViRA state machine with a fake clock, worker results, and
   publisher; verify the exact rotate/stop/translate/stop sequence.
5. Run a local ZMQ integration test across ports 5558 and 5563 with synthetic
   ActorRay/IMU data; verify `TURN-BYPASS`, filtered translation, stale command
   stop, and cancellation stop.
6. Test in MuJoCo before hardware.
7. On initial hardware runs, operators may override the upstream 8 m maximum
   from the CLI with a smaller value; the implementation default remains the
   agent-nav value required by this design.

Completion requires relevant unit tests to pass and evidence that the existing
keyboard planner path still launches unchanged by default.
