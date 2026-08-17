# Base Pose YOLOE Keyboard Planner Window Design

> **Superseded behavior:** The exclusive-heartbeat design below was rejected
> after operator review. The final behavior keeps YOLOE selected initially and
> preserves the unmodified `agent_full` key-triggered keyboard publisher. Each
> `w/s/a/d/q/e/Space` message overrides YOLOE only for its existing 0.5-second
> duration; the relay then automatically returns to YOLOE. `x` exits the
> keyboard source. No idle keyboard heartbeat is published. The keyboard runs
> visibly in `inference` pane 5 rather than in a separate tmux window. This
> note is the authoritative correction to the historical design sections below.

## Goal

When `launch_inference.py` runs with `planner_input=base_pose` and
`base_pose_mode=raw_yoloe_servo`, add a dedicated tmux window that provides the
same immediate planner keyboard controls used by the `agent_full` branch:

- `w` / `s`: forward / backward;
- `a` / `d`: left / right translation;
- `q` / `e`: left / right yaw;
- `Space`: stop;
- `x`: exit manual control and return control to YOLOE.

Do not port `agent_full`'s ControlGateway or change any other planner mode.

## Architecture

The launcher creates a `base_keyboard` tmux window only for raw YOLOE BasePose.
That window runs the existing `keyboard_planner_thread_server.py` on a dedicated
manual-command port. Its existing keys, speeds, and command JSON remain the
source of truth.

An opt-in exclusive mode makes the keyboard server publish at its configured
rate. A key command remains active for the existing command duration and then
the server publishes zero velocity while it is idle. This continuous zero
heartbeat keeps manual control exclusive and prevents YOLOE from resuming as
soon as a key is released.

`lavira_sonic_relay.py` accepts an optional manual source in addition to its
existing autonomous source. A fresh manual heartbeat always wins. If the
keyboard process exits or fails, its heartbeat becomes stale and the relay
falls back to the latest valid YOLOE command. The relay remains the only process
that binds the SONIC planner output port, so the change introduces no competing
publishers.

## Launch and Data Flow

1. BasePose YOLOE publishes autonomous velocity JSON on the existing port 5558.
2. The new keyboard window publishes compatible velocity JSON on a dedicated
   port (default 5566).
3. The relay subscribes to both ports and publishes exactly one selected SONIC
   planner stream on the existing output port 5563.
4. While keyboard heartbeats are fresh, the selected stream is the keyboard
   velocity, including idle zero velocity.
5. After `x`, process failure, or window closure, the heartbeat expires and the
   selected stream returns to YOLOE.

The launcher returns focus to the main `inference` window after creating and
starting `base_keyboard`.

## Safety and Failure Behavior

- Manual takeover begins with a zero-velocity heartbeat, so merely starting the
  keyboard window cannot cause motion.
- Releasing a movement key produces zero after its configured duration; it does
  not resume YOLOE.
- `Space` immediately selects zero velocity and keeps manual takeover active.
- `x` sends the keyboard server's existing final stop burst before exiting;
  YOLOE resumes only after the manual-source timeout.
- Invalid manual messages are ignored without affecting the autonomous source.
- If the keyboard process crashes, the bounded heartbeat timeout restores YOLOE
  rather than leaving a stale nonzero manual command active.
- Existing velocity bounds and orientation integration remain in the relay and
  apply identically to both sources.

## Compatibility

The keyboard server's new continuous/exclusive behavior is opt-in. Existing
`planner_input=keyboard`, LaViRA, RGB, and depth-based BasePose launches retain
their current behavior and tmux layout. The user's existing uncommitted launch
configuration, including the camera pitch override, must be preserved.

## Verification

- Unit tests prove the keyboard's exclusive state emits a timed movement then
  continuous zero heartbeats and exits on `x`.
- Relay tests prove fresh manual input overrides YOLOE and stale manual input
  falls back to YOLOE.
- Launcher tests prove the dedicated command, port wiring, raw-YOLOE-only
  condition, and tmux window creation.
- Focused tests, Python compilation, and `git diff --check` run without starting
  robot or ZMQ hardware processes.

