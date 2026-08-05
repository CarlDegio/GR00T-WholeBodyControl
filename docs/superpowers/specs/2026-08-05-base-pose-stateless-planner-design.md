# Base-Pose Stateless Planner Design

## Context

The base-pose launch path added a shared ready/request token between
`run_vla_inference.py`, `base_pose_planner.py`, and `lavira_sonic_relay.py`.
The relay also subscribed to `g1_debug` and attached frozen upper-body and hand
targets to every planner message.

Live logs showed that the relay successfully latched the current upper body and
hands, but repeated the latch roughly every planner tick because the request
file remained present. This repeatedly rewrote the ready file. BasePose treated
any missing or changed token while inference or motion was active as
`cpp_planner_not_ready` and cancelled the generation. Stopping the VLA Python
process also removed the marker files and triggered the same cancellation.

The operator has chosen to remove this synchronization and frozen-pose path
entirely for base-pose adjustment. C++ is responsible for retaining its own
upper-body state while the Python planner publishes only base motion.

## Goals

- Restore the `k`, `i`, `o`, and `p` interpretation in
  `run_vla_inference.py` to the behavior on the repository's `replay_real`
  branch.
- Make the first `k` send C++ planner start and the second `k` send stop without
  marker-file or robot-state coordination.
- Make BasePose planning and execution independent of ready/request files and
  frozen upper-body or hand state.
- Make `Space` cancel any pending inference or active motion and immediately
  publish a configured sequence of all-zero base-velocity commands.
- Preserve the existing `5558 -> relay -> 5563` process and socket topology.

## Non-Goals

- Do not change Codex prompting, model selection, reasoning effort, camera
  capture, or motion-command generation.
- Do not change the planner binary protocol or C++ controller.
- Do not make BasePose publish directly to port 5563.
- Do not remove the generic frozen-pose capability from
  `lavira_sonic_relay.py`; BasePose simply stops enabling or depending on it.
- Do not restart or send commands to live robot processes during automated
  verification.

## Architecture

### Pane 1: Keyboard publisher

The embedded keyboard publisher in `launch_inference.py` remains behaviorally
identical to `replay_real`. It reads one line at a time and sends the raw key to
port 5580, except `t <text>`, which is sent as `prompt:<text>`.

This pane has no control-loop state. It does not decide whether `k` means start
or stop.

### Pane 2: VLA inference and C++ control state

`run_vla_inference.py` owns the local `cpp_loop_running` and `cpp_mode` state.
Its keyboard branches match `replay_real`:

- If `cpp_loop_running` is false, `k` sends
  `start=True, stop=False, planner=True`. On a successful send it records
  `cpp_loop_running=True` and `cpp_mode="PLANNER"`.
- If `cpp_loop_running` is true, `k` sends
  `start=False, stop=True`, with the current planner flag. On a successful send
  it records `cpp_loop_running=False` and `cpp_mode="OFF"`.
- `i` publishes the initial-pose ramp and switches a running planner loop to
  POSE mode, as on `replay_real`.
- `o` switches a running POSE loop to PLANNER mode.
- `p` pauses or resumes policy inference outside PLANNER mode.

The BasePose launch command no longer passes hold-ready paths or timeouts to
this process. The marker helpers, direct-toggle helper, frozen-frame checks,
and shutdown marker cleanup are removed.

### Pane 3: BasePose planner

BasePose has only three operator commands:

- `N`: when idle, enqueue one inference generation and publish an immediate
  zero base-velocity command while inference runs.
- `Space`: invalidate the current generation, discard queued work, cancel any
  active motion, and immediately publish `final_stop_count` zero base-velocity
  commands.
- `X`: perform the same zero-velocity cancellation and then exit.

BasePose does not accept a planner-ready-file option, read token files, store a
ready token, reject `N` for missing readiness, or monitor token changes during
inference or motion. The `HOLD cpp_planner_not_ready` state and log are removed.
Cancellation logs use `STOP <reason>` to reflect zero-velocity stopping rather
than pose locking.

Each zero command has source velocity `[vx, vy, wz] = [0.0, 0.0, 0.0]`.

### Pane 4: Stateless SONIC relay

For BasePose, `launch_inference.py` starts `lavira_sonic_relay.py` without:

- `--freeze-current-upper-body`;
- `--state-host` or `--state-port`;
- `--hold-ready-file`.

The relay therefore does not subscribe to port 5557 and never adds
upper-body, upper-body-velocity, left-hand, or right-hand fields. It converts
the BasePose source velocity to the existing SONIC planner message at 20 Hz.
For a zero source velocity, the encoded movement is all zero and speed is zero.

## Data Flow

1. Pane 1 publishes a keyboard string to port 5580.
2. Pane 2 interprets `k/i/o/p` and publishes C++ command messages to port 5556.
3. Pane 3 independently interprets `N/Space/X` and publishes JSON base-velocity
   commands to port 5558.
4. Pane 4 converts the latest base velocity to a SONIC planner message on port
   5563.
5. Pane 2 forwards planner messages from port 5563 to the C++ action publisher
   on port 5556 while in PLANNER mode.

No file or robot-state edge participates in this control path.

## Error and Shutdown Behavior

- A failed C++ command send leaves Pane 2's local C++ state unchanged.
- `N` remains rejected while another inference or motion generation is active.
- `Space` is always accepted and publishes the zero sequence even if the
  runtime is already idle.
- Stale worker results are rejected through the existing generation check.
- BasePose shutdown publishes the zero sequence before closing its socket.
- Relay shutdown preserves its existing zero-message and C++ stop sequence;
  for BasePose those zero messages carry no frozen body or hand fields.

## Test Strategy

- Verify the `k` start/stop calls and local mode transitions match
  `replay_real`, with no marker-file side effects.
- Verify the BasePose launch commands contain no ready, request, freeze-current,
  or state-port arguments.
- Verify `N` starts without any marker file.
- Verify creating, changing, or deleting a former marker path cannot cancel a
  BasePose generation.
- Verify `Space` cancels inference and active motion and publishes exactly
  `final_stop_count` messages whose velocity is `[0.0, 0.0, 0.0]`.
- Verify stateless relay messages omit all upper-body and hand fields.
- Run focused unit tests and Python compilation without connecting to live ZMQ
  robot ports.

## Success Criteria

- First `k` starts C++ in PLANNER mode and second `k` stops it, using the
  `replay_real` state machine.
- BasePose can start with `N` without a ready file.
- `HOLD cpp_planner_not_ready` is unreachable because the readiness state and
  transition no longer exist.
- `Space` always produces immediate all-zero base planner velocity and has no
  dependency on robot state or upper-body/hand latching.
- No BasePose launch process creates or reads
  `/tmp/sonic_base_pose_hold_*.ready` or its `.request` companion.
