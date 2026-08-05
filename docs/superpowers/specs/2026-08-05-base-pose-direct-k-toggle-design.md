# Base-pose direct K toggle design

## Goal

In base-pose mode, the Python keyboard handler must treat `k` as a direct
control-loop toggle:

- the first `k` immediately sends a planner-mode start command;
- the second `k` immediately sends a stop command.

Neither transition may wait for the planner hold-ready marker or for a robot
state sample.

## Design

Keep `run_vla_inference.py` as the owner of the local C++ loop state. When the
loop is off, handling `k` sends `start=True`, `stop=False`, and `planner=True`
before performing any optional pose-latch work. When the loop is running,
handling `k` sends `start=False`, `stop=True`, and preserves the current planner
mode in the command.

After a successful start, Python may request a fresh frozen upper-body and hand
pose, but this request must be non-blocking. The relay may satisfy it after C++
enters CONTROL and begins publishing `g1_debug`. Failure to create or satisfy
the request is diagnostic only and must not undo or delay the start command.

Stopping removes stale hold-ready and request markers without waiting for the
relay.

## Error handling

A ZMQ send exception keeps the existing behavior: print a warning and do not
change the local running state. Hold-request failures print a warning but do not
change a successfully started C++ state.

## Tests

Add focused keyboard-handler coverage proving that:

1. with no hold-ready marker, the first `k` sends start immediately;
2. the first `k` does not call a blocking hold wait or synchronous hold forward;
3. after start, the second `k` sends stop immediately;
4. stale hold marker files are cleaned when stopping.

The change must not alter non-base-pose keyboard behavior, planner motion
generation, camera handling, or Codex inference.
