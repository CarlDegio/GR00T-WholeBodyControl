# Recoverable Camera Stale and Asynchronous Diagnostics Design

## Goal

Keep the raw-YOLOE visual-servo process and active generation alive through
transient continuous-tracking delays while preventing stale motion commands.
Remove diagnostic encoding and disk synchronization from the control path
without losing normally completed per-frame telemetry or every-fifth-frame
review artifacts.

## Scope

This change applies after raw-YOLOE initialization, during continuous tracking.
It does not change grounding, first-frame selection, YOLOE model parameters,
camera calibration, tracking-loss limits, or hard worker-error handling.

## Recoverable soft stale

The default camera soft-stale threshold is `0.4 s`.

While `RawServoRuntime` is aligning, a soft stale immediately changes the
published velocity to zero but does not:

- make `VisualServoController` terminal;
- call `RawServoRuntime._finish`;
- cancel the current generation;
- stop the worker; or
- require a new operator `N`.

The runtime remains in `aligning` with a `soft_stale` flag. It publishes zero at
the normal planner rate while the flag is active. The first fresh, valid
observation clears the flag and is processed normally; its newly computed
command may be published immediately. No multi-frame recovery debounce is
required.

Entering and leaving soft stale each produce one runtime event. Repeated planner
ticks while soft stale do not repeat the warning.

## Event timing

Every worker-created `RawServoEvent` carries `produced_at_monotonic`, captured
after the frame's perception result is ready and immediately before the event is
offered to runtime queues.

Runtime also captures `received_at_monotonic` separately for every event it
removes from a queue. `poll_events` must call the monotonic clock again for each
event; it must not pass one loop-start timestamp to an entire batch.

Worker production time is the source of observation freshness. Receipt time
measures delivery and main-loop delay. A valid observation can resume motion
only when its production age is within `0.4 s`; receiving an already-old event
does not make it fresh. Tests and externally constructed events that omit
production time fall back to their individual receipt time.

## Control event routing

Events are divided into two control paths:

- Lifecycle and safety events (`initialized`, `invalid`, and `error`) use a
  reliable FIFO queue and are never conflated.
- Valid continuous `observation` events use a single-slot latest mailbox.

An observation currently being processed is never interrupted. If another
observation is waiting and a newer observation arrives, only the waiting
observation is displaced. The next control-loop iteration processes the newest
waiting observation. Each main-loop iteration handles at most one observation,
then runs publication and watchdog work before the next iteration.

Every event retains its frame index. Runtime rejects observations older than or
equal to the last applied frame index so a scheduling race cannot apply control
out of order.

This routing bounds queue wait without changing YOLOE inference itself. The
worker continues to run tracking sequentially and the camera subscriber
continues to conflate camera messages before each capture.

## Independent diagnostic data flow

Per-frame diagnostics use one lossless in-process queue and one dedicated
writer thread. Control code never performs review PNG/JPEG encoding,
`raw_servo_frames.jsonl` writes, artifact file creation, or artifact `fsync`.
The small `raw_servo_events.jsonl` lifecycle/control event log remains in
runtime and is outside the per-frame writer protocol.

For each frame, the worker enqueues a produced-frame diagnostic item before it
offers the corresponding control event. The item owns immutable copies of RGB
and masks. A second diagnostic item records the frame's control disposition:

- Applied frames have `control_applied: true` plus the controller snapshot and
  actual command selected for the frame.
- Observations displaced from the latest mailbox have
  `control_applied: false`, `controller: null`, and `command: null`.
- A waiting observation cleared by cancellation or shutdown is also recorded as
  not applied.

The writer joins produced-frame and disposition items by `(generation,
frame_index)`, preserves frame-index order per generation, and then delegates
file formatting to `FrameDiagnosticsWriter`. `raw_servo_frames.jsonl` therefore
continues to contain one row for every produced frame. Every fifth frame still
writes one lossless raw RGB PNG and the available target/table mask PNGs.
Annotated bbox/mask JPEG reconstruction remains post-run only.

The offline replay reader treats null controller/command values as empty display
metadata while continuing to reconstruct bbox and masks from perception data.

## Shutdown and completeness

Stopping robot motion has priority over diagnostic flushing. Normal shutdown:

1. publishes the configured zero-velocity stop sequence;
2. stops and joins the perception worker;
3. marks any observation left in the latest mailbox as not applied;
4. closes the diagnostic queue with a sentinel;
5. waits for the writer to drain and join; and
6. reports `diagnostics flushed`.

Normal task completion, operator cancellation, and soft stale do not discard
already queued diagnostics. A later generation may start while the writer is
finishing an earlier output directory because every item includes its resolved
output directory and generation.

A forced process kill or power loss can still lose items that exist only in
memory. Preventing that would require returning synchronous durability to the
control path and is outside this design.

## Diagnostic failures

Diagnostic encoding or writing failures are not control failures. The writer:

- logs one warning containing the output directory and error;
- disables diagnostics for that generation;
- discards remaining diagnostic items for that disabled generation; and
- continues serving later generations.

It never calls `VisualServoController.note_invalid`, never makes the controller
terminal, and never cancels motion. Worker, camera, calibration, tracking, and
control errors keep their existing safety behavior.

## Alternatives considered

### Dedicated writer process

A process isolates CPU and library failures more strongly but requires
serializing large RGB/mask arrays and more complex generation-aware shutdown.
The extra IPC cost and failure modes are not justified for OpenCV encoding and
disk I/O that can run in a thread.

### Only move PNG encoding off-thread

This is smaller but leaves JSONL/file work on the control path and leaves stale
observation backlog intact. It does not meet the requested latency or
latest-observation behavior.

### Selected design

Use a writer thread, a reliable lifecycle queue, and a single-slot observation
mailbox. This makes the smallest architectural change that separates control
freshness from diagnostic durability.

## Tests

Automated tests must prove:

- the configured soft-stale default is exactly `0.4 s`;
- crossing `0.4 s` publishes zero while phase stays `aligning`, generation stays
  active, and controller remains nonterminal;
- one fresh valid observation immediately resumes control;
- an old received observation cannot resume soft stale;
- each polled event receives a fresh monotonic receipt time;
- lifecycle/invalid/error events are reliable while waiting observations are
  latest-only;
- replacing an observation never interrupts the observation already processing;
- out-of-order or duplicate frame indices are not applied;
- displaced observations are written with `control_applied: false` and null
  controller/command;
- applied observations retain their exact controller/command metadata;
- slow PNG/JSON writes do not delay control-event acceptance or publication;
- normal close drains all JSONL and every-fifth-frame artifacts;
- a diagnostic exception emits a warning and disables that generation without
  terminating control; and
- offline replay reconstructs sampled frames whose control metadata is null.

## Success criteria

The raw servo safely holds zero through a transient observation gap, resumes on
the first fresh valid observation, and does not terminate solely because of the
`0.4 s` soft-stale watchdog or diagnostic I/O. Continuous control consumes only
the newest waiting observation, while a normal shutdown still produces complete
perception diagnostics for every worker-produced frame.
