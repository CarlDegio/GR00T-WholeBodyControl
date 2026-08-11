# Recoverable Camera Stale and Asynchronous Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace terminal 0.35-second camera stale handling with a recoverable 0.4-second zero-velocity hold, consume only the newest waiting observation, and move per-frame diagnostics to a drainable writer thread.

**Architecture:** Lifecycle and safety events remain reliable FIFO messages while continuous valid observations use a single-slot latest mailbox. Worker frames and runtime control dispositions flow through a separate lossless queue; a dedicated thread merges them by generation/frame index and writes JSONL plus every-fifth-frame artifacts.

**Tech Stack:** Python 3.10, `threading`, `queue`, NumPy, OpenCV, pytest

## Global Constraints

- The soft-stale threshold is exactly `0.4 s`.
- Soft stale publishes zero but never terminates or cancels the active generation.
- The first fresh valid observation resumes immediately.
- In-flight control processing is never cancelled; only a waiting observation may be replaced.
- Every worker-produced frame is diagnosed; displaced frames use `control_applied=false` and null control metadata.
- Per-frame diagnostic failure warns once and disables only that generation's diagnostics.
- Preserve unrelated dirty and untracked workspace changes.

---

### Task 1: Recoverable soft-stale timing

**Files:**
- Modify: `gear_sonic/scripts/base_pose_planner.py`
- Modify: `gear_sonic/utils/inference/base_pose_visual_servo.py`
- Test: `gear_sonic/tests/test_base_pose_visual_servo.py`

**Interfaces:**
- Produces: `RawServoEvent.produced_at_monotonic: float | None`
- Produces: `RawServoRuntime.soft_stale: bool`
- Produces: individual receipt times from an injected monotonic clock

- [ ] **Step 1: Write failing soft-stale tests**

Test the exact `0.4` default, zero publication without `_finish`, preservation of `aligning`/generation/nonterminal state, immediate recovery on one fresh observation, rejection of an already-old observation, and a fresh receipt timestamp for each polled event.

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv_inference/bin/python -m pytest gear_sonic/tests/test_base_pose_visual_servo.py -k 'soft_stale or event_receipt_clock' -v`

Expected: failures because the default is `0.35`, stale is terminal, and events have no production time.

- [ ] **Step 3: Implement minimal timing behavior**

Change the default, add production/receipt fields and clock injection, replace terminal stale with a zero-command flag, and only clear it for a fresh valid observation.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `.venv_inference/bin/python -m pytest gear_sonic/tests/test_base_pose_visual_servo.py -k 'soft_stale or event_receipt_clock' -v`

### Task 2: Reliable events and latest-observation mailbox

**Files:**
- Modify: `gear_sonic/utils/inference/base_pose_visual_servo.py`
- Test: `gear_sonic/tests/test_base_pose_visual_servo.py`

**Interfaces:**
- Produces: FIFO delivery for `initialized`, `invalid`, and `error`
- Produces: `queue.Queue(maxsize=1)` replacement for `observation`
- Produces: `last_applied_frame_index` ordering guard

- [ ] **Step 1: Write failing routing tests**

Test that two waiting observations leave only the newest, safety events are not displaced, in-flight acceptance completes, each loop handles at most one observation, and duplicate/out-of-order indices do not update control.

- [ ] **Step 2: Run routing tests and verify RED**

Run: `.venv_inference/bin/python -m pytest gear_sonic/tests/test_base_pose_visual_servo.py -k 'mailbox or lifecycle or out_of_order' -v`

- [ ] **Step 3: Implement routing and ordering**

Add the reliable queue and one-slot mailbox, route worker events by kind, and process at most one mailbox observation per control-loop iteration.

- [ ] **Step 4: Run routing plus invalid-frame safety tests**

Run: `.venv_inference/bin/python -m pytest gear_sonic/tests/test_base_pose_visual_servo.py -k 'mailbox or lifecycle or out_of_order or invalid' -v`

### Task 3: Drainable asynchronous per-frame diagnostics

**Files:**
- Modify: `gear_sonic/utils/inference/base_pose_visual_servo_diagnostics.py`
- Test: `gear_sonic/tests/test_base_pose_visual_servo_diagnostics.py`

**Interfaces:**
- Produces: `AsyncFrameDiagnosticsWriter.submit_frame(...) -> None`
- Produces: `AsyncFrameDiagnosticsWriter.submit_decision(...) -> None`
- Produces: `AsyncFrameDiagnosticsWriter.close(drain=True) -> None`
- Extends: `FrameDiagnosticsWriter.write` with `control_applied` and nullable control metadata

- [ ] **Step 1: Write failing row-schema tests**

Assert applied rows retain exact control data, while displaced rows have `control_applied=false`, null controller/command, and unchanged every-fifth-frame RGB/mask artifacts.

- [ ] **Step 2: Run schema tests and verify RED**

Run: `.venv_inference/bin/python -m pytest gear_sonic/tests/test_base_pose_visual_servo_diagnostics.py -k 'control_applied or displaced' -v`

- [ ] **Step 3: Implement nullable synchronous row formatting**

Preserve existing paths and `annotated_image: null` while adding truthful applied/displaced metadata.

- [ ] **Step 4: Write failing writer-thread tests**

Use synchronization events, not sleeps, to prove blocked disk work does not block submit calls; prove `close(drain=True)` writes every pending row in frame order; inject a write exception and assert one warning plus per-generation disable.

- [ ] **Step 5: Implement queue, merge, drain, and failure isolation**

Use a lossless queue, immutable frame ownership, a pending map keyed by `(generation, frame_index)`, per-output synchronous writers, a sentinel, and deterministic join.

- [ ] **Step 6: Run the diagnostics module**

Run: `.venv_inference/bin/python -m pytest gear_sonic/tests/test_base_pose_visual_servo_diagnostics.py -q`

### Task 4: Integrate worker, mailbox disposition, and shutdown

**Files:**
- Modify: `gear_sonic/utils/inference/base_pose_visual_servo.py`
- Modify: `gear_sonic/scripts/base_pose_planner.py`
- Test: `gear_sonic/tests/test_base_pose_visual_servo.py`
- Test: `gear_sonic/tests/test_base_pose_visual_servo_diagnostics.py`

**Interfaces:**
- Consumes: `AsyncFrameDiagnosticsWriter`
- Produces: worker frame submission before control routing
- Produces: applied/displaced decision submission
- Produces: post-worker `flush_diagnostics()` shutdown hook

- [ ] **Step 1: Write failing integration tests**

Prove displaced observations remain recorded, applied observations record the actual post-update command, a blocked writer does not delay `accept_event`, cancellation marks the waiting observation not applied, and writer failure leaves runtime aligning/nonterminal.

- [ ] **Step 2: Run integration tests and verify RED**

Run: `.venv_inference/bin/python -m pytest gear_sonic/tests/test_base_pose_visual_servo.py gear_sonic/tests/test_base_pose_visual_servo_diagnostics.py -k 'diagnostic or displaced or mailbox' -v`

- [ ] **Step 3: Wire frame and decision messages**

Submit immutable frames from the worker, decisions from runtime/mailbox displacement, remove synchronous `_record_frame`, and retain lightweight runtime event logging.

- [ ] **Step 4: Implement stop-first drain ordering**

Stop robot motion, join perception, mark the final waiting observation displaced, close/drain/join diagnostics, and log `diagnostics flushed`.

- [ ] **Step 5: Run runtime and diagnostic tests**

Run: `.venv_inference/bin/python -m pytest gear_sonic/tests/test_base_pose_visual_servo.py gear_sonic/tests/test_base_pose_visual_servo_diagnostics.py -q`

### Task 5: Replay compatibility, documentation, and verification

**Files:**
- Modify: `gear_sonic/scripts/replay_raw_yoloe_servo.py`
- Modify: `gear_sonic/tests/test_replay_raw_yoloe_servo.py`
- Modify: `docs/base_pose_adjustment.md`

**Interfaces:**
- Consumes: nullable controller/command diagnostic rows
- Preserves: every-fifth-frame reconstructed bbox/mask JPEG output

- [ ] **Step 1: Write a failing replay test with null control metadata**

Create sampled displaced rows and assert the renderer still emits the expected JPEGs.

- [ ] **Step 2: Run the replay test and verify RED**

Run: `.venv_inference/bin/python -m pytest gear_sonic/tests/test_replay_raw_yoloe_servo.py -k 'null_control' -v`

- [ ] **Step 3: Make replay null-tolerant and update documentation**

Document the 0.4-second zero hold, single-frame recovery, latest mailbox, background drain, and `diagnostics flushed` handoff.

- [ ] **Step 4: Run full component regression and syntax checks**

Run: `.venv_inference/bin/python -m pytest gear_sonic/tests -q`

Run: `.venv_inference/bin/python -m py_compile gear_sonic/utils/inference/base_pose_visual_servo.py gear_sonic/utils/inference/base_pose_visual_servo_diagnostics.py gear_sonic/scripts/base_pose_planner.py gear_sonic/scripts/replay_raw_yoloe_servo.py`

Run: `git diff --check`

- [ ] **Step 5: Inspect final scoped diff**

Preserve unrelated work and leave implementation uncommitted if staging would capture pre-existing user changes in untracked visual-servo files.
