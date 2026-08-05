# Base-Pose Stateless Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore replay-real keyboard control behavior and make BasePose publish base-only planner velocities with immediate all-zero `Space` cancellation and no ready-file, robot-state, upper-body, or hand latch dependency.

**Architecture:** Keep the existing `5558 -> lavira_sonic_relay -> 5563` topology, but launch the relay without state or frozen-pose options. Remove the ready-token state from BasePose and VLA inference, restore the `replay_real` `k/i/o/p` branches, and let C++ retain its own upper-body state while Python publishes only base velocity.

**Tech Stack:** Python 3.10, ZeroMQ, dataclasses, `unittest`, existing pytest-style contract files, SONIC packed planner messages.

## Global Constraints

- Pane 1 remains a dumb keyboard string publisher on port 5580.
- The first `k` sends `start=True, stop=False, planner=True`; the second sends `start=False, stop=True`.
- The `k`, `i`, `o`, and `p` branch behavior in `run_vla_inference.py` must match `replay_real`.
- BasePose must not create, read, compare, or delete ready/request marker files.
- BasePose relay launch must not use `--freeze-current-upper-body`, `--state-host`, `--state-port`, or `--hold-ready-file`.
- BasePose planner messages must not contain upper-body or hand target fields.
- `Space` must immediately publish exactly `final_stop_count` source commands with `[vx, vy, wz] = [0.0, 0.0, 0.0]` after cancelling work.
- Preserve the generic frozen-pose capability in `lavira_sonic_relay.py`; only the BasePose launch path stops using it.
- Do not change Codex prompting, model selection, reasoning effort, camera capture, motion-plan validation, or C++ code.
- Do not restart, stop, or send commands to any live robot process during automated verification.
- The worktree already contains user-approved prompt/ultra edits in some target files. Preserve them, inspect every staged diff, and never silently bundle unrelated hunks into a stateless-planner commit.
- No available local environment contains pytest. Persist pytest-style tests for CI, but use the provided `.venv_inference` assertion scripts and stdlib `unittest` commands for local RED/GREEN evidence. Do not install packages from the network.

---

### Task 1: Launch BasePose with the stateless relay

**Files:**
- Modify: `gear_sonic/scripts/launch_inference.py:160-230,338-482,800-965`
- Modify: `gear_sonic/tests/test_lavira_planner.py:25-31,190-238`

**Interfaces:**
- Consumes: `build_planner_input_command(config, repo_root) -> str` and `build_reasan_planner_command(config, repo_root) -> str`.
- Produces: BasePose launch commands with no marker path, state subscription, or frozen-pose option.

- [ ] **Step 1: Write the failing launch contract tests**

In `gear_sonic/tests/test_lavira_planner.py`, remove the
`base_pose_hold_ready_file` import. Add the negative assertion to the existing
RGB planner command test:

```python
assert "--planner-ready-file" not in command
```

Replace `test_base_pose_always_uses_direct_frozen_pose_relay_without_reasan`
with:

```python
def test_base_pose_uses_stateless_direct_relay_without_reasan() -> None:
    config = InferenceLaunchConfig(planner_input="base_pose", reasan_avoidance=True)

    command = build_reasan_planner_command(config, Path("/workspace/sonic"))

    assert "lavira_sonic_relay.py" in command
    assert "--freeze-current-upper-body" not in command
    assert "--state-host" not in command
    assert "--state-port" not in command
    assert "--hold-ready-file" not in command
    assert "reasan_planner.py" not in command
    assert not uses_reasan_avoidance(config)
```

- [ ] **Step 2: Run a software-only assertion and verify RED**

Run:

```bash
./.venv_inference/bin/python - <<'PY'
from pathlib import Path
from gear_sonic.scripts.launch_inference import (
    InferenceLaunchConfig,
    build_planner_input_command,
    build_reasan_planner_command,
)

config = InferenceLaunchConfig(
    planner_input="base_pose",
    base_pose_task="put the paper ball in the basket",
)
planner = build_planner_input_command(config, Path("/workspace/sonic"))
relay = build_reasan_planner_command(config, Path("/workspace/sonic"))
assert "--planner-ready-file" not in planner
for forbidden in (
    "--freeze-current-upper-body",
    "--state-host",
    "--state-port",
    "--hold-ready-file",
):
    assert forbidden not in relay, (forbidden, relay)
PY
```

Expected: `AssertionError` because the current planner command contains
`--planner-ready-file` and the relay contains all four forbidden options.

- [ ] **Step 3: Remove BasePose marker and state wiring from launch**

In `gear_sonic/scripts/launch_inference.py`:

1. Delete `InferenceLaunchConfig.base_pose_hold_ready_timeout_seconds`.
2. Delete `base_pose_hold_ready_file`.
3. Remove `--planner-ready-file` from `_base_pose_planner_command`.
4. Change the BasePose branch of `build_reasan_planner_command` to:

```python
if config.planner_input == "base_pose":
    return (
        f"cd {shlex.quote(str(repo_root))} && "
        f"source .venv_teleop/bin/activate && "
        f"python gear_sonic/scripts/lavira_sonic_relay.py "
        f"--source tcp://127.0.0.1:{config.keyboard_planner_port} "
        f"--output 'tcp://*:{config.reasan_planner_port}' --hz 20"
    )
```

5. Delete the `planner_input == "base_pose"` block that appends
   `--planner-hold-ready-file` and `--planner-hold-ready-timeout-seconds` to
   `inference_cmd`.
6. Change the BasePose workflow text from “latches current upper body/hands” to
   “starts C++ in PLANNER mode”.
7. Describe Pane 4 as a stateless BasePose-to-SONIC relay rather than a hold
   relay.

Do not alter the embedded Pane 1 keyboard script.

- [ ] **Step 4: Verify GREEN locally**

Re-run the exact assertion script from Step 2. Expected: exit 0.

Also run:

```bash
python3 -m py_compile gear_sonic/scripts/launch_inference.py
```

Expected: exit 0.

- [ ] **Step 5: Inspect and commit only Task 1 hunks**

Run:

```bash
git diff --check -- \
  gear_sonic/scripts/launch_inference.py \
  gear_sonic/tests/test_lavira_planner.py
git diff -- \
  gear_sonic/scripts/launch_inference.py \
  gear_sonic/tests/test_lavira_planner.py
```

Confirm the diff preserves the already-approved `ultra` defaults. Stage only
the stateless-launch hunks; if the prompt/ultra hunks are still unstaged, keep
them out of this commit. Before committing, `git diff --cached --name-only`
must list only the two Task 1 files and `git diff --cached` must contain only
Task 1 changes.

```bash
git add -p \
  gear_sonic/scripts/launch_inference.py \
  gear_sonic/tests/test_lavira_planner.py
git diff --cached --check
git diff --cached
git commit -m "fix: launch base-pose with stateless relay"
```

At each `git add -p` prompt, accept only the Task 1 marker/state-removal and
stateless-relay test hunks; reject the pre-existing reasoning-effort hunks.

---

### Task 2: Remove BasePose HOLD readiness and make Space an explicit zero stop

**Files:**
- Modify: `gear_sonic/scripts/base_pose_planner.py:41-60,268-372,449-481,590-620`
- Modify: `gear_sonic/tests/test_base_pose_adjustment.py:190-335`

**Interfaces:**
- Produces: `BasePosePlannerRuntime.cancel_and_stop(reason: str, now: float) -> None`.
- Consumes: existing `BasePoseSequenceController.stop_command() -> VelocityCommand` and `build_velocity_message(command, action=...) -> str`.

- [ ] **Step 1: Replace the marker-cancellation contract with stateless stop contracts**

Delete `test_cpp_k_stop_marker_removal_discards_remaining_motion` from
`gear_sonic/tests/test_base_pose_adjustment.py` and add:

```python
def test_runtime_exposes_no_planner_ready_state() -> None:
    assert "planner_ready_file" not in BasePosePlannerConfig.__dataclass_fields__


def test_space_publishes_exact_zero_stop_sequence(tmp_path: Path) -> None:
    messages: list[str] = []
    logs: list[str] = []
    config = BasePosePlannerConfig(
        task="adjust",
        output_root=str(tmp_path),
        final_stop_count=4,
    )
    runtime = BasePosePlannerRuntime(config, publish=messages.append, logger=logs.append)
    assert runtime.handle_key("n", now=1.0) == "started"
    count_before_space = len(messages)

    assert runtime.handle_key(" ", now=1.1) == "cancelled"

    stops = [json.loads(message) for message in messages[count_before_space:]]
    assert len(stops) == config.final_stop_count
    assert all(
        stop["velocity"] == {"vx": 0.0, "vy": 0.0, "wz": 0.0}
        for stop in stops
    )
    assert all(stop["action"] == "stop" for stop in stops)
    assert logs[-1] == "[BasePose] STOP operator_space"
```

Keep the existing late-result and all-motion-direction Space tests.

- [ ] **Step 2: Run a software-only assertion and verify RED**

Run:

```bash
./.venv_inference/bin/python - <<'PY'
import json
from tempfile import TemporaryDirectory
from gear_sonic.scripts.base_pose_planner import (
    BasePosePlannerConfig,
    BasePosePlannerRuntime,
)

assert "planner_ready_file" not in BasePosePlannerConfig.__dataclass_fields__
with TemporaryDirectory() as directory:
    messages = []
    logs = []
    config = BasePosePlannerConfig(
        task="adjust",
        output_root=directory,
        final_stop_count=4,
    )
    runtime = BasePosePlannerRuntime(
        config, publish=messages.append, logger=logs.append
    )
    assert runtime.handle_key("n", now=1.0) == "started"
    before = len(messages)
    assert runtime.handle_key(" ", now=1.1) == "cancelled"
    stops = [json.loads(message) for message in messages[before:]]
    assert len(stops) == 4
    assert all(
        item["velocity"] == {"vx": 0.0, "vy": 0.0, "wz": 0.0}
        for item in stops
    )
    assert all(item["action"] == "stop" for item in stops)
    assert logs[-1] == "[BasePose] STOP operator_space"
PY
```

Expected: `AssertionError` because the dataclass still exposes
`planner_ready_file`; the existing stop messages also use action `hold` and
the log uses `HOLD`.

- [ ] **Step 3: Implement the stateless runtime**

In `gear_sonic/scripts/base_pose_planner.py`:

1. Delete `BasePosePlannerConfig.planner_ready_file`.
2. Delete `self.planner_ready_token` and `_read_planner_ready_token`.
3. Change `_publish_stop` to:

```python
def _publish_stop(self) -> None:
    self.publish(
        build_velocity_message(self.controller.stop_command(), action="stop")
    )
```

4. Rename `cancel_and_hold` to `cancel_and_stop`; keep generation invalidation,
   queue draining, controller cancellation, and the immediate
   `_publish_stop_sequence()` call. Record `cancel_and_stop` and log:

```python
self._record_event("cancel_and_stop", reason=reason, phase=previous_phase)
...
self.logger(f"[BasePose] STOP {reason}")
```

5. Route `X`, `Space`, and shutdown through `cancel_and_stop`.
6. In the `N` branch, delete the ready-token read, `not_ready` return, and token
   assignment. Retain the idle/busy guard, generation queueing, and immediate
   zero command.
7. Delete the entire `publish_due` condition that compares planner-ready
   tokens and calls `cancel_and_hold("cpp_planner_not_ready", ...)`.
8. Change startup help to:

```python
print("[BasePose] N plan | Space cancel-and-stop | X stop-and-exit")
```

- [ ] **Step 4: Verify GREEN locally**

Re-run the exact assertion script from Step 2. Expected: exit 0.

Also run:

```bash
python3 -m py_compile \
  gear_sonic/scripts/base_pose_planner.py \
  gear_sonic/tests/test_base_pose_adjustment.py
```

Expected: exit 0.

- [ ] **Step 5: Inspect and commit only Task 2 hunks**

Run:

```bash
git diff --check -- \
  gear_sonic/scripts/base_pose_planner.py \
  gear_sonic/tests/test_base_pose_adjustment.py
git diff -- \
  gear_sonic/scripts/base_pose_planner.py \
  gear_sonic/tests/test_base_pose_adjustment.py
```

Confirm no prompt, model, reasoning, camera, plan validation, or controller
timing behavior changed. Stage only Task 2 hunks and inspect the cached diff.

```bash
git add -p gear_sonic/scripts/base_pose_planner.py
git add gear_sonic/tests/test_base_pose_adjustment.py
git diff --cached --check
git diff --cached
git commit -m "fix: make base-pose stop stateless"
```

At the `git add -p` prompt, reject the pre-existing reasoning-effort hunk and
accept only the Task 2 runtime hunks.

---

### Task 3: Restore replay-real keyboard interpretation in VLA inference

**Files:**
- Modify: `gear_sonic/scripts/run_vla_inference.py:29-40,131-225,730-830,995-1080`
- Modify: `gear_sonic/tests/test_run_vla_inference_delay.py`

**Interfaces:**
- Consumes: existing nested `send_cpp_control_command(start: bool, planner: bool = False) -> bool`.
- Produces: the `replay_real` `k/i/o/p` keyboard behavior with no marker-file side effects.

- [ ] **Step 1: Replace ready-helper tests with a no-ready-state contract**

In `gear_sonic/tests/test_run_vla_inference_delay.py`:

1. Import `fields` from `dataclasses`.
2. Import the module as
   `from gear_sonic.scripts import run_vla_inference as inference_module`.
3. Retain imports for `SIMULATED_INFERENCE_DELAY_SECONDS` and
   `_inference_worker_loop`.
4. Remove imports and tests for `_execute_cpp_control_toggle`,
   `_planner_message_has_frozen_targets`, and `wait_for_planner_hold_ready`.
5. Remove now-unused `Path`, `TemporaryDirectory`, and
   `build_planner_message` imports.
6. Add:

```python
def test_base_pose_control_exposes_no_hold_ready_state(self):
    config_fields = {field.name for field in fields(inference_module.InferenceConfig)}
    self.assertNotIn("planner_hold_ready_file", config_fields)
    self.assertNotIn("planner_hold_ready_timeout_seconds", config_fields)
    for removed_helper in (
        "_clear_planner_hold",
        "_request_planner_hold",
        "_execute_cpp_control_toggle",
        "wait_for_planner_hold_ready",
        "_planner_message_has_frozen_targets",
        "_forward_frozen_planner_hold",
    ):
        self.assertFalse(hasattr(inference_module, removed_helper), removed_helper)
```

- [ ] **Step 2: Run the new unittest and verify RED**

Run:

```bash
./.venv_inference/bin/python -m unittest \
  gear_sonic.tests.test_run_vla_inference_delay.InferenceWorkerDelayTest.test_base_pose_control_exposes_no_hold_ready_state \
  -v
```

Expected: FAIL because both config fields and all listed helpers currently
exist.

- [ ] **Step 3: Remove the hold-ready implementation**

In `gear_sonic/scripts/run_vla_inference.py`:

1. Remove stdlib imports used only by the ready/frozen helpers:
   `Callable`, `Path`, `Literal`, `json`, and `math`.
2. Remove `HEADER_SIZE` if no remaining code uses it.
3. Delete both `planner_hold_ready_*` config fields.
4. Delete `_clear_planner_hold`, `_request_planner_hold`,
   `_execute_cpp_control_toggle`, and `wait_for_planner_hold_ready`.
5. Delete `_planner_message_has_frozen_targets` and
   `_forward_frozen_planner_hold`.
6. Delete shutdown marker cleanup from `finally`.

Use `rg` after editing to ensure each removed name has zero references.

- [ ] **Step 4: Restore the replay-real keyboard branches exactly**

Remove the BasePose-specific early return from the `i` branch.

Replace the `k` branch with the reference behavior:

```python
elif key == "k":
    if cpp_loop_running:
        current_planner = cpp_mode == "PLANNER"
        print(f"Stopping C++ control loop (from {cpp_mode} mode)...")
        if send_cpp_control_command(start=False, planner=current_planner):
            print("Stopped C++ control loop")
    else:
        print("Starting C++ control loop in PLANNER mode...")
        if send_cpp_control_command(start=True, planner=True):
            print("Started C++ control loop in PLANNER mode")
            print("Press 'i' to send initial pose and switch to POSE mode")
            if pause_loop:
                print("Note: Policy loop is paused - press 'p' to resume")
```

Do not modify Pane 1 publishing, `send_cpp_control_command`, the `o` branch,
the `p` branch, initial-pose interpolation, inference delay, or planner message
forwarding.

- [ ] **Step 5: Verify GREEN and replay-real parity**

Run:

```bash
./.venv_inference/bin/python -m unittest \
  gear_sonic.tests.test_run_vla_inference_delay \
  -v

python3 -m py_compile \
  gear_sonic/scripts/run_vla_inference.py \
  gear_sonic/tests/test_run_vla_inference_delay.py
```

Expected: all remaining delay tests plus the no-ready-state test pass, and
compilation exits 0.

Then compare the reference and current keyboard regions:

```bash
git show replay_real:gear_sonic/scripts/run_vla_inference.py | sed -n '610,700p'
sed -n '720,825p' gear_sonic/scripts/run_vla_inference.py
```

Confirm `i/o/p/k` have the same decisions, command arguments, and local-state
effects. Formatting may differ; BasePose marker checks must be absent.

- [ ] **Step 6: Inspect and commit Task 3**

Run:

```bash
git diff --check -- \
  gear_sonic/scripts/run_vla_inference.py \
  gear_sonic/tests/test_run_vla_inference_delay.py
git diff -- \
  gear_sonic/scripts/run_vla_inference.py \
  gear_sonic/tests/test_run_vla_inference_delay.py
```

Confirm the existing simulated inference-delay behavior is unchanged. Stage
only these two files and inspect the cached diff.

```bash
git add \
  gear_sonic/scripts/run_vla_inference.py \
  gear_sonic/tests/test_run_vla_inference_delay.py
git diff --cached --check
git diff --cached
git commit -m "fix: restore replay-real keyboard control"
```

---

### Task 4: End-to-end software verification and operator handoff

**Files:**
- Verify: `gear_sonic/scripts/launch_inference.py`
- Verify: `gear_sonic/scripts/run_vla_inference.py`
- Verify: `gear_sonic/scripts/base_pose_planner.py`
- Verify: `gear_sonic/scripts/lavira_sonic_relay.py`
- Verify: `gear_sonic/tests/test_lavira_planner.py`
- Verify: `gear_sonic/tests/test_base_pose_adjustment.py`
- Verify: `gear_sonic/tests/test_run_vla_inference_delay.py`

**Interfaces:**
- Consumes: completed Tasks 1-3.
- Produces: evidence that BasePose has no state-latch edge and Space emits base-only zero planner velocity.

- [ ] **Step 1: Run all locally available regression checks**

Run the full stdlib module:

```bash
./.venv_inference/bin/python -m unittest \
  gear_sonic.tests.test_run_vla_inference_delay \
  -v
```

Re-run the Task 1 and Task 2 assertion scripts verbatim. Expected: all exit 0.

- [ ] **Step 2: Verify the stateless relay message fields**

Run:

```bash
./.venv_inference/bin/python - <<'PY'
import json
import struct
import numpy as np
from gear_sonic.scripts.lavira_sonic_relay import PlannerState

message = PlannerState().message(np.zeros(3, dtype=np.float32), 0.05)
header = json.loads(message[7 : 7 + 1280].rstrip(b"\x00"))
payload = message[7 + 1280 :]
names = {field["name"] for field in header["fields"]}
assert struct.unpack_from("<i", payload, 0)[0] == 1
assert not names.intersection(
    {
        "upper_body_position",
        "upper_body_velocity",
        "left_hand_joints",
        "right_hand_joints",
    }
)
PY
```

Expected: exit 0. This does not open a socket.

- [ ] **Step 3: Compile all changed Python files**

Run:

```bash
python3 -m py_compile \
  gear_sonic/scripts/launch_inference.py \
  gear_sonic/scripts/run_vla_inference.py \
  gear_sonic/scripts/base_pose_planner.py \
  gear_sonic/scripts/lavira_sonic_relay.py \
  gear_sonic/tests/test_lavira_planner.py \
  gear_sonic/tests/test_base_pose_adjustment.py \
  gear_sonic/tests/test_run_vla_inference_delay.py
```

Expected: exit 0.

- [ ] **Step 4: Prove the BasePose path has no latch references**

Run:

```bash
rg -n \
  "base_pose_hold_ready|planner_hold_ready|planner_ready_file|cpp_planner_not_ready|cancel_and_hold|freeze-current-upper-body|hold-ready-file" \
  gear_sonic/scripts/launch_inference.py \
  gear_sonic/scripts/run_vla_inference.py \
  gear_sonic/scripts/base_pose_planner.py
```

Expected: no matches. Do not apply this assertion to
`lavira_sonic_relay.py`, because its generic frozen-pose capability is
intentionally preserved for non-BasePose callers.

- [ ] **Step 5: Review final scope and status**

Run:

```bash
git diff --check
git status --short
git log --oneline -6
```

Confirm no live process was restarted or controlled. Report that the already
running tmux session still uses its old loaded code; the operator must stop the
robot safely and restart `launch_inference.py` before testing the new keyboard
and Space behavior.
