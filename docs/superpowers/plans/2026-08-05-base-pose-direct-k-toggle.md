# Base-pose Direct K Toggle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the first base-pose `k` press send planner-mode start immediately and the second `k` press send stop immediately, without blocking on robot state or hold-ready.

**Architecture:** Extract the start/stop orchestration from the nested keyboard handler into a small testable helper. The helper sends the control command first, then creates a non-blocking pose-latch request after a successful start; stop sends first and then removes stale marker files. The existing relay and regular planner-message forwarding perform the eventual pose latch after C++ begins publishing state.

**Tech Stack:** Python 3.12, ZeroMQ, `unittest`, existing SONIC packed-message utilities.

## Global Constraints

- The first `k` must send `start=True`, `stop=False`, `planner=True` without waiting for a hold-ready marker or robot-state frame.
- The second `k` must send `start=False`, `stop=True` immediately.
- Pose-latch request creation is allowed only after a successful start send and must not poll or sleep.
- Pose-latch request failure is diagnostic only and must not roll back a successful start.
- Stop must remove both the ready marker and its `.request` file without waiting.
- Do not change planner motion generation, camera handling, Codex inference, or non-`k` keyboard behavior.
- Do not restart or send commands to the currently running robot processes during automated verification.

---

### Task 1: Direct non-blocking C++ control toggle

**Files:**
- Modify: `gear_sonic/scripts/run_vla_inference.py:155-180,645-812`
- Test: `gear_sonic/tests/test_run_vla_inference_delay.py`

**Interfaces:**
- Produces: `_request_planner_hold(path: str) -> bool`, which replaces stale marker state with a new `.request` token without polling.
- Produces: `_clear_planner_hold(path: str) -> None`, which removes the ready and request files.
- Produces: `_execute_cpp_control_toggle(*, cpp_loop_running: bool, cpp_mode: str, planner_hold_ready_file: str, send_control_command: Callable[[bool, bool], bool]) -> Literal["started", "stopped", "failed"]`.
- Consumes: the existing nested `send_cpp_control_command(start: bool, planner: bool) -> bool`; it remains responsible for packing/sending the ZMQ command and updating the nested local C++ state.

- [ ] **Step 1: Write the failing direct-toggle test**

Extend `gear_sonic/tests/test_run_vla_inference_delay.py` to import `_execute_cpp_control_toggle` and add:

```python
    def test_base_pose_k_starts_before_request_and_second_k_stops(self):
        with TemporaryDirectory() as directory:
            marker = Path(directory) / "hold.ready"
            request = Path(f"{marker}.request")
            events = []

            def send_control(start, planner):
                events.append((start, planner, request.exists()))
                return True

            self.assertEqual(
                _execute_cpp_control_toggle(
                    cpp_loop_running=False,
                    cpp_mode="OFF",
                    planner_hold_ready_file=str(marker),
                    send_control_command=send_control,
                ),
                "started",
            )
            self.assertEqual(events, [(True, True, False)])
            self.assertTrue(request.is_file())

            marker.write_text("ready\n", encoding="utf-8")
            self.assertEqual(
                _execute_cpp_control_toggle(
                    cpp_loop_running=True,
                    cpp_mode="PLANNER",
                    planner_hold_ready_file=str(marker),
                    send_control_command=send_control,
                ),
                "stopped",
            )
            self.assertEqual(events[-1], (False, True, True))
            self.assertFalse(marker.exists())
            self.assertFalse(request.exists())
```

This assertion proves ordering with real filesystem state: the start callback observes no request, and the request appears only after the send returns.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
./.venv_inference/bin/python -m unittest \
  gear_sonic.tests.test_run_vla_inference_delay.InferenceWorkerDelayTest.test_base_pose_k_starts_before_request_and_second_k_stops \
  -v
```

Expected: FAIL during import because `_execute_cpp_control_toggle` does not exist yet.

- [ ] **Step 3: Implement non-blocking marker helpers**

In `gear_sonic/scripts/run_vla_inference.py`, import `Callable` from
`collections.abc` and `Literal` from `typing`, then add the marker operations
near `wait_for_planner_hold_ready`:

```python
def _clear_planner_hold(path: str) -> None:
    if not path:
        return
    marker = Path(path).expanduser()
    marker.unlink(missing_ok=True)
    Path(f"{marker}.request").unlink(missing_ok=True)


def _request_planner_hold(path: str) -> bool:
    if not path:
        return True
    marker = Path(path).expanduser()
    request = Path(f"{marker}.request")
    try:
        _clear_planner_hold(path)
        request.parent.mkdir(parents=True, exist_ok=True)
        request.write_text(f"{time.time_ns()}\n", encoding="utf-8")
    except OSError:
        return False
    return True
```

Refactor `wait_for_planner_hold_ready` to call `_request_planner_hold(path)` for its setup so the existing helper keeps identical semantics for any external caller.

- [ ] **Step 4: Implement the direct toggle helper**

Add the orchestration helper at module scope:

```python
def _execute_cpp_control_toggle(
    *,
    cpp_loop_running: bool,
    cpp_mode: str,
    planner_hold_ready_file: str,
    send_control_command: Callable[[bool, bool], bool],
) -> Literal["started", "stopped", "failed"]:
    if cpp_loop_running:
        sent = send_control_command(False, cpp_mode == "PLANNER")
        if not sent:
            return "failed"
        _clear_planner_hold(planner_hold_ready_file)
        return "stopped"

    if not send_control_command(True, True):
        return "failed"
    if planner_hold_ready_file and not _request_planner_hold(
        planner_hold_ready_file
    ):
        print("Warning: failed to request post-start planner hold state")
    return "started"
```

The only blocking-capable functions, `wait_for_planner_hold_ready` and `_forward_frozen_planner_hold`, must not be called by this helper.

- [ ] **Step 5: Route the `k` branch through the helper**

Replace the existing pre-start hold-ready block inside `check_keyboard_input` with one call to `_execute_cpp_control_toggle`. Preserve the current user-facing start/stop logs, using the returned `"started"`, `"stopped"`, or `"failed"` status. Do not call `wait_for_planner_hold_ready` or `_forward_frozen_planner_hold` from the `k` branch.

The resulting sequence must be:

```python
result = _execute_cpp_control_toggle(
    cpp_loop_running=cpp_loop_running,
    cpp_mode=cpp_mode,
    planner_hold_ready_file=config.planner_hold_ready_file,
    send_control_command=send_cpp_control_command,
)
```

The nested `send_cpp_control_command` continues updating `cpp_loop_running` and `cpp_mode` only after its ZMQ send succeeds.

- [ ] **Step 6: Verify GREEN and run the focused regression module**

Run:

```bash
./.venv_inference/bin/python -m unittest \
  gear_sonic.tests.test_run_vla_inference_delay.InferenceWorkerDelayTest.test_base_pose_k_starts_before_request_and_second_k_stops \
  -v

./.venv_inference/bin/python -m unittest \
  gear_sonic.tests.test_run_vla_inference_delay \
  -v

python3 -m py_compile \
  gear_sonic/scripts/run_vla_inference.py \
  gear_sonic/tests/test_run_vla_inference_delay.py
```

Expected: the new test passes, all tests in the existing unittest module pass, and compilation exits zero.

- [ ] **Step 7: Review the production diff and commit only direct-toggle files**

Run:

```bash
git diff --check -- \
  gear_sonic/scripts/run_vla_inference.py \
  gear_sonic/tests/test_run_vla_inference_delay.py

git diff -- \
  gear_sonic/scripts/run_vla_inference.py \
  gear_sonic/tests/test_run_vla_inference_delay.py
```

Confirm the diff contains no camera, prompt, Codex, planner-motion, or C++ changes. Then commit only these two files:

```bash
git add \
  gear_sonic/scripts/run_vla_inference.py \
  gear_sonic/tests/test_run_vla_inference_delay.py
git commit -m "fix: make base-pose k toggle non-blocking"
```

Do not include the pre-existing modified or untracked files in this commit.
