# FAST-LIO Process-Group Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure every FAST-LIO restart and shutdown waits for the complete ROS launch process group to exit, preventing orphaned `fastlio_mapping` processes from accumulating.

**Architecture:** Keep the existing dedicated POSIX session created by `start_new_session=True`, and use its leader PID as the process-group ID. Separate process-group existence checks from `Popen` parent reaping, then make cleanup escalate SIGINT → SIGTERM → SIGKILL based on group liveness rather than the launch parent's status.

**Tech Stack:** Python 3.10, POSIX process groups, pytest

## Global Constraints

- Modify only FAST-LIO supervisor process ownership and cleanup; do not change localization, LiDAR, IMU, time-sync, or recovery-threshold configuration.
- Signal only the dedicated process group created by this supervisor; do not use `pkill`, process-name matching, or system-wide cleanup.
- Preserve the existing bounded escalation timeouts: SIGINT 5.0 seconds, SIGTERM 2.0 seconds, and SIGKILL 1.0 second.
- Do not launch a replacement FAST-LIO process until the previous process group is confirmed absent.
- Treat a process group that survives SIGKILL as a fatal cleanup error.

---

### Task 1: Make FAST-LIO cleanup process-group aware

**Files:**
- Modify: `gear_sonic/scripts/run_fastlio_supervisor.py:68-94`
- Test: `gear_sonic/tests/test_fastlio_supervisor.py`

**Interfaces:**
- Consumes: `subprocess.Popen.pid` as the process-group ID established by `_start_fastlio(..., start_new_session=True)`.
- Produces: `_process_group_exists(process_group_id: int) -> bool`.
- Produces: `_wait_for_process_group_exit(process: subprocess.Popen, process_group_id: int, timeout_s: float) -> bool`.
- Produces: `_signal_process_group(process_group_id: int, signum: signal.Signals) -> None`.
- Preserves: `_stop_fastlio(process: subprocess.Popen) -> None`, now returning only after the whole process group disappears or raising `RuntimeError` after failed SIGKILL cleanup.

- [ ] **Step 1: Add failing tests for the observed parent-first exit and cleanup boundaries**

Add these imports and tests to `gear_sonic/tests/test_fastlio_supervisor.py`:

```python
import signal

import pytest


class _ExitedLaunchProcess:
    pid = 4242

    def __init__(self) -> None:
        self.poll_count = 0

    def poll(self) -> int:
        self.poll_count += 1
        return 0


def test_group_wait_does_not_treat_exited_launch_parent_as_cleanup_complete(
    monkeypatch,
) -> None:
    process = _ExitedLaunchProcess()
    group_states = iter([True, False])
    monkeypatch.setattr(
        run_fastlio_supervisor,
        "_process_group_exists",
        lambda _process_group_id: next(group_states),
    )
    monkeypatch.setattr(run_fastlio_supervisor.time, "sleep", lambda _seconds: None)

    assert run_fastlio_supervisor._wait_for_process_group_exit(
        process,
        process.pid,
        timeout_s=1.0,
    )
    assert process.poll_count == 2


def test_stop_escalates_when_parent_exits_but_process_group_survives_sigint(
    monkeypatch,
) -> None:
    process = _ExitedLaunchProcess()
    signals: list[signal.Signals] = []
    wait_results = iter([False, True])
    monkeypatch.setattr(
        run_fastlio_supervisor,
        "_process_group_exists",
        lambda _process_group_id: True,
    )
    monkeypatch.setattr(
        run_fastlio_supervisor,
        "_signal_process_group",
        lambda _process_group_id, signum: signals.append(signum),
    )
    monkeypatch.setattr(
        run_fastlio_supervisor,
        "_wait_for_process_group_exit",
        lambda _process, _process_group_id, _timeout_s: next(wait_results),
    )

    run_fastlio_supervisor._stop_fastlio(process)

    assert signals == [signal.SIGINT, signal.SIGTERM]


def test_stop_is_noop_when_process_group_is_already_gone(monkeypatch) -> None:
    process = _ExitedLaunchProcess()
    monkeypatch.setattr(
        run_fastlio_supervisor,
        "_process_group_exists",
        lambda _process_group_id: False,
    )
    monkeypatch.setattr(
        run_fastlio_supervisor,
        "_signal_process_group",
        lambda *_args: pytest.fail("dead process group must not be signalled"),
    )

    run_fastlio_supervisor._stop_fastlio(process)

    assert process.poll_count == 1


def test_stop_raises_when_process_group_survives_sigkill(monkeypatch) -> None:
    process = _ExitedLaunchProcess()
    signals: list[signal.Signals] = []
    monkeypatch.setattr(
        run_fastlio_supervisor,
        "_process_group_exists",
        lambda _process_group_id: True,
    )
    monkeypatch.setattr(
        run_fastlio_supervisor,
        "_signal_process_group",
        lambda _process_group_id, signum: signals.append(signum),
    )
    monkeypatch.setattr(
        run_fastlio_supervisor,
        "_wait_for_process_group_exit",
        lambda _process, _process_group_id, _timeout_s: False,
    )

    with pytest.raises(RuntimeError, match="survived SIGKILL"):
        run_fastlio_supervisor._stop_fastlio(process)

    assert signals == [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]
```

The first test catches any implementation that returns as soon as `process.poll()` reports the launch parent exited. The escalation test catches the former `_signal_process_group`/`process.wait()` behavior that could leave a child alive. The no-op and SIGKILL tests protect idempotency and the no-stacked-restart safety boundary.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
pytest -q \
  gear_sonic/tests/test_fastlio_supervisor.py::test_group_wait_does_not_treat_exited_launch_parent_as_cleanup_complete \
  gear_sonic/tests/test_fastlio_supervisor.py::test_stop_escalates_when_parent_exits_but_process_group_survives_sigint \
  gear_sonic/tests/test_fastlio_supervisor.py::test_stop_is_noop_when_process_group_is_already_gone \
  gear_sonic/tests/test_fastlio_supervisor.py::test_stop_raises_when_process_group_survives_sigkill
```

Expected: FAIL because `_wait_for_process_group_exit` and `_process_group_exists` do not exist and `_signal_process_group` still accepts a `Popen` object instead of an explicit process-group ID.

- [ ] **Step 3: Implement process-group liveness, bounded waiting, and escalation**

Replace the existing `_signal_process_group` and `_stop_fastlio` implementation in `gear_sonic/scripts/run_fastlio_supervisor.py` with:

```python
def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_group_exit(
    process: subprocess.Popen,
    process_group_id: int,
    timeout_s: float,
) -> bool:
    deadline = time.monotonic() + timeout_s
    while True:
        # Reap the launch parent when it exits, but keep the group as the
        # ownership/liveness boundary because mapping descendants may remain.
        process.poll()
        if not _process_group_exists(process_group_id):
            return True
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0.0:
            return False
        time.sleep(min(0.05, remaining_s))


def _signal_process_group(
    process_group_id: int,
    signum: signal.Signals,
) -> None:
    os.killpg(process_group_id, signum)


def _stop_fastlio(process: subprocess.Popen) -> None:
    process_group_id = process.pid
    if not _process_group_exists(process_group_id):
        process.poll()
        return

    for signum, timeout_s in (
        (signal.SIGINT, 5.0),
        (signal.SIGTERM, 2.0),
        (signal.SIGKILL, 1.0),
    ):
        try:
            _signal_process_group(process_group_id, signum)
        except ProcessLookupError:
            process.poll()
            return
        if _wait_for_process_group_exit(
            process,
            process_group_id,
            timeout_s,
        ):
            return

    raise RuntimeError(
        f"FAST-LIO process group {process_group_id} survived SIGKILL"
    )
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```bash
pytest -q gear_sonic/tests/test_fastlio_supervisor.py
```

Expected: all supervisor tests PASS.

- [ ] **Step 5: Run a real POSIX process-group integration check**

Run this bounded check using the repository's Python interpreter:

```bash
python -c 'import subprocess,sys; from gear_sonic.scripts.run_fastlio_supervisor import _process_group_exists,_stop_fastlio; p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"],start_new_session=True); pgid=p.pid; _stop_fastlio(p); assert p.poll() is not None; assert not _process_group_exists(pgid); print("process-group cleanup: PASS")'
```

Expected: `process-group cleanup: PASS` and exit status 0.

- [ ] **Step 6: Run the regression suite for supervisor and recovery behavior**

Run:

```bash
pytest -q \
  gear_sonic/tests/test_fastlio_supervisor.py \
  gear_sonic/tests/test_slam_recovery.py \
  gear_sonic/tests/test_launch_tmux_panes.py
python -m py_compile gear_sonic/scripts/run_fastlio_supervisor.py
```

Expected: all tests PASS and `py_compile` exits 0.

- [ ] **Step 7: Commit the leak fix**

```bash
git add \
  gear_sonic/scripts/run_fastlio_supervisor.py \
  gear_sonic/tests/test_fastlio_supervisor.py
git commit -m "fix(fastlio): wait for process group cleanup"
```
