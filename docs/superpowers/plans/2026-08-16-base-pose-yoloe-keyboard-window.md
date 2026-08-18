# Base Pose YOLOE Keyboard Planner Window Implementation Plan

> **Superseded behavior:** Task 1's exclusive heartbeat and the related
> `--exclusive` arguments were removed after operator review. The final
> implementation starts with YOLOE selected, uses the unchanged `agent_full`
> key-triggered publisher, applies a manual key for its existing 0.5-second
> command duration, and then automatically falls back to YOLOE. The manual
> port 5566 and relay arbitration remain, but the keyboard now runs visibly in
> `inference` pane 5 instead of a dedicated tmux window. This note overrides
> the historical execution steps below.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated, exclusive WASD/QE keyboard planner tmux window to raw-YOLOE BasePose launches, with safe zero hold and automatic YOLOE recovery after the keyboard exits.

**Architecture:** Extend the existing keyboard planner server with an opt-in continuous heartbeat mode, and extend the existing BasePose relay with an optional higher-priority manual input. The launcher alone decides when to enable and display this path: only `planner_input=base_pose` plus `base_pose_mode=raw_yoloe_servo` creates the window and wires its dedicated port.

**Tech Stack:** Python 3, ZeroMQ/pyzmq, NumPy, tmux, tyro, pytest.

## Global Constraints

- Do not port or depend on `agent_full`'s ControlGateway.
- Preserve the exact existing key map and speeds: `w/s` = `+/-0.3 m/s`, `a/d` = `+/-0.15 m/s`, `q/e` = `+/-0.5 rad/s`, `Space` = stop, `x` = exit.
- A movement key remains active for the existing `0.5 s` command duration, then the live keyboard window holds zero velocity.
- YOLOE resumes only after the keyboard process exits or fails and its manual heartbeat becomes stale.
- Enable the new window only for `planner_input="base_pose"` and `base_pose_mode="raw_yoloe_servo"`.
- Preserve all pre-existing uncommitted work, especially `base_pose_camera_pitch_deg = -38.0` in `launch_inference.py`.
- Do not start robot, camera, tmux, or live ZMQ processes during automated verification.
- Because the shared worktree is dirty and `launch_inference.py` already contains user changes, do not create implementation commits unless the user's hunks can be excluded exactly.

---

### Task 1: Add opt-in exclusive heartbeat behavior to the keyboard planner

**Files:**
- Create: `gear_sonic/tests/test_keyboard_planner_thread_server.py`
- Modify: `gear_sonic/scripts/keyboard_planner_thread_server.py`

**Interfaces:**
- Produces: `ExclusiveKeyboardState(duration: float)` with `handle(action: str, velocity: tuple[float, float, float], now: float) -> None` and `sample(now: float) -> tuple[str, tuple[float, float, float]]`.
- Produces: `KeyboardPlannerConfig.exclusive: bool = False`; tyro exposes it as `--exclusive`.
- Preserves: `build_navila_message`, `key_commands`, all default non-exclusive behavior and existing message schema.

- [ ] **Step 1: Write failing state and configuration tests**

```python
from gear_sonic.scripts.keyboard_planner_thread_server import (
    ExclusiveKeyboardState,
    KeyboardPlannerConfig,
    key_commands,
)


def test_exclusive_keyboard_times_motion_then_holds_zero() -> None:
    config = KeyboardPlannerConfig(duration=0.5)
    state = ExclusiveKeyboardState(duration=config.duration)
    action, velocity = key_commands(config)["w"]

    state.handle(action, velocity, now=10.0)

    assert state.sample(10.49) == ("forward", (0.3, 0.0, 0.0))
    assert state.sample(10.51) == ("stop", (0.0, 0.0, 0.0))


def test_exclusive_mode_is_opt_in() -> None:
    assert KeyboardPlannerConfig().exclusive is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest -q gear_sonic/tests/test_keyboard_planner_thread_server.py`

Expected: FAIL because `ExclusiveKeyboardState` and `exclusive` do not exist.

- [ ] **Step 3: Add the minimal exclusive state**

```python
@dataclass
class ExclusiveKeyboardState:
    duration: float
    action: str = "stop"
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    active_until: float = 0.0

    def handle(
        self,
        action: str,
        velocity: tuple[float, float, float],
        now: float,
    ) -> None:
        self.action = action
        self.velocity = velocity
        self.active_until = now + self.duration

    def sample(self, now: float) -> tuple[str, tuple[float, float, float]]:
        if now <= self.active_until:
            return self.action, self.velocity
        return "stop", (0.0, 0.0, 0.0)
```

Add `exclusive: bool = False` to `KeyboardPlannerConfig`. Validate `hz > 0` only when the exclusive loop uses it.

- [ ] **Step 4: Implement the exclusive input/publish loop**

Keep the current blocking key-triggered loop byte-for-byte in the `exclusive=False` branch. For `exclusive=True`, use `select.select([sys.stdin], [], [], timeout)` while the TTY remains in cbreak mode. At `1 / config.hz` intervals:

```python
state = ExclusiveKeyboardState(config.duration)
action, velocity = state.sample(time.monotonic())
socket.send_string(build_navila_message(action, velocity, config.duration))
```

On `w/s/a/d/q/e/Space`, call `state.handle(...)`. On `x`, leave the loop. Preserve the existing three-message final stop burst in `finally`. The first tick must publish zero, which safely establishes takeover before any movement key.

- [ ] **Step 5: Run focused tests and compilation**

Run: `pytest -q gear_sonic/tests/test_keyboard_planner_thread_server.py`

Expected: PASS.

Run: `python3 -m py_compile gear_sonic/scripts/keyboard_planner_thread_server.py`

Expected: exit 0.

---

### Task 2: Add higher-priority manual input to the direct SONIC relay

**Files:**
- Modify: `gear_sonic/scripts/lavira_sonic_relay.py`
- Modify: `gear_sonic/tests/test_lavira_reasan_contract.py`

**Interfaces:**
- Consumes: unchanged `navila_reasan_velocity_command` JSON from both autonomous and keyboard publishers.
- Produces: `LatestCommand.is_fresh(now: float, timeout: float) -> bool`.
- Produces: `select_velocity(automatic: LatestCommand, manual: LatestCommand | None, *, now: float, timeout: float) -> np.ndarray`.
- Produces: optional CLI argument `--manual-source`; empty by default.

- [ ] **Step 1: Write failing arbitration and CLI tests**

```python
def _latest(velocity: list[float], *, received_at: float) -> LatestCommand:
    latest = LatestCommand()
    latest.update(
        {"velocity": np.asarray(velocity, dtype=np.float32), "duration": 2.0},
        received_at,
    )
    return latest


def test_direct_relay_fresh_manual_command_overrides_automatic() -> None:
    automatic = _latest([0.3, 0.0, 0.0], received_at=10.0)
    manual = _latest([0.0, 0.0, 0.0], received_at=10.4)

    selected = select_velocity(automatic, manual, now=10.5, timeout=0.7)

    assert selected.tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_direct_relay_stale_manual_falls_back_to_automatic() -> None:
    automatic = _latest([0.3, 0.0, 0.0], received_at=10.4)
    manual = _latest([0.0, 0.0, 0.5], received_at=9.0)

    selected = select_velocity(automatic, manual, now=10.5, timeout=0.7)

    assert selected.tolist() == pytest.approx([0.3, 0.0, 0.0])


def test_direct_relay_manual_source_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["lavira_sonic_relay.py"])
    assert parse_direct_relay_args().manual_source == ""
```

- [ ] **Step 2: Run the selected tests to verify they fail**

Run: `pytest -q gear_sonic/tests/test_lavira_reasan_contract.py -k 'manual or stale_command'`

Expected: FAIL because `select_velocity`, `is_fresh`, and `manual_source` do not exist.

- [ ] **Step 3: Implement freshness and arbitration**

```python
def is_fresh(self, now: float, timeout: float) -> bool:
    if self.value is None or self.received_at is None:
        return False
    age = max(0.0, now - self.received_at)
    return age <= min(timeout, float(self.value["duration"]))


def select_velocity(
    automatic: LatestCommand,
    manual: LatestCommand | None,
    *,
    now: float,
    timeout: float,
) -> np.ndarray:
    if manual is not None and manual.is_fresh(now, timeout):
        return manual.velocity(now, timeout)
    return automatic.velocity(now, timeout)
```

Make `LatestCommand.velocity` call `is_fresh` so there is only one freshness definition.

- [ ] **Step 4: Wire the optional manual SUB socket**

Add `parser.add_argument("--manual-source", default="")`. If non-empty, create a second `zmq.SUB` with the same subscribe-all, conflate, linger, decode, and lateral-bounding behavior as the autonomous source. Drain both sockets every loop and call:

```python
velocity = select_velocity(
    latest,
    manual_latest,
    now=now,
    timeout=args.timeout,
)
planner_message = planner.message(velocity, period, frozen_pose=frozen_pose)
```

Log the manual endpoint when enabled and close the optional socket in `finally`. An invalid manual message must only log and leave the last valid manual/automatic states unchanged.

- [ ] **Step 5: Run relay regression tests and compilation**

Run: `pytest -q gear_sonic/tests/test_lavira_reasan_contract.py gear_sonic/tests/test_filter_velocity_relay.py`

Expected: PASS.

Run: `python3 -m py_compile gear_sonic/scripts/lavira_sonic_relay.py`

Expected: exit 0.

---

### Task 3: Launch and document the raw-YOLOE keyboard tmux window

**Files:**
- Modify: `gear_sonic/scripts/launch_inference.py`
- Modify: `gear_sonic/tests/test_lavira_planner.py`
- Modify: `gear_sonic/tests/test_launch_tmux_panes.py`

**Interfaces:**
- Consumes: Task 1's `--exclusive` flag and Task 2's `--manual-source` relay argument.
- Produces: `InferenceLaunchConfig.base_pose_manual_keyboard_port: int = 5566`.
- Produces: `uses_base_pose_manual_keyboard(config: InferenceLaunchConfig) -> bool`.
- Produces: `build_base_pose_manual_keyboard_command(config: InferenceLaunchConfig, repo_root: Path) -> str`.
- Produces: `_launch_base_pose_manual_keyboard_window(config: InferenceLaunchConfig, repo_root: Path) -> None`.

- [ ] **Step 1: Write failing command and mode-gating tests**

```python
def test_raw_yoloe_base_pose_builds_exclusive_manual_keyboard_and_relay() -> None:
    config = InferenceLaunchConfig(
        planner_input="base_pose",
        base_pose_mode="raw_yoloe_servo",
        base_pose_manual_keyboard_port=6002,
    )

    keyboard = build_base_pose_manual_keyboard_command(config, Path("/workspace/sonic"))
    relay = build_reasan_planner_command(config, Path("/workspace/sonic"))

    assert keyboard == (
        "cd /workspace/sonic && source .venv_teleop/bin/activate && "
        "python gear_sonic/scripts/keyboard_planner_thread_server.py "
        "--port 6002 --hz 20 --host localhost --exclusive"
    )
    assert "--manual-source tcp://127.0.0.1:6002" in relay


@pytest.mark.parametrize(
    ("planner_input", "base_pose_mode", "expected"),
    [
        ("base_pose", "raw_yoloe_servo", True),
        ("base_pose", "rgb", False),
        ("lavira", "raw_yoloe_servo", False),
        ("keyboard", "raw_yoloe_servo", False),
    ],
)
def test_manual_keyboard_window_is_raw_yoloe_base_pose_only(
    planner_input: str,
    base_pose_mode: str,
    expected: bool,
) -> None:
    config = InferenceLaunchConfig(
        planner_input=planner_input,  # type: ignore[arg-type]
        base_pose_mode=base_pose_mode,  # type: ignore[arg-type]
    )
    assert uses_base_pose_manual_keyboard(config) is expected
```

- [ ] **Step 2: Run the command tests to verify they fail**

Run: `pytest -q gear_sonic/tests/test_lavira_planner.py -k 'manual_keyboard or raw_yoloe_base_pose_builds'`

Expected: FAIL because the config field and helpers do not exist and the relay lacks the manual source.

- [ ] **Step 3: Implement configuration, predicate, and command builders**

Add the port field beside the other BasePose ports. Use `shlex.quote(str(repo_root))` in the keyboard builder. In `build_reasan_planner_command`, append the manual source only when `uses_base_pose_manual_keyboard(config)` is true; RGB/RGB-D paths must remain byte-for-byte unchanged.

```python
def uses_base_pose_manual_keyboard(config: InferenceLaunchConfig) -> bool:
    return (
        config.planner_input == "base_pose"
        and config.base_pose_mode == "raw_yoloe_servo"
    )
```

- [ ] **Step 4: Write a failing tmux window test**

Mock `subprocess.run` and call `_launch_base_pose_manual_keyboard_window`. Assert the exact three calls:

```python
[
    ["tmux", "new-window", "-t", SESSION_NAME, "-n", "base_keyboard"],
    ["tmux", "send-keys", "-t", f"{SESSION_NAME}:base_keyboard", command, "C-m"],
    ["tmux", "select-window", "-t", f"{SESSION_NAME}:inference"],
]
```

Also assert the helper makes no subprocess calls when `planner_input="keyboard"`.

- [ ] **Step 5: Implement the tmux helper and main-flow call**

Create/start the window after pane 4's direct relay is started and before the optional data-exporter window. Return focus to `inference`. Add summary/help text:

```text
Window 'base_keyboard':
  W/S forward/back | A/D lateral | Q/E yaw | Space stop | X return to YOLOE
```

The navigation help must mention next/previous tmux window whenever this window exists, even without simulation or data export.

- [ ] **Step 6: Run launcher regressions and compilation**

Run: `pytest -q gear_sonic/tests/test_lavira_planner.py gear_sonic/tests/test_launch_tmux_panes.py`

Expected: PASS.

Run: `python3 -m py_compile gear_sonic/scripts/launch_inference.py`

Expected: exit 0.

---

### Task 4: Verify the complete change without touching hardware

**Files:**
- Verify only; no new implementation files.

**Interfaces:**
- Consumes: all interfaces produced by Tasks 1-3.
- Produces: evidence that the raw-YOLOE launch path, keyboard state, relay arbitration, and unchanged legacy paths pass together.

- [ ] **Step 1: Run the focused feature suite**

Run:

```bash
pytest -q \
  gear_sonic/tests/test_keyboard_planner_thread_server.py \
  gear_sonic/tests/test_lavira_reasan_contract.py \
  gear_sonic/tests/test_filter_velocity_relay.py \
  gear_sonic/tests/test_lavira_planner.py \
  gear_sonic/tests/test_launch_tmux_panes.py
```

Expected: all selected tests PASS.

- [ ] **Step 2: Compile all modified Python modules**

Run:

```bash
python3 -m py_compile \
  gear_sonic/scripts/keyboard_planner_thread_server.py \
  gear_sonic/scripts/lavira_sonic_relay.py \
  gear_sonic/scripts/launch_inference.py \
  gear_sonic/tests/test_keyboard_planner_thread_server.py \
  gear_sonic/tests/test_lavira_reasan_contract.py \
  gear_sonic/tests/test_lavira_planner.py \
  gear_sonic/tests/test_launch_tmux_panes.py
```

Expected: exit 0.

- [ ] **Step 3: Inspect scope and whitespace**

Run: `git diff --check`

Expected: no output.

Run: `git diff -- gear_sonic/scripts/keyboard_planner_thread_server.py gear_sonic/scripts/lavira_sonic_relay.py gear_sonic/scripts/launch_inference.py gear_sonic/tests/test_keyboard_planner_thread_server.py gear_sonic/tests/test_lavira_reasan_contract.py gear_sonic/tests/test_lavira_planner.py gear_sonic/tests/test_launch_tmux_panes.py`

Expected: only the approved keyboard heartbeat, relay arbitration, raw-YOLOE tmux window, tests, plus the pre-existing camera-pitch hunk in `launch_inference.py`.

