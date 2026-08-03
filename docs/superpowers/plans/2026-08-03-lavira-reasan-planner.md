# LaViRA REASAN Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tmux-triggered, single-cycle LaViRA/AgentNav command source to `replay_real` that sends rotate-then-translate velocity commands through the existing REASAN safety filter.

**Architecture:** Add RGB-D support to the composed camera, port the AgentNav camera/Codex and pure geometry layers, and add an independent `lavira_planner.py` state machine in pane 3. LaViRA publishes the existing `navila_reasan_velocity_command` format on port 5558; current REASAN and VLA relay code remain unchanged.

**Tech Stack:** Python 3, NumPy, OpenCV, pyrealsense2, ZeroMQ/pyzmq, tyro, Codex CLI, pytest, tmux.

## Global Constraints

- Target branch is `replay_real`; preserve unrelated user changes.
- Do not modify `gear_sonic/scripts/reasan_planner.py`.
- Do not modify `gear_sonic/scripts/keyboard_planner_thread_server.py`.
- Do not modify `gear_sonic/scripts/run_vla_inference.py`.
- Do not modify `tools/mid360_reasan_open3d.py`.
- Do not add `navila_planner.py`, Uni-LaViRA sidecars, direct SONIC publication, continuous replanning, or VLA handoff.
- Keep the existing REASAN `TURN-BYPASS` behavior for pure yaw.
- Keep `planner_input="keyboard"` as the launch default.
- Copy AgentNav automatic-motion defaults exactly: rotation 0.4 rad/s, forward 0.3 m/s, target standoff 0.0 m, maximum direct travel 8.0 m, minimum rotation 2 degrees, 20 Hz publication, 0.5 s transition pause, maximum speed 0.5 m/s, maximum duration 30 s, and maximum relative yaw pi.
- Rename upstream `safe_distance` to `target_standoff_distance` without changing its 0.0 m default or formula.
- Every failure, cancellation, and shutdown path must publish repeated zero velocity and must not publish a later stale worker result.
- Implement with tests first and commit after every task.

---

## File map

- `gear_sonic/camera/sensor_server.py`: wire schema, RGB/depth serialization, calibration metadata.
- `gear_sonic/camera/drivers/realsense.py`: aligned depth acquisition and camera calibration.
- `gear_sonic/camera/composed_camera.py`: chest-only depth selection and multi-camera metadata merge.
- `start_camera_server.zsh`: enable chest depth at launch.
- `gear_sonic/utils/inference/object_nav_geometry.py`: pure bbox/depth-to-motion calculation.
- `gear_sonic/utils/inference/object_nav.py`: one RGB-D capture/Codex policy cycle.
- `gear_sonic/scripts/lavira_planner.py`: keyboard, worker, state machine, timing, and REASAN JSON publisher.
- `gear_sonic/scripts/launch_inference.py`: select keyboard or LaViRA in tmux pane 3.
- `gear_sonic/tests/test_camera_rgbd_protocol.py`: RGB-D protocol and metadata compatibility.
- `gear_sonic/tests/test_object_nav_geometry.py`: pure geometry and command-limit coverage.
- `gear_sonic/tests/test_object_nav_codex.py`: policy parsing and single-cycle failure behavior.
- `gear_sonic/tests/test_lavira_planner.py`: command adapter, state machine, cancellation, and stop guarantees.

---

### Task 1: RGB-D wire protocol

**Files:**
- Modify: `gear_sonic/camera/sensor_server.py`
- Create: `gear_sonic/tests/test_camera_rgbd_protocol.py`

**Interfaces:**
- Produces: `ImageMessageSchema(timestamps, images, camera_info={})`.
- Produces: depth keys ending in `_depth` as lossless two-dimensional `np.uint16` PNG data.
- Preserves: existing RGB string, JPEG bytes, ndarray, and msgpack ndarray decoding.

- [ ] **Step 1: Add failing protocol tests**

Add these tests, with the repository imports and fixtures needed to construct
the schema:

```python
def test_rgbd_schema_round_trip_preserves_uint16_and_camera_info():
    depth = np.array([[0, 1000], [2345, 65535]], dtype=np.uint16)
    schema = ImageMessageSchema(
        timestamps={"chest_view": 1.0, "chest_view_depth": 1.0},
        images={"chest_view": np.zeros((2, 2, 3), np.uint8), "chest_view_depth": depth},
        camera_info={"chest_view": {"fx": 500.0, "fy": 501.0, "cx": 1.0, "cy": 1.0,
                                            "width": 2, "height": 2, "depth_scale_m": 0.001,
                                            "depth_aligned_to": "chest_view"}},
    )
    wire = schema.serialize()
    decoded = ImageMessageSchema.deserialize(wire)
    np.testing.assert_array_equal(decoded.images["chest_view_depth"], depth)
    assert decoded.images["chest_view_depth"].dtype == np.uint16
    assert decoded.camera_info == schema.camera_info
    assert wire["schema_version"] == 2

def test_depth_encoder_rejects_wrong_dtype_and_shape():
    with pytest.raises(ValueError, match="2D uint16"):
        ImageUtils.encode_depth_image(np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="2D uint16"):
        ImageUtils.encode_depth_image(np.zeros((2, 2, 1), dtype=np.uint16))

def test_legacy_rgb_message_without_camera_info_still_decodes():
    decoded = ImageMessageSchema.deserialize({"timestamps": {}, "images": {}})
    assert decoded.camera_info == {}
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run: `pytest -q gear_sonic/tests/test_camera_rgbd_protocol.py`

Expected: FAIL because `camera_info`/schema v2 are absent and depth validation is incomplete.

- [ ] **Step 3: Implement the minimal schema change**

Add `camera_info: dict[str, Any] = field(default_factory=dict)`. Emit
`schema_version=2`; encode keys ending in `_depth` with
`ImageUtils.encode_depth_image`; decode them using `cv2.IMREAD_UNCHANGED` or
`decode_depth_image`; include `camera_info` in `deserialize()` and `asdict()`.
Require depth input to be exactly a 2-D `np.uint16` array and raise
`ValueError("depth image must be a 2D uint16 array")` otherwise.

- [ ] **Step 4: Run focused and nearby camera tests**

Run: `pytest -q gear_sonic/tests/test_camera_rgbd_protocol.py gear_sonic/tests -k 'camera and not hardware'`

Expected: new tests PASS; no existing software-only camera regression.

- [ ] **Step 5: Commit**

```bash
git add gear_sonic/camera/sensor_server.py gear_sonic/tests/test_camera_rgbd_protocol.py
git commit -m "feat: add composed camera RGB-D protocol"
```

### Task 2: Aligned chest depth and calibration

**Files:**
- Modify: `gear_sonic/camera/drivers/realsense.py`
- Modify: `gear_sonic/camera/composed_camera.py`
- Modify: `start_camera_server.zsh`
- Test: `gear_sonic/tests/test_camera_rgbd_protocol.py`

**Interfaces:**
- Consumes: Task 1 `ImageMessageSchema.camera_info`.
- Produces: `camera_info["chest_view"]` with `fx`, `fy`, `cx`, `cy`, `width`, `height`, `depth_scale_m`, and `depth_aligned_to`.
- Produces: `images["chest_view_depth"]` aligned to `images["chest_view"]`.

- [ ] **Step 1: Add failing mocked RealSense and composition tests**

Test that enabling composed-camera depth affects only `chest_view`, that
`rs.align(rs.stream.color).process(frames)` is used, and that composition
merges `camera_info`. Assert this exact calibration shape:

```python
assert result["camera_info"]["chest_view"] == {
    "fx": 500.0, "fy": 501.0, "cx": 320.0, "cy": 240.0,
    "width": 640, "height": 480,
    "depth_scale_m": 0.001,
    "depth_aligned_to": "chest_view",
}
```

- [ ] **Step 2: Run the new cases and confirm failure**

Run: `pytest -q gear_sonic/tests/test_camera_rgbd_protocol.py -k 'realsense or composed'`

Expected: FAIL because alignment/calibration and composed metadata merge are absent.

- [ ] **Step 3: Implement aligned depth and chest-only enablement**

Capture the pipeline profile returned by `pipeline.start()`, create the color
aligner only when depth is enabled, read the device depth scale, align frames
before extraction, and publish calibration from the aligned color profile.
In `ComposedCameraSensor`, set `enable_depth` only when both the global flag is
true and `mount_position == "chest_view"`; merge `camera_info` beside images
and timestamps.

- [ ] **Step 4: Enable depth in the checked-in launch script**

Add exactly `--realsense-enable-depth` to `start_camera_server.zsh`, leaving
camera IDs and port unchanged.

- [ ] **Step 5: Run tests and shell syntax validation**

Run: `pytest -q gear_sonic/tests/test_camera_rgbd_protocol.py`

Run: `zsh -n start_camera_server.zsh`

Expected: PASS for both commands.

- [ ] **Step 6: Commit**

```bash
git add gear_sonic/camera/drivers/realsense.py gear_sonic/camera/composed_camera.py start_camera_server.zsh gear_sonic/tests/test_camera_rgbd_protocol.py
git commit -m "feat: publish aligned chest depth calibration"
```

### Task 3: Pure AgentNav geometry

**Files:**
- Create: `gear_sonic/utils/inference/object_nav_geometry.py`
- Create: `gear_sonic/tests/test_object_nav_geometry.py`

**Interfaces:**
- Produces: `measure_object_nav_target(policy, depth_mm, fx, cx) -> dict[str, Any]`.
- Produces: `build_object_nav_commands_from_frames(policy, frames, *, rotation_speed=0.4, forward_speed=0.3, target_standoff_distance=0.0, max_direct_travel=8.0) -> tuple[dict[str, Any], dict[str, Any]]`.
- Output command shape: `{"commands": [{"vx", "vy", "wz", "duration"}, {"vx", "vy", "wz", "duration"}]}`.

- [ ] **Step 1: Port focused failing tests from agent-nav with renamed parameter**

Cover centered, left, and right targets; two-degree rotation suppression; five
frames with at least three valid measurements; invalid bbox/depth/intrinsics;
0.0 target standoff; and rejection above 8.0 m. Include:

```python
commands, geometry = build_object_nav_commands_from_frames(policy, frames)
assert commands["commands"][0] == {"vx": 0.0, "vy": 0.0, "wz": 0.0, "duration": 0.0}
assert commands["commands"][1]["vx"] == 0.3
assert commands["commands"][1]["duration"] == pytest.approx(geometry["travel"] / 0.3, abs=1e-3)
```

- [ ] **Step 2: Run and confirm import failure**

Run: `pytest -q gear_sonic/tests/test_object_nav_geometry.py`

Expected: FAIL because `object_nav_geometry.py` does not exist.

- [ ] **Step 3: Port the pure implementation**

Port the implementation from
`origin/agent-nav:gear_sonic/utils/inference/object_nav_geometry.py`, changing
only `SAFE_DISTANCE`/`safe_distance` to
`TARGET_STANDOFF_DISTANCE`/`target_standoff_distance`. Preserve all numeric
defaults, five-frame policy, rounding behavior, and error checks.

- [ ] **Step 4: Run geometry tests**

Run: `pytest -q gear_sonic/tests/test_object_nav_geometry.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gear_sonic/utils/inference/object_nav_geometry.py gear_sonic/tests/test_object_nav_geometry.py
git commit -m "feat: add AgentNav RGB-D geometry"
```

### Task 4: Single-cycle RGB-D and Codex runner

**Files:**
- Create: `gear_sonic/utils/inference/object_nav.py`
- Create: `gear_sonic/tests/test_object_nav_codex.py`

**Interfaces:**
- Consumes: Task 3 `build_object_nav_commands_from_frames`.
- Produces: `ObjectNavConfig` with `target_standoff_distance`.
- Produces: immutable `ObjectNavResult(outcome, policy, commands, geometry, output_dir, error=None)`.
- Produces: `ObjectNavRunner(config, camera=None, codex=None).run_once() -> ObjectNavResult` and `.close()`.
- Outcomes: `NAVIGATE`, `STOP`, `FAILED`, or `REJECTED`.

- [ ] **Step 1: Add failing policy and runner tests**

Port the relevant tests from `origin/agent-nav` for strict policy keys, finite
confidence, bbox ordering, target types, rotation directions, low confidence,
Codex failure, malformed camera data, and successful five-frame navigation.
Use fake camera/Codex objects; never invoke the real CLI or hardware.

```python
result = ObjectNavRunner(config, camera=fake_camera, codex=fake_codex).run_once()
assert result.outcome == "NAVIGATE"
assert len(result.commands["commands"]) == 2
```

- [ ] **Step 2: Run and confirm import failure**

Run: `pytest -q gear_sonic/tests/test_object_nav_codex.py`

Expected: FAIL because `object_nav.py` does not exist.

- [ ] **Step 3: Port and trim the runner**

Port `RGBDSnapshot`, `ObjectNavConfig`, `ObjectNavResult`,
`ComposedRGBDCamera`, `CodexBBoxClient`, diagnostic writers, and
`ObjectNavRunner` from `origin/agent-nav`. Rename the standoff parameter and
its call site. Remove `SonicPlannerRequestError`, `send_object_nav_commands`,
and every direct planner/Sonic dependency. Preserve camera timeout 3000 ms,
Codex timeout 180 s, confidence 0.6, raw-depth diagnostics, strict policy
validation, and fail-closed results.

- [ ] **Step 4: Run runner, geometry, and camera tests**

Run: `pytest -q gear_sonic/tests/test_object_nav_codex.py gear_sonic/tests/test_object_nav_geometry.py gear_sonic/tests/test_camera_rgbd_protocol.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gear_sonic/utils/inference/object_nav.py gear_sonic/tests/test_object_nav_codex.py
git commit -m "feat: add single-cycle AgentNav runner"
```

### Task 5: LaViRA-to-REASAN state machine

**Files:**
- Create: `gear_sonic/scripts/lavira_planner.py`
- Create: `gear_sonic/tests/test_lavira_planner.py`

**Interfaces:**
- Consumes: Task 4 `ObjectNavRunner` and `ObjectNavResult`.
- Consumes: existing `build_navila_message(action, velocity, duration_s, raw_text)`.
- Produces: `LaviraPlannerConfig` CLI dataclass.
- Produces: immutable `VelocityCommand(vx: float, vy: float, wz: float, duration: float)`.
- Produces: `build_reasan_velocity_message(command: VelocityCommand, *, action: str) -> str`.
- Produces: `validate_object_nav_batch(payload, *, max_speed=0.5, max_duration=30.0, max_abs_yaw=math.pi, min_positive_duration=0.05) -> ObjectNavBatch`.
- Produces: `LaviraPlannerController.start(result, now)`, `.cancel(reason)`, and `.step(now) -> VelocityCommand` with phases `idle`, `rotating`, `transition_pause`, `translating`, and `final_stop`.

- [ ] **Step 1: Add failing validation and state-machine tests**

Test exact two-command validation, pure rotation/translation, finite values,
20 Hz minimum positive duration, maximum speed/duration/yaw, zero-duration
phase skipping, 0.5 s transition pause, and final stop.

```python
controller.start(navigate_result, now=10.0)
assert controller.step(10.0).velocity == (0.0, 0.0, 0.4)
assert controller.step(rotation_deadline).velocity == (0.0, 0.0, 0.0)
assert controller.step(rotation_deadline + 0.5).velocity == (0.3, 0.0, 0.0)
```

- [ ] **Step 2: Add failing worker-generation and keyboard tests**

Test `n` starts only while idle, repeated `n` is rejected, Space/X/manual keys
increment the generation and publish stop, late results are discarded, and
manual mappings reuse current keyboard defaults.

- [ ] **Step 3: Run and confirm import failure**

Run: `pytest -q gear_sonic/tests/test_lavira_planner.py`

Expected: FAIL because `lavira_planner.py` does not exist.

- [ ] **Step 4: Implement validation, controller, and JSON adapter**

Port only the validation and nonblocking executor semantics from
`origin/agent-nav:gear_sonic/utils/inference/uni_lavira_planner.py`. Do not
port direct SONIC modes, heading integration, JSON REP bridge, or sidecars.
Map each active `VelocityCommand` through `build_navila_message`, using action
names `turn_left`, `turn_right`, `move_forward`, and `stop`.

- [ ] **Step 5: Implement responsive main loop**

Use a one-slot request queue, one-slot result queue, daemon inference worker,
generation counter, nonblocking terminal reads, monotonic deadlines, and a
ZMQ PUB bound to configured host/port 5558. Publish the active command every
`1 / planner_hz`; publish zero throughout the 0.5 s transition pause and at
least three times on final stop, cancellation, exception, signal, and exit.

- [ ] **Step 6: Run LaViRA tests**

Run: `pytest -q gear_sonic/tests/test_lavira_planner.py`

Expected: PASS without camera, Codex, terminal, or ZMQ network access.

- [ ] **Step 7: Run all new feature tests together**

Run: `pytest -q gear_sonic/tests/test_camera_rgbd_protocol.py gear_sonic/tests/test_object_nav_geometry.py gear_sonic/tests/test_object_nav_codex.py gear_sonic/tests/test_lavira_planner.py`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add gear_sonic/scripts/lavira_planner.py gear_sonic/tests/test_lavira_planner.py
git commit -m "feat: publish LaViRA commands through REASAN"
```

### Task 6: tmux launch selection

**Files:**
- Modify: `gear_sonic/scripts/launch_inference.py`
- Test: `gear_sonic/tests/test_lavira_planner.py`

**Interfaces:**
- Consumes: Task 5 `lavira_planner.py` CLI.
- Produces: `InferenceLaunchConfig.planner_input: Literal["keyboard", "lavira"] = "keyboard"`.
- Preserves: existing keyboard command string byte-for-byte when the default is selected.

- [ ] **Step 1: Add failing launch-command tests**

Extract a pure `build_planner_input_command(config, repo_root) -> str` helper
and test both selections. Assert keyboard uses `.venv_teleop` and the existing
script; assert LaViRA uses `.venv_inference`, `lavira_planner.py`, quoted
mission/target values, camera endpoint, output port 5558, and all approved
AgentNav values.

- [ ] **Step 2: Run the focused launch tests and confirm failure**

Run: `pytest -q gear_sonic/tests/test_lavira_planner.py -k launch`

Expected: FAIL because the selection/helper does not exist.

- [ ] **Step 3: Implement launch configuration and pane selection**

Add `Literal` and `shlex` imports, LaViRA fields matching
`LaviraPlannerConfig`, prerequisite validation for nonempty mission and global
target when selected, the pure command builder, and pane 3 label/help updates.
Do not alter pane indices, ports 5555/5558/5562/5563, the REASAN command, the
MID-360 command, or the VLA command.

- [ ] **Step 4: Run launch and feature tests**

Run: `pytest -q gear_sonic/tests/test_lavira_planner.py gear_sonic/tests/test_object_nav_codex.py`

Run: `python -m py_compile gear_sonic/scripts/launch_inference.py gear_sonic/scripts/lavira_planner.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gear_sonic/scripts/launch_inference.py gear_sonic/tests/test_lavira_planner.py
git commit -m "feat: launch LaViRA command source in tmux"
```

### Task 7: Regression and local integration verification

**Files:**
- Modify only if a verified defect is found in files already listed above.
- Test: all tests introduced by Tasks 1-6 and existing REASAN/keyboard tests.
- Create: `gear_sonic/tests/test_lavira_reasan_contract.py`

**Interfaces:**
- Verifies: default keyboard path unchanged.
- Verifies: LaViRA publishes on 5558 and REASAN publishes on 5563.
- Verifies: pure yaw remains `TURN-BYPASS`; translation reaches ONNX filtering.

- [ ] **Step 1: Run formatting/static checks used by the repository**

Run: `git diff --check`

Run: `python -m py_compile gear_sonic/camera/sensor_server.py gear_sonic/camera/drivers/realsense.py gear_sonic/camera/composed_camera.py gear_sonic/utils/inference/object_nav_geometry.py gear_sonic/utils/inference/object_nav.py gear_sonic/scripts/lavira_planner.py gear_sonic/scripts/launch_inference.py`

Expected: no output and exit status 0.

- [ ] **Step 2: Run focused regression suite**

Run: `pytest -q gear_sonic/tests/test_camera_rgbd_protocol.py gear_sonic/tests/test_object_nav_geometry.py gear_sonic/tests/test_object_nav_codex.py gear_sonic/tests/test_lavira_planner.py gear_sonic/tests/test_keyboard_command_publisher.py`

If `test_keyboard_command_publisher.py` is not present on `replay_real`, run
the four new test files plus the repository's existing keyboard planner tests.
Expected: PASS.

- [ ] **Step 3: Run existing REASAN tests**

Run: `pytest -q gear_sonic/tests -k 'reasan or planner_control'`

Expected: PASS; no existing REASAN behavior changed.

- [ ] **Step 4: Add and run an exact LaViRA/REASAN protocol contract test**

Create a test that builds the LaViRA rotation, translation, and stop JSON via
the Task 5 adapter, passes each string to the unchanged
`reasan_planner.decode_velocity_command`, and asserts the decoded duration and
velocity arrays. Also reproduce the unchanged pure-yaw predicate and assert
rotation selects bypass while translation does not:

```python
rotation = decode_velocity_command(build_reasan_velocity_message(
    VelocityCommand(0.0, 0.0, 0.4, 1.0), action="turn_left"
))
translation = decode_velocity_command(build_reasan_velocity_message(
    VelocityCommand(0.3, 0.0, 0.0, 1.0), action="move_forward"
))
assert rotation["velocity"].tolist() == pytest.approx([0.0, 0.0, 0.4])
assert translation["velocity"].tolist() == pytest.approx([0.3, 0.0, 0.0])
assert abs(float(rotation["velocity"][0])) <= 1e-6
assert abs(float(rotation["velocity"][1])) <= 1e-6
assert abs(float(rotation["velocity"][2])) > 1e-6
assert abs(float(translation["velocity"][0])) > 1e-6
```

Run: `pytest -q gear_sonic/tests/test_lavira_reasan_contract.py`

Expected: PASS without robot hardware or a running ONNX session.

- [ ] **Step 5: Run the complete software-only suite**

Run: `pytest -q gear_sonic/tests`

Expected: PASS, excluding only tests explicitly marked for unavailable robot,
camera, simulator, or network hardware. Record exact skips or environment-only
failures in the handoff.

- [ ] **Step 6: Commit the contract test and any verified fixes**

```bash
git add gear_sonic/tests/test_lavira_reasan_contract.py
git commit -m "test: verify LaViRA REASAN protocol contract"
```

If production files were also fixed, add only the already-scoped production
files changed to correct the verified defect and use commit message
`fix: harden LaViRA REASAN integration` instead.

### Task 8: MuJoCo and hardware rollout handoff

**Files:**
- No source changes expected.
- Update the implementation handoff with commands, observed outputs, and any unavailable hardware checks.

**Interfaces:**
- Verifies operational behavior beyond software-only tests.

- [ ] **Step 1: Launch the existing stack in simulation**

Run:

```bash
python gear_sonic/scripts/launch_inference.py --sim \
  --planner-input lavira \
  --lavira-mission "approach the red chair" \
  --lavira-global-target "red chair"
```

Confirm pane 3 shows `IDLE` and pane 4 reports healthy ActorRay/IMU. Then
terminate the session and launch once with `--sim` and no `--planner-input`;
confirm pane 3 uses the existing keyboard source.

- [ ] **Step 2: Exercise cancellation before motion**

Press `n`, then Space while Codex is pending. Expected: pane 3 reports cancel,
no late result moves the robot, and pane 4 reaches `SAFE-STOP`.

- [ ] **Step 3: Exercise one valid synthetic or staged cycle**

Expected sequence in pane 3:

```text
INFERENCING -> ROTATING -> STOP_GAP -> FORWARD -> FINAL_STOP -> IDLE
```

Expected pane 4 status: rotation uses `TURN-BYPASS`; forward uses `OK` and the
ONNX-filtered output; final state is zero velocity.

- [ ] **Step 4: Perform hardware verification only with operator approval**

Use the CLI to override `--max-direct-travel` to 0.3-0.5 m for the first real
robot run while retaining the implementation default of 8.0 m. Verify stop,
cancel, stale command, and obstacle suppression before increasing distance.

- [ ] **Step 5: Record final evidence**

Report commits, exact test commands and results, skipped hardware checks,
observed tmux state sequence, REASAN status sequence, and any recommended
follow-up. Do not claim hardware validation if it was not run.
