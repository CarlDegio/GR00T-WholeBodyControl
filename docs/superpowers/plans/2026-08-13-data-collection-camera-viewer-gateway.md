# Data Collection Camera Viewer Gateway Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `run_camera_viewer.py` read decoded RGB camera frames exclusively through the local SensorGateway while preserving its existing display and MP4 recording behavior.

**Architecture:** Add a small camera-shaped adapter around `SensorGatewayClient`: it discovers decoded `camera/*` RGB streams through Gateway health, requests them through the Snapshot API, and returns the existing `{"images": ...}` shape consumed by the viewer loop. Change the data-collection launcher to pass its runtime profile to the viewer instead of a direct camera host and port.

**Tech Stack:** Python 3.10, ZeroMQ SensorGateway RPC, shared-memory Snapshot API, NumPy, OpenCV, Tyro, pytest.

## Global Constraints

- The viewer is Gateway-only; do not retain `camera_host`, `camera_port`, or a direct-camera compatibility mode.
- Do not modify `run_operator_cv_viewer.py`.
- Preserve the current horizontal tile layout, RGB-to-BGR conversion, labels, FPS pacing, `R` MP4 recording, `Q` quit, output directory, and codec behavior.
- Ignore all depth and non-camera Gateway streams.
- Do not add dataset depth recording, readiness gating, camera endpoint default changes, or camera installation changes.

---

### Task 1: Gateway-backed camera adapter

**Files:**
- Modify: `gear_sonic/scripts/run_camera_viewer.py`
- Test: `gear_sonic/tests/test_camera_viewer.py`

**Interfaces:**
- Consumes: `load_runtime_profile(path)`, `SensorGatewayClient.health()`, `SensorGatewayClient.read_snapshot(SnapshotRequest)`, and decoded `camera/<name>` shared-memory arrays.
- Produces: `_gateway_rgb_streams(health: Mapping[str, Any]) -> tuple[str, ...]` and `GatewayCameraClient.read(blocking: bool = False) -> dict[str, dict[str, np.ndarray]] | None` with the same `images` shape used by the existing viewer loop.

- [ ] **Step 1: Write failing discovery and adapter tests**

Extend `gear_sonic/tests/test_camera_viewer.py` with a fake Gateway client and tests equivalent to:

```python
def test_gateway_rgb_streams_are_sorted_and_exclude_depth_and_non_camera():
    health = {
        "streams": {
            "camera/right_wrist": {},
            "camera/ego_view_depth": {},
            "source/camera_server": {},
            "camera/ego_view": {},
        }
    }
    assert _gateway_rgb_streams(health) == (
        "camera/ego_view",
        "camera/right_wrist",
    )


def test_gateway_camera_client_materializes_rgb_images_by_camera_name():
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    fake = FakeGatewayClient(
        health={"streams": {"camera/ego_view": {}}},
        arrays={"camera/ego_view": rgb},
    )
    camera = GatewayCameraClient("inproc://unused", client=fake)

    message = camera.read()

    assert message is not None
    np.testing.assert_array_equal(message["images"]["ego_view"], rgb)
    assert fake.requests[0].streams == ("camera/ego_view",)
```

Add a third test where health or snapshot RPC raises `SensorGatewayClientError`; assert `read()` returns `None`. Include an invalid depth-shaped or non-`uint8` camera array and assert it is omitted rather than returned as RGB.

- [ ] **Step 2: Run the new tests to verify RED**

Run:

```bash
.venv_teleop/bin/python -m pytest -q \
  gear_sonic/tests/test_camera_viewer.py
```

Expected: collection fails because `_gateway_rgb_streams` and `GatewayCameraClient` do not exist.

- [ ] **Step 3: Implement the minimal Gateway adapter**

In `run_camera_viewer.py`:

- replace `ComposedCameraClientSensor` with imports for `load_runtime_profile`, `SensorGatewayClient`, `SensorGatewayClientError`, and `SnapshotRequest`;
- replace `CameraViewerConfig.camera_host` and `.camera_port` with a `profile` field defaulting to `default_runtime_profile_path()`;
- implement `_gateway_rgb_streams()` using the health response's `streams` keys;
- implement `GatewayCameraClient` with injectable `client`, cached discovery, one multi-stream snapshot request, `max_age_ms=1500.0`, `max_skew_ms=5.0`, and `retries=0`;
- accept only `HxWx3 uint8` arrays, strip the `camera/` prefix, and return `None` on `SensorGatewayClientError`;
- construct the adapter from `profile.endpoint_uri("sensor_gateway_metadata")` in `main()`;
- ensure the no-first-frame exit also closes the adapter.

Do not change the display or recording loop beyond replacing its data source.

- [ ] **Step 4: Run focused tests to verify GREEN**

Run:

```bash
.venv_teleop/bin/python -m pytest -q \
  gear_sonic/tests/test_camera_viewer.py \
  gear_sonic/tests/test_sensor_gateway_client.py
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the Gateway adapter**

```bash
git add gear_sonic/scripts/run_camera_viewer.py gear_sonic/tests/test_camera_viewer.py
git commit -m "feat: read data collection viewer from sensor gateway"
```

---

### Task 2: Launch the viewer through the runtime profile

**Files:**
- Modify: `gear_sonic/scripts/launch_data_collection.py`
- Test: `gear_sonic/tests/test_camera_viewer.py`

**Interfaces:**
- Consumes: `DataCollectionLaunchConfig.runtime_profile` and the Gateway-only `run_camera_viewer.py --profile <path>` CLI from Task 1.
- Produces: `build_camera_viewer_command(config: DataCollectionLaunchConfig, repo_root: Path) -> str`, used by the tmux launcher and directly testable without starting tmux.

- [ ] **Step 1: Write the failing launcher command test**

Add a test equivalent to:

```python
def test_data_collection_launcher_viewer_uses_profile_without_direct_camera():
    command = build_camera_viewer_command(
        DataCollectionLaunchConfig(runtime_profile="/tmp/runtime profile.yaml"),
        Path("/workspace/sonic"),
    )
    assert "run_camera_viewer.py" in command
    assert "--profile '/tmp/runtime profile.yaml'" in command
    assert "--camera-host" not in command
    assert "--camera-port" not in command
```

- [ ] **Step 2: Run the launcher test to verify RED**

Run:

```bash
.venv_teleop/bin/python -m pytest -q \
  gear_sonic/tests/test_camera_viewer.py::test_data_collection_launcher_viewer_uses_profile_without_direct_camera
```

Expected: collection fails because `build_camera_viewer_command` does not exist.

- [ ] **Step 3: Implement and use the command builder**

Add `build_camera_viewer_command()` to `launch_data_collection.py`. Quote both the repository path and runtime profile with `shlex.quote()`, activate `.venv_data_collection`, and invoke:

```text
python gear_sonic/scripts/run_camera_viewer.py --profile <quoted-profile>
```

Replace the inline direct-viewer command in `main()` with this helper. Keep `camera_host` and `camera_port` in `DataCollectionLaunchConfig` because SensorGateway and the simulator still consume them; update their docstrings so they no longer claim the viewer consumes them.

- [ ] **Step 4: Run focused tests to verify GREEN**

Run:

```bash
.venv_teleop/bin/python -m pytest -q \
  gear_sonic/tests/test_camera_viewer.py
```

Expected: all viewer and launcher command tests pass.

- [ ] **Step 5: Commit launcher integration**

```bash
git add gear_sonic/scripts/launch_data_collection.py gear_sonic/tests/test_camera_viewer.py
git commit -m "refactor: launch camera viewer through sensor gateway"
```

---

### Task 3: Regression verification

**Files:**
- Verify: `gear_sonic/scripts/run_camera_viewer.py`
- Verify: `gear_sonic/scripts/launch_data_collection.py`
- Verify: related tests and CLI entry points

**Interfaces:**
- Consumes: the Gateway adapter and launcher command builder from Tasks 1-2.
- Produces: fresh evidence that the Gateway-only viewer preserves existing behavior and imports in both relevant virtual environments.

- [ ] **Step 1: Run viewer, Gateway, launcher, and exporter regression tests**

Run:

```bash
.venv_teleop/bin/python -m pytest -q \
  gear_sonic/tests/test_camera_viewer.py \
  gear_sonic/tests/test_operator_cv_viewer.py \
  gear_sonic/tests/test_sensor_gateway.py \
  gear_sonic/tests/test_sensor_gateway_client.py \
  gear_sonic/tests/test_data_exporter_sensor_gateway.py \
  gear_sonic/tests/test_runtime_config.py
```

Expected: all selected tests pass.

- [ ] **Step 2: Verify both CLIs parse their new commands**

Run:

```bash
.venv_data_collection/bin/python gear_sonic/scripts/run_camera_viewer.py --help
.venv_data_collection/bin/python gear_sonic/scripts/launch_data_collection.py --help
```

Expected: both commands exit 0; viewer help contains `--profile` and does not contain `--camera-host` or `--camera-port`; launcher help retains camera host/port for SensorGateway and simulation.

- [ ] **Step 3: Check formatting and review the final diff**

Run:

```bash
git diff --check HEAD~2..HEAD
git status --short
```

Expected: no whitespace errors; only the pre-existing untracked `Log/` remains outside committed work.
