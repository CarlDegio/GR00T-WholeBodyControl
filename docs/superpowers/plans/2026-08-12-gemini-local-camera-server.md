# Local Gemini Camera Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Install the official Orbbec Python SDK locally and run one Gemini 345Lg RGB-D camera through the existing composed-camera server on port 5555.

**Architecture:** Add an `orbbec` factory branch to the existing composed camera, plus a small Python preflight launcher that selects exactly one Gemini and verifies its USB 3 speed before replacing itself with `gear_sonic.camera.composed_camera`. A dedicated zsh wrapper uses the repository's `.venv_camera` interpreter and leaves the existing four-RealSense launcher untouched.

**Tech Stack:** Python 3.10, `pyorbbecsdk2`/`pyorbbecsdk`, NumPy, OpenCV, Tyro, ZMQ, pytest, zsh, Linux sysfs.

## Global Constraints

- Keep `start_camera_server.zsh` unchanged.
- Start only `ego_view`; do not configure chest or wrist cameras.
- Publish RGB `uint8[H,W,3]` and color-aligned depth `uint16[H,W]`.
- Use the existing RealSense-compatible `timestamps`, `images`, and `camera_info` payload.
- Reject zero or multiple Orbbec devices instead of selecting ambiguously.
- Require the selected Gemini 345Lg to negotiate at least 5000 Mb/s.
- Use ZMQ port 5555.

---

### Task 1: Register Orbbec With the Composed Camera

**Files:**
- Modify: `gear_sonic/camera/composed_camera.py:51-95,380-431`
- Create: `gear_sonic/tests/test_composed_camera_orbbec.py`

**Interfaces:**
- Consumes: `OrbbecConfig`, `OrbbecSensor`, `ComposedCameraConfig.fps`, and the selected mount/device ID.
- Produces: `ComposedCameraConfig.orbbec_enable_depth: bool` and `_instantiate_camera(..., camera_type="orbbec", ...) -> OrbbecSensor`.

- [ ] **Step 1: Write the failing factory test**

Create a fake `gear_sonic.camera.drivers.orbbec` module, construct a
`ComposedCameraSensor` without running its threaded constructor, and verify
the consumer-visible sensor configuration:

```python
def test_orbbec_factory_propagates_fps_depth_mount_and_serial(monkeypatch):
    fake_module = types.ModuleType("gear_sonic.camera.drivers.orbbec")

    class FakeConfig:
        fps = 30
        enable_depth = True

    captured = {}

    class FakeSensor:
        def __init__(self, *, config, mount_position, device_id):
            captured.update(
                config=config,
                mount_position=mount_position,
                device_id=device_id,
            )

    fake_module.OrbbecConfig = FakeConfig
    fake_module.OrbbecSensor = FakeSensor
    monkeypatch.setitem(sys.modules, "gear_sonic.camera.drivers.orbbec", fake_module)

    composed = object.__new__(ComposedCameraSensor)
    composed.config = ComposedCameraConfig(fps=10, orbbec_enable_depth=True)
    sensor = composed._instantiate_camera("ego_view", "orbbec", "CPMD464001G")

    assert isinstance(sensor, FakeSensor)
    assert captured["config"].fps == 10
    assert captured["config"].enable_depth is True
    assert captured["mount_position"] == "ego_view"
    assert captured["device_id"] == "CPMD464001G"
```

The production mutation caught is omitting the `orbbec` branch or failing to
propagate FPS/depth/device selection.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
python3 -m pytest gear_sonic/tests/test_composed_camera_orbbec.py -q
```

Expected: FAIL because `ComposedCameraConfig` does not accept
`orbbec_enable_depth` or `_instantiate_camera()` reports an unsupported camera.

- [ ] **Step 3: Implement the minimal factory branch**

Add this config field:

```python
orbbec_enable_depth: bool = False
"""Whether Orbbec cameras should publish aligned depth alongside color."""
```

Add this branch before replay/USB handling:

```python
elif camera_type == "orbbec":
    from gear_sonic.camera.drivers.orbbec import OrbbecConfig, OrbbecSensor

    print(
        f"Initializing Orbbec sensor for camera type: {camera_type}, "
        f"device: {device_id}"
    )
    orbbec_config = OrbbecConfig()
    orbbec_config.fps = self.config.fps
    orbbec_config.enable_depth = self.config.orbbec_enable_depth
    return OrbbecSensor(
        config=orbbec_config,
        mount_position=mount_position,
        device_id=device_id,
    )
```

Also list `orbbec` in the class/module camera-type documentation.

- [ ] **Step 4: Run focused and related tests**

Run:

```bash
python3 -m pytest \
  gear_sonic/tests/test_composed_camera_orbbec.py \
  gear_sonic/tests/test_orbbec_driver.py \
  gear_sonic/tests/test_camera_rgbd_protocol.py -q
```

Expected: all tests pass.

### Task 2: Add a Testable Single-Gemini Preflight Launcher

**Files:**
- Create: `gear_sonic/camera/gemini_server_launcher.py`
- Create: `gear_sonic/tests/test_gemini_server_launcher.py`
- Create: `start_gemini_camera_server.zsh`

**Interfaces:**
- Consumes: `pyorbbecsdk.Context().query_devices()`, `/sys/bus/usb/devices`, and `sys.executable`.
- Produces: `discover_single_gemini(context) -> str`, `read_gemini_usb_speed(serial, sysfs_root) -> int`, `build_server_argv(serial) -> list[str]`, and a zsh entry point.

- [ ] **Step 1: Write failing discovery, USB, and argv tests**

Use a temporary sysfs tree with literal `idVendor=2bc5`, `idProduct=0813`,
`serial=CPMD464001G`, and `speed=5000`. Use complete fake device-list methods
for zero, one, and two-device cases. Assert:

```python
assert discover_single_gemini(one_device_context) == "CPMD464001G"
assert read_gemini_usb_speed("CPMD464001G", tmp_path) == 5000
assert build_server_argv("CPMD464001G") == [
    sys.executable,
    "-m",
    "gear_sonic.camera.composed_camera",
    "--ego-view-camera",
    "orbbec",
    "--ego-view-device-id",
    "CPMD464001G",
    "--orbbec-enable-depth",
    "--port",
    "5555",
]
```

Assert zero and multiple devices raise `RuntimeError`, a missing sysfs match
raises `RuntimeError`, and speed `480` raises a message containing `USB 3` and
`480` from `validate_usb3_speed(480)`.

The production mutations caught are selecting the first of multiple devices,
matching another vendor/product, accepting USB 2, or accidentally enabling
additional cameras.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
python3 -m pytest gear_sonic/tests/test_gemini_server_launcher.py -q
```

Expected: collection fails because
`gear_sonic.camera.gemini_server_launcher` does not exist.

- [ ] **Step 3: Implement the preflight module**

Implement lazy SDK import inside `main()`, exact serial discovery, exact
VID/PID/serial sysfs matching, integer speed validation, and process replacement:

```python
def main() -> None:
    from pyorbbecsdk import Context

    serial = discover_single_gemini(Context())
    speed = read_gemini_usb_speed(serial, Path("/sys/bus/usb/devices"))
    validate_usb3_speed(speed)
    argv = build_server_argv(serial)
    print(f"Starting Gemini {serial} at {speed} Mb/s", flush=True)
    os.execv(sys.executable, argv)
```

`build_server_argv()` must contain only the ego-view Orbbec options shown in
Step 1, so all other camera defaults remain `None`.

- [ ] **Step 4: Add the dedicated zsh wrapper**

Create an executable script that resolves its own repository directory and
then replaces itself with the Python launcher:

```zsh
#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
PYTHON="$SCRIPT_DIR/.venv_camera/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    print -u2 "ERROR: missing $PYTHON"
    print -u2 "Create .venv_camera and install pyorbbecsdk2 first."
    exit 1
fi

exec "$PYTHON" -m gear_sonic.camera.gemini_server_launcher
```

Run `chmod +x start_gemini_camera_server.zsh`.

- [ ] **Step 5: Run launcher tests and static checks**

Run:

```bash
python3 -m pytest gear_sonic/tests/test_gemini_server_launcher.py -q
python3 -m compileall -q \
  gear_sonic/camera/gemini_server_launcher.py \
  gear_sonic/camera/composed_camera.py
zsh -n start_gemini_camera_server.zsh
```

Expected: all commands exit 0.

### Task 3: Install the SDK and Verify Real RGB-D Frames

**Files:**
- Runtime environment only: `.venv_camera/` (git-ignored)
- No tracked source file is created in this task.

**Interfaces:**
- Consumes: the locally connected Gemini 345Lg at USB speed 5000 Mb/s and the driver from `gear_sonic/camera/drivers/orbbec.py`.
- Produces: a working local Python environment and hardware evidence for RGB, aligned depth, intrinsics, depth scale, and single-camera server readiness.

- [ ] **Step 1: Create the local Python 3.10 camera environment**

Use `uv` when available; otherwise use `python3 -m venv`. Do not run the
repository's destructive installer because it deletes an existing camera
environment.

```bash
uv venv .venv_camera --python 3.10 --prompt gear_sonic_camera
```

- [ ] **Step 2: Install only the required local runtime and test packages**

```bash
uv pip install --python .venv_camera/bin/python --no-deps -e gear_sonic
uv pip install --python .venv_camera/bin/python \
  pyorbbecsdk2==2.1.2 \
  numpy==1.26.4 \
  opencv-python \
  pyzmq \
  msgpack \
  msgpack-numpy \
  tyro \
  pytest
```

Expected: installation exits 0 and
`.venv_camera/bin/python -c 'import pyorbbecsdk'` succeeds.

- [ ] **Step 3: Re-run all focused tests inside the deployment environment**

Run:

```bash
.venv_camera/bin/python -m pytest \
  gear_sonic/tests/test_composed_camera_orbbec.py \
  gear_sonic/tests/test_gemini_server_launcher.py \
  gear_sonic/tests/test_orbbec_driver.py \
  gear_sonic/tests/test_camera_rgbd_protocol.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Run a bounded real-driver RGB-D test**

Instantiate `OrbbecSensor` with the discovered serial and depth enabled, read
up to 30 times, and exit successfully only after validating literal protocol
properties:

```python
assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
assert depth.dtype == np.uint16 and depth.shape == rgb.shape[:2]
assert info["width"] == rgb.shape[1]
assert info["height"] == rgb.shape[0]
assert info["depth_scale_m"] > 0
```

Always call `sensor.close()` in `finally`.

- [ ] **Step 5: Run the same RGB-D path through the composed sensor**

Construct `ComposedCameraSensor` with only:

```python
ComposedCameraConfig(
    ego_view_camera="orbbec",
    ego_view_device_id="CPMD464001G",
    fps=30,
    orbbec_enable_depth=True,
    run_as_server=False,
    server=False,
)
```

Poll for a bounded 15 seconds, require a message whose sole mount key is
`ego_view`, verify its images contain `ego_view` and `ego_view_depth`, and
always call `close()`.

- [ ] **Step 6: Smoke-test the long-running launch entry**

Run `./start_gemini_camera_server.zsh`, wait until output reports the one
camera ready and the server running on port 5555, then terminate it with
SIGINT. Confirm no RealSense, chest, or wrist initialization appears.

- [ ] **Step 7: Final regression and diff checks**

Run:

```bash
.venv_camera/bin/python -m pytest \
  gear_sonic/tests/test_composed_camera_orbbec.py \
  gear_sonic/tests/test_gemini_server_launcher.py \
  gear_sonic/tests/test_orbbec_driver.py \
  gear_sonic/tests/test_camera_rgbd_protocol.py -q
git diff --check
git status --short
```

Expected: tests pass, diff check is clean, and `.venv_camera` is not tracked.
