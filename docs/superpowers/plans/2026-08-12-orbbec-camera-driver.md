# Orbbec Camera Driver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an official Orbbec SDK v2 RGB-D driver whose `read()` output is wire-compatible with `RealSenseSensor`.

**Architecture:** A new `OrbbecSensor` owns Orbbec device discovery, stream configuration, software depth-to-color alignment, NumPy conversion, and the existing sensor-server lifecycle. The compiled SDK is the only mocked boundary in tests; the real driver and serialization code are exercised.

**Tech Stack:** Python 3.10, NumPy, `pyorbbecsdk2`/`pyorbbecsdk`, pytest, existing Gear Sonic camera protocol.

## Global Constraints

- The runtime target is Linux ARM64 on the G1.
- Color output is RGB `uint8[H,W,3]`.
- Depth output is color-aligned raw `uint16[H,W]`.
- `depth_scale_m` is metres per raw depth unit.
- `read()` has only `timestamps`, `images`, and `camera_info` top-level keys.
- The selected serial is exposed as `sensor.serial_number`, not added to the wire payload.

---

### Task 1: Orbbec RGB-D Driver

**Files:**
- Create: `gear_sonic/camera/drivers/orbbec.py`
- Create: `gear_sonic/tests/test_orbbec_driver.py`

**Interfaces:**
- Consumes: official `pyorbbecsdk` `Context`, `Pipeline`, `Config`, stream-profile, frame-set, and `AlignFilter` APIs.
- Produces: `OrbbecConfig` and `OrbbecSensor` with the same constructor/lifecycle shape and `read() -> dict[str, Any] | None` payload as `RealSenseSensor`.

- [x] **Step 1: Write the failing RGB-D contract test**

Create a complete fake `pyorbbecsdk` module with two serial-numbered devices, exact RGB/Y16 profile selection, raw frames, and distinct aligned frames. Import the real driver and assert exact serial selection, RGB channel order, aligned `uint16` depth, shared timestamp, color intrinsics, metre scale, and wire serialization.

- [x] **Step 2: Run the contract test to verify RED**

Run: `python -m pytest gear_sonic/tests/test_orbbec_driver.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'gear_sonic.camera.drivers.orbbec'`.

- [x] **Step 3: Implement the minimal driver**

Implement serial-sorted selection using `DeviceList.get_device_serial_number_by_index()` and `get_device_by_serial_number()`. Request the configured RGB and Y16 profiles, require complete frame sets, start `Pipeline(device)`, and create `AlignFilter(OBStreamType.COLOR_STREAM)`. Convert SDK buffers with `np.frombuffer(...).reshape(...).copy()`, and build RealSense-compatible metadata from the selected color profile and aligned depth frame.

- [x] **Step 4: Run the contract test to verify GREEN**

Run: `python -m pytest gear_sonic/tests/test_orbbec_driver.py -q`

Expected: all tests pass.

- [x] **Step 5: Run related regression and static checks**

Run: `python -m pytest gear_sonic/tests/test_orbbec_driver.py gear_sonic/tests/test_camera_rgbd_protocol.py -q`

Run: `python -m compileall -q gear_sonic/camera/drivers/orbbec.py gear_sonic/tests/test_orbbec_driver.py`

Expected: both commands exit 0.
