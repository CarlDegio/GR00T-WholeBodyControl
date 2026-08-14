# GROOT RGB Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show every RGB camera used by GROOT data collection in one non-blocking OpenCV preview window.

**Architecture:** A focused `RgbPreviewWorker` owns the GUI thread and a latest-frame cache. `CameraZmqIngress` publishes each fresh encoded camera message to that worker inside the GUI-capable SensorGateway process; the data-collection launcher enables it without changing data returned to the collector.

**Tech Stack:** Python 3.10+, threading, NumPy, OpenCV, pytest

## Global Constraints

- The preview is enabled while the data-collection SensorGateway is alive.
- All enabled RGB cameras share one `GROOT RGB Preview` window.
- Pressing `q` closes only the preview; preview failures never stop data collection.
- Existing encoded and decoded collection payloads remain unchanged.

---

### Task 1: RGB preview worker

**Files:**
- Create: `gear_sonic/runtime/rgb_preview.py`
- Test: `gear_sonic/tests/test_rgb_preview.py`

**Interfaces:**
- Consumes: `dict[str, bytes | str | numpy.ndarray]` camera images and an `encoded: bool` flag.
- Produces: `RgbPreviewWorker.start()`, `publish(images)`, and `close()`.

- [x] **Step 1: Write failing tests** for JPEG/RGB conversion, labelled tiled output, latest-frame replacement, `q` shutdown, and display-error isolation.
- [x] **Step 2: Verify RED** with `python -m pytest gear_sonic/tests/test_rgb_preview.py -q` and confirm the module/API is missing.
- [x] **Step 3: Implement minimal worker** using a condition-protected latest-frame cache and one daemon thread.
- [x] **Step 4: Verify GREEN** with the same focused pytest command.

### Task 2: SensorGateway integration

**Files:**
- Modify: `gear_sonic/runtime/sensor_gateway.py`
- Modify: `gear_sonic/scripts/run_sensor_gateway.py`
- Modify: `gear_sonic/scripts/launch_data_collection.py`
- Modify: `gear_sonic/tests/test_sensor_gateway.py`
- Modify: `gear_sonic/tests/test_run_sensor_gateway.py`
- Modify: `gear_sonic/tests/test_camera_viewer.py`

**Interfaces:**
- Consumes: `RgbPreviewWorker` from Task 1 and `preview_rgb: bool` ingress configuration.
- Produces: Camera ingress lifecycle wiring and a data-collection launcher command that enables it.

- [x] **Step 1: Write failing integration tests** proving camera messages reach the preview worker and the launcher enables it.
- [x] **Step 2: Verify RED** with focused SensorGateway and launcher tests.
- [x] **Step 3: Add `preview_rgb=True` to Camera ingress and start/publish/close the worker there.**
- [x] **Step 4: Verify GREEN** with focused tests, then run the related regression suite.
- [x] **Step 5: Commit only feature files** on `feature/groot-rgb-preview`.
