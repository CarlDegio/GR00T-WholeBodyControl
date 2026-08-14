# GROOT RGB Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show every RGB camera used by GROOT data collection in one non-blocking OpenCV preview window.

**Architecture:** A focused `RgbPreviewWorker` owns the GUI thread and a latest-frame cache. `DataExporterSensorGatewayIngress` publishes each fresh camera message to that worker without changing the data returned to the collector.

**Tech Stack:** Python 3.10+, threading, NumPy, OpenCV, pytest

## Global Constraints

- The preview is enabled by default while `run_data_exporter` is alive.
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

### Task 2: DataExporter SensorGateway integration

**Files:**
- Modify: `gear_sonic/runtime/data_exporter_sensor_gateway.py`
- Modify: `gear_sonic/scripts/run_data_exporter.py`
- Modify: `gear_sonic/tests/test_data_exporter_sensor_gateway.py`

**Interfaces:**
- Consumes: `RgbPreviewWorker` from Task 1 and `preview_rgb: bool` configuration.
- Produces: ingress lifecycle wiring that starts, updates, and closes the preview worker.

- [x] **Step 1: Write failing integration tests** proving camera messages reach the preview worker and lifecycle cleanup is idempotent.
- [x] **Step 2: Verify RED** with the focused DataExporter SensorGateway test file.
- [x] **Step 3: Add `preview_rgb=True` configuration** and inject/start/publish/close the worker from the ingress.
- [x] **Step 4: Verify GREEN** with focused tests, then run the related regression suite.
- [x] **Step 5: Commit only feature files** on `feature/groot-rgb-preview`.
