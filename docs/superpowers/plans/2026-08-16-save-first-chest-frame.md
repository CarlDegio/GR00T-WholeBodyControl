# Save First Chest RGB Frame Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone command that saves the first valid `chest_view` RGB frame from the composed camera server.

**Architecture:** Mirror the existing ego-frame utility in a separate module so the established command-line and ZMQ behavior remain unchanged. Select only `images["chest_view"]`, validate the RGB array, save it through OpenCV, and always close the camera client.

**Tech Stack:** Python, NumPy, OpenCV, `ComposedCameraClientSensor`, pytest

## Global Constraints

- Do not modify the existing head-camera utility or camera-server launchers.
- Capture only `chest_view` RGB; do not capture depth.
- Keep the CLI parallel to `save_first_ego_frame.py`.

---

### Task 1: Chest RGB first-frame utility

**Files:**
- Create: `gear_sonic/scripts/save_first_chest_frame.py`
- Test: `gear_sonic/tests/test_save_first_chest_frame.py`

**Interfaces:**
- Consumes: `ComposedCameraClientSensor(server_ip: str, port: int, decode_images: bool)` and messages containing `images["chest_view"]`.
- Produces: `extract_chest_rgb(message) -> np.ndarray | None`, `save_rgb_png(image_rgb, output_path, overwrite=False) -> None`, and `capture_first_chest_frame(camera_host, camera_port, output_path, timeout_sec, ready_file=None, overwrite=False) -> None`.

- [ ] **Step 1: Write failing tests for stream selection, PNG output, and capture cleanup**

Create `gear_sonic/tests/test_save_first_chest_frame.py` with tests that:

```python
image = np.zeros((2, 3, 3), dtype=np.uint8)
assert extract_chest_rgb({"images": {"chest_view": image}}) is image
assert extract_chest_rgb({"images": {"ego_view": image}}) is None
```

Also verify malformed shape/dtype errors, exact RGB round-trip through PNG,
overwrite rejection, readiness-file creation, first-valid-frame selection, and
client closure using a fake `ComposedCameraClientSensor`.

- [ ] **Step 2: Run the new tests and verify the missing module fails**

Run:

```bash
.venv_inference/bin/python -m pytest gear_sonic/tests/test_save_first_chest_frame.py -q
```

Expected: collection fails because `gear_sonic.scripts.save_first_chest_frame` does not exist.

- [ ] **Step 3: Implement the standalone chest-frame command**

Create `gear_sonic/scripts/save_first_chest_frame.py` with the same validation,
timeout loop, ready-file behavior, RGB-to-BGR conversion, overwrite protection,
and CLI arguments as `save_first_ego_frame.py`. Replace the selected image key
and user-facing text with `chest_view`, and call `client.close()` in `finally`.

- [ ] **Step 4: Run the focused tests**

Run:

```bash
.venv_inference/bin/python -m pytest gear_sonic/tests/test_save_first_chest_frame.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Verify import syntax and repository formatting**

Run:

```bash
.venv_inference/bin/python -m py_compile gear_sonic/scripts/save_first_chest_frame.py
git diff --check
```

Expected: both commands exit successfully with no output.

- [ ] **Step 6: Commit the implementation**

```bash
git add docs/superpowers/plans/2026-08-16-save-first-chest-frame.md gear_sonic/scripts/save_first_chest_frame.py gear_sonic/tests/test_save_first_chest_frame.py
git commit -m "feat: add chest RGB first-frame capture"
```
