# Right-Wrist OpenCV Rotation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rotate only the right-wrist camera image displayed by the unified OpenCV operator viewer by 180 degrees.

**Architecture:** Keep all camera and SensorGateway data unchanged. Apply the physical-mount display correction at `gateway_frame_to_bgr()`, after RGB-to-BGR decoding and before the viewer-local cache and canvas composer.

**Tech Stack:** Python 3.10, NumPy, OpenCV, pytest.

## Global Constraints

- Rotate only stream `camera/right_wrist`.
- Use `cv2.rotate(image, cv2.ROTATE_180)` after RGB-to-BGR conversion.
- Head, chest, left-wrist, navigation, RGB-D, and LingBot display orientation remains unchanged.
- Do not modify SensorGateway payloads, camera drivers, model inputs, shared memory, or dataset recording.
- Preserve unrelated user changes and the already completed Livox implementation.

---

### Task 1: Right-Wrist Display Decode Correction

**Files:**
- Modify: `gear_sonic/tests/test_operator_cv_viewer.py`
- Modify: `gear_sonic/scripts/run_operator_cv_viewer.py:140-151`

**Interfaces:**
- Consumes: `gateway_frame_to_bgr(stream: str, payload: np.ndarray)` and `RIGHT_WRIST_RGB_STREAM`.
- Produces: right-wrist BGR frames rotated 180 degrees; every other supported stream retains current decoding behavior.

- [ ] **Step 1: Write asymmetric failing orientation tests**

Replace the existing one-pixel RGB/BGR test with an asymmetric chest fixture and add a right-wrist fixture:

```python
def test_non_right_wrist_camera_is_converted_to_bgr_without_rotation() -> None:
    rgb = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 0]],
        ],
        dtype=np.uint8,
    )

    bgr = gateway_frame_to_bgr(CHEST_RGB_STREAM, rgb)

    assert bgr is not None
    np.testing.assert_array_equal(
        bgr,
        np.array(
            [
                [[0, 0, 255], [0, 255, 0]],
                [[255, 0, 0], [0, 255, 255]],
            ],
            dtype=np.uint8,
        ),
    )


def test_right_wrist_camera_is_converted_to_bgr_and_rotated_180_degrees() -> None:
    rgb = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 0]],
        ],
        dtype=np.uint8,
    )

    bgr = gateway_frame_to_bgr(RIGHT_WRIST_RGB_STREAM, rgb)

    assert bgr is not None
    np.testing.assert_array_equal(
        bgr,
        np.array(
            [
                [[0, 255, 255], [255, 0, 0]],
                [[0, 255, 0], [0, 0, 255]],
            ],
            dtype=np.uint8,
        ),
    )
```

- [ ] **Step 2: Run the viewer tests and verify the right-wrist test fails**

Run:

```bash
/home/user/Project/GR00T-WholeBodyControl/.venv_teleop/bin/python -m pytest \
  gear_sonic/tests/test_operator_cv_viewer.py -v
```

Expected: existing canvas tests and the non-right-wrist test pass; the right-wrist orientation test fails because the current decoder only performs RGB-to-BGR conversion.

- [ ] **Step 3: Implement the display-only rotation**

Change only the camera branch in `gateway_frame_to_bgr()`:

```python
    if stream in CAMERA_RGB_STREAMS:
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            return None
        image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        if stream == RIGHT_WRIST_RGB_STREAM:
            return cv2.rotate(image, cv2.ROTATE_180)
        return image
```

- [ ] **Step 4: Run the focused and combined regression tests**

Run:

```bash
/home/user/Project/GR00T-WholeBodyControl/.venv_teleop/bin/python -m pytest \
  gear_sonic/tests/test_operator_cv_viewer.py \
  gear_sonic/tests/test_livox_front_sector_filter.py \
  gear_sonic/tests/test_launch_tmux_panes.py -v
git diff --check
```

Expected: all viewer, Livox filter, and launcher tests pass; `git diff --check` prints nothing.

- [ ] **Step 5: Commit the isolated viewer fix**

```bash
git add \
  gear_sonic/scripts/run_operator_cv_viewer.py \
  gear_sonic/tests/test_operator_cv_viewer.py
git commit -m "fix: rotate right-wrist OpenCV view"
```

### Task 2: Final Combined Verification

**Files:**
- Verify: all Livox and right-wrist files changed since `cd67559`.

**Interfaces:**
- Consumes: completed Livox shared filter and right-wrist display decoder.
- Produces: fresh evidence that both modifications pass their focused and integration regressions without altering unrelated files.

- [ ] **Step 1: Compile all modified production Python files**

Run:

```bash
/home/user/Project/GR00T-WholeBodyControl/.venv_teleop/bin/python -m compileall -q \
  gear_sonic/scripts/run_livox_front_sector_filter.py \
  gear_sonic/scripts/launch_inference.py \
  gear_sonic/scripts/run_operator_cv_viewer.py
```

Expected: exit status 0 and no output.

- [ ] **Step 2: Run the complete relevant regression set**

Run:

```bash
/home/user/Project/GR00T-WholeBodyControl/.venv_teleop/bin/python -m pytest \
  gear_sonic/tests/test_livox_front_sector_filter.py \
  gear_sonic/tests/test_launch_tmux_panes.py \
  gear_sonic/tests/test_operator_cv_viewer.py \
  gear_sonic/tests/test_navdp_readiness_gate.py \
  gear_sonic/tests/test_sensor_gateway.py \
  gear_sonic/tests/test_runtime_config.py \
  gear_sonic/tests/test_runtime_contracts.py -v
git diff --check
```

Expected: all selected tests pass and the diff check is clean.

- [ ] **Step 3: Audit branch scope and working-tree cleanliness**

Run:

```bash
git diff --name-status cd67559..HEAD
git status --short
```

Expected: only the Livox implementation, the OpenCV viewer/test, and their specification/plan documents differ from the base; the feature worktree has no uncommitted files.
