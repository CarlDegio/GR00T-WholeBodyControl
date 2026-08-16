# Local Chest RGB Capture Launcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local Bash launcher that connects the chest-frame Python client to the camera server running on the robot.

**Architecture:** A repository-root Bash script validates the robot host and port, chooses a local Python environment, creates a timestamped local output path by default, and executes `gear_sonic.scripts.save_first_chest_frame`. It never starts or stops the robot camera server.

**Tech Stack:** Bash, Python module CLI, pytest subprocess test

## Global Constraints

- The first argument is the robot hostname or IP.
- Optional arguments are local output PNG path and camera port, in that order.
- Default port is `5555`; default output is under `outputs/camera_startup/`.
- Do not launch or manage any camera-server process.

---

### Task 1: Local remote-capture launcher

**Files:**
- Create: `save_first_chest_frame.sh`
- Modify: `gear_sonic/tests/test_save_first_chest_frame.py`

**Interfaces:**
- Consumes: `ROBOT_IP [OUTPUT_PNG] [CAMERA_PORT]` and optional `FIRST_CHEST_FRAME_PYTHON`.
- Produces: one invocation of `python -m gear_sonic.scripts.save_first_chest_frame` with host, port, and output arguments.

- [ ] **Step 1: Add a failing subprocess test**

Invoke the launcher with `FIRST_CHEST_FRAME_PYTHON=/bin/echo`, robot host `192.168.123.164`, explicit PNG path, and port `6000`. Assert exit code zero and that stdout contains the module, host, port, and output arguments.

- [ ] **Step 2: Verify the launcher test fails because the script is absent**

```bash
.venv_inference/bin/python -m pytest gear_sonic/tests/test_save_first_chest_frame.py::test_local_launcher_forwards_remote_camera_arguments -q
```

- [ ] **Step 3: Implement the launcher**

Create an executable Bash script with `set -eu`, required robot host handling, integer port validation, timestamped local PNG default, the `FIRST_CHEST_FRAME_PYTHON` override, fallback search through local inference/data/camera environments, and execution of the Python module. Do not start a camera server.

- [ ] **Step 4: Run launcher and Python tests**

```bash
.venv_inference/bin/python -m pytest gear_sonic/tests/test_save_first_chest_frame.py -q
./save_first_chest_frame.sh --help
```

- [ ] **Step 5: Verify formatting and commit**

```bash
bash -n save_first_chest_frame.sh
git diff --check
git add save_first_chest_frame.sh gear_sonic/tests/test_save_first_chest_frame.py docs/superpowers/specs/2026-08-16-save-first-chest-frame-design.md docs/superpowers/plans/2026-08-16-save-first-chest-frame-launcher.md
git commit -m "feat: add local chest frame capture launcher"
```
