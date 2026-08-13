# JPEG Quality 95 Camera Stream Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Compare JPEG quality 80 and 95 on the Sonic camera server with repeatable PC-side measurements of wire rate, frame rate, image throughput, end-to-end latency, and decode cost.

**Architecture:** Add a default-preserving `jpeg_quality` option to the host-encoded RGB JPEG path, separate from OAK's on-device `mjpeg_quality`, and set quality 95 only in the experiment launch script. Add a read-only ZMQ subscriber that measures raw msgpack bytes before unpacking and derives per-stream rates and timestamp latency while decoding every image on the PC. Run the same timed benchmark against quality 80 and 95, retaining machine-readable JSON and a concise comparison report.

**Tech Stack:** Python 3.10, OpenCV, NumPy, msgpack, pyzmq, tyro, pytest, zsh, Git worktrees, SSH.

## Global Constraints

- Work only on branch `experiment/jpeg-quality-95` in `.worktrees/jpeg-quality-95`.
- Preserve four `640x480@30` RGB streams, two lossless `uint16` depth streams, schema version 2, Base64 representation, timestamps, and camera metadata.
- Keep software JPEG quality at 80 unless `--jpeg-quality` explicitly overrides it; do not reinterpret OAK-only `--mjpeg-quality`.
- Measure each quality for 60 seconds after camera warm-up using the same PC, Sonic host, endpoint, decode behavior, and expected stream set.
- Report composed-message FPS, per-stream unique timestamp FPS, wire MiB/s and Mbit/s, images/s, payload size, latency mean/p50/p95, and PC decode mean/p50/p95.
- Back up every modified Sonic file, verify pre-deployment hashes match the isolated worktree baseline, and do not touch unrelated dirty files or processes.

---

### Task 1: Add deterministic PC stream statistics using TDD

**Files:**
- Create: `gear_sonic/scripts/benchmark_camera_stream.py`
- Create: `gear_sonic/tests/test_benchmark_camera_stream.py`

**Interfaces:**
- Consumes: raw msgpack messages with `timestamps` and encoded `images` from `tcp://192.168.123.164:5555`.
- Produces: `CameraStreamStats.add_message(packed: bytes, received_at: float, decode_ms: float) -> None` and `CameraStreamStats.summary(elapsed_s: float) -> dict[str, object]`.

- [x] **Step 1: Write failing tests for rate, throughput, and latency math**

Use two literal messages of 100 and 140 packed bytes received over 2 seconds, with two camera timestamps per message. Assert the hand-derived results:

```python
assert summary["messages"] == 2
assert summary["message_fps"] == 1.0
assert summary["wire_bytes_per_second"] == 120.0
assert summary["images_per_second"] == 2.0
assert summary["streams"]["ego_view"]["unique_fps"] == 1.0
assert summary["latency_ms"]["p50"] == 100.0
assert summary["decode_ms"]["mean"] == 3.0
```

The production change caught by this test is missing or incorrect denominator, unique-timestamp, percentile, or byte accounting.

- [x] **Step 2: Run the focused test and verify RED**

Run:

```bash
/home/user/Project/GR00T-WholeBodyControl/.venv_camera/bin/python -m pytest \
  gear_sonic/tests/test_benchmark_camera_stream.py -q
```

Expected: collection fails because `gear_sonic.scripts.benchmark_camera_stream` does not exist.

- [x] **Step 3: Implement the minimal statistics accumulator and subscriber**

Implement real msgpack parsing and JPEG/PNG decoding, a 5-second warm-up followed by a default 60-second measurement, and JSON output. The CLI contract is:

```text
python -m gear_sonic.scripts.benchmark_camera_stream \
  --host 192.168.123.164 --port 5555 \
  --warmup-seconds 5 --duration-seconds 60 \
  --label jpeg-quality-80 --output-json <path>
```

Percentiles use nearest-rank interpolation over stored samples. Latency is `received_at - camera_timestamp`; duplicate per-stream timestamps count toward messages but not `unique_fps`.

- [x] **Step 4: Verify GREEN and CLI syntax**

Run:

```bash
/home/user/Project/GR00T-WholeBodyControl/.venv_camera/bin/python -m pytest \
  gear_sonic/tests/test_benchmark_camera_stream.py -q
/home/user/Project/GR00T-WholeBodyControl/.venv_camera/bin/python \
  -m gear_sonic.scripts.benchmark_camera_stream --help
```

Expected: tests pass and help lists the endpoint, warm-up, duration, label, and JSON path options.

### Task 2: Make software JPEG quality configurable using TDD

**Files:**
- Modify: `gear_sonic/tests/test_camera_rgbd_protocol.py`
- Modify: `gear_sonic/camera/sensor_server.py`
- Modify: `gear_sonic/camera/composed_camera.py`
- Modify: `start_camera_server.zsh`

**Interfaces:**
- Consumes: `ComposedCameraConfig.jpeg_quality: int`, default 80.
- Produces: `ImageMessageSchema.serialize(executor: Executor | None = None, jpeg_quality: int = 80) -> dict[str, Any]` and `ImageUtils.encode_image(image: np.ndarray, quality: int = 80) -> str`.

- [x] **Step 1: Write failing real-encoder tests**

Create a deterministic noisy RGB image with `np.random.default_rng(0)`. Serialize it through a hardware-free `ComposedCameraSensor` at quality 80 and 95, then assert both payloads decode to the original shape and the quality-95 payload is larger. Also assert `ComposedCameraConfig(jpeg_quality=0)` and `jpeg_quality=101` raise `ValueError`.

The production changes caught by these tests are ignoring the composed configuration or accepting invalid encoder quality.

- [x] **Step 2: Run focused tests and verify RED**

Run:

```bash
/home/user/Project/GR00T-WholeBodyControl/.venv_camera/bin/python -m pytest \
  gear_sonic/tests/test_camera_rgbd_protocol.py -q
```

Expected: failures because `jpeg_quality` is not a composed config field and serialization always uses 80.

- [x] **Step 3: Implement the minimal quality path**

Pass `jpeg_quality` only for NumPy RGB images; retain raw JPEG bytes and lossless depth PNG behavior. Validate the inclusive range 1-100 in `ComposedCameraConfig.__post_init__`. Add this launch option:

```zsh
--jpeg-quality 95 \
```

- [x] **Step 4: Verify GREEN and camera regressions**

Run:

```bash
/home/user/Project/GR00T-WholeBodyControl/.venv_camera/bin/python -m pytest \
  gear_sonic/tests/test_camera_rgbd_protocol.py \
  gear_sonic/tests/test_orbbec_driver.py \
  gear_sonic/tests/test_benchmark_camera_stream.py -q
zsh -n start_camera_server.zsh
```

Expected: all selected tests pass and the launch script is valid zsh.

### Task 3: Run the quality-80 baseline and quality-95 experiment

**Files:**
- Create: `experiments/jpeg_quality_95/quality_80.json`
- Create: `experiments/jpeg_quality_95/quality_95.json`
- Create: `experiments/jpeg_quality_95/sonic_quality_80.log`
- Create: `experiments/jpeg_quality_95/sonic_quality_95.log`

**Interfaces:**
- Consumes: Sonic repository `/home/unitree/GR00T-WholeBodyControl` and PC benchmark from Task 1.
- Produces: two measurements with identical duration and stream expectations.

- [x] **Step 1: Start the existing quality-80 camera script on Sonic**

Confirm the five synchronized camera files still match the isolated baseline hashes. Start only `/home/unitree/GR00T-WholeBodyControl/start_camera_server.zsh`, log its output, wait for TCP port 5555 and all four cameras, and confirm no second composed-camera process exists.

- [x] **Step 2: Measure quality 80 from the PC**

Run the benchmark for 5 seconds warm-up plus 60 seconds measurement. Require all six keys (`ego_view`, `ego_view_depth`, `chest_view`, `chest_view_depth`, `left_wrist`, `right_wrist`), save `quality_80.json`, and copy the corresponding Sonic server log.

- [x] **Step 3: Back up and deploy the quality-95 experiment files**

Create `/home/unitree/GR00T-WholeBodyControl/Log/jpeg_quality_95_backup_<timestamp>/`, copy only `sensor_server.py`, `composed_camera.py`, and `start_camera_server.zsh`, upload replacements with `.codex-new` suffixes, compare SHA-256, then atomically rename. Stop only the identified composed-camera PID and restart through `start_camera_server.zsh`.

- [x] **Step 4: Measure quality 95 from the PC**

Repeat the exact 5-second warm-up plus 60-second measurement, save `quality_95.json`, and copy the server log. Verify all expected shapes, depth dtypes, per-stream activity, and absence of camera reconnect errors.

### Task 4: Analyze, verify, and commit the isolated experiment

**Files:**
- Create: `experiments/jpeg_quality_95/README.md`
- Include: files from Tasks 1-3.

**Interfaces:**
- Consumes: `quality_80.json`, `quality_95.json`, both server logs, and repository tests.
- Produces: a side-by-side table with absolute values and quality-95 percentage deltas.

- [x] **Step 1: Generate the comparison report**

Report wire rate, FPS, images/s, mean payload, mean/p50/p95 latency, mean/p50/p95 decode time, and all six unique timestamp rates. State the measurement duration, endpoint, clock synchronization status, image/depth shapes, and whether the 29 FPS target is met.

- [x] **Step 2: Run final verification**

Run:

```bash
/home/user/Project/GR00T-WholeBodyControl/.venv_camera/bin/python -m pytest \
  gear_sonic/tests/test_camera_rgbd_protocol.py \
  gear_sonic/tests/test_orbbec_driver.py \
  gear_sonic/tests/test_benchmark_camera_stream.py -q
zsh -n start_camera_server.zsh
git diff --check
```

Expected: all tests pass, zsh syntax is valid, and no whitespace errors are reported.

- [x] **Step 3: Commit the experiment branch**

```bash
git add docs/superpowers/plans/2026-08-13-jpeg-quality-95-stream-benchmark.md \
  gear_sonic/scripts/benchmark_camera_stream.py \
  gear_sonic/tests/test_benchmark_camera_stream.py \
  gear_sonic/tests/test_camera_rgbd_protocol.py \
  gear_sonic/camera/sensor_server.py \
  gear_sonic/camera/composed_camera.py \
  start_camera_server.zsh \
  experiments/jpeg_quality_95
git commit -m "test: benchmark camera stream at JPEG quality 95"
```
