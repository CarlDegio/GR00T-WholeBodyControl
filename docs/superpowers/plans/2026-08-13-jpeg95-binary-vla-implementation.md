# JPEG 95 Binary VLA Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** 将所有生产 JPEG/MJPEG 质量统一为 95，用 msgpack binary 传输相机 JPEG/PNG，并让 VLA 经 Gateway 复用相机 JPEG，使 OpenPI 只解码一次且不改变现有 VLA 时序。

**Architecture:** 相机保持单个 msgpack 消息，图像从 Base64 string 改为 raw bytes；Gateway 先发布 camera_encoded/*，再同步解码并发布原 camera/*。VLA 现有单 worker 仍按 camera→state 顺序轮询，只切换相机 stream 和 payload 表示；OpenPI 兼容新旧 marker。

**Tech Stack:** Python 3.11、NumPy、OpenCV、msgpack/msgpack-numpy、pyzmq、pytest、Ruff、zsh、SensorGateway shared memory。

## Global Constraints

- GR00T 只在 /home/user/Project/GR00T-WholeBodyControl/.worktrees/jpeg-quality-95 的 experiment/jpeg-quality-95 分支改动。
- OpenPI 使用 /home/user/Project/openpi_sonic/.worktrees/jpeg95-direct-camera-jpeg 和 experiment/jpeg95-direct-camera-jpeg；主工作区 replay_data/ 不移动、不删除。
- 所有生产 JPEG/MJPEG 质量为 95；PNG 深度继续无损编码。
- VLA 只访问 Gateway，不直接订阅相机 ZMQ。
- VlaSensorGatewayIngress._run、_request、_poll_state、_fresh 和 camera→state 顺序不得修改。
- 不改变 max_age_ms、max_skew_ms、sequence 去重和 retries=0；不增加时间比较、等待、重试、sleep、配对或调度分支。
- encoded adapter 的线格式分支替换原 HWC shape/dtype 校验，不叠加新的 readiness/freshness check。
- 不创建 VLA 新线程、解码线程池、新 socket 或新端口。
- 逐帧详细统计只在 benchmark 中执行，不进入生产热路径。
- 旧 Base64 相机消息和 __opencv_jpeg_rgb__ marker 继续可读。
- 每个任务先观察测试失败，再做最小实现、通过聚焦测试并提交。

---

### Task 1: OpenPI 向后兼容地支持相机原始 JPEG

**Files:**
- Create: /home/user/Project/openpi_sonic/.worktrees/jpeg95-direct-camera-jpeg/scripts/serve_g1_sonic_zmq_policy_test.py
- Modify: /home/user/Project/openpi_sonic/.worktrees/jpeg95-direct-camera-jpeg/scripts/serve_g1_sonic_zmq_policy.py:127-163

**Interfaces:**
- Consumes: {"__camera_jpeg_rgb__": True, "data": bytes} 和旧 marker dict。
- Produces: _decode_jpeg_rgb_video(value: Any, *, name: str) -> Any；新 marker 返回 [1, 1, H, W, 3] RGB uint8。

- [ ] **Step 1: 创建 OpenPI worktree**

    cd /home/user/Project/openpi_sonic
    git worktree add .worktrees/jpeg95-direct-camera-jpeg -b experiment/jpeg95-direct-camera-jpeg main
    git -C .worktrees/jpeg95-direct-camera-jpeg status --short --branch

Expected: 新 worktree/分支存在，原工作区仍只有未跟踪 replay_data/。

- [ ] **Step 2: 写新旧 marker 和 decode 次数测试**

    import cv2
    import numpy as np
    from scripts import serve_g1_sonic_zmq_policy as server

    def _jpeg(image_rgb):
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        assert ok
        return encoded.tobytes()

    def test_camera_marker_decodes_once_to_bt_rgb(monkeypatch):
        image = np.zeros((24, 32, 3), np.uint8)
        image[..., 0] = 240
        real = cv2.imdecode
        calls = []
        def counted(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)
        monkeypatch.setattr(server.cv2, "imdecode", counted)
        result = server._decode_jpeg_rgb_video(
            {server.CAMERA_JPEG_VIDEO_MARKER: True, "data": _jpeg(image)},
            name="video.ego_view",
        )
        assert result.shape == (1, 1, 24, 32, 3)
        assert result[0, 0, ..., 0].mean() > result[0, 0, ..., 2].mean()
        assert len(calls) == 1

    def test_legacy_marker_still_decodes():
        image = np.zeros((12, 16, 3), np.uint8)
        result = server._decode_jpeg_rgb_video(
            {server.JPEG_VIDEO_MARKER: True, "shape": (1, 1, 12, 16, 3),
             "dtype": "uint8", "data": _jpeg(image)},
            name="video.chest_view",
        )
        assert result.shape == (1, 1, 12, 16, 3)

- [ ] **Step 3: 确认新 marker 测试失败**

    cd /home/user/Project/openpi_sonic/.worktrees/jpeg95-direct-camera-jpeg
    /home/user/Project/openpi_sonic/.venv/bin/pytest scripts/serve_g1_sonic_zmq_policy_test.py -v

Expected: CAMERA_JPEG_VIDEO_MARKER 未定义导致 FAIL；legacy 用例通过。

- [ ] **Step 4: 实现共享 decode 和双 marker 分派**

    JPEG_VIDEO_MARKER = "__opencv_jpeg_rgb__"
    CAMERA_JPEG_VIDEO_MARKER = "__camera_jpeg_rgb__"

    def _decode_jpeg_bytes(encoded: bytes | bytearray, *, name: str) -> np.ndarray:
        image_bgr = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"cv2.imdecode failed for {name}")
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

新 marker 分支验证 data 为 bytes，调用一次 _decode_jpeg_bytes 后增加 [1, 1] 维。旧 marker 的 shape/dtype 分支改为调用同一 helper；gr00t_observation_to_openpi 的四个调用点不改。

    if isinstance(value, dict) and value.get(CAMERA_JPEG_VIDEO_MARKER):
        encoded = value.get("data")
        if not isinstance(encoded, bytes | bytearray):
            raise ValueError(f"{name} JPEG payload must contain bytes in 'data'")
        return _decode_jpeg_bytes(encoded, name=name)[np.newaxis, np.newaxis]

- [ ] **Step 5: 验证并提交**

    /home/user/Project/openpi_sonic/.venv/bin/pytest scripts/serve_g1_sonic_zmq_policy_test.py -v
    /home/user/Project/openpi_sonic/.venv/bin/ruff check scripts/serve_g1_sonic_zmq_policy.py scripts/serve_g1_sonic_zmq_policy_test.py
    git add scripts/serve_g1_sonic_zmq_policy.py scripts/serve_g1_sonic_zmq_policy_test.py
    git commit -m "feat: accept direct camera JPEG observations"

Expected: 两个测试和 Ruff 通过。

---

### Task 2: 统一生产 JPEG/MJPEG 质量为 95

**Files:**
- Create: gear_sonic/camera/constants.py
- Create: gear_sonic/tests/test_jpeg95_production_defaults.py
- Modify: gear_sonic/camera/sensor_server.py:121-135,275-282
- Modify: gear_sonic/camera/composed_camera.py:115-125
- Modify: gear_sonic/camera/drivers/oak.py:36-44,365-369
- Modify: gear_sonic/camera/gemini_server_launcher.py:86-101
- Modify: gear_sonic/runtime/visualization.py:22-29
- Modify: gear_sonic/navdp/gateway.py:70-82
- Modify: gear_sonic/utils/mujoco_sim/sensor_server.py:15-80
- Modify: docs/source/tutorials/data_collection.md:122
- Test: gear_sonic/tests/test_gemini_server_launcher.py
- Test: gear_sonic/tests/test_runtime_visualization.py
- Test: gear_sonic/tests/test_navdp_sensor_gateway.py

**Interfaces:**
- Produces: PRODUCTION_JPEG_QUALITY: Final[int] = 95。
- Consumes: 各生产默认配置/encoder；显式相机 CLI 参数仍可覆盖。

- [ ] **Step 1: 写生产默认值失败测试**

    import inspect
    from gear_sonic.camera.composed_camera import ComposedCameraConfig
    from gear_sonic.camera.constants import PRODUCTION_JPEG_QUALITY
    from gear_sonic.camera.sensor_server import ImageMessageSchema, ImageUtils
    from gear_sonic.runtime.visualization import VisualizationPublisher

    def test_production_jpeg_defaults_are_95():
        assert PRODUCTION_JPEG_QUALITY == 95
        assert ComposedCameraConfig().jpeg_quality == 95
        assert ComposedCameraConfig().mjpeg_quality == 95
        assert inspect.signature(ImageMessageSchema.serialize).parameters["jpeg_quality"].default == 95
        assert inspect.signature(ImageUtils.encode_image).parameters["quality"].default == 95
        publisher = VisualizationPublisher("inproc://jpeg-quality-default")
        try:
            assert publisher.jpeg_quality == 95
        finally:
            publisher.close()

Gemini argv 预期加入 "--jpeg-quality", "95"；NavDP 用 monkeypatch 断言 JPEG imencode 参数为 [cv2.IMWRITE_JPEG_QUALITY, 95]。

- [ ] **Step 2: 确认默认值测试失败**

    cd /home/user/Project/GR00T-WholeBodyControl/.worktrees/jpeg-quality-95
    /home/user/Project/GR00T-WholeBodyControl/.venv_teleop/bin/python -m pytest \
      gear_sonic/tests/test_jpeg95_production_defaults.py \
      gear_sonic/tests/test_gemini_server_launcher.py \
      gear_sonic/tests/test_runtime_visualization.py \
      gear_sonic/tests/test_navdp_sensor_gateway.py -v

Expected: 常量缺失或 80/85 默认值导致 FAIL。

- [ ] **Step 3: 添加常量并替换生产默认值**

    from typing import Final
    PRODUCTION_JPEG_QUALITY: Final[int] = 95

schema、组合相机、OAK 配置/CLI、可视化、NavDP、MuJoCo 使用该值；Gemini argv 显式加入 --jpeg-quality 95；教程写明生产默认 95。start_camera_server.zsh 已为 95，保持不变。

- [ ] **Step 4: 验证并提交**

    /home/user/Project/GR00T-WholeBodyControl/.venv_teleop/bin/python -m pytest \
      gear_sonic/tests/test_jpeg95_production_defaults.py \
      gear_sonic/tests/test_gemini_server_launcher.py \
      gear_sonic/tests/test_runtime_visualization.py \
      gear_sonic/tests/test_navdp_sensor_gateway.py -v
    git add gear_sonic/camera/constants.py gear_sonic/camera/sensor_server.py \
      gear_sonic/camera/composed_camera.py gear_sonic/camera/drivers/oak.py \
      gear_sonic/camera/gemini_server_launcher.py gear_sonic/runtime/visualization.py \
      gear_sonic/navdp/gateway.py gear_sonic/utils/mujoco_sim/sensor_server.py \
      gear_sonic/tests/test_jpeg95_production_defaults.py \
      gear_sonic/tests/test_gemini_server_launcher.py \
      gear_sonic/tests/test_runtime_visualization.py \
      gear_sonic/tests/test_navdp_sensor_gateway.py \
      docs/source/tutorials/data_collection.md
    git commit -m "feat: use JPEG quality 95 in production"

Expected: 聚焦测试全过；benchmark 中显式 80/95 对照值没有被统一替换。

---

### Task 3: 相机消息使用 raw binary 和 latest-first sender

**Files:**
- Modify: gear_sonic/camera/sensor_server.py:107-180,205-229,273-300
- Modify: gear_sonic/utils/mujoco_sim/sensor_server.py:15-80
- Test: gear_sonic/tests/test_camera_rgbd_protocol.py
- Test: gear_sonic/tests/test_jpeg95_production_defaults.py

**Interfaces:**
- Produces: ImageUtils.encode_image(...) -> bytes、encode_depth_image(...) -> bytes、CAMERA_SEND_HWM = 1。
- Preserves: ImageMessageSchema.deserialize 接受 bytes 和旧 Base64 string。

- [ ] **Step 1: 写 binary、颜色和兼容失败测试**

    def test_schema_emits_binary_rgb_depth_and_preserves_rgb_order():
        rgb = np.zeros((48, 64, 3), np.uint8)
        rgb[..., 0] = 240
        depth = np.arange(48 * 64, dtype=np.uint16).reshape(48, 64)
        schema = ImageMessageSchema(
            timestamps={"ego_view": 1.0, "ego_view_depth": 1.0},
            images={"ego_view": rgb, "ego_view_depth": depth},
        )
        wire = schema.serialize()
        assert isinstance(wire["images"]["ego_view"], bytes)
        assert isinstance(wire["images"]["ego_view_depth"], bytes)
        decoded = ImageMessageSchema.deserialize(wire)
        assert decoded.images["ego_view"][..., 0].mean() > decoded.images["ego_view"][..., 2].mean()
        np.testing.assert_array_equal(decoded.images["ego_view_depth"], depth)

另一个测试将这两个 bytes 转成 Base64 ASCII string 后 deserialize，断言旧 RGB/深度仍可解码；同时断言 CAMERA_SEND_HWM == 1。

- [ ] **Step 2: 确认当前 producer 类型/HWM 测试失败**

    /home/user/Project/GR00T-WholeBodyControl/.venv_teleop/bin/python -m pytest \
      gear_sonic/tests/test_camera_rgbd_protocol.py \
      gear_sonic/tests/test_jpeg95_production_defaults.py -v

Expected: binary 类型或 HWM 断言 FAIL。

- [ ] **Step 3: 实现 bytes encoder 与 RGB/BGR 边界**

    CAMERA_SEND_HWM = 1

    def encode_image(image: np.ndarray, quality: int = PRODUCTION_JPEG_QUALITY) -> bytes:
        image_bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(
            ".jpg", image_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
        )
        if not ok:
            raise RuntimeError("failed to encode RGB image as JPEG")
        return encoded.tobytes()

深度 PNG 成功后直接返回 bytes；旧 decode_image/decode_depth_image 保留。主相机与 MuJoCo sender 使用 SNDHWM=1。

- [ ] **Step 4: 验证并提交**

    /home/user/Project/GR00T-WholeBodyControl/.venv_teleop/bin/python -m pytest \
      gear_sonic/tests/test_camera_rgbd_protocol.py \
      gear_sonic/tests/test_camera_viewer.py \
      gear_sonic/tests/test_runtime_fake_services.py \
      gear_sonic/tests/test_orbbec_driver.py \
      gear_sonic/tests/test_jpeg95_production_defaults.py -v
    git add gear_sonic/camera/sensor_server.py gear_sonic/utils/mujoco_sim/sensor_server.py \
      gear_sonic/tests/test_camera_rgbd_protocol.py gear_sonic/tests/test_jpeg95_production_defaults.py
    git commit -m "perf: send camera images as msgpack binary"

Expected: 测试全过，quality 95 bytes 大于 quality 80 bytes，RGB 通道正确。

---

### Task 4: Gateway encoded-first 发布

**Files:**
- Modify: gear_sonic/runtime/sensor_gateway.py:332-415
- Test: gear_sonic/tests/test_sensor_gateway.py:223-307

**Interfaces:**
- Produces: camera_encoded/{name} 一维 uint8，encoding 为 jpeg_bytes 或 base64_jpeg。
- Preserves: camera/{name}、return count、时间戳、序号和现有 RPC thread。

- [ ] **Step 1: 写发布顺序失败测试**

monkeypatch core.publish_array 记录 stream，发送四 RGB/两 depth binary 消息：

    encoded_positions = [
        publish_order.index(f"camera_encoded/{name}")
        for name in ("ego_view", "chest_view", "left_wrist", "right_wrist")
    ]
    decoded_positions = [
        index for index, stream in enumerate(publish_order)
        if stream.startswith("camera/")
    ]
    assert max(encoded_positions) < min(decoded_positions)
    assert encoded_frame.attributes["encoding"] == "jpeg_bytes"

再手工发送 Base64 JPEG，断言 encoded attribute 是 base64_jpeg。

- [ ] **Step 2: 确认当前发布顺序失败**

    /home/user/Project/GR00T-WholeBodyControl/.venv_teleop/bin/python -m pytest \
      gear_sonic/tests/test_sensor_gateway.py::test_camera_and_cpp_ingress_are_read_only_copies_of_current_wires -v

Expected: 当前先 decode、后 encoded，顺序断言 FAIL。

- [ ] **Step 3: 重排为两个同步循环**

    encoded_schema = ImageMessageSchema.deserialize(payload, decode_images=False)
    for name, encoded in encoded_schema.images.items():
        if name.endswith("_depth") or not isinstance(encoded, bytes | bytearray | str):
            continue
        encoded_array = np.frombuffer(
            encoded.encode("utf-8") if isinstance(encoded, str) else encoded,
            dtype=np.uint8,
        )
        wire_encoding = "base64_jpeg" if isinstance(encoded, str) else "jpeg_bytes"
        self.core.publish_array(
            f"camera_encoded/{name}", encoded_array,
            received_ns=received_ns,
            source_timestamp_ns=max(0, int(float(encoded_schema.timestamps.get(name, 0.0)) * 1e9)),
            source_clock="camera_unix" if encoded_schema.timestamps.get(name, 0.0) else "unknown",
            expected_hz=self.expected_hz,
            attributes={"encoding": wire_encoding, "decoded_color_order": "RGB",
                        "camera_info": dict(encoded_schema.camera_info.get(name, {}))},
        )

    schema = ImageMessageSchema.deserialize(payload)
    for name, image in schema.images.items():
        if not isinstance(image, np.ndarray):
            continue
        timestamp_s = float(schema.timestamps.get(name, 0.0))
        base_name = name.removesuffix("_depth")
        attributes = {
            "encoding": "numpy",
            "camera_info": dict(schema.camera_info.get(base_name, {})),
        }
        if not name.endswith("_depth"):
            attributes["color_order"] = "RGB"
        self.core.publish_array(
            f"camera/{name}", image, received_ns=received_ns,
            source_timestamp_ns=max(0, int(timestamp_s * 1e9)),
            source_clock="camera_unix" if timestamp_s > 0.0 else "unknown",
            expected_hz=self.expected_hz, attributes=attributes,
        )
        count += 1

两个循环复用 received_ns、source timestamp 和 camera_info。不要修改 SensorGatewayRpcServer，不增加 thread、queue、snapshot 或时间条件。

- [ ] **Step 4: 验证并提交**

    /home/user/Project/GR00T-WholeBodyControl/.venv_teleop/bin/python -m pytest \
      gear_sonic/tests/test_sensor_gateway.py \
      gear_sonic/tests/test_sensor_gateway_client.py \
      gear_sonic/tests/test_run_sensor_gateway.py -v
    git add gear_sonic/runtime/sensor_gateway.py gear_sonic/tests/test_sensor_gateway.py
    git commit -m "perf: publish encoded camera frames before decode"

Expected: Gateway 测试全过；production 仍只有已有 ingress thread 和 RPC thread。

---

### Task 5: VLA 原 worker 改读 encoded 侧链

**Files:**
- Modify: gear_sonic/runtime/vla_sensor_gateway.py:17-67,140-149
- Test: gear_sonic/tests/test_vla_sensor_gateway.py

**Interfaces:**
- Produces: VLA_CAMERA_STREAMS 为四个 camera_encoded/*。
- Produces: camera_message_from_snapshot 中 images 为 dict[str, bytes]。
- Preserves: 原 _run/_request/_poll_state/_fresh、age/skew/sequence 和轮询顺序。

- [ ] **Step 1: 把 fixture 改成 encoded arrays 并写 Base64 兼容测试**

    arrays = {
        f"camera_encoded/{name}": np.frombuffer(payload, np.uint8).copy()
        for name, payload in jpeg_payloads.items()
    }

SharedMemoryFrame attributes 设置 encoding="jpeg_bytes" 和原 camera_info。断言四个 stream 名及输出 bytes 完全一致。另一个 fixture 放 Base64 ASCII uint8、encoding="base64_jpeg"，断言输出还原为原 JPEG bytes，而不是 RGB。

- [ ] **Step 2: 确认当前 HWC 校验失败**

    /home/user/Project/GR00T-WholeBodyControl/.venv_teleop/bin/python -m pytest \
      gear_sonic/tests/test_vla_sensor_gateway.py -v

Expected: 当前请求 camera/* 且要求 HxWx3，测试 FAIL。

- [ ] **Step 3: 只修改 stream 常量和 payload helper**

    import base64
    VLA_CAMERA_STREAMS = tuple(f"camera_encoded/{name}" for name in VLA_CAMERA_NAMES)

    payload = np.asarray(snapshot.arrays[stream], dtype=np.uint8).reshape(-1).tobytes()
    encoding = frame.attributes.get("encoding")
    if encoding == "base64_jpeg":
        payload = base64.b64decode(payload)
    elif encoding != "jpeg_bytes":
        raise ValueError(f"VLA camera {name!r} has unsupported encoding {encoding!r}")
    images[name] = payload

encoding 分支替换原 HWC shape/dtype 判断，只做线格式归一，不做时间判断。timestamp/camera_info 复制代码保持原样。

- [ ] **Step 4: 验证、审计限定 diff 并提交**

    /home/user/Project/GR00T-WholeBodyControl/.venv_teleop/bin/python -m pytest \
      gear_sonic/tests/test_vla_sensor_gateway.py -v
    git diff c90d208 -- gear_sonic/runtime/vla_sensor_gateway.py
    git add gear_sonic/runtime/vla_sensor_gateway.py gear_sonic/tests/test_vla_sensor_gateway.py
    git commit -m "perf: feed VLA encoded camera snapshots"

Expected: 测试通过；diff 没有 _run、_request、_poll_state、_fresh hunk。

---

### Task 6: VLA 请求直接封装 JPEG

**Files:**
- Modify: gear_sonic/scripts/run_vla_inference.py:336-385,471-535
- Modify: gear_sonic/runtime/vla_timing.py:12-25
- Test: gear_sonic/tests/test_run_vla_inference_delay.py
- Test: gear_sonic/tests/test_vla_timing.py

**Interfaces:**
- Produces: CAMERA_JPEG_VIDEO_MARKER = "__camera_jpeg_rgb__"。
- Produces: wrap_camera_jpeg_for_video(encoded) -> marker/data dict。
- Produces: timing 名 jpeg_prepare；移除生产 jpeg_encode timing。

- [ ] **Step 1: 写零 codec 封装和 timing 失败测试**

    def test_camera_jpeg_wrapper_keeps_bytes_without_codec():
        payload = b"already-encoded-camera-jpeg"
        assert wrap_camera_jpeg_for_video(payload) == {
            CAMERA_JPEG_VIDEO_MARKER: True,
            "data": payload,
        }

    def test_timing_uses_jpeg_prepare():
        window = VlaTimingWindow(window_size=2)
        window.record({"jpeg_prepare": 0.05, "worker_total": 1.0})
        snapshot = window.snapshot()
        assert snapshot["segments_ms"]["jpeg_prepare"]["last"] == 0.05
        assert "jpeg_encode" not in snapshot["segments_ms"]

prepare_observation 测试让 fake Gateway 返回四个故意不可解码的 byte strings，并 monkeypatch prepare_observation_for_eval 为 identity；成功构建 observation 即证明本端没有 decode/re-encode。

- [ ] **Step 2: 确认 helper/timing 缺失而失败**

    /home/user/Project/GR00T-WholeBodyControl/.venv_teleop/bin/python -m pytest \
      gear_sonic/tests/test_run_vla_inference_delay.py \
      gear_sonic/tests/test_vla_timing.py -v

Expected: wrapper 和 jpeg_prepare 尚不存在，测试 FAIL。

- [ ] **Step 3: 删除二次编码并封装 bytes**

    CAMERA_JPEG_VIDEO_MARKER = "__camera_jpeg_rgb__"

    def wrap_camera_jpeg_for_video(encoded: bytes | bytearray | memoryview) -> dict[str, Any]:
        return {CAMERA_JPEG_VIDEO_MARKER: True, "data": bytes(encoded)}

video comprehension 直接调用该 helper，timing key 改为 jpeg_prepare。删除 JPEG_VIDEO_QUALITY、旧 encode helper 和只供它使用的 cv2 import；VLA_TIMING_SEGMENTS 同步更名。

- [ ] **Step 4: 验证、静态检查并提交**

    /home/user/Project/GR00T-WholeBodyControl/.venv_teleop/bin/python -m pytest \
      gear_sonic/tests/test_run_vla_inference_delay.py \
      gear_sonic/tests/test_vla_timing.py -v
    rg -n "cv2\.imencode|jpeg_encode|JPEG_VIDEO_QUALITY" \
      gear_sonic/scripts/run_vla_inference.py gear_sonic/runtime/vla_timing.py
    git add gear_sonic/scripts/run_vla_inference.py gear_sonic/runtime/vla_timing.py \
      gear_sonic/tests/test_run_vla_inference_delay.py gear_sonic/tests/test_vla_timing.py
    git commit -m "perf: remove VLA camera JPEG re-encoding"

Expected: pytest 通过；rg 无输出。

---

### Task 7: 独立测量延迟、FPS 和吞吐

**Files:**
- Create: gear_sonic/scripts/benchmark_vla_jpeg_pipeline.py
- Create: gear_sonic/tests/test_benchmark_vla_jpeg_pipeline.py
- Modify: gear_sonic/scripts/benchmark_camera_stream.py:35-126
- Modify: gear_sonic/tests/test_benchmark_camera_stream.py

**Interfaces:**
- Produces: camera JSON 的 actual wire、同包 Base64 counterfactual、binary savings。
- Produces: VLA JSON 的 encoded/decoded unique FPS、images/s、bytes/request、Mbit/s、Gateway RPC、prepare、pack、OpenPI 顺序 decode 的 mean/P50/P95。
- Consumes: SensorGateway IPC 和 OpenPI worktree path。

- [ ] **Step 1: 写 stats 失败测试**

    def test_binary_stats_report_base64_counterfactual():
        message = {
            "timestamps": {"ego_view": 10.0},
            "images": {"ego_view": b"123456", "ego_view_depth": b"abcdefghi"},
        }
        packed = msgpack.packb(message, use_bin_type=True)
        stats = CameraStreamStats()
        stats.add_message(packed, received_at=10.1, decode_ms=1.0)
        summary = stats.summary(1.0)
        assert summary["binary_images"] == 2
        assert summary["base64_counterfactual_wire_bytes"] > summary["wire_bytes"]
        assert summary["binary_wire_savings_percent"] > 0.0

    def test_vla_stats_report_rate_throughput_latency():
        stats = VlaPipelineStats()
        stats.record(
            encoded_sequences={"ego_view": 1, "chest_view": 2,
                               "left_wrist": 3, "right_wrist": 4},
            decoded_sequences={"ego_view": 5, "ego_view_depth": 6},
            request_bytes=1000, camera_rpc_ms=1.0,
            jpeg_prepare_ms=0.1, request_pack_ms=0.2,
            openpi_decode_ms=4.0,
        )
        summary = stats.summary(2.0)
        assert summary["vla_request_fps"] == 0.5
        assert summary["vla_mbit_per_second"] == 0.004
        assert summary["latency_ms"]["openpi_decode"]["p50"] == 4.0

- [ ] **Step 2: 确认统计字段/类缺失而失败**

    /home/user/Project/GR00T-WholeBodyControl/.venv_teleop/bin/python -m pytest \
      gear_sonic/tests/test_benchmark_camera_stream.py \
      gear_sonic/tests/test_benchmark_vla_jpeg_pipeline.py -v

Expected: counterfactual 字段和 VlaPipelineStats 缺失导致 FAIL。

- [ ] **Step 3: 实现同包 Base64 反事实**

    def _base64_counterfactual_size(message):
        value = dict(message)
        value["images"] = {
            name: base64.b64encode(data).decode("ascii")
            if isinstance(data, bytes | bytearray) else data
            for name, data in message.get("images", {}).items()
        }
        return len(msgpack.packb(value, use_bin_type=True))

CameraStreamStats 对同一个 unpacked message 累计 actual 和 counterfactual，避免场景变化污染 25% 结论。

- [ ] **Step 4: 实现单线程 Gateway/VLA/OpenPI benchmark**

每个循环按以下固定顺序执行，不修改 production ingress：

    encoded = client.read_snapshot(
        SnapshotRequest(streams=VLA_CAMERA_STREAMS,
                        max_age_ms=1000.0, max_skew_ms=5.0),
        retries=0,
    )
    camera = camera_message_from_snapshot(encoded)
    video = {
        name: wrap_camera_jpeg_for_video(camera["images"][name])
        for name in VLA_CAMERA_NAMES
    }
    packed = msgpack_numpy.packb(
        {"endpoint": "get_action", "data": {"observation": {"video": video}}}
    )
    decoded = [
        openpi_module._decode_jpeg_rgb_video(video[name], name=f"video.{name}")
        for name in VLA_CAMERA_NAMES
    ]

随后同线程读取六路 camera/* 只统计 sequence/FPS。所有阶段用 perf_counter_ns，重复 encoded sequence 不重复计 codec；同时对同一 JPEG 运行旧 RGB re-encode 微基准，输出 old/new codec P50/P95。

- [ ] **Step 5: 验证并提交 benchmark**

    /home/user/Project/GR00T-WholeBodyControl/.venv_teleop/bin/python -m pytest \
      gear_sonic/tests/test_benchmark_camera_stream.py \
      gear_sonic/tests/test_benchmark_vla_jpeg_pipeline.py -v
    /home/user/Project/GR00T-WholeBodyControl/.venv_teleop/bin/python -m ruff check \
      gear_sonic/scripts/benchmark_camera_stream.py \
      gear_sonic/scripts/benchmark_vla_jpeg_pipeline.py \
      gear_sonic/tests/test_benchmark_camera_stream.py \
      gear_sonic/tests/test_benchmark_vla_jpeg_pipeline.py
    git add gear_sonic/scripts/benchmark_camera_stream.py \
      gear_sonic/scripts/benchmark_vla_jpeg_pipeline.py \
      gear_sonic/tests/test_benchmark_camera_stream.py \
      gear_sonic/tests/test_benchmark_vla_jpeg_pipeline.py
    git commit -m "test: benchmark binary camera and VLA JPEG pipeline"

Expected: pytest/Ruff 通过，benchmark 未修改任何 production 调度代码。

---

### Task 8: 全量验证与 60 秒现场验收

**Files:**
- Create: experiments/jpeg95_binary_vla/camera_binary_60s.json
- Create: experiments/jpeg95_binary_vla/vla_pipeline_60s.json
- Create: experiments/jpeg95_binary_vla/check_acceptance.py
- Create: experiments/jpeg95_binary_vla/README.md
- Verify: 两个仓库的改动文件。

**Acceptance:**
- camera/Gateway message FPS ≥ 29。
- 每路 encoded FPS ≥ 29；正常六路 decoded 合计 ≥ 174 images/s。
- 同包 binary 相比 Base64 反事实至少减少 20%。
- jpeg_prepare P95 < 1 ms；四 JPEG 顺序 decode P95 ≤ 6 ms。
- 新 codec P50/P95 均小于旧二次编码路径。
- VLA 时序函数无修改。

- [ ] **Step 1: 运行两仓测试、Ruff 和 diff 检查**

    cd /home/user/Project/GR00T-WholeBodyControl/.worktrees/jpeg-quality-95
    /home/user/Project/GR00T-WholeBodyControl/.venv_teleop/bin/python -m pytest gear_sonic/tests -q
    /home/user/Project/GR00T-WholeBodyControl/.venv_teleop/bin/python -m ruff check \
      gear_sonic/camera gear_sonic/runtime gear_sonic/navdp \
      gear_sonic/scripts/benchmark_camera_stream.py \
      gear_sonic/scripts/benchmark_vla_jpeg_pipeline.py
    git diff --check c90d208..HEAD

    cd /home/user/Project/openpi_sonic/.worktrees/jpeg95-direct-camera-jpeg
    /home/user/Project/openpi_sonic/.venv/bin/pytest scripts/serve_g1_sonic_zmq_policy_test.py -q
    /home/user/Project/openpi_sonic/.venv/bin/ruff check scripts/serve_g1_sonic_zmq_policy.py scripts/serve_g1_sonic_zmq_policy_test.py
    git diff --check main..HEAD

Expected: pytest 零失败/错误，Ruff 通过，diff check 无输出。

- [ ] **Step 2: 审计无新增 VLA 时序逻辑**

    cd /home/user/Project/GR00T-WholeBodyControl/.worktrees/jpeg-quality-95
    git diff c90d208..HEAD -- gear_sonic/runtime/vla_sensor_gateway.py
    rg -n "camera_encoded|jpeg_prepare|__camera_jpeg_rgb__" \
      gear_sonic/runtime/vla_sensor_gateway.py \
      gear_sonic/scripts/run_vla_inference.py \
      gear_sonic/runtime/vla_timing.py

Expected: vla_sensor_gateway diff 不含 _run、_request、_poll_state、_fresh hunk，也没有新 thread、sleep、重试、时间比较、等待或配对。

- [ ] **Step 3: 按兼容顺序部署协议**

1. 先在 OpenPI worktree 运行新旧 marker 协议测试，不加载模型。
2. 再用 GR00T worktree 启动 PC SensorGateway 和协议 benchmark。
3. 最后在 Sonic 的 /home/unitree/GR00T-WholeBodyControl 做文件级日期备份、SHA-256 校验和原子替换。
4. 只停止命令行为 gear_sonic.camera.composed_camera 的已确认 PID，再用 start_camera_server.zsh 启动，确认命令行含 --jpeg-quality 95。

Expected: Sonic 切 binary 前所有接收端已经兼容 binary/Base64。

- [ ] **Step 4: 运行相机 60 秒测试**

    cd /home/user/Project/GR00T-WholeBodyControl/.worktrees/jpeg-quality-95
    /home/user/Project/GR00T-WholeBodyControl/.venv_teleop/bin/python \
      gear_sonic/scripts/benchmark_camera_stream.py \
      --host 192.168.123.164 --port 5555 \
      --warmup-seconds 5 --duration-seconds 60 \
      --label jpeg95-binary \
      --output-json experiments/jpeg95_binary_vla/camera_binary_60s.json

Expected: 六路存在、message FPS ≥ 29、所有图像为 binary、反事实 savings ≥ 20%。

- [ ] **Step 5: 运行 Gateway/VLA/OpenPI 协议 60 秒测试**

    /home/user/Project/GR00T-WholeBodyControl/.venv_teleop/bin/python \
      gear_sonic/scripts/benchmark_vla_jpeg_pipeline.py \
      --sensor-gateway-endpoint ipc:///tmp/sonic_sensor_gateway.ipc \
      --openpi-repo /home/user/Project/openpi_sonic/.worktrees/jpeg95-direct-camera-jpeg \
      --duration-seconds 60 \
      --output-json experiments/jpeg95_binary_vla/vla_pipeline_60s.json

Expected: encoded/decoded FPS、请求吞吐和各段 P50/P95 完整；不包含模型推理时间。

- [ ] **Step 6: 机器检查验收阈值**

    import json
    from pathlib import Path

    root = Path("experiments/jpeg95_binary_vla")
    camera = json.loads((root / "camera_binary_60s.json").read_text())
    vla = json.loads((root / "vla_pipeline_60s.json").read_text())
    assert camera["message_fps"] >= 29.0
    assert camera["binary_wire_savings_percent"] >= 20.0
    assert min(x["unique_fps"] for x in vla["encoded_streams"].values()) >= 29.0
    assert vla["decoded_images_per_second"] >= 174.0
    assert vla["latency_ms"]["jpeg_prepare"]["p95"] < 1.0
    assert vla["latency_ms"]["openpi_decode"]["p95"] <= 6.0
    assert vla["codec_comparison_ms"]["new_p50"] < vla["codec_comparison_ms"]["old_p50"]
    assert vla["codec_comparison_ms"]["new_p95"] < vla["codec_comparison_ms"]["old_p95"]

Run:

    /home/user/Project/GR00T-WholeBodyControl/.venv_teleop/bin/python \
      experiments/jpeg95_binary_vla/check_acceptance.py

Expected: exit code 0。该 check_acceptance.py 使用上面的完整代码并随报告提交；失败时保留 JSON 且不宣称优化完成。

- [ ] **Step 7: 写报告并提交结果**

README 必须并列 JPEG 80、JPEG 95 Base64、JPEG 95 binary 的 FPS/Mbit/s/latency，列出 encoded/decoded FPS、images/s、drop/stale、old/new codec 和 OpenPI decode P50/P95；明确延迟下降来自删除 Base64、encoded-first、删除 VLA imencode 和 SNDHWM=1；明确 VLA 时序函数未修改。

    git add docs/superpowers/specs/2026-08-13-jpeg95-binary-vla-design.md \
      docs/superpowers/plans/2026-08-13-jpeg95-binary-vla-implementation.md \
      experiments/jpeg95_binary_vla
    git commit -m "docs: report binary JPEG VLA validation"
