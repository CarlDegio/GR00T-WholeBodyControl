# PICO SensorGateway Video Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a workstation service that reads `camera_encoded/ego_view` exclusively through SensorGateway and displays it in XRoboToolkit Remote Vision on PICO, with a robot-free local mock path.

**Architecture:** A bounded latest-frame reader materializes JPEG bytes from SensorGateway shared memory, decodes and duplicates the mono head image into a side-by-side frame, and feeds raw RGB to a low-latency FFmpeg H.264 subprocess. A TCP control server implements the XRoboToolkit `OPEN_CAMERA`/`CLOSE_CAMERA` protocol and starts one video session that sends length-prefixed Annex-B access units to the headset. A mock camera publisher and a fake PICO receiver make the entire chain testable on the workstation.

**Tech Stack:** Python 3.10+, NumPy, OpenCV, pyzmq/msgpack, existing `gear_sonic.runtime` SensorGateway APIs, FFmpeg (`h264_nvenc` in production and `libx264` as a portable test fallback), pytest.

## Global Constraints

- All camera input to the bridge MUST use `SensorGatewayClient`; the bridge must never subscribe directly to camera ZMQ.
- The production stream is `camera_encoded/ego_view`, whose shared-memory array contains JPEG bytes and whose metadata has `encoding=jpeg_bytes`.
- Output is mono duplicated SBS at 1280x480, 30 FPS, H.264 Annex B, one four-byte big-endian length prefix per access unit.
- The control protocol has an outer four-byte big-endian body length and an inner little-endian command/data envelope.
- Keep at most one pending decoded frame and one active PICO video session. Prefer current frames over preserving old ones.
- Never silently freeze the last camera frame. Missing, invalid, or stale input renders a conspicuous status card at 2 FPS.
- Do not run a test command until the user has received the requested pre-test reminder. PICO and robot/hardware tests require a separate explicit confirmation.
- In every command below, `python` means `/home/user/Project/GR00T-WholeBodyControl/.venv_teleop/bin/python`; the isolated worktree intentionally does not contain a second virtual environment.
- Preserve the original checkout's modified `gear_sonic/scripts/run_data_exporter.py` and untracked `Log/`; all work stays on `feature/pico-sensorgateway-video` in `.worktrees/pico-sensorgateway-video`.

---

### Task 1: Implement the XRoboToolkit wire protocols

**Files:**

- Create: `gear_sonic/pico_video/__init__.py`
- Create: `gear_sonic/pico_video/protocol.py`
- Create: `gear_sonic/tests/test_pico_video_protocol.py`

- [ ] **Step 1: Write failing protocol tests**

Cover the actual vendor byte layout with hand-derived fixtures, including fragmented and coalesced TCP input:

```python
import struct

import pytest

from gear_sonic.pico_video.protocol import (
    CameraRequest,
    ControlFrameDecoder,
    ProtocolError,
    frame_video_access_unit,
    parse_camera_request,
)


def _control_packet(command: bytes, data: bytes) -> bytes:
    body = struct.pack("<I", len(command)) + command
    body += struct.pack("<I", len(data)) + data
    return struct.pack(">I", len(body)) + body


def test_control_decoder_handles_fragmented_and_coalesced_packets() -> None:
    open_packet = _control_packet(b"OPEN_CAMERA", b"abc")
    close_packet = _control_packet(b"CLOSE_CAMERA", b"")
    decoder = ControlFrameDecoder()

    assert decoder.feed(open_packet[:3]) == ()
    assert decoder.feed(open_packet[3:] + close_packet) == (
        ("OPEN_CAMERA", b"abc"),
        ("CLOSE_CAMERA", b""),
    )


def test_camera_request_matches_vendor_little_endian_payload() -> None:
    payload = b"\xca\xfe\x01" + struct.pack(
        "<7i", 1280, 480, 30, 4_000_000, 0, 0, 12345
    ) + b"\x03ZED\x0b192.0.2.10"

    assert parse_camera_request(payload) == CameraRequest(
        width=1280,
        height=480,
        fps=30,
        bitrate=4_000_000,
        enable_mv_hevc=False,
        render_mode=0,
        port=12345,
        camera="ZED",
        ip="192.0.2.10",
    )


@pytest.mark.parametrize("payload", [b"", b"\xca\xfe\x02", b"\xca\xfe\x01" + b"\0" * 28])
def test_camera_request_rejects_truncated_or_unsupported_payload(payload: bytes) -> None:
    with pytest.raises(ProtocolError):
        parse_camera_request(payload)


def test_video_access_unit_uses_big_endian_length_prefix() -> None:
    assert frame_video_access_unit(b"\x00\x00\x00\x01\x09\xf0") == (
        b"\x00\x00\x00\x06\x00\x00\x00\x01\x09\xf0"
    )
```

- [ ] **Step 2: Run the focused test and observe the expected import failure**

Run: `python -m pytest -q gear_sonic/tests/test_pico_video_protocol.py`

Expected: FAIL because `gear_sonic.pico_video.protocol` does not exist.

- [ ] **Step 3: Implement the minimal validated parser**

Use bounded buffering and explicit byte order; reject oversized bodies before allocating them:

```python
class ControlFrameDecoder:
    def __init__(self, *, max_body_bytes: int = 2 * 1024 * 1024) -> None:
        self._buffer = bytearray()
        self._max_body_bytes = max_body_bytes

    def feed(self, chunk: bytes) -> tuple[tuple[str, bytes], ...]:
        self._buffer.extend(chunk)
        messages: list[tuple[str, bytes]] = []
        while len(self._buffer) >= 4:
            body_size = struct.unpack_from(">I", self._buffer)[0]
            if body_size == 0 or body_size > self._max_body_bytes:
                raise ProtocolError(f"invalid control body size: {body_size}")
            if len(self._buffer) < 4 + body_size:
                break
            body = bytes(self._buffer[4 : 4 + body_size])
            del self._buffer[: 4 + body_size]
            messages.append(_parse_control_body(body))
        return tuple(messages)
```

`parse_camera_request()` must validate magic `CA FE`, version `1`, positive dimensions/FPS/bitrate, TCP port range, complete compact strings, no trailing bytes, a valid IP literal, and `camera == "ZED"`. `frame_video_access_unit()` rejects empty payloads and returns `struct.pack(">I", len(payload)) + payload`.

- [ ] **Step 4: Run focused tests and static compilation**

Run: `python -m pytest -q gear_sonic/tests/test_pico_video_protocol.py && python -m compileall -q gear_sonic/pico_video`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gear_sonic/pico_video gear_sonic/tests/test_pico_video_protocol.py
git commit -m "feat: implement PICO video wire protocol"
```

---

### Task 2: Build deterministic camera and status frames

**Files:**

- Create: `gear_sonic/pico_video/frames.py`
- Create: `gear_sonic/tests/test_pico_video_frames.py`

- [ ] **Step 1: Write failing image-behavior tests**

The expected values are literal pixel/shape behavior rather than implementation details:

```python
from datetime import datetime, timezone

import cv2
import numpy as np

from gear_sonic.pico_video.frames import (
    compose_mono_sbs,
    decode_jpeg_rgb,
    render_status_card,
    render_test_card,
)


def test_mono_sbs_has_identical_left_and_right_eyes() -> None:
    source = np.zeros((24, 32, 3), dtype=np.uint8)
    source[:, :16] = (255, 0, 0)
    output = compose_mono_sbs(source, eye_width=64, height=48)

    assert output.shape == (48, 128, 3)
    np.testing.assert_array_equal(output[:, :64], output[:, 64:])
    assert output[:, :64, 0].mean() > output[:, :64, 2].mean()


def test_jpeg_decoder_returns_rgb_pixels() -> None:
    bgr = np.zeros((8, 8, 3), dtype=np.uint8)
    bgr[:] = (0, 0, 255)
    ok, encoded = cv2.imencode(".jpg", bgr)
    assert ok

    rgb = decode_jpeg_rgb(encoded.tobytes())

    assert rgb.shape == (8, 8, 3)
    assert rgb[..., 0].mean() > 240
    assert rgb[..., 2].mean() < 15


def test_test_card_changes_with_frame_number_and_marks_orientation() -> None:
    now = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)
    first = render_test_card(640, 480, frame_number=7, wall_time=now)
    second = render_test_card(640, 480, frame_number=8, wall_time=now)

    assert first.shape == (480, 640, 3)
    assert first.dtype == np.uint8
    assert not np.array_equal(first, second)
    assert tuple(first[12, 12]) == (255, 0, 0)
    assert tuple(first[12, -13]) == (0, 255, 0)


def test_status_card_is_visibly_red_and_contains_no_old_frame() -> None:
    card = render_status_card(1280, 480, "SENSORGATEWAY STALE")
    assert card.shape == (480, 1280, 3)
    assert card[..., 0].mean() > card[..., 1].mean() * 1.5
```

- [ ] **Step 2: Run the focused test and observe failure**

Run: `python -m pytest -q gear_sonic/tests/test_pico_video_frames.py`

Expected: FAIL because frame functions do not exist.

- [ ] **Step 3: Implement RGB decode, aspect-safe SBS, test card, and status card**

`compose_mono_sbs()` must preserve aspect ratio with black letterboxing, then concatenate two independent copies. `render_test_card()` uses OpenCV primitives for a grid, asymmetric red/green top corner markers, `FRAME 000007`, and ISO wall time. `render_status_card()` starts from a fresh red background every call and overlays the status string plus current wall time.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest -q gear_sonic/tests/test_pico_video_frames.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gear_sonic/pico_video/frames.py gear_sonic/tests/test_pico_video_frames.py
git commit -m "feat: compose PICO side-by-side video frames"
```

---

### Task 3: Read encoded head-camera frames only through SensorGateway

**Files:**

- Create: `gear_sonic/pico_video/gateway_source.py`
- Create: `gear_sonic/tests/test_pico_video_gateway_source.py`

- [ ] **Step 1: Write failing tests against real gateway contracts**

Use the real `MaterializedSnapshot`, `SensorSnapshot`, `SharedMemoryFrame`, and `MessageMetadata` structures; replace only the RPC/shared-memory boundary:

```python
from types import MappingProxyType

import numpy as np
import pytest

from gear_sonic.pico_video.gateway_source import GatewayFrame, SensorGatewayVideoSource
from gear_sonic.runtime.client import MaterializedSnapshot, SensorGatewayClientError
from gear_sonic.runtime.contracts import MessageMetadata, SharedMemoryFrame
from gear_sonic.runtime.snapshot import SensorSnapshot, TimestampBasis


def _materialized(jpeg: bytes, *, sequence: int, generation: int = 3) -> MaterializedSnapshot:
    metadata = MessageMetadata(
        source="camera_zmq",
        sequence=sequence,
        timestamp_ns=10_000_000_000,
        ttl_ms=1000,
        generation=generation,
    )
    frame = SharedMemoryFrame(
        metadata=metadata,
        stream="camera_encoded/ego_view",
        shared_memory="test-ring",
        shape=(len(jpeg),),
        dtype="uint8",
        offset_bytes=0,
        size_bytes=len(jpeg),
        source_timestamp_ns=9_900_000_000,
        source_clock="camera_time",
        attributes={"encoding": "jpeg_bytes", "image_shape": [480, 640, 3]},
    )
    snapshot = SensorSnapshot(
        complete=True,
        reason="",
        anchor_timestamp_ns=metadata.timestamp_ns,
        timestamp_basis=TimestampBasis.RECEIVE,
        frames=MappingProxyType({frame.stream: frame}),
        skew_ms=0.0,
        ages_ms=MappingProxyType({frame.stream: 1.0}),
    )
    return MaterializedSnapshot(
        snapshot=snapshot,
        arrays=MappingProxyType({frame.stream: np.frombuffer(jpeg, dtype=np.uint8)}),
        attempts=1,
    )


def test_source_returns_valid_jpeg_once_and_deduplicates_generation_sequence() -> None:
    client = SequenceClient([_materialized(b"\xff\xd8data\xff\xd9", sequence=8)] * 2)
    source = SensorGatewayVideoSource(client, max_age_ms=250.0)

    assert source.poll() == GatewayFrame(
        jpeg=b"\xff\xd8data\xff\xd9",
        generation=3,
        sequence=8,
        received_timestamp_ns=10_000_000_000,
        source_timestamp_ns=9_900_000_000,
        source_shape=(480, 640, 3),
    )
    assert source.poll() is None


def test_source_requests_only_the_encoded_ego_stream() -> None:
    client = SequenceClient([_materialized(b"\xff\xd8x\xff\xd9", sequence=1)])
    source = SensorGatewayVideoSource(client, max_age_ms=250.0)

    source.poll()

    assert client.requests[0].streams == ("camera_encoded/ego_view",)
    assert client.requests[0].max_age_ms == 250.0


def test_source_reports_gateway_and_encoding_failures_as_status() -> None:
    source = SensorGatewayVideoSource(
        SequenceClient([SensorGatewayClientError("offline")]), max_age_ms=250.0
    )
    assert source.poll() is None
    assert source.status == "SENSORGATEWAY OFFLINE"
```

`SequenceClient` is a strict test double: it records complete `SnapshotRequest` objects, returns real materialized snapshot structures, and raises supplied real exception types. Add branches for wrong dtype/shape, wrong `encoding`, invalid JPEG markers, generation changes, and recovery after an error.

- [ ] **Step 2: Run the focused test and observe failure**

Run: `python -m pytest -q gear_sonic/tests/test_pico_video_gateway_source.py`

Expected: FAIL because the source adapter does not exist.

- [ ] **Step 3: Implement the source adapter**

```python
class SensorGatewayVideoSource:
    STREAM = "camera_encoded/ego_view"

    def __init__(self, client: SensorGatewayClient, *, max_age_ms: float) -> None:
        self._client = client
        self._request = SnapshotRequest(
            streams=(self.STREAM,), max_age_ms=max_age_ms, max_skew_ms=0.0
        )
        self._last_key: tuple[int, int] | None = None
        self.status = "WAITING FOR SENSORGATEWAY"

    def poll(self) -> GatewayFrame | None:
        try:
            materialized = self._client.read_snapshot(self._request, retries=0)
            reference = materialized.snapshot.frames[self.STREAM]
            encoded = materialized.arrays[self.STREAM]
            if encoded.dtype != np.uint8 or encoded.ndim != 1:
                raise InvalidGatewayFrameError("encoded frame must be 1-D uint8")
            if reference.attributes.get("encoding") != "jpeg_bytes":
                raise InvalidGatewayFrameError("encoded frame must use jpeg_bytes")
            jpeg = encoded.tobytes()
            if len(jpeg) < 4 or not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
                raise InvalidGatewayFrameError("encoded frame is not a complete JPEG")
            key = (reference.metadata.generation, reference.metadata.sequence)
            if key == self._last_key:
                self.status = "READY"
                return None
            image_shape = tuple(int(value) for value in reference.attributes.get("image_shape", ()))
            if len(image_shape) != 3 or image_shape[2] != 3:
                raise InvalidGatewayFrameError("encoded frame has invalid image_shape")
            self._last_key = key
            self.status = "READY"
            return GatewayFrame(
                jpeg=jpeg,
                generation=key[0],
                sequence=key[1],
                received_timestamp_ns=reference.metadata.timestamp_ns,
                source_timestamp_ns=reference.source_timestamp_ns,
                source_shape=image_shape,
            )
        except SensorGatewayClientError:
            self.status = "SENSORGATEWAY OFFLINE"
            return None
        except InvalidGatewayFrameError:
            self.status = "INVALID CAMERA FRAME"
            return None
```

Copy the NumPy data to immutable `bytes` before returning. Deduplicate `(generation, sequence)` only after a sample passes validation so a corrected generation can recover.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest -q gear_sonic/tests/test_pico_video_gateway_source.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gear_sonic/pico_video/gateway_source.py gear_sonic/tests/test_pico_video_gateway_source.py
git commit -m "feat: read PICO video from SensorGateway"
```

---

### Task 4: Stream low-latency H.264 access units

**Files:**

- Create: `gear_sonic/pico_video/encoder.py`
- Create: `gear_sonic/tests/test_pico_video_encoder.py`
- Create: `gear_sonic/tests/test_pico_video_encoder_integration.py`

- [ ] **Step 1: Write failing access-unit and command tests**

```python
from gear_sonic.pico_video.encoder import AnnexBAccessUnitParser, EncoderSettings, ffmpeg_command


def test_annex_b_parser_groups_arbitrary_chunks_by_aud() -> None:
    first = b"\x00\x00\x00\x01\x09\xf0\x00\x00\x01\x65\xaa"
    second = b"\x00\x00\x01\x09\xf0\x00\x00\x01\x41\xbb"
    parser = AnnexBAccessUnitParser()

    assert parser.feed((first + second)[:9]) == ()
    assert parser.feed((first + second)[9:15]) == ()
    assert parser.feed((first + second)[15:]) == (first,)
    assert parser.flush() == (second,)


def test_nvenc_command_requests_annex_b_aud_and_no_b_frames() -> None:
    command = ffmpeg_command(
        EncoderSettings(width=1280, height=480, fps=30, bitrate=4_000_000, encoder="h264_nvenc")
    )

    assert command[:4] == ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    assert command[-3:] == ["-f", "h264", "pipe:1"]
    assert ["-bf", "0"] == command[command.index("-bf") : command.index("-bf") + 2]
    assert "aud=1" in command[command.index("-bsf:v") + 1]
```

Also test parser limits, garbage before the first start code, 3/4-byte start codes, SPS/PPS staying with their access unit, libx264 fallback options, non-RGB frame rejection, and subprocess early exit propagation.

- [ ] **Step 2: Run focused tests and observe failure**

Run: `python -m pytest -q gear_sonic/tests/test_pico_video_encoder.py`

Expected: FAIL because encoder code does not exist.

- [ ] **Step 3: Implement FFmpeg settings, process wrapper, and AU parser**

The process input contract is exact `rgb24`, 1280x480, fixed 30 FPS. Use these production encoder options:

```python
if settings.encoder == "h264_nvenc":
    codec = ["-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ull", "-rc", "cbr"]
elif settings.encoder == "libx264":
    codec = ["-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency"]
else:
    raise ValueError(f"unsupported encoder: {settings.encoder}")

return [
    "ffmpeg", "-hide_banner", "-loglevel", "error",
    "-f", "rawvideo", "-pix_fmt", "rgb24",
    "-s:v", f"{settings.width}x{settings.height}",
    "-r", str(settings.fps), "-i", "pipe:0",
    *codec, "-b:v", str(settings.bitrate), "-maxrate", str(settings.bitrate),
    "-bufsize", str(settings.bitrate), "-g", str(settings.fps), "-bf", "0",
    "-pix_fmt", "yuv420p", "-bsf:v", "h264_metadata=aud=insert",
    "-f", "h264", "pipe:1",
]
```

`FfmpegH264Encoder` owns the subprocess and two I/O methods: `write_frame(rgb)` writes exactly one frame; `iter_access_units(stop_event)` reads stdout chunks, yields complete AUs, flushes once at EOF, and raises a diagnostic containing bounded stderr if FFmpeg exits unexpectedly. `close()` closes stdin and terminates only its own live child with a short grace period.

- [ ] **Step 4: Run focused tests and one portable FFmpeg encode/decode smoke test**

Run: `python -m pytest -q gear_sonic/tests/test_pico_video_encoder.py`

Run: `python -m pytest -q gear_sonic/tests/test_pico_video_encoder_integration.py -k libx264` after adding an integration case that encodes three generated frames and probes the output as `1280x480`, H.264, with `ffprobe`.

Expected: PASS; skip only if `ffmpeg`/`ffprobe` is genuinely absent.

- [ ] **Step 5: Commit**

```bash
git add gear_sonic/pico_video/encoder.py gear_sonic/tests/test_pico_video_encoder.py gear_sonic/tests/test_pico_video_encoder_integration.py
git commit -m "feat: encode and frame low-latency H264"
```

---

### Task 5: Add one-session PICO control and video service

**Files:**

- Create: `gear_sonic/pico_video/bridge.py`
- Create: `gear_sonic/scripts/run_pico_video_bridge.py`
- Create: `gear_sonic/tests/test_pico_video_bridge.py`
- Create: `gear_sonic/tests/test_run_pico_video_bridge.py`

- [ ] **Step 1: Write failing lifecycle tests with real loopback sockets**

Exercise the public server behavior, not thread internals:

```python
def test_open_camera_streams_framed_access_units_to_requested_loopback_port() -> None:
    with FakePicoVideoReceiver() as pico, running_bridge(
        source=RepeatingSource(TEST_JPEG), encoder="libx264"
    ) as bridge:
        with socket.create_connection(bridge.control_address, timeout=1.0) as control:
            control.sendall(open_camera_packet(ip="127.0.0.1", port=pico.port))
            access_unit = pico.receive_access_unit(timeout_s=3.0)

    assert access_unit.startswith((b"\x00\x00\x00\x01", b"\x00\x00\x01"))
    assert b"\x00\x00\x01\x09" in access_unit or b"\x00\x00\x00\x01\x09" in access_unit


def test_close_camera_stops_video_without_hanging_control_connection() -> None:
    with FakePicoVideoReceiver() as pico, running_bridge(
        source=RepeatingSource(TEST_JPEG), encoder="libx264"
    ) as bridge:
        with socket.create_connection(bridge.control_address, timeout=1.0) as control:
            control.sendall(open_camera_packet(ip="127.0.0.1", port=pico.port))
            pico.receive_access_unit(timeout_s=3.0)
            control.sendall(close_camera_packet())
            assert pico.wait_for_disconnect(timeout_s=2.0)


def test_second_open_replaces_first_session() -> None:
    source = RepeatingSource(TEST_JPEG)
    with FakePicoVideoReceiver() as first, FakePicoVideoReceiver() as second, running_bridge(
        source=source, encoder="libx264"
    ) as bridge:
        send_open(bridge.control_address, first.port)
        first.receive_access_unit(timeout_s=3.0)
        send_open(bridge.control_address, second.port)
        assert first.wait_for_disconnect(timeout_s=2.0)
        assert second.receive_access_unit(timeout_s=3.0)
```

Add tests for invalid request isolation, control client disconnect, unreachable video endpoint, capacity-one replacement, source becoming stale, and a fresh status card replacing camera pixels.

- [ ] **Step 2: Run focused tests and observe failure**

Run: `python -m pytest -q gear_sonic/tests/test_pico_video_bridge.py gear_sonic/tests/test_run_pico_video_bridge.py`

Expected: FAIL because the bridge/server CLI does not exist.

- [ ] **Step 3: Implement the producer/session/control lifecycles**

Use three owned components:

```python
@dataclass(frozen=True)
class BridgeSettings:
    gateway_endpoint: str
    stream: str = "camera_encoded/ego_view"
    control_host: str = "0.0.0.0"
    control_port: int = 13579
    width: int = 1280
    height: int = 480
    fps: int = 30
    bitrate: int = 4_000_000
    max_age_ms: float = 250.0
    stale_fps: float = 2.0
    encoder: str = "h264_nvenc"


class LatestFrameSlot:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._frame: np.ndarray | None = None

    def put(self, frame: np.ndarray) -> None:
        with self._condition:
            self._frame = frame
            self._condition.notify()

    def get(self, timeout_s: float) -> np.ndarray | None:
        with self._condition:
            if self._frame is None:
                self._condition.wait(timeout_s)
            frame, self._frame = self._frame, None
            return frame
```

`PicoVideoBridge.serve_forever()` owns the listening socket and producer thread; `PicoVideoBridge.stop()` sets the common stop event and performs bounded resource shutdown. One producer thread continuously polls `SensorGatewayVideoSource`, decodes valid new JPEGs, and replaces the slot. A duplicate sample with source status `READY` leaves the slot unchanged; an unavailable, stale, or invalid source status publishes a newly rendered status frame no faster than `stale_fps`. The active `VideoSession` connects to the request's IP/port, starts FFmpeg with the request bitrate/FPS only after enforcing the fixed 1280x480 `SONIC_HEAD` profile, feeds the slot at the session cadence, and sends `sendall(frame_video_access_unit(au))`.

The control server uses `SO_REUSEADDR`, accepts multiple sequential clients, applies `ControlFrameDecoder`, handles only `OPEN_CAMERA`/`CLOSE_CAMERA`, logs malformed/unknown messages, and contains errors to the client/session that caused them. `stop()` must set events, close listening and active sockets, close the SensorGateway client, and join owned threads with bounded deadlines.

The CLI resolves `sensor_gateway_metadata` through `load_runtime_profile()` unless `--gateway-endpoint` is supplied. It must expose profile/overlay, bind, stream, max-age, encoder, dimensions/FPS/bitrate, and stats interval flags. Signal handlers call `bridge.stop()`.

- [ ] **Step 4: Run lifecycle and script tests**

Run: `python -m pytest -q gear_sonic/tests/test_pico_video_bridge.py gear_sonic/tests/test_run_pico_video_bridge.py`

Expected: PASS with no leaked subprocesses or non-daemon threads.

- [ ] **Step 5: Commit**

```bash
git add gear_sonic/pico_video/bridge.py gear_sonic/scripts/run_pico_video_bridge.py gear_sonic/tests/test_pico_video_bridge.py gear_sonic/tests/test_run_pico_video_bridge.py
git commit -m "feat: serve SensorGateway video to PICO"
```

---

### Task 6: Add the robot-free mock camera and full workstation loop

**Files:**

- Create: `gear_sonic/pico_video/mock_camera.py`
- Create: `gear_sonic/scripts/run_mock_camera_server.py`
- Create: `gear_sonic/tests/test_pico_video_mock_camera.py`
- Create: `gear_sonic/tests/test_pico_video_workstation_integration.py`

- [ ] **Step 1: Write failing publisher contract test**

Subscribe over a real in-process ZMQ socket, unpack the real message, and verify `ImageMessageSchema.deserialize()`:

```python
def test_mock_camera_publishes_changing_ego_view_in_camera_schema() -> None:
    with running_mock_camera(endpoint) as publisher, subscribed(endpoint) as subscriber:
        first = receive_camera_schema(subscriber)
        second = receive_camera_schema(subscriber)

    assert set(first.images) == {"ego_view"}
    assert first.images["ego_view"].shape == (480, 640, 3)
    assert first.image_shapes["ego_view"] == [480, 640, 3]
    assert first.timestamps["ego_view"] < second.timestamps["ego_view"]
    assert not np.array_equal(first.images["ego_view"], second.images["ego_view"])
```

- [ ] **Step 2: Run the focused test and observe failure**

Run: `python -m pytest -q gear_sonic/tests/test_pico_video_mock_camera.py`

Expected: FAIL because the mock publisher does not exist.

- [ ] **Step 3: Implement the mock publisher and CLI**

`MockCameraPublisher` renders `render_test_card(640, 480, frame_number, wall_time)`, wraps it in `ImageMessageSchema(timestamps={"ego_view": time.time()}, images={"ego_view": rgb})`, serializes with the production JPEG quality, and sends via `SensorServer` at a monotonic 30 Hz deadline. The CLI defaults to port 5555 and supports width, height, FPS, and JPEG quality.

- [ ] **Step 4: Run the publisher test**

Run: `python -m pytest -q gear_sonic/tests/test_pico_video_mock_camera.py`

Expected: PASS.

- [ ] **Step 5: Write and run the full loopback integration test**

Start these real components in-process on reserved loopback ports:

1. `MockCameraPublisher` ZMQ PUB.
2. `CameraZmqIngress` + `SensorGatewayCore` + `SensorGatewayRpcServer`.
3. Real `SensorGatewayClient` and `PicoVideoBridge` with `libx264`.
4. Fake PICO control client and video receiver.

The test waits for at least three access units, writes only their H.264 payloads to a temporary file, and invokes:

```bash
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height \
  -of default=noprint_wrappers=1 /tmp/pico-loopback.h264
```

Assert literal output contains `codec_name=h264`, `width=1280`, and `height=480`. Decode the first frame with FFmpeg to RGB and assert the two 640-pixel halves are equal within lossy H.264 tolerance (mean absolute difference under 3). Then stop the mock camera and assert a later decoded frame is red-dominant, proving stale input is marked instead of frozen.

Run: `python -m pytest -q gear_sonic/tests/test_pico_video_workstation_integration.py`

Expected: PASS; skip only when FFmpeg is absent, never for application errors.

- [ ] **Step 6: Commit**

```bash
git add gear_sonic/pico_video/mock_camera.py gear_sonic/scripts/run_mock_camera_server.py gear_sonic/tests/test_pico_video_mock_camera.py gear_sonic/tests/test_pico_video_workstation_integration.py
git commit -m "test: add PICO workstation video loop"
```

---

### Task 7: Document operation and perform gated verification

**Files:**

- Create: `docs/source/tutorials/pico_remote_vision.md`
- Modify: `docs/source/index.rst`
- Modify: `README.md`

- [ ] **Step 1: Write the operator runbook**

Document the local sequence with four terminals:

```bash
# Terminal 1: camera-compatible animated test source
python gear_sonic/scripts/run_mock_camera_server.py --port 5555

# Terminal 2: real SensorGateway, camera ingress only
python gear_sonic/scripts/run_sensor_gateway.py \
  --camera-host 127.0.0.1 --camera-port 5555 \
  --rpc-bind-host 127.0.0.1 \
  --no-enable-depth-anything --no-enable-cpp-state --no-enable-ros \
  --no-enable-visualization --no-enable-vla-timing

# Terminal 3: PICO bridge (portable CPU encoder for first workstation check)
python gear_sonic/scripts/run_pico_video_bridge.py \
  --gateway-endpoint tcp://127.0.0.1:5560 --encoder libx264

# Terminal 4: bridge diagnostics / fake-PICO integration test, only after reminder
python -m pytest -q gear_sonic/tests/test_pico_video_workstation_integration.py
```

Then document the NVIDIA production command (`--encoder h264_nvenc`), firewall ports 13579/12345, PICO `SONIC_HEAD` profile (ZED, 1280x480, 30 FPS, 4 Mbps), expected stats, shutdown order, stale-card meanings, and troubleshooting for SensorGateway timeout, FFmpeg encoder failure, and headset connection refusal.

Keep two explicit checklists:

- **PICO gate:** notify the user and wait before launching/connecting to the actual headset.
- **Robot gate:** after power is available, notify the user and wait before connecting the actual camera endpoint; do not start motion/control processes as part of video validation.

- [ ] **Step 2: Run implementation plan self-review**

Confirm every design requirement has an implementation or test owner:

```bash
rg -n "TODO|FIXME|placeholder|pass$|NotImplemented" gear_sonic/pico_video \
  gear_sonic/scripts/run_pico_video_bridge.py gear_sonic/scripts/run_mock_camera_server.py \
  gear_sonic/tests/test_pico_video_*
git diff --check HEAD~6..HEAD
```

Expected: no placeholders, no whitespace errors. Any deliberate abstract/protocol `pass` must be reviewed rather than accepted by the scan.

- [ ] **Step 3: Run the complete focused verification suite**

Run:

```bash
python -m pytest -q \
  gear_sonic/tests/test_pico_video_protocol.py \
  gear_sonic/tests/test_pico_video_frames.py \
  gear_sonic/tests/test_pico_video_gateway_source.py \
  gear_sonic/tests/test_pico_video_encoder.py \
  gear_sonic/tests/test_pico_video_encoder_integration.py \
  gear_sonic/tests/test_pico_video_bridge.py \
  gear_sonic/tests/test_run_pico_video_bridge.py \
  gear_sonic/tests/test_pico_video_mock_camera.py \
  gear_sonic/tests/test_pico_video_workstation_integration.py
python -m compileall -q gear_sonic/pico_video gear_sonic/scripts/run_pico_video_bridge.py gear_sonic/scripts/run_mock_camera_server.py
```

Expected: all focused tests PASS.

- [ ] **Step 4: Run adjacent SensorGateway regression tests**

Run:

```bash
python -m pytest -q \
  gear_sonic/tests/test_sensor_gateway.py \
  gear_sonic/tests/test_sensor_gateway_client.py \
  gear_sonic/tests/test_run_sensor_gateway.py \
  gear_sonic/tests/test_runtime_shared_memory.py \
  gear_sonic/tests/test_runtime_snapshot.py
```

Expected: PASS.

- [ ] **Step 5: Inspect branch scope and commit documentation**

```bash
git status --short
git diff --stat agent_full...HEAD
git diff --check agent_full...HEAD
git add docs/source/tutorials/pico_remote_vision.md docs/source/index.rst README.md
git commit -m "docs: explain PICO SensorGateway video setup"
```

Stage only documentation files that actually changed. Verify the diff excludes `gear_sonic/scripts/run_data_exporter.py` and `Log/`.

- [ ] **Step 6: Stop before hardware**

Report the workstation test evidence and exact remaining PICO/robot commands. Do not run either hardware gate until the user explicitly authorizes it.
