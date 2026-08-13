# PICO SensorGateway Video Bridge Design

## Goal

Display SONIC's head-mounted `ego_view` camera in the existing XRoboToolkit
PICO Remote Vision window without coupling video failures to robot control. The
bridge runs on the operator workstation and consumes camera data exclusively
through SensorGateway.

The first milestone is fully local because the robot is unavailable: a
deterministic animated test card enters the normal camera wire format, passes
through the real SensorGateway ingress and snapshot API, and is encoded and
sent to a fake PICO receiver. Live PICO and robot tests are separate, explicit
later gates.

## Scope

### Milestone 1: Local mock pipeline

- Publish a dynamic `640x480@30` RGB test card containing a frame number and
  capture timestamp using the existing `ImageMessageSchema` camera protocol.
- Run the existing `CameraZmqIngress` and SensorGateway RPC service rather than
  introducing a second camera data path.
- Read only `camera_encoded/ego_view` through `SensorGatewayClient`.
- Decode fresh JPEG frames, duplicate the mono view into a `1280x480`
  side-by-side frame, and encode low-latency H.264 with workstation FFmpeg.
- Implement the XRoboToolkit `OPEN_CAMERA` and `CLOSE_CAMERA` control flow and
  its four-byte big-endian H.264 payload framing.
- Validate the complete software pipeline with a fake PICO endpoint and an
  independent FFmpeg decode check.

### Milestone 2: PICO display

- Add a `SONIC_HEAD` XRoboToolkit video-source profile configured as
  `1280x480@30` with identical mono images for both eyes.
- Start Remote Vision on the PICO and confirm that the animated test card is
  visible, current, correctly oriented, and correctly scaled.
- Measure end-to-end latency using the rendered timestamps.

This test starts only after the user is reminded and confirms the headset is
ready.

### Milestone 3: Live robot camera

- Replace the local mock camera publisher with the deployed composed-camera
  source.
- Keep the bridge input unchanged at `camera_encoded/ego_view` on
  SensorGateway.
- Validate disconnect, stale-frame, and recovery behavior with the robot.

This test starts only after the user is reminded and confirms the robot and
headset are ready.

## Non-goals

- The bridge does not connect directly to the camera server on port 5555.
- The bridge does not publish or forward robot commands.
- The first version does not implement WebRTC, audio, depth streaming,
  panoramic projection, or true stereo reconstruction.
- The first version does not modify the XRoboToolkit Unity application. Its
  persistent `video_source.yml` configuration is sufficient.
- The first version does not support multiple simultaneous headsets.
- Software and Jetson encoders are not first-class targets. A `libx264`
  fallback may be used by automated tests when NVENC is unavailable, but the
  operator workstation runtime target is `h264_nvenc` on the RTX 3090.

## Architecture

```text
Local milestone

AnimatedTestCard
      |
      | ImageMessageSchema, ZMQ PUB
      v
MockCameraServer :5555
      |
      v
CameraZmqIngress -> SensorGatewayCore -> shared-memory ring
                                      -> SensorGateway RPC :5560
                                                   |
                                                   | camera_encoded/ego_view
                                                   v
                                         PicoVideoBridge :13579
                                           | JPEG decode
                                           | mono -> identical SBS
                                           | FFmpeg H.264
                                           v
                                      fake PICO :12345

Live milestone

Robot ComposedCamera -> workstation SensorGateway -> same PicoVideoBridge
```

The mock source stops at the existing camera-server boundary. All bridge input,
including local development input, crosses SensorGateway. This makes mock and
live operation differ only in the producer feeding `CameraZmqIngress`.

## Components

### Mock camera publisher

`run_mock_camera_server.py` owns a ZMQ PUB socket and emits the current
`ImageMessageSchema` payload. Each RGB test card contains:

- a high-contrast color pattern and orientation markers;
- `LEFT`, `RIGHT`, `TOP`, and `BOTTOM` labels;
- a monotonically increasing frame number;
- a wall-clock capture timestamp suitable for visual latency checks.

The publisher stores the JPEG bytes under `images["ego_view"]`, source time
under `timestamps["ego_view"]`, and `[480, 640, 3]` under
`image_shapes["ego_view"]`. It never writes directly into SensorGateway.

### SensorGateway video source

`SensorGatewayVideoSource` resolves the `sensor_gateway_metadata` endpoint from
the normal runtime profile unless a CLI endpoint override is supplied. It
requests a single stream:

```text
camera_encoded/ego_view
```

Snapshots use a configurable maximum age, defaulting to `250 ms`, and no skew
allowance is needed for the single stream. The source:

- accepts only one-dimensional `uint8` arrays;
- accepts only the `jpeg_bytes` encoding attribute;
- uses `(generation, sequence)` to suppress duplicate frames;
- returns the JPEG bytes, Gateway receive timestamp, source timestamp, and
  image-shape metadata;
- converts missing, stale, overwritten, and temporarily unavailable snapshots
  into typed source-status results instead of terminating the process.

The bridge deliberately uses the encoded Gateway stream. `CameraZmqIngress`
publishes it before performing its decoded RGB publication, and the compact
shared-memory copy avoids moving a full RGB frame through the snapshot RPC
reader.

### Frame compositor

The compositor decodes JPEG to RGB and verifies a three-channel image. The
production mono frame is resized with aspect ratio preserved and padded if
needed, then copied into both halves of the negotiated side-by-side output.

For the `SONIC_HEAD` profile, each eye is `640x480`, producing `1280x480`. No
synthetic disparity is introduced. The two halves must be byte-identical.

When the source is unavailable or stale, the compositor generates a red status
card containing `SENSOR GATEWAY UNAVAILABLE` or `SENSOR FRAME STALE` and the
current local time. It must never resend an unmarked frozen camera frame.

### XRoboToolkit control server

The control server listens on TCP port `13579`. It accepts one active
XRoboToolkit control connection and supports the existing length-framed
commands:

- `OPEN_CAMERA`: parse width, height, FPS, bitrate, requested camera type,
  headset IP, and headset video port;
- `CLOSE_CAMERA`: stop and dispose the current video session;
- unknown commands: log and ignore without closing a valid control session.

The initial profile uses camera type `ZED` because the released XRoboToolkit
client already recognizes that Remote Vision flow. The bridge validates that
the negotiated dimensions are positive, even, and within a bounded pixel
count, and that FPS, bitrate, and port are safe positive values.

Only one streaming session is permitted. A new `OPEN_CAMERA` replaces the old
session after the old encoder and video socket have shut down.

### H.264 encoder and video sender

The FFmpeg encoder receives packed RGB24 frames over standard input and emits
Annex-B H.264 over standard output. The workstation command uses:

- `h264_nvenc`;
- the lowest-latency preset/tune supported by the installed FFmpeg;
- zero B frames;
- GOP length equal to the negotiated FPS;
- access-unit delimiters and repeated SPS/PPS at keyframes;
- the bitrate requested by the PICO profile, capped by local safety limits.

The encoder reader groups Annex-B NAL units into access units using AUD
boundaries. Each access unit is sent as:

```text
4-byte unsigned big-endian payload length | H.264 access-unit bytes
```

The video socket connects to the headset IP and port from `OPEN_CAMERA`; the
PICO remains the video TCP listener. Writes have bounded timeouts. Encoder exit,
socket errors, or `CLOSE_CAMERA` terminate only the current streaming session.

## Process and configuration model

The bridge CLI exposes:

- runtime profile and overlay paths;
- optional SensorGateway endpoint override;
- control bind host and port, default `0.0.0.0:13579`;
- stream name, default `camera_encoded/ego_view`;
- maximum source age, default `250 ms`;
- encoder choice, default `h264_nvenc`;
- stale-card refresh rate, default `2 FPS`;
- verbosity and periodic statistics interval.

The live SensorGateway endpoint is resolved from the repository's existing
runtime profile. PICO video IP and port remain session data negotiated by
`OPEN_CAMERA`; they are not added as static runtime endpoints.

For the local milestone, the existing SensorGateway is launched with camera
ingress enabled and unrelated sources disabled, with its camera host overridden
to `127.0.0.1`. The mock publisher and SensorGateway remain separate processes
so the real ingress boundary is exercised.

## Lifecycle and concurrency

The main process owns a stop event and four bounded activities:

1. the control accept/read loop;
2. SensorGateway polling for the newest source frame;
3. FFmpeg stdin/stdout handling for the current video session;
4. video socket writes.

Queues have a capacity of one frame. A producer replaces an unconsumed frame
rather than blocking or accumulating latency. Session objects own their FFmpeg
process, socket, and worker threads, and shutdown joins them before a
replacement session starts.

SIGINT and SIGTERM stop accepting control connections, close the active video
session, terminate FFmpeg with a bounded grace period, close SensorGateway, and
return a nonzero exit status only for unrecoverable process-level failures.

## Safety and failure behavior

- SensorGateway timeout: retry with bounded backoff and display an unavailable
  status card if a video session is active.
- Missing stream: treat as unavailable and include the stream name in logs.
- Stale or overwritten frame: discard it; never encode it as a fresh frame.
- Invalid JPEG: count and drop it, then display an invalid-frame status card if
  no valid fresh frame arrives.
- Duplicate Gateway sequence: do not decode or encode it again.
- Encoder failure: close the video socket, report the FFmpeg diagnostic, and
  keep the control server ready for a new `OPEN_CAMERA`.
- PICO disconnect: close the video session without affecting SensorGateway.
- Control disconnect: close the associated video session immediately.
- No video code imports or calls the SONIC control gateways or C++ command
  endpoints.

## Observability

At a configurable interval, the bridge prints one compact statistics line:

- Gateway state and reconnect count;
- latest Gateway generation/sequence and receive age;
- unique source FPS;
- decoded, invalid, duplicate, and stale frame counts;
- encoded and sent FPS;
- sent bitrate and socket error count;
- active/inactive PICO session state.

Logs never print raw image bytes. Normal frame traffic does not emit one line
per frame.

## Test strategy and gates

No tests are run until the user receives the requested reminder.

### Automated unit tests

- Control framing handles fragmented and coalesced TCP input.
- `OPEN_CAMERA` parsing rejects invalid magic, versions, lengths, dimensions,
  FPS, bitrate, and ports.
- Video framing uses a four-byte big-endian payload length.
- Annex-B grouping emits complete access units across arbitrary read chunks.
- The test-card generator produces the requested RGB shape and changes its
  timestamp/frame-number region between frames.
- Mono-to-SBS output has byte-identical left and right halves.
- SensorGateway source accepts valid `jpeg_bytes`, suppresses duplicate
  generation/sequence pairs, and rejects stale or malformed frames.
- Bounded latest-frame queues drop the old item rather than blocking.
- Stale and unavailable states produce explicit red status cards.

### Automated integration tests

- A mock camera message passes through real `CameraZmqIngress`,
  `SensorGatewayCore`, shared memory, and `SensorGatewayClient` before reaching
  the bridge source.
- A fake PICO control client sends fragmented `OPEN_CAMERA`, listens on a local
  video port, and receives valid length-framed H.264.
- The captured H.264 stream is independently decoded with FFmpeg and contains
  the expected `1280x480` frames.
- `CLOSE_CAMERA`, fake-PICO disconnect, and stale SensorGateway input each stop
  or transition the video session without hanging.

### Manual gates

1. Before any PICO test, remind the user and wait for confirmation that the
   headset is ready.
2. Before any robot test, remind the user and wait for confirmation that the
   robot and headset are ready.

## Acceptance criteria

Milestone 1 is complete when:

- the bridge reads the mock image only through
  `camera_encoded/ego_view` from SensorGateway;
- the fake PICO receives a decodable `1280x480@30` H.264 stream;
- decoded left and right views are identical and retain the test-card
  orientation markers;
- frame numbers advance without an unbounded queue;
- stale input results in a visible status frame rather than a frozen camera
  frame;
- all targeted automated tests pass with clean output;
- documentation gives exact commands for the three local processes.

No live PICO or robot result is required to complete Milestone 1.
