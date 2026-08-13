# PICO Remote Vision from the SONIC Head Camera

This tutorial displays SONIC's `ego_view` head-camera stream in the existing
XRoboToolkit Remote Vision window. The workstation bridge consumes the camera
exclusively through SensorGateway:

```text
camera server -> CameraZmqIngress -> SensorGateway shared memory/RPC
              -> camera_encoded/ego_view -> PICO bridge -> H.264 -> PICO
```

The bridge is video-only. It does not send robot commands and does not require
the SONIC motion-control process. Start with the local animated camera while
the robot is powered off, then cross the PICO and robot gates separately.

## Prerequisites

Use the repository's teleoperation environment and install FFmpeg on the
workstation:

```bash
source .venv_teleop/bin/activate
sudo apt-get update
sudo apt-get install ffmpeg

ffmpeg -hide_banner -encoders | grep -E 'h264_nvenc|libx264'
```

The production workstation path uses `h264_nvenc`. `libx264` is available for
the first local check and automated tests.

## Local validation without a robot or PICO

Open three terminals at the repository root and activate `.venv_teleop` in
each terminal.

### Terminal 1: animated camera server

```bash
python gear_sonic/scripts/run_mock_camera_server.py --port 5555
```

The source publishes a changing 640x480 RGB card under `ego_view` using the
normal `ImageMessageSchema` binary-JPEG wire format. Its frame number, time,
corner colors, and orientation labels make frozen, mirrored, or rotated video
visible.

### Terminal 2: SensorGateway with camera ingress only

```bash
python gear_sonic/scripts/run_sensor_gateway.py \
  --camera-host 127.0.0.1 --camera-port 5555 \
  --rpc-bind-host 127.0.0.1 \
  --no-enable-lingbot-depth --no-enable-cpp-state --no-enable-ros \
  --no-enable-visualization --no-enable-vla-timing
```

This is the required data boundary. Do not connect the bridge directly to
port 5555.

### Terminal 3: Remote Vision bridge

For the first CPU-only workstation check:

```bash
python gear_sonic/scripts/run_pico_video_bridge.py \
  --gateway-endpoint tcp://127.0.0.1:5560 \
  --encoder libx264 --verbose
```

For normal use on the RTX workstation:

```bash
python gear_sonic/scripts/run_pico_video_bridge.py \
  --gateway-endpoint tcp://127.0.0.1:5560 \
  --encoder h264_nvenc
```

The bridge listens for XRoboToolkit control on TCP 13579. It does not connect
to a video receiver until an `OPEN_CAMERA` request arrives. While there is no
PICO, this idle state is expected.

The full robot-free loopback check is:

```bash
python -m pytest -q gear_sonic/tests/test_pico_video_workstation_integration.py
```

It starts a fake PICO listener, decodes the resulting H.264 independently, and
checks 1280x480 output, identical left/right eyes, live-frame updates, and the
red stale-source card.

## PICO gate: add the SONIC_HEAD profile

Do not begin this section until the operator has been notified and confirms
that the PICO is ready. Enable Developer Mode and USB debugging, then pull the
XRoboToolkit profile (v1.1.0 or newer):

```bash
adb pull /sdcard/Android/data/com.xrobotoolkit.client/files/video_source.yml
```

Append this entry to `video_source.yml`:

```yaml
- name: "SONIC_HEAD"
  camera: "ZED"
  description: "SONIC mono head camera duplicated as stereo SBS"
  properties:
    - name: "visibleRatio"
      type: "float"
      value: 0.555
    - name: "contentRatio"
      type: "float"
      value: 1.8
    - name: "heightCompressionFactor"
      type: "float"
      value: 1.333333
    - name: "RawImageRectSize"
      type: "string"
      value: "600x225"
    - name: "CamWidth"
      type: "int"
      value: 1280
    - name: "CamHeight"
      type: "int"
      value: 480
    - name: "CamFPS"
      type: "int"
      value: 30
    - name: "CamBitrate"
      type: "int"
      value: 4000000
    - name: "AudioStreamPort"
      type: "int"
      value: 13580
```

Push the file back and restart XRoboToolkit:

```bash
adb push video_source.yml /sdcard/Android/data/com.xrobotoolkit.client/files/video_source.yml
```

If the profile must be reset, remove the device copy and restart the app:

```bash
adb shell rm /sdcard/Android/data/com.xrobotoolkit.client/files/video_source.yml
```

### Network setup

- PICO and workstation must be on the same low-latency LAN/Wi-Fi.
- Allow inbound TCP 13579 on the workstation for Remote Vision control.
- The PICO listens on TCP 12345; the workstation opens the outbound video
  connection to the IP and port supplied by the headset.
- With UFW enabled, the workstation rule is `sudo ufw allow 13579/tcp`.

In XRoboToolkit Remote Vision, select **SONIC_HEAD**, enter the workstation's
LAN IPv4 address, and press **Listen**. Expected results are:

- the animated frame counter and timestamp keep changing;
- `LEFT`, `RIGHT`, `TOP`, and `BOTTOM` have the correct orientation;
- both eyes show the same image without disparity or double vision;
- stopping Terminal 1 replaces the image with a red `SENSOR FRAME STALE` card;
- restarting Terminal 1 restores live video without restarting the bridge.

## Robot gate: switch from mock to the live head camera

Do not begin this section until the operator has been notified and confirms
that the robot camera and PICO are ready. No motion/control process is needed
for this video check.

1. Stop `run_mock_camera_server.py`.
2. Start the normal composed camera server on the robot.
3. Start SensorGateway with the deployed runtime profile. If a temporary
   override is needed, use the camera endpoint only:

   ```bash
   python gear_sonic/scripts/run_sensor_gateway.py \
     --camera-host 192.168.123.164 --camera-port 5555 \
     --no-enable-lingbot-depth --no-enable-cpp-state --no-enable-ros \
     --no-enable-visualization --no-enable-vla-timing
   ```

4. Keep `run_pico_video_bridge.py` unchanged. Its input remains
   `camera_encoded/ego_view` from SensorGateway.
5. Confirm disconnect, stale-card, and automatic recovery behavior before
   combining the video path with robot teleoperation.

## Troubleshooting

| Symptom | Check |
|---|---|
| `SENSORGATEWAY OFFLINE` | SensorGateway RPC is listening at the bridge's `--gateway-endpoint`; default local port is 5560. |
| `SENSOR FRAME STALE` | The camera publisher stopped, the stream name is missing, or the newest Gateway frame is older than `--max-age-ms` (default 250 ms). |
| `INVALID CAMERA FRAME` | `camera_encoded/ego_view` must be a 1-D uint8 shared-memory array with `encoding=jpeg_bytes` and `image_shape=[H,W,3]`. |
| FFmpeg exits immediately | Run the encoder listing command above; use `--encoder libx264` to distinguish NVENC/driver problems from pipeline problems. |
| PICO cannot connect | Verify the app uses the workstation LAN IP, TCP 13579 is reachable, and another bridge is not already bound to the port. |
| Bridge cannot connect to PICO video | Keep the Remote Vision window listening, verify the PICO IP in `OPEN_CAMERA`, and confirm both devices are on the same network. |
| Image is stretched or double | Select `SONIC_HEAD`; the bridge intentionally accepts only 1280x480@30 and duplicates one 640x480 eye image. |

Stop in reverse order: close Remote Vision, stop the bridge, stop
SensorGateway, then stop the camera publisher. Video failures are isolated
from SONIC control throughout this flow.
