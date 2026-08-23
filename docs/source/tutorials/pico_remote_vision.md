# PICO Remote Vision from the SONIC Head Camera

This tutorial displays SONIC's `ego_view` head-camera stream in the existing
XRoboToolkit Remote Vision window. The workstation bridge consumes the camera
exclusively through SensorGateway:

```text
camera server -> CameraZmqIngress -> SensorGateway shared memory/RPC
              -> camera_encoded/ego_view -> PICO bridge -> H.264 -> PICO
```

The bridge is video-only. It does not send robot commands and does not require
the SONIC motion-control process. Bring up the live camera and cross the PICO
and robot gates separately.

## Prerequisites

Use the repository's teleoperation environment and install FFmpeg on the
workstation:

```bash
source .venv_teleop/bin/activate
sudo apt-get update
sudo apt-get install ffmpeg

ffmpeg -hide_banner -encoders | grep -E 'h264_nvenc|libx264'
```

The production workstation path uses `h264_nvenc`. `libx264` remains available
for diagnosing encoder or driver problems.

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

### Native USBOnly network

Connect the PICO over USB and select **USBOnly** in XRoboToolkit. Leave the
network under the ownership of the PICO OS and NetworkManager:

- do not create a static `pico0` connection or rename the RNDIS interface;
- do not change the headset USB gadget functions;
- do not add ADB forward/reverse tunnels;
- do not add interface-name firewall rules for XRoboToolkit.

The PICO provides DHCP over its native RNDIS link. Both the subnet and the
workstation address may change after reconnecting, so neither belongs in a
checked-in profile. The production bridge discovers the active PICO RNDIS
interface, workstation address, and PICO gateway every time it starts. It
binds TCP 13579 only to that workstation USB address and binds the outbound
H.264 socket to the same source address. Startup fails closed when USBOnly is
absent, ambiguous, or routed through another interface unless supervised mode
is explicitly enabled.

With SensorGateway running, start the production bridge without a network
override:

```bash
python -m gear_sonic.utils.pico_video.service \
  --profile gear_sonic/config/launch_inference.yaml \
  --encoder h264_nvenc
```

Use `--stay-alive` for a long-running collection process. In this mode the
bridge waits while USBOnly is absent, checks the active native link every two
seconds, and fully recreates the listener, SensorGateway client, and video
session after a disconnect or DHCP address change:

```bash
python -m gear_sonic.utils.pico_video.service \
  --profile gear_sonic/config/launch_inference.yaml \
  --encoder h264_nvenc --stay-alive
```

If multiple PICO headsets are attached, select the intended native interface
explicitly:

```bash
python -m gear_sonic.utils.pico_video.service \
  --profile gear_sonic/config/launch_inference.yaml \
  --pico-usb-interface enx0123456789ab
```

The PICO listens on TCP 12345 and advertises that endpoint in `OPEN_CAMERA`.
The bridge accepts it only when both the control peer and video target match
the discovered PICO USB peer. Wi-Fi and ordinary Ethernet clients cannot open
a video session.

## Data collection launcher

`launch_data_collection.py` enables the supervised PICO video bridge by
default. The `gateways` tmux window contains SensorGateway, ControlGateway,
and a third PICO Video pane. The PICO does not have to be connected when data
collection starts: the third pane waits for native USBOnly and begins serving
after XRoboToolkit connects it. A later USB disconnect or address change is
rediscovered without restarting data collection.

```bash
python gear_sonic/scripts/launch_data_collection.py
```

Only when no PICO video is wanted, disable that pane explicitly:

```bash
python gear_sonic/scripts/launch_data_collection.py --no-pico-video
```

The launcher still takes camera frames only from
`camera_encoded/ego_view` in SensorGateway. It does not open the camera server
directly and does not manage NetworkManager, USB gadget mode, routes, or
firewall rules.

In XRoboToolkit Remote Vision, select **SONIC_HEAD** and press **Listen**.
Expected results are:

- the live head-camera image keeps updating;
- the image has the correct orientation;
- both eyes show the same image without disparity or double vision;
- stopping the camera source replaces the image with a red `SENSOR FRAME STALE` card;
- restarting the camera source restores live video without restarting the bridge.

## Robot gate: start the live head camera

Do not begin this section until the operator has been notified and confirms
that the robot camera and PICO are ready. No motion/control process is needed
for this video check.

1. Start the normal composed camera server on the robot.
2. Start SensorGateway with the deployed runtime profile. If a temporary
   camera change is needed, put it in an overlay:

   ```bash
   printf 'endpoints:\n  camera_server: {host: 192.168.123.164, port: 5555}\n' \
     > /tmp/sonic-pico-camera.yaml
   python -m gear_sonic.runtime.gateway.services.sensor \
     --overlay /tmp/sonic-pico-camera.yaml \
     --no-enable-depth-anything --no-enable-cpp-state --no-enable-ros \
     --no-enable-visualization --no-enable-vla-timing
   ```

3. Keep `gear_sonic.utils.pico_video.service` unchanged. Its input remains
   `camera_encoded/ego_view` from SensorGateway.
4. Confirm disconnect, stale-card, and automatic recovery behavior before
   combining the video path with robot teleoperation.

## Troubleshooting

| Symptom | Check |
|---|---|
| `SENSORGATEWAY OFFLINE` | SensorGateway RPC is listening at the endpoint selected by `--profile`/`--overlay`; `--gateway-endpoint` is reserved for `local-test`. |
| `SENSOR FRAME STALE` | The camera publisher stopped, the stream name is missing, or the newest Gateway frame is older than `--max-age-ms` (default 250 ms). |
| `INVALID CAMERA FRAME` | `camera_encoded/ego_view` must be a 1-D uint8 shared-memory array with `encoding=jpeg_bytes` and `image_shape=[H,W,3]`. |
| FFmpeg exits immediately | Run the encoder listing command above; use `--encoder libx264` to distinguish NVENC/driver problems from pipeline problems. |
| `PICO USBOnly RNDIS network not found` | Confirm USB is attached, select USBOnly, and wait for NetworkManager DHCP. Do not create a static replacement profile. |
| PICO cannot connect | Confirm XRoboToolkit control is connected through USBOnly, the bridge reports the current USB workstation address, and another bridge is not already bound to TCP 13579. |
| Bridge cannot connect to PICO video | Keep Remote Vision listening and verify that the discovered PICO USB peer is reachable on TCP 12345. The bridge intentionally does not fall back to Wi-Fi. |
| `OPEN_CAMERA` is rejected | Select `SONIC_HEAD`; the bridge intentionally accepts only 1280x480@30 at the configured 4 Mbps, with a video IP matching the control peer. |
| Image is stretched or double | The bridge duplicates one 640x480 eye image; verify the `SONIC_HEAD` display properties were copied exactly. |

Stop in reverse order: close Remote Vision, stop the bridge, stop
SensorGateway, then stop the camera publisher. Video failures are isolated
from SONIC control throughout this flow.
