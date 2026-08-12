# Local Gemini Camera Server Design

## Goal

Run the USB-connected Orbbec Gemini 345Lg on the current local host through
the same `gear_sonic.camera.composed_camera` server used by the robot, while
starting no RealSense, chest, or wrist cameras. The server must publish both
RGB and depth aligned to the RGB coordinate system.

## Chosen Approach

Keep the existing four-RealSense `start_camera_server.zsh` unchanged and add a
dedicated `start_gemini_camera_server.zsh`. The new script invokes
`.venv_camera/bin/python -m gear_sonic.camera.composed_camera`, configures only
the `ego_view` mount as `orbbec`, enables Orbbec depth, and uses ZMQ port 5555.

This is preferred over replacing the existing launcher because it preserves
the current robot deployment. A direct standalone driver test is useful for
diagnostics but is not the final launch path because it would bypass the
composed-camera server contract.

## Components and Data Flow

1. `.venv_camera` contains Python 3.10, the existing camera-server
   dependencies, and the official `pyorbbecsdk2` package (imported as
   `pyorbbecsdk`).
2. `ComposedCameraConfig` exposes an Orbbec depth toggle, and
   `ComposedCameraSensor._instantiate_camera()` maps camera type `orbbec` to
   `OrbbecSensor`.
3. `start_gemini_camera_server.zsh` discovers the one connected Gemini serial
   through the SDK and rejects zero or multiple Orbbec devices rather than
   selecting an ambiguous device.
4. The launcher starts one `ego_view` worker. `OrbbecSensor` requests RGB and
   Y16 streams, aligns depth to color, and returns the existing RealSense-
   compatible payload (`timestamps`, `images`, and `camera_info`).
5. `ComposedCameraSensor` serializes that payload and publishes it on port
   5555 exactly as it does for the existing camera types.

## Failure Handling

The launcher fails early with a clear message if `.venv_camera` is missing,
the SDK cannot be imported, no Gemini is connected, more than one Orbbec
device is connected, or the negotiated USB speed is below 5000 Mb/s. The
camera worker retains the existing retry and reconnect behavior for runtime
frame failures.

The USB-speed check reads the selected device's sysfs node using VID `2bc5`
and PID `0813`; the expected local connection is 5000 Mb/s. It does not modify
udev rules or system USB configuration.

## Verification

Automated tests cover Orbbec registration, propagation of the depth option,
and the dedicated launcher's single-camera arguments without opening physical
hardware. Existing camera protocol and Orbbec driver tests remain green.

The final hardware test must prove all of the following on the current host:

- Gemini 345Lg enumerates at 5000 Mb/s.
- The SDK opens the selected serial.
- At least one RGB frame and one aligned depth frame are returned.
- RGB has shape `H x W x 3` and dtype `uint8`.
- Aligned depth has shape `H x W` and dtype `uint16`, matching RGB spatial
  dimensions.
- Camera intrinsics and `depth_scale_m` are present.
- The composed-camera server becomes ready with exactly `ego_view` configured.

The server is stopped cleanly after the bounded validation run; the launcher
remains available for normal long-running use.
