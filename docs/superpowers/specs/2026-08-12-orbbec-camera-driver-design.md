# Orbbec Camera Driver Design

## Scope

Add `gear_sonic/camera/drivers/orbbec.py` for an Orbbec Gemini 345Lg connected to the G1 by USB. This change only supplies the driver; camera composition and launch registration are separate follow-up changes.

## Interface

`OrbbecConfig` mirrors the existing RealSense configuration: 640×480 color, 640×480 depth, 30 FPS, `ego_view` mounting, and depth enabled by default.

`OrbbecSensor` mirrors `RealSenseSensor` construction, lifecycle, serialization, observation-space, and server behavior. It selects a camera by exact serial number when `device_id` is provided, otherwise by stable serial-sorted index. The selected serial is exposed as `sensor.serial_number` and logged at startup.

`read()` returns exactly the existing camera protocol keys:

- `timestamps`: one shared host timestamp under `<mount>` and `<mount>_depth`.
- `images`: RGB `uint8[H,W,3]` under `<mount>` and color-aligned depth `uint16[H,W]` under `<mount>_depth`.
- `camera_info`: color intrinsics and depth scale under `<mount>`, with the same field names used by `RealSenseSensor`.

No serial-number field is added to `read()` because that would change the RealSense-compatible wire contract.

## SDK Data Flow

Use the official Orbbec SDK v2 Python package (`pyorbbecsdk2`, imported as `pyorbbecsdk`). Configure an RGB stream and a depth stream, require complete frame sets, and apply software depth-to-color alignment with `AlignFilter(OBStreamType.COLOR_STREAM)`.

Gemini 345Lg reports USB PID `0x0813`, which the official v2 example classifies as a Dabai A-series device. For that family, apply `UnDistortionFilter` to the color stream before alignment when the installed SDK exposes that newer API. Older SDK v2 builds that do not expose the filter continue with the stable `AlignFilter` path.

Color frames are requested as `OBFormat.RGB`, so their byte buffers are reshaped without a BGR conversion. Depth buffers remain raw `uint16`; metric depth is computed by consumers using `depth_scale_m = depth_frame.get_depth_scale() * 0.001`, converting the SDK's millimetres-per-unit scale to metres-per-unit as required by the existing protocol.

## Errors and Cleanup

Construction raises descriptive errors for no camera, an unknown serial, an invalid index, unsupported profiles, or pipeline startup failure. `read()` logs and returns `None` for timeouts, incomplete/invalid frames, alignment failures, conversion failures, empty buffers, or mismatched color/depth dimensions. `close()` stops the optional server first and then the camera pipeline.

## Tests

Tests replace only the external compiled SDK module with a complete fake and exercise the real driver. They verify serial selection, aligned frame use, RGB/depth dtype and shape, color intrinsics, metre depth scale, protocol-compatible serialization, and resource cleanup. The production mutations these tests must catch include publishing raw depth, converting RGB to BGR, returning metric float depth instead of raw `uint16`, using depth rather than color intrinsics, and reporting millimetres rather than metres in `depth_scale_m`.
