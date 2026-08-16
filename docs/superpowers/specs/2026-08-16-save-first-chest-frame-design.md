# Save First Chest RGB Frame Design

## Goal

Add a standalone Python utility that saves the first valid chest-camera RGB
frame published by the existing composed camera server.

## Design

Create `gear_sonic/scripts/save_first_chest_frame.py` as the chest-camera
counterpart of `save_first_ego_frame.py`. It connects through
`ComposedCameraClientSensor`, reads `images["chest_view"]`, validates that the
frame is a non-empty `uint8` `HxWx3` NumPy array, converts RGB to BGR for
OpenCV, and saves a lossless PNG.

The command-line interface remains parallel to the existing utility:
`--camera-host`, `--camera-port`, `--output-path`, `--timeout-sec`,
`--ready-file`, and `--overwrite`. The client is always closed, including on
timeouts and malformed input. Existing head-camera files and camera-server
launchers remain unchanged.

## Verification

Add focused unit tests covering chest-stream selection and validation, RGB
channel preservation and overwrite protection, first-frame capture, readiness
signalling, and client cleanup. Run the new test module and Python compilation.

## Non-goals

- Starting or stopping the camera server.
- Changing base-pose or YOLOE runtime behavior.
- Capturing depth images or any stream other than `chest_view` RGB.
