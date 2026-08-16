# Save First Chest RGB Frame Design

## Goal

Add a standalone Python utility and local launcher that save the first valid
chest-camera RGB frame published by the composed camera server running on the
robot.

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

Create `save_first_chest_frame.sh` at the repository root. It requires the
robot hostname or IP, accepts optional output path and port arguments, and only
starts the local screenshot client; it never starts or stops a camera server.

## Verification

Add focused unit tests covering chest-stream selection and validation, RGB
channel preservation and overwrite protection, first-frame capture, readiness
signalling, client cleanup, and launcher argument forwarding. Run the new test
module, launcher help, and Python compilation.

## Non-goals

- Starting or stopping the camera server.
- Changing base-pose or YOLOE runtime behavior.
- Capturing depth images or any stream other than `chest_view` RGB.
