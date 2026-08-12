# Right-Wrist OpenCV Rotation Design

## Goal

Correct the upside-down right-wrist camera image in the unified OpenCV operator
viewer by rotating that displayed image 180 degrees.

## Scope

The correction applies only to the `camera/right_wrist` stream inside
`gear_sonic/scripts/run_operator_cv_viewer.py`. The SensorGateway payload,
camera server output, VLA/model input, shared-memory data, and recorded dataset
remain byte-for-byte unchanged.

Head, chest, and left-wrist camera displays retain their current orientation.
Navigation, head RGB-D, and LingBot visualization panels are unaffected.

## Data Flow

`gateway_frame_to_bgr()` remains the display decoding boundary. It first
validates the RGB camera payload and converts RGB to OpenCV BGR. When and only
when the stream name is `camera/right_wrist`, it then rotates the decoded BGR
array with `cv2.ROTATE_180` before the frame enters the viewer's local cache and
canvas composer.

No rotation flag is added to the camera driver or runtime configuration because
the physical mounting correction is required only for operator visualization.

## Testing

An asymmetric 2-by-2 RGB fixture proves that `camera/right_wrist` is both
converted to BGR and rotated 180 degrees. A matching asymmetric fixture for
another camera stream proves that RGB-to-BGR conversion still occurs without
rotation. Existing canvas composition tests continue to verify all seven views.

Only these files change:

- `gear_sonic/scripts/run_operator_cv_viewer.py`
- `gear_sonic/tests/test_operator_cv_viewer.py`
