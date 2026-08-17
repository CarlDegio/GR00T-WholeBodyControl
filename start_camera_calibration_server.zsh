#!/usr/bin/env zsh

# One-shot calibration mode: publish complete metadata while streaming the
# same dual RGB-D image set used by normal operation.
"$(dirname "$0")/.venv_camera/bin/python" \
    -m gear_sonic.camera.composed_camera \
    --ego-view-camera orbbec --ego-view-device-id CPMD464001G \
    --chest-camera realsense --chest-device-id 408122070390 \
    --left-wrist-camera realsense --left-wrist-device-id 218622279421 \
    --right-wrist-camera realsense --right-wrist-device-id 352122270966 \
    --orbbec-enable-depth \
    --realsense-enable-depth \
    --realsense-depth-mounts chest_view \
    --publish-camera-info \
    --port 5555
