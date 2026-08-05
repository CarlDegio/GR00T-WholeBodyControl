#!/usr/bin/env zsh

# Base-pose adjustment uses aligned depth from the head/ego RealSense. Keep the
# existing start_camera_server.zsh unchanged for chest-depth AgentNav runs.
"$(dirname "$0")/.venv_camera/bin/python" -m gear_sonic.camera.composed_camera \
    --ego-view-camera realsense --ego-view-device-id 347522071257 \
    --chest-camera realsense --chest-device-id 408122070390 \
    --left-wrist-camera realsense --left-wrist-device-id 218622279421 \
    --right-wrist-camera realsense --right-wrist-device-id 352122270966 \
    --realsense-enable-depth \
    --realsense-depth-mount ego_view \
    --port 5555
