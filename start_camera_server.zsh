source .venv_camera/bin/activate

python -m gear_sonic.camera.composed_camera \
    --ego-view-camera realsense --ego-view-device-id 347522071257 \
#    --left-wrist-camera realsense --left-wrist-device-id 218622279421 \
#    --right-wrist-camera realsense --right-wrist-device-id 352122270966 \
    --port 5555
