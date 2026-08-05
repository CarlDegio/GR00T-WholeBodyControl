#!/usr/bin/env zsh

# Save the first received ego_view RGB frame. Local modes also start one of the
# existing camera servers; remote mode only subscribes to a server on the robot
# and writes the image on this deployment machine.

set -eu

script_dir="${0:A:h}"
mode="${1:-base_pose}"
camera_host="127.0.0.1"
camera_port="5555"
camera_script=""
launch_local_camera=true
output_argument=""

case "$mode" in
    base_pose|start_base_pose_camera_server.zsh|./start_base_pose_camera_server.zsh)
        camera_script="$script_dir/start_base_pose_camera_server.zsh"
        output_argument="${2:-}"
        ;;
    original|start_camera_server.zsh|./start_camera_server.zsh)
        camera_script="$script_dir/start_camera_server.zsh"
        output_argument="${2:-}"
        ;;
    remote)
        if [[ -z "${2:-}" ]]; then
            print -u2 "ERROR: remote mode requires the robot camera-server hostname or IP"
            print -u2 "Usage: $0 remote ROBOT_IP [OUTPUT_PNG] [CAMERA_PORT]"
            exit 2
        fi
        launch_local_camera=false
        camera_host="$2"
        output_argument="${3:-}"
        camera_port="${4:-5555}"
        ;;
    -h|--help)
        print "Usage:"
        print "  $0 [base_pose|original] [OUTPUT_PNG]"
        print "  $0 remote ROBOT_IP [OUTPUT_PNG] [CAMERA_PORT]"
        print ""
        print "Examples:"
        print "  $0 base_pose"
        print "  $0 original outputs/camera_startup/agentnav_ego.png"
        print "  $0 remote 192.168.123.164"
        exit 0
        ;;
    *)
        print -u2 "ERROR: camera mode must be 'base_pose', 'original', or 'remote', got '$mode'"
        exit 2
        ;;
esac

timestamp="$(date +%Y%m%d_%H%M%S)"
output_path="${output_argument:-$script_dir/outputs/camera_startup/${timestamp}_ego_view_rgb.png}"

if [[ "$camera_port" != <-> ]] || (( camera_port < 1 || camera_port > 65535 )); then
    print -u2 "ERROR: invalid camera port: $camera_port"
    exit 2
fi

if [[ -n "${FIRST_EGO_FRAME_PYTHON:-}" ]]; then
    python_bin="$FIRST_EGO_FRAME_PYTHON"
elif $launch_local_camera; then
    python_bin="$script_dir/.venv_camera/bin/python"
else
    python_bin=""
    for candidate in \
        "$script_dir/.venv_inference/bin/python" \
        "$script_dir/.venv_data_collection/bin/python" \
        "$script_dir/.venv_camera/bin/python"; do
        if [[ -x "$candidate" ]]; then
            python_bin="$candidate"
            break
        fi
    done
fi

if [[ -z "$python_bin" || ! -x "$python_bin" ]]; then
    print -u2 "ERROR: no compatible Python environment was found"
    print -u2 "Set FIRST_EGO_FRAME_PYTHON to an environment containing OpenCV, NumPy, and pyzmq."
    exit 1
fi
if $launch_local_camera && [[ ! -x "$camera_script" ]]; then
    print -u2 "ERROR: camera startup script is not executable: $camera_script"
    exit 1
fi

sync_dir="$(mktemp -d "${TMPDIR:-/tmp}/g1-first-ego-frame.XXXXXX")"
ready_file="$sync_dir/subscriber_ready"
capture_pid=""
camera_pid=""

cleanup() {
    if [[ -n "$capture_pid" ]] && kill -0 "$capture_pid" 2>/dev/null; then
        kill "$capture_pid" 2>/dev/null || true
        wait "$capture_pid" 2>/dev/null || true
    fi
    if [[ -n "$camera_pid" ]] && kill -0 "$camera_pid" 2>/dev/null; then
        kill "$camera_pid" 2>/dev/null || true
        wait "$camera_pid" 2>/dev/null || true
    fi
    [[ ! -e "$ready_file" ]] || rm -f -- "$ready_file"
    [[ ! -d "$sync_dir" ]] || rmdir -- "$sync_dir" 2>/dev/null || true
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

print "Preparing ego_view first-frame subscriber..."
(
    cd "$script_dir"
    exec "$python_bin" -m gear_sonic.scripts.save_first_ego_frame \
        --camera-host "$camera_host" \
        --camera-port "$camera_port" \
        --output-path "$output_path" \
        --timeout-sec 120 \
        --ready-file "$ready_file"
) &
capture_pid=$!

attempt=0
while [[ ! -f "$ready_file" ]]; do
    if ! kill -0 "$capture_pid" 2>/dev/null; then
        wait "$capture_pid" || true
        print -u2 "ERROR: first-frame subscriber exited before it became ready"
        exit 1
    fi
    attempt=$((attempt + 1))
    if (( attempt >= 50 )); then
        print -u2 "ERROR: first-frame subscriber was not ready within 5 seconds"
        exit 1
    fi
    sleep 0.1
done

if ! $launch_local_camera; then
    print "Waiting for the first ego_view RGB frame from $camera_host:$camera_port ..."
    print "The image will be stored on this deployment machine at:"
    print "  $output_path"
    capture_status=0
    wait "$capture_pid" || capture_status=$?
    capture_pid=""
    exit "$capture_status"
fi

print "Starting '$mode' camera server; first ego_view RGB will be saved to:"
print "  $output_path"
"$camera_script" &
camera_pid=$!

camera_status=0
wait "$camera_pid" || camera_status=$?
camera_pid=""
exit "$camera_status"
