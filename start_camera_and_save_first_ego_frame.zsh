#!/usr/bin/env zsh

# Start one of the existing multi-camera servers and save the first ego_view
# RGB frame received after this wrapper starts. The camera server remains in
# the foreground until interrupted, just like the original startup scripts.

set -eu

script_dir="${0:A:h}"
mode="${1:-base_pose}"

case "$mode" in
    base_pose|start_base_pose_camera_server.zsh|./start_base_pose_camera_server.zsh)
        camera_script="$script_dir/start_base_pose_camera_server.zsh"
        ;;
    original|start_camera_server.zsh|./start_camera_server.zsh)
        camera_script="$script_dir/start_camera_server.zsh"
        ;;
    -h|--help)
        print "Usage: $0 [base_pose|original] [OUTPUT_PNG]"
        print ""
        print "Examples:"
        print "  $0 base_pose"
        print "  $0 original outputs/camera_startup/agentnav_ego.png"
        exit 0
        ;;
    *)
        print -u2 "ERROR: camera mode must be 'base_pose' or 'original', got '$mode'"
        exit 2
        ;;
esac

timestamp="$(date +%Y%m%d_%H%M%S)"
output_path="${2:-$script_dir/outputs/camera_startup/${timestamp}_ego_view_rgb.png}"
python_bin="$script_dir/.venv_camera/bin/python"
capture_script="$script_dir/gear_sonic/scripts/save_first_ego_frame.py"

if [[ ! -x "$python_bin" ]]; then
    print -u2 "ERROR: camera Python environment not found: $python_bin"
    exit 1
fi
if [[ ! -x "$camera_script" ]]; then
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
"$python_bin" "$capture_script" \
    --camera-host 127.0.0.1 \
    --camera-port 5555 \
    --output-path "$output_path" \
    --timeout-sec 120 \
    --ready-file "$ready_file" &
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

print "Starting '$mode' camera server; first ego_view RGB will be saved to:"
print "  $output_path"
"$camera_script" &
camera_pid=$!

camera_status=0
wait "$camera_pid" || camera_status=$?
camera_pid=""
exit "$camera_status"
