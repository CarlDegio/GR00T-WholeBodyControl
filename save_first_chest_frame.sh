#!/usr/bin/env bash

# Run the local screenshot client against a composed camera server on the robot.

set -eu

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    printf '%s\n' "Usage: $0 ROBOT_IP [OUTPUT_PNG] [CAMERA_PORT]"
    printf '%s\n' "Example: $0 192.168.123.164"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if [[ -z "${1:-}" ]]; then
    usage >&2
    exit 2
fi
if (( $# > 3 )); then
    usage >&2
    exit 2
fi

camera_host="$1"
timestamp="$(date +%Y%m%d_%H%M%S)"
output_path="${2:-$script_dir/outputs/camera_startup/${timestamp}_chest_view_rgb.png}"
camera_port="${3:-5555}"

if [[ ! "$camera_port" =~ ^[0-9]+$ ]] \
    || (( camera_port < 1 || camera_port > 65535 )); then
    printf '%s\n' "ERROR: invalid camera port: $camera_port" >&2
    exit 2
fi

if [[ -n "${FIRST_CHEST_FRAME_PYTHON:-}" ]]; then
    python_bin="$FIRST_CHEST_FRAME_PYTHON"
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
    printf '%s\n' "ERROR: no compatible local Python environment was found" >&2
    printf '%s\n' \
        "Set FIRST_CHEST_FRAME_PYTHON to an environment containing OpenCV, NumPy, and pyzmq." >&2
    exit 1
fi

cd "$script_dir"
exec "$python_bin" -m gear_sonic.scripts.save_first_chest_frame \
    --camera-host "$camera_host" \
    --camera-port "$camera_port" \
    --output-path "$output_path"
