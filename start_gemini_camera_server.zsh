#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
PYTHON="$SCRIPT_DIR/.venv_camera/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    print -u2 "ERROR: missing $PYTHON"
    print -u2 "Create .venv_camera and install pyorbbecsdk2 first."
    exit 1
fi

exec "$PYTHON" -m gear_sonic.camera.gemini_server_launcher
