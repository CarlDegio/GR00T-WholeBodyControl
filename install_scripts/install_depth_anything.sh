#!/usr/bin/env bash
# Install the isolated runtime used by the RGB-only metric depth sidecar.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DA_ROOT="/home/user/Project/Depth-Anything-V2"
METRIC_ROOT="$DA_ROOT/metric_depth"
CHECKPOINT_DIR="$METRIC_ROOT/checkpoints"
CHECKPOINT="$CHECKPOINT_DIR/depth_anything_v2_metric_hypersim_vitb.pth"
CHECKPOINT_URL="https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Base/resolve/main/depth_anything_v2_metric_hypersim_vitb.pth?download=true"

if ! command -v uv >/dev/null 2>&1; then
    echo "[ERROR] uv is required; install it before running this script." >&2
    exit 1
fi

if [ ! -d "$METRIC_ROOT/depth_anything_v2" ]; then
    echo "[INFO] Cloning the official Depth Anything V2 source into $DA_ROOT"
    git clone https://github.com/DepthAnything/Depth-Anything-V2 "$DA_ROOT"
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "[INFO] Downloading the official indoor Metric Base checkpoint"
    mkdir -p "$CHECKPOINT_DIR"
    curl --fail --location "$CHECKPOINT_URL" --output "$CHECKPOINT"
fi

cd "$REPO_ROOT"
if [ ! -x .venv_depth_anything/bin/python ]; then
    uv python install 3.10
    MANAGED_PY="$(uv python find --no-project 3.10)"
    uv venv .venv_depth_anything --python "$MANAGED_PY" \
        --prompt gear_sonic_depth_anything
fi

uv pip install --python .venv_depth_anything/bin/python \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv_depth_anything/bin/python \
    numpy==1.26.4 opencv-python pyzmq msgpack tyro

echo "[OK] Depth Anything source: $METRIC_ROOT"
echo "[OK] Metric Base checkpoint: $CHECKPOINT"
echo "[OK] Runtime: $REPO_ROOT/.venv_depth_anything"
