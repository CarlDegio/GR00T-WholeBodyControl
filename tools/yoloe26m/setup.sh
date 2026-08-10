#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv_inference"
PYTHON_BIN="${VENV_DIR}/bin/python"
MODEL_DIR="${SCRIPT_DIR}/weights"
ASSET_DIR="${SCRIPT_DIR}/assets"
CONFIG_DIR="${SCRIPT_DIR}/config"
DATASET_DIR="${SCRIPT_DIR}/datasets"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"

command -v uv >/dev/null 2>&1 || {
    echo "ERROR: uv is required but was not found in PATH." >&2
    exit 1
}

mkdir -p "${MODEL_DIR}" "${ASSET_DIR}" "${CONFIG_DIR}" "${DATASET_DIR}" "${OUTPUT_DIR}"

[[ -x "${PYTHON_BIN}" ]] || {
    echo "ERROR: expected existing environment at ${VENV_DIR}." >&2
    exit 1
}


uv pip install --python "${PYTHON_BIN}" ultralytics==8.4.117

# YOLOE text prompting imports Ultralytics' maintained CLIP fork. Installing it
# now prevents a surprise package installation during the first inference.
uv pip install \
    --python "${PYTHON_BIN}" \
    "git+https://github.com/ultralytics/CLIP.git@488e81a6711eea7346872b46ea928b367da8889d"

curl \
    --fail \
    --location \
    --retry 5 \
    --retry-delay 2 \
    --continue-at - \
    --output "${MODEL_DIR}/yoloe-26m-seg.pt" \
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26m-seg.pt"

curl \
    --fail \
    --location \
    --retry 5 \
    --retry-delay 2 \
    --continue-at - \
    --output "${ASSET_DIR}/bus.jpg" \
    "https://ultralytics.com/images/bus.jpg"

YOLO_CONFIG_DIR="${CONFIG_DIR}" "${PYTHON_BIN}" - \
    "${MODEL_DIR}" "${DATASET_DIR}" "${OUTPUT_DIR}" <<'PY'
from pathlib import Path
import sys

from ultralytics import settings

model_dir, dataset_dir, output_dir = map(lambda value: str(Path(value).resolve()), sys.argv[1:])
settings.update(
    {
        "weights_dir": model_dir,
        "datasets_dir": dataset_dir,
        "runs_dir": output_dir,
        "sync": False,
    }
)
print("Ultralytics settings:", settings)
PY

# This first prediction also downloads and caches the text encoder, so later
# offline runs have all files they need.
YOLO_CONFIG_DIR="${CONFIG_DIR}" "${PYTHON_BIN}" "${SCRIPT_DIR}/infer.py" \
    "${ASSET_DIR}/bus.jpg" \
    --classes person bus \
    --name setup_smoke_test \
    --exist-ok

echo
echo "YOLOE-26M setup and smoke test completed successfully."
echo "Activate with: source ${VENV_DIR}/bin/activate"
