#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv_inference"
PYTHON_BIN="${VENV_DIR}/bin/python"
MODEL_DIR="${SCRIPT_DIR}/weights"
CONFIG_DIR="${SCRIPT_DIR}/config"
DATASET_DIR="${SCRIPT_DIR}/datasets"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"
YOLOE_MODEL="${MODEL_DIR}/yoloe-26m-seg.pt"
MOBILECLIP_MODEL="${REPO_ROOT}/mobileclip2_b.ts"

command -v uv >/dev/null 2>&1 || {
    echo "ERROR: uv is required but was not found in PATH." >&2
    exit 1
}

[[ -x "${PYTHON_BIN}" ]] || {
    echo "ERROR: expected existing environment at ${VENV_DIR}." >&2
    exit 1
}

mkdir -p "${MODEL_DIR}" "${CONFIG_DIR}" "${DATASET_DIR}" "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

uv pip install \
    --python "${PYTHON_BIN}" \
    "ultralytics==8.4.117" \
    "git+https://github.com/ultralytics/CLIP.git@488e81a6711eea7346872b46ea928b367da8889d"

if [[ ! -f "${YOLOE_MODEL}" ]]; then
    curl \
        --fail \
        --location \
        --retry 5 \
        --retry-delay 2 \
        --continue-at - \
        --output "${YOLOE_MODEL}" \
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26m-seg.pt"
fi

# Configure Ultralytics' local paths, then exercise text prompting once so the
# repository-root runtime cache is populated before BasePose starts.
YOLO_CONFIG_DIR="${CONFIG_DIR}" "${PYTHON_BIN}" - \
    "${YOLOE_MODEL}" "${MODEL_DIR}" "${DATASET_DIR}" "${OUTPUT_DIR}" <<'PY'
from pathlib import Path
import sys

from ultralytics import YOLOE, settings

model_path, weights_dir, datasets_dir, output_dir = map(
    lambda value: str(Path(value).resolve()),
    sys.argv[1:],
)
settings.update(
    {
        "weights_dir": weights_dir,
        "datasets_dir": datasets_dir,
        "runs_dir": output_dir,
        "sync": False,
    }
)
model = YOLOE(model_path)
embeddings = model.get_text_pe(["person", "desk"])
if embeddings.ndim != 3 or embeddings.shape[1] != 2:
    raise RuntimeError(f"unexpected YOLOE text embedding shape: {tuple(embeddings.shape)}")
print("YOLOE CLIP text encoder ready:", tuple(embeddings.shape))
PY

[[ -f "${MOBILECLIP_MODEL}" ]] || {
    echo "ERROR: MobileCLIP2 was not cached at ${MOBILECLIP_MODEL}." >&2
    exit 1
}

echo "YOLOE and CLIP setup completed successfully."
