#!/usr/bin/env bash
# Install TensorRT 10.13.3 from the local-repo DEB and configure TensorRT_ROOT
# for gear_sonic_deploy (CMake expects $TensorRT_ROOT/include + /lib).
#
# Usage (requires sudo password):
#   bash gear_sonic_deploy/scripts/install_tensorrt_from_deb.sh
#
# Default DEB path matches bb0027's download; override with TENSORRT_DEB=...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEB="${TENSORRT_DEB:-$HOME/Downloads/nv-tensorrt-local-repo-ubuntu2204-10.13.3-cuda-13.0_1.0-1_amd64.deb}"
REPO_VAR="/var/nv-tensorrt-local-repo-ubuntu2204-10.13.3-cuda-13.0"
TENSORRT_ROOT="${TENSORRT_ROOT:-$HOME/TensorRT}"

if [ ! -f "$DEB" ]; then
    echo "[ERROR] DEB not found: $DEB"
    exit 1
fi

echo "[1/4] Register local apt repo …"
sudo dpkg -i "$DEB"

echo "[2/4] Import GPG key …"
sudo cp "${REPO_VAR}/"*-keyring.gpg /usr/share/keyrings/ 2>/dev/null || \
    sudo cp "${REPO_VAR}/2F9B36B1.pub" /usr/share/keyrings/nv-tensorrt-local-ubuntu2204-10.13.3-cuda-13.0.gpg

echo "[3/4] Install TensorRT packages (≈4 GB, may take several minutes) …"
sudo apt-get update
sudo apt-get install -y \
    libnvinfer10 \
    libnvinfer-dev \
    libnvinfer-plugin10 \
    libnvinfer-plugin-dev \
    libnvonnxparsers10 \
    libnvonnxparsers-dev \
    libnvinfer-bin

echo "[4/4] Configure TensorRT_ROOT at ${TENSORRT_ROOT} …"
INCLUDE_SRC="/usr/include/x86_64-linux-gnu"
LIB_SRC="/usr/lib/x86_64-linux-gnu"

if [ ! -f "${INCLUDE_SRC}/NvInfer.h" ]; then
    echo "[ERROR] ${INCLUDE_SRC}/NvInfer.h not found after install."
    exit 1
fi

mkdir -p "${TENSORRT_ROOT}/include" "${TENSORRT_ROOT}/lib"

# Headers
for h in "${INCLUDE_SRC}"/NvInfer*.h "${INCLUDE_SRC}"/NvOnnxParser*.h; do
    [ -e "$h" ] || continue
    ln -sfn "$h" "${TENSORRT_ROOT}/include/$(basename "$h")"
done

# Libraries required by gear_sonic_deploy FindTensorRT.cmake
for pattern in libnvinfer libnvinfer_plugin libnvonnxparser; do
    for lib in "${LIB_SRC}/${pattern}"*.so*; do
        [ -e "$lib" ] || continue
        ln -sfn "$lib" "${TENSORRT_ROOT}/lib/$(basename "$lib")"
    done
done

BASHRC="${HOME}/.bashrc"
MARKER="# >>> TensorRT (gear_sonic_deploy) >>>"
if ! grep -qF "$MARKER" "$BASHRC" 2>/dev/null; then
    cat >> "$BASHRC" <<EOF

${MARKER}
export TensorRT_ROOT="${TENSORRT_ROOT}"
export LD_LIBRARY_PATH="\${TensorRT_ROOT}/lib:\${LD_LIBRARY_PATH:-}"
# <<< TensorRT (gear_sonic_deploy) <<<
EOF
    echo "Appended TensorRT_ROOT to ~/.bashrc"
else
    echo "TensorRT block already present in ~/.bashrc"
fi

export TensorRT_ROOT="${TENSORRT_ROOT}"
export LD_LIBRARY_PATH="${TENSORRT_ROOT}/lib:${LD_LIBRARY_PATH:-}"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  TensorRT 10.13.3 installed and TensorRT_ROOT set."
echo ""
echo "  Verify:"
echo "    source ~/.bashrc"
echo "    ls \$TensorRT_ROOT/include/NvInfer.h"
echo "    ls \$TensorRT_ROOT/lib/libnvinfer.so*"
echo ""
echo "  Next:"
echo "    cd gear_sonic_deploy && ./scripts/install_deps.sh"
echo "    source scripts/setup_env.sh"
echo "    cmake -B build -S . && cmake --build build -j\$(nproc)"
echo "════════════════════════════════════════════════════════"
