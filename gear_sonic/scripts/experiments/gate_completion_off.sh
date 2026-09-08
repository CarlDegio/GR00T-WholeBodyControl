#!/usr/bin/env bash
set -euo pipefail
EXP_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$EXP_REPO_ROOT"
exec "$EXP_REPO_ROOT/.venv_inference/bin/python" -m gear_sonic.experiments.run gate_completion_off "$@"
