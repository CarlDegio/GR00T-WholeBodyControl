#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION_NAME="${1:-sonic_filter_mujoco}"
BASH_BIN=/usr/bin/bash
BASE_PATH=/home/user/.local/bin:/usr/local/cuda-13.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
READY_FILE=/tmp/sonic_filter_velocity_ready

command -v tmux >/dev/null
for path in \
    "$ROOT_DIR/.venv_sim/bin/activate" \
    "$ROOT_DIR/.venv_teleop/bin/activate" \
    "$ROOT_DIR/gear_sonic/scripts/run_sim_loop.py" \
    "$ROOT_DIR/gear_sonic/scripts/filter_velocity_relay.py" \
    "$ROOT_DIR/gear_sonic_deploy/scripts/setup_env.sh"; do
    [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done

if tmux has-session -t "=$SESSION_NAME" 2>/dev/null; then
    echo "tmux session '$SESSION_NAME' already exists." >&2
    exit 1
fi
rm -f "$READY_FILE"

tmux new-session -d -s "$SESSION_NAME" -n relay "$BASH_BIN" --noprofile --norc
tmux new-window -d -t "=$SESSION_NAME" -n mujoco "$BASH_BIN" --noprofile --norc
tmux new-window -d -t "=$SESSION_NAME" -n deploy "$BASH_BIN" --noprofile --norc

tmux send-keys -t "=$SESSION_NAME:relay" \
    "export PATH='$BASE_PATH'; cd '$ROOT_DIR'; source .venv_teleop/bin/activate; python gear_sonic/scripts/filter_velocity_relay.py --source tcp://127.0.0.1:5558 --output 'tcp://*:5556' --hz 10 --timeout 1.0 --ready-file '$READY_FILE'; rc=\$?; echo '[relay] exited with code' \$rc; exec '$BASH_BIN' --noprofile --norc" C-m

tmux send-keys -t "=$SESSION_NAME:mujoco" \
    "export PATH='$BASE_PATH'; cd '$ROOT_DIR'; source .venv_sim/bin/activate; echo '[mujoco] waiting for first Filter velocity'; while [[ ! -e '$READY_FILE' ]]; do sleep 0.1; done; python gear_sonic/scripts/run_sim_loop.py; rc=\$?; echo '[mujoco] exited with code' \$rc; exec '$BASH_BIN' --noprofile --norc" C-m

tmux send-keys -t "=$SESSION_NAME:deploy" \
    "export PATH='$BASE_PATH'; export TensorRT_ROOT=/usr; export CUDAToolkit_ROOT=/usr/local/cuda-13.0; export onnxruntime_ROOT=/home/user/.local/onnxruntime; export LD_LIBRARY_PATH=/home/user/.local/onnxruntime/lib:/usr/local/cuda-13.0/lib64:\${LD_LIBRARY_PATH:-}; cd '$ROOT_DIR/gear_sonic_deploy'; source scripts/setup_env.sh; just run g1_deploy_onnx_ref lo policy/release/model_decoder.onnx reference/example/ --obs-config policy/release/observation_config.yaml --encoder-file policy/release/model_encoder.onnx --planner-file planner/target_vel/V2/planner_sonic.onnx --input-type zmq_manager --output-type all --zmq-host localhost --zmq-port 5556 --disable-crc-check; rc=\$?; echo '[deploy] exited with code' \$rc; exec '$BASH_BIN' --noprofile --norc" C-m

tmux select-window -t "=$SESSION_NAME:mujoco"
echo "Started tmux session: $SESSION_NAME"
echo "Attach with: tmux attach -t '$SESSION_NAME'"
