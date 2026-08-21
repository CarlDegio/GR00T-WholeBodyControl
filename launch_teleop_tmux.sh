#!/usr/bin/env bash
#
# Launch the MuJoCo + SONIC deploy + PICO teleoperation stack in tmux.
# Every tmux window explicitly runs a clean Bash, even when this script is
# invoked from zsh, Cursor, or another shell.
#
# Usage:
#   ./launch_teleop_tmux.sh
#       Start the three windows in tmux session "groot_teleop".
#
#   ./launch_teleop_tmux.sh my_session
#       Start them in a custom tmux session named "my_session".
#
#   ./launch_teleop_tmux.sh --record
#   ./launch_teleop_tmux.sh my_session --record
#       Enable temporary diagnostic recording. The script prints the generated
#       /tmp/groot_teleop_<timestamp> directory after startup. It contains PICO
#       pose batches, deploy state CSV files, and the policy-input CSV. SONIC
#       streamed/planner motions are additionally recorded under
#       gear_sonic_deploy/reference/recorded_motion/<date>/.
#
# Windows created:
#   0  mujoco  - MuJoCo simulator with the live SMPL tracking overlay
#   1  deploy  - SONIC C++ policy deployment in simulation mode
#   2  pico    - PICO ZMQ manager and tracking streamer
#
# View the windows:
#   tmux attach -t groot_teleop       # use the custom name when specified
#   Ctrl-b, then 0                    # switch to MuJoCo
#   Ctrl-b, then 1                    # switch to deploy
#   Ctrl-b, then 2                    # switch to PICO
#   Ctrl-b, then d                    # detach without stopping processes
#
# Inspect a window without attaching:
#   tmux capture-pane -p -t groot_teleop:mujoco
#   tmux capture-pane -p -t groot_teleop:deploy
#   tmux capture-pane -p -t groot_teleop:pico
#
# Stop this teleoperation stack:
#   tmux kill-session -t groot_teleop
# Be careful to use the correct session name so unrelated training sessions
# are not stopped.
#
# Options:
#   --record       Enable temporary state/SMPL/action recording for diagnosis.
#   session-name   Optional single positional argument; default: groot_teleop.
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION_NAME="groot_teleop"
SESSION_NAME_SET=false
RECORD_MODE=""
BASH_BIN="/usr/bin/bash"
BASE_PATH="/home/user/.local/bin:/usr/local/cuda-13.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
RECORD_ROOT=""
PICO_RECORD_ARGS=""
DEPLOY_RECORD_ARGS=""

for arg in "$@"; do
    case "$arg" in
        --record)
            RECORD_MODE="--record"
            ;;
        -*)
            echo "Unknown option: $arg" >&2
            echo "Usage: $0 [session-name] [--record]" >&2
            exit 2
            ;;
        *)
            if [[ "$SESSION_NAME_SET" == true ]]; then
                echo "Only one session name may be specified." >&2
                exit 2
            fi
            SESSION_NAME="$arg"
            SESSION_NAME_SET=true
            ;;
    esac
done

if [[ "$RECORD_MODE" == "--record" ]]; then
    RECORD_ROOT="/tmp/groot_teleop_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$RECORD_ROOT/pico" "$RECORD_ROOT/deploy"
    PICO_RECORD_ARGS="--record_dir $RECORD_ROOT/pico"
    DEPLOY_RECORD_ARGS="--enable-motion-recording --enable-csv-logs --logs-dir $RECORD_ROOT/deploy --policy-input-logfile $RECORD_ROOT/policy_input.csv"
fi

require_path() {
    if [[ ! -e "$1" ]]; then
        echo "Missing required path: $1" >&2
        exit 1
    fi
}

command -v tmux >/dev/null 2>&1 || {
    echo "tmux is not installed or not in PATH." >&2
    exit 1
}

require_path "$ROOT_DIR/.venv_sim/bin/activate"
require_path "$ROOT_DIR/.venv_teleop/bin/activate"
require_path "$ROOT_DIR/gear_sonic/utils/mujoco_sim/service.py"
require_path "$ROOT_DIR/gear_sonic/utils/teleop/pico_manager.py"
require_path "$ROOT_DIR/gear_sonic_deploy/deploy.sh"
require_path "$ROOT_DIR/gear_sonic_deploy/scripts/setup_env.sh"

if tmux has-session -t "=$SESSION_NAME" 2>/dev/null; then
    echo "tmux session '$SESSION_NAME' already exists." >&2
    echo "Attach with: tmux attach -t '$SESSION_NAME'" >&2
    exit 1
fi

# Every window starts an explicit clean Bash. This avoids zsh's special `path`
# array changing PATH while gear_sonic_deploy/scripts/setup_env.sh is sourced.
tmux new-session -d -s "$SESSION_NAME" -n mujoco \
    "$BASH_BIN" --noprofile --norc
tmux new-window -d -t "=$SESSION_NAME" -n deploy \
    "$BASH_BIN" --noprofile --norc
tmux new-window -d -t "=$SESSION_NAME" -n pico \
    "$BASH_BIN" --noprofile --norc

tmux send-keys -t "=$SESSION_NAME:mujoco" \
    "export PATH='$BASE_PATH'; cd '$ROOT_DIR'; source .venv_sim/bin/activate; python -m gear_sonic.utils.mujoco_sim.service --show-smpl-tracking; rc=\$?; echo '[mujoco] exited with code' \$rc; exec '$BASH_BIN' --noprofile --norc" C-m

tmux send-keys -t "=$SESSION_NAME:deploy" \
    "export PATH='$BASE_PATH'; export TensorRT_ROOT=/usr; export CUDAToolkit_ROOT=/usr/local/cuda-13.0; export onnxruntime_ROOT=/home/user/.local/onnxruntime; export LD_LIBRARY_PATH=/home/user/.local/onnxruntime/lib:/usr/local/cuda-13.0/lib64:\${LD_LIBRARY_PATH:-}; cd '$ROOT_DIR/gear_sonic_deploy'; source scripts/setup_env.sh; just run g1_deploy_onnx_ref lo policy/release/model_decoder.onnx reference/example/ --obs-config policy/release/observation_config.yaml --encoder-file policy/release/model_encoder.onnx --planner-file planner/target_vel/V2/planner_sonic.onnx --input-type zmq_manager --output-type all --zmq-host localhost --disable-crc-check $DEPLOY_RECORD_ARGS; rc=\$?; echo '[deploy] exited with code' \$rc; exec '$BASH_BIN' --noprofile --norc" C-m

tmux send-keys -t "=$SESSION_NAME:pico" \
    "export PATH='$BASE_PATH'; cd '$ROOT_DIR'; source .venv_teleop/bin/activate; python -m gear_sonic.utils.teleop.pico_manager --manager --vis_vr3pt --vis_smpl $PICO_RECORD_ARGS; rc=\$?; echo '[pico] exited with code' \$rc; exec '$BASH_BIN' --noprofile --norc" C-m

tmux select-window -t "=$SESSION_NAME:deploy"

echo "Started tmux session: $SESSION_NAME"
echo "  0 mujoco  - simulator with SMPL overlay"
echo "  1 deploy  - SONIC C++ deployment"
echo "  2 pico    - PICO ZMQ manager"
echo
echo "Attach: tmux attach -t '$SESSION_NAME'"
echo "Switch windows inside tmux: Ctrl-b then 0, 1, or 2"
if [[ -n "$RECORD_ROOT" ]]; then
    echo
    echo "Diagnostic recording directory: $RECORD_ROOT"
    echo "$RECORD_ROOT" > "/tmp/${SESSION_NAME}_recording_path"
fi
