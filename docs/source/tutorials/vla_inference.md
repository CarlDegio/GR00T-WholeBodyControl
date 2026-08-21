# VLA Inference

This guide covers running a trained Isaac-GR00T VLA policy on the Unitree G1 robot
using the Sonic whole-body control stack.

## Overview

The inference pipeline consists of:

1. **Isaac-GR00T PolicyServer** — loads the VLA model and serves actions over ZMQ
2. **VLA inference client** (`gear_sonic.utils.inference.vla.service`) — reads camera + robot state,
   queries the PolicyServer, and publishes actions to the C++ control loop
3. **C++ deploy** (`gear_sonic_deploy`) — executes whole-body control on the robot
4. **Camera server** — provides camera images over ZMQ (runs as a systemd service)
5. **Data exporter** (optional) — records episodes during inference

```
┌──────────────────────┐
│  Isaac-GR00T         │
│  PolicyServer        │
│  (GPU machine)       │
└──────┬───────────────┘
       │ ZMQ REQ/REP
       ▼
┌─────────────────────┐    ZMQ TCP    ┌──────────────────────┐
│  VLA Inference      │ ◄─────────── │  Camera Server       │
│  (VLA service)      │              │  (on robot)          │
└────┬───────────┬────┘              └──────────────────────┘
     │           │
     │ ZMQ PUB   │ ZMQ SUB
     │ (actions) │ (state)
     ▼           ▼
┌─────────────────────┐
│  C++ Deploy         │
│  (gear_sonic_deploy)│
└─────────────────────┘
```

## Prerequisites

### 1. Isaac-GR00T PolicyServer

The PolicyServer runs on a machine with a GPU. It loads your finetuned VLA model
and serves inference over ZMQ.

Install [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) and start the server:

```bash
# On the GPU machine (from the Isaac-GR00T repo)
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path /path/to/your/finetuned_model \
    --embodiment-tag UNITREE_G1_SONIC \
    --device cuda:0 \
    --port 5550
```

### 2. Inference Environment

On the inference machine (can be the same as the PolicyServer or a separate PC):

```bash
bash install_scripts/install_inference.sh
```

This creates `.venv_inference` with the Isaac-GR00T PolicyClient and all
inference dependencies.

### 3. Camera Server

The camera server should be running as a systemd service on the robot.
See [Data Collection](data_collection.md) for camera server setup.

The production VLA path reads four `camera_encoded/*` streams from SensorGateway.
Those streams must carry `jpeg_bytes`; legacy Base64 strings are supported only by
the ordinary camera schema decoder and are rejected by VLA ingress. VLA wraps the
existing JPEG bytes in OpenPI's `__opencv_jpeg_rgb__` marker without shape inference,
fallback decode/re-encode, or color correction.

### 4. C++ Deploy

The `gear_sonic_deploy` binary must be built. See the main README.

## Action Space

The Sonic embodiment (`unitree_g1_sonic`) uses a 78-dimensional action
space: 64-dim motion token + 7-dim left hand joints + 7-dim right hand joints.

## Quick Start — tmux Launcher

The easiest way to run inference is with the all-in-one tmux launcher:

Set machine addresses once in `gear_sonic/config/launch_inference.yaml`. Runtime
addresses live only under `endpoints`; for example:

```yaml
endpoints:
  policy_server: {host: 127.0.0.1, port: 29999}
  camera_server: {host: 192.168.123.164, port: 5555}
components:
  vla:
    prompt: pick up the apple
```

```bash
# Real robot
python gear_sonic/scripts/launch_inference.py

# Simulation
python gear_sonic/scripts/launch_inference.py --sim

# Without data recording
python gear_sonic/scripts/launch_inference.py --no-data-exporter
```

The launcher creates two core tmux windows. The default `overview` window keeps
the operator-facing information in three panes:

| Pane | Component | Description |
|------|-----------|-------------|
| Performance (top-left) | SensorGateway | Sensor health and VLA timing |
| Control (bottom-left) | Operator CLI | Type keyboard commands here |
| Events (right) | ControlGateway | Control routing and event output |

The `workers` window preserves six independent worker panes and their full
scrollback: C++ Deploy, VLA Inference, LaViRA/planner input, NavDP + server,
PlannerExecutor, and Base-Pose. Simulation and data recording add optional
windows when enabled.

### Keyboard Controls

Type these keys in the `overview` window's **Control** pane:

| Key | Action |
|-----|--------|
| `k` | Start / stop the C++ control loop |
| `i` | Send initial pose and switch to POSE mode |
| `p` | Pause / resume policy inference |
| `[` | Toggle left hand open/closed (initial pose) |
| `]` | Toggle right hand open/closed (initial pose) |
| `t <text>` | Change the inference prompt (e.g., `t pick up the cup`) |
| `c` | Start recording an episode (data exporter) |
| `e` | Stop recording — success (data exporter) |
| `f` | Stop recording — failure / discard (data exporter) |

### Typical Workflow

1. Wait for both core windows to initialize; the launcher selects
   `overview:control` automatically
2. Press `k` to start the C++ control loop (starts in PLANNER mode)
3. Press `i` to send the initial pose (switches to POSE mode)
   > **Note:** The initial motion token in `gear_sonic/utils/inference/initial_poses.py`
   > is specific to the SONIC checkpoint used during training. If you change the
   > SONIC checkpoint, you must update `LATENT_INITIAL_MOTION_TOKEN` to a safe
   > standing pose from the new checkpoint's latent space.
4. Press `p` to unpause the inference loop
5. The robot will begin executing VLA-predicted actions
6. Press `p` to pause, `k` to stop the control loop when done

Use `Ctrl-b 0` for `overview` and `Ctrl-b 1` for `workers`.

## Manual Setup (Without tmux)

If you prefer to run each component in separate terminals:

### Terminal 1 — Isaac-GR00T PolicyServer (GPU machine)

```bash
# From the Isaac-GR00T repo
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path /path/to/your/finetuned_model \
    --embodiment-tag UNITREE_G1_SONIC \
    --device cuda:0 \
    --port 5550
```

### Terminal 2 — C++ Deploy

```bash
cd gear_sonic_deploy
./deploy.sh --input-type zmq_manager real
```

### Terminal 3 — VLA Inference

```bash
source .venv_inference/bin/activate
python -m gear_sonic.utils.inference.vla.service \
    --profile gear_sonic/config/launch_inference.yaml
```

The selected profile supplies the PolicyServer, SensorGateway, ControlGateway,
planner relay, timing, action endpoints, and every VLA runtime setting under
`components.vla`. Use an `--overlay` file for temporary changes; the production
runner does not accept direct runtime or endpoint overrides.

### Terminal 4 — Data Exporter (optional)

```bash
source .venv_data_collection/bin/activate
python -m gear_sonic.utils.data_collection.service \
    --profile gear_sonic/config/launch_inference.yaml \
    --task-prompt "pick up the apple"
```

## Configuration Reference

### VLA Inference (`gear_sonic.utils.inference.vla.service`)

| Flag | Default | Description |
|------|---------|-------------|
| `--profile` | default runtime profile | YAML source for every communication endpoint |
| `--overlay` | none | Partial YAML overlay; may be repeated |

Set `embodiment_tag`, `prompt`, `action_publish_rate`, `action_horizon`,
`inference_hz`, Gateway cache limits, and `verbose_timing` once under
`components.vla` in the selected profile.

### tmux Launcher (`launch_inference.py`)

The launcher exposes only orchestration switches such as simulation, optional
workers, planner selection, and diagnostics. Process settings live under
`components.*`, while addresses live under `endpoints`. Run
`python gear_sonic/scripts/launch_inference.py --help` for the full list.

## Remote PolicyServer

When running the PolicyServer on a separate GPU machine, update the runtime
profile:

```yaml
endpoints:
  policy_server: {host: <gpu_machine_ip>, port: 5550}
```

Then launch normally with `python gear_sonic/scripts/launch_inference.py`.

Make sure port 5550 (or your chosen port) is accessible between the two machines.

## Latency Compensation

The inference loop automatically compensates for network and compute latency.
When a new action chunk arrives, the system calculates how many actions in the
chunk are already "stale" based on the time elapsed since inference started,
and skips to the appropriate action index. This is controlled by
`components.vla.action_publish_rate` and `components.vla.action_horizon`.
