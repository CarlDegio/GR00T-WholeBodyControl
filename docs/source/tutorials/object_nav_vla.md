# ObjectNav to VLA Workflow

This workflow uses the composed chest RGB-D stream to navigate toward an
object with Codex, then hands control to the SONIC VLA inference client.

## Prerequisites

- Start the composed-camera server with aligned `chest_view_depth` enabled.
- Start the Isaac-GR00T PolicyServer used by `run_vla_inference.py`.
- Build `gear_sonic_deploy` and install `.venv_inference`.
- Log the Codex CLI into a ChatGPT subscription with `codex login`.

ObjectNav gives Codex the proxy defaults `http://127.0.0.1:7897/` for HTTP(S)
and `socks://127.0.0.1:7897/` for catch-all traffic. Override or disable them
with `OBJECT_NAV_CODEX_HTTP_PROXY` and `OBJECT_NAV_CODEX_ALL_PROXY`; an empty
value disables that proxy group.

## Launch the Combined Workflow

Repeated navigation continues until Codex explicitly returns `STOP`:

```bash
python gear_sonic/scripts/launch_object_nav_vla.py \
  --mission "walk to the red chair, then pick up the bottle" \
  --global-target "red chair" \
  --vla-prompt "pick up the bottle" \
  --navigation-mode repeat-until-stop \
  --camera-host 192.168.123.164 \
  --policy-host localhost
```

For exactly one ObjectNav command batch before VLA handoff, use:

```bash
python gear_sonic/scripts/launch_object_nav_vla.py \
  --mission "approach the table" \
  --global-target "table" \
  --navigation-mode once \
  --camera-host 192.168.123.164
```

The launcher opens four tmux panes for C++ deploy, keyboard input,
ObjectNav/planner logs, and VLA inference. Confirm the C++ deploy prompt, then
type `N` in the keyboard pane. No image capture, Codex call, or C++ start
command occurs before `N` is received.

After successful navigation, the ObjectNav publisher first aborts any unfinished
motion to zero velocity, confirms `start=true, planner=true`, and releases action
port 5556 without stopping the C++ loop. The VLA pane starts
`run_vla_inference.py` paused while inheriting that running PLANNER state. In the
keyboard pane, type:

1. `i` to publish the initial pose and switch from PLANNER to POSE mode.
2. `p` to unpause VLA inference.

The default ObjectNav motion settings intentionally match the selected source
workspace: zero stopping margin and an eight-metre maximum direct travel.
Override them with `--safe-distance` and `--max-direct-travel` when needed.

## Run One ObjectNav Cycle Directly

With the controlled planner server already running in PLANNER mode:

```bash
python gear_sonic/scripts/run_object_nav.py \
  --mission "find the chair" \
  --global-target "chair" \
  --camera-host 192.168.123.164 \
  --execute-sonic
```

The command prints only the planner-compatible `{"commands": [...]}` object
to stdout. RGB input, five lossless raw depth frames, camera calibration,
Codex policy, fused geometry, commands, and bbox visualization are written
under `outputs/object_nav/`.

Failures remain fail-closed for VLA handoff. A camera/Codex/depth error, low
confidence, rejected planner request, timeout, or operator interruption prevents
VLA startup. Before the ObjectNav publisher exits, it aborts unfinished motion to
zero velocity and keeps C++ in PLANNER mode; it never intentionally transitions
the robot to OFF. The tmux panes remain available for diagnosis.
