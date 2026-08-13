# JPEG 95 binary camera and VLA validation

## Result

The camera transport thresholds passed: 30.001 messages/s, every observed image
payload was msgpack binary, and binary was 24.980% smaller than the same-message
Base64 counterfactual. The corrected physical-frame acceptance also passed. A fresh
sequential SensorGateway/OpenPI run observed 30.016 Gateway publications/s on every
stream, physical encoded rates of 29.666--30.016 FPS, and a six-stream physical
decoded aggregate of 179.012 images/s.

The benchmark covers camera transport, Gateway snapshot RPC, JPEG wrapping/request
packing, and the existing OpenPI JPEG decoder. It deliberately excludes policy/model
inference time.

## Camera comparison

The JPEG 80 and JPEG 95 Base64 results are the earlier 60-second artifacts in
`experiments/jpeg_quality_95`. The JPEG 95 binary result is the new live run. These
were separate runs, so the same-packet counterfactual in the last row is the rigorous
binary/Base64 wire comparison.

| Metric | JPEG 80 Base64 | JPEG 95 Base64 | JPEG 95 binary |
|---|---:|---:|---:|
| Message FPS | 30.002 | 30.004 | 30.001 |
| Six-image throughput (images/s) | 180.013 | 180.021 | 180.005 |
| Wire payload (Mbit/s) | 131.965 | 192.613 | 144.702 |
| Same-packet Base64 counterfactual (Mbit/s) | not recorded | not recorded | 192.885 |
| Binary saving vs same-packet Base64 | not applicable | not applicable | 24.980% |
| End-to-end latency P50 / P95 (ms) | 41.265 / 63.047 | 43.449 / 61.122 | 40.653 / 55.233 |
| Six-image PC decode P50 / P95 (ms) | 8.926 / 12.710 | 9.993 / 13.440 | 9.282 / 12.356 |

Absolute end-to-end latency is sensitive to separate-run camera timing and host clock
conditions; this run does not claim an absolute latency improvement over the older
runs. Its direct transport evidence is the 144.702 Mbit/s actual payload versus the
192.885 Mbit/s same-message Base64 counterfactual.

### Per-camera source timestamp rate

This table counts distinct physical-camera timestamps rather than composed messages.
The composed publisher may reuse the latest device frame while still publishing at
30 Hz.

| Stream | JPEG 80 Base64 | JPEG 95 Base64 | JPEG 95 binary | Binary reused timestamps |
|---|---:|---:|---:|---:|
| `ego_view` | 28.236 | 29.204 | 29.818 | 11 |
| `ego_view_depth` | 28.236 | 29.204 | 29.818 | 11 |
| `chest_view` | 29.919 | 29.870 | 29.834 | 10 |
| `chest_view_depth` | 29.919 | 29.870 | 29.834 | 10 |
| `left_wrist` | 30.002 | 29.987 | 29.801 | 12 |
| `right_wrist` | 29.802 | 29.537 | 29.984 | 1 |

The acceptance FPS threshold applies to distinct positive producer source timestamps,
not composed-message or Gateway-created sequence rates. The controlled NI=0 camera
and fresh pipeline runs below pass those physical thresholds.

### Controlled scheduler A/B

The initial Task 8 launcher used `nohup ./start_camera_server.zsh ... &` from Zsh.
Zsh's default `BG_NICE` option automatically assigned that background job nice +5;
the camera child inherited it. The start script itself contains no `nice` command.
A controlled A/B restarted only the owned camera with `unsetopt BG_NICE`, keeping the
same deployed files, JPEG quality 95, device IDs, depth settings, port, and command
line. The launcher/camera moved from PID 5187/5190 at NI=5 to PID 218685/218687 at
NI=0.

| Physical source | NI=5 pipeline FPS | NI=0 camera A/B FPS |
|---|---:|---:|
| Ego | 27.164 | 29.818 |
| Chest | 29.263 | 29.834 |
| Left wrist | 29.597 | 29.801 |
| Right wrist | 28.797 | 29.984 |

The directly comparable raw-camera artifact improved from 54.737/80.124 ms P50/P95
at NI=5 to 40.653/55.233 ms at NI=0 while composed-message rate remained about
30 FPS. This isolates the earlier physical-frame reuse and latency regression to the
background launch scheduling condition. It is not evidence that JPEG 95 or binary
transport caused the loss.

## SensorGateway and OpenPI protocol

No comparable JPEG 80 or JPEG 95 Base64 Gateway/OpenPI pipeline artifact exists, so
those baseline cells are explicitly unavailable rather than estimated.

| Metric | JPEG 80 Base64 | JPEG 95 Base64 | JPEG 95 binary |
|---|---:|---:|---:|
| Gateway publication FPS, every stream | not measured | not measured | 30.016 |
| Physical encoded FPS, four RGB range | not measured | not measured | 29.666–30.016 |
| Physical decoded FPS, six-stream range | not measured | not measured | 29.666–30.016 |
| Physical decoded aggregate (images/s) | not measured | not measured | 179.012 (pass) |
| VLA request throughput (requests/s) | not measured | not measured | 134.663 |
| VLA request payload (Mbit/s) | not measured | not measured | 370.645 |
| Gateway RPC P50 / P95 (ms) | not measured | not measured | 0.513 / 2.234 |
| JPEG prepare P50 / P95 (ms) | not measured | not measured | 0.045 / 0.067 |
| Request pack P50 / P95 (ms) | not measured | not measured | 0.072 / 0.111 |
| OpenPI four-JPEG decode P50 / P95 (ms) | not measured | not measured | 3.717 / 4.806 |
| New JPEG wrapper P50 / P95 (ms) | not measured | not measured | 0.00047 / 0.00287 |
| Old RGB-to-JPEG codec P50 / P95 (ms) | not measured | not measured | 0.71260 / 0.94543 |

The codec comparison contains 7,204 unique stream/frame samples. The benchmark made
8,080 sequential requests; requests between new camera publications legitimately
reused the latest Gateway sequence and were excluded from codec unique-frame counts.

| Stream | Gateway publication FPS | Physical unique FPS | Source timestamp reuses |
|---|---:|---:|---:|
| `ego_view` | 30.016 | 29.666 | 21 |
| `ego_view_depth` | 30.016 | 29.666 | 21 |
| `chest_view` | 30.016 | 29.916 | 6 |
| `chest_view_depth` | 30.016 | 29.916 | 6 |
| `left_wrist` | 30.016 | 29.833 | 11 |
| `right_wrist` | 30.016 | 30.016 | 0 |

## Drop and stale evidence

- Sonic's `message dropped: 0` counter means the nonblocking `socket.send` call had
  zero `zmq.Again` failures. It does not observe messages discarded later by PUB/HWM,
  TCP, SUB conflation, or a receiver.
- The fresh benchmark observed zero Gateway sequence gaps or regressions. Those are
  gaps in Gateway-created ring publications as sampled by this benchmark, not proof
  of zero camera transport drops.
- Producer source timestamps had zero missing values and zero regressions, but the
  reuse counts in the table show composed publications frequently reused physical
  frames.
- Snapshot rejection accounting was encoded=0, decoded=0, total=0. Rejections would
  be counted and skipped rather than silently terminating the benchmark.
- PUB/HWM transport drops are unobservable in this protocol because the producer
  message has no independent frame identifier. No zero-drop claim is made.

## Why the protocol path is shorter

The intended latency reduction comes from four scoped changes: raw msgpack binary
removes Base64 encode/decode and expansion; encoded-first Gateway publication makes
the original camera JPEG available before decoded arrays; VLA no longer calls
`cv2.imencode` and instead wraps the existing JPEG; and camera `SNDHWM=1` keeps the
publisher latest-first under receiver pressure.

The production VLA scheduling and timing functions were not changed: no new thread,
sleep, retry, wait, freshness, pairing, or scheduling branch was added. The telemetry
label changed from `jpeg_encode` to `jpeg_prepare`, and the observation builder now
records the wrapping duration under that label.

VLA encoded ingress accepts only `jpeg_bytes`. The ordinary camera schema decoder
retains historical Base64 decoding for non-VLA consumers, but VLA does not Base64
decode, infer shapes, fall back to RGB decode/re-encode, or perform color correction.

## Live topology and deployment

- PC binary receiver: this worktree's read-only SensorGateway, bound to
  `ipc:///tmp/sonic_sensor_gateway.ipc`, was started before the producer change.
- Sonic camera: `unitree@192.168.123.164`, port 5555, confirmed command line includes
  `--jpeg-quality 95`. The final evidence uses launcher PID 218685 and camera PID
  218687 at NI=0; no production source/config file was changed for the scheduler A/B.
- Only `gear_sonic/camera/constants.py`, `composed_camera.py`, and `sensor_server.py`
  were deployed. Exact overwritten files were backed up at
  `/home/unitree/GR00T-WholeBodyControl/.task8-backups/20260813T154131+0800` before
  same-directory atomic replacement.
- The modified Sonic Realsense/Orbbec drivers, start script, and unrelated dirty
  worktree files were not overwritten.
- `/home/user/Project/openpi_sonic` was not modified; its tracked diff against
  `origin/main` remained empty. Its pre-existing untracked `replay_data/` directory
  was left untouched.

## Artifacts and reproduction

- `camera_binary_60s.json`: raw camera receiver statistics.
- `vla_pipeline_60s.json`: raw Gateway/OpenPI protocol statistics.
- `check_acceptance.py`: the executable acceptance assertions from the plan.

Run the checker from the repository root:

```bash
/home/user/Project/GR00T-WholeBodyControl/.venv_teleop/bin/python \
  experiments/jpeg95_binary_vla/check_acceptance.py
```

For the committed fresh NI=0 run this command exits 0: all specified camera,
physical-rate, aggregate-throughput, JPEG preparation, OpenPI decode, and codec
comparison thresholds pass.
