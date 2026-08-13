# JPEG 95 binary camera and VLA validation

## Result

The camera transport thresholds passed: 30.007 messages/s, every observed image
payload was msgpack binary, and binary was 24.980% smaller than the same-message
Base64 counterfactual. The corrected physical-frame acceptance did **not** pass. A
fresh sequential SensorGateway/OpenPI run observed 30.013 Gateway publications/s on
every stream, but physical encoded rates ranged from 27.164 to 29.597 FPS and the
six-stream physical decoded aggregate was 171.247 images/s, below the required 174.

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
| Message FPS | 30.002 | 30.004 | 30.007 |
| Six-image throughput (images/s) | 180.013 | 180.021 | 180.040 |
| Wire payload (Mbit/s) | 131.965 | 192.613 | 145.403 |
| Same-packet Base64 counterfactual (Mbit/s) | not recorded | not recorded | 193.819 |
| Binary saving vs same-packet Base64 | not applicable | not applicable | 24.980% |
| End-to-end latency P50 / P95 (ms) | 41.265 / 63.047 | 43.449 / 61.122 | 54.737 / 80.124 |
| Six-image PC decode P50 / P95 (ms) | 8.926 / 12.710 | 9.993 / 13.440 | 9.082 / 12.073 |

Absolute end-to-end latency is sensitive to separate-run camera timing and host clock
conditions; this run does not claim an absolute latency improvement over the older
runs. Its direct transport evidence is the 145.403 Mbit/s actual payload versus the
193.819 Mbit/s same-message Base64 counterfactual.

### Per-camera source timestamp rate

This table counts distinct physical-camera timestamps rather than composed messages.
The composed publisher may reuse the latest device frame while still publishing at
30 Hz.

| Stream | JPEG 80 Base64 | JPEG 95 Base64 | JPEG 95 binary | Binary reused timestamps |
|---|---:|---:|---:|---:|
| `ego_view` | 28.236 | 29.204 | 27.224 | 167 |
| `ego_view_depth` | 28.236 | 29.204 | 27.224 | 167 |
| `chest_view` | 29.919 | 29.870 | 28.141 | 112 |
| `chest_view_depth` | 29.919 | 29.870 | 28.141 | 112 |
| `left_wrist` | 30.002 | 29.987 | 28.790 | 73 |
| `right_wrist` | 29.802 | 29.537 | 27.974 | 122 |

The acceptance FPS threshold applies to distinct positive producer source timestamps,
not composed-message or Gateway-created sequence rates. The corrected fresh pipeline
run below therefore fails the physical threshold honestly.

## SensorGateway and OpenPI protocol

No comparable JPEG 80 or JPEG 95 Base64 Gateway/OpenPI pipeline artifact exists, so
those baseline cells are explicitly unavailable rather than estimated.

| Metric | JPEG 80 Base64 | JPEG 95 Base64 | JPEG 95 binary |
|---|---:|---:|---:|
| Gateway publication FPS, every stream | not measured | not measured | 30.013 |
| Physical encoded FPS, four RGB range | not measured | not measured | 27.164–29.597 |
| Physical decoded FPS, six-stream range | not measured | not measured | 27.164–29.597 |
| Physical decoded aggregate (images/s) | not measured | not measured | 171.247 (fail) |
| VLA request throughput (requests/s) | not measured | not measured | 118.820 |
| VLA request payload (Mbit/s) | not measured | not measured | 329.786 |
| Gateway RPC P50 / P95 (ms) | not measured | not measured | 0.659 / 2.739 |
| JPEG prepare P50 / P95 (ms) | not measured | not measured | 0.052 / 0.073 |
| Request pack P50 / P95 (ms) | not measured | not measured | 0.082 / 0.120 |
| OpenPI four-JPEG decode P50 / P95 (ms) | not measured | not measured | 3.899 / 5.354 |
| New JPEG wrapper P50 / P95 (ms) | not measured | not measured | 0.00049 / 0.00321 |
| Old RGB-to-JPEG codec P50 / P95 (ms) | not measured | not measured | 0.72052 / 1.00331 |

The codec comparison contains 7,204 unique stream/frame samples. The benchmark made
7,130 sequential requests; requests between new camera publications legitimately
reused the latest Gateway sequence and were excluded from codec unique-frame counts.

| Stream | Gateway publication FPS | Physical unique FPS | Source timestamp reuses |
|---|---:|---:|---:|
| `ego_view` | 30.013 | 27.164 | 171 |
| `ego_view_depth` | 30.013 | 27.164 | 171 |
| `chest_view` | 30.013 | 29.263 | 45 |
| `chest_view_depth` | 30.013 | 29.263 | 45 |
| `left_wrist` | 30.013 | 29.597 | 25 |
| `right_wrist` | 30.013 | 28.797 | 73 |

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
  `--jpeg-quality 95`.
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

For the committed fresh run this command exits 1 at the physical encoded FPS
assertion. The JSON is preserved as failure evidence; full performance acceptance is
not claimed.
