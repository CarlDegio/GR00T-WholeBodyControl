# JPEG 95 binary camera and VLA validation

## Result

The 60-second live run passed all eight machine-checked performance thresholds. The
camera published at 30.007 messages/s, every observed image payload was msgpack
binary, and the same-message Base64 counterfactual was 24.980% larger. The
sequential SensorGateway/OpenPI protocol benchmark observed 30.014 encoded frames/s
per RGB stream and 180.082 decoded images/s across all six streams.

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

The acceptance FPS threshold applies to Gateway encoded-frame sequences, measured
below. The lower physical-camera timestamp rates above are reported separately and
must not be hidden by the composed-message rate.

## SensorGateway and OpenPI protocol

No comparable JPEG 80 or JPEG 95 Base64 Gateway/OpenPI pipeline artifact exists, so
those baseline cells are explicitly unavailable rather than estimated.

| Metric | JPEG 80 Base64 | JPEG 95 Base64 | JPEG 95 binary |
|---|---:|---:|---:|
| Encoded FPS, each of four RGB streams | not measured | not measured | 30.014 |
| Decoded FPS, each of six streams | not measured | not measured | 30.014 |
| Decoded aggregate (images/s) | not measured | not measured | 180.082 |
| VLA request throughput (requests/s) | not measured | not measured | 117.388 |
| VLA request payload (Mbit/s) | not measured | not measured | 323.759 |
| Gateway RPC P50 / P95 (ms) | not measured | not measured | 0.671 / 2.708 |
| JPEG prepare P50 / P95 (ms) | not measured | not measured | 0.052 / 0.070 |
| Request pack P50 / P95 (ms) | not measured | not measured | 0.085 / 0.120 |
| OpenPI four-JPEG decode P50 / P95 (ms) | not measured | not measured | 3.973 / 5.292 |
| New JPEG wrapper P50 / P95 (ms) | not measured | not measured | 0.00042 / 0.00303 |
| Old RGB-to-JPEG codec P50 / P95 (ms) | not measured | not measured | 0.72479 / 0.99827 |

The codec comparison contains 7,204 unique stream/frame samples. The benchmark made
7,044 sequential requests; requests between new camera publications legitimately
reused the latest Gateway sequence and were excluded from codec unique-frame counts.

## Drop and stale evidence

- The Sonic producer log reported `message dropped: 0` through the live run.
- Post-run Gateway health reported zero dropped messages, zero failures, and zero
  out-of-order messages for the source, all encoded streams, and all decoded streams.
- The pipeline benchmark aborts on stale/unavailable snapshots; it completed all
  7,044 requests without such an error. No separate stale counter is emitted.
- Source timestamp reuse is listed in the per-camera table and is distinct from a
  transport drop or stale Gateway snapshot.

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

## Live topology and deployment

- PC compatibility receiver: this worktree's read-only SensorGateway, bound to
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
