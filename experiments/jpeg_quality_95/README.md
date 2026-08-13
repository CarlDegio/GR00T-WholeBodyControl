# Sonic Camera JPEG Quality 80 vs 95

## Result

Raising the four software-encoded RGB streams from JPEG quality 80 to 95 increased
application payload bandwidth by **45.96%** while composed publish throughput stayed
at **30.00 FPS**. Mean end-to-end latency increased by **1.33 ms (3.01%)**, median
latency increased by **2.18 ms (5.29%)**, and mean PC decode time increased by
**1.07 ms (11.37%)**.

| Metric | Quality 80 | Quality 95 | Change |
|---|---:|---:|---:|
| Composed messages | 1,801 | 1,801 | 0 |
| Composed FPS | 30.002 | 30.004 | +0.005% |
| Image throughput | 180.013 images/s | 180.021 images/s | +0.005% |
| Mean composed payload | 536.93 KiB | 783.65 KiB | +45.95% |
| Wire payload rate | 15.731 MiB/s | 22.961 MiB/s | +45.96% |
| Wire payload bitrate | 131.965 Mbit/s | 192.613 Mbit/s | +45.96% |
| End-to-end latency mean | 44.188 ms | 45.517 ms | +3.01% |
| End-to-end latency p50 | 41.265 ms | 43.449 ms | +5.29% |
| End-to-end latency p95 | 63.047 ms | 61.122 ms | -3.05% |
| PC decode mean | 9.413 ms | 10.483 ms | +11.37% |
| PC decode p50 | 8.926 ms | 9.993 ms | +11.95% |
| PC decode p95 | 12.710 ms | 13.440 ms | +5.74% |

The p95 latency decrease should be treated as run-to-run variation, not as evidence
that quality 95 improves tail latency. Each quality was measured once for 60 seconds.

## Per-stream unique frame rate

The composed publisher may reuse a camera's latest frame. Unique FPS counts distinct
camera timestamps and therefore exposes that reuse.

| Stream | Quality 80 | Quality 95 | Change |
|---|---:|---:|---:|
| `ego_view` | 28.236 | 29.204 | +3.43% |
| `ego_view_depth` | 28.236 | 29.204 | +3.43% |
| `chest_view` | 29.919 | 29.870 | -0.16% |
| `chest_view_depth` | 29.919 | 29.870 | -0.16% |
| `left_wrist` | 30.002 | 29.987 | -0.05% |
| `right_wrist` | 29.802 | 29.537 | -0.89% |

Quality 95 met the 29 FPS unique-frame target on all six streams in this run. The
quality-80 Orbbec ego stream did not. This difference is acquisition variability and
does not establish that JPEG quality changes sensor capture rate; the composed
publisher itself remained at 30 FPS in both runs.

## Method

- Sonic endpoint: `tcp://192.168.123.164:5555`.
- Camera set: four `640x480 uint8` RGB streams and two `640x480 uint16` lossless
  depth streams.
- JPEG quality applies only to NumPy RGB images encoded by OpenCV. Depth remains PNG,
  and OAK-only on-device `mjpeg_quality` is unchanged.
- Each run used 5 seconds of warm-up followed by 60 seconds of measured traffic.
- The PC subscribed without ZMQ conflation, measured raw msgpack payload length,
  unpacked every message, and decoded all six images.
- Wire rate is application payload goodput and excludes Ethernet/IP/TCP framing.
- End-to-end latency is PC receive wall time minus Sonic camera timestamp. Both hosts
  reported NTP synchronization; chrony RMS offsets were below 0.2 ms, while absolute
  root dispersion was about 11 ms. Relative A/B latency is more reliable than its
  absolute value.
- Both server logs reported zero ZMQ message drops and no camera warnings, errors, or
  reconnects during the captured periods.

## Artifacts

- `quality_80.json`: complete quality-80 receiver statistics.
- `quality_95.json`: complete quality-95 receiver statistics.
- `sonic_quality_80.log`: Sonic quality-80 startup and publisher log.
- `sonic_quality_95.log`: Sonic quality-95 startup and publisher log.

The active Sonic process was left running with `--jpeg-quality 95`. The replaced
Sonic files are backed up under
`/home/unitree/GR00T-WholeBodyControl/Log/jpeg_quality_95_backup_20260813_131148/`.
