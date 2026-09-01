# PICO disconnect baseline: 2026-08-30 16:46:16

This directory preserves the complete evidence available for the abnormal
disconnect that occurred after DataCollection had started and the PICO link
was carrying traffic in both directions.

## Incident identity

- DataCollection launch: 2026-08-30 16:44:41 Asia/Shanghai
- RNDIS recovery used by the failing run: 16:45:32
- PICO `Send`/RoboticsService connection active: approximately 16:45:37
- Remote Vision request sent to `192.168.30.170:13579`: 16:45:57
- Abnormal USB leaf disconnect: 16:46:16.626
- PICO battery shortly before failure: 19%
- PICO VBUS shortly before failure: approximately 4.390 V
- PICO remained powered and later enumerated as `2d40:00b5`, without RNDIS

## Preserved evidence

`evidence_20260830_164616.tar.zst` contains:

- complete captured PICO logcat and host USB/network monitors;
- complete RoboticsService logs from before and after the service restart;
- current-boot kernel journal and the focused 16:30:00--16:47:30 slice;
- all seven available `sonic_data_collection` tmux pane histories;
- USB topology, interfaces, routes, sockets, process and host state;
- the exact runtime configuration and relevant source files;
- the worktree patch and Git revision used by the run;
- `outputs/2026-08-30-16-44-57` exactly as created. It contains metadata but
  no recorded frames because `cpp/state_msgpack`/proprioception was missing;
- an internal per-file `SHA256SUMS` and file manifest.

Archive SHA-256:

```text
2407c3bc6d2768e79fff7c02e550dea9653267fa691063af82bd94fb338e0ffb
```

## Cable A/B protocol

Use the current cable first, then the original PICO cable. Change only the
cable between trials.

Keep these conditions identical:

1. Charge PICO to the same starting band for both trials (recommended
   60--70%); recharge before trial B.
2. Use the same PC USB port and the same hub path.
3. Start the same DataCollection profile.
4. Restore USBOnly/RNDIS, then enable XRoboToolkit `Send`.
5. Enable the same image and Remote Vision options in the same order.
6. Run each trial for at least 10 minutes, or until the first USB leaf
   disconnect.
7. Do not restart RoboticsService between the two trials unless it becomes
   necessary for both; record any restart as a changed condition.

Interpretation:

- Current cable fails and original cable remains stable under matching
  conditions: cable/connector path is strongly implicated.
- Both fail in a similar time window: investigate PICO power delivery and the
  RNDIS/DWC3 gadget stack rather than the cable alone.
- Only low-battery trials fail: power margin is the dominant variable and a
  cable conclusion is not justified.
