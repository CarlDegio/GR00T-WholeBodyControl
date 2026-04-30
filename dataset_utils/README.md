# SONIC Dataset Utils

These scripts parse the LeRobot-format datasets exported by
`gear_sonic/scripts/run_data_exporter.py`.

Use the data collection environment:

```bash
source .venv_data_collection/bin/activate
```

Inspect a dataset and preview one episode:

```bash
python dataset_utils/inspect_sonic_dataset.py \
    outputs/2026-04-17-20-47-36 \
    --episode-index 0
```

Export one episode to JSONL:

```bash
python dataset_utils/export_sonic_episode.py \
    outputs/2026-04-17-20-47-36 \
    0
```

The exporter now also writes a full CSV table for the whole parquet episode.
Array-valued fields are serialized into one cell as strings such as
`[1, 2, 3]`.

Export one episode and decode a few video frames:

```bash
python dataset_utils/export_sonic_episode.py \
    outputs/2026-04-17-20-47-36 \
    0 \
    --extract-video-frames
```
