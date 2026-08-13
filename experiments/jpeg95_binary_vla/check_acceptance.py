import json
from pathlib import Path

root = Path("experiments/jpeg95_binary_vla")
camera = json.loads((root / "camera_binary_60s.json").read_text())
vla = json.loads((root / "vla_pipeline_60s.json").read_text())
assert camera["message_fps"] >= 29.0
assert camera["binary_wire_savings_percent"] >= 20.0
assert min(x["unique_fps"] for x in vla["encoded_streams"].values()) >= 29.0
assert vla["decoded_images_per_second"] >= 174.0
assert vla["latency_ms"]["jpeg_prepare"]["p95"] < 1.0
assert vla["latency_ms"]["openpi_decode"]["p95"] <= 6.0
assert vla["codec_comparison_ms"]["new_p50"] < vla["codec_comparison_ms"]["old_p50"]
assert vla["codec_comparison_ms"]["new_p95"] < vla["codec_comparison_ms"]["old_p95"]
