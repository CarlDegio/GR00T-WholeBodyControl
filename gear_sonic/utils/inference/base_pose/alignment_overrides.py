"""The alignment settings a mission skill may override for one ALIGN call."""

import math


DISTANCE_FIELDS = ("raw_head_target_distance_m", "raw_chest_target_distance_m")
CONFIDENCE_FIELDS = ("raw_head_target_confidence", "raw_chest_target_confidence")
OVERRIDE_FIELDS = DISTANCE_FIELDS + CONFIDENCE_FIELDS + (
    "raw_stable_frames", "raw_lateral_tolerance_m",
)


def validate_alignment_overrides(value):
    if not isinstance(value, dict) or set(value) - set(OVERRIDE_FIELDS):
        raise ValueError("base_pose supports only head/chest target distances, confidences, raw_stable_frames and raw_lateral_tolerance_m")
    if any(type(v) not in {int, float} or not math.isfinite(v) or v <= 0 for v in value.values()):
        raise ValueError("base_pose overrides must be finite positive numbers")
    if any(value[key] > 1 for key in CONFIDENCE_FIELDS if key in value):
        raise ValueError("base_pose target confidences must be in (0, 1]")
    if "raw_stable_frames" in value and type(value["raw_stable_frames"]) is not int:
        raise ValueError("base_pose raw_stable_frames must be a positive integer")
    return {key: setting if key == "raw_stable_frames" else float(setting) for key, setting in value.items()}
