"""Resolve and validate a mission overlay into a launchable YAML; never launch processes."""

import argparse
from dataclasses import asdict
from pathlib import Path

import yaml

from gear_sonic.runtime.profile import load_runtime_profile
from gear_sonic.utils.inference.lavira.mission import validate_mission_profile


def prepare_mission(output, *, profile=None, overlays=()):
    runtime = load_runtime_profile(profile, overlays=overlays)
    validate_mission_profile(runtime)
    payload = {
        "schema": "sonic.runtime_profile", "version": 1, "profile": runtime.name,
        "endpoints": {name: asdict(endpoint) for name, endpoint in runtime.endpoints.items()},
        "ros_topics": dict(runtime.ros_topics),
        "components": {name: dict(component) for name, component in runtime.components.items()},
    }
    output = Path(output).resolve()
    if output in runtime.source_files:
        raise ValueError("output must differ from the source profile and overlays")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--overlay", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(prepare_mission(args.output, profile=args.profile, overlays=args.overlay))


if __name__ == "__main__":
    main()
