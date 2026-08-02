"""Run one Codex + RGB-D ObjectNav cycle, optionally through SONIC."""

from __future__ import annotations

from dataclasses import dataclass
import json
import sys

import tyro

from gear_sonic.utils.inference.object_nav import (
    ObjectNavConfig,
    ObjectNavRunner,
    SonicPlannerRequestError,
    send_object_nav_commands,
)


@dataclass
class ObjectNavCLIConfig:
    mission: str
    """Navigation mission supplied to Codex."""

    global_target: str
    """Final object/place whose verified arrival produces STOP."""

    camera_host: str = "localhost"
    camera_port: int = 5555
    camera_timeout_ms: int = 3000
    output_root: str = "outputs/object_nav"

    codex_timeout_seconds: float = 180.0
    min_confidence: float = 0.6
    rotation_speed: float = 0.4
    forward_speed: float = 0.3
    safe_distance: float = 0.0
    max_direct_travel: float = 8.0

    execute_sonic: bool = False
    """Send NAVIGATE commands to the acknowledged planner endpoint."""

    sonic_host: str = "127.0.0.1"
    sonic_json_port: int = 5559
    sonic_timeout_ms: int = 70000


def to_runtime_config(config: ObjectNavCLIConfig) -> ObjectNavConfig:
    return ObjectNavConfig(
        mission=config.mission,
        global_target=config.global_target,
        camera_host=config.camera_host,
        camera_port=config.camera_port,
        camera_timeout_ms=config.camera_timeout_ms,
        codex_timeout_seconds=config.codex_timeout_seconds,
        min_confidence=config.min_confidence,
        rotation_speed=config.rotation_speed,
        forward_speed=config.forward_speed,
        safe_distance=config.safe_distance,
        max_direct_travel=config.max_direct_travel,
        output_root=config.output_root,
    )


def main(config: ObjectNavCLIConfig) -> int:
    runner = ObjectNavRunner(to_runtime_config(config))
    exit_code = 0
    try:
        result = runner.run_once()
        if result.outcome == "NAVIGATE" and config.execute_sonic:
            try:
                reply = send_object_nav_commands(
                    result.commands,
                    host=config.sonic_host,
                    port=config.sonic_json_port,
                    timeout_ms=config.sonic_timeout_ms,
                )
            except (SonicPlannerRequestError, ValueError) as exc:
                print(f"[ObjectNav] Sonic execution failed: {exc}", file=sys.stderr)
                exit_code = 1
            else:
                print(
                    "[ObjectNav] Sonic execution completed at heading "
                    f"{reply.get('heading_rad', 'unknown')}",
                    file=sys.stderr,
                )
        elif result.outcome in {"FAILED", "REJECTED"}:
            exit_code = 1
        sys.stdout.write(json.dumps(result.commands, ensure_ascii=False) + "\n")
        sys.stdout.flush()
        return exit_code
    finally:
        runner.close()


if __name__ == "__main__":
    raise SystemExit(main(tyro.cli(ObjectNavCLIConfig)))
