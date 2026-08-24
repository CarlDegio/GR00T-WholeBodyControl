"""Terminate repository-owned inference processes left behind by dead panes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
import subprocess
import time
from typing import Any


FASTLIO_PROCESS_PATTERNS = (
    r"(^|/)ros2 launch fast_lio mapping\.launch\.py( |$)",
    r"(^|/)fastlio_mapping( |$)",
)

NAVDP_PROCESS_PATTERNS = (
    r"(^|/)(python|python3) -m gear_sonic\.utils\.inference\.navdp\.service( |$)",
    r"(^|/)(python|python3) -m eval\.src\.policy_server( |$)",
    r"(^|/)(python|python3) ([^ ]*/)?gear_sonic/scripts/navdp_planner\.py( |$)",
)

POLICY_PROCESS_PATTERNS = (
    r"(^|/)(python|python3) -m gear_sonic\.utils\.inference\.vla\.service( |$)",
)

GATEWAY_PROCESS_PATTERNS = (
    r"(^|/)(python|python3) -m gear_sonic\.runtime\.gateway\.services\.sensor( |$)",
)

INFERENCE_PROCESS_PATTERNS = (
    *FASTLIO_PROCESS_PATTERNS,
    *NAVDP_PROCESS_PATTERNS,
    *POLICY_PROCESS_PATTERNS,
    *GATEWAY_PROCESS_PATTERNS,
)


def terminate_matching_processes(
    patterns: Sequence[str],
    *,
    runner: Callable[..., Any] = subprocess.run,
    sleeper: Callable[[float], None] = time.sleep,
    grace_period_s: float = 0.5,
) -> None:
    """Send TERM and then KILL to exact inference entry-point matches."""

    for pattern in patterns:
        runner(["pkill", "-TERM", "-f", pattern], capture_output=True)
    sleeper(grace_period_s)
    for pattern in patterns:
        runner(["pkill", "-KILL", "-f", pattern], capture_output=True)


def cleanup_inference_processes() -> None:
    """Clean all known inference processes; safe to invoke repeatedly."""

    terminate_matching_processes(INFERENCE_PROCESS_PATTERNS)
    cleanup_shared_memory_files()


def cleanup_shared_memory_files(
    shared_memory_dir: Path = Path("/dev/shm"),
) -> tuple[Path, ...]:
    """Unlink SONIC segments after their owning gateway has stopped."""

    removed: list[Path] = []
    for path in shared_memory_dir.glob("sonic-*"):
        if not path.is_file():
            continue
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        removed.append(path)
    return tuple(removed)


if __name__ == "__main__":
    cleanup_inference_processes()
