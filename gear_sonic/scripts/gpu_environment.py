"""Select locally cached driver libraries matching the running NVIDIA kernel."""

import os
from pathlib import Path
import re
from typing import Mapping


def gpu_environment(
    repo_root: Path,
    *,
    environ: Mapping[str, str] | None = None,
    driver_info: Path = Path("/proc/driver/nvidia/version"),
) -> tuple[str, ...]:
    """Return explicit env arguments so old tmux servers cannot drop library paths."""
    environ = os.environ if environ is None else environ
    paths = [p for p in environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    if driver_info.is_file():
        version = re.search(r"Kernel Module\s+(\d+\.\d+\.\d+)", driver_info.read_text())
        if version:
            cached = repo_root.parent / ".tools" / f"nvidia-{version[1]}" / "usr/lib/x86_64-linux-gnu"
            if all((cached / lib).is_file() for lib in ("libcuda.so.1", "libnvidia-ml.so.1")):
                paths.insert(0, str(cached))
    paths = list(dict.fromkeys(paths))
    return (f"LD_LIBRARY_PATH={':'.join(paths)}",) if paths else ()
