"""Shared entrypoint used by the fifteen executable experiment wrappers."""

import sys

from .config import launch

if __name__ == "__main__":
    raise SystemExit(launch(sys.argv[1], sys.argv[2:]))
