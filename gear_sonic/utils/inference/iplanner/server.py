"""Run the original LaViRA server under a repository-owned process entry point."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--root", required=True)
    args, server_args = parser.parse_known_args()
    root = Path(args.root).expanduser().resolve()
    server = root / "iplanner_server.py"
    if not server.is_file():
        parser.error(f"LaViRA iPlanner server not found: {server}")
    sys.path.insert(0, str(root))
    os.chdir(root)
    sys.argv = [str(server), *server_args]
    runpy.run_path(str(server), run_name="__main__")


if __name__ == "__main__":
    main()
