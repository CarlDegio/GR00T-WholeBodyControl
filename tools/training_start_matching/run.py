"""Reproduce geometry, metrics, and figures from reviewed annotations."""
from pathlib import Path
import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-root', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true', help='Re-extract first frames from the frozen source manifest before analysis.')
    args = parser.parse_args()
    root = args.analysis_root.expanduser().resolve()
    if not (root / 'preflight.json').is_file():
        parser.error('The analysis root must contain a frozen preflight.json and the downloaded source data.')
    env = dict(os.environ)
    env['TRAINING_START_MATCHING_ROOT'] = str(root)
    env['MPLCONFIGDIR'] = str(root / '.cache/matplotlib')
    scripts = (['prepare.py'] if args.prepare else []) + ['validate_math.py', 'estimate.py', 'diagnose_geometry.py', 'metrics.py', 'present.py', 'report.py', 'validate_artifacts.py']
    for script in scripts:
        print(f'Running {script}', flush=True)
        subprocess.run([sys.executable, str(Path(__file__).resolve().parent / script)], env=env, check=True)


if __name__ == '__main__':
    main()
