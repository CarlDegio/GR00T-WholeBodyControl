"""Resolve a prepared analysis directory, independently of the code location."""
from pathlib import Path
import os

_configured = os.environ.get('TRAINING_START_MATCHING_ROOT')
ROOT = Path(_configured).expanduser().resolve() if _configured else Path(__file__).resolve().parents[1]
if not (ROOT / 'preflight.json').is_file():
    raise RuntimeError('Use run.py --analysis-root PATH, or set TRAINING_START_MATCHING_ROOT to a prepared directory containing preflight.json.')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
