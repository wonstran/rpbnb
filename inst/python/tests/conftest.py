"""Put inst/python on sys.path so `pytest inst/python/tests/` works from the
repository root without a PYTHONPATH incantation."""

import sys
from pathlib import Path

_PKG_ROOT = str(Path(__file__).resolve().parent.parent)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
