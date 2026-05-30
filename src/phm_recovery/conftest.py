"""Make phm_recovery importable when running pytest from the package dir or repo root.

Adds the package directory to ``sys.path`` so ``import phm_recovery`` resolves
without an editable install. Mirrors the pattern in src/phm_core/conftest.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))
