"""Make the phm_core package importable when running pytest from the repo root.

Adds this package directory (which contains the ``phm_core/`` package) to
``sys.path`` so ``import phm_core`` resolves without an editable install. This
keeps ``pytest src/phm_core -q`` working straight from a fresh checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))
