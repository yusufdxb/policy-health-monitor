"""Pytest config for the benchmark.

Puts the benchmark dir (for ``import lib...``) and the phm_core package root
(for ``from phm_core.calibration import ...``) on sys.path, mirroring the
PYTHONPATH the harness is run with. Keeps the benchmark self-contained so
``pytest benchmark`` works without a sourced ROS overlay.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PHM_CORE = REPO / "src" / "phm_core"

for p in (str(HERE), str(PHM_CORE)):
    if p not in sys.path:
        sys.path.insert(0, p)
