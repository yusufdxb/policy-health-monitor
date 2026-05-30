"""Pytest configuration for phm_arbiter tests.

Adds this package directory (which contains the ``phm_arbiter/`` package) and
the sibling ``phm_core`` package root to ``sys.path`` so ``import phm_arbiter``
and ``import phm_core`` resolve without an editable install. Also prunes ROS
workspace and ``/opt/ros`` entries so ament pytest plugins are not auto-loaded.
Mirrors the pattern in src/phm_core/conftest.py and src/phm_ood/conftest.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Prune ROS overlay entries (same pattern as repo-root conftest.py).
_ROS_MARKERS = ("/opt/ros/", "/ros2_ws/")
sys.path[:] = [p for p in sys.path if not any(m in p for m in _ROS_MARKERS)]

# Add phm_arbiter package root so `import phm_arbiter` works.
_PKG_ROOT = Path(__file__).resolve().parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

# Add phm_core package root (sibling) so `import phm_core` works.
_CORE_ROOT = _PKG_ROOT.parent / "phm_core"
if str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))
