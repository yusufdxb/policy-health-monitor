"""pytest configuration for phm_sim.

Adds phm_core and phm_sim to sys.path so tests can import both without
installing them, and prunes /opt/ros paths to prevent ROS 2 pytest plugin
auto-loading. Mirrors the approach in src/phm_core/conftest.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add this package directory (which contains the ``phm_sim/`` package) so
# ``import phm_sim`` resolves without a colcon build or pip install.
_PKG_ROOT = str(Path(__file__).resolve().parent)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

# Add the sibling ``phm_core`` package root so ``import phm_core`` resolves.
_CORE_ROOT = str(Path(__file__).resolve().parent.parent / "phm_core")
if _CORE_ROOT not in sys.path:
    sys.path.insert(0, _CORE_ROOT)

# Prune ROS workspace paths (same logic as repo-root conftest.py).
_ROS_MARKERS = ("/opt/ros/", "/ros2_ws/")
sys.path[:] = [p for p in sys.path if not any(m in p for m in _ROS_MARKERS)]
