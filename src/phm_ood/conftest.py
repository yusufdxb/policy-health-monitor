"""Pytest configuration for phm_ood tests.

Adds the phm_core source directory to sys.path so tests can import phm_core
without a colcon install. Also prunes any ROS workspace entries from sys.path
so ament pytest plugins are not auto-loaded.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Prune ROS overlay entries (same pattern as repo-root conftest.py).
_ROS_MARKERS = ("/opt/ros/", "/ros2_ws/")
sys.path[:] = [p for p in sys.path if not any(m in p for m in _ROS_MARKERS)]

# Add phm_core source so `import phm_core` works without a colcon install.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PHM_CORE_SRC = str(_REPO_ROOT / "src" / "phm_core")
if _PHM_CORE_SRC not in sys.path:
    sys.path.insert(0, _PHM_CORE_SRC)

# Add phm_ood source directory itself so `import phm_ood` works.
_PHM_OOD_SRC = str(Path(__file__).resolve().parent)
if _PHM_OOD_SRC not in sys.path:
    sys.path.insert(0, _PHM_OOD_SRC)
