"""Make phm_detectors and phm_core importable when running pytest without an
editable install.

Adds both package roots to ``sys.path`` so ``import phm_detectors`` and
``import phm_core`` resolve from a fresh checkout, the same pattern used by
``src/phm_core/conftest.py``.

Also prunes ROS workspace and /opt/ros entries from sys.path to prevent pytest
from loading ROS plugins (ament_cmake, launch_testing) that assume a ROS graph
and pull in deps the pure-Python venv does not carry.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add phm_detectors package root.
_DET_ROOT = Path(__file__).resolve().parent
if str(_DET_ROOT) not in sys.path:
    sys.path.insert(0, str(_DET_ROOT))

# Add phm_core package root (sibling package).
_CORE_ROOT = _DET_ROOT.parent / "phm_core"
if str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))

# Prune ROS overlays to prevent auto-loading ROS pytest plugins.
_ROS_MARKERS = ("/opt/ros/", "/ros2_ws/")
sys.path[:] = [p for p in sys.path if not any(m in p for m in _ROS_MARKERS)]
