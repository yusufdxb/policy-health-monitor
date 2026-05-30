"""Repo-root pytest configuration for the pure-Python packages.

phm_core has no ROS dependency. On a machine where a ROS 2 overlay has been
sourced into the shell, ``PYTHONPATH`` injects ``/opt/ros`` and workspace
``install`` site-packages into ``sys.path``, which makes pytest try to autoload
ROS pytest plugins (ament_cmake, launch_testing). Those assume a ROS graph and
pull in deps (pyyaml, etc.) the pure-Python venv does not carry.

This module prunes ROS workspace and ``/opt/ros`` entries from ``sys.path`` so a
plain ``pytest src/phm_core`` collects only this repo. CI runs in a clean
setup-python environment where these paths are absent, so this is a no-op there.
"""

from __future__ import annotations

import sys

_ROS_MARKERS = ("/opt/ros/", "/ros2_ws/")
sys.path[:] = [p for p in sys.path if not any(m in p for m in _ROS_MARKERS)]
