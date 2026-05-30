"""phm_arbiter: fusion node that arbitrates DetectorVerdict -> /phm/health.

The pure arbitration logic lives in ``_core.py`` (no ROS deps, testable
standalone). The rclpy node is in ``arbiter_node.py``.
"""

from __future__ import annotations

from phm_arbiter._core import (
    ACTION_HOLD,
    ACTION_LOG_ONLY,
    ACTION_NONE,
    ACTION_REWIND,
    ACTION_STOP_AND_HOLD,
    ARBITER_ALLOWLIST,
    STATE_DEGRADED,
    STATE_INTERVENE,
    STATE_OK,
    STATE_STOP,
    PolicyHealthStatusData,
    arbitrate,
)

__all__ = [
    "ACTION_HOLD",
    "ACTION_LOG_ONLY",
    "ACTION_NONE",
    "ACTION_REWIND",
    "ACTION_STOP_AND_HOLD",
    "ARBITER_ALLOWLIST",
    "STATE_DEGRADED",
    "STATE_INTERVENE",
    "STATE_OK",
    "STATE_STOP",
    "PolicyHealthStatusData",
    "arbitrate",
]
