"""phm_recovery: ROS 2 recovery node for the Policy Health Monitor.

The rclpy node in recovery_node.py is a thin wrapper. All testable logic lives
in _core.py (pure Python, no ROS dependency).
"""

from phm_recovery._core import (
    ACTION_HOLD,
    ACTION_LOG_ONLY,
    ACTION_NONE,
    ACTION_REWIND,
    ACTION_STOP_AND_HOLD,
    STATE_DEGRADED,
    STATE_INTERVENE,
    STATE_OK,
    STATE_STOP,
    EnvelopeResult,
    HealthActionDecision,
    HealthToActionMapper,
    RewindHook,
    SafetyEnvelope,
)

__all__ = [
    "ACTION_HOLD",
    "ACTION_LOG_ONLY",
    "ACTION_NONE",
    "ACTION_REWIND",
    "ACTION_STOP_AND_HOLD",
    "EnvelopeResult",
    "HealthActionDecision",
    "HealthToActionMapper",
    "RewindHook",
    "SafetyEnvelope",
    "STATE_DEGRADED",
    "STATE_INTERVENE",
    "STATE_OK",
    "STATE_STOP",
]
