"""phm_core: pure-Python detector logic for the Policy Health Monitor.

No ROS dependency, so these modules unit-test fast and without a ROS graph. The
rclpy nodes in the sibling packages are thin wrappers over this logic.
"""

from __future__ import annotations

from phm_core.calibration import calibrate_threshold, loco_fpr, rolling_spread
from phm_core.detector import (
    ACTION_HOLD,
    ACTION_LOG_ONLY,
    ACTION_NONE,
    ACTION_REWIND,
    ACTION_STOP_AND_HOLD,
    Detector,
    DetectorVerdictData,
)
from phm_core.hysteresis import Hysteresis
from phm_core.severity import (
    DEGRADED_THRESHOLD,
    INTERVENE_THRESHOLD,
    STATE_DEGRADED,
    STATE_INTERVENE,
    STATE_OK,
    STATE_STOP,
    STOP_THRESHOLD,
    Severity,
    classify,
    normalize,
)

__all__ = [
    "ACTION_HOLD",
    "ACTION_LOG_ONLY",
    "ACTION_NONE",
    "ACTION_REWIND",
    "ACTION_STOP_AND_HOLD",
    "DEGRADED_THRESHOLD",
    "Detector",
    "DetectorVerdictData",
    "Hysteresis",
    "INTERVENE_THRESHOLD",
    "STATE_DEGRADED",
    "STATE_INTERVENE",
    "STATE_OK",
    "STATE_STOP",
    "STOP_THRESHOLD",
    "Severity",
    "calibrate_threshold",
    "classify",
    "loco_fpr",
    "normalize",
    "rolling_spread",
]
