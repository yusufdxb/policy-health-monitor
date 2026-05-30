"""phm_detectors: bridges BlackBoxRS detector patterns into phm_core verdicts.

Three adapters (pure Python, no rclpy):

- ``FrequencyDropAdapter``   -- topic rate below learned baseline
- ``StaticThresholdAdapter`` -- metric above static limit
- ``DeadTopicAdapter``       -- topic silent for > timeout_sec

The rclpy node (``phm_detectors_node``) runs these adapters off the live ROS
graph and publishes ``phm_msgs/DetectorVerdict`` to ``/phm/verdicts``.
"""

from __future__ import annotations

from phm_detectors._core import (
    DeadTopicAdapter,
    DeadTopicSample,
    FrequencyDropAdapter,
    FrequencySample,
    StaticThresholdAdapter,
    ThresholdSample,
)

__all__ = [
    "DeadTopicAdapter",
    "DeadTopicSample",
    "FrequencyDropAdapter",
    "FrequencySample",
    "StaticThresholdAdapter",
    "ThresholdSample",
]
