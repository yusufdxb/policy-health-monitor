"""Detector ABC and the plain dataclass that mirrors DetectorVerdict.msg.

The dataclass decouples detector logic from rclpy: detectors return a
``DetectorVerdictData`` (pure Python) and a thin ROS 2 node wrapper converts it
to ``phm_msgs/DetectorVerdict`` for publishing. This lets phm_core be unit-tested
with no ROS graph.

Interface adapted from BlackBoxRS
``blackboxrs/anomaly_engine/detectors/base.py:35`` (BaseDetector: ``name`` plus a
single check method that returns an event or None). Here ``check`` is renamed
``update`` per the spec and the return type is the pure-Python verdict dataclass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

# suggested_action enum, mirrors PolicyHealthStatus.msg / the arbiter allowlist.
ACTION_NONE = 0
ACTION_LOG_ONLY = 1
ACTION_HOLD = 2
ACTION_STOP_AND_HOLD = 3
ACTION_REWIND = 4


@dataclass
class DetectorVerdictData:
    """Pure-Python mirror of ``phm_msgs/DetectorVerdict.msg``.

    Field names and order match the .msg exactly (minus the std_msgs/Header,
    which the ROS node stamps at publish time):

    - ``source``: which detector produced this verdict.
    - ``score``: normalized severity in [0, 1], 0 healthy, 1 worst.
    - ``violating``: post-hysteresis boolean.
    - ``reason``: human-readable explanation.
    - ``suggested_action``: one of the ACTION_* constants.
    """

    source: str
    score: float
    violating: bool
    reason: str
    suggested_action: int = ACTION_NONE


class Detector(ABC):
    """Abstract base for all PHM detectors.

    A detector watches one logical input (a topic, a metric, a policy
    embedding stream) and turns each sample into a partial verdict. The
    arbiter fuses verdicts from every detector into a single health signal.

    Subclasses provide ``name`` and ``target_topic`` and implement
    :meth:`update`, returning a :class:`DetectorVerdictData` when there is a
    verdict to report, or ``None`` to stay silent for this sample.
    """

    #: Unique, human-readable identifier, also used as DetectorVerdict.source.
    name: str
    #: The logical input this detector watches, e.g. "/policy/embedding".
    target_topic: str

    @abstractmethod
    def update(self, sample: Any) -> DetectorVerdictData | None:
        """Process one sample and optionally emit a verdict.

        Args:
            sample: the latest input for this detector. Type is detector
                specific (an embedding array, a timestamp, a metric value).

        Returns:
            A :class:`DetectorVerdictData` if this detector has a verdict to
            report for the sample, otherwise ``None``.
        """
        raise NotImplementedError
