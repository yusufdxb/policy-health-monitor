"""Tests for the Detector ABC and the DetectorVerdictData dataclass."""

from __future__ import annotations

import dataclasses

import pytest

from phm_core.detector import (
    ACTION_HOLD,
    ACTION_NONE,
    Detector,
    DetectorVerdictData,
)


def test_verdict_dataclass_mirrors_msg_fields():
    # Field names and order must match phm_msgs/DetectorVerdict.msg (minus the
    # std_msgs/Header the ROS node stamps at publish time).
    fields = [f.name for f in dataclasses.fields(DetectorVerdictData)]
    assert fields == ["source", "score", "violating", "reason", "suggested_action"]


def test_verdict_defaults_to_action_none():
    v = DetectorVerdictData(source="phm_ood", score=0.4, violating=True, reason="x")
    assert v.suggested_action == ACTION_NONE


def test_detector_abc_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Detector()  # abstract update method


def test_concrete_detector_returns_verdict():
    class SpreadDetector(Detector):
        name = "phm_ood"
        target_topic = "/policy/embedding"

        def update(self, sample):
            if sample < 0.5:
                return DetectorVerdictData(
                    source=self.name,
                    score=0.9,
                    violating=True,
                    reason=f"spread {sample} below floor",
                    suggested_action=ACTION_HOLD,
                )
            return None

    det = SpreadDetector()
    assert det.update(0.8) is None
    verdict = det.update(0.1)
    assert verdict is not None
    assert verdict.source == "phm_ood"
    assert verdict.violating is True
    assert verdict.suggested_action == ACTION_HOLD
