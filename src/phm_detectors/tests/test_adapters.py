"""Unit tests for phm_detectors adapters.

All tests operate on pure-Python objects (FrequencyDropAdapter,
StaticThresholdAdapter, DeadTopicAdapter) and their sample dataclasses. No
rclpy is imported anywhere in this file. The tests exercise:

1. Healthy samples -> non-violating verdict.
2. Frequency drop below tolerance -> violating verdict after hysteresis.
3. Static threshold breach -> violating verdict after hysteresis.
4. Dead topic (silence > timeout) -> violating verdict.
5. Recovery: after a violating run a healthy sample resets hysteresis.
6. Score range is in [0, 1] and monotonically increases toward worst.
7. ``suggested_action`` values are from the phm_core action constants.
"""

from __future__ import annotations

import pytest
from phm_core.detector import (
    ACTION_HOLD,
    ACTION_NONE,
    ACTION_STOP_AND_HOLD,
)

from phm_detectors._core import (
    DeadTopicAdapter,
    DeadTopicSample,
    FrequencyDropAdapter,
    FrequencySample,
    StaticThresholdAdapter,
    ThresholdSample,
)

# ---------------------------------------------------------------------------
# FrequencyDropAdapter tests
# ---------------------------------------------------------------------------

_TOPIC = "/scan"
_BASELINE_HZ = 10.0
_LEARNING = 10  # must match _core._LEARNING_SAMPLES


def _build_freq_adapter(tolerance: float = 20.0, min_consec: int = 2):
    return FrequencyDropAdapter(_TOPIC, tolerance_percent=tolerance, min_consecutive=min_consec)


def _feed_learning(adapter: FrequencyDropAdapter, hz: float = _BASELINE_HZ) -> None:
    """Feed enough samples to exit the learning phase."""
    for _ in range(_LEARNING):
        adapter.update(FrequencySample(topic=_TOPIC, frequency_hz=hz))


class TestFrequencyDropAdapter:
    def test_learning_phase_returns_none(self):
        adapter = _build_freq_adapter()
        for i in range(_LEARNING - 1):
            result = adapter.update(FrequencySample(topic=_TOPIC, frequency_hz=_BASELINE_HZ))
            assert result is None, f"Expected None during learning (sample {i})"

    def test_wrong_topic_returns_none(self):
        adapter = _build_freq_adapter()
        result = adapter.update(FrequencySample(topic="/odom", frequency_hz=5.0))
        assert result is None

    def test_healthy_after_learning_not_violating(self):
        adapter = _build_freq_adapter(tolerance=20.0, min_consec=2)
        _feed_learning(adapter)
        # Send healthy sample (at baseline rate)
        v = adapter.update(FrequencySample(topic=_TOPIC, frequency_hz=_BASELINE_HZ))
        assert v is not None
        assert v.violating is False
        assert v.score >= 0.0
        assert v.score <= 1.0
        assert v.source == f"freq:{_TOPIC}"

    def test_frequency_drop_single_sample_not_yet_violating(self):
        # With min_consecutive=2, one violating sample is NOT enough to fire.
        adapter = _build_freq_adapter(tolerance=20.0, min_consec=2)
        _feed_learning(adapter)
        # Drop rate: baseline=10.0, floor=8.0; send 5.0 Hz (below floor).
        v = adapter.update(FrequencySample(topic=_TOPIC, frequency_hz=5.0))
        assert v is not None
        assert v.violating is False  # hysteresis not satisfied yet

    def test_frequency_drop_consecutive_violating(self):
        # Two consecutive violating samples -> violating=True.
        adapter = _build_freq_adapter(tolerance=20.0, min_consec=2)
        _feed_learning(adapter)
        dropped = FrequencySample(topic=_TOPIC, frequency_hz=3.0)  # far below floor
        v1 = adapter.update(dropped)
        v2 = adapter.update(dropped)
        assert v1 is not None and v1.violating is False
        assert v2 is not None and v2.violating is True
        assert v2.score > 0.0
        assert "below floor" in v2.reason

    def test_frequency_drop_score_range(self):
        adapter = _build_freq_adapter(tolerance=20.0, min_consec=1)
        _feed_learning(adapter)
        # Complete silence: score should approach 1.0
        v = adapter.update(FrequencySample(topic=_TOPIC, frequency_hz=0.0))
        assert v is not None
        assert 0.0 <= v.score <= 1.0
        assert v.score == pytest.approx(1.0)

    def test_frequency_drop_recovery_resets_hysteresis(self):
        # Violate twice, then recover, then violate once: should not fire.
        adapter = _build_freq_adapter(tolerance=20.0, min_consec=2)
        _feed_learning(adapter)
        dropped = FrequencySample(topic=_TOPIC, frequency_hz=1.0)
        healthy = FrequencySample(topic=_TOPIC, frequency_hz=_BASELINE_HZ)
        adapter.update(dropped)
        adapter.update(dropped)  # violating=True
        v_recovery = adapter.update(healthy)
        assert v_recovery is not None
        assert v_recovery.violating is False
        # One more drop should not violate (hysteresis reset)
        v_after = adapter.update(dropped)
        assert v_after is not None
        assert v_after.violating is False

    def test_frequency_drop_suggested_action_at_half_baseline(self):
        # 5.0 Hz out of 10.0 baseline -> score ~0.5 -> INTERVENE -> ACTION_HOLD.
        adapter = _build_freq_adapter(tolerance=20.0, min_consec=1)
        _feed_learning(adapter)
        v = adapter.update(FrequencySample(topic=_TOPIC, frequency_hz=5.0))
        assert v is not None
        assert v.suggested_action == ACTION_HOLD


# ---------------------------------------------------------------------------
# StaticThresholdAdapter tests
# ---------------------------------------------------------------------------

_METRIC = "cpu_percent"
_LIMIT = 80.0


def _build_thresh_adapter(min_consec: int = 2):
    return StaticThresholdAdapter("system:cpu", metric=_METRIC, min_consecutive=min_consec)


class TestStaticThresholdAdapter:
    def test_wrong_metric_returns_none(self):
        adapter = _build_thresh_adapter()
        result = adapter.update(
            ThresholdSample(metric="memory_percent", value=95.0, threshold=85.0)
        )
        assert result is None

    def test_healthy_sample_not_violating(self):
        adapter = _build_thresh_adapter(min_consec=2)
        v = adapter.update(ThresholdSample(metric=_METRIC, value=50.0, threshold=_LIMIT))
        assert v is not None
        assert v.violating is False
        assert v.source == f"threshold:{_METRIC}"
        assert 0.0 <= v.score <= 1.0

    def test_threshold_breach_single_not_yet_violating(self):
        adapter = _build_thresh_adapter(min_consec=2)
        v = adapter.update(ThresholdSample(metric=_METRIC, value=90.0, threshold=_LIMIT))
        assert v is not None
        assert v.violating is False  # need 2 consecutive

    def test_threshold_breach_consecutive_violating(self):
        adapter = _build_thresh_adapter(min_consec=2)
        s = ThresholdSample(metric=_METRIC, value=95.0, threshold=_LIMIT)
        v1 = adapter.update(s)
        v2 = adapter.update(s)
        assert v1 is not None and v1.violating is False
        assert v2 is not None and v2.violating is True
        assert "exceeds" in v2.reason

    def test_threshold_breach_score_at_double_limit(self):
        # value == threshold * 2 -> score == 1.0
        adapter = _build_thresh_adapter(min_consec=1)
        v = adapter.update(ThresholdSample(metric=_METRIC, value=_LIMIT * 2, threshold=_LIMIT))
        assert v is not None
        assert v.score == pytest.approx(1.0)

    def test_threshold_healthy_score_below_half(self):
        # value == threshold (exactly at limit) -> score == 0.5
        adapter = _build_thresh_adapter(min_consec=1)
        v = adapter.update(ThresholdSample(metric=_METRIC, value=_LIMIT, threshold=_LIMIT))
        assert v is not None
        # value == threshold: score = normalize(80, 0, 160) = 0.5
        assert v.score == pytest.approx(0.5)

    def test_threshold_recovery_resets_hysteresis(self):
        adapter = _build_thresh_adapter(min_consec=2)
        breach = ThresholdSample(metric=_METRIC, value=95.0, threshold=_LIMIT)
        healthy = ThresholdSample(metric=_METRIC, value=30.0, threshold=_LIMIT)
        adapter.update(breach)
        adapter.update(breach)  # violating now
        v_ok = adapter.update(healthy)
        assert v_ok is not None
        assert v_ok.violating is False
        # one more breach: not violating (needs 2 consecutive again)
        v_one = adapter.update(breach)
        assert v_one is not None
        assert v_one.violating is False

    def test_threshold_action_at_threshold(self):
        # At exactly threshold, score=0.5 -> INTERVENE -> ACTION_HOLD.
        adapter = _build_thresh_adapter(min_consec=1)
        v = adapter.update(ThresholdSample(metric=_METRIC, value=_LIMIT, threshold=_LIMIT))
        assert v is not None
        assert v.suggested_action == ACTION_HOLD

    def test_threshold_action_healthy(self):
        # Below threshold -> healthy -> ACTION_NONE.
        adapter = _build_thresh_adapter(min_consec=1)
        v = adapter.update(ThresholdSample(metric=_METRIC, value=20.0, threshold=_LIMIT))
        assert v is not None
        assert v.suggested_action == ACTION_NONE


# ---------------------------------------------------------------------------
# DeadTopicAdapter tests
# ---------------------------------------------------------------------------

_DEAD_TOPIC = "/imu/data"
_TIMEOUT = 5.0


def _build_dead_adapter(timeout: float = _TIMEOUT):
    return DeadTopicAdapter(_DEAD_TOPIC, timeout_sec=timeout)


class TestDeadTopicAdapter:
    def test_wrong_topic_returns_none(self):
        adapter = _build_dead_adapter()
        result = adapter.update(
            DeadTopicSample(topic="/odom", last_seen_sec=0.0, now_sec=10.0)
        )
        assert result is None

    def test_alive_topic_not_violating(self):
        adapter = _build_dead_adapter(timeout=5.0)
        v = adapter.update(DeadTopicSample(topic=_DEAD_TOPIC, last_seen_sec=100.0, now_sec=101.0))
        assert v is not None
        assert v.violating is False
        assert v.source == f"dead:{_DEAD_TOPIC}"

    def test_dead_topic_violating(self):
        adapter = _build_dead_adapter(timeout=5.0)
        # Silent for 10 seconds (> 5s timeout)
        v = adapter.update(DeadTopicSample(topic=_DEAD_TOPIC, last_seen_sec=0.0, now_sec=10.0))
        assert v is not None
        assert v.violating is True
        assert v.suggested_action == ACTION_STOP_AND_HOLD
        assert "silent" in v.reason

    def test_dead_topic_score_at_double_timeout(self):
        # elapsed == timeout * 2 -> score == 1.0
        adapter = _build_dead_adapter(timeout=5.0)
        v = adapter.update(
            DeadTopicSample(topic=_DEAD_TOPIC, last_seen_sec=0.0, now_sec=10.0)
        )
        assert v is not None
        assert v.score == pytest.approx(1.0)

    def test_dead_topic_score_range_alive(self):
        adapter = _build_dead_adapter(timeout=5.0)
        v = adapter.update(
            DeadTopicSample(topic=_DEAD_TOPIC, last_seen_sec=99.5, now_sec=100.0)
        )
        assert v is not None
        assert 0.0 <= v.score <= 1.0

    def test_dead_topic_recovery_clears_alert(self):
        adapter = _build_dead_adapter(timeout=5.0)
        # Fire the dead verdict.
        v1 = adapter.update(DeadTopicSample(topic=_DEAD_TOPIC, last_seen_sec=0.0, now_sec=10.0))
        assert v1 is not None and v1.violating is True
        # Mark alive.
        adapter.mark_alive(now_sec=10.0)
        # Now check: elapsed 0 -> healthy.
        v2 = adapter.update(DeadTopicSample(topic=_DEAD_TOPIC, last_seen_sec=10.0, now_sec=10.5))
        assert v2 is not None
        assert v2.violating is False

    def test_dead_topic_just_at_timeout_boundary(self):
        adapter = _build_dead_adapter(timeout=5.0)
        # elapsed == timeout exactly -> raw_violating is False (not > timeout)
        v = adapter.update(
            DeadTopicSample(topic=_DEAD_TOPIC, last_seen_sec=95.0, now_sec=100.0)
        )
        assert v is not None
        # elapsed == 5.0, raw_violating = (5.0 > 5.0) = False
        assert v.violating is False

    def test_dead_topic_just_over_timeout(self):
        adapter = _build_dead_adapter(timeout=5.0)
        v = adapter.update(
            DeadTopicSample(topic=_DEAD_TOPIC, last_seen_sec=94.9, now_sec=100.0)
        )
        assert v is not None
        assert v.violating is True  # elapsed = 5.1 > 5.0


# ---------------------------------------------------------------------------
# Integration: all three detectors on a "robot health check" sequence
# ---------------------------------------------------------------------------

class TestIntegration:
    """Synthetic end-to-end scenario: healthy system, then three simultaneous
    faults, then recovery."""

    def test_healthy_system_produces_no_violations(self):
        freq = _build_freq_adapter(tolerance=20.0, min_consec=2)
        thresh = _build_thresh_adapter(min_consec=2)
        dead = _build_dead_adapter(timeout=5.0)

        _feed_learning(freq)

        now = 1000.0
        for _i in range(5):
            fv = freq.update(FrequencySample(_TOPIC, _BASELINE_HZ))
            tv = thresh.update(ThresholdSample(_METRIC, 40.0, _LIMIT))
            dv = dead.update(DeadTopicSample(_DEAD_TOPIC, now - 0.5, now))
            now += 1.0
            assert fv is not None and fv.violating is False
            assert tv is not None and tv.violating is False
            assert dv is not None and dv.violating is False

    def test_simultaneous_faults_all_violate(self):
        freq = _build_freq_adapter(tolerance=20.0, min_consec=2)
        thresh = _build_thresh_adapter(min_consec=2)
        dead = _build_dead_adapter(timeout=5.0)

        _feed_learning(freq)

        # Two rounds of faults to satisfy hysteresis on freq and thresh.
        dropped = FrequencySample(_TOPIC, 1.0)
        breach = ThresholdSample(_METRIC, 95.0, _LIMIT)
        now = 200.0

        for _ in range(2):
            freq.update(dropped)
            thresh.update(breach)

        fv = freq.update(dropped)
        tv = thresh.update(breach)
        dv = dead.update(DeadTopicSample(_DEAD_TOPIC, 0.0, now))

        assert fv is not None and fv.violating is True
        assert tv is not None and tv.violating is True
        assert dv is not None and dv.violating is True

    def test_recovery_after_faults(self):
        freq = _build_freq_adapter(tolerance=20.0, min_consec=2)
        thresh = _build_thresh_adapter(min_consec=2)
        dead = _build_dead_adapter(timeout=5.0)

        _feed_learning(freq)

        dropped = FrequencySample(_TOPIC, 1.0)
        breach = ThresholdSample(_METRIC, 95.0, _LIMIT)
        now = 300.0

        # Induce violation.
        freq.update(dropped)
        freq.update(dropped)
        thresh.update(breach)
        thresh.update(breach)
        dead.update(DeadTopicSample(_DEAD_TOPIC, 0.0, now))

        # Recover.
        dead.mark_alive(now)
        now += 0.1
        fv = freq.update(FrequencySample(_TOPIC, _BASELINE_HZ))
        tv = thresh.update(ThresholdSample(_METRIC, 30.0, _LIMIT))
        dv = dead.update(DeadTopicSample(_DEAD_TOPIC, now - 0.05, now))

        assert fv is not None and fv.violating is False
        assert tv is not None and tv.violating is False
        assert dv is not None and dv.violating is False
