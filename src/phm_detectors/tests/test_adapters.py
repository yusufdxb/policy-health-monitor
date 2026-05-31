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

import numpy as np
import pytest
from phm_core.calibration import rolling_spread
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
    RecurrentSpreadSample,
    RecurrentTemporalSpreadAdapter,
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


# ---------------------------------------------------------------------------
# RecurrentTemporalSpreadAdapter tests (Phantom-Braking E6)
# ---------------------------------------------------------------------------

_EMB_TOPIC = "/policy/embedding"


def _build_spread_adapter(window: int = 30, threshold: float = 0.0,
                          min_consec: int = 2):
    return RecurrentTemporalSpreadAdapter(
        _EMB_TOPIC, window=window, threshold=threshold,
        min_consecutive=min_consec,
    )


def _emb(vec) -> RecurrentSpreadSample:
    return RecurrentSpreadSample(topic=_EMB_TOPIC, embedding=np.asarray(vec, float))


class TestRecurrentTemporalSpreadRegistration:
    """The detector is discoverable through the package registration surface
    (the package __init__ __all__, the same mechanism the other adapters use)."""

    def test_importable_from_package_root(self):
        import phm_detectors

        assert hasattr(phm_detectors, "RecurrentTemporalSpreadAdapter")
        assert hasattr(phm_detectors, "RecurrentSpreadSample")
        assert "RecurrentTemporalSpreadAdapter" in phm_detectors.__all__
        assert "RecurrentSpreadSample" in phm_detectors.__all__

    def test_is_a_phm_detector(self):
        from phm_core.detector import Detector

        adapter = _build_spread_adapter()
        assert isinstance(adapter, Detector)
        # Honors the Detector interface: name, target_topic, update().
        assert adapter.name == f"recurrent_temporal_spread:{_EMB_TOPIC}"
        assert adapter.target_topic == _EMB_TOPIC
        assert callable(adapter.update)


class TestRecurrentTemporalSpreadNumerics:
    """rolling_spread matches the Phantom-Braking reference on a fixed input,
    and threshold calibration returns a sane percentile."""

    def test_rolling_spread_matches_reference_fixed_input(self):
        # Fixed synthetic input: T=4 frames, D=2 dims, window=3.
        # Expected values computed by hand from the population (ddof=0) variance:
        #   t=2 (frames 0,1,2): col0=[0,1,2] var=2/3, col1=[0,0,0] var=0 -> 2/3
        #   t=3 (frames 1,2,3): col0=[1,2,3] var=2/3, col1=[0,0,3] var=2   -> 8/3
        H = np.array(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 3.0]], dtype=np.float64
        )
        s = rolling_spread(H, window=3)
        assert np.isnan(s[0]) and np.isnan(s[1])
        assert s[2] == pytest.approx(2.0 / 3.0)
        assert s[3] == pytest.approx(8.0 / 3.0)
        # Cross-check against the definition used in the Phantom-Braking source
        # (e6_detector.py:22): float(np.var(window, axis=0).sum()).
        assert s[2] == pytest.approx(float(np.var(H[0:3], axis=0).sum()))
        assert s[3] == pytest.approx(float(np.var(H[1:4], axis=0).sum()))

    def test_adapter_last_spread_matches_rolling_spread(self):
        # The adapter's per-window spread equals rolling_spread()[-1] on the
        # same window, i.e. it is the same math as the Phantom-Braking source.
        rng = np.random.default_rng(7)
        window = 8
        frames = rng.normal(size=(window, 5))
        adapter = _build_spread_adapter(window=window, threshold=0.0)
        v = None
        for f in frames:
            v = adapter.update(_emb(f))
        assert v is not None
        expected = float(rolling_spread(frames, window)[-1])
        assert adapter.last_spread == pytest.approx(expected)

    def test_calibrate_threshold_is_sane_percentile(self):
        # At the 1st percentile, ~1% of in-distribution frames fall below the
        # threshold (the Phantom-Braking calibration property). Use enough
        # frames that the empirical fraction is stable.
        rng = np.random.default_rng(0)
        window = 30
        ref = rng.normal(size=(500, 8))
        adapter = _build_spread_adapter(window=window)
        thr = adapter.calibrate_from_data(ref, percentile=1.0)
        assert adapter.threshold == thr
        spreads = rolling_spread(ref, window)
        valid = spreads[~np.isnan(spreads)]
        # Threshold lies inside the spread distribution.
        assert valid.min() <= thr <= valid.max()
        # ~1% of frames below the 1st-percentile threshold (tolerant band).
        frac_below = float(np.mean(valid < thr))
        assert 0.0 <= frac_below <= 0.05


class TestRecurrentTemporalSpreadBehavior:
    def test_wrong_topic_returns_none(self):
        adapter = _build_spread_adapter(window=3, threshold=1.0)
        v = adapter.update(
            RecurrentSpreadSample(topic="/other", embedding=np.zeros(4))
        )
        assert v is None

    def test_warmup_returns_healthy_until_window_fills(self):
        adapter = _build_spread_adapter(window=3, threshold=1.0)
        # First two frames are warm-up (buffer 1/3, 2/3).
        for i in range(2):
            v = adapter.update(_emb([float(i), 0.0]))
            assert v is not None
            assert v.violating is False
            assert v.score == 0.0
            assert "warming up" in v.reason

    def test_healthy_spread_not_violating(self):
        # A moving (spread-out) recurrent state stays above a low threshold.
        adapter = _build_spread_adapter(window=3, threshold=0.1, min_consec=1)
        adapter.update(_emb([0.0, 0.0]))
        adapter.update(_emb([1.0, 0.0]))
        v = adapter.update(_emb([2.0, 0.0]))  # spread = 2/3 >= 0.1
        assert v is not None
        assert v.violating is False
        assert v.source == f"recurrent_temporal_spread:{_EMB_TOPIC}"
        assert 0.0 <= v.score <= 1.0
        assert ">=" in v.reason

    def test_collapse_fires_after_hysteresis(self):
        # A frozen recurrent state has spread 0 < threshold -> OOD. With
        # min_consecutive=2, the first collapsed frame must NOT fire yet.
        adapter = _build_spread_adapter(window=3, threshold=1.0, min_consec=2)
        # Fill window with a moving state first (healthy), then freeze.
        adapter.update(_emb([0.0, 0.0]))
        adapter.update(_emb([5.0, 0.0]))
        adapter.update(_emb([10.0, 0.0]))  # window full, spread high, healthy
        frozen = _emb([2.0, 2.0])
        # Each subsequent frame is the same point; once the window is all the
        # same point spread -> 0.
        adapter.update(frozen)
        adapter.update(frozen)
        v_pre = adapter.update(frozen)  # window now fully frozen -> spread 0
        v_fire = adapter.update(frozen)
        assert v_pre is not None and v_pre.violating is False  # 1st below-thr
        assert v_fire is not None and v_fire.violating is True
        assert v_fire.score == pytest.approx(1.0)  # spread 0 -> worst
        assert v_fire.suggested_action == ACTION_STOP_AND_HOLD
        assert "<" in v_fire.reason

    def test_recovery_resets_hysteresis(self):
        adapter = _build_spread_adapter(window=2, threshold=1.0, min_consec=2)
        frozen = _emb([1.0, 1.0])
        moving_a = _emb([0.0, 0.0])
        moving_b = _emb([10.0, 10.0])
        # First frozen frame is warm-up (buffer 1/2); the next two fill an
        # all-frozen window (spread 0) and accumulate the two below-threshold
        # observations hysteresis needs to fire.
        adapter.update(frozen)  # warm-up, hysteresis untouched
        v1 = adapter.update(frozen)  # window full, 1st below-thr
        v_fire = adapter.update(frozen)  # 2nd below-thr -> fires
        assert v1 is not None and v1.violating is False
        assert v_fire is not None and v_fire.violating is True
        # Recover with a spread-out window.
        adapter.update(moving_a)
        v_ok = adapter.update(moving_b)  # window [moving_a, moving_b], high spread
        assert v_ok is not None and v_ok.violating is False
        # One more collapsed frame should not fire immediately (hysteresis
        # reset): it is again only the 1st below-threshold observation.
        adapter.update(frozen)  # window [moving_b, frozen]: still has spread
        v_after = adapter.update(frozen)  # window all-frozen, 1st below-thr again
        assert v_after is not None and v_after.violating is False

    def test_score_range_is_unit_interval(self):
        adapter = _build_spread_adapter(window=2, threshold=2.0, min_consec=1)
        rng = np.random.default_rng(3)
        for _ in range(20):
            v = adapter.update(_emb(rng.normal(size=4)))
            assert v is not None
            assert 0.0 <= v.score <= 1.0

    def test_window_must_be_at_least_two(self):
        with pytest.raises(ValueError):
            RecurrentTemporalSpreadAdapter(_EMB_TOPIC, window=1)

    def test_low_spread_is_the_unhealthy_direction(self):
        # Direction check vs the threshold adapter: here a LOW value fires.
        adapter = _build_spread_adapter(window=2, threshold=1.0, min_consec=1)
        # High-spread window: healthy.
        adapter.update(_emb([0.0, 0.0]))
        v_high = adapter.update(_emb([10.0, 10.0]))  # spread = 100 >= 1.0
        assert v_high is not None and v_high.violating is False
        # Collapsed window: same point -> spread 0 < 1.0 -> fires.
        adapter.update(_emb([4.0, 4.0]))
        v_low = adapter.update(_emb([4.0, 4.0]))  # spread 0
        assert v_low is not None and v_low.violating is True
