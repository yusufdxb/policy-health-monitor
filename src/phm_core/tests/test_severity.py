"""Tests for the severity banding (the single source of state thresholds)."""

from __future__ import annotations

import pytest

from phm_core.severity import (
    ACTION_HOLD,
    ACTION_LOG_ONLY,
    ACTION_NONE,
    ACTION_STOP_AND_HOLD,
    DEGRADED_THRESHOLD,
    INTERVENE_THRESHOLD,
    STATE_DEGRADED,
    STATE_INTERVENE,
    STATE_OK,
    STATE_STOP,
    STOP_THRESHOLD,
    classify,
    normalize,
)


@pytest.mark.parametrize(
    "score,exp_state,exp_action",
    [
        (0.0, STATE_OK, ACTION_NONE),
        (0.10, STATE_OK, ACTION_NONE),
        (0.249, STATE_OK, ACTION_NONE),
        (0.25, STATE_DEGRADED, ACTION_LOG_ONLY),
        (0.40, STATE_DEGRADED, ACTION_LOG_ONLY),
        (0.499, STATE_DEGRADED, ACTION_LOG_ONLY),
        (0.50, STATE_INTERVENE, ACTION_HOLD),
        (0.70, STATE_INTERVENE, ACTION_HOLD),
        (0.799, STATE_INTERVENE, ACTION_HOLD),
        (0.80, STATE_STOP, ACTION_STOP_AND_HOLD),
        (0.95, STATE_STOP, ACTION_STOP_AND_HOLD),
        (1.0, STATE_STOP, ACTION_STOP_AND_HOLD),
    ],
)
def test_classify_bands(score, exp_state, exp_action):
    sev = classify(score)
    assert sev.state == exp_state
    assert sev.suggested_action == exp_action


def test_band_edges_are_inclusive_at_lower_edge():
    # Exactly at each threshold the higher-severity band wins.
    assert classify(DEGRADED_THRESHOLD).state == STATE_DEGRADED
    assert classify(INTERVENE_THRESHOLD).state == STATE_INTERVENE
    assert classify(STOP_THRESHOLD).state == STATE_STOP


def test_classify_clamps_out_of_range_scores():
    low = classify(-0.5)
    assert low.score == 0.0
    assert low.state == STATE_OK
    high = classify(1.7)
    assert high.score == 1.0
    assert high.state == STATE_STOP


def test_normalize_forward_direction():
    # healthy=0, worst=10: raw 5 -> 0.5, raw 2.5 -> 0.25, clamps below/above.
    assert normalize(5.0, 0.0, 10.0) == pytest.approx(0.5)
    assert normalize(2.5, 0.0, 10.0) == pytest.approx(0.25)
    assert normalize(-3.0, 0.0, 10.0) == 0.0
    assert normalize(99.0, 0.0, 10.0) == 1.0


def test_normalize_inverted_direction_for_rolling_spread():
    # Rolling-spread collapse: a LOW value is unhealthy. healthy=1.0 (high
    # spread), worst=0.0 (collapsed). raw 0.5 -> 0.5, raw 0.0 -> 1.0 (worst).
    assert normalize(0.5, 1.0, 0.0) == pytest.approx(0.5)
    assert normalize(0.0, 1.0, 0.0) == pytest.approx(1.0)
    assert normalize(1.0, 1.0, 0.0) == pytest.approx(0.0)
    assert normalize(2.0, 1.0, 0.0) == 0.0  # even healthier than healthy, clamps


def test_normalize_rejects_degenerate_scale():
    with pytest.raises(ValueError):
        normalize(1.0, 3.0, 3.0)


def test_normalized_spread_classifies_to_stop():
    # A fully collapsed hidden state (raw spread 0, healthy 1) is worst-case ->
    # STOP through the full normalize+classify pipeline.
    score = normalize(0.0, 1.0, 0.0)
    sev = classify(score)
    assert sev.state == STATE_STOP
    assert sev.suggested_action == ACTION_STOP_AND_HOLD
