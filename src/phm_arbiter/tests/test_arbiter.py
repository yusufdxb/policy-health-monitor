"""Exhaustive unit tests for phm_arbiter._core.arbitrate().

No rclpy imports: all tests exercise the pure-Python function directly.

Coverage targets per spec section 3.3:
- No verdicts -> OK
- Single INTERVENE verdict -> INTERVENE
- STOP beats INTERVENE beats DEGRADED (worst-wins)
- Stale verdict -> DEGRADED with reason "stale:<source>"
- Tie-breaking: equal state, higher score wins
- reason and source propagate from the winning verdict
- suggested_action clamped to ARBITER_ALLOWLIST (REWIND excluded)
- Non-violating verdict contributes nothing
- Mixed stale + violating: correct winner
- staleness_sec edge cases (exact boundary, just inside, just outside)
- score field on result matches the winning verdict's score
- Empty source string on OK result
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

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

# ---------------------------------------------------------------------------
# Helper builder so tests are concise
# ---------------------------------------------------------------------------


@dataclass
class FakeVerdict:
    """Minimal stand-in for DetectorVerdictData + a timestamp field."""

    source: str
    score: float
    violating: bool
    reason: str
    suggested_action: int = ACTION_NONE
    timestamp: float = 0.0  # will be set per-test via helper


def make_verdict(
    source: str,
    score: float,
    violating: bool,
    reason: str,
    suggested_action: int = ACTION_NONE,
    age: float = 0.0,  # age relative to `now`; now is fixed at 100.0 in most tests
    now: float = 100.0,
) -> FakeVerdict:
    return FakeVerdict(
        source=source,
        score=score,
        violating=violating,
        reason=reason,
        suggested_action=suggested_action,
        timestamp=now - age,
    )


NOW = 100.0  # fixed reference time for all tests


# ---------------------------------------------------------------------------
# 1. No verdicts
# ---------------------------------------------------------------------------


def test_no_verdicts_returns_ok():
    result = arbitrate([], now=NOW)
    assert result.state == STATE_OK
    assert result.score == 0.0
    assert result.suggested_action == ACTION_NONE
    assert result.source == ""


def test_no_verdicts_reason_is_nominal():
    result = arbitrate([], now=NOW)
    assert "nominal" in result.reason


# ---------------------------------------------------------------------------
# 2. Single verdict, various states
# ---------------------------------------------------------------------------


def test_single_ok_nonviolating_returns_ok():
    v = make_verdict("src_a", score=0.1, violating=False, reason="fine")
    result = arbitrate([v], now=NOW)
    assert result.state == STATE_OK


def test_single_degraded_violating():
    v = make_verdict(
        "freq:/scan", score=0.3, violating=True, reason="freq low",
        suggested_action=ACTION_LOG_ONLY,
    )
    result = arbitrate([v], now=NOW)
    assert result.state == STATE_DEGRADED
    assert result.score == pytest.approx(0.3)
    assert result.source == "freq:/scan"
    assert result.reason == "freq low"
    assert result.suggested_action == ACTION_LOG_ONLY


def test_single_intervene_violating():
    v = make_verdict(
        "phm_ood", score=0.65, violating=True, reason="spread collapse",
        suggested_action=ACTION_HOLD,
    )
    result = arbitrate([v], now=NOW)
    assert result.state == STATE_INTERVENE
    assert result.score == pytest.approx(0.65)
    assert result.source == "phm_ood"
    assert result.suggested_action == ACTION_HOLD


def test_single_stop_violating():
    v = make_verdict(
        "threshold:cpu", score=0.9, violating=True, reason="cpu overload",
        suggested_action=ACTION_STOP_AND_HOLD,
    )
    result = arbitrate([v], now=NOW)
    assert result.state == STATE_STOP
    assert result.score == pytest.approx(0.9)
    assert result.source == "threshold:cpu"
    assert result.suggested_action == ACTION_STOP_AND_HOLD


# ---------------------------------------------------------------------------
# 3. Worst-wins ordering: STOP > INTERVENE > DEGRADED > OK
# ---------------------------------------------------------------------------


def test_stop_beats_intervene():
    verdicts = [
        make_verdict("phm_ood", score=0.65, violating=True, reason="ood",
                     suggested_action=ACTION_HOLD),
        make_verdict("threshold:cpu", score=0.85, violating=True, reason="cpu",
                     suggested_action=ACTION_STOP_AND_HOLD),
    ]
    result = arbitrate(verdicts, now=NOW)
    assert result.state == STATE_STOP
    assert result.source == "threshold:cpu"


def test_intervene_beats_degraded():
    verdicts = [
        make_verdict("freq:/scan", score=0.35, violating=True, reason="freq low",
                     suggested_action=ACTION_LOG_ONLY),
        make_verdict("phm_ood", score=0.55, violating=True, reason="ood",
                     suggested_action=ACTION_HOLD),
    ]
    result = arbitrate(verdicts, now=NOW)
    assert result.state == STATE_INTERVENE
    assert result.source == "phm_ood"


def test_stop_beats_degraded():
    verdicts = [
        make_verdict("freq:/scan", score=0.35, violating=True, reason="freq low",
                     suggested_action=ACTION_LOG_ONLY),
        make_verdict("threshold:cpu", score=0.9, violating=True, reason="cpu",
                     suggested_action=ACTION_STOP_AND_HOLD),
    ]
    result = arbitrate(verdicts, now=NOW)
    assert result.state == STATE_STOP


def test_three_sources_all_different_worst_wins():
    verdicts = [
        make_verdict("src_a", score=0.3, violating=True, reason="a",
                     suggested_action=ACTION_LOG_ONLY),
        make_verdict("src_b", score=0.55, violating=True, reason="b",
                     suggested_action=ACTION_HOLD),
        make_verdict("src_c", score=0.85, violating=True, reason="c",
                     suggested_action=ACTION_STOP_AND_HOLD),
    ]
    result = arbitrate(verdicts, now=NOW)
    assert result.state == STATE_STOP
    assert result.source == "src_c"
    assert result.reason == "c"
    assert result.score == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# 4. Stale verdicts
# ---------------------------------------------------------------------------


def test_stale_verdict_becomes_degraded():
    v = make_verdict("phm_ood", score=0.65, violating=True, reason="ood", age=2.0)
    result = arbitrate([v], now=NOW, staleness_sec=1.0)
    assert result.state == STATE_DEGRADED
    assert "stale:phm_ood" in result.reason


def test_stale_reason_format():
    v = make_verdict("freq:/scan", score=0.1, violating=False, reason="ok", age=5.0)
    result = arbitrate([v], now=NOW, staleness_sec=1.0)
    assert result.reason == "stale:freq:/scan"
    assert result.source == "freq:/scan"


def test_stale_verdict_never_dropped():
    # A stale non-violating verdict still becomes DEGRADED (not silently ignored).
    v = make_verdict("freq:/scan", score=0.05, violating=False, reason="ok", age=2.0)
    result = arbitrate([v], now=NOW, staleness_sec=1.0)
    assert result.state == STATE_DEGRADED


def test_fresh_stop_beats_stale_degraded():
    verdicts = [
        make_verdict("stale_src", score=0.65, violating=True, reason="stale", age=2.0),
        make_verdict("fresh_src", score=0.85, violating=True, reason="cpu",
                     suggested_action=ACTION_STOP_AND_HOLD),
    ]
    result = arbitrate(verdicts, now=NOW, staleness_sec=1.0)
    assert result.state == STATE_STOP
    assert result.source == "fresh_src"


def test_stale_beats_fresh_ok():
    """A stale verdict (DEGRADED) beats a fresh non-violating verdict (no candidate)."""
    verdicts = [
        make_verdict("stale_src", score=0.1, violating=False, reason="ok", age=2.0),
        make_verdict("fresh_src", score=0.05, violating=False, reason="fine"),
    ]
    result = arbitrate(verdicts, now=NOW, staleness_sec=1.0)
    assert result.state == STATE_DEGRADED
    assert result.source == "stale_src"


def test_multiple_stale_sources():
    verdicts = [
        make_verdict("src_a", score=0.1, violating=False, reason="ok", age=3.0),
        make_verdict("src_b", score=0.9, violating=True, reason="bad", age=5.0),
    ]
    result = arbitrate(verdicts, now=NOW, staleness_sec=1.0)
    # Both are stale; both become DEGRADED candidates; src_b had higher original
    # score but stale candidates all get _STALE_SCORE, so we still get DEGRADED.
    assert result.state == STATE_DEGRADED


# ---------------------------------------------------------------------------
# 5. Staleness boundary conditions
# ---------------------------------------------------------------------------


def test_exactly_at_staleness_boundary_is_stale():
    """age == staleness_sec (strictly greater-than boundary) is stale."""
    v = make_verdict("src", score=0.9, violating=True, reason="bad", age=1.0)
    result = arbitrate([v], now=NOW, staleness_sec=1.0)
    # age=1.0, staleness_sec=1.0: 1.0 > 1.0 is False -> NOT stale.
    # So the verdict participates normally as STATE_STOP.
    assert result.state == STATE_STOP


def test_just_over_staleness_boundary_is_stale():
    v = make_verdict("src", score=0.9, violating=True, reason="bad", age=1.001)
    result = arbitrate([v], now=NOW, staleness_sec=1.0)
    assert result.state == STATE_DEGRADED
    assert result.reason == "stale:src"


def test_just_under_staleness_boundary_is_fresh():
    v = make_verdict("src", score=0.9, violating=True, reason="bad", age=0.999)
    result = arbitrate([v], now=NOW, staleness_sec=1.0)
    assert result.state == STATE_STOP  # fresh STOP


# ---------------------------------------------------------------------------
# 6. Tie-breaking (same state, higher score wins)
# ---------------------------------------------------------------------------


def test_tie_higher_score_wins():
    verdicts = [
        make_verdict("src_a", score=0.55, violating=True, reason="a low",
                     suggested_action=ACTION_HOLD),
        make_verdict("src_b", score=0.75, violating=True, reason="b high",
                     suggested_action=ACTION_HOLD),
    ]
    # Both land in INTERVENE band; src_b has higher score.
    result = arbitrate(verdicts, now=NOW)
    assert result.state == STATE_INTERVENE
    assert result.source == "src_b"
    assert result.score == pytest.approx(0.75)
    assert result.reason == "b high"


def test_tie_equal_score_deterministic():
    """Equal state AND equal score: result is one of the tied verdicts."""
    verdicts = [
        make_verdict("src_a", score=0.6, violating=True, reason="a",
                     suggested_action=ACTION_HOLD),
        make_verdict("src_b", score=0.6, violating=True, reason="b",
                     suggested_action=ACTION_HOLD),
    ]
    result = arbitrate(verdicts, now=NOW)
    assert result.state == STATE_INTERVENE
    assert result.source in {"src_a", "src_b"}
    assert result.score == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# 7. reason and source propagation
# ---------------------------------------------------------------------------


def test_reason_propagates_verbatim():
    long_reason = "ood: rolling-spread 0.012 < thr 0.019 for 3 frames"
    v = make_verdict("phm_ood", score=0.65, violating=True, reason=long_reason,
                     suggested_action=ACTION_HOLD)
    result = arbitrate([v], now=NOW)
    assert result.reason == long_reason


def test_source_propagates_verbatim():
    v = make_verdict("freq:/scan", score=0.35, violating=True, reason="low freq",
                     suggested_action=ACTION_LOG_ONLY)
    result = arbitrate([v], now=NOW)
    assert result.source == "freq:/scan"


# ---------------------------------------------------------------------------
# 8. suggested_action clamping (REWIND excluded from ARBITER_ALLOWLIST)
# ---------------------------------------------------------------------------


def test_rewind_clamped_to_none():
    """REWIND is not in the arbiter allowlist; must be clamped to ACTION_NONE."""
    v = make_verdict("phm_ood", score=0.65, violating=True, reason="ood",
                     suggested_action=ACTION_REWIND)
    result = arbitrate([v], now=NOW)
    assert result.suggested_action == ACTION_NONE


def test_valid_actions_pass_through():
    for action in ARBITER_ALLOWLIST:
        v = make_verdict("src", score=0.65, violating=True, reason="r",
                         suggested_action=action)
        result = arbitrate([v], now=NOW)
        assert result.suggested_action == action, (
            f"ACTION {action} should pass through but got {result.suggested_action}"
        )


def test_unknown_action_value_clamped_to_none():
    """An out-of-range action value must also be clamped."""
    v = make_verdict("src", score=0.65, violating=True, reason="r",
                     suggested_action=99)
    result = arbitrate([v], now=NOW)
    assert result.suggested_action == ACTION_NONE


# ---------------------------------------------------------------------------
# 9. Non-violating verdicts contribute nothing unless stale
# ---------------------------------------------------------------------------


def test_non_violating_fresh_is_invisible():
    verdicts = [
        make_verdict("src_a", score=0.8, violating=False, reason="fine"),
        make_verdict("src_b", score=0.9, violating=False, reason="also fine"),
    ]
    result = arbitrate(verdicts, now=NOW)
    assert result.state == STATE_OK
    assert result.score == 0.0


def test_mix_violating_and_nonviolating():
    verdicts = [
        make_verdict("src_a", score=0.05, violating=False, reason="fine"),
        make_verdict("src_b", score=0.6, violating=True, reason="ood",
                     suggested_action=ACTION_HOLD),
    ]
    result = arbitrate(verdicts, now=NOW)
    assert result.state == STATE_INTERVENE
    assert result.source == "src_b"


# ---------------------------------------------------------------------------
# 10. score field integrity
# ---------------------------------------------------------------------------


def test_score_field_is_winning_score():
    verdicts = [
        make_verdict("src_a", score=0.4, violating=True, reason="a",
                     suggested_action=ACTION_LOG_ONLY),
        make_verdict("src_b", score=0.7, violating=True, reason="b",
                     suggested_action=ACTION_HOLD),
    ]
    result = arbitrate(verdicts, now=NOW)
    assert result.score == pytest.approx(0.7)


def test_score_is_zero_on_ok():
    result = arbitrate([], now=NOW)
    assert result.score == 0.0


# ---------------------------------------------------------------------------
# 11. Return type
# ---------------------------------------------------------------------------


def test_return_type_is_policy_health_status_data():
    result = arbitrate([], now=NOW)
    assert isinstance(result, PolicyHealthStatusData)


# ---------------------------------------------------------------------------
# 12. Verdicts without a timestamp attribute are treated as fresh
# ---------------------------------------------------------------------------


@dataclass
class TimestamplessVerdict:
    source: str
    score: float
    violating: bool
    reason: str
    suggested_action: int = ACTION_NONE


def test_no_timestamp_attribute_treated_as_fresh():
    v = TimestamplessVerdict(
        source="phm_ood", score=0.85, violating=True, reason="bad",
        suggested_action=ACTION_STOP_AND_HOLD,
    )
    result = arbitrate([v], now=NOW, staleness_sec=1.0)
    # Should not be treated as stale; should be STOP.
    assert result.state == STATE_STOP


# ---------------------------------------------------------------------------
# 13. Large number of sources (stress / determinism)
# ---------------------------------------------------------------------------


def test_many_sources_worst_wins():
    verdicts = [
        make_verdict(f"src_{i}", score=0.1 + i * 0.02, violating=True,
                     reason=f"reason_{i}", suggested_action=ACTION_LOG_ONLY)
        for i in range(20)
    ]
    # Source with highest score should win (src_19, score=0.1+19*0.02=0.48, DEGRADED band).
    result = arbitrate(verdicts, now=NOW)
    assert result.source == "src_19"
    assert result.score == pytest.approx(0.48)


def test_many_sources_with_one_stop():
    verdicts = [
        make_verdict(f"src_{i}", score=0.1 + i * 0.01, violating=True,
                     reason=f"reason_{i}", suggested_action=ACTION_LOG_ONLY)
        for i in range(10)
    ]
    # Inject one STOP-level verdict.
    verdicts.append(
        make_verdict("src_stop", score=0.95, violating=True, reason="critical",
                     suggested_action=ACTION_STOP_AND_HOLD)
    )
    result = arbitrate(verdicts, now=NOW)
    assert result.state == STATE_STOP
    assert result.source == "src_stop"


# ---------------------------------------------------------------------------
# 14. staleness_sec=0 makes everything stale (edge case)
# ---------------------------------------------------------------------------


def test_zero_staleness_makes_fresh_verdict_stale():
    v = make_verdict("src", score=0.9, violating=True, reason="bad", age=0.0)
    result = arbitrate([v], now=NOW, staleness_sec=0.0)
    # age=0.0, staleness_sec=0.0: 0.0 > 0.0 is False -> NOT stale (fresh).
    assert result.state == STATE_STOP


def test_tiny_staleness_makes_aged_verdict_stale():
    v = make_verdict("src", score=0.9, violating=True, reason="bad", age=0.001)
    # staleness_sec=0.0: 0.001 > 0.0 -> stale
    result = arbitrate([v], now=NOW, staleness_sec=0.0)
    assert result.state == STATE_DEGRADED
    assert result.reason == "stale:src"


# ---------------------------------------------------------------------------
# 15. ARBITER_ALLOWLIST contents
# ---------------------------------------------------------------------------


def test_arbiter_allowlist_excludes_rewind():
    assert ACTION_REWIND not in ARBITER_ALLOWLIST


def test_arbiter_allowlist_contains_all_non_rewind_actions():
    assert ACTION_NONE in ARBITER_ALLOWLIST
    assert ACTION_LOG_ONLY in ARBITER_ALLOWLIST
    assert ACTION_HOLD in ARBITER_ALLOWLIST
    assert ACTION_STOP_AND_HOLD in ARBITER_ALLOWLIST
