"""Safety-logic regression tests for phm_arbiter._core.arbitrate().

These tests target the adversarial-review (skeptic-core) findings in
docs/REVIEW_PUNCHLIST.md. They reproduce the exact verified inputs from that
punch list and assert the LOCKED fail-safe behavior:

1. STALENESS never de-escalates a violating verdict (stale + violating +
   score>=0.80 stays STOP).
2. VIOLATING FLOOR: a non-stale violating verdict is never STATE_OK; it floors
   at STATE_DEGRADED (violating=True + score=0.1 -> state >= DEGRADED).
6. NaN / non-finite guard at the trust boundary: score=NaN + violating does NOT
   force STOP, the published score is finite, and arbitration is deterministic.

No rclpy imports: all tests exercise the pure-Python function directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from phm_arbiter._core import (
    ACTION_STOP_AND_HOLD,
    STATE_DEGRADED,
    STATE_STOP,
    arbitrate,
)


@dataclass
class FakeVerdict:
    source: str
    score: float
    violating: bool
    reason: str
    suggested_action: int = 0
    timestamp: float = 0.0


NOW = 100.0


# ---------------------------------------------------------------------------
# LOCKED decision 1: staleness never de-escalates a violating verdict.
# ---------------------------------------------------------------------------


def test_stale_violating_stop_stays_stop():
    """Verified punch-list input: a 1.1s-old score=0.95 STOP verdict.

    A stale-but-violating CRITICAL verdict must NOT be downgraded to DEGRADED.
    Staleness on a critical channel escalates (or at minimum never drops below
    the live severity), never de-escalates.
    """
    v = FakeVerdict(
        source="threshold:cpu",
        score=0.95,
        violating=True,
        reason="cpu overload",
        suggested_action=ACTION_STOP_AND_HOLD,
        timestamp=NOW - 1.1,  # age 1.1s > staleness 1.0s -> stale
    )
    result = arbitrate([v], now=NOW, staleness_sec=1.0)
    assert result.state == STATE_STOP, (
        f"stale+violating+score=0.95 must stay STOP, got state={result.state}"
    )
    assert result.score >= 0.80, f"stale STOP score must stay high, got {result.score}"
    assert "stale:threshold:cpu" in result.reason


def test_stale_violating_intervene_stays_at_least_intervene():
    """Stale + violating in the INTERVENE band must not drop below INTERVENE."""
    from phm_arbiter._core import STATE_INTERVENE

    v = FakeVerdict(
        source="phm_ood",
        score=0.65,
        violating=True,
        reason="ood",
        timestamp=NOW - 2.0,
    )
    result = arbitrate([v], now=NOW, staleness_sec=1.0)
    assert result.state >= STATE_INTERVENE
    assert result.reason == "stale:phm_ood"


def test_stale_non_violating_becomes_degraded():
    """Stale + non-violating still floors at DEGRADED (never silently dropped)."""
    v = FakeVerdict(
        source="freq:/scan",
        score=0.05,
        violating=False,
        reason="ok",
        timestamp=NOW - 2.0,
    )
    result = arbitrate([v], now=NOW, staleness_sec=1.0)
    assert result.state == STATE_DEGRADED
    assert result.reason == "stale:freq:/scan"


# ---------------------------------------------------------------------------
# LOCKED decision 2: violating floor (a violating verdict is never STATE_OK).
# ---------------------------------------------------------------------------


def test_violating_low_score_floors_at_degraded():
    """Verified punch-list input: violating=True with score=0.10.

    _score_to_state(0.10) is STATE_OK, but a verdict that explicitly asserts
    'I am violating' must never resolve to STATE_OK. It floors at DEGRADED.
    """
    v = FakeVerdict(
        source="phm_ood",
        score=0.10,
        violating=True,
        reason="sub-floor ood",
        timestamp=NOW,
    )
    result = arbitrate([v], now=NOW, staleness_sec=1.0)
    assert result.state >= STATE_DEGRADED, (
        f"violating=True+score=0.1 must be >= DEGRADED, got state={result.state}"
    )


def test_violating_zero_score_floors_at_degraded():
    v = FakeVerdict(
        source="src",
        score=0.0,
        violating=True,
        reason="violating but zero score",
        timestamp=NOW,
    )
    result = arbitrate([v], now=NOW, staleness_sec=1.0)
    assert result.state >= STATE_DEGRADED


# ---------------------------------------------------------------------------
# LOCKED decision 6: NaN / non-finite guard at the trust boundary.
# ---------------------------------------------------------------------------


def test_nan_violating_does_not_force_stop():
    """Verified punch-list input: score=NaN + violating=True.

    A single poisoned detector with a NaN score must not pin the monitor to
    STOP-with-NaN nor produce non-deterministic arbitration. The non-finite
    score is treated as DEGRADED with a finite sentinel.
    """
    v = FakeVerdict(
        source="phm_ood",
        score=float("nan"),
        violating=True,
        reason="poisoned",
        timestamp=NOW,
    )
    result = arbitrate([v], now=NOW, staleness_sec=1.0)
    assert result.state != STATE_STOP, "NaN must not force STOP"
    assert math.isfinite(result.score), f"published score must be finite, got {result.score}"
    assert "bad-score:phm_ood" in result.reason


def test_nan_arbitration_is_deterministic():
    """Repeated arbitration on a NaN verdict yields identical, finite output."""
    v = FakeVerdict(
        source="phm_ood",
        score=float("nan"),
        violating=True,
        reason="poisoned",
        timestamp=NOW,
    )
    r1 = arbitrate([v], now=NOW, staleness_sec=1.0)
    r2 = arbitrate([v], now=NOW, staleness_sec=1.0)
    assert (r1.state, r1.score, r1.source) == (r2.state, r2.score, r2.source)
    assert math.isfinite(r1.score)


def test_nan_does_not_win_over_real_stop():
    """A real STOP from another detector must still win over a NaN-degraded one."""
    nan_v = FakeVerdict(
        source="phm_ood", score=float("nan"), violating=True, reason="poisoned",
        timestamp=NOW,
    )
    real_stop = FakeVerdict(
        source="threshold:cpu", score=0.92, violating=True, reason="cpu",
        suggested_action=ACTION_STOP_AND_HOLD, timestamp=NOW,
    )
    result = arbitrate([nan_v, real_stop], now=NOW, staleness_sec=1.0)
    assert result.state == STATE_STOP
    assert result.source == "threshold:cpu"
    assert math.isfinite(result.score)


def test_inf_violating_does_not_force_stop():
    v = FakeVerdict(
        source="src", score=float("inf"), violating=True, reason="inf",
        timestamp=NOW,
    )
    result = arbitrate([v], now=NOW, staleness_sec=1.0)
    assert result.state != STATE_STOP
    assert math.isfinite(result.score)
    assert "bad-score:src" in result.reason
