"""Integration test for the arbiter -> recovery seam (LOCKED decisions 4 + 5).

This is the cross-module test the adversarial review flagged as MISSING: it
threads the arbiter's ``arbitrate()`` output directly into the recovery
``HealthToActionMapper`` and asserts the composed behavior is fail-safe.

The key invariant under test (LOCKED decision 4): NO state >= STATE_INTERVENE
may ever produce ``hold_active=False`` out of the mapper, for ANY suggested
action the arbiter can emit. The original bug was an INTERVENE+REWIND request
that the arbiter clamped to NONE and the recovery mapper then turned into a
LOG_ONLY with the hold released, a silent no-op on a real safety action.

No rclpy import: both modules are pure-Python ``_core`` logic. The phm_arbiter
package root is added to ``sys.path`` here (the recovery conftest only injects
phm_recovery) so this single cross-module test can import the arbiter fusion.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# Make phm_arbiter importable for this cross-module integration test. The
# recovery conftest only adds phm_recovery; the arbiter package lives alongside.
_ARBITER_ROOT = Path(__file__).resolve().parents[2] / "phm_arbiter"
if str(_ARBITER_ROOT) not in sys.path:
    sys.path.insert(0, str(_ARBITER_ROOT))

import pytest  # noqa: E402
from phm_arbiter._core import (  # noqa: E402
    ACTION_HOLD as A_HOLD,
)
from phm_arbiter._core import (  # noqa: E402
    ACTION_LOG_ONLY as A_LOG,
)
from phm_arbiter._core import (  # noqa: E402
    ACTION_NONE as A_NONE,
)
from phm_arbiter._core import (  # noqa: E402
    ACTION_REWIND as A_REWIND,
)
from phm_arbiter._core import (  # noqa: E402
    ACTION_STOP_AND_HOLD as A_STOP,
)
from phm_arbiter._core import (  # noqa: E402
    STATE_INTERVENE as ST_INTERVENE,
)
from phm_arbiter._core import (  # noqa: E402
    STATE_STOP as ST_STOP,
)
from phm_arbiter._core import arbitrate  # noqa: E402

from phm_recovery._core import HealthToActionMapper  # noqa: E402

NOW = 100.0

# Score bands that drive each state under the arbiter's _score_to_state:
#   >= 0.80 -> STOP, >= 0.50 -> INTERVENE, >= 0.25 -> DEGRADED, else OK (violating
#   floors at DEGRADED). We sweep one representative score per band.
_SCORES = [0.0, 0.10, 0.30, 0.65, 0.95]
_ACTIONS = [A_NONE, A_LOG, A_HOLD, A_STOP, A_REWIND]


@dataclass
class FakeVerdict:
    source: str
    score: float
    violating: bool
    reason: str
    suggested_action: int = 0
    timestamp: float = NOW


@pytest.mark.parametrize("score", _SCORES)
@pytest.mark.parametrize("action", _ACTIONS)
@pytest.mark.parametrize("violating", [True, False])
def test_no_intervene_or_stop_ever_releases_hold(score, action, violating):
    """Thread arbitrate() -> HealthToActionMapper for every (score, action,
    violating) combination. Any arbitrated state >= INTERVENE must actuate a
    hold (hold_active=True). This is the seam invariant from decision 4.
    """
    v = FakeVerdict(
        source="phm_ood",
        score=score,
        violating=violating,
        reason="seam test",
        suggested_action=action,
        timestamp=NOW,
    )
    health = arbitrate([v], now=NOW, staleness_sec=1.0)

    mapper = HealthToActionMapper()
    decision = mapper.map(
        state=health.state,
        suggested_action=health.suggested_action,
        source=health.source,
        reason=health.reason,
    )

    if health.state >= ST_INTERVENE:
        assert decision.hold_active is True, (
            f"state={health.state} (>=INTERVENE) with arbiter action="
            f"{health.suggested_action} released the hold: {decision}"
        )


def test_intervene_plus_rewind_seam_holds_and_rewinds():
    """The exact original bug: INTERVENE-band verdict requesting REWIND. The
    arbiter now passes REWIND through (decision 4), and the recovery mapper
    must return ACTION_REWIND with hold_active=True (NOT a LOG_ONLY no-op).
    """
    v = FakeVerdict(
        source="phm_ood",
        score=0.65,  # INTERVENE band
        violating=True,
        reason="rewind requested",
        suggested_action=A_REWIND,
        timestamp=NOW,
    )
    health = arbitrate([v], now=NOW, staleness_sec=1.0)
    assert health.state == ST_INTERVENE
    assert health.suggested_action == A_REWIND  # passed through, not clamped

    mapper = HealthToActionMapper()
    decision = mapper.map(
        state=health.state,
        suggested_action=health.suggested_action,
        source=health.source,
        reason=health.reason,
    )
    assert decision.action == A_REWIND
    assert decision.hold_active is True


def test_intervene_with_action_none_still_holds():
    """A common composition: arbiter emits INTERVENE with suggested_action NONE
    (e.g. a low-severity OOD detector). The recovery layer must still HOLD.
    """
    v = FakeVerdict(
        source="phm_ood",
        score=0.55,
        violating=True,
        reason="ood",
        suggested_action=A_NONE,
        timestamp=NOW,
    )
    health = arbitrate([v], now=NOW, staleness_sec=1.0)
    assert health.state == ST_INTERVENE

    mapper = HealthToActionMapper()
    decision = mapper.map(
        state=health.state,
        suggested_action=health.suggested_action,
        source=health.source,
        reason=health.reason,
    )
    assert decision.hold_active is True


def test_stop_seam_forces_stop_and_hold():
    """A STOP-level verdict threads through to STOP_AND_HOLD with the hold on."""
    v = FakeVerdict(
        source="threshold:cpu",
        score=0.95,
        violating=True,
        reason="cpu overload",
        suggested_action=A_STOP,
        timestamp=NOW,
    )
    health = arbitrate([v], now=NOW, staleness_sec=1.0)
    assert health.state == ST_STOP

    mapper = HealthToActionMapper()
    decision = mapper.map(
        state=health.state,
        suggested_action=health.suggested_action,
        source=health.source,
        reason=health.reason,
    )
    assert decision.action == A_STOP
    assert decision.hold_active is True
