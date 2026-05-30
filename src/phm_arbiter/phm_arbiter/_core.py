"""Pure-Python arbitration logic for the Policy Health Monitor.

This module contains NO rclpy imports. All ROS 2 coupling lives in
``arbiter_node.py``. Tests import only this file so they run without a ROS
graph.

Spec reference: docs/superpowers/specs/2026-05-29-policy-health-monitor-design.md
section 3.3 (phm_arbiter fusion rule).

Worst-wins rule:
1. Collect the latest DetectorVerdictData from every registered source within a
   staleness window (staleness_sec, default 1.0 s).
2. State = max severity state across non-stale, violating verdicts (OK if none).
3. score = max score; source/reason from the verdict that set the winning state.
4. suggested_action = the action of the winning verdict, clamped to ARBITER_ALLOWLIST.
5. A stale verdict (timestamp older than staleness_sec) is promoted to DEGRADED
   with reason "stale:<source>", never silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

# State enum constants mirror phm_msgs/PolicyHealthStatus.msg exactly.
STATE_OK = 0
STATE_DEGRADED = 1
STATE_INTERVENE = 2
STATE_STOP = 3

# Suggested-action enum constants mirror phm_msgs/PolicyHealthStatus.msg exactly.
ACTION_NONE = 0
ACTION_LOG_ONLY = 1
ACTION_HOLD = 2
ACTION_STOP_AND_HOLD = 3
ACTION_REWIND = 4

# The arbiter clamps suggested_action to this allowlist.
# REWIND (4) is excluded: it is a recovery-policy decision, not a fusion output.
ARBITER_ALLOWLIST: frozenset[int] = frozenset(
    {ACTION_NONE, ACTION_LOG_ONLY, ACTION_HOLD, ACTION_STOP_AND_HOLD}
)

# A stale verdict is synthesized as DEGRADED, score 0.5, LOG_ONLY action.
# Score 0.5 is at the boundary of INTERVENE per severity.py but the state is
# hard-coded to DEGRADED here because stale-ness is a data-quality issue, not a
# policy-quality issue. The arbiter still participates in worst-wins, so if
# another source is worse, that wins.
_STALE_SCORE = 0.25  # minimum score that places a verdict in DEGRADED band
_STALE_ACTION = ACTION_LOG_ONLY


@dataclass
class PolicyHealthStatusData:
    """Pure-Python mirror of phm_msgs/PolicyHealthStatus.msg (minus the header).

    Field names match the .msg exactly so the rclpy node can assign them
    by name without a translation layer.

    - state: one of STATE_OK / STATE_DEGRADED / STATE_INTERVENE / STATE_STOP
    - score: worst normalized severity score in [0, 1] across sources
    - reason: human-readable explanation of what drove this state
    - source: which detector drove the state
    - suggested_action: one of ACTION_* constants, clamped to ARBITER_ALLOWLIST
    """

    state: int
    score: float
    reason: str
    source: str
    suggested_action: int


@dataclass
class _Candidate:
    """Internal working struct used during arbitration."""

    state: int
    score: float
    reason: str
    source: str
    suggested_action: int


def arbitrate(
    verdicts: list[object],  # list of DetectorVerdictData (typed as object to stay pure)
    now: float,
    staleness_sec: float = 1.0,
) -> PolicyHealthStatusData:
    """Fuse a list of DetectorVerdictData into a single PolicyHealthStatusData.

    Each verdict must expose: source (str), score (float), violating (bool),
    reason (str), suggested_action (int), and timestamp (float, seconds since
    epoch or ROS time in seconds). If a verdict has no ``timestamp`` attribute
    it is treated as fresh.

    Worst-wins fusion:
    - Stale verdict (age > staleness_sec): synthesized DEGRADED, reason "stale:<source>".
    - Non-stale, non-violating verdict: contributes nothing to worst-wins.
    - Non-stale, violating verdict: competes by state (highest int wins), then
      by score (highest wins) for tie-breaking.

    When no candidate is violating or stale, returns OK with score 0.0.

    Args:
        verdicts: iterable of DetectorVerdictData or any object with the above fields.
        now: current time in seconds (float). Must match the verdict timestamp units.
        staleness_sec: age threshold in seconds; verdicts older than this are stale.

    Returns:
        A PolicyHealthStatusData with the arbitrated result.
    """
    candidates: list[_Candidate] = []

    for v in verdicts:
        # Determine age. Verdicts without a timestamp attribute are treated as fresh.
        ts = getattr(v, "timestamp", now)
        age = now - ts

        if age > staleness_sec:
            # Stale: synthesize a DEGRADED candidate, never drop silently.
            candidates.append(
                _Candidate(
                    state=STATE_DEGRADED,
                    score=_STALE_SCORE,
                    reason=f"stale:{v.source}",
                    source=v.source,
                    suggested_action=_STALE_ACTION,
                )
            )
        elif v.violating:
            # Non-stale and actively violating: enter worst-wins competition.
            action = v.suggested_action
            if action not in ARBITER_ALLOWLIST:
                action = ACTION_NONE
            # Map the verdict score to a state using the same bands as severity.py:
            #   score <  0.25 -> OK        (should never reach here if violating is correct)
            #   0.25 <= score < 0.50 -> DEGRADED
            #   0.50 <= score < 0.80 -> INTERVENE
            #   score >= 0.80 -> STOP
            # We do NOT call severity.classify() here to keep _core.py self-contained.
            state = _score_to_state(v.score)
            candidates.append(
                _Candidate(
                    state=state,
                    score=float(v.score),
                    reason=v.reason,
                    source=v.source,
                    suggested_action=action,
                )
            )
        # else: non-stale, non-violating -> contributes nothing, OK for this source.

    if not candidates:
        return PolicyHealthStatusData(
            state=STATE_OK,
            score=0.0,
            reason="all detectors nominal",
            source="",
            suggested_action=ACTION_NONE,
        )

    # Worst-wins: primary key = state (higher int = worse), secondary = score.
    winner = max(candidates, key=lambda c: (c.state, c.score))

    return PolicyHealthStatusData(
        state=winner.state,
        score=winner.score,
        reason=winner.reason,
        source=winner.source,
        suggested_action=winner.suggested_action,
    )


def _score_to_state(score: float) -> int:
    """Map a normalized score to a STATE_* constant using the spec banding.

    Bands match phm_core/severity.py classify() exactly:
    - score >= 0.80 -> STATE_STOP
    - score >= 0.50 -> STATE_INTERVENE
    - score >= 0.25 -> STATE_DEGRADED
    - else          -> STATE_OK

    Defined here (instead of importing from phm_core) so _core.py has zero
    external dependencies and tests run with PYTHONPATH pointing only at this
    package.
    """
    s = max(0.0, min(1.0, float(score)))
    if s >= 0.80:
        return STATE_STOP
    if s >= 0.50:
        return STATE_INTERVENE
    if s >= 0.25:
        return STATE_DEGRADED
    return STATE_OK
