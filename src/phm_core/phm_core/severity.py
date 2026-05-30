"""Severity mapping: raw detector signal -> (score, state, suggested_action).

This is the single place that defines the OK / DEGRADED / INTERVENE / STOP
thresholds for the whole monitor. Both the per-detector verdicts and the
arbiter read these bands so the meaning of a score is consistent everywhere.

State enum and action enum match ``phm_msgs/PolicyHealthStatus.msg`` exactly:

- state:  0 OK, 1 DEGRADED, 2 INTERVENE, 3 STOP
- action: 0 NONE, 1 LOG_ONLY, 2 HOLD, 3 STOP_AND_HOLD, 4 REWIND

The default banding on a normalized score in [0, 1] (0 healthy, 1 worst):

- score <  0.25  -> OK,        action NONE
- 0.25 <= score < 0.50 -> DEGRADED,  action LOG_ONLY
- 0.50 <= score < 0.80 -> INTERVENE, action HOLD
- score >= 0.80  -> STOP,      action STOP_AND_HOLD

ASSUMPTION: the spec ("severity.py: maps a detector's raw signal to a
normalized score in [0,1] and a state/suggested_action ... Single place that
defines the OK/DEGRADED/INTERVENE/STOP thresholds") fixes the four states and
the action enum but does not pin the numeric cut points or the state-to-action
map. These DEGRADED=0.25 / INTERVENE=0.50 / STOP=0.80 bands and the
NONE/LOG_ONLY/HOLD/STOP_AND_HOLD action pairing are a reasonable default chosen
here; they are centralized so a later spec revision changes one place. REWIND is
not auto-selected by score banding (it is a recovery-policy choice the arbiter
or recovery node makes), so it is exposed as a constant but never returned by
:func:`classify`.
"""

from __future__ import annotations

from dataclasses import dataclass

# State enum, mirrors PolicyHealthStatus.msg.
STATE_OK = 0
STATE_DEGRADED = 1
STATE_INTERVENE = 2
STATE_STOP = 3

# Suggested-action enum, mirrors PolicyHealthStatus.msg.
ACTION_NONE = 0
ACTION_LOG_ONLY = 1
ACTION_HOLD = 2
ACTION_STOP_AND_HOLD = 3
ACTION_REWIND = 4

# Score cut points. A score >= the band's lower edge selects that band.
DEGRADED_THRESHOLD = 0.25
INTERVENE_THRESHOLD = 0.50
STOP_THRESHOLD = 0.80


@dataclass(frozen=True)
class Severity:
    """The classified severity of a normalized score."""

    score: float
    state: int
    suggested_action: int


def normalize(raw: float, healthy: float, worst: float) -> float:
    """Map a raw signal onto a normalized severity score in [0, 1].

    Linearly maps ``raw`` so that ``healthy`` -> 0.0 and ``worst`` -> 1.0, then
    clamps to [0, 1]. Works in either direction: pass ``healthy > worst`` when a
    *low* raw value is the unhealthy one (the rolling-spread case, where the
    spread collapses toward zero out of distribution).

    Args:
        raw: the detector's raw signal value.
        healthy: the raw value that corresponds to perfectly healthy (score 0).
        worst: the raw value that corresponds to worst case (score 1).

    Returns:
        A float in [0, 1].

    Raises:
        ValueError: if ``healthy == worst`` (degenerate, no scale).
    """
    if healthy == worst:
        raise ValueError("healthy and worst must differ to define a scale")
    frac = (raw - healthy) / (worst - healthy)
    if frac < 0.0:
        return 0.0
    if frac > 1.0:
        return 1.0
    return float(frac)


def classify(score: float) -> Severity:
    """Classify a normalized severity score into a state and suggested action.

    Args:
        score: normalized severity in [0, 1] (0 healthy, 1 worst). Values
            outside [0, 1] are clamped before classification.

    Returns:
        A :class:`Severity` with the clamped score, the state, and the
        suggested action for that band.
    """
    s = 0.0 if score < 0.0 else 1.0 if score > 1.0 else float(score)

    if s >= STOP_THRESHOLD:
        return Severity(s, STATE_STOP, ACTION_STOP_AND_HOLD)
    if s >= INTERVENE_THRESHOLD:
        return Severity(s, STATE_INTERVENE, ACTION_HOLD)
    if s >= DEGRADED_THRESHOLD:
        return Severity(s, STATE_DEGRADED, ACTION_LOG_ONLY)
    return Severity(s, STATE_OK, ACTION_NONE)
