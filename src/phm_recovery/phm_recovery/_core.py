"""phm_recovery core: SafetyEnvelope, HealthToActionMapper, RewindHook.

Pure Python, no ROS dependency. The rclpy node in recovery_node.py is a thin
wrapper over these classes so they can be unit-tested without a ROS graph.

SafetyEnvelope is ported (not imported) from:
  HELIX helix_recovery/recovery_node.py:36
  (SafetyEnvelope class, per-action cooldown, RESUME exempt, allowlist check).
The port adds support for the PHM action set (HOLD, STOP_AND_HOLD, REWIND,
LOG_ONLY, NONE) and decouples the allowlist from the HELIX message types.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PHM action constants, mirror PolicyHealthStatus.msg / phm_core.detector.
# ---------------------------------------------------------------------------
ACTION_NONE: int = 0
ACTION_LOG_ONLY: int = 1
ACTION_HOLD: int = 2
ACTION_STOP_AND_HOLD: int = 3
ACTION_REWIND: int = 4

_ACTION_NAMES: dict[int, str] = {
    ACTION_NONE: "NONE",
    ACTION_LOG_ONLY: "LOG_ONLY",
    ACTION_HOLD: "HOLD",
    ACTION_STOP_AND_HOLD: "STOP_AND_HOLD",
    ACTION_REWIND: "REWIND",
}

# PHM health state constants, mirror PolicyHealthStatus.msg.
STATE_OK: int = 0
STATE_DEGRADED: int = 1
STATE_INTERVENE: int = 2
STATE_STOP: int = 3

# The set of actions the SafetyEnvelope will accept for actuation.
# LOG_ONLY and NONE do not actuate (envelope reports ACCEPTED but publish=False).
_DEFAULT_ALLOWLIST: frozenset[int] = frozenset(
    {ACTION_NONE, ACTION_LOG_ONLY, ACTION_HOLD, ACTION_STOP_AND_HOLD, ACTION_REWIND}
)
_ACTUATING_ACTIONS: frozenset[int] = frozenset({ACTION_HOLD, ACTION_STOP_AND_HOLD, ACTION_REWIND})

# RESUME is the sentinel action that clears a hold. It is represented as a
# string internally (no msg constant) because it is an envelope-internal
# concept only: no detector emits RESUME, the recovery node synthesizes it when
# the health state returns to OK.
_ACTION_RESUME_SENTINEL: str = "__RESUME__"


@dataclass
class EnvelopeResult:
    """Result of SafetyEnvelope.evaluate.

    Ported from HELIX helix_recovery/recovery_node.py:30-33.
    """

    # 'ACCEPTED' | 'SUPPRESSED_DISABLED' | 'SUPPRESSED_ALLOWLIST' | 'SUPPRESSED_COOLDOWN'
    status: str
    publish: bool  # whether the caller should actuate (publish a command)
    reason: str


class SafetyEnvelope:
    """Per-action cooldown gate with allowlist and RESUME exemption.

    Ported from HELIX helix_recovery/recovery_node.py:36-60.

    The key invariant is: RESUME (clearing a hold) is NEVER rate-limited by
    the cooldown of the stop it is clearing. This matches the HELIX comment:
    "Cooldown exists only to damp STOP_AND_HOLD flapping." If RESUME were also
    gated, a safety stop could suppress its own release.

    Cooldown is keyed by (action, fault_key): two different fault sources can
    each fire at full rate independently, but the same source cannot re-fire
    the same action until the cooldown expires.
    """

    def __init__(
        self,
        enabled: bool,
        cooldown_seconds: float,
        allowlist: frozenset[int] = _DEFAULT_ALLOWLIST,
    ) -> None:
        self.enabled = enabled
        self.cooldown_seconds = cooldown_seconds
        self._allowlist = allowlist
        # Key: (action_int, fault_key_str) -> last fired wall-clock time.
        self._last_action_time: dict[tuple[int, str], float] = {}

    def evaluate(self, action: int, fault_key: str, now: float) -> EnvelopeResult:
        """Gate a requested action through the safety envelope.

        Ported from HELIX helix_recovery/recovery_node.py:44-60.

        Args:
            action: one of the ACTION_* constants.
            fault_key: a string key identifying the fault source (e.g. the
                health state reason or the publishing source name). Used to
                scope the cooldown so independent faults do not block each other.
            now: current time in seconds (wall clock or sim clock).

        Returns:
            An :class:`EnvelopeResult` describing whether the action was
            accepted and whether the caller should publish/actuate.
        """
        if not self.enabled:
            return EnvelopeResult(
                "SUPPRESSED_DISABLED", False, "recovery.enabled is false"
            )
        if action not in self._allowlist:
            name = _ACTION_NAMES.get(action, str(action))
            return EnvelopeResult(
                "SUPPRESSED_ALLOWLIST", False, f"action {name} not in allowlist"
            )

        # RESUME is exempt from cooldown (HELIX recovery_node.py:53-57).
        # For PHM, RESUME is represented as the ACTION_NONE action with a
        # clearing call. The caller uses evaluate_resume() for explicit resumes;
        # this branch is the guard so a raw ACTION_NONE also bypasses cooldown.
        # (HOLD and STOP_AND_HOLD are the ones that damp flapping.)
        if action in (ACTION_HOLD, ACTION_STOP_AND_HOLD):
            key = (action, fault_key)
            last = self._last_action_time.get(key)
            if last is not None and (now - last) < self.cooldown_seconds:
                elapsed = now - last
                return EnvelopeResult(
                    "SUPPRESSED_COOLDOWN",
                    False,
                    f"cooldown active for {fault_key} "
                    f"({elapsed:.2f}s < {self.cooldown_seconds:.2f}s)",
                )
            self._last_action_time[key] = now

        publish = action in _ACTUATING_ACTIONS
        name = _ACTION_NAMES.get(action, str(action))
        return EnvelopeResult("ACCEPTED", publish, f"action {name} accepted")

    def evaluate_resume(self, fault_key: str, now: float) -> EnvelopeResult:
        """Evaluate a RESUME request (clear a hold).

        RESUME is exempt from cooldown. This mirrors HELIX recovery_node.py:53:
        'RESUME ends a STOP_AND_HOLD. It must never be rate-limited.'

        The fault_key is accepted for logging symmetry but not used for gating.
        """
        if not self.enabled:
            return EnvelopeResult(
                "SUPPRESSED_DISABLED", False, "recovery.enabled is false"
            )
        return EnvelopeResult("ACCEPTED", True, f"RESUME accepted for {fault_key} (cooldown exempt)")  # noqa: E501


# ---------------------------------------------------------------------------
# Health-state to action mapping (pure, testable)
# ---------------------------------------------------------------------------

@dataclass
class HealthActionDecision:
    """Output of HealthToActionMapper.map."""

    action: int         # ACTION_* constant
    hold_active: bool   # True when the node should publish zero-velocity
    reason: str


class HealthToActionMapper:
    """Maps a PolicyHealthStatus state+suggested_action to an actuation decision.

    This is the pure-Python logic that the recovery node delegates to. It
    tracks whether a hold is currently active (so the zero-vel timer keeps
    publishing) and decides whether to start, continue, or clear a hold.

    Rules:
    - STATE_STOP -> force STOP_AND_HOLD regardless of suggested_action.
    - STATE_INTERVENE with suggested_action >= HOLD -> activate HOLD.
    - suggested_action == ACTION_REWIND -> activate REWIND (hold + log).
    - STATE_OK or STATE_DEGRADED (with low suggested_action) -> clear hold.
    - LOG_ONLY events pass through without actuating.
    """

    def __init__(self) -> None:
        self._hold_active: bool = False

    @property
    def hold_active(self) -> bool:
        return self._hold_active

    def map(  # noqa: PLR0911
        self, state: int, suggested_action: int, source: str, reason: str
    ) -> HealthActionDecision:
        """Compute the actuation decision for one health status update.

        Args:
            state: PolicyHealthStatus.state (STATE_* constant).
            suggested_action: PolicyHealthStatus.suggested_action (ACTION_*).
            source: the health status source field (for logging).
            reason: the health status reason field (for logging).

        Returns:
            A :class:`HealthActionDecision` describing the action to take and
            whether the zero-velocity hold should be active.
        """
        if state == STATE_STOP:
            self._hold_active = True
            return HealthActionDecision(
                ACTION_STOP_AND_HOLD,
                True,
                f"STATE_STOP from {source}: {reason}",
            )

        if state == STATE_INTERVENE:
            if suggested_action == ACTION_REWIND:
                self._hold_active = True
                return HealthActionDecision(
                    ACTION_REWIND,
                    True,
                    f"ACTION_REWIND from {source}: {reason}",
                )
            if suggested_action >= ACTION_HOLD:
                self._hold_active = True
                return HealthActionDecision(
                    ACTION_HOLD,
                    True,
                    f"STATE_INTERVENE/HOLD from {source}: {reason}",
                )
            # INTERVENE with LOG_ONLY or NONE suggested: log but no hold.
            return HealthActionDecision(
                ACTION_LOG_ONLY,
                self._hold_active,  # preserve existing hold state
                f"STATE_INTERVENE/LOG from {source}: {reason}",
            )

        if state in (STATE_OK, STATE_DEGRADED):
            if self._hold_active:
                # Clearing hold: state has recovered.
                self._hold_active = False
                return HealthActionDecision(
                    ACTION_NONE,
                    False,
                    f"hold cleared: state={state} from {source}",
                )
            action = suggested_action if suggested_action in (ACTION_LOG_ONLY,) else ACTION_NONE
            return HealthActionDecision(action, False, f"state {state} from {source}: no actuation")

        # Unknown state: treat conservatively as STOP.
        self._hold_active = True
        return HealthActionDecision(
            ACTION_STOP_AND_HOLD,
            True,
            f"unknown state {state} from {source}: conservative STOP_AND_HOLD",
        )

    def clear_hold(self) -> None:
        """Programmatically clear the hold (for RESUME from outside)."""
        self._hold_active = False


# ---------------------------------------------------------------------------
# RewindHook: pluggable callback for ACTION_REWIND
# ---------------------------------------------------------------------------

# Type alias for the rewind callback.
RewindCallbackType = Callable[[], None]


@dataclass
class RewindHook:
    """Pluggable rewind hook called when the recovery node sees ACTION_REWIND.

    v0 behavior: log the rewind request and activate a hold. A host stack
    overrides the callback to implement return-to-last-safe-waypoint. The
    callback receives no arguments; the host stack is responsible for reading
    its own waypoint state.

    Usage:
        hook = RewindHook()
        hook.register(my_callback)   # optional override
        hook.trigger()               # called by the recovery node
    """

    _callback: RewindCallbackType | None = field(default=None, repr=False)

    def register(self, callback: RewindCallbackType) -> None:
        """Register an override callback for the rewind action."""
        self._callback = callback

    def trigger(self) -> None:
        """Trigger the rewind action.

        If no callback is registered (v0 default), logs the request and holds.
        If a callback is registered, calls it. The recovery node activates a
        hold independently of whether the callback does anything.
        """
        if self._callback is None:
            logger.warning(
                "ACTION_REWIND triggered but no rewind callback registered; "
                "holding and logging. Register a callback via RewindHook.register() "
                "to implement return-to-last-safe-waypoint."
            )
        else:
            self._callback()
