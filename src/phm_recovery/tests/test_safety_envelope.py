"""Tests for SafetyEnvelope.

Verifies:
- Cooldown blocks re-fire of HOLD/STOP_AND_HOLD for the same fault_key.
- RESUME (evaluate_resume) is NEVER blocked by cooldown.
- Allowlist rejects unknown action integers.
- Disabled envelope suppresses all actions.
- Independent fault_keys have independent cooldowns.
- Cooldown expires after enough time.
- LOG_ONLY and NONE are accepted but do not actuate (publish=False for NONE,
  publish=False for LOG_ONLY since they are not in _ACTUATING_ACTIONS).

These tests import only phm_recovery._core (no rclpy).
"""

from __future__ import annotations

from phm_recovery._core import (
    ACTION_HOLD,
    ACTION_LOG_ONLY,
    ACTION_NONE,
    ACTION_REWIND,
    ACTION_STOP_AND_HOLD,
    SafetyEnvelope,
)

# A sentinel integer not in any allowlist.
_UNKNOWN_ACTION = 99


def _envelope(enabled: bool = True, cooldown: float = 5.0) -> SafetyEnvelope:
    return SafetyEnvelope(enabled=enabled, cooldown_seconds=cooldown)


class TestAllowlist:
    def test_unknown_action_suppressed(self):
        env = _envelope()
        result = env.evaluate(_UNKNOWN_ACTION, "src", now=0.0)
        assert result.status == "SUPPRESSED_ALLOWLIST"
        assert result.publish is False

    def test_known_actions_accepted(self):
        env = _envelope()
        all_actions = (ACTION_NONE, ACTION_LOG_ONLY, ACTION_HOLD, ACTION_STOP_AND_HOLD,
                       ACTION_REWIND)
        for action in all_actions:
            result = env.evaluate(action, "src", now=0.0)
            msg = f"action {action} should be ACCEPTED, got {result.status}"
            assert result.status == "ACCEPTED", msg

    def test_hold_actuates(self):
        env = _envelope()
        result = env.evaluate(ACTION_HOLD, "src", now=0.0)
        assert result.publish is True

    def test_stop_and_hold_actuates(self):
        env = _envelope()
        result = env.evaluate(ACTION_STOP_AND_HOLD, "src", now=0.0)
        assert result.publish is True

    def test_rewind_actuates(self):
        env = _envelope()
        result = env.evaluate(ACTION_REWIND, "src", now=0.0)
        assert result.publish is True

    def test_none_does_not_actuate(self):
        env = _envelope()
        result = env.evaluate(ACTION_NONE, "src", now=0.0)
        assert result.status == "ACCEPTED"
        assert result.publish is False

    def test_log_only_does_not_actuate(self):
        env = _envelope()
        result = env.evaluate(ACTION_LOG_ONLY, "src", now=0.0)
        assert result.status == "ACCEPTED"
        assert result.publish is False


class TestCooldown:
    def test_cooldown_blocks_re_fire(self):
        env = _envelope(cooldown=5.0)
        first = env.evaluate(ACTION_STOP_AND_HOLD, "fault_a", now=0.0)
        assert first.status == "ACCEPTED"
        second = env.evaluate(ACTION_STOP_AND_HOLD, "fault_a", now=2.0)  # within cooldown
        assert second.status == "SUPPRESSED_COOLDOWN"
        assert second.publish is False

    def test_cooldown_expires(self):
        env = _envelope(cooldown=5.0)
        env.evaluate(ACTION_STOP_AND_HOLD, "fault_a", now=0.0)
        result = env.evaluate(ACTION_STOP_AND_HOLD, "fault_a", now=5.01)  # just past cooldown
        assert result.status == "ACCEPTED"
        assert result.publish is True

    def test_cooldown_exactly_at_boundary_is_accepted(self):
        # The cooldown check is strict: (now - last) < cooldown_seconds.
        # At exactly cooldown_seconds elapsed the window has expired -> ACCEPTED.
        # This matches HELIX recovery_node.py:55 (strict < not <=).
        env = _envelope(cooldown=5.0)
        env.evaluate(ACTION_STOP_AND_HOLD, "fault_a", now=0.0)
        result = env.evaluate(ACTION_STOP_AND_HOLD, "fault_a", now=5.0)
        assert result.status == "ACCEPTED"

    def test_independent_fault_keys_have_independent_cooldowns(self):
        env = _envelope(cooldown=5.0)
        env.evaluate(ACTION_STOP_AND_HOLD, "fault_a", now=0.0)
        # fault_b has never fired: should be accepted.
        result = env.evaluate(ACTION_STOP_AND_HOLD, "fault_b", now=1.0)
        assert result.status == "ACCEPTED"

    def test_hold_also_subject_to_cooldown(self):
        env = _envelope(cooldown=5.0)
        env.evaluate(ACTION_HOLD, "fault_a", now=0.0)
        result = env.evaluate(ACTION_HOLD, "fault_a", now=2.0)
        assert result.status == "SUPPRESSED_COOLDOWN"

    def test_hold_and_stop_have_separate_cooldown_keys(self):
        # HOLD and STOP_AND_HOLD are different actions, so they key separately.
        env = _envelope(cooldown=5.0)
        env.evaluate(ACTION_HOLD, "fault_a", now=0.0)
        result = env.evaluate(ACTION_STOP_AND_HOLD, "fault_a", now=1.0)
        # Different (action, fault_key) key: not gated by HOLD cooldown.
        assert result.status == "ACCEPTED"

    def test_rewind_not_subject_to_cooldown(self):
        # REWIND is in _ACTUATING_ACTIONS but NOT in the cooldown check
        # (only HOLD and STOP_AND_HOLD are damped). This matches the spec intent:
        # cooldown damps hold flapping; rewind is rare and must not be blocked.
        env = _envelope(cooldown=5.0)
        env.evaluate(ACTION_REWIND, "fault_a", now=0.0)
        result = env.evaluate(ACTION_REWIND, "fault_a", now=0.1)
        assert result.status == "ACCEPTED"  # no cooldown on REWIND


class TestCooldownCouplesToActuation:
    """LOCKED decision 5: cooldown damps NEW holds only; a continued hold is
    cooldown-exempt and is never released by the cooldown gate.
    """

    def test_continued_hold_within_cooldown_keeps_publishing(self):
        """A re-asserted hold within the cooldown window, when the hold is
        already active, stays publishing (cooldown-exempt continuation).
        """
        env = _envelope(cooldown=5.0)
        first = env.evaluate(ACTION_STOP_AND_HOLD, "fault_a", now=0.0)
        assert first.publish is True
        # Re-assert within cooldown, but the hold is ALREADY active: continuation.
        second = env.evaluate(
            ACTION_STOP_AND_HOLD, "fault_a", now=2.0, hold_already_active=True
        )
        assert second.status == "ACCEPTED"
        assert second.publish is True  # ongoing hold keeps publishing, NOT released

    def test_new_hold_within_cooldown_is_suppressed(self):
        """A NEW hold (none active yet) within cooldown is still damped."""
        env = _envelope(cooldown=5.0)
        env.evaluate(ACTION_STOP_AND_HOLD, "fault_a", now=0.0)
        # hold_already_active defaults False -> a NEW assertion -> cooldown damps it.
        second = env.evaluate(ACTION_STOP_AND_HOLD, "fault_a", now=2.0)
        assert second.status == "SUPPRESSED_COOLDOWN"
        assert second.publish is False

    def test_repeated_intervene_continuation_never_releases(self):
        """Many re-asserts within cooldown, all as continuations, keep the hold."""
        env = _envelope(cooldown=5.0)
        env.evaluate(ACTION_HOLD, "fault_a", now=0.0)
        for t in (0.5, 1.0, 1.5, 2.0, 2.5):
            r = env.evaluate(ACTION_HOLD, "fault_a", now=t, hold_already_active=True)
            assert r.publish is True, f"continuation at t={t} must keep publishing"


class TestResumeExemption:
    """RESUME must never be blocked by cooldown. HELIX recovery_node.py:53."""

    def test_resume_accepted_immediately_after_stop(self):
        env = _envelope(cooldown=5.0)
        env.evaluate(ACTION_STOP_AND_HOLD, "fault_a", now=0.0)
        result = env.evaluate_resume("fault_a", now=0.0001)  # essentially same tick
        assert result.status == "ACCEPTED"
        assert result.publish is True

    def test_resume_accepted_without_prior_stop(self):
        env = _envelope(cooldown=5.0)
        result = env.evaluate_resume("fault_a", now=0.0)
        assert result.status == "ACCEPTED"

    def test_resume_not_blocked_by_repeated_stops(self):
        env = _envelope(cooldown=5.0)
        for t in range(5):
            env.evaluate(ACTION_STOP_AND_HOLD, "fault_a", now=float(t) * 10.0)
        result = env.evaluate_resume("fault_a", now=41.0)
        assert result.status == "ACCEPTED"
        assert result.publish is True


class TestDisabled:
    def test_disabled_suppresses_all_actions(self):
        env = _envelope(enabled=False)
        all_actions = (ACTION_HOLD, ACTION_STOP_AND_HOLD, ACTION_REWIND, ACTION_LOG_ONLY,
                       ACTION_NONE)
        for action in all_actions:
            result = env.evaluate(action, "fault_a", now=0.0)
            assert result.status == "SUPPRESSED_DISABLED"
            assert result.publish is False

    def test_disabled_suppresses_resume(self):
        env = _envelope(enabled=False)
        result = env.evaluate_resume("fault_a", now=0.0)
        assert result.status == "SUPPRESSED_DISABLED"
        assert result.publish is False
