"""Tests for HealthToActionMapper and RewindHook.

Verifies:
- STATE_STOP -> STOP_AND_HOLD, hold_active=True.
- STATE_INTERVENE + ACTION_HOLD -> HOLD, hold_active=True.
- STATE_INTERVENE + ACTION_REWIND -> REWIND, hold_active=True.
- STATE_INTERVENE + ACTION_LOG_ONLY -> LOG_ONLY, preserves existing hold.
- STATE_OK / STATE_DEGRADED -> clear hold (ACTION_NONE, hold_active=False).
- Hold stays active through consecutive STOP messages.
- RewindHook: default (no callback) logs a warning; registered callback is called.

These tests import only phm_recovery._core (no rclpy).
"""

from __future__ import annotations

import logging

from phm_recovery._core import (
    ACTION_HOLD,
    ACTION_LOG_ONLY,
    ACTION_NONE,
    ACTION_REWIND,
    ACTION_STOP_AND_HOLD,
    STATE_DEGRADED,
    STATE_INTERVENE,
    STATE_OK,
    STATE_STOP,
    HealthToActionMapper,
    RewindHook,
)

# ---------------------------------------------------------------------------
# HealthToActionMapper
# ---------------------------------------------------------------------------

class TestStateMappings:
    def _mapper(self) -> HealthToActionMapper:
        return HealthToActionMapper()

    def test_stop_forces_stop_and_hold(self):
        m = self._mapper()
        d = m.map(STATE_STOP, ACTION_NONE, "phm_ood", "ood threshold exceeded")
        assert d.action == ACTION_STOP_AND_HOLD
        assert d.hold_active is True

    def test_stop_overrides_suggested_action(self):
        """Even if suggested_action is LOG_ONLY, STATE_STOP still forces STOP_AND_HOLD."""
        m = self._mapper()
        d = m.map(STATE_STOP, ACTION_LOG_ONLY, "phm_ood", "ood")
        assert d.action == ACTION_STOP_AND_HOLD
        assert d.hold_active is True

    def test_intervene_hold_activates_hold(self):
        m = self._mapper()
        d = m.map(STATE_INTERVENE, ACTION_HOLD, "freq:/scan", "scan timeout")
        assert d.action == ACTION_HOLD
        assert d.hold_active is True

    def test_intervene_stop_and_hold_activates_hold(self):
        m = self._mapper()
        d = m.map(STATE_INTERVENE, ACTION_STOP_AND_HOLD, "freq:/scan", "scan timeout")
        assert d.action == ACTION_HOLD
        assert d.hold_active is True

    def test_intervene_rewind_triggers_rewind(self):
        m = self._mapper()
        d = m.map(STATE_INTERVENE, ACTION_REWIND, "phm_ood", "rewind requested")
        assert d.action == ACTION_REWIND
        assert d.hold_active is True

    def test_intervene_log_only_no_new_hold(self):
        """INTERVENE with LOG_ONLY does not start a hold (if none was active)."""
        m = self._mapper()
        d = m.map(STATE_INTERVENE, ACTION_LOG_ONLY, "threshold:cpu", "cpu 90%")
        assert d.action == ACTION_LOG_ONLY
        assert d.hold_active is False  # was not active before

    def test_intervene_log_only_preserves_existing_hold(self):
        """INTERVENE/LOG_ONLY preserves a hold started by a prior STOP."""
        m = self._mapper()
        m.map(STATE_STOP, ACTION_NONE, "ood", "")  # start a hold
        d = m.map(STATE_INTERVENE, ACTION_LOG_ONLY, "threshold:cpu", "cpu 90%")
        assert d.hold_active is True  # prior hold preserved

    def test_ok_clears_hold(self):
        m = self._mapper()
        m.map(STATE_STOP, ACTION_NONE, "ood", "")  # start hold
        assert m.hold_active is True
        d = m.map(STATE_OK, ACTION_NONE, "ood", "recovered")
        assert d.action == ACTION_NONE
        assert d.hold_active is False
        assert m.hold_active is False

    def test_degraded_clears_hold(self):
        m = self._mapper()
        m.map(STATE_STOP, ACTION_NONE, "ood", "")
        d = m.map(STATE_DEGRADED, ACTION_LOG_ONLY, "ood", "slightly degraded")
        assert d.hold_active is False

    def test_ok_when_no_hold_active_returns_none(self):
        m = self._mapper()
        d = m.map(STATE_OK, ACTION_NONE, "ood", "all good")
        assert d.action == ACTION_NONE
        assert d.hold_active is False

    def test_hold_stays_active_through_consecutive_stops(self):
        m = self._mapper()
        for _ in range(5):
            d = m.map(STATE_STOP, ACTION_NONE, "ood", "")
            assert d.hold_active is True
        assert m.hold_active is True

    def test_clear_hold_programmatic(self):
        m = self._mapper()
        m.map(STATE_STOP, ACTION_NONE, "ood", "")
        m.clear_hold()
        assert m.hold_active is False

    def test_unknown_state_conservative_stop(self):
        m = self._mapper()
        d = m.map(99, ACTION_NONE, "unknown", "")
        assert d.action == ACTION_STOP_AND_HOLD
        assert d.hold_active is True


class TestHoldActiveProperty:
    def test_initially_false(self):
        m = HealthToActionMapper()
        assert m.hold_active is False

    def test_becomes_true_on_stop(self):
        m = HealthToActionMapper()
        m.map(STATE_STOP, ACTION_NONE, "s", "")
        assert m.hold_active is True

    def test_returns_false_after_clear(self):
        m = HealthToActionMapper()
        m.map(STATE_STOP, ACTION_NONE, "s", "")
        m.map(STATE_OK, ACTION_NONE, "s", "")
        assert m.hold_active is False


# ---------------------------------------------------------------------------
# RewindHook
# ---------------------------------------------------------------------------

class TestRewindHook:
    def test_default_trigger_logs_warning(self, caplog):
        hook = RewindHook()
        with caplog.at_level(logging.WARNING, logger="phm_recovery._core"):
            hook.trigger()
        assert any("REWIND" in r.message for r in caplog.records)

    def test_registered_callback_is_called(self):
        hook = RewindHook()
        called = []
        hook.register(lambda: called.append(True))
        hook.trigger()
        assert called == [True]

    def test_callback_called_on_each_trigger(self):
        hook = RewindHook()
        count = []
        hook.register(lambda: count.append(1))
        hook.trigger()
        hook.trigger()
        assert len(count) == 2

    def test_register_replaces_previous_callback(self):
        hook = RewindHook()
        calls_a, calls_b = [], []
        hook.register(lambda: calls_a.append(1))
        hook.register(lambda: calls_b.append(1))  # replaces
        hook.trigger()
        assert calls_a == []
        assert calls_b == [1]
