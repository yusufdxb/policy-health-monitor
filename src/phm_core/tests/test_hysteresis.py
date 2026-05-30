"""Tests for the consecutive-violation hysteresis counter."""

from __future__ import annotations

import pytest

from phm_core.hysteresis import Hysteresis


def test_fires_only_after_min_consecutive():
    h = Hysteresis(min_consecutive=3)
    assert h.observe(True) is False  # 1
    assert h.observe(True) is False  # 2
    assert h.observe(True) is True   # 3 -> fires


def test_stays_fired_while_run_continues():
    h = Hysteresis(min_consecutive=2)
    assert h.observe(True) is False
    assert h.observe(True) is True
    assert h.observe(True) is True   # still violating, still fired
    assert h.observe(True) is True


def test_healthy_sample_resets_the_run():
    h = Hysteresis(min_consecutive=3)
    h.observe(True)
    h.observe(True)
    assert h.count == 2
    assert h.observe(False) is False  # healthy resets
    assert h.count == 0
    # Must again accumulate the full run before firing.
    assert h.observe(True) is False
    assert h.observe(True) is False
    assert h.observe(True) is True


def test_intermittent_violations_never_fire():
    h = Hysteresis(min_consecutive=3)
    for _ in range(10):
        assert h.observe(True) is False
        assert h.observe(False) is False  # broken by a healthy sample each time


def test_min_consecutive_one_fires_immediately():
    h = Hysteresis(min_consecutive=1)
    assert h.observe(True) is True
    assert h.observe(False) is False
    assert h.observe(True) is True


def test_explicit_reset_clears_run():
    h = Hysteresis(min_consecutive=2)
    h.observe(True)
    assert h.count == 1
    h.reset()
    assert h.count == 0
    assert h.observe(True) is False  # run restarts after reset


def test_rejects_min_consecutive_below_one():
    with pytest.raises(ValueError):
        Hysteresis(min_consecutive=0)
    with pytest.raises(ValueError):
        Hysteresis(min_consecutive=-1)


def test_count_and_min_consecutive_properties():
    h = Hysteresis(min_consecutive=4)
    assert h.min_consecutive == 4
    assert h.count == 0
    h.observe(True)
    h.observe(True)
    assert h.count == 2
