"""Tests for the TRACK D harness.

Verified properties:
  - the OOD threshold calibrated on clean features is finite,
  - the OOD signal (monitor fired-fraction) rises with alpha,
  - the lead-time equals output_collapses_at - monitor_fires_at exactly,
  - the monitor fires at a lower-or-equal alpha than output collapse on the
    shipped configuration (the headline early-warning claim),
  - the reused phm_core math is the same object the harness imports.

Run:
    /usr/bin/python3 -m pytest vla_monitor_demo -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Make the harness importable regardless of the pytest invocation cwd.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import harness as H  # noqa: E402


@pytest.fixture(scope="module")
def trained():
    """Train the stand-in policy once for the module (a few hundred ms)."""
    policy, mse = H.train_policy()
    return policy, mse


@pytest.fixture(scope="module")
def result(trained):
    policy, _ = trained
    return H.run_sweep(policy)


def test_policy_trains_to_low_error(trained):
    """The stand-in policy actually learns the synthetic control law."""
    _, mse = trained
    assert math.isfinite(mse)
    assert mse < 0.05, f"policy did not learn the task (MSE={mse})"


def test_policy_exposes_tappable_hidden(trained):
    """forward() returns (action, hidden); hidden is the policy embedding."""
    import torch

    policy, _ = trained
    x = torch.zeros(4, 16)
    action, hidden = policy(x)
    assert action.shape == (4, 2)
    assert hidden.shape[0] == 4 and hidden.shape[1] == policy.head.in_features


def test_calibrated_threshold_is_finite(result):
    """calibrate_threshold (reused from phm_core) yields a finite threshold."""
    assert math.isfinite(result.threshold)
    assert result.threshold > 0.0


def test_clean_fpr_is_small(result):
    """The clean-batch false-positive rate sits at or below the 1st percentile.

    calibrate_threshold uses the 1st percentile, so by construction the clean
    flag rate is ~0.01. We assert it is small (a sane calibration), giving the
    sensitive tripwire (0.05) real headroom above the clean floor.
    """
    assert 0.0 <= result.clean_fpr <= 0.05
    assert result.monitor_fire_fraction > result.clean_fpr


def test_ood_signal_rises_with_alpha(result):
    """The monitor fired-fraction trends up from clean to full shift.

    We check the endpoints and a positive rank correlation rather than strict
    per-step monotonicity (the signal is stochastic per alpha).
    """
    fired = result.fired_fraction
    assert fired[-1] > fired[0], "fired fraction did not rise from alpha 0 to 1"

    # Spearman-style rank correlation without scipy.
    a_rank = np.argsort(np.argsort(result.alphas))
    f_rank = np.argsort(np.argsort(fired))
    a_c = a_rank - a_rank.mean()
    f_c = f_rank - f_rank.mean()
    rho = float((a_c * f_c).sum() / (np.sqrt((a_c**2).sum() * (f_c**2).sum())))
    assert rho > 0.7, f"OOD signal does not rise with alpha (rho={rho:.3f})"


def test_ood_score_collapses_with_alpha(result):
    """The mean rolling spread of the embedding falls as the input shifts.

    Lower spread = the hidden state collapsing toward saturation, the same
    collapse direction phantom-braking detects (e6_detector.py:1-7).
    """
    ood = result.ood_score
    assert ood[-1] < ood[0], "mean rolling spread did not fall under shift"


def test_lead_time_matches_documented_formula(result):
    """lead_time == output_collapses_at - monitor_fires_at, exactly."""
    expected = result.output_collapses_at - result.monitor_fires_at
    assert result.lead_time == pytest.approx(expected, abs=1e-12)


def test_both_events_fire_within_sweep(result):
    """Both the monitor and the output collapse happen inside alpha in [0,1]."""
    assert math.isfinite(result.monitor_fires_at)
    assert math.isfinite(result.output_collapses_at)
    assert 0.0 <= result.monitor_fires_at <= 1.0
    assert 0.0 <= result.output_collapses_at <= 1.0


def test_headline_positive_lead_time(result):
    """Headline claim: the monitor fires at a lower-or-equal alpha than output
    collapse on the shipped config (early warning / non-negative lead-time).

    This is asserted as the measured, reproducible result, not a hard guarantee
    of the methodology: if the design changed and the effect vanished, this test
    would fail loudly rather than let the README overclaim.
    """
    assert result.monitor_fires_at <= result.output_collapses_at
    assert result.lead_time >= 0.0


def test_reuses_phm_core_math_not_a_copy():
    """The harness imports rolling_spread / calibrate_threshold from phm_core,
    not a local re-implementation (the spec forbids duplicating the math)."""
    import phm_core.calibration as cal

    assert H.rolling_spread is cal.rolling_spread
    assert H.calibrate_threshold is cal.calibrate_threshold


def test_rolling_spread_drops_on_frozen_hidden():
    """Sanity check on the reused math: a frozen (constant) hidden state has
    zero rolling spread, a varying one has positive spread."""
    frozen = np.ones((100, 8), dtype=np.float64)
    varying = np.random.default_rng(0).standard_normal((100, 8))
    sf = H.rolling_spread(frozen, 30)
    sv = H.rolling_spread(varying, 30)
    assert np.nanmax(sf) == pytest.approx(0.0, abs=1e-12)
    assert np.nanmean(sv) > 0.1


def test_perturb_identity_at_alpha_zero():
    """alpha = 0 must leave the inputs untouched (clean reference is exact)."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((10, 16)).astype(np.float32)
    out = H.perturb_inputs(x, 0.0, rng)
    assert np.array_equal(x, out)


def test_perturb_moves_inputs_at_alpha_one():
    """alpha = 1 must move the inputs off the clean manifold."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((10, 16)).astype(np.float32)
    out = H.perturb_inputs(x, 1.0, rng)
    assert not np.array_equal(x, out)
    assert float(np.abs(out - x).mean()) > 0.1
