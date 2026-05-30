"""Parity tests for the rolling-spread calibration ported from Phantom-Braking.

Fixture values were hand-computed and cross-checked against the source math (see
e6_detector.py:16-53). Each expected number below is derived by hand in the
docstrings, not lifted from a run of the code under test.
"""

from __future__ import annotations

import numpy as np
import pytest

from phm_core.calibration import calibrate_threshold, loco_fpr, rolling_spread

# Fixture: T=4, D=2, window=2.
HIDDEN = np.array(
    [
        [0.0, 1.0],
        [2.0, 1.0],
        [2.0, 4.0],
        [5.0, 4.0],
    ],
    dtype=np.float64,
)
WINDOW = 2


def test_rolling_spread_matches_hand_computation():
    # t=2 (rows 0,1): col0 var([0,2])=1.0, col1 var([1,1])=0.0 -> 1.0
    # t=3 (rows 1,2): col0 var([2,2])=0.0, col1 var([1,4])=2.25 -> 2.25
    # t=4 (rows 2,3): col0 var([2,5])=2.25, col1 var([4,4])=0.0 -> 2.25
    out = rolling_spread(HIDDEN, WINDOW)
    assert out.shape == (4,)
    assert np.isnan(out[0])  # window not yet filled
    np.testing.assert_allclose(out[1:], [1.0, 2.25, 2.25], rtol=0, atol=1e-12)


def test_rolling_spread_nan_until_window_fills():
    # With window=3 over T=4, only t=3 and t=4 produce values.
    out = rolling_spread(HIDDEN, 3)
    assert np.isnan(out[0])
    assert np.isnan(out[1])
    assert not np.isnan(out[2])
    assert not np.isnan(out[3])


def test_calibrate_threshold_first_percentile():
    # Valid spreads are [1.0, 2.25, 2.25]; numpy's 1st percentile with linear
    # interpolation over a length-3 sorted array gives 1.0 + 0.01*(2.25-1.0)*2
    # = 1.0 + 0.025 = 1.025.
    out = rolling_spread(HIDDEN, WINDOW)
    thr = calibrate_threshold(out, percentile=1.0)
    assert thr == pytest.approx(1.025, abs=1e-12)


def test_calibrate_threshold_ignores_nan():
    spreads = np.array([np.nan, 1.0, 2.25, 2.25])
    # Median (50th pct) of [1.0, 2.25, 2.25] is 2.25.
    assert calibrate_threshold(spreads, percentile=50.0) == pytest.approx(2.25, abs=1e-12)


def test_loco_fpr_two_corpora():
    # Corpus 0 = HIDDEN (large spreads), corpus 1 = a near-constant corpus with
    # tiny spreads. Hand-derived expectations:
    #   fold held-out=0: calibrate on corpus 1 (tiny spreads), threshold tiny
    #     (0.0025), so none of corpus 0's spreads fall below it -> FPR 0.0.
    #   fold held-out=1: calibrate on corpus 0, threshold 1.025, all of corpus
    #     1's tiny spreads are below it -> FPR 1.0.
    #   mean = 0.5, max = 1.0.
    c1 = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
        ],
        dtype=np.float64,
    )
    res = loco_fpr([HIDDEN, c1], window=WINDOW, percentile=1.0)
    assert set(res["folds"].keys()) == {0, 1}
    assert res["folds"][0]["fpr"] == pytest.approx(0.0, abs=1e-12)
    assert res["folds"][1]["fpr"] == pytest.approx(1.0, abs=1e-12)
    assert res["folds"][0]["calibrated_on"] == [1]
    assert res["folds"][1]["calibrated_on"] == [0]
    assert res["folds"][1]["threshold"] == pytest.approx(1.025, abs=1e-12)
    assert res["fpr_mean"] == pytest.approx(0.5, abs=1e-12)
    assert res["fpr_max"] == pytest.approx(1.0, abs=1e-12)


def test_loco_parity_against_inlined_source_math():
    """Independent reimplementation of the source loop must match loco_fpr.

    This is the byte-faithfulness guard: if the port drifts from
    e6_detector.py:34-53, this diverges.
    """
    rng = np.random.default_rng(0)
    corpora = [rng.normal(size=(40, 3)), rng.normal(size=(35, 3)),
               rng.normal(size=(50, 3))]
    window = 5
    percentile = 1.0

    # Inlined faithful copy of the source arithmetic.
    idx = list(range(len(corpora)))
    expected = {}
    for ho in idx:
        ck = [k for k in idx if k != ho]
        ch = np.concatenate([corpora[k] for k in ck], axis=0)
        thr = calibrate_threshold(rolling_spread(ch, window), percentile)
        hs = rolling_spread(corpora[ho], window)
        v = hs[~np.isnan(hs)]
        expected[ho] = float(np.mean(v < thr))

    res = loco_fpr(corpora, window=window, percentile=percentile)
    for ho in idx:
        assert res["folds"][ho]["fpr"] == pytest.approx(expected[ho], abs=1e-12)
    assert res["fpr_mean"] == pytest.approx(
        float(np.mean(list(expected.values()))), abs=1e-12
    )
    assert res["fpr_max"] == pytest.approx(
        float(np.max(list(expected.values()))), abs=1e-12
    )
