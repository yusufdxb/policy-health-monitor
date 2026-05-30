"""Rolling-spread OOD calibration, ported from Phantom-Braking.

Source: ``/home/yusuf/Projects/phantom-braking/src/e6_detector.py:16-53``
(functions ``rolling_spread``, ``calibrate_threshold``, ``loco_fpr``).

The math here is byte-faithful to that source. The OOD signal is the per-frame
trace of the rolling covariance of a policy's hidden state: when the hidden
state collapses (freezes to a point), the windowed variance sum drops toward
zero, so a low spread (below a threshold calibrated on real in-distribution
data) flags out-of-distribution behavior.

Divergence from the source, intentional and spec-driven: the spec (section 3.2)
asks for ``loco_fpr(corpora: list[np.ndarray], window, percentile)`` taking a
list of corpora, whereas the source takes a ``dict[str, np.ndarray]`` keyed by
corpus name. This module's :func:`loco_fpr` accepts the spec's list form and
keys folds by integer index; the per-fold arithmetic (calibrate on the other
corpora, evaluate held-out FPR, report mean and max) is identical to the source.
"""

from __future__ import annotations

import numpy as np


def rolling_spread(hidden: np.ndarray, window: int) -> np.ndarray:
    """Per-frame trace of the rolling covariance of the hidden state.

    ``hidden`` is shape (T, D). Returns shape (T,), NaN before the window fills.

    Ported verbatim from e6_detector.py:16-23.
    """
    T, D = hidden.shape
    out = np.full(T, np.nan, dtype=np.float64)
    for t in range(window, T + 1):
        out[t - 1] = float(np.var(hidden[t - window:t], axis=0).sum())
    return out


def calibrate_threshold(real_spreads: np.ndarray, percentile: float = 1.0) -> float:
    """Below this spread = OOD.

    Pick the ``percentile``-th percentile of the real-driving spread
    distribution so real drives stay above it ~99% of the time (at the default
    1st percentile).

    Ported verbatim from e6_detector.py:26-31.
    """
    s = real_spreads[~np.isnan(real_spreads)]
    return float(np.percentile(s, percentile))


def loco_fpr(corpora: list[np.ndarray], window: int, percentile: float) -> dict:
    """Leave-one-corpus-out false-positive rate.

    Calibrate the threshold on N-1 corpora, evaluate the held-out corpus, repeat
    for each corpus, and report per-fold FPR plus mean and max.

    Ported from e6_detector.py:34-53. The source keys corpora by name (a dict);
    per the spec this takes a list and keys each fold by integer index. The
    per-fold computation (concatenate the calibration corpora, compute their
    rolling spread, calibrate the threshold, then measure the fraction of valid
    held-out spreads below that threshold) matches the source exactly.

    Args:
        corpora: list of in-distribution hidden-state arrays, each shape (T, D).
        window: rolling-spread window length.
        percentile: percentile used by :func:`calibrate_threshold`.

    Returns:
        dict with ``folds`` (per-index threshold, fpr, calibrated_on indices),
        ``fpr_mean`` and ``fpr_max``.
    """
    folds: dict[int, dict] = {}
    indices = list(range(len(corpora)))
    for held_out in indices:
        calib_keys = [k for k in indices if k != held_out]
        calib_hidden = np.concatenate([corpora[k] for k in calib_keys], axis=0)
        calib_spreads = rolling_spread(calib_hidden, window)
        thr = calibrate_threshold(calib_spreads, percentile)
        held_spreads = rolling_spread(corpora[held_out], window)
        valid = held_spreads[~np.isnan(held_spreads)]
        fpr = float(np.mean(valid < thr)) if len(valid) else float("nan")
        folds[held_out] = {"threshold": thr, "fpr": fpr,
                           "calibrated_on": calib_keys}
    fprs = np.array([f["fpr"] for f in folds.values()])
    return {"folds": folds, "fpr_mean": float(fprs.mean()),
            "fpr_max": float(fprs.max())}
