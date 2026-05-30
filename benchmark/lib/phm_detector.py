"""Adapter that turns the PHM rolling-spread detector into a higher-is-OOD score.

The detector under test is ``phm_core.calibration.rolling_spread`` +
``calibrate_threshold`` (the internal-feature OOD score). Source:
``/home/yusuf/Projects/policy-health-monitor/src/phm_core/phm_core/calibration.py``
(itself ported from phantom-braking e6_detector.py:16-31).

PHM convention: LOWER rolling spread = more OOD (hidden-state collapse). The
benchmark metrics use the common convention HIGHER = more OOD, so this adapter
emits a per-frame signed-deficit score

    score_t = threshold - spread_t

where ``threshold`` is calibrated on the ID spread distribution. A collapsed
(low-spread) frame gives a large positive score; a healthy frame gives a
negative score. This is a strictly monotone (order-reversing of spread) map, so
AUROC / AUPR / FPR@95 are invariant to the exact affine offset and only the
direction flip matters. Frames before the rolling window fills are NaN and are
dropped by the metrics layer.

This adapter does NOT modify phm_core; it imports the public functions.
"""

from __future__ import annotations

import numpy as np

from phm_core.calibration import calibrate_threshold, rolling_spread


def phm_scores(features_id: np.ndarray, features_test: np.ndarray,
               window: int = 20, percentile: float = 1.0) -> np.ndarray:
    """Per-frame higher-is-OOD PHM score for ``features_test``.

    Calibration is the rolling-spread threshold fit on ``features_id`` at the
    given ID percentile (default 1st percentile, the calibrate_threshold
    default). Returns an array of len(features_test) with NaN before the window
    fills.
    """
    id_spread = rolling_spread(np.asarray(features_id, dtype=np.float64), window)
    thr = calibrate_threshold(id_spread, percentile=percentile)
    test_spread = rolling_spread(np.asarray(features_test, dtype=np.float64), window)
    return (thr - test_spread).astype(np.float64)
