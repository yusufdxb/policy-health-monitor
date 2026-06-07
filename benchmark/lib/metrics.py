"""Threshold-free OOD detection metrics and bootstrap confidence intervals.

Ported (re-implemented, not cross-imported) from
``/home/yusuf/Projects/phantom-braking/src/metrics.py:1-118``.

Divergence from the source, intentional and environment-driven: the source
wraps ``sklearn.metrics`` (``roc_auc_score``, ``average_precision_score``,
``roc_curve``, ``precision_recall_curve``). The benchmark venv has numpy but
NOT scikit-learn (verified: ``import sklearn`` -> ModuleNotFoundError), and the
sandbox has no PyPI access, so this module re-implements the same three metrics
(AUROC, AUPR, FPR@95TPR) and the stratified bootstrap CI in pure numpy. The
definitions match sklearn's:

- AUROC: Mann-Whitney-U / rank-based area under the ROC curve, midrank tie
  handling (identical to ``roc_auc_score`` for binary labels).
- AUPR: average precision = sum over recall increments of precision
  (identical to ``average_precision_score``, the step-function AP, not the
  trapezoidal PR-AUC).
- FPR@95TPR: FPR at the smallest threshold whose TPR >= 0.95, matching the
  source's ``np.searchsorted(tpr, 0.95, side="left")`` over the monotone ROC.

The numpy AUROC and AUPR are unit-tested against hand-computed values on a
tiny fixture in ../tests/test_metrics.py.

Conventions (same as source):
- Higher score = more OOD.
- Label 1 = OOD, label 0 = ID.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def _clean(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # metrics.py:28-32
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    mask = np.isfinite(s)
    return s[mask], y[mask]


def _roc_curve(s: np.ndarray, y: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ROC curve (fpr, tpr, thresholds), numpy port of sklearn.roc_curve.

    Sort scores descending; at each distinct score sweep the threshold and
    accumulate true/false positives. Prepends the (0, 0) origin so the curve
    starts at TPR=FPR=0, matching sklearn's drop_intermediate=False output
    shape for the points we use (searchsorted only needs monotone tpr).
    """
    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]

    # Indices where the score changes -> threshold points.
    distinct = np.where(np.diff(s_sorted))[0]
    threshold_idx = np.r_[distinct, s_sorted.size - 1]

    tps = np.cumsum(y_sorted)[threshold_idx]
    fps = 1 + threshold_idx - tps  # count of negatives seen so far

    p = float((y == 1).sum())
    n = float((y == 0).sum())
    tpr = np.r_[0.0, tps / p]
    fpr = np.r_[0.0, fps / n]
    thr = np.r_[np.inf, s_sorted[threshold_idx]]
    return fpr, tpr, thr


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve. Higher score = more OOD; label 1 = OOD.

    Rank-based (Mann-Whitney U) with midrank tie handling, equal to
    sklearn.roc_auc_score for binary labels. metrics.py:35-40.
    """
    s, y = _clean(scores, labels)
    if len(np.unique(y)) < 2:
        return float("nan")
    # Midranks of all scores (1-indexed average ranks for ties).
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(sorted_s):
        j = i
        while j + 1 < len(sorted_s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-indexed midrank
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    pos = y == 1
    n_pos = float(pos.sum())
    n_neg = float((~pos).sum())
    sum_ranks_pos = float(ranks[pos].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return float(auc)


def aupr(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision (step-function AP), equal to
    sklearn.average_precision_score. metrics.py:43-48.

    AP = sum_n (R_n - R_{n-1}) * P_n over the ranked predictions, where the
    threshold sweeps from the highest score downward.
    """
    s, y = _clean(scores, labels)
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]
    p = float((y == 1).sum())

    # Collapse ties: precision/recall are only defined at distinct thresholds.
    distinct = np.where(np.diff(s_sorted))[0]
    threshold_idx = np.r_[distinct, s_sorted.size - 1]
    tps = np.cumsum(y_sorted)[threshold_idx]
    fps = 1 + threshold_idx - tps
    precision = tps / np.maximum(tps + fps, 1e-12)
    recall = tps / p

    # AP = sum of precision[k] * (recall[k] - recall[k-1]), recall[-1] = 0.
    recall_prev = np.r_[0.0, recall[:-1]]
    ap = float(np.sum(precision * (recall - recall_prev)))
    return ap


def fpr_at_tpr(scores: np.ndarray, labels: np.ndarray,
               tpr_target: float = 0.95) -> float:
    """FPR at the smallest threshold whose TPR >= tpr_target. metrics.py:51-65."""
    s, y = _clean(scores, labels)
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = _roc_curve(s, y)
    idx = np.searchsorted(tpr, tpr_target, side="left")
    if idx >= len(tpr):
        return float("nan")
    return float(fpr[idx])


def roc_curve_points(scores: np.ndarray, labels: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(fpr, tpr, thresholds) for plotting / analysis. metrics.py:68-73."""
    s, y = _clean(scores, labels)
    return _roc_curve(s, y)


def bootstrap_ci(metric_fn: Callable[[np.ndarray, np.ndarray], float],
                 scores: np.ndarray, labels: np.ndarray,
                 n_bootstrap: int = 1000, ci: float = 0.95,
                 seed: int = 42) -> tuple[float, float, float]:
    """Stratified bootstrap CI for any (scores, labels) -> float metric.

    Ported verbatim from metrics.py:84-117. Resamples ID and OOD frames
    independently with replacement so the per-class count is preserved.
    Returns (mean, lo, hi).
    """
    s, y = _clean(scores, labels)
    rng = np.random.RandomState(int(seed))
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    if len(idx_pos) == 0 or len(idx_neg) == 0:
        return float("nan"), float("nan"), float("nan")
    vals = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        rp = rng.choice(idx_pos, size=len(idx_pos), replace=True)
        rn = rng.choice(idx_neg, size=len(idx_neg), replace=True)
        sel = np.concatenate([rp, rn])
        vals[b] = metric_fn(s[sel], y[sel])
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(vals, alpha))
    hi = float(np.quantile(vals, 1.0 - alpha))
    return float(vals.mean()), lo, hi


def lead_time(scores, threshold, onset):
    """Frames between the first alarm and the ground-truth failure onset.

    Positive = the detector alarmed BEFORE the failure (warned early); negative =
    it lagged. Returns None if there is no onset (onset < 0) or the detector never
    alarms over the stream. `scores` are higher-is-OOD; an alarm is score>=threshold.
    """
    s = np.asarray(scores, dtype=np.float64)
    if onset is None or onset < 0:
        return None
    alarms = np.where(s >= threshold)[0]
    if alarms.size == 0:
        return None
    return int(onset - alarms[0])
