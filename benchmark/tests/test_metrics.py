"""Metrics correctness on tiny, hand-computable fixtures.

The benchmark venv has no sklearn, so these pin the pure-numpy AUROC / AUPR /
FPR@95 implementations to values computed by hand.
"""

import numpy as np
from lib import metrics
from lib.metrics import aggregate_seeds, calibrate_threshold_at_tpr, lead_time


def test_auroc_perfect_separation():
    # All OOD scores above all ID scores -> AUROC 1.0.
    scores = np.array([0.1, 0.2, 0.3, 0.9, 1.0, 1.1])
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert metrics.auroc(scores, labels) == 1.0


def test_auroc_inverted_separation():
    # OOD scores all below ID -> AUROC 0.0.
    scores = np.array([0.9, 1.0, 1.1, 0.1, 0.2, 0.3])
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert metrics.auroc(scores, labels) == 0.0


def test_auroc_known_value():
    # 2 OOD (scores 2, 0.5), 2 ID (scores 1, 0). Pairwise wins:
    # OOD=2 beats both ID -> 2; OOD=0.5 beats ID=0 only -> 1. Total 3 / 4 = 0.75.
    scores = np.array([1.0, 0.0, 2.0, 0.5])
    labels = np.array([0, 0, 1, 1])
    assert abs(metrics.auroc(scores, labels) - 0.75) < 1e-12


def test_auroc_tie_midrank():
    # One tie between an ID and an OOD at score 1.0 counts as half a win.
    # OOD scores: 1.0 (tie with ID 1.0 -> 0.5, beats ID 0.0 -> 1) = 1.5
    # OOD score 2.0 beats both -> 2. Total 3.5 / (2*2) = 0.875.
    scores = np.array([1.0, 0.0, 1.0, 2.0])
    labels = np.array([0, 0, 1, 1])
    assert abs(metrics.auroc(scores, labels) - 0.875) < 1e-12


def test_aupr_perfect():
    scores = np.array([0.1, 0.2, 0.9, 1.0])
    labels = np.array([0, 0, 1, 1])
    assert abs(metrics.aupr(scores, labels) - 1.0) < 1e-12


def test_aupr_known_value():
    # Ranked by score desc: scores [3,2,1,0], labels [1,0,1,0].
    # k=1: tp=1,fp=0 -> P=1.0, R=0.5; recall step 0.5 -> +1.0*0.5 = 0.5
    # k=2: tp=1,fp=1 -> P=0.5, R=0.5; recall step 0 -> +0
    # k=3: tp=2,fp=1 -> P=2/3, R=1.0; recall step 0.5 -> +2/3*0.5 = 1/3
    # k=4: tp=2,fp=2 -> P=0.5, R=1.0; recall step 0 -> +0
    # AP = 0.5 + 1/3 = 0.8333...
    scores = np.array([3.0, 2.0, 1.0, 0.0])
    labels = np.array([1, 0, 1, 0])
    assert abs(metrics.aupr(scores, labels) - (0.5 + 1.0 / 3.0)) < 1e-12


def test_fpr_at_tpr_perfect():
    # Perfect separation -> FPR@95 is 0.
    scores = np.array([0.1, 0.2, 0.3, 0.9, 1.0, 1.1])
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert metrics.fpr_at_tpr(scores, labels, 0.95) == 0.0


def test_fpr_at_tpr_known():
    # 4 OOD, 4 ID. To reach TPR>=0.75 we must cross the 3rd-highest OOD.
    # scores: OOD=[4,3,2,1], ID=[3.5,2.5,1.5,0.5]. Threshold sweep:
    # at score 2 (3rd OOD) TPR=3/4=0.75; negatives above 2: 3.5,2.5 -> FPR=2/4=0.5.
    scores = np.array([4, 3, 2, 1, 3.5, 2.5, 1.5, 0.5], dtype=float)
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    assert abs(metrics.fpr_at_tpr(scores, labels, 0.75) - 0.5) < 1e-12


def test_single_class_returns_nan():
    scores = np.array([1.0, 2.0, 3.0])
    labels = np.array([0, 0, 0])
    assert np.isnan(metrics.auroc(scores, labels))
    assert np.isnan(metrics.aupr(scores, labels))
    assert np.isnan(metrics.fpr_at_tpr(scores, labels))


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    s_id = rng.normal(0, 1, 200)
    s_ood = rng.normal(2, 1, 200)
    scores = np.concatenate([s_id, s_ood])
    labels = np.concatenate([np.zeros(200), np.ones(200)]).astype(int)
    point = metrics.auroc(scores, labels)
    mean, lo, hi = metrics.bootstrap_ci(metrics.auroc, scores, labels,
                                        n_bootstrap=300)
    assert lo <= point <= hi
    assert lo < hi


def test_lead_time_positive_when_alarm_precedes_onset():
    scores = np.zeros(100)
    scores[45:] = 1.0
    assert lead_time(scores, threshold=0.5, onset=50) == 5


def test_lead_time_negative_when_alarm_lags_onset():
    scores = np.zeros(100)
    scores[55:] = 1.0
    assert lead_time(scores, threshold=0.5, onset=50) == -5


def test_lead_time_none_when_no_alarm():
    scores = np.zeros(100)
    assert lead_time(scores, threshold=0.5, onset=50) is None


def test_lead_time_none_when_no_onset():
    scores = np.ones(100)
    assert lead_time(scores, threshold=0.5, onset=-1) is None


def test_aggregate_seeds_mean_std():
    vals = [0.90, 0.92, 0.94, 0.96, 0.98]
    agg = aggregate_seeds(vals)
    assert abs(agg["mean"] - 0.94) < 1e-9
    assert abs(agg["std"] - np.std(vals, ddof=1)) < 1e-9
    assert agg["n"] == 5


def test_aggregate_seeds_ignores_none():
    agg = aggregate_seeds([0.9, None, 0.8, None])
    assert agg["n"] == 2
    assert abs(agg["mean"] - 0.85) < 1e-9


def test_aggregate_seeds_all_none():
    agg = aggregate_seeds([None, None])
    assert agg["n"] == 0
    assert agg["mean"] is None


def test_calibrate_threshold_at_tpr_hits_target():
    rng = np.random.default_rng(0)
    _id_scores = rng.normal(0, 1, 1000)
    ood_scores = rng.normal(6, 1, 1000)
    thr = calibrate_threshold_at_tpr(ood_scores, target_tpr=0.95)
    tpr = (ood_scores >= thr).mean()
    assert abs(tpr - 0.95) < 0.03
