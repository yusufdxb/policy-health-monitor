"""Detector smoke tests: each detector separates its target failure mode.

These are coarse (AUROC > 0.5 thresholds), not pinned numbers, since the
detectors are stochastic in their data. They guard the harness wiring: every
detector returns a finite per-frame higher-is-OOD score of the right length and
beats chance on the failure mode it targets.
"""

import numpy as np
from lib import baselines, metrics
from lib.generator import StreamSpec, generate_stream
from lib.phm_detector import phm_scores
from lib.rnd import rnd_numpy


def _streams(mode, seed=11):
    spec = StreamSpec(dim=64, n_id=400, n_ood=400, ood_mode=mode, seed=seed)
    fid, _ = generate_stream(spec)
    eval_spec = StreamSpec(dim=64, n_id=400, n_ood=400, ood_mode=mode, seed=seed + 1)
    id_s, ood_s = generate_stream(eval_spec)
    return fid, id_s, ood_s


def _auroc(scorer, fid, id_s, ood_s):
    s_id = np.asarray(scorer(fid, id_s), dtype=float)
    s_ood = np.asarray(scorer(fid, ood_s), dtype=float)
    scores = np.concatenate([s_id, s_ood])
    labels = np.concatenate([np.zeros(len(s_id)), np.ones(len(s_ood))]).astype(int)
    return metrics.auroc(scores, labels)


def test_phm_detects_collapse():
    fid, id_s, ood_s = _streams("collapse")
    au = _auroc(lambda a, b: phm_scores(a, b, window=20), fid, id_s, ood_s)
    assert au > 0.9, f"PHM AUROC on collapse was {au}"


def test_mahalanobis_detects_shift():
    fid, id_s, ood_s = _streams("shift")
    au = _auroc(lambda a, b: baselines.mahalanobis(a, b), fid, id_s, ood_s)
    assert au > 0.9, f"Mahalanobis AUROC on shift was {au}"


def test_knn_unnormalized_detects_shift():
    # Unnormalized KNN keeps the radial (magnitude) component that carries a
    # mean shift, so it separates the shifted cluster.
    fid, id_s, ood_s = _streams("shift")
    au = _auroc(lambda a, b: baselines.knn_distance(a, b, k=50, normalize=False),
                fid, id_s, ood_s)
    assert au > 0.9, f"unnormalized KNN AUROC on shift was {au}"


def test_knn_normalized_loses_magnitude_shift():
    # L2-normalized KNN (Sun et al. 2022 default, the phantom-braking default)
    # projects onto the unit sphere and discards the radial magnitude that the
    # shift lives in, so it is at-or-below chance on a pure magnitude shift.
    # This is a documented property, not a harness bug: it is why the benchmark
    # reports both variants.
    fid, id_s, ood_s = _streams("shift")
    au = _auroc(lambda a, b: baselines.knn_distance(a, b, k=50, normalize=True),
                fid, id_s, ood_s)
    assert au < 0.5, f"normalized KNN AUROC on shift was {au}"


def test_rnd_numpy_finite_and_right_length():
    fid, id_s, ood_s = _streams("shift")
    s = rnd_numpy(fid, ood_s)
    assert s.shape == (len(ood_s),)
    assert np.all(np.isfinite(s))


def test_rnd_numpy_detects_shift():
    fid, id_s, ood_s = _streams("shift")
    au = _auroc(lambda a, b: rnd_numpy(a, b), fid, id_s, ood_s)
    assert au > 0.7, f"RND AUROC on shift was {au}"


def test_phm_scores_length_and_nan_prefix():
    fid, id_s, ood_s = _streams("collapse")
    s = phm_scores(fid, ood_s, window=20)
    assert s.shape == (len(ood_s),)
    # First window-1 frames are NaN (rolling window not yet filled).
    assert np.isnan(s[:19]).all()
    assert np.isfinite(s[19:]).all()


# --- v1.0 fair second-order / additional baselines -------------------------
# Each baseline is tested on the failure mode it ACTUALLY detects, using the
# same independent fit(seed)/eval(seed+1) split as every other test here.
#
# Empirically (correct split, measured 2026-06-07):
#   - Hotelling-T2 windowed variance is the FAIR second-order baseline: it
#     catches collapse robustly (AUROC ~0.99, like PHM rolling-spread), so
#     PHM's collapse advantage over FIRST-order baselines is not an artifact of
#     weak baselines. A second-order baseline confirms collapse is detectable.
#   - PCA-residual and temporal-RND are first-order-ish: they catch SHIFT
#     (location leaving the ID subspace) but NOT a pure collapse (a frozen point
#     INSIDE the ID subspace leaves little residual). They are included as
#     additional honest baselines, characterized on shift.


def test_hotelling_t2_is_fair_second_order_baseline_on_collapse():
    # The point of a FAIR comparison: a principled second-order baseline also
    # sees collapse (so PHM is not just beating first-order strawmen).
    fid, id_s, ood_s = _streams("collapse")
    au = _auroc(lambda a, b: baselines.hotelling_t2_window(a, b, window=20),
                fid, id_s, ood_s)
    assert au > 0.9, f"Hotelling-T2 AUROC on collapse was {au}"


def test_hotelling_t2_finite_and_right_length():
    fid, id_s, ood_s = _streams("collapse")
    s = baselines.hotelling_t2_window(fid, ood_s, window=20)
    assert s.shape == (len(ood_s),)
    assert np.all(np.isfinite(s))


def test_pca_residual_detects_shift():
    # PCA-residual catches location shift (the embedding leaves the ID subspace),
    # NOT pure collapse. Tested on shift, where it is its strongest.
    fid, id_s, ood_s = _streams("shift")
    au = _auroc(lambda a, b: baselines.pca_residual(a, b, n_components=8),
                fid, id_s, ood_s)
    assert au > 0.9, f"PCA-residual AUROC on shift was {au}"


def test_pca_residual_finite_and_right_length():
    fid, id_s, ood_s = _streams("shift")
    s = baselines.pca_residual(fid, ood_s, n_components=8)
    assert s.shape == (len(ood_s),)
    assert np.all(np.isfinite(s))


def test_temporal_rnd_detects_shift():
    fid, id_s, ood_s = _streams("shift")
    au = _auroc(lambda a, b: baselines.temporal_rnd(a, b, window=8, hidden=64, seed=0),
                fid, id_s, ood_s)
    assert au > 0.7, f"temporal-RND AUROC on shift was {au}"


def test_temporal_rnd_finite_and_right_length():
    fid, id_s, ood_s = _streams("shift")
    s = baselines.temporal_rnd(fid, ood_s, window=8, hidden=64, seed=0)
    assert s.shape == (len(ood_s),)
    assert np.all(np.isfinite(s))
