"""Canonical post-hoc feature-space OOD baselines.

Ported (re-implemented, not cross-imported) from
``/home/yusuf/Projects/phantom-braking/src/baselines.py``:

- ``_fit_gaussian``                  <- baselines.py:112-131
- ``mahalanobis``                    <- baselines.py:134-145
- ``_fit_background_gmm``            <- baselines.py:148-191
- ``relative_mahalanobis``           <- baselines.py:194-216
- ``knn_distance``                   <- baselines.py:219-251

All return per-frame OOD scores where HIGHER means more OOD, matching the
source convention so a single threshold (real-data high-quantile) holds.

The PHM rolling-spread detector under test flips this internally (lower spread
= more OOD); the harness converts it to higher-is-OOD by negation before
computing metrics, exactly as the phantom-braking baselines_results.md report
converts both to a common direction.

ViM / MSP / Energy are N/A in this regression / embedding setting: there are no
classifier logits and no softmax head over a closed label set. This mirrors the
not-applicable analysis in baselines.py:60-107 and is recorded in RESULTS.md
rather than silently skipped.
"""

from __future__ import annotations

import numpy as np


def _fit_gaussian(features_id: np.ndarray, reg: float = 0.1
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Empirical mean and precision (inverse cov) of ID features.

    baselines.py:112-131. Trace-relative ridge shrinkage keeps the inverse
    well-conditioned when N < D.
    """
    mu = features_id.mean(axis=0)
    X = features_id - mu
    cov = (X.T @ X) / max(len(X) - 1, 1)
    trace_mean = float(np.trace(cov) / cov.shape[0]) if cov.shape[0] else 1.0
    cov = cov + (reg * trace_mean + 1e-6) * np.eye(cov.shape[0], dtype=cov.dtype)
    prec = np.linalg.inv(cov)
    return mu, prec


def mahalanobis(features_id: np.ndarray, features_test: np.ndarray,
                reg: float = 0.1) -> np.ndarray:
    """Squared Mahalanobis distance from the ID distribution. Higher = OOD.

    Single-Gaussian fit. Lee et al. NeurIPS 2018 (arXiv:1807.03888).
    baselines.py:134-145.
    """
    mu, prec = _fit_gaussian(features_id, reg=reg)
    X = features_test - mu
    return np.einsum("ij,jk,ik->i", X, prec, X).astype(np.float64)


def _fit_background_gmm(features_id: np.ndarray, n_components: int = 2,
                        reg: float = 0.1, seed: int = 0
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Coarse 2-component background fit for Relative Mahalanobis.

    Tiny K-means split + per-cluster Gaussians, sklearn-free.
    baselines.py:148-191.
    """
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(features_id), size=n_components, replace=False)
    centers = features_id[idx].copy()
    assign = np.zeros(len(features_id), dtype=np.int64)
    for _ in range(20):
        d = ((features_id[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assign = d.argmin(axis=1)
        new = np.stack([
            features_id[assign == k].mean(axis=0) if (assign == k).any()
            else centers[k]
            for k in range(n_components)
        ])
        if np.allclose(new, centers, atol=1e-6):
            centers = new
            break
        centers = new
    mus = []
    precs = []
    for k in range(n_components):
        cluster = features_id[assign == k]
        if len(cluster) < 2:
            cluster = features_id
        m, p = _fit_gaussian(cluster, reg=reg)
        mus.append(m)
        precs.append(p)
    return np.stack(mus), np.stack(precs)


def relative_mahalanobis(features_id: np.ndarray, features_test: np.ndarray,
                         reg: float = 0.1) -> np.ndarray:
    """Relative Mahalanobis distance. Higher = OOD.

    RMD = min_k M_component_k(x) - M_id(x). Ren et al. 2021
    (arXiv:2106.09022). baselines.py:194-216.
    """
    m_id = mahalanobis(features_id, features_test, reg=reg)
    bg_mus, bg_precs = _fit_background_gmm(features_id, n_components=2, reg=reg)
    comp_scores = []
    for mu, prec in zip(bg_mus, bg_precs, strict=True):
        X = features_test - mu
        comp_scores.append(np.einsum("ij,jk,ik->i", X, prec, X))
    comp_min = np.min(np.stack(comp_scores), axis=0)
    return (comp_min - m_id).astype(np.float64)


def knn_distance(features_id: np.ndarray, features_test: np.ndarray,
                 k: int = 50, normalize: bool = True) -> np.ndarray:
    """Distance to the k-th nearest ID neighbour. Higher = OOD.

    Sun et al. ICML 2022 (arXiv:2204.06507). L2-normalises by default.
    baselines.py:219-251.
    """
    if normalize:
        def _norm(x: np.ndarray) -> np.ndarray:
            n = np.linalg.norm(x, axis=1, keepdims=True)
            n = np.where(n == 0, 1.0, n)
            return x / n
        A = _norm(features_id.astype(np.float64))
        B = _norm(features_test.astype(np.float64))
    else:
        A = features_id.astype(np.float64)
        B = features_test.astype(np.float64)
    if k > len(A):
        k = len(A)
    out = np.empty(len(B), dtype=np.float64)
    a_norm = (A * A).sum(axis=1)
    chunk = 1024
    for i in range(0, len(B), chunk):
        b = B[i:i + chunk]
        b_norm = (b * b).sum(axis=1, keepdims=True)
        d2 = b_norm + a_norm[None, :] - 2.0 * b @ A.T
        d2 = np.clip(d2, 0.0, None)
        part = np.partition(d2, k - 1, axis=1)[:, k - 1]
        out[i:i + chunk] = np.sqrt(part)
    return out


def pca_residual(features_id, features_test, n_components: int = 32):
    """Second-order OOD score: residual energy outside the ID principal subspace.

    Fit PCA on ID features (mean + top-k right singular vectors). Score each test
    frame by the squared norm of the part NOT captured by the ID subspace. A
    collapsed/frozen embedding sits off the ID manifold's normal spread and leaves
    a residual the location baselines miss. Higher = more OOD.
    """
    fid = np.asarray(features_id, dtype=np.float64)
    ft = np.asarray(features_test, dtype=np.float64)
    mu = fid.mean(axis=0)
    Xc = fid - mu
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(n_components, vt.shape[0])
    basis = vt[:k]
    tc = ft - mu
    proj = tc @ basis.T @ basis
    resid = tc - proj
    return np.sum(resid * resid, axis=1)


def hotelling_t2_window(features_id, features_test, window: int = 20):
    """Second-order score: deviation of the windowed within-stream variance from
    the ID variance level. A collapse drives the trailing-window total variance
    toward zero; the absolute log-ratio against the ID variance is high for both
    collapse (too low) and erratic (too high) windows. Higher = more OOD.
    """
    fid = np.asarray(features_id, dtype=np.float64)
    ft = np.asarray(features_test, dtype=np.float64)
    id_var = np.trace(np.cov(fid.T)) if fid.shape[0] > 1 else 1.0
    id_var = max(id_var, 1e-12)
    T = ft.shape[0]
    out = np.zeros(T, dtype=np.float64)
    for t in range(T):
        lo = max(0, t - window + 1)
        win = ft[lo : t + 1]
        v = np.trace(np.cov(win.T)) if win.shape[0] > 1 else 0.0
        out[t] = abs(np.log((v + 1e-12) / id_var))
    return out


def temporal_rnd(features_id, features_test, window: int = 8,
                 hidden: int = 64, seed: int = 0):
    """RND (Burda et al. 2019) over a temporal window, numpy closed-form.

    Input = the flattened trailing `window` of frames (zero-padded at the start),
    capturing temporal structure a per-frame RND misses. A fixed random target
    net and a ridge-fit predictor are trained on ID windows; the test score is the
    predictor's squared error. Higher = more OOD.
    """
    rng = np.random.default_rng(seed)

    def windows(F):
        F = np.asarray(F, dtype=np.float64)
        T, d = F.shape
        W = np.zeros((T, window * d), dtype=np.float64)
        for t in range(T):
            lo = max(0, t - window + 1)
            chunk = F[lo : t + 1].reshape(-1)
            W[t, -chunk.size :] = chunk
        return W

    Wid = windows(features_id)
    Wt = windows(features_test)
    in_dim = Wid.shape[1]
    proj = rng.normal(size=(in_dim, hidden)) / np.sqrt(in_dim)
    target_id = np.tanh(Wid @ proj)
    target_t = np.tanh(Wt @ proj)
    lam = 1e-2
    A = Wid.T @ Wid + lam * np.eye(in_dim)
    B = np.linalg.solve(A, Wid.T @ target_id)
    pred_t = Wt @ B
    return np.sum((pred_t - target_t) ** 2, axis=1)
