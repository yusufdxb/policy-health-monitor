"""RND-style novelty detector (Random Network Distillation).

RND (Burda et al. ICLR 2019, arXiv:1810.12894) trains a predictor network to
match the output of a fixed, randomly initialised target network on the
in-distribution data. The per-sample prediction error (target minus predictor)
is small on data the predictor has seen the manifold of, and large on
out-of-distribution data, so the squared error is an OOD novelty score with the
standard convention HIGHER = more OOD.

This is the 4th method required by the benchmark spec, alongside the three
phantom-braking baselines (mahalanobis / relative_mahalanobis / knn).

Two backends, same math, same higher-is-OOD output:

- ``rnd_numpy`` (default, runs in the numpy-only benchmark venv): the target is
  a fixed random linear-ReLU projection f_T(x) = ReLU(x W_T) V_T; the predictor
  is a linear map fit IN CLOSED FORM by ridge least squares on the ID features
  to reproduce the target outputs. This is the "fixed random target vs a small
  trained predictor" form of RND with a convex (closed-form) predictor, so it
  is deterministic and needs no gradient descent. Reconstruction error on the
  test stream is the novelty score.

- ``rnd_torch`` (run out-of-process via /usr/bin/python3, which has torch
  2.11.0+cu128): a fixed random MLP target and a separately initialised MLP
  predictor trained by Adam to match it on the ID features, then evaluated on
  the test stream. Used to corroborate the numpy form with a real gradient-
  trained predictor. The harness shells out to scripts/rnd_torch_worker.py.

The numpy form is the one wired into the head-to-head table so the whole
harness runs in one interpreter; the torch form is reported as a corroborating
datapoint with its real output pasted.
"""

from __future__ import annotations

import numpy as np


def rnd_numpy(features_id: np.ndarray, features_test: np.ndarray,
              proj_dim: int = 128, reg: float = 1e-2,
              seed: int = 0) -> np.ndarray:
    """RND novelty score with a fixed random target and a closed-form predictor.

    Args:
        features_id: (N_id, D) in-distribution features the predictor is fit on.
        features_test: (N_test, D) features to score.
        proj_dim: width of the random target projection.
        reg: ridge regularisation for the closed-form predictor solve.
        seed: RNG seed for the fixed random target network.

    Returns:
        (N_test,) per-frame squared prediction error. Higher = more OOD.
    """
    features_id = np.asarray(features_id, dtype=np.float64)
    features_test = np.asarray(features_test, dtype=np.float64)
    d = features_id.shape[1]
    rng = np.random.RandomState(seed)

    # Fixed random target network: ReLU(X W_T) V_T, never trained.
    w_t = rng.randn(d, proj_dim) / np.sqrt(d)
    v_t = rng.randn(proj_dim, proj_dim) / np.sqrt(proj_dim)

    def target(x: np.ndarray) -> np.ndarray:
        h = np.maximum(x @ w_t, 0.0)
        return h @ v_t

    t_id = target(features_id)

    # Standardise the input on ID stats so the predictor solve is conditioned.
    mu = features_id.mean(axis=0)
    sd = features_id.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)

    def feat(x: np.ndarray) -> np.ndarray:
        z = (x - mu) / sd
        # Augment with a bias column.
        return np.hstack([z, np.ones((len(z), 1))])

    phi_id = feat(features_id)
    # Closed-form ridge predictor:  W = (Phi^T Phi + reg I)^-1 Phi^T T_id.
    p = phi_id.shape[1]
    gram = phi_id.T @ phi_id + reg * np.eye(p)
    w_pred = np.linalg.solve(gram, phi_id.T @ t_id)

    phi_test = feat(features_test)
    pred_test = phi_test @ w_pred
    t_test = target(features_test)
    err = ((t_test - pred_test) ** 2).mean(axis=1)
    return err.astype(np.float64)
