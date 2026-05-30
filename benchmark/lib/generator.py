"""In-distribution vs OOD embedding-stream generator for the benchmark.

Reuses the (in_dist, ood) generator concept from
``/home/yusuf/Projects/policy-health-monitor/src/phm_sim/phm_sim/_sim_core.py:25-65``
(``generate_embeddings``): the in-distribution stream is a stable multivariate
Gaussian with a healthy rolling spread, and the OOD stream is a collapsed,
near-zero-variance embedding. This module generalises it for the benchmark:

- N (n_frames) and dim are configurable, as required by the spec.
- ID embeddings are temporally correlated (an AR(1) drift over a non-isotropic
  covariance) so the stream looks like a real policy hidden state rather than
  iid noise, which makes the rolling-spread signal meaningful.
- Two OOD modes:
    * "collapse": low-variance frozen embedding (phm_sim's original OOD mode);
      this is the failure the PHM rolling-spread detector is designed to catch.
    * "shift": a mean-shifted, rescaled Gaussian in a different region of the
      embedding space (location shift), the failure the Mahalanobis / KNN
      baselines are designed to catch.

The harness scores ID frames as label 0 and OOD frames as label 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StreamSpec:
    """Configuration for one ID-vs-OOD embedding stream."""

    dim: int = 64
    n_id: int = 600
    n_ood: int = 600
    ood_mode: str = "collapse"          # "collapse" | "shift"
    in_dist_scale: float = 1.0
    ood_scale: float = 0.02             # collapse: near-zero variance
    ood_shift: float = 4.0              # shift: mean offset in sigma units
    ar_rho: float = 0.85                # AR(1) temporal correlation of ID stream
    seed: int = 42


def _random_spd(dim: int, rng: np.random.Generator) -> np.ndarray:
    """A non-isotropic symmetric positive-definite covariance for the ID stream."""
    a = rng.normal(size=(dim, dim)) / np.sqrt(dim)
    cov = a @ a.T + 0.5 * np.eye(dim)
    return cov


def _ar1_gaussian(n: int, mean: np.ndarray, cov: np.ndarray, rho: float,
                  rng: np.random.Generator) -> np.ndarray:
    """AR(1) process x_t = mean + rho*(x_{t-1}-mean) + sqrt(1-rho^2)*eps_t.

    Stationary covariance is ``cov``; rho controls temporal correlation. With
    rho=0 this is iid N(mean, cov), matching phm_sim's iid generator.
    """
    dim = mean.shape[0]
    chol = np.linalg.cholesky(cov)
    innov_scale = np.sqrt(max(1.0 - rho * rho, 1e-9))
    out = np.empty((n, dim), dtype=np.float64)
    x = mean + chol @ rng.normal(size=dim)
    for t in range(n):
        eps = chol @ rng.normal(size=dim)
        x = mean + rho * (x - mean) + innov_scale * eps
        out[t] = x
    return out


def generate_stream(spec: StreamSpec) -> tuple[np.ndarray, np.ndarray]:
    """Return (id_frames, ood_frames).

    id_frames:  (n_id, dim)  stable, temporally correlated, healthy spread.
    ood_frames: (n_ood, dim) collapsed or shifted, per ``spec.ood_mode``.
    """
    rng = np.random.default_rng(spec.seed)
    mean = np.zeros(spec.dim)
    cov = (spec.in_dist_scale ** 2) * _random_spd(spec.dim, rng)

    id_frames = _ar1_gaussian(spec.n_id, mean, cov, spec.ar_rho, rng)

    if spec.ood_mode == "collapse":
        # Frozen embedding near a single point: variance collapses by
        # (ood_scale / in_dist_scale)^2, the phm_sim collapse failure.
        anchor = id_frames[-1]            # collapse around the last ID state
        ood_frames = anchor + rng.normal(
            scale=spec.ood_scale, size=(spec.n_ood, spec.dim)
        )
    elif spec.ood_mode == "shift":
        # Mean-shifted, rescaled Gaussian in a different region of the space.
        sigma = np.sqrt(np.diag(cov))
        shifted_mean = mean + spec.ood_shift * sigma
        ood_frames = _ar1_gaussian(
            spec.n_ood, shifted_mean, 0.5 * cov, spec.ar_rho, rng
        )
    else:
        raise ValueError(f"unknown ood_mode {spec.ood_mode!r}")

    return id_frames.astype(np.float64), ood_frames.astype(np.float64)


def rolling_spread_trace(frames: np.ndarray, window: int) -> float:
    """Mean per-frame trace-of-rolling-covariance, a scalar summary of spread.

    Used by tests to assert ID spread > OOD spread. Matches the math of
    phm_core.calibration.rolling_spread (windowed trace of variance) averaged
    over valid frames.
    """
    T, _ = frames.shape
    vals = []
    for t in range(window, T + 1):
        vals.append(float(np.var(frames[t - window:t], axis=0).sum()))
    return float(np.mean(vals)) if vals else float("nan")
