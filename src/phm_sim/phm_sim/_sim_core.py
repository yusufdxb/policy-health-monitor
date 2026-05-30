"""Pure-Python embedding generator for phm_sim.

No rclpy dependency. Returns (in_dist_array, ood_array) or streams frames
one at a time. The rclpy node in embedding_publisher_node.py is a thin adapter
over this module.

Design intent (spec section 4 / Track A):
- In-distribution phase: samples from a stable multivariate Gaussian with a
  non-degenerate covariance. The rolling spread of this stream is high and
  consistently above the calibrated threshold, producing an OK verdict.
- OOD (collapse) phase: samples clustered near a single point (near-zero
  variance). The rolling spread collapses below the calibrated threshold,
  driving the OOD detector to INTERVENE and the arbiter to match.

Both phases use the same random seed convention so tests are deterministic.
The in_dist_scale and ood_scale parameters let the caller tune the separation
between phases.
"""

from __future__ import annotations

import numpy as np


def generate_embeddings(
    *,
    dim: int = 64,
    n_frames: int = 200,
    in_dist_scale: float = 1.0,
    ood_scale: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a batch of in-distribution and OOD embedding arrays.

    Both arrays have shape (n_frames, dim). Useful for offline tests and
    spread-comparison assertions.

    The in-distribution array is drawn from N(0, in_dist_scale * I_dim) so the
    rolling trace-of-variance is approximately dim * in_dist_scale^2 per frame.
    The OOD array is drawn from N(0, ood_scale * I_dim), collapsing the rolling
    spread by a factor of (ood_scale / in_dist_scale)^2.

    Args:
        dim: embedding dimensionality.
        n_frames: number of time steps in each array.
        in_dist_scale: std-dev of the in-distribution Gaussian; higher = more
            spread = healthier signal.
        ood_scale: std-dev of the OOD Gaussian; keep << in_dist_scale so the
            rolling spread collapses clearly below the calibrated threshold.
        seed: NumPy RNG seed for reproducibility.

    Returns:
        Tuple (in_dist, ood):
            in_dist: float64 array of shape (n_frames, dim), high variance.
            ood:     float64 array of shape (n_frames, dim), near-zero variance.

    Example:
        >>> in_d, ood = generate_embeddings(dim=32, n_frames=100)
        >>> in_d.shape
        (100, 32)
    """
    rng = np.random.default_rng(seed)
    in_dist = rng.normal(loc=0.0, scale=in_dist_scale, size=(n_frames, dim))
    ood = rng.normal(loc=0.0, scale=ood_scale, size=(n_frames, dim))
    return in_dist.astype(np.float64), ood.astype(np.float64)


class EmbeddingStream:
    """Stateful frame-by-frame embedding generator for the publisher node.

    Advances through two phases:
    1. in_dist phase (first ``n_in_dist`` frames): stable multivariate Gaussian.
    2. ood phase (subsequent frames, loops): near-zero-variance collapsed state.

    Call :meth:`next_frame` once per timer tick to get the next embedding vector.

    Args:
        dim: embedding dimensionality.
        n_in_dist: number of frames in the in-distribution phase before
            switching to the OOD phase. Must be >= 1.
        in_dist_scale: std-dev of the in-distribution Gaussian.
        ood_scale: std-dev of the OOD (collapse) Gaussian.
        policy_id: label embedded in each frame, forwarded to PolicyEmbedding.
        seed: RNG seed.
    """

    def __init__(
        self,
        *,
        dim: int = 64,
        n_in_dist: int = 100,
        in_dist_scale: float = 1.0,
        ood_scale: float = 0.01,
        policy_id: str = "phm_sim",
        seed: int = 42,
    ) -> None:
        if n_in_dist < 1:
            raise ValueError(f"n_in_dist must be >= 1, got {n_in_dist}")
        self._dim = dim
        self._n_in_dist = n_in_dist
        self._in_dist_scale = float(in_dist_scale)
        self._ood_scale = float(ood_scale)
        self._policy_id = policy_id
        self._rng = np.random.default_rng(seed)
        self._frame_index: int = 0

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        return self._dim

    @property
    def policy_id(self) -> str:
        """Policy identifier forwarded to PolicyEmbedding.policy_id."""
        return self._policy_id

    @property
    def frame_index(self) -> int:
        """Zero-based index of the next frame to be generated."""
        return self._frame_index

    @property
    def is_ood_phase(self) -> bool:
        """True once the in-distribution phase has ended."""
        return self._frame_index >= self._n_in_dist

    def next_frame(self) -> np.ndarray:
        """Return the next embedding vector of shape (dim,).

        In the in-distribution phase: sample from N(0, in_dist_scale * I).
        In the OOD phase:             sample from N(0, ood_scale * I).

        The frame index is incremented on each call, so
        :attr:`is_ood_phase` flips to True at frame ``n_in_dist``.
        """
        if self._frame_index < self._n_in_dist:
            vec = self._rng.normal(
                loc=0.0, scale=self._in_dist_scale, size=(self._dim,)
            )
        else:
            vec = self._rng.normal(
                loc=0.0, scale=self._ood_scale, size=(self._dim,)
            )
        self._frame_index += 1
        return vec.astype(np.float64)

    def reset(self) -> None:
        """Reset the stream to frame 0 (restarts both phases)."""
        self._frame_index = 0

    def trigger_ood(self) -> None:
        """Force-transition to the OOD phase immediately, regardless of frame count.

        The publisher node calls this when a ROS service or parameter trigger
        is received. Does nothing if already in the OOD phase.
        """
        if self._frame_index < self._n_in_dist:
            self._frame_index = self._n_in_dist
