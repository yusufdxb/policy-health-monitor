"""Unit tests for phm_sim._sim_core (pure Python, no rclpy).

These tests verify:
1. generate_embeddings returns correct shapes and dtypes.
2. The in-distribution stream has strictly higher mean rolling spread than
   the OOD stream (the core invariant the OOD pipeline depends on).
3. A threshold calibrated on in-distribution data correctly classifies the
   in-dist stream as mostly OK and the OOD stream as mostly violating.
4. EmbeddingStream: phase transitions, trigger_ood, reset, reproducibility.
5. EmbeddingStream.next_frame produces the right shape and dtype.

No rclpy import anywhere in this file. Import guard: the top-level
phm_sim.__init__ re-exports from _sim_core directly, so 'from phm_sim import
...' also avoids touching the rclpy node module.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

# Safety guard: rclpy must not be pulled in by anything in this test file.
_rclpy_before = "rclpy" in sys.modules


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_no_rclpy_imported() -> None:
    """Fail if rclpy was imported as a side-effect of our test imports."""
    if not _rclpy_before and "rclpy" in sys.modules:
        pytest.fail("rclpy was imported by a phm_sim._sim_core import, violating the no-ROS rule.")


# ---------------------------------------------------------------------------
# Imports under test (after guard is set)
# ---------------------------------------------------------------------------

from phm_core.calibration import calibrate_threshold, rolling_spread  # noqa: E402

from phm_sim._sim_core import EmbeddingStream, generate_embeddings  # noqa: E402


# Verify import side-effects.
def test_no_rclpy_imported_by_sim_core() -> None:
    """Importing phm_sim._sim_core must not pull in rclpy."""
    _assert_no_rclpy_imported()


# ---------------------------------------------------------------------------
# generate_embeddings
# ---------------------------------------------------------------------------

class TestGenerateEmbeddings:
    def test_output_shapes(self) -> None:
        in_d, ood = generate_embeddings(dim=32, n_frames=50)
        assert in_d.shape == (50, 32)
        assert ood.shape == (50, 32)

    def test_output_dtype(self) -> None:
        in_d, ood = generate_embeddings(dim=16, n_frames=20)
        assert in_d.dtype == np.float64
        assert ood.dtype == np.float64

    def test_in_dist_has_higher_variance_than_ood(self) -> None:
        """Per-dimension variance of in_dist must be >> ood at the default scales."""
        in_d, ood = generate_embeddings(dim=64, n_frames=500, seed=0)
        # in_dist_scale=1.0, ood_scale=0.01 -> variance ratio ~10000x.
        in_var = float(np.var(in_d))
        ood_var = float(np.var(ood))
        assert in_var > ood_var * 100, (
            f"Expected in_dist variance >> ood variance, got {in_var:.4f} vs {ood_var:.6f}"
        )

    def test_reproducibility_same_seed(self) -> None:
        in1, ood1 = generate_embeddings(seed=7)
        in2, ood2 = generate_embeddings(seed=7)
        np.testing.assert_array_equal(in1, in2)
        np.testing.assert_array_equal(ood1, ood2)

    def test_different_seeds_differ(self) -> None:
        in1, _ = generate_embeddings(seed=1)
        in2, _ = generate_embeddings(seed=2)
        assert not np.array_equal(in1, in2)


# ---------------------------------------------------------------------------
# Rolling spread invariant (the core assertion the OOD pipeline depends on)
# ---------------------------------------------------------------------------

class TestRollingSpreadInvariant:
    """Assert that in-distribution rolling spread > OOD rolling spread.

    This is the invariant the phm_ood node relies on: calibrate a threshold
    on in-distribution data, and OOD frames consistently fall below it.
    """

    WINDOW = 20   # window size used by phm_ood (matches calibration default)
    DIM = 64
    N_FRAMES = 300

    def _spreads(self, arr: np.ndarray) -> np.ndarray:
        """Strip NaN prefix from rolling_spread output."""
        s = rolling_spread(arr, self.WINDOW)
        return s[~np.isnan(s)]

    def test_mean_in_dist_spread_greater_than_ood(self) -> None:
        """Mean in-dist rolling spread must exceed mean OOD rolling spread."""
        in_d, ood = generate_embeddings(
            dim=self.DIM, n_frames=self.N_FRAMES, seed=42
        )
        in_spreads = self._spreads(in_d)
        ood_spreads = self._spreads(ood)
        assert len(in_spreads) > 0, "in-dist produced no valid spread values"
        assert len(ood_spreads) > 0, "OOD produced no valid spread values"
        mean_in = float(np.mean(in_spreads))
        mean_ood = float(np.mean(ood_spreads))
        assert mean_in > mean_ood, (
            f"In-dist mean spread ({mean_in:.4f}) must exceed OOD mean spread ({mean_ood:.6f})"
        )

    def test_calibrated_threshold_separates_phases(self) -> None:
        """A threshold calibrated on in-dist data must classify OOD frames as violating.

        Uses phm_core.calibration.calibrate_threshold (ported from
        Phantom-Braking e6_detector.py:26-31).

        Expectations:
        - >= 90% of in-dist frames are above the threshold (low FPR).
        - >= 90% of OOD frames are below the threshold (high TPR).
        """
        in_d, ood = generate_embeddings(
            dim=self.DIM, n_frames=self.N_FRAMES, seed=42
        )
        in_spreads = self._spreads(in_d)
        ood_spreads = self._spreads(ood)

        # Calibrate on in-distribution data, 1st percentile (same as Phantom E6).
        thr = calibrate_threshold(in_spreads, percentile=1.0)

        # Low FPR: most in-dist frames should be above threshold (healthy).
        in_dist_above_thr = float(np.mean(in_spreads >= thr))
        assert in_dist_above_thr >= 0.90, (
            f"Only {in_dist_above_thr:.1%} of in-dist frames above threshold {thr:.4f}; "
            "expected >= 90%"
        )

        # High TPR: most OOD frames should be below threshold (violating).
        ood_below_thr = float(np.mean(ood_spreads < thr))
        assert ood_below_thr >= 0.90, (
            f"Only {ood_below_thr:.1%} of OOD frames below threshold {thr:.4f}; "
            "expected >= 90%"
        )

    def test_threshold_strictly_positive(self) -> None:
        """Calibrated threshold must be > 0 (non-degenerate in-dist spread)."""
        in_d, _ = generate_embeddings(dim=self.DIM, n_frames=self.N_FRAMES, seed=42)
        in_spreads = self._spreads(in_d)
        thr = calibrate_threshold(in_spreads, percentile=1.0)
        assert thr > 0.0, f"Expected positive threshold, got {thr}"

    def test_spread_ratio_exceeds_100x(self) -> None:
        """Mean in-dist spread / mean OOD spread must be > 100.

        At in_dist_scale=1.0, ood_scale=0.01 the theoretical ratio is
        (1.0/0.01)^2 = 10000. We assert a conservative lower bound.
        """
        in_d, ood = generate_embeddings(
            dim=self.DIM, n_frames=self.N_FRAMES, seed=42
        )
        ratio = float(np.mean(self._spreads(in_d))) / float(np.mean(self._spreads(ood)))
        assert ratio > 100.0, (
            f"Spread ratio {ratio:.1f} too low; expected > 100 with default scales"
        )


# ---------------------------------------------------------------------------
# EmbeddingStream
# ---------------------------------------------------------------------------

class TestEmbeddingStream:
    def test_frame_shape_and_dtype(self) -> None:
        stream = EmbeddingStream(dim=32, n_in_dist=10)
        frame = stream.next_frame()
        assert frame.shape == (32,)
        assert frame.dtype == np.float64

    def test_phase_transition_at_n_in_dist(self) -> None:
        stream = EmbeddingStream(dim=8, n_in_dist=5)
        for _ in range(5):
            assert not stream.is_ood_phase, "should be in_dist before n_in_dist frames"
            stream.next_frame()
        assert stream.is_ood_phase, "should switch to OOD after n_in_dist frames"

    def test_frame_index_increments(self) -> None:
        stream = EmbeddingStream(dim=4, n_in_dist=10)
        for i in range(7):
            assert stream.frame_index == i
            stream.next_frame()
        assert stream.frame_index == 7

    def test_trigger_ood_forces_phase(self) -> None:
        stream = EmbeddingStream(dim=8, n_in_dist=100)
        assert not stream.is_ood_phase
        stream.trigger_ood()
        assert stream.is_ood_phase

    def test_trigger_ood_idempotent(self) -> None:
        """Calling trigger_ood when already in OOD phase is a no-op."""
        stream = EmbeddingStream(dim=8, n_in_dist=2)
        stream.next_frame()
        stream.next_frame()
        assert stream.is_ood_phase
        idx_before = stream.frame_index
        stream.trigger_ood()
        assert stream.frame_index == idx_before

    def test_reset_restarts_stream(self) -> None:
        stream = EmbeddingStream(dim=8, n_in_dist=5)
        for _ in range(10):
            stream.next_frame()
        assert stream.is_ood_phase
        stream.reset()
        assert stream.frame_index == 0
        assert not stream.is_ood_phase

    def test_reproducibility_same_seed(self) -> None:
        s1 = EmbeddingStream(dim=16, n_in_dist=10, seed=99)
        s2 = EmbeddingStream(dim=16, n_in_dist=10, seed=99)
        for _ in range(20):
            np.testing.assert_array_equal(s1.next_frame(), s2.next_frame())

    def test_different_seeds_differ(self) -> None:
        s1 = EmbeddingStream(dim=16, n_in_dist=10, seed=1)
        s2 = EmbeddingStream(dim=16, n_in_dist=10, seed=2)
        frames1 = [s1.next_frame() for _ in range(5)]
        frames2 = [s2.next_frame() for _ in range(5)]
        # At least one frame pair must differ.
        assert any(not np.array_equal(a, b) for a, b in zip(frames1, frames2, strict=True))

    def test_invalid_n_in_dist_raises(self) -> None:
        with pytest.raises(ValueError, match="n_in_dist"):
            EmbeddingStream(dim=8, n_in_dist=0)

    def test_in_dist_frames_have_higher_variance_than_ood(self) -> None:
        """Frames from each phase must match their expected scale."""
        n = 200
        stream = EmbeddingStream(
            dim=64, n_in_dist=n, in_dist_scale=1.0, ood_scale=0.01, seed=0
        )
        in_frames = np.stack([stream.next_frame() for _ in range(n)])
        ood_frames = np.stack([stream.next_frame() for _ in range(n)])
        in_var = float(np.var(in_frames))
        ood_var = float(np.var(ood_frames))
        assert in_var > ood_var * 100, (
            f"In-dist var {in_var:.4f} not >> OOD var {ood_var:.6f}"
        )

    def test_stream_rolling_spread_in_dist_greater_than_ood(self) -> None:
        """EmbeddingStream rolling spread invariant: same as generate_embeddings test
        but using the stateful stream interface."""
        n = 300
        window = 20
        stream = EmbeddingStream(dim=64, n_in_dist=n, seed=42)
        in_frames = np.stack([stream.next_frame() for _ in range(n)])
        ood_frames = np.stack([stream.next_frame() for _ in range(n)])

        in_spreads = rolling_spread(in_frames, window)
        ood_spreads = rolling_spread(ood_frames, window)
        in_spreads = in_spreads[~np.isnan(in_spreads)]
        ood_spreads = ood_spreads[~np.isnan(ood_spreads)]

        assert float(np.mean(in_spreads)) > float(np.mean(ood_spreads)), (
            "EmbeddingStream: in-dist mean rolling spread must exceed OOD"
        )
