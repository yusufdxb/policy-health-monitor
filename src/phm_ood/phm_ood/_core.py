"""Pure-Python OOD detector logic for phm_ood.

This module contains all decision logic for the rolling-spread OOD detector.
The rclpy lifecycle node (node.py) is a thin adapter over this class: it
receives PolicyEmbedding messages, calls OodCore.update(), and publishes the
returned DetectorVerdictData as a DetectorVerdict message. Tests target this
module directly without spinning a ROS graph.

OOD signal: low rolling spread = out of distribution.
The policy's hidden state collapses toward a single point when the policy
encounters OOD conditions (first documented in Phantom-Braking
``src/e6_detector.py`` for openpilot supercombo). A monitor watching the
trace of the rolling covariance (sum of per-dimension variances) fires when
the spread drops below a calibrated threshold.

Threshold direction: spread < threshold -> OOD (violating). This mirrors
Phantom-Braking e6_detector.py:calibrate_threshold, where the threshold is the
``percentile``-th percentile of the real-driving distribution, so in-distribution
drives stay ABOVE it with high probability.
"""

from __future__ import annotations

import numpy as np
from phm_core.calibration import calibrate_threshold, rolling_spread
from phm_core.detector import (
    ACTION_HOLD,
    ACTION_LOG_ONLY,
    ACTION_NONE,
    ACTION_STOP_AND_HOLD,
    DetectorVerdictData,
)
from phm_core.hysteresis import Hysteresis
from phm_core.severity import (
    DEGRADED_THRESHOLD,
    INTERVENE_THRESHOLD,
    STOP_THRESHOLD,
    normalize,
)

SOURCE = "phm_ood"


class OodCore:
    """Rolling-spread OOD detector, no ROS dependency.

    Maintains a rolling buffer of policy embedding vectors. Each call to
    :meth:`update` appends one vector, computes rolling spread when the buffer
    has filled, compares against the calibrated threshold (spread < threshold
    means OOD), and applies hysteresis before emitting a DetectorVerdictData.

    Parameters
    ----------
    window:
        Number of consecutive embedding frames to include in the rolling
        covariance. Must be >= 2. Default 30 (spec param ``window``).
    threshold:
        Calibrated spread threshold below which the detector fires.
        ``spread < threshold`` -> OOD. Default 0.0 (must be set via
        :meth:`set_threshold` or the constructor before use).
    hysteresis_count:
        Consecutive violating frames required before the verdict flips to
        ``violating=True``. Default 3 (spec param ``hysteresis_count``).
    compute_every:
        Frequency gate: only compute spread every N frames. 1 = every frame
        (default). A value > 1 re-uses the last verdict on intervening frames.
    embed_dim:
        Expected embedding dimension for input validation. 0 = skip validation.
    """

    def __init__(
        self,
        window: int = 30,
        threshold: float = 0.0,
        hysteresis_count: int = 3,
        compute_every: int = 1,
        embed_dim: int = 0,
    ) -> None:
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        if hysteresis_count < 1:
            raise ValueError(
                f"hysteresis_count must be >= 1, got {hysteresis_count}"
            )
        if compute_every < 1:
            raise ValueError(f"compute_every must be >= 1, got {compute_every}")

        self._window = window
        self._threshold = float(threshold)
        self._hysteresis = Hysteresis(hysteresis_count)
        self._compute_every = compute_every
        self._embed_dim = embed_dim

        # Rolling buffer: list of 1-D numpy arrays, capped at window length.
        self._buffer: list[np.ndarray] = []
        # Frame counter (counts every embedding received, not just computed).
        self._frame_count: int = 0
        # Last emitted verdict (re-used on frequency-gated skipped frames).
        self._last_verdict: DetectorVerdictData | None = None
        # Last computed raw spread (for logging).
        self._last_spread: float | None = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_threshold(self, threshold: float) -> None:
        """Set the calibrated spread threshold. Thread-unsafe (call once)."""
        self._threshold = float(threshold)

    def calibrate_from_data(
        self, in_dist_hidden: np.ndarray, percentile: float = 1.0
    ) -> float:
        """Compute and store the threshold from in-distribution embeddings.

        Args:
            in_dist_hidden: shape (T, D) array of in-distribution hidden states.
            percentile: percentile of the spread distribution to use as the
                threshold (default 1.0, the 1st percentile, so ~99% of real
                frames stay above it).

        Returns:
            The computed threshold (also stored in self._threshold).
        """
        spreads = rolling_spread(in_dist_hidden, self._window)
        thr = calibrate_threshold(spreads, percentile)
        self._threshold = thr
        return thr

    @property
    def threshold(self) -> float:
        """Current calibrated spread threshold."""
        return self._threshold

    @property
    def window(self) -> int:
        """Rolling-spread window length."""
        return self._window

    @property
    def last_spread(self) -> float | None:
        """Most recently computed rolling spread, or None if not yet computed."""
        return self._last_spread

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(
        self, embedding: np.ndarray, policy_id: str = ""
    ) -> DetectorVerdictData:
        """Process one embedding frame and return a verdict.

        Args:
            embedding: 1-D float array, the policy's hidden state vector.
            policy_id: identifier from the PolicyEmbedding message (used in
                reason strings only).

        Returns:
            A DetectorVerdictData with source='phm_ood'. The verdict is always
            returned (never None) so the arbiter always has a fresh signal.
        """
        emb = np.asarray(embedding, dtype=np.float64).ravel()

        # Dimension validation (skipped when embed_dim == 0).
        if self._embed_dim > 0 and emb.shape[0] != self._embed_dim:
            # Return a degraded verdict with a clear reason; don't crash.
            return DetectorVerdictData(
                source=SOURCE,
                score=0.5,
                violating=False,
                reason=(
                    f"dim mismatch: expected {self._embed_dim},"
                    f" got {emb.shape[0]}"
                ),
                suggested_action=ACTION_NONE,
            )

        # Update rolling buffer.
        self._buffer.append(emb)
        if len(self._buffer) > self._window:
            self._buffer.pop(0)

        self._frame_count += 1

        # Buffer not yet full: return an OK verdict.
        if len(self._buffer) < self._window:
            return self._ok_verdict(
                reason=f"warming up: {len(self._buffer)}/{self._window} frames"
            )

        # Frequency gate: re-use last verdict on non-compute frames.
        if self._last_verdict is not None and (
            self._frame_count % self._compute_every != 0
        ):
            return self._last_verdict

        # Compute rolling spread from the current buffer.
        hidden = np.stack(self._buffer, axis=0)  # (window, D)
        spread_series = rolling_spread(hidden, self._window)
        # Only the last element is valid (window == window), rest are NaN.
        spread = float(spread_series[-1])
        self._last_spread = spread

        raw_violating = spread < self._threshold
        fired = self._hysteresis.observe(raw_violating)

        verdict = self._make_verdict(spread, raw_violating, fired, policy_id)
        self._last_verdict = verdict
        return verdict

    # ------------------------------------------------------------------
    # Verdict construction
    # ------------------------------------------------------------------

    def _ok_verdict(self, reason: str = "ok") -> DetectorVerdictData:
        """Return a healthy, non-violating verdict."""
        return DetectorVerdictData(
            source=SOURCE,
            score=0.0,
            violating=False,
            reason=reason,
            suggested_action=ACTION_NONE,
        )

    def _make_verdict(
        self,
        spread: float,
        raw_violating: bool,
        fired: bool,
        policy_id: str,
    ) -> DetectorVerdictData:
        """Build a DetectorVerdictData from the latest spread and hysteresis.

        Score mapping (threshold direction: low spread = bad):
            - spread >= threshold: healthy (score 0.0 -> 0.0, no severity)
            - spread < threshold: OOD; score = normalize(spread,
                healthy=threshold, worst=0.0)
              so spread == threshold -> 0.0, spread == 0.0 -> 1.0.

        Action banding follows severity.py thresholds (LOCKED decision 3):
            - score < DEGRADED_THRESHOLD (0.25): below the severity floor; NOT a
              violation (LOCKED decision 2). violating=False, action NONE.
            - DEGRADED <= score < INTERVENE (0.25..0.50) and fired -> LOG_ONLY
            - INTERVENE <= score < STOP (0.50..0.80) and fired     -> HOLD
            - score >= STOP_THRESHOLD (0.80) and fired             -> STOP_AND_HOLD
        """
        pid_str = f"[{policy_id}] " if policy_id else ""
        n_consec = self._hysteresis.count

        if not raw_violating:
            # In-distribution.
            reason = (
                f"{pid_str}ood: rolling-spread {spread:.4f}"
                f" >= thr {self._threshold:.4f}"
            )
            return DetectorVerdictData(
                source=SOURCE,
                score=0.0,
                violating=False,
                reason=reason,
                suggested_action=ACTION_NONE,
            )

        # OOD: compute normalized severity.
        if self._threshold > 0.0:
            # normalize(raw, healthy=threshold, worst=0) -> high score near 0.
            score = normalize(spread, healthy=self._threshold, worst=0.0)
        else:
            # Degenerate threshold; any OOD is max score.
            score = 1.0

        reason = (
            f"{pid_str}ood: rolling-spread {spread:.4f}"
            f" < thr {self._threshold:.4f}"
            f" for {n_consec} frame(s)"
        )

        if not fired:
            # Pre-hysteresis: violating raw but not yet confirmed.
            return DetectorVerdictData(
                source=SOURCE,
                score=score,
                violating=False,
                reason=reason + " (pre-hysteresis)",
                suggested_action=ACTION_NONE,
            )

        # VIOLATING FLOOR + TOTALITY (LOCKED decision 2): a post-hysteresis score
        # below the DEGRADED floor is a zero-severity signal. Do NOT emit it as a
        # violation: a violating=True verdict with a STATE_OK-band score makes the
        # arbiter worst-wins map non-total (it would carry source/score yet resolve
        # to OK). Report it as a non-violating pass so the arbiter ignores it.
        if score < DEGRADED_THRESHOLD:
            return DetectorVerdictData(
                source=SOURCE,
                score=score,
                violating=False,
                reason=reason + " (below severity floor)",
                suggested_action=ACTION_NONE,
            )

        # Post-hysteresis confirmed OOD, action banding (LOCKED decision 3):
        #   [STOP, 1.0]        -> STOP_AND_HOLD
        #   [INTERVENE, STOP)  -> HOLD
        #   [DEGRADED, INTERVENE) -> LOG_ONLY
        if score >= STOP_THRESHOLD:
            action = ACTION_STOP_AND_HOLD
            band = "stop"
        elif score >= INTERVENE_THRESHOLD:
            action = ACTION_HOLD
            band = "intervene"
        else:
            action = ACTION_LOG_ONLY
            band = "degraded"

        reason = reason + f" [{band}]"

        return DetectorVerdictData(
            source=SOURCE,
            score=score,
            violating=True,
            reason=reason,
            suggested_action=action,
        )
