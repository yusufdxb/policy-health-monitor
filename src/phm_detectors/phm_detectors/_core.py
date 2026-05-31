"""Pure-Python adapter layer: bridges BlackBoxRS detector patterns into phm_core
Detector verdicts.

blackboxrs is NOT imported here because it is not installed in the PHM venv.
Instead, the core logic from each BlackBoxRS detector is ported verbatim with a
comment citing the source file and line range. The overall structure mirrors
BlackBoxRS ``blackboxrs/anomaly_engine/detectors/`` but returns
``DetectorVerdictData`` objects (phm_core) instead of ``BlackBoxEvent``
anomalies (BlackBoxRS).

Three adapters are provided:

- ``FrequencyDropAdapter``  -- topic rate below tolerance (ported from
  BlackBoxRS ``anomaly_engine/detectors/frequency.py:52-152``).
- ``StaticThresholdAdapter`` -- cpu / mem / temp above static limit (ported
  from BlackBoxRS ``anomaly_engine/detectors/threshold.py:75-154``).
- ``DeadTopicAdapter``  -- no messages for N seconds (ported from BlackBoxRS
  ``anomaly_engine/detectors/dead_topic.py:50-127``).

Each adapter is a concrete subclass of ``phm_core.Detector`` (the PHM ABC) and
holds its own ``phm_core.Hysteresis`` instance. The rclpy node in
``phm_detectors_node.py`` calls ``adapter.update(sample)`` on every graph-stats
tick and publishes the returned verdict (if any) to ``/phm/verdicts``.

Sample types accepted by each adapter (plain dataclasses, no ROS):

- ``FrequencySample(topic: str, frequency_hz: float)``
- ``ThresholdSample(metric: str, value: float, threshold: float)``
- ``DeadTopicSample(topic: str, last_seen_sec: float, now_sec: float)``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from phm_core.calibration import calibrate_threshold, rolling_spread
from phm_core.detector import (
    ACTION_NONE,
    ACTION_STOP_AND_HOLD,
    Detector,
    DetectorVerdictData,
)
from phm_core.hysteresis import Hysteresis
from phm_core.severity import classify, normalize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sample dataclasses (pure Python, no rclpy)
# ---------------------------------------------------------------------------


@dataclass
class FrequencySample:
    """One frequency observation for a single topic.

    Mirrors the data dict fields extracted by BlackBoxRS FrequencyDetector
    (``frequency.py:82-83``): ``topic`` and ``frequency_hz``.
    """

    topic: str
    frequency_hz: float


@dataclass
class ThresholdSample:
    """One system-metric observation.

    Mirrors the (rule.data_key, value, threshold) triple extracted by BlackBoxRS
    ThresholdDetector (``threshold.py:100-105``).
    """

    metric: str
    value: float
    threshold: float


@dataclass
class RecurrentSpreadSample:
    """One policy hidden-state frame for the recurrent-temporal-spread detector.

    ``embedding`` is the policy's internal recurrent feature for a single frame
    (1-D, e.g. the 512-D supercombo recurrent vector). The adapter buffers these
    frames and watches the trace of their rolling covariance.

    ``topic`` lets the node route only the embedding stream this detector
    watches, mirroring the topic-match guard the other adapters use.
    """

    topic: str
    embedding: np.ndarray


@dataclass
class DeadTopicSample:
    """A clock tick that carries the last-seen timestamp for one topic.

    The rclpy node builds this from the ROS graph stats and the current
    ``rclpy.clock.Clock`` time. The ``DeadTopicAdapter`` only needs these
    two floats to compute elapsed silence.
    """

    topic: str
    last_seen_sec: float
    now_sec: float


# ---------------------------------------------------------------------------
# FrequencyDropAdapter
# ---------------------------------------------------------------------------

_LEARNING_SAMPLES = 10  # ported from BlackBoxRS frequency.py:20


class FrequencyDropAdapter(Detector):
    """Fires when topic frequency drops below a tolerance floor.

    Logic ported from BlackBoxRS
    ``blackboxrs/anomaly_engine/detectors/frequency.py:52-152``.

    Two phases per topic:

    1. Learning: collects ``_LEARNING_SAMPLES`` readings and averages them to
       establish a baseline. Verdicts are suppressed during learning.
    2. Monitoring: checks every reading against
       ``baseline * (1 - tolerance_percent / 100)``. If the rate falls below the
       floor for ``min_consecutive`` consecutive samples, fires a violating
       verdict. A healthy sample resets the counter.

    Score is normalized on [0, 1] where 0 means rate == baseline and 1 means
    rate == 0 Hz. Scores below 0.5 map to DEGRADED (LOG_ONLY); at or above 0.5
    they escalate to INTERVENE/STOP per ``phm_core.severity.classify``.

    Args:
        target_topic: the ROS topic this adapter monitors (used as
            ``Detector.target_topic``; actual routing is done by the node).
        tolerance_percent: frequency must stay above
            ``baseline * (1 - tolerance_percent/100)`` to be healthy.
            Ported from BlackBoxRS ``FrequencyConfig.tolerance_percent``.
        min_consecutive: hysteresis window. A verdict fires only after this
            many consecutive violating samples. Ported from BlackBoxRS
            ``FrequencyConfig.min_consecutive_samples``.
    """

    def __init__(
        self,
        target_topic: str,
        tolerance_percent: float = 20.0,
        min_consecutive: int = 2,
    ) -> None:
        self.name = f"freq:{target_topic}"
        self.target_topic = target_topic
        self._tolerance_pct = tolerance_percent
        self._hysteresis = Hysteresis(min_consecutive)
        # Per-topic learning buffers (ported from frequency.py:55-56).
        self._samples: list[float] = []
        self._baseline: float | None = None

    def update(self, sample: FrequencySample) -> DetectorVerdictData | None:
        """Process one frequency observation.

        Accepts only observations whose ``topic`` matches ``target_topic``.
        During the learning phase returns ``None``. After learning, returns a
        verdict on every call.

        Args:
            sample: a ``FrequencySample`` for a single topic.

        Returns:
            A ``DetectorVerdictData`` once learning is complete, or ``None``
            during learning or if the sample is for a different topic.
        """
        # Only handle matching topic (frequency.py:79).
        if sample.topic != self.target_topic:
            return None

        hz = sample.frequency_hz

        # -- Learning phase (frequency.py:88-101) ----------------------------
        if self._baseline is None:
            self._samples.append(hz)
            if len(self._samples) < _LEARNING_SAMPLES:
                return None
            self._baseline = sum(self._samples) / len(self._samples)
            self._samples.clear()
            logger.info(
                "FrequencyDropAdapter baseline for %s: %.2f Hz",
                self.target_topic,
                self._baseline,
            )
            return None

        # -- Monitoring phase (frequency.py:103-152) -------------------------
        baseline = self._baseline
        floor = baseline * (1.0 - self._tolerance_pct / 100.0)
        raw_violating = hz < floor

        post_hyst = self._hysteresis.observe(raw_violating)

        # Score: 0 = healthy (hz == baseline), 1 = dead (hz == 0).
        # healthy anchor = baseline, worst anchor = 0 Hz.
        score = normalize(hz, healthy=baseline, worst=0.0) if baseline > 0.0 else 0.0
        # Clamp: if hz > baseline (overclock) score stays 0.
        score = max(0.0, min(1.0, score))

        sev = classify(score)

        if raw_violating:
            reason = (
                f"freq:{self.target_topic} {hz:.2f} Hz below floor "
                f"{floor:.2f} Hz (baseline {baseline:.2f} Hz, "
                f"tolerance {self._tolerance_pct}%)"
            )
            logger.warning(  # noqa: PLE1205 -- throttling done by the node
                "FrequencyDropAdapter %s: %.2f Hz < floor %.2f Hz",
                self.target_topic,
                hz,
                floor,
            )
        else:
            reason = (
                f"freq:{self.target_topic} {hz:.2f} Hz healthy "
                f"(baseline {baseline:.2f} Hz)"
            )

        return DetectorVerdictData(
            source=self.name,
            score=score,
            violating=post_hyst,
            reason=reason,
            suggested_action=sev.suggested_action,
        )


# ---------------------------------------------------------------------------
# StaticThresholdAdapter
# ---------------------------------------------------------------------------


class StaticThresholdAdapter(Detector):
    """Fires when a single named metric exceeds a static threshold.

    Logic ported from BlackBoxRS
    ``blackboxrs/anomaly_engine/detectors/threshold.py:75-154``.

    Accepts ``ThresholdSample`` objects. The calling node is responsible for
    mapping system-monitor data to ``ThresholdSample(metric, value, threshold)``
    triplets. The adapter applies hysteresis and normalizes the score.

    Score normalization: 0 = value == 0, 1 = value == threshold * 2. This
    ensures the score reaches 1.0 well before the metric hits pathological
    values while keeping 0.5 at the threshold (a breach maps immediately to at
    least INTERVENE territory).

    Args:
        target_topic: logical input name used by the arbiter, e.g.
            ``"system:cpu"`` or ``"system:mem"``.
        metric: the metric identifier string (e.g. ``"cpu_percent"``). Used in
            ``Detector.name`` and verdict source.
        min_consecutive: hysteresis window. Ported from BlackBoxRS
            ``AnomalyThresholds.min_consecutive_samples`` (``threshold.py:78``).
    """

    def __init__(
        self,
        target_topic: str,
        metric: str,
        min_consecutive: int = 2,
    ) -> None:
        self.name = f"threshold:{metric}"
        self.target_topic = target_topic
        self._metric = metric
        self._hysteresis = Hysteresis(min_consecutive)

    def update(self, sample: ThresholdSample) -> DetectorVerdictData | None:
        """Process one threshold observation.

        Accepts only ``ThresholdSample`` objects whose ``metric`` matches
        the adapter's ``metric``.

        Args:
            sample: a ``ThresholdSample`` for one metric.

        Returns:
            A ``DetectorVerdictData`` always (post-learning), or ``None`` if
            the sample is for a different metric.
        """
        if sample.metric != self._metric:
            return None

        value = sample.value
        threshold = sample.threshold

        # Violating if value exceeds the configured limit (threshold.py:107).
        raw_violating = value > threshold
        post_hyst = self._hysteresis.observe(raw_violating)

        # Score: 0 = value == 0, 1 = value == threshold * 2.
        # Healthy anchor = 0, worst anchor = threshold * 2.
        worst = threshold * 2.0 if threshold > 0.0 else 1.0
        score = normalize(value, healthy=0.0, worst=worst)

        sev = classify(score)

        if raw_violating:
            reason = (
                f"threshold:{self._metric} {value:.2f} exceeds "
                f"limit {threshold:.2f}"
            )
            logger.warning(
                "StaticThresholdAdapter %s: %.2f > %.2f",
                self._metric,
                value,
                threshold,
            )
        else:
            reason = (
                f"threshold:{self._metric} {value:.2f} within "
                f"limit {threshold:.2f}"
            )

        return DetectorVerdictData(
            source=self.name,
            score=score,
            violating=post_hyst,
            reason=reason,
            suggested_action=sev.suggested_action,
        )


# ---------------------------------------------------------------------------
# DeadTopicAdapter
# ---------------------------------------------------------------------------


class DeadTopicAdapter(Detector):
    """Fires when a topic has been silent for longer than ``timeout_sec``.

    Logic ported from BlackBoxRS
    ``blackboxrs/anomaly_engine/detectors/dead_topic.py:50-127``.

    Unlike the BlackBoxRS version (which is driven by an event stream and
    therefore can only detect silence when OTHER events arrive), this adapter is
    driven by explicit clock ticks from the rclpy node. The node calls
    ``update(DeadTopicSample(...))`` on a timer; the adapter only needs the
    ``last_seen_sec`` and ``now_sec`` fields to compute elapsed silence.

    The ``_alerted`` flag mirrors ``self._alerted`` in BlackBoxRS
    ``dead_topic.py:53``: once a dead-topic verdict is issued it is not re-issued
    until the topic becomes active again. The node must call
    ``mark_alive(topic)`` when a new message is observed on the topic to reset
    the alert flag and update ``last_seen_sec``.

    Score: 0 = just seen (elapsed == 0), 1 = elapsed == timeout * 2.

    Args:
        target_topic: the ROS topic to watch.
        timeout_sec: silence duration after which the topic is considered dead.
            Ported from BlackBoxRS ``DeadTopicConfig.timeout_sec``
            (``dead_topic.py:51``).
    """

    def __init__(self, target_topic: str, timeout_sec: float = 5.0) -> None:
        self.name = f"dead:{target_topic}"
        self.target_topic = target_topic
        self._timeout_sec = timeout_sec
        self._alerted = False

    def mark_alive(self, now_sec: float) -> None:
        """Record that a message was just received on the watched topic.

        Mirrors BlackBoxRS ``dead_topic.py:83-85`` (``_last_seen`` update and
        ``_alerted.discard``). The node calls this from its subscription
        callback; the adapter just clears the alert flag.

        Args:
            now_sec: current time in seconds (float), used only for logging.
        """
        self._alerted = False
        logger.debug(
            "DeadTopicAdapter %s: topic alive at %.3f",
            self.target_topic,
            now_sec,
        )

    def update(self, sample: DeadTopicSample) -> DetectorVerdictData | None:
        """Check whether the topic has been silent too long.

        Args:
            sample: a ``DeadTopicSample`` carrying ``topic``, ``last_seen_sec``,
                and ``now_sec``. If ``topic`` does not match ``target_topic``
                returns ``None``.

        Returns:
            A ``DetectorVerdictData`` if the topic is dead (or alive, so the
            arbiter has a fresh healthy verdict), or ``None`` if the topic does
            not match.
        """
        if sample.topic != self.target_topic:
            return None

        elapsed = sample.now_sec - sample.last_seen_sec
        raw_violating = elapsed > self._timeout_sec

        # Score: 0 = just seen, 1 = elapsed == timeout * 2.
        worst = self._timeout_sec * 2.0 if self._timeout_sec > 0.0 else 1.0
        score = normalize(elapsed, healthy=0.0, worst=worst)

        sev = classify(score)

        if raw_violating:
            # Only log the FIRST alert per silence window (dead_topic.py:94).
            if not self._alerted:
                logger.warning(
                    "DeadTopicAdapter %s: silent %.1fs > timeout %.1fs",
                    self.target_topic,
                    elapsed,
                    self._timeout_sec,
                )
                self._alerted = True
            reason = (
                f"dead:{self.target_topic} silent {elapsed:.1f}s "
                f"(timeout {self._timeout_sec}s)"
            )
            # Use STOP_AND_HOLD: a dead topic is an unrecoverable health signal
            # until the topic resumes; the arbiter may downgrade per its policy.
            action = ACTION_STOP_AND_HOLD
        else:
            # Topic is alive; clear alert (matches dead_topic.py:85 discard).
            self._alerted = False
            reason = (
                f"dead:{self.target_topic} alive "
                f"(last seen {elapsed:.2f}s ago)"
            )
            action = sev.suggested_action

        return DetectorVerdictData(
            source=self.name,
            score=score,
            violating=raw_violating,
            reason=reason,
            suggested_action=action,
        )


# ---------------------------------------------------------------------------
# RecurrentTemporalSpreadAdapter
# ---------------------------------------------------------------------------


class RecurrentTemporalSpreadAdapter(Detector):
    """Fires when a policy's recurrent feature freezes (rolling spread collapses).

    Origin: this is the E6 self-aware OOD monitor from Phantom-Braking
    (``supercombo-blindspot``, github.com/yusufdxb/supercombo-blindspot),
    ``src/e6_detector.py:16-31`` (``rolling_spread`` + ``calibrate_threshold``).
    There it showed that openpilot's supercombo driving model freezes its 512-D
    recurrent feature to a single point out of distribution, and that a monitor
    watching the rolling temporal spread of that feature, calibrated at the 1st
    percentile of real-driving spread (reference threshold 0.078873), catches the
    collapse. The byte-faithful math lives in ``phm_core.calibration``
    (rolling_spread / calibrate_threshold), itself ported from that source; this
    adapter reuses those functions rather than reimplementing them.

    Signal direction (opposite of the threshold / frequency adapters): a LOW
    rolling spread is the unhealthy one. The spread is the trace of the windowed
    covariance (sum of per-dimension variances over the last ``window`` frames);
    it drops toward zero when the recurrent state stops moving. So the adapter
    fires when ``spread < threshold`` rather than above it.

    The adapter buffers the last ``window`` embedding frames. Until the buffer
    fills it returns a warm-up healthy verdict (never ``None`` once a matching
    sample arrives, so the arbiter always has a fresh signal). After the buffer
    fills it computes the rolling spread of the window, compares against the
    calibrated threshold, applies hysteresis, and emits a verdict.

    Calibration: the threshold can be supplied at construction time or learned
    from a corpus of in-distribution embeddings via :meth:`calibrate_from_data`
    (the deploy-time hook), which delegates to
    ``phm_core.calibration.calibrate_threshold`` at the given percentile.

    Score normalization (low spread = bad): ``normalize(spread,
    healthy=threshold, worst=0.0)`` so ``spread == threshold`` maps to 0.0 and a
    fully frozen state (spread 0) maps to 1.0. When the threshold is
    non-positive (uncalibrated) any below-threshold spread is treated as the
    worst case (score 1.0).

    Args:
        target_topic: the embedding topic this adapter watches (used as
            ``Detector.target_topic``; the node routes by topic).
        window: number of consecutive frames in the rolling covariance. Must be
            >= 2. Default 30, matching the Phantom-Braking E6 window.
        threshold: calibrated spread below which the detector fires. Default 0.0
            (uncalibrated); set via the constructor or
            :meth:`calibrate_from_data` before deployment.
        min_consecutive: hysteresis window. A verdict fires only after this many
            consecutive below-threshold samples. Default 2, matching the other
            adapters.
    """

    def __init__(
        self,
        target_topic: str,
        window: int = 30,
        threshold: float = 0.0,
        min_consecutive: int = 2,
    ) -> None:
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        self.name = f"recurrent_temporal_spread:{target_topic}"
        self.target_topic = target_topic
        self._window = window
        self._threshold = float(threshold)
        self._hysteresis = Hysteresis(min_consecutive)
        # Rolling buffer of 1-D embedding frames, capped at window length.
        self._buffer: list[np.ndarray] = []
        self._last_spread: float | None = None

    @property
    def window(self) -> int:
        """Rolling-spread window length."""
        return self._window

    @property
    def threshold(self) -> float:
        """Current calibrated spread threshold (spread below this fires)."""
        return self._threshold

    @property
    def last_spread(self) -> float | None:
        """Most recently computed rolling spread, or None if not yet computed."""
        return self._last_spread

    def set_threshold(self, threshold: float) -> None:
        """Set the calibrated spread threshold. Call once before monitoring."""
        self._threshold = float(threshold)

    def calibrate_from_data(
        self, in_dist_hidden: np.ndarray, percentile: float = 1.0
    ) -> float:
        """Learn and store the threshold from in-distribution embeddings.

        Delegates to ``phm_core.calibration`` (Phantom-Braking E6): computes the
        rolling spread of ``in_dist_hidden`` and takes its ``percentile``-th
        percentile, so ~99% of real frames stay above the threshold at the
        default 1st percentile.

        Args:
            in_dist_hidden: shape (T, D) array of in-distribution hidden states.
            percentile: percentile of the spread distribution (default 1.0).

        Returns:
            The computed threshold (also stored on the adapter).
        """
        spreads = rolling_spread(
            np.asarray(in_dist_hidden, dtype=np.float64), self._window
        )
        thr = calibrate_threshold(spreads, percentile)
        self._threshold = thr
        return thr

    def update(
        self, sample: RecurrentSpreadSample
    ) -> DetectorVerdictData | None:
        """Process one embedding frame and optionally emit a verdict.

        Accepts only samples whose ``topic`` matches ``target_topic``. While the
        rolling buffer is filling it returns a warm-up healthy verdict. Once full
        it computes the rolling spread, compares against the threshold (spread <
        threshold -> OOD), applies hysteresis, and returns a verdict.

        Args:
            sample: a ``RecurrentSpreadSample`` carrying one embedding frame.

        Returns:
            A ``DetectorVerdictData`` for a matching topic (warm-up or scored),
            or ``None`` if the sample is for a different topic.
        """
        if sample.topic != self.target_topic:
            return None

        emb = np.asarray(sample.embedding, dtype=np.float64).ravel()
        self._buffer.append(emb)
        if len(self._buffer) > self._window:
            self._buffer.pop(0)

        # Warm-up: buffer not yet full -> healthy verdict, hysteresis untouched.
        if len(self._buffer) < self._window:
            return DetectorVerdictData(
                source=self.name,
                score=0.0,
                violating=False,
                reason=(
                    f"recurrent_temporal_spread:{self.target_topic} warming up: "
                    f"{len(self._buffer)}/{self._window} frames"
                ),
                suggested_action=ACTION_NONE,
            )

        # Rolling spread over the full window: only the last value is non-NaN.
        hidden = np.stack(self._buffer, axis=0)  # (window, D)
        spread = float(rolling_spread(hidden, self._window)[-1])
        self._last_spread = spread

        # Low spread = OOD (Phantom-Braking E6: spread < threshold -> firing).
        raw_violating = spread < self._threshold
        post_hyst = self._hysteresis.observe(raw_violating)

        # Score: 0 = spread == threshold (healthy edge), 1 = spread == 0 (frozen).
        if self._threshold > 0.0:
            score = normalize(spread, healthy=self._threshold, worst=0.0)
        else:
            # Uncalibrated/degenerate threshold: any breach is worst case.
            score = 1.0 if raw_violating else 0.0

        sev = classify(score)

        if raw_violating:
            reason = (
                f"recurrent_temporal_spread:{self.target_topic} spread "
                f"{spread:.6f} < threshold {self._threshold:.6f} "
                f"(window {self._window})"
            )
            logger.warning(
                "RecurrentTemporalSpreadAdapter %s: spread %.6f < threshold %.6f",
                self.target_topic,
                spread,
                self._threshold,
            )
        else:
            reason = (
                f"recurrent_temporal_spread:{self.target_topic} spread "
                f"{spread:.6f} >= threshold {self._threshold:.6f} "
                f"(window {self._window})"
            )

        return DetectorVerdictData(
            source=self.name,
            score=score,
            violating=post_hyst,
            reason=reason,
            suggested_action=sev.suggested_action,
        )
