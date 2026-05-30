"""Consecutive-violation hysteresis counter.

This extracts the duplicated debounce pattern found in two source repos:

- BlackBoxRS ``blackboxrs/anomaly_engine/detectors/threshold.py:113``: a healthy
  sample resets the per-metric counter to 0; a violating sample increments it;
  the anomaly only fires once ``count >= min_consecutive_samples``.
- HELIX ``src/helix_core/helix_core/anomaly_detector.py:198``: increments
  ``_consecutive[metric]`` on each Z-score violation and only emits a fault once
  ``consecutive >= self._consecutive_trigger``, resetting otherwise.

Both implementations are the same finite state: count consecutive violating
observations, fire at a threshold, reset on any healthy observation. This module
is the single shared home for that logic so detectors no longer each reimplement
it.
"""

from __future__ import annotations


class Hysteresis:
    """Fires only after a run of consecutive violating observations.

    ``observe(violating)`` returns ``True`` once at least ``min_consecutive``
    violating observations have arrived in an unbroken run, and keeps returning
    ``True`` while the run continues. A single healthy observation resets the
    run, so the next firing again requires ``min_consecutive`` in a row.
    """

    def __init__(self, min_consecutive: int) -> None:
        """Create a hysteresis counter.

        Args:
            min_consecutive: number of consecutive violating observations
                required before :meth:`observe` returns ``True``. Must be >= 1;
                a value of 1 fires on the first violating observation.

        Raises:
            ValueError: if ``min_consecutive`` is less than 1.
        """
        if min_consecutive < 1:
            raise ValueError(
                f"min_consecutive must be >= 1, got {min_consecutive}"
            )
        self._min_consecutive = int(min_consecutive)
        self._count = 0

    def observe(self, violating: bool) -> bool:
        """Record one observation and report whether the detector should fire.

        Mirrors BlackBoxRS threshold.py:108-117: reset on healthy, increment on
        violating, fire at ``count >= min_consecutive``.

        Args:
            violating: ``True`` if this observation breached the detector's
                raw condition, ``False`` if healthy.

        Returns:
            ``True`` if there have now been at least ``min_consecutive``
            consecutive violating observations, otherwise ``False``.
        """
        if not violating:
            # Healthy sample: reset the run (threshold.py:108-109).
            self._count = 0
            return False

        # Violating sample: extend the run (threshold.py:112-114).
        self._count += 1
        return self._count >= self._min_consecutive

    def reset(self) -> None:
        """Clear the current run of violating observations."""
        self._count = 0

    @property
    def count(self) -> int:
        """Length of the current unbroken run of violating observations."""
        return self._count

    @property
    def min_consecutive(self) -> int:
        """Number of consecutive violations required to fire."""
        return self._min_consecutive
