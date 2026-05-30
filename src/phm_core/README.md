# phm_core

Pure-Python detector logic for the Policy Health Monitor. No ROS dependency, so
the unit tests run fast and without a ROS graph.

Modules:

- `detector.py`: `Detector` ABC and `DetectorVerdictData` (mirrors `DetectorVerdict.msg`).
- `hysteresis.py`: `Hysteresis(min_consecutive).observe(violating) -> bool`, the shared consecutive-violation debounce extracted from BlackBoxRS and HELIX.
- `calibration.py`: `rolling_spread`, `calibrate_threshold`, `loco_fpr`, ported byte-faithfully from Phantom-Braking's E6 OOD detector.
- `severity.py`: the single place that maps a normalized score to a state (OK / DEGRADED / INTERVENE / STOP) and a suggested action.

Run the tests:

```bash
../../.venv/bin/python -m pytest src/phm_core -q
```
