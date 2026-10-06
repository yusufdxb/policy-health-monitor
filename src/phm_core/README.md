# phm_core

ROS-independent C++17 library for the Policy Health Monitor. The ROS 2 nodes in the
sibling packages are thin wrappers over it, and it builds with plain CMake as well as
colcon.

| Header | Contents |
|---|---|
| `severity.hpp` | State / action enums, the OK / DEGRADED / INTERVENE / STOP score bands, `normalize`, `classify`. |
| `hysteresis.hpp` | `Hysteresis(min_consecutive).observe(violating)`, the shared consecutive-violation debounce. |
| `detector.hpp` | `VerdictData` (mirrors `DetectorVerdict.msg`) and the `Detector<Sample>` interface. |
| `calibration.hpp` | `rolling_spread`, `calibrate_threshold`, `loco_fpr` (supercombo-blindspot E6). |
| `spread_backend.hpp` | The rolling-spread kernel: `plain` (always), `eigen` (when found), `libtorch` (opt-in). |
| `ood_core.hpp` | `OodCore`: ring-buffered window, frequency gate, threshold, hysteresis, severity banding, non-finite and malformed-input verdicts. |
| `adapters.hpp` | Frequency-drop, static-threshold, dead-topic and recurrent-spread detectors. |
| `system_metrics.hpp` | Host CPU, memory and GPU-temperature reader (procfs / sysfs). |
| `arbiter.hpp` | Worst-wins `arbitrate()` with stale-never-de-escalates and the bad-score sentinel. |
| `recovery.hpp` | `SafetyEnvelope`, `HealthToActionMapper`, `RewindHook`, `RecoveryController`. |
| `sim.hpp` | Synthetic healthy / collapsed embedding streams. |
| `numpy_random.hpp` | NumPy-compatible `default_rng` (PCG64) and legacy `RandomState` (MT19937) streams. |
| `numerics.hpp` | NumPy-order sum, mean, percentile and quantile. |
| `npy.hpp` | `.npy` / `.npz` read and write (np.savez-compatible). |

Random streams, reductions, rolling spread and calibration reproduce NumPy bit for bit
(`test/test_numpy_random.cpp`, `test/test_calibration.cpp`), so a seed or a calibration
file means the same thing it meant in the NumPy implementation this library replaced.
The library is compiled with `-ffp-contract=off` so that holds on aarch64 as well.

Build and test without ROS (from the repository root):

```bash
cmake -S src/phm_core -B build/phm_core -DCMAKE_BUILD_TYPE=Release
cmake --build build/phm_core -j
ctest --test-dir build/phm_core --output-on-failure
build/phm_core/phm_collapse_example
```

Dependencies: zlib (required), Eigen 3 (optional backend), LibTorch (optional backend,
`-DPHM_WITH_LIBTORCH=ON`), GoogleTest (tests).
