# Policy Health Monitor (PHM)

Policy Health Monitor is a runtime reliability layer for learned robot policies. It
consumes a policy's internals (embeddings, actions), perception, and node and sensor
health, then emits a single arbitrated signal (`OK`, `DEGRADED`, `INTERVENE`, `STOP`),
each with a human-readable reason and a recommended action, so a learned controller can
be supervised in the loop.

The differentiating component is an out-of-distribution detector that runs on the policy's
internal embeddings and fires *before* behavior collapses, rather than after the robot has
already misbehaved. It consolidates the detector, hysteresis, and threshold-free metrics
work from three prior projects (HELIX, BlackBoxRS, Phantom-Braking) into one
runtime-assurance stack.

The system architecture (signal flow from policy internals through the OOD detectors and
worst-wins arbiter to the recovery layer) is in
[docs/architecture.svg](docs/architecture.svg).

## Packages

| Package | Role |
|---|---|
| `phm_msgs` | ROS 2 messages: `PolicyHealthStatus`, `PolicyEmbedding`, `DetectorVerdict`. |
| `phm_core` | Pure-Python detector logic (Detector ABC, Hysteresis, calibration, severity). No ROS dependency. |
| `phm_detectors` | Concrete detectors, including the rolling-spread / collapse detector ported from Phantom-Braking E6. |
| `phm_ood` | OOD scoring on policy internals. |
| `phm_arbiter` | Worst-wins arbiter that fuses detector verdicts + node/sensor health into one `PolicyHealthStatus` (total ordering, stale-critical safety). |
| `phm_recovery` | Safe-fallback layer (`cmd_vel` hold + rewind hook). |
| `phm_ood_cpp` | C++ `rclcpp` runtime node with plain / Eigen / LibTorch backends. |
| `phm_sim` | Synthetic policy-stream harness for end-to-end tests. |

## Benchmark

The PHM OOD detector is benchmarked against Mahalanobis, Relative Mahalanobis, KNN, and
RND (closed-form and torch-trained) on two synthetic failure families. Metrics are
threshold-free (AUROC, AUPR, FPR@95TPR) with stratified-bootstrap 95% CIs. Full numbers
and methodology in [benchmark/RESULTS.md](benchmark/RESULTS.md).

Headline: on the **collapse** failure (a frozen, low-variance embedding) the PHM
rolling-spread detector is perfect (AUROC 1.000, FPR@95 0.000) while every location-based
baseline is at or below chance (AUROC 0.03 to 0.42). A collapse is a second-order anomaly
(within-window variance drops while the embedding's location does not move), so first-order
location detectors are structurally blind to it. On the **shift** failure both the location
baselines and PHM are perfect. The two scenarios make the contrast explicit; PHM covers the
failure mode the standard baselines miss.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install numpy==1.26.4 pytest ruff
# pure-Python stack (no ROS graph, no rclpy):
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  src/phm_core src/phm_detectors src/phm_ood src/phm_arbiter src/phm_recovery src/phm_sim benchmark -q
```

`phm_core` and the detector/arbiter/recovery logic carry no ROS dependency, so their tests
run fast without `rclpy`. The ROS 2 nodes are thin wrappers and build with `colcon`. The
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` flag avoids ROS's `launch_testing` pytest plugin when a
ROS distro is sourced in the same shell.

## Status

The full stack builds (`colcon`) and the pure-Python suite passes (290 tests). The benchmark
runs on synthetic policy streams. On-device validation on a Jetson Orin (real-policy
embeddings, on-device latency and false-positive rate, and an induced-failure hardware
demo) is pending a compute target; the owed numbers and pre-flight are tracked in
[docs/lab_card_pending_hardware.md](docs/lab_card_pending_hardware.md).

## License

MIT, see [LICENSE](LICENSE).
