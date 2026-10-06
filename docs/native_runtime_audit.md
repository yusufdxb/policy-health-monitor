# Native Runtime Evidence

This document records the externally reproducible evidence for the C++ runtime: the
`phm_core` library, the ROS 2 nodes built on it, and the offline tools. It describes what
the repository proves, the ROS 2 contracts that must remain stable, and the limits of the
current validation.

## Evidence summary

| Area | Repository evidence | Current limit |
|---|---|---|
| Rolling-spread math | `phm_core` gtests pin the spread, calibration and random streams to values NumPy produced (bit-exact), and cover nominal, collapsed, dimension-mismatch and non-finite inputs. | The optional LibTorch backend has no dedicated parity test; Eigen agrees with the plain kernel to 1e-12 relative, not bit for bit. |
| Decision logic | Ported from the earlier Python implementation and compared on randomized scenarios before that implementation was removed: OOD core, health detectors, arbiter and recovery decisions were identical, including scores and reason strings. The gtests keep those cases as fixtures. | A parity statement about the removed implementation, not about the robot. |
| Bad-input handling | The OOD core returns `BAD_INPUT_SCORE` for a non-finite spread and caches it across frequency-gated frames. | The verdict is evidence of detector health, not evidence that the policy is OOD, and the arbiter does not escalate it: `/phm/health` stays `OK`. |
| Lifecycle behavior | `phm_ood_cpp/test/test_ood_node.cpp` drives configure / activate / deactivate / cleanup on a live rclcpp graph for both OOD node interfaces. | One process, one host. |
| ROS graph behavior | gtests on live rclcpp graphs for the detectors (raw generic subscriptions, dead and dropped topics), arbiter (worst wins, staleness, non-finite scores, transient-local health) and recovery (holds only when enabled, cooldown, rewind). | Synthetic inputs. |
| ROS install overlay | `scripts/check_install_overlay.sh` resolves every package through the ament index, checks installed executables and their shared libraries, launch files and configs, the absence of project Python, and a `find_package(phm_core)` consumer. | Proves installation, not a live multi-node graph. |
| End-to-end chain | `scripts/measure_chain.sh` runs the detectors, OOD, arbiter and recovery nodes on an isolated localhost graph and times embedding to verdict, health and zero-velocity output with `phm_chain_bench`, plus CPU and RSS per node. | Desktop machine, synthetic input; not the robot. |
| Detector-core latency | `bench_latency` times `OodCore::update` alone. | Excludes DDS, executor scheduling and publication. |

## Safety behavior

### Non-finite embeddings

A NaN or infinite element propagates into the rolling-spread statistic. The detector
checks that statistic before thresholding. When it is non-finite, the detector returns:

- score: `BAD_INPUT_SCORE` (`0.5`)
- violating: `false`
- suggested action: `ACTION_NONE`
- reason: `non-finite spread: embedding contains NaN or Inf`

The finite score keeps the verdict out of the arbiter's non-finite-score path, and the
verdict does not misrepresent the condition as a confirmed OOD violation. Caching the
result prevents `compute_every > 1` from replaying a previous healthy verdict on the next
gated frame.

The arbiter does not escalate a fresh non-violating verdict, so this verdict reaches
`/phm/verdicts` but `/phm/health` stays `OK`; the same applies to dimension-mismatch,
empty-embedding and oversized-embedding verdicts. This has been the arbiter's rule since
0.1.0. Escalating bad-input verdicts to a health degradation would be a behavior change
and is not implemented.

### Malformed embeddings

- `phm_ood_cpp` drops a frame whose `dim` field disagrees with its length, and an empty
  frame. `phm_ood` publishes a bad-input verdict for a frame whose length differs from the
  pinned dimension.
- Without `embed_dim`, the first non-empty frame pins the dimension. Empty frames before
  it are scored as zero-dimension frames (an all-empty stream reads as a collapse, as the
  NumPy core scored it) but never pin the dimension.
- A frame that would need more than `kMaxWindowFloats` (2^28) floats of window buffer is
  rejected as bad input. Both nodes fail configuration for `window` outside [2, 100000] and
  for hysteresis or `compute_every` values outside [1, INT_MAX].

### Threshold readiness

Rolling spread is non-negative, so `threshold <= 0` makes the collapse test inert. Both
OOD node interfaces emit a warning during configuration when this occurs. The parameter
remains accepted for backward compatibility, but deployments must load a calibrated
positive threshold before treating the detector as active safety evidence.

### Embedding QoS

The OOD nodes subscribe with best-effort reliability, keep-last depth 10, and volatile
durability. A best-effort subscriber can connect to best-effort or reliable publishers.
This avoids the silent no-data state caused by a reliable subscriber paired with a
best-effort publisher.

### Actuation

`/phm/health` carries the fused state; publishing `STOP` there does not stop a robot. Only
`phm_recovery`, with `recovery.enabled: true`, publishes zero-velocity `geometry_msgs/Twist`
on `/phm/cmd_vel`. Its default configuration has actuation disabled.

## ROS 2 interface contract

### Detector input and output

| Direction | Topic | Type | QoS |
|---|---|---|---|
| Input | `/policy/embedding` | `phm_msgs/PolicyEmbedding` | best effort, keep last 10, volatile |
| Output | `/phm/verdicts` | `phm_msgs/DetectorVerdict` | reliable, keep last 10, volatile |

The OOD runtime is a `rclcpp_lifecycle::LifecycleNode` with two entry points:
`ros2 run phm_ood_cpp ood_node` (node `/phm_ood_cpp`, source `phm_ood_cpp`) and
`ros2 run phm_ood phm_ood_node` (node `/phm_ood`, source `phm_ood`). Both publish only
while ACTIVE. Configure and activate explicitly:

```bash
ros2 run phm_ood_cpp ood_node --ros-args -p threshold:=0.05
ros2 lifecycle set /phm_ood_cpp configure
ros2 lifecycle set /phm_ood_cpp activate
```

`ood_node` clears its rolling window on deactivation; frames received while inactive do
not contribute to a later verdict.

### Fused consumer contract

Consumers should subscribe to the arbiter output rather than directly to a single
detector:

| Topic | Type | QoS |
|---|---|---|
| `/phm/health` | `phm_msgs/PolicyHealthStatus` | reliable, keep last 1, transient local |
| `/phm/cmd_vel` | `geometry_msgs/Twist` | reliable, keep last 10, volatile |

The arbiter applies worst-wins ordering across detector verdicts. It also degrades stale
or non-finite detector inputs. A consumer must request transient local durability to
receive the retained health state.

## Reproduce the evidence

### ROS-free build and tests

```bash
cmake -S standalone -B build/standalone -DCMAKE_BUILD_TYPE=Release
cmake --build build/standalone -j
ctest --test-dir build/standalone --output-on-failure
scripts/check_public_hygiene.sh
scripts/check_no_project_python.sh
scripts/check_benchmark_reproduces.sh build/standalone/benchmark/phm_benchmark
```

### ROS 2 build and package tests

From a sourced ROS 2 Humble environment:

```bash
source /opt/ros/humble/setup.bash
rosdep install --from-paths src integrations --ignore-src -y --rosdistro humble
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
scripts/check_install_overlay.sh
colcon test
colcon test-result --verbose
```

### Latency

```bash
source install/setup.bash
PHM_BACKEND=plain ros2 run phm_ood_cpp bench_latency 100000 30 384
PHM_BACKEND=plain scripts/measure_chain.sh /tmp/phm_chain
```

`bench_latency` times `OodCore::update` with pre-generated frames. `measure_chain.sh`
times the running chain on an isolated localhost graph (`ROS_LOCALHOST_ONLY=1`, a private
domain id) and records CPU and peak RSS per node. Results are host-specific and must be
reported with the run configuration and machine state. They are not target-platform
evidence, and a C++ implementation is not by itself a hard real-time guarantee.

## Proof boundary

Validated by the repository:

- ROS-free detector, calibration, arbitration and recovery behavior
- bit-exact agreement of random streams, rolling spread and calibration with NumPy
- package compilation and tests on ROS 2 Humble, including live rclcpp graphs
- install-space discovery after sourcing the colcon overlay
- end-to-end timing of the chain on a desktop machine with synthetic input

Not yet validated:

- the C++ nodes on a physical robot or its onboard computer
- a live safety intervention on hardware
- the optional LibTorch backend in CI

These limits are intentional proof boundaries. Build, unit-test, replay or desktop
measurements must not be presented as hardware safety validation.
