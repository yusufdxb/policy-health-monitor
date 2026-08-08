# Native Runtime Evidence

This document records the externally reproducible evidence for the
`phm_ood_cpp` runtime and its Python reference implementation. It describes
what the repository proves, the ROS 2 contracts that must remain stable, and
the limits of the current validation.

## Evidence summary

| Area | Repository evidence | Current limit |
|---|---|---|
| Rolling-spread math | Python unit tests and C++ gtests cover nominal, collapsed, dimension-mismatch, and non-finite inputs. | The optional LibTorch backend has no dedicated parity test. |
| Bad-input handling | Python and C++ return the shared `BAD_INPUT_SCORE` for non-finite spread and cache the detector-health verdict across frequency-gated frames. | The verdict is evidence of detector health, not evidence that the policy is OOD. |
| Lifecycle behavior | The C++ node builds as a managed lifecycle node and resets its rolling window on deactivation. | No launch test drives lifecycle transitions in CI. |
| ROS install overlay | `scripts/check_install_imports.py` verifies all packages through the sourced ament index and rejects source-tree path injection. | This proves installation and imports, not a live multi-node graph. |
| Latency | `bench_latency` measures the ROS-free `OodCore::update` path. | It excludes DDS, executor scheduling, message conversion, publication, and target-platform effects. |

## Safety behavior

### Non-finite embeddings

A NaN or infinite element propagates into the rolling-spread statistic. In both
implementations, the detector now checks that statistic before thresholding.
When it is non-finite, the detector returns:

- score: `BAD_INPUT_SCORE` (`0.5`)
- violating: `false`
- suggested action: `ACTION_NONE`
- reason: `non-finite spread: embedding contains NaN or Inf`

The finite score lets the downstream arbiter classify the condition as a
detector-health degradation without misrepresenting it as a confirmed OOD
violation. Caching the result prevents `compute_every > 1` from replaying a
previous healthy verdict on the next gated frame.

### Threshold readiness

Rolling spread is non-negative, so `threshold <= 0` makes the collapse test
inert. Both runtime nodes emit a warning during configuration when this occurs.
The parameter remains accepted for backward compatibility, but deployments
must load a calibrated positive threshold before treating the detector as
active safety evidence.

### Embedding QoS

The Python and C++ embedding subscribers use best-effort reliability, keep-last
depth 10, and volatile durability. A best-effort subscriber can connect to
best-effort or reliable publishers. This avoids the silent no-data state caused
by a reliable subscriber paired with a best-effort publisher.

## ROS 2 interface contract

### Detector input and output

| Direction | Topic | Type | QoS |
|---|---|---|---|
| Input | `/policy/embedding` | `phm_msgs/PolicyEmbedding` | best effort, keep last 10, volatile |
| Output | `/phm/verdicts` | `phm_msgs/DetectorVerdict` | reliable, keep last 10, volatile |

The C++ runtime node name is `/phm_ood_cpp`. It is a
`rclcpp_lifecycle::LifecycleNode` and publishes only while ACTIVE. Configure
and activate it explicitly:

```bash
ros2 run phm_ood_cpp ood_node --ros-args -p threshold:=0.05
ros2 lifecycle set /phm_ood_cpp configure
ros2 lifecycle set /phm_ood_cpp activate
```

Deactivation clears the rolling window. Frames received while inactive do not
contribute to a later verdict.

### Fused consumer contract

Consumers should subscribe to the arbiter output rather than directly to a
single detector:

| Topic | Type | QoS |
|---|---|---|
| `/phm/health` | `phm_msgs/PolicyHealthStatus` | reliable, keep last 1, transient local |

The arbiter applies worst-wins ordering across detector verdicts. It also
degrades stale or non-finite detector inputs. A consumer must request transient
local durability to receive the retained health state.

## Reproduce the evidence

### Pure Python

```bash
python3 -m venv .venv
.venv/bin/pip install numpy==1.26.4 pytest ruff
.venv/bin/ruff check src/ scripts/
.venv/bin/python scripts/check_public_hygiene.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  src/phm_core src/phm_detectors src/phm_ood src/phm_arbiter \
  src/phm_recovery src/phm_sim benchmark -q
```

The current suite result is 295 passing tests.

### ROS 2 build and package tests

From a sourced ROS 2 Humble environment:

```bash
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -y --rosdistro humble
colcon build
source install/setup.bash

repo_dir=$PWD
smoke_dir=$(mktemp -d)
cd "$smoke_dir"
python3 "$repo_dir/scripts/check_install_imports.py"
cd "$repo_dir"

colcon test
colcon test-result --verbose
```

The current package result is 44 tests with no errors or failures across all
eight packages. The C++ OOD gtest target contains 11 passing cases.

### CPU micro-benchmark

```bash
source install/setup.bash
ros2 run phm_ood_cpp bench_latency 100000 30 512
```

The benchmark times `OodCore::update` with pre-generated frames. Results are
host-specific and must include the run configuration and machine state when
reported. They are not end-to-end ROS latency or target-platform evidence.

## Proof boundary

Validated by the repository:

- ROS-free Python detector and arbitration behavior
- C++ detector-core behavior for the default build backends
- package compilation and package-level tests on ROS 2 Humble
- install-space discovery and imports after sourcing the colcon overlay
- static agreement of topic names, message types, and QoS settings

Not yet validated:

- physical-robot or target-platform behavior
- end-to-end DDS and executor latency
- a live multi-node safety intervention
- automated Python-to-C++ numerical parity over shared fixtures
- the optional LibTorch backend in CI

These limits are intentional proof boundaries. Build, unit-test, or host
micro-benchmark results must not be presented as hardware safety validation.
