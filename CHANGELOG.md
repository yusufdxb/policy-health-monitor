# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions follow semantic versioning.

## [Unreleased]

### Changed (0.2.0: C++ migration)
- Every package is C++ (`ament_cmake`). The detector, calibration, severity, hysteresis,
  OOD, health-detector, arbitration, recovery and synthetic-stream logic is the
  ROS-independent library `phm_core`, which also builds with plain CMake and installs a
  `find_package(phm_core)` config. The Python packages, their setuptools / pip / pytest /
  ruff metadata and the pip distribution are removed.
- `phm_ood` and `phm_ood_cpp` run one lifecycle-node implementation over one OOD core.
  Each keeps its node name, verdict source, parameters, QoS, stamping and malformed-input
  behavior. `phm_ood` now declares its parameters at construction (so configure, cleanup,
  configure works) and answers a mid-stream dimension change with a bad-input verdict
  instead of raising.
- The plain rolling-spread kernel reads the window in place from a ring buffer (no
  per-frame copy or allocation) and reproduces NumPy's evaluation order, so a threshold
  calibrated offline scores identically at runtime. It is built with `-ffp-contract=off`.
- Launch files are XML (`*.launch.xml`), replacing `*.launch.py`.
- The GO2 integration is the colcon package `phm_go2`: `phoenix_shadow_embedder` (ONNX
  Runtime C++ API; observation assembly ported from go2-phoenix and pinned by a parity
  fixture), `phm_go2_probe`, `phm_go2_summarize`, `phm_go2_plot` (SVG figures) and the
  offline `phoenix_shadow_replay`. `onboard_session.sh` runs these executables directly.
- The benchmark (`phm_benchmark`) and the alpha-sweep demo (`vla_monitor_demo`) are C++,
  with NumPy-compatible random streams and PyTorch-identical initialization. Benchmark
  rows `RND (numpy)` and `RND (torch, gradient-trained)` are now `RND (closed-form)` and
  `RND (MLP, gradient-trained)`; the demo figure is `alpha_sweep.svg`.
- CI builds the ROS-free superbuild (`standalone/`) and the colcon packages, and runs
  `scripts/check_public_hygiene.sh`, `scripts/check_no_project_python.sh`,
  `scripts/check_install_overlay.sh` and `scripts/check_benchmark_reproduces.sh`.

### Added
- `phm_tools` (header-only JSON / SVG / small-MLP helpers for the offline tools).
- `phm_sim`'s `phm_chain_bench` and `scripts/measure_chain.sh`: end-to-end timing of a
  running graph (embedding to verdict, health and recovery output, CPU and memory).
- ROS graph tests (gtest on live rclcpp graphs) for the detectors, OOD, arbiter and
  recovery nodes.

### Fixed
- Treat non-finite rolling spread as a detector-health fault.
- Align the C++ embedding subscriber with the best-effort QoS contract.
- The 0.1.x Python `phm_ood` node created its verdict publisher in `on_activate` but never
  activated it, so on a live graph it dropped every verdict (reproduced: 0 verdicts on
  `/phm/verdicts` during 6 s of 10 Hz embeddings). The C++ node activates its publisher,
  and `phm_ood_cpp/test/test_ood_node.cpp` checks that both interfaces publish.
- Warn when a non-positive threshold leaves either OOD node inert.
- The docs said the arbiter turns a non-finite-spread verdict into a health degradation. It
  does not: a fresh non-violating verdict never changes `/phm/health`. The docs now state
  this limit; the behavior is unchanged.
- Hardened the `.npz` / `.npy` reader against malformed files (out-of-range lengths and
  offsets, overflowing shapes), which could read past the buffer.
- An empty first embedding no longer pins the OOD dimension to 0; `window` beyond 100000
  and hysteresis / `compute_every` values that do not fit in an int now fail configure for
  both OOD node interfaces (`phm_ood` declares no parameter ranges); the hysteresis run
  counter saturates instead of overflowing.
- `phoenix_shadow_embedder` refuses a model without an `action` output at startup;
  `phoenix_shadow_replay` validates its input arrays.
- GO2 session: `onboard_session.sh` measured CPU in whole seconds (`ps -o times`), which
  rounds the C++ nodes (about 1% of a core) to zero; it now reads clock ticks from `/proc`.
  Its `ARB_STALENESS` default is 1.5 s instead of 1.0 s: with 1 Hz detector verdicts, a
  1.0 s limit produced brief stale `DEGRADED` reports when the arbiter's timer lined up
  with the detectors' tick (seen in a desktop rehearsal: 4 of 1200 nominal health
  messages). `preflight.sh` checks the setup before a session.
- `vla_monitor_demo/README.md` stated a +0.050 lead time on a 41-point alpha grid
  (monitor 0.325, output 0.375). Neither the Python demo nor its C++ port reproduces that:
  both give +0.025 (monitor 0.350, output 0.375). The README now says so; the 21-point
  headline (+0.05) is unchanged.

### Changed
- Publish reproducible native-runtime evidence and add a CI guard against publishing
  machine-local paths or private hardware identifiers.

## [0.1.0] - 2026-06-06

First public release.

### Added
- Top-level `pyproject.toml`: the ROS-free stack (`phm_core`, `phm_detectors`, `phm_ood`,
  `phm_arbiter`, `phm_recovery`, `phm_sim`) is now `pip install`-able without ROS.
- CI jobs for a clean-environment `pip install` + import smoke and the full pure-Python
  test suite, alongside the existing `colcon build` job on `ros:humble`.
- Installation section in the README (pip for the library, colcon for the full ROS 2 graph).

### Fixed
- Corrected the author email in `phm_core` package metadata.

### Notes
- Foundation stack from the initial build: 8 ROS 2 packages (`phm_msgs`, `phm_core`,
  `phm_detectors`, `phm_ood`, `phm_arbiter`, `phm_recovery`, `phm_ood_cpp`, `phm_sim`),
  a threshold-free reliability benchmark vs Mahalanobis / RMD / KNN / RND, and a C++
  `rclcpp` runtime node (plain / Eigen / LibTorch backends).
- Target-platform validation with real-policy embeddings, on-device latency, false-positive
  rate, and an induced-failure hardware demo remains pending.
