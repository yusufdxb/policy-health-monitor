# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions follow semantic versioning.

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
- On-device validation on a Jetson Orin (real-policy embeddings, on-device latency / FPR,
  induced-failure hardware demo) remains pending a compute target. See
  `docs/lab_card_pending_hardware.md`.
