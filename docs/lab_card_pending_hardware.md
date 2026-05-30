# Track F: deployment validation (pending a compute target)

Status: BLOCKED on hardware (no Jetson Orin available as of 2026-05-29). The GO2's
onboard compute IS the Orin, so without one there is no on-robot deploy and no real
edge number. Everything below is pre-wired so this becomes scripted validation, not
development, the moment a compute target exists (a borrowed Jetson, an Orin Nano dev
kit as a stand-in, or a GO2 driven from an external machine at CaresLab).

## What is deploy-ready now (sim + workstation certified)
- `phm_msgs` contract (PolicyHealthStatus / PolicyEmbedding / DetectorVerdict).
- The full ROS 2 pipeline: `phm_ood` (Python) and `phm_ood_cpp` (C++) -> `phm_detectors` -> `phm_arbiter` -> `/phm/health` -> `phm_recovery` (safe cmd_vel hold + rewind hook).
- C++ runtime node builds with plain / Eigen / LibTorch backends; LibTorch links against the installed torch (CUDA 12.8) on the workstation.
- Benchmark harness (`benchmark/`) and the alpha-sweep demo (`vla_monitor_demo/`).

## Numbers still OWED (do not claim until measured on the target)
1. Real Jetson Orin NX per-frame latency for `phm_ood_cpp` (plain + LibTorch + a TensorRT engine built from JetPack's native TensorRT). Compare to the workstation 93 us/frame (plain CPU).
2. End-to-end `/policy/embedding -> /phm/health -> /phm/cmd_vel` hold latency on a live ROS 2 graph (not just node-module imports).
3. Real-policy OOD: tap `go2-phoenix`'s deployed locomotion policy hidden state on hardware, calibrate the OOD threshold on real-robot traces, report a real-data FPR (the current 1.05% is synthetic).
4. The capstone video: monitor fires -> robot refuses / holds / recovers, on the real GO2.

## Pre-flight when hardware arrives
- Build TensorRT backend natively on the Orin (JetPack ships libnvinfer + headers; the workstation could not, 2 pip attempts failed).
- Wire `go2-phoenix` to publish `PolicyEmbedding` on `/policy/embedding`.
- Run the existing nodes; confirm the QoS contract (TRANSIENT_LOCAL on `/phm/health`) connects on the real DDS.
- Record latency/FPR, drop them into `benchmark/RESULTS.md` and `site/index.html` (replacing the workstation-CPU labels).

This card is the single source of truth for what "done on hardware" means. No edge or
Jetson claim is published anywhere in this repo until the number is measured here.
