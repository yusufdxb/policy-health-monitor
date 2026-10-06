# PHM onboard a Unitree GO2 (shadow mode)

This directory runs the full PHM graph on a Unitree GO2's onboard computer against a real
learned policy, without letting that policy or PHM command the robot. A shadow node runs
the [go2-phoenix](https://github.com/yusufdxb/go2-phoenix) stand-v3 locomotion policy on
the robot's live `/lowstate` at 50 Hz and publishes the policy's 384-D hidden-layer
latent. A session script calibrates PHM on that latent, records a nominal window, and
runs repeated induced faults. Two tools turn the recorded session into numbers and
figures. Everything here is the colcon package `phm_go2` (C++).

The results of the session are in the top-level
[README](../../README.md#on-a-real-robot). This page is how to reproduce it.

## Files

| File | Role |
|---|---|
| `phoenix_shadow_embedder` (`src/shadow_embedder_node.cpp`) | Shadow policy node. Runs the latent ONNX with the ONNX Runtime C++ API on live `/lowstate` at 50 Hz, publishes `phm_msgs/PolicyEmbedding` on `/policy/embedding`. Optional `freeze_obs` or `stop` fault. |
| `preflight.sh` | Read-only checks before a session: executables and libraries, model files, no stray or actuating processes, `/lowstate` rate, disk space. |
| `onboard_session.sh` | Runs one session: calibrate, start PHM, nominal window, induced-fault trials, per-process CPU accounting. |
| `phm_go2_probe` (`src/probe.cpp`) | `calibrate`: threshold from live latents. `record`: every `/phm/health`, `/phm/verdicts`, and embedding arrival to JSONL. Subscriber only. |
| `phm_go2_detectors.yaml` | `phm_detectors` parameters: rate and dead-topic checks on the robot's state topics and the latent, plus CPU, memory, and GPU-temperature limits. |
| `phm_go2_summarize` (`tools/summarize_session.cpp`) | Reduces a session directory to one JSON object. No ROS needed. |
| `phm_go2_plot` (`tools/plot_session.cpp`) | Renders the README figures (SVG) from a session directory. No ROS needed. |
| `phoenix_shadow_replay` (`src/shadow_replay.cpp`) | Runs the shadow policy offline on a recorded or synthetic `/lowstate` sequence (`.npz`) and saves every observation, action and latent, for parity checks. |

## Safety: nothing here commands the robot

| Process | Started by `onboard_session.sh` | Publishes (besides `/rosout` and `/parameter_events`) |
|---|---|---|
| `phoenix_shadow_embedder` | yes | `/policy/embedding` only |
| `phm_detectors_node` | yes | `/phm/verdicts` |
| `phm_ood_cpp` `ood_node` | yes | `/phm/verdicts` |
| `phm_arbiter` | yes | `/phm/health` |
| `phm_go2_probe` | yes | nothing |
| `phm_recovery` | **never** | would publish `/phm/cmd_vel` |

- The policy's action output is used for one thing: the `last_action` term of its next
  observation, as the go2-phoenix deploy node feeds it. It is never published. Because
  it is never applied, the policy runs open loop on that term; the joint, IMU, and
  gravity terms are the robot's real state. The `freeze_obs` fault freezes the whole
  observation, last-action term included; the `stop` fault stops inference and
  publication entirely.
- `phm_recovery` is the only PHM package that acts on a `STOP`, and the session never
  launches it. A `STOP` on `/phm/health` changes nothing on the robot in this setup.
- The script does not read or change the robot's mode. In the reported session the robot
  stood still under its own controller (Unitree sport mode).

## Prerequisites on the robot's onboard computer

- ROS 2 Humble, and Unitree's `unitree_go` message package (from `unitree_ros2`), built
  and sourced, so `/lowstate` deserializes as `unitree_go/msg/LowState`.
- ONNX Runtime for the board's architecture: an `onnxruntime-linux-<arch>-<version>`
  release directory with `include/` and `lib/`. The shadow node uses only the CPU
  execution provider, with one thread.
- This repository built with colcon against both, so `phm_go2` includes the shadow node:

  ```bash
  source /opt/ros/humble/setup.bash
  source <unitree_ros2 workspace>/install/setup.bash
  colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release \
    -DONNXRUNTIME_ROOT=<onnxruntime release directory>
  ```

  Without ONNX Runtime or `unitree_go` the package still builds the probe and the session
  tools and skips the shadow node (CMake says so).
- Data actually flowing from the robot: `ros2 topic hz /lowstate` shows about 500 Hz. A
  CycloneDDS configuration bound to the wrong network interface still lists every topic
  but delivers no data, so check the rate, not the topic list.
- A deps directory (default `$HOME/phm_go2_deps`, override with `PHM_GO2_DEPS`) holding:

  ```text
  $HOME/phm_go2_deps/models/stand_v3_latent.onnx
  $HOME/phm_go2_deps/models/stand_v3_latent.onnx.data
  ```

The shadow node assembles each observation with a C++ port of go2-phoenix's
`assemble_policy_observation` (`src/phoenix/sim2real/observation.py` at commit `23c6fb5`)
for this policy: zero `base_lin_vel` (the stand-v3 deploy's "zeros" source), IMU gyroscope,
projected gravity, zero velocity command, joint positions relative to the training pose
and joint velocities in policy joint order, and the previous action. The port is pinned by
`test/test_observation.cpp` to observations that function produced, bit for bit, and the
whole shadow tick (observation, action, latent, last-action feedback, and both faults) was
compared against the Python node it replaces on a 600-tick synthetic `/lowstate` replay
with identical results; see [Parity](#parity).

`stand_v3_latent.onnx` is a stand-v3 checkpoint exported with go2-phoenix's exporter and
`--emit-latent`, which adds a `latent` output (the policy's concatenated hidden
activations, 384-D for stand-v3) beside `action`. In a go2-phoenix environment at the
same commit:

```bash
python -m phoenix.sim2real.export --checkpoint <stand-v3 checkpoint> \
  --output stand_v3_latent.onnx --emit-latent
```

Copy `stand_v3_latent.onnx.data` with it: it holds the weights, and ONNX Runtime loads it
from the same directory. The shadow node refuses to start if the model lacks the `obs`
input or the `action` or `latent` output. The checkpoint is not distributed with this
repository.

## Run a session

```bash
source /opt/ros/humble/setup.bash
source <unitree_ros2 workspace>/install/setup.bash    # unitree_go messages
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=<config bound to the robot-facing network interface>
source <policy-health-monitor>/install/setup.bash
ros2 daemon stop          # drop discovery state cached from an earlier environment

cd <policy-health-monitor>
integrations/go2/preflight.sh
PHM_GO2_DEPS=$HOME/phm_go2_deps NOM_SEC=300 TRIALS=5 \
  integrations/go2/onboard_session.sh session_out
```

`preflight.sh` is read-only. It checks that the session's executables and their libraries
resolve, the model files exist, no PHM process from an earlier run is still up,
`phm_recovery` is not running and nothing publishes `/phm/cmd_vel`, `/lowstate` arrives at
400 Hz or more (`MIN_LOWSTATE_HZ`), and there is room for the output. Start the session
only when it prints `preflight passed`.

The session runs in four phases:

1. **Calibrate.** Starts the shadow policy alone, records `CAL_SEC` of live latents, and
   sets the threshold at the `PERCENTILE` percentile of the `WINDOW`-frame rolling spread.
2. **Start PHM.** Starts the detectors node with `phm_go2_detectors.yaml`, the arbiter,
   and the C++ OOD node with the calibrated threshold, configures and activates the OOD
   node through its lifecycle, then waits 12 s so the rate checks learn their baselines.
3. **Nominal.** Records `NOM_SEC` with no fault, plus the CPU time of each process group.
4. **Faults.** `TRIALS` rounds, each one `freeze_obs` trial then one `stop` trial. Each
   trial restarts the shadow policy with the fault armed `FAULT_AT` seconds after it
   starts and records `FAULT_SEC`.

Every process starts in its own process group and is stopped by group. The script runs the
installed executables directly from the sourced overlay, with no `ros2 run` wrapper
process per node; it still uses `ros2 lifecycle set` for the OOD node's transitions.

| Variable | Default | Meaning |
|---|---|---|
| `PHM_GO2_DEPS` | `$HOME/phm_go2_deps` | deps directory described above |
| `PHM_GO2_ONNX` | `$PHM_GO2_DEPS/models/stand_v3_latent.onnx` | latent ONNX |
| `CAL_SEC` | 60 | calibration length, seconds |
| `NOM_SEC` | 120 | nominal phase length, seconds |
| `TRIALS` | 1 | trials per fault type |
| `FAULT_SEC` | 45 | recorded length of each fault trial, seconds |
| `FAULT_AT` | 20 | seconds after the shadow policy starts that the fault is injected |
| `WINDOW` | 30 | rolling-spread window, frames (0.6 s at 50 Hz) |
| `PERCENTILE` | 1.0 | calibration percentile |
| `MIN_CONSEC` | 3 | consecutive below-threshold frames before the C++ OOD node reports a violation |
| `ARB_STALENESS` | 1.5 | arbiter staleness limit, seconds. Must exceed the detectors node's 1 s verdict period: at 1.0 the arbiter can briefly report a verdict as stale (`DEGRADED`, reason `stale:...`) whenever its timer lines up with the detectors' tick, which the C++ nodes' near-simultaneous startup makes likely. The 0.1.x sessions used 1.0 |
| `PHM_BACKEND` | `plain` | C++ rolling-spread backend; `plain` measured about 2x faster than `eigen` on the GO2 |

The faults, both injected inside the shadow node and both still without actuation:

- `freeze_obs`: from the fault time on, the policy is fed the last observation it saw, as
  a policy wired to a stale sensor snapshot would be. The latent stops varying.
- `stop`: from the fault time on, the node stops publishing, as a crashed policy process
  would. `/policy/embedding` goes silent.

## Reduce a session to numbers and figures

From a sourced colcon overlay:

```bash
ros2 run phm_go2 phm_go2_summarize session_out > session_out/summary.json
ros2 run phm_go2 phm_go2_plot session_out --out docs/go2
```

The ROS-free superbuild in `standalone/` builds them too, so they run on a workstation
without ROS after copying the session directory back:

```bash
cmake -S standalone -B build/standalone -DCMAKE_BUILD_TYPE=Release
cmake --build build/standalone -j
build/standalone/phm_go2/phm_go2_summarize session_out > session_out/summary.json
build/standalone/phm_go2/phm_go2_plot session_out --out docs/go2
```

`phm_go2_plot` writes
`go2_freeze_fault.svg`, `go2_stop_fault.svg`, and `go2_nominal.svg` to `--out` (for a
session without fault trials whose nominal phase contains a stand-up burst, it writes
`go2_standup.svg` instead), computing every number on a figure from the recorded files.
Both tools read a session's `.jsonl` and `.log` files directly or from their `.gz` copies.

How the numbers are defined:

- **Detection latency** runs from the shadow node's `FAULT INJECTED` log timestamp to the
  time the recorder, a separate subscriber, receives the first matching message. Both are
  the onboard computer's system clock, so the latency includes delivery and detector
  compute. Each fault reports the median, minimum, and maximum over its trials, and how
  many trials reached each level (`DEGRADED` or worse, `INTERVENE` or worse, `STOP`).
- **Which verdicts count.** For `freeze_obs`, only the C++ OOD node's verdicts. For
  `stop`, the dead-topic and rate checks on `/policy/embedding` and the C++ OOD node.
- **Nominal false alarms.** Any `/phm/health` message that is not `OK`, and any violating
  OOD verdict, during the nominal phase.
- **Pre-fault rows.** Each trial also reports its rows before the fault, excluding the
  first 5 s of the trial: the shadow node has just restarted, and the previous trial's
  `STOP` is still clearing.

## Output files

| File | Contents |
|---|---|
| `session.txt` | start and end time, host, device model, ONNX hash prefix, every parameter, PHM git revision (`unknown` outside a git checkout), calibrated threshold, and `DONE` when the session finished |
| `events.log` | shadow node starts, phase markers, lifecycle transition output |
| `calibrate.json` | calibration summary: frame count, rate, threshold, spread minimum, median, and maximum |
| `calib.npz` | threshold, window, percentile, and the raw calibration latents (float32), so the threshold can be recomputed offline |
| `nominal.jsonl` | every `/phm/health`, `/phm/verdicts`, and `/policy/embedding` arrival in the nominal phase; each embedding is stored as its stamp and rolling spread, not the vector |
| `fault_freeze_<n>.jsonl`, `fault_stop_<n>.jsonl` | the same, one file per trial |
| `embedder_calibrate.log` | shadow node log for the calibration and nominal phases, with ONNX Runtime p50 and p99 every 10 s |
| `embedder_freeze_<n>.log`, `embedder_stop_<n>.log` | shadow node log per trial, including the `FAULT INJECTED` line that latencies are measured from |
| `detectors.log`, `arbiter.log`, `ood_cpp.log` | PHM node logs |
| `cpu_nominal.csv` | per process group: CPU seconds, wall seconds, and percent of one core over the nominal phase, plus `nproc` |
| `top_nominal.txt` | one `top` snapshot at the end of the nominal phase |

## Proof boundary

- The robot stood still under its own controller (Unitree sport mode). It did not walk.
- The policy ran in shadow mode and never controlled the robot. Because its actions were
  never applied, it ran open loop on its own last-action term. On a standing-still robot
  the latent can vary only through small changes in the real sensor readings and through
  that open-loop term, so the calibrated threshold describes this shadow setup, not a
  policy in control.
- The threshold was calibrated, and the false-alarm rate measured, on the standing-still
  state only.
- Both faults were induced in software. `freeze_obs` holds the whole observation
  constant, including the last-action term, so the latent becomes exactly constant: the
  easiest case for a rolling-spread detector. A stale sensor feeding a policy whose
  last-action term still changes was not tested. `stop` removes the embedding stream, so
  it tests liveness checks rather than the latent statistics.
- This is not a real policy failure in the field, and it is not evidence that PHM warns
  before a behavioral failure.
- Open: a rolling-spread collapse detector calibrated on a still robot can confuse "the
  robot is still" with "the policy collapsed" when the robot goes from moving to still.
  This has not been tested.
- The opposite change was seen once, in an earlier 0.1.x session calibrated with the robot
  lying down: when the operator stood the robot up, the rolling spread peaked at about
  80,000 times its lying-down median and settled about 1.6 times higher while standing.
  PHM stayed `OK` throughout (4,369 of 4,369 health messages, 0 of 10,922 OOD verdicts
  violating). The test is one-sided, so a burst of variance never raises it.
- `phm_recovery` was never launched, so no recovery action was exercised on hardware.
- The reported session ran the 0.2.0 C++ executables described here (shadow node, probe,
  detectors, arbiter and OOD node) at commit `54cc61a`. The stand-up observation above
  comes from an earlier 0.1.x session.

## Parity

The C++ tools replace Python ones; these checks were run on a desktop machine before the
Python versions were removed:

- Observation assembly: bit-identical to go2-phoenix's `assemble_policy_observation` on 64
  random readings (`test/test_observation.cpp`, kept as a test).
- Shadow tick on the real stand-v3 latent model: `phoenix_shadow_replay` against the
  Python node's logic on the same 600-tick synthetic standing `/lowstate` sequence, ONNX
  Runtime 1.23.2 through the C++ API on one side and the Python API on the other:
  observations, actions and latents bit-identical with no fault, with `freeze_obs` at
  tick 300, and with `stop` at tick 300 (300 outputs, then silence). The same held with
  ONNX Runtime 1.28.0 on the C++ side.
- The same 600-tick replay on the robot's onboard computer (aarch64, ONNX Runtime 1.23.2)
  against the desktop output, in all three modes: observation sensor terms bit-identical,
  actions within 9.6e-7 and latents within 1.9e-6 (ONNX Runtime uses different CPU kernels
  on ARM).
- Session reduction: `phm_go2_summarize` output byte-identical to the Python summarizer
  on two recorded sessions and on a copy with gzipped files; `phm_go2_plot` prints the
  same summary lines as the Python plotter. The figures are SVG instead of PNG.
