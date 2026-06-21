# Proof of Concept: Runtime OOD / Failure Prediction for Robot Policies

## 0. What this document is

This is a technical validation plan for the smallest library that proves one claim:

> Given a trained robot policy and a stream of observations, internal-feature OOD
> scoring can predict impending policy failure significantly better than chance,
> with measurable lead time, at negligible runtime cost.

It is not a product spec. There is no dashboard, pricing, company, or roadmap here.
The only question is whether the idea is technically valid, and the success criteria
in Section 10 decide that with numbers.

### Relationship to the existing PHM repo

This repo (Policy Health Monitor) already contains the detector mathematics, the
threshold-free metric harness, hysteresis, and severity normalization, but it has two
gaps that block the proof:

1. It consumes embedding *arrays and streams*. It does not wrap a live `torch.nn.Module`
   and hook its features during inference.
2. Every published number is on **synthetic** policy streams. There is no real-policy
   failure-prediction result, which is exactly the claim under test.

The proof closes both gaps with a thin wrapper layer (`sentinel`) over the existing
detector code, plus three detectors that are not yet implemented (ViM, Energy, MSP),
plus a real-policy benchmark. Everything else is reuse.

What already exists and is reused as-is:

- `benchmark/lib/baselines.py`: `mahalanobis`, `relative_mahalanobis`, `knn_distance`.
- `phm_core/calibration.py`: `rolling_spread`, `calibrate_threshold`, `loco_fpr`.
- `phm_core/severity.py`: `normalize`, `classify`.
- `phm_core/hysteresis.py`: `Hysteresis` (debounces flapping near threshold).
- The bootstrap-CI AUROC / AUPR / FPR@95TPR harness in `benchmark/`.

What is new and is the actual deliverable of the proof:

- `sentinel.wrap()`: forward-hook feature capture on an arbitrary PyTorch policy.
- Three new detectors: `vim`, `energy`, `msp`.
- The three unified scalar outputs (`ood_score`, `confidence`, `failure_risk`).
- A logistic risk head that maps detector scores to a calibrated failure probability.
- The real-policy benchmark (robomimic + openpilot corruption sweep).

---

## 1. Architecture

The library is one process, no services, no IPC. It sits beside the policy and reads
the policy's own forward pass through a hook. Signal flow, top to bottom:

- **FeatureExtractor.** Registers a forward hook on one named layer of the policy
  (default: the penultimate layer, auto-resolved as the last layer feeding the action
  head). On each `policy(obs)` it captures that layer's activation, pools it to a fixed
  vector (`flatten` for MLP heads, global average pool for conv feature maps), and
  caches it. The policy runs exactly once; the wrapper reads the cache. If the policy
  exposes logits or a discrete action head, those are captured too (needed by MSP and
  Energy).

- **Detector bank.** A set of fitted detectors, each implementing `fit(feats[, logits])`
  and `score(feats[, logits]) -> float` where higher means more anomalous:
  - feature-based (work on any policy): `mahalanobis`, `relative_mahalanobis`, `knn`,
    `vim`, plus the existing PHM `rolling_spread` collapse detector.
  - logit-based (only when the policy emits logits): `energy`, `msp`.

- **ScoreCalibrator.** Each detector's raw score is mapped to `[0, 1]` using the
  empirical CDF of that detector's scores on the in-distribution calibration set
  (a stored sorted array; score is its quantile rank). This makes detectors with
  different units comparable and gives `ood_score` a fixed, interpretable scale.

- **Aggregator.** Combines per-detector calibrated scores into the three outputs
  (Section 4). For sequential policies it applies windowed temporal aggregation
  and an EMA before thresholding, because a single noisy frame must not trip an alarm.

- **RiskHead.** A logistic regression mapping the calibrated detector scores to a
  probability of episode failure. Fitted on labeled rollouts (Section 4.3). Optional:
  if no labeled rollouts are available, `failure_risk` falls back to the aggregated
  `ood_score` and is flagged uncalibrated.

The existing `docs/architecture.svg` shows the larger PHM graph (arbiter, recovery,
ROS nodes). The proof uses only the left half of that diagram: features in, scores out.

### Temporal correctness (non-negotiable)

For any recurrent or history-conditioned policy (RNN, transformer, VLA, openpilot-style
stacked-frame model), the policy's own recurrent state must roll normally; the wrapper
is read-only and must never zero or reset it. The detector consumes a *window* of recent
features, not a single frame, and the OOD score is smoothed (EMA + min-consecutive
hysteresis) so that the multi-second init transient of a recurrent policy is not
mistaken for an anomaly. This is enforced in tests, not left to convention. (Lesson
carried from Phantom-Braking: mishandled recurrent state produces a fake "phantom"
anomaly at episode start.)

---

## 2. API

The whole public surface is one `wrap` call plus three reads.

```python
import torch
from sentinel import wrap

policy = load_policy()  # any torch.nn.Module; not modified

mon = wrap(
    policy,
    feature_layer=None,                 # None = auto-resolve penultimate layer
    methods=("mahalanobis", "knn", "vim", "energy", "msp"),
    temporal_window=16,                 # frames; 1 disables temporal aggregation
    ema=0.2,                            # smoothing on the OOD score
)

# --- 1. Calibrate on in-distribution data (no labels needed) ---
for obs in id_calibration_loader:      # observations the policy was trained for
    policy(obs)                        # wrapper captures features via the hook
mon.fit()                              # fits detectors + score calibrators on captured ID features

# --- 2. (optional) Calibrate the failure-risk head on labeled rollouts ---
# rollouts: iterable of (sequence_of_obs, failed: bool)
mon.calibrate_risk(rollouts)           # fits logistic risk head; sets ECE-reportable calibration

# --- 3. Inference: policy use is unchanged ---
action = policy(obs)
report = mon.last()                    # reads the scores from the hook capture
report.ood_score        # float in [0, 1], higher = more out-of-distribution
report.confidence       # float in [0, 1], higher = policy more in-distribution
report.failure_risk     # float in [0, 1], calibrated P(this episode fails) if risk head fitted
report.per_method       # dict[str, float], calibrated score per detector
report.flag             # bool, ood_score over threshold AND survived hysteresis

# Convenience: run policy and read in one call
action, report = mon.step(obs)
```

Design rules for the API:

- `wrap` returns a monitor that owns the hook; deleting it removes the hook cleanly.
- The policy object is never mutated and its outputs are bit-identical with and without
  the wrapper. A parity test asserts this.
- `fit()` with no labels gives `ood_score` and `confidence`. `failure_risk` requires
  `calibrate_risk`; without it, `failure_risk` returns the aggregated OOD score and
  `report.risk_calibrated == False`.
- All three scores are always in `[0, 1]`. No raw distances leak into the public API.

---

## 3. Repository layout

The proof adds one new top-level package (`sentinel`) and one benchmark suite. It does
not touch the ROS packages.

```
policy-health-monitor/
  src/
    sentinel/                  # NEW: the minimal proof library
      sentinel/
        __init__.py            # exports wrap(), Report
        wrap.py                # Monitor class, hook lifecycle, step()/last()
        features.py            # FeatureExtractor: layer resolution, hook, pooling
        detectors/
          base.py              # Detector protocol: fit/score
          mahalanobis.py       # wraps benchmark/lib baselines (reuse)
          knn.py               # wraps benchmark/lib (reuse)
          vim.py               # NEW
          energy.py            # NEW
          msp.py               # NEW
          collapse.py          # wraps phm_core rolling_spread (reuse)
        calibrate.py           # ScoreCalibrator (empirical CDF) + RiskHead (logreg)
        aggregate.py           # temporal window, EMA, hysteresis, three-score fusion
      tests/                   # unit tests (parity, hook, calibration, temporal)
  benchmark/                   # EXISTS: synthetic harness + metric code (reused)
    real/                      # NEW: the real-policy proof
      robomimic_eval.py        # rollout -> features -> episode failure AUROC
      openpilot_sweep.py       # corruption alpha gradient on supercombo
      vision_sanity.py         # CIFAR-10 vs SVHN/CIFAR-100, machinery check
      RESULTS_REAL.md          # generated: the headline numbers + CIs
  proof_of_concept.md          # this file
```

`benchmark/lib/baselines.py` (Mahalanobis, Relative Mahalanobis, kNN) and the
threshold-free metric harness already exist; the `detectors/` modules are thin
adapters around them so there is a single source of truth for the math.

---

## 4. The three scores

### 4.1 OOD score

For each fitted detector, the raw anomaly score on a frame (or window) is mapped to its
quantile against the ID calibration distribution, giving a per-detector value in `[0, 1]`.
The aggregate `ood_score` is the mean of the calibrated per-detector scores (mean, not
max, so one noisy detector cannot dominate; `per_method` exposes the individuals for
debugging). For sequential policies the score is taken over the temporal window and then
EMA-smoothed.

### 4.2 Confidence score

`confidence = 1 - ood_score`. When the policy emits logits, the MSP value (max softmax
probability) is also reported under `per_method["msp"]` as a second, independent
confidence reading, because MSP is the canonical confidence baseline and reviewers
expect it. The headline `confidence` field stays defined for all policy types (including
continuous control with no logits) by deriving from the OOD aggregate.

### 4.3 Failure-risk score

This is the score the headline claim is really about, and it is the one that needs
labels. A logistic regression maps the vector of calibrated detector scores to
`P(episode fails)`. It is fit on rollouts where the ground-truth outcome is known:

- collect, for each rollout, the per-frame detector scores and the episode label
  (`success`/`failure`) from the simulator's own success check;
- aggregate each rollout's frame scores to a feature vector (mean and max over the
  episode, plus the max over the last K frames so impending failure is weighted);
- fit logistic regression; report calibration with reliability curve and ECE.

If no labeled rollouts exist, `failure_risk` degrades to the OOD aggregate and is marked
uncalibrated. The proof's main result uses the calibrated head.

---

## 5. Dependencies

Core (kept minimal so "install the library" is one line and works on a Jetson):

- `torch` (the policy and the hook; the only heavy dep, already required by any user).
- `numpy` (already a PHM dependency).
- `scikit-learn` (logistic risk head, `NearestNeighbors` for kNN, reliability/ECE).
- `scipy` (covariance, chi-square scaling for Mahalanobis).

Benchmark-only extras (not required to use the library, only to reproduce the proof):

- `robomimic` + `robosuite` + `mujoco` (manipulation policies with success labels).
- `onnxruntime` (openpilot supercombo, reused from the Phantom-Braking lineage).
- `torchvision` (CIFAR-10 / SVHN / CIFAR-100, machinery sanity check).
- `faiss-cpu` (optional, only if kNN needs to scale past ~50k ID samples).

No new core dependency beyond `scikit-learn` and `scipy`. Pinned `numpy==1.26.4` to
match the rest of the mewtwo / Jetson stack.

---

## 6. Methods

All five requested methods, plus the existing collapse detector. None are novel; each is
the standard published formulation. Applicability differs by policy type and that is
stated honestly, not hidden.

| Method | Needs | Applies to | Source |
|---|---|---|---|
| Mahalanobis | features | all policies | Lee et al. 2018 (reuse) |
| Relative Mahalanobis | features | all policies | Ren et al. 2021 (reuse) |
| kNN | features | all policies | Sun et al. 2022 (reuse) |
| ViM | features + (weight, bias) of action head | policies with a linear head | Wang et al. 2022 (new) |
| Energy | logits | discrete-action / classification policies | Liu et al. 2020 (new) |
| MSP | logits | discrete-action / classification policies | Hendrycks and Gimpel 2017 (new) |
| Collapse (rolling spread) | feature window | sequential policies | PHM, ported from Phantom-Braking E6 |

Honest applicability boundary, stated up front because it determines which testbeds use
which methods:

- **Continuous-control policies (robomimic BC/BC-RNN, locomotion) have no logits.**
  MSP and Energy do not apply to them. Those testbeds use the feature-based methods only
  (Mahalanobis, Relative Mahalanobis, kNN, ViM, collapse). The proof does not pretend
  otherwise.
- **ViM needs the action head's linear weight and bias** to compute the virtual-logit
  residual. If the head is non-linear, ViM falls back to its feature-residual term only
  and is flagged degraded.
- **MSP and Energy are validated on the classification testbed** (CIFAR sanity check)
  and on any discrete-action policy, where logits exist and the methods are defined.

---

## 7. Benchmarks

Three testbeds, ordered from "proves the code is correct" to "proves the claim".

### 7.1 Vision classification sanity check (machinery correctness)

CIFAR-10 as in-distribution, SVHN and CIFAR-100 as out-of-distribution, on a standard
pretrained ResNet. This is not a robot policy; its only job is to prove the detector
implementations reproduce published OOD-detection numbers. If our kNN, Mahalanobis,
Energy, MSP, and ViM land near their literature AUROC on this canonical benchmark, the
math is correct and the later robot numbers are trustworthy. If they do not, the proof
stops here and the bug is fixed before anything else.

### 7.2 Robomimic manipulation (the headline claim)

Pretrained robomimic BC-RNN policies on the `lift`, `can`, and `square` tasks. Procedure:

1. Run M in-distribution rollouts; capture per-frame penultimate features; `fit()` the
   detectors on a held-out ID subset (no leakage: calibration rollouts are disjoint
   from evaluation rollouts).
2. Run N evaluation rollouts under a mix of nominal and shifted conditions (object pose
   perturbation, camera shift, lighting change, sensor noise). Each rollout gets a
   ground-truth `success`/`failure` from robosuite's own task check.
3. Per rollout, aggregate frame scores and compute episode-level failure prediction
   AUROC, AUPR, FPR@95TPR with stratified bootstrap 95% CIs.
4. Compute lead time: at a threshold fixed to a target false-alarm rate on successful
   episodes, how many frames before terminal failure the flag first fires.

This is the test that decides the proof. Manipulation is chosen because success/failure
is unambiguous and cheap to label, and pretrained policies are public.

### 7.3 openpilot supercombo corruption sweep (distribution-shift gradient)

Reuse the Phantom-Braking supercombo setup. Sweep an alpha-blend from clean frames to a
corruption (fog, blur, low light, frame freeze). Report OOD AUROC for separating clean
from corrupted input as a function of alpha, expecting a monotone rise. This validates
that the score tracks shift severity on a real shipped vision model, and connects the
proof to existing verified work (parity already established in Phantom-Braking).

---

## 8. Evaluation methodology

- **Metrics are threshold-free first.** AUROC, AUPR, and FPR@95TPR, each with a
  stratified bootstrap 95% CI (1000 resamples). Threshold-dependent numbers (lead time,
  false-alarm rate) are reported only after fixing the operating point on a held-out
  calibration split, never on the test split.
- **Baselines, always.** Every result is reported against (a) random chance (AUROC 0.5)
  and (b) MSP, the standard cheap baseline. A method only "counts" if it beats both with
  a CI that excludes chance. On continuous-control testbeds where MSP is undefined, the
  baseline is chance plus the simplest feature baseline (raw Mahalanobis).
- **No leakage.** Calibration / fit data, risk-head training data, and evaluation data
  are disjoint at the rollout level. Asserted in the harness.
- **Calibration reporting.** The risk head reports a reliability diagram and ECE on a
  held-out rollout set, so `failure_risk` is shown to be a real probability, not a score
  in probability clothing.
- **Multiple tasks and seeds.** Robomimic results span at least three tasks and three
  seeds so the headline is not an N=1 anecdote. Per-task and pooled numbers both reported.
- **Overhead measured, not assumed.** Added latency is measured against the bare policy
  forward pass on both mewtwo (RTX 5070) and, when a target is available, the Jetson
  Orin NX, and reported as a percentage and absolute milliseconds.

---

## 9. Threats to validity (stated, not buried)

- **OOD is not the same as failure.** A robust policy can absorb mild shift and still
  succeed; a policy can fail in-distribution for reasons unrelated to OOD. The headline
  claim is precisely that internal-feature OOD *correlates with and precedes* failure
  well enough to beat chance, and Section 10 sets the bar. A null or weak result on 7.2
  is a real finding and is reported as such, not spun.
- **Single-checkpoint generalization.** Results are per-policy. The proof uses multiple
  tasks to reduce, not eliminate, this risk. The claim is bounded to the tested policy
  classes.
- **MSP / Energy inapplicability** to continuous control narrows method coverage on the
  most important testbed; the feature-based methods carry the claim there.
- **VLA is a stretch.** A VLA policy (for example OpenVLA) is feasibility-flagged: the
  hook and feature path are identical in principle (pooled vision-language token), but
  the model likely does not fit alongside the benchmark in 12 GB, and the score may be
  dominated by language conditioning rather than visual shift. VLA is attempted only if
  it fits; if it does not, the proof reports machinery-only feasibility, not a result.

---

## 10. Success criteria

The proof succeeds if and only if all of M1, M2, and M3 hold, with M4 and M5 as
quality gates. Every number is measurable from the harness.

- **M1 (machinery correct).** On the CIFAR-10 vs SVHN/CIFAR-100 sanity check, each
  implemented detector reaches at least: kNN AUROC >= 0.90, Mahalanobis >= 0.88,
  Energy >= 0.85, MSP >= 0.80, ViM >= 0.88, each within +/- 0.03 of its published value
  for the same setup. This proves the detectors are implemented correctly before any
  robot claim is made.

- **M2 (headline: beats chance on real failure prediction).** On at least two robot
  testbeds (robomimic being mandatory), the best detector predicts episode failure with
  **AUROC >= 0.75 and a bootstrap 95% CI lower bound > 0.60**. The 0.60 lower bound, not
  merely > 0.50, is what makes "significantly better than chance" defensible with margin.
  The best detector must also beat the MSP baseline (where defined) by a CI-separated
  margin.

- **M3 (useful lead time).** At an operating threshold whose false-alarm rate on
  successful episodes is <= 10%, the monitor flags failure at least 1.0 second (or >= 10
  control steps if the rate is unknown) before terminal failure in **>= 60%** of failed
  episodes. A detector that only fires at the moment of failure is not useful and fails
  M3 even if it passes M2.

- **M4 (calibrated risk).** The `failure_risk` head has **ECE <= 0.15** on held-out
  rollouts. Below this bar `failure_risk` is reported as a score, not a probability.

- **M5 (negligible overhead).** Added latency is **< 15%** of the policy forward pass and
  **< 2 ms** absolute for feature-based detectors on GPU. A monitor that doubles
  inference cost is not a runtime tool.

Failure of M2 is the most informative outcome: it would mean internal-feature OOD does
not track robot-policy failure well enough to be useful, which is a publishable negative
result and the honest end of the idea as scoped.

---

## 11. What the proof deliberately leaves out

No arbiter, no recovery layer, no ROS graph, no multi-signal fusion, no hardware demo,
no UI. Those exist or are planned in the larger PHM stack and are out of scope here. The
proof is exactly: wrap a PyTorch policy, score it, and show on real policies that the
score predicts failure better than chance, with lead time, cheaply. Nothing more.
