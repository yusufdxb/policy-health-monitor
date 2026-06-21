# 30-Day Validation Plan

Companion to `proof_of_concept.md`. The single question this plan answers in 30 days:
**does internal-feature OOD predict robot-policy failure better than chance, with
useful lead time, and does anyone need us to build it?**

## Assumptions and budget

- Solo developer, 20 hours/week, 4 weeks = **80 hours total**.
- Existing robotics + OOD experience, so no ramp time on theory or tooling.
- Reuse: the PHM detector math (Mahalanobis, Relative Mahalanobis, kNN), the metric
  harness (AUROC / AUPR / FPR@95TPR with bootstrap CIs), hysteresis, severity. New code
  is the wrapper, three detectors (ViM, Energy, MSP), the risk head, and the real
  benchmarks.

## Sequencing principle: kill fast

The riskiest claim is M2 (real-policy failure prediction). The plan reaches a **real
robomimic AUROC by end of Week 2**, not Week 4, so a dead idea dies at the halfway
point and saves 40 hours. Week 1 is machinery you can trust; everything after is the
real claim and its honesty checks.

Two hard gates: a **Week 2 mid-point gate** (rough real signal) and the **Week 4
decision**. Failing the Week 2 gate triggers either a scoped pivot or early termination,
documented, not pushed through.

Every task below names its artifact. "Done" means the artifact exists and its number is
recorded, not that code "looks right".

---

## Week 1: Machinery you can trust (M1) + landscape check

Goal: a working wrapper, all five detectors, and proof the math is correct on a canonical
benchmark before touching a robot. Plus: confirm we are not rebuilding something that
already exists.

| Task | Hours | Measurable artifact |
|---|---|---|
| `sentinel.wrap()` + forward-hook FeatureExtractor (layer auto-resolve, pooling, logit capture) | 5 | `src/sentinel/` importable; parity test asserting policy outputs bit-identical with/without wrapper passes |
| Adapter detectors over existing PHM math (Mahalanobis, Rel-Maha, kNN) + ScoreCalibrator (empirical-CDF to [0,1]) | 3 | unit tests green; calibrated scores in [0,1] on a fixture |
| Implement ViM, Energy, MSP (no novel algorithms, standard formulations) | 5 | unit tests per detector; values match hand-computed references on a toy tensor |
| CIFAR-10 vs SVHN/CIFAR-100 sanity benchmark (`benchmark/real/vision_sanity.py`) | 5 | `RESULTS_REAL.md` table: per-detector AUROC + CI vs published numbers |
| Open-source landscape scan (pytorch-ood, OpenOOD, others) for "does this already exist for *robot policies*" | 2 | `docs/landscape.md`: what exists, what they cover, the specific gap (live policy wrap + failure prediction, not classification OOD) |

**Week 1 deliverables**
- `src/sentinel/` with `wrap()`, five detectors, calibrator, passing unit + parity tests.
- `benchmark/real/RESULTS_REAL.md` machinery section with per-detector AUROC + CIs.
- `docs/landscape.md` competitive scan.

**Week 1 exit gate (M1):** kNN AUROC >= 0.90, Mahalanobis >= 0.88, Energy >= 0.85,
MSP >= 0.80, ViM >= 0.88 on CIFAR-10 vs SVHN/CIFAR-100, each within +/- 0.03 of
published. If a detector misses, fix it now; do not carry broken math into the robot
benchmark. If `docs/landscape.md` finds an existing tool that already wraps live robot
policies and predicts failure, stop and reassess scope before Week 2.

---

## Week 2: First real signal (early M2 read) + mid-point gate

Goal: stand up the robomimic pipeline and get a rough but honest failure-prediction AUROC
on a real policy. This is the make-or-break week.

| Task | Hours | Measurable artifact |
|---|---|---|
| robomimic + robosuite + mujoco install; load pretrained BC-RNN on `lift` | 3 | script runs N nominal rollouts, prints success rate matching the model card |
| Rollout harness: capture per-frame penultimate features + episode success/failure label; enforce disjoint calibrate/eval splits | 5 | `benchmark/real/robomimic_eval.py`; cached feature + label arrays on disk, leakage assertion passes |
| Induce shift (object-pose perturb, camera shift, lighting, sensor noise) to produce a failure population | 3 | rollout set with a non-trivial failure rate (target 20 to 50%), recorded |
| Fit detectors on ID, score eval rollouts, compute episode-level failure-prediction AUROC + CI on `lift` | 6 | `RESULTS_REAL.md` row: best-detector AUROC + 95% CI on `lift`, vs chance and MSP-where-defined |
| Write up the mid-point read honestly (including a null, if that is what it is) | 3 | `docs/midpoint_gate.md` with the number and the go/no-go call |

**Week 2 deliverables**
- `benchmark/real/robomimic_eval.py` end-to-end on `lift`.
- A real episode-level failure-prediction AUROC with CI on one task.
- `docs/midpoint_gate.md` decision memo.

**Week 2 mid-point gate (rough M2):** best detector on `lift` reaches **AUROC >= 0.70 with
95% CI lower bound > 0.55**. This is a softer bar than the final M2 (one task, single
seed) but enough to justify spending Weeks 3 to 4. If it clears, continue. If it lands
0.55 to 0.70, continue but flag risk. If the CI includes chance, **stop here**: the core
idea has not shown signal on a real policy, and 40 hours are saved. Record the negative
result.

---

## Week 3: Strengthen the claim (full M2, M3, M4)

Goal: turn one rough number into a defensible result. Multiple tasks and seeds, lead time,
calibrated risk, and the second testbed.

| Task | Hours | Measurable artifact |
|---|---|---|
| Extend robomimic eval to `can` and `square`, 3 seeds each; per-task + pooled AUROC/AUPR/FPR@95 with CIs | 6 | `RESULTS_REAL.md` full table, multi-task multi-seed, vs chance + baseline |
| Lead-time metric: fix operating point at <=10% false-alarm on successes, measure frames-before-failure distribution | 4 | lead-time histogram + "% of failures flagged >= 1.0s early" number |
| RiskHead (logistic) on labeled rollouts + reliability diagram + ECE on held-out rollouts | 4 | `failure_risk` calibrated; reliability plot + ECE value in `RESULTS_REAL.md` |
| openpilot supercombo corruption alpha-sweep (reuse Phantom-Braking setup) | 6 | `benchmark/real/openpilot_sweep.py`; AUROC-vs-alpha curve, monotonicity check |

**Week 3 deliverables**
- Full multi-task, multi-seed robomimic table (M2).
- Lead-time result (M3) and risk calibration / ECE (M4).
- Second testbed: openpilot corruption sweep curve.

**Week 3 exit gate:** on track for final M2 (AUROC >= 0.75, CI lower bound > 0.60 on
>= 2 testbeds), M3 (>= 60% of failures flagged >= 1.0s early at <=10% false-alarm), and
M4 (ECE <= 0.15). Any miss is logged with the actual number; do not paper over it.

---

## Week 4: Overhead, deployment advantage, decision (M5 + verdict)

Goal: prove it is cheap enough to run, decide whether it earns its place, and make the
continue/terminate call with numbers.

| Task | Hours | Measurable artifact |
|---|---|---|
| Latency benchmark: added cost vs bare policy forward on mewtwo (RTX 5070); Jetson Orin NX if available | 4 | `RESULTS_REAL.md` overhead row: % and absolute ms per detector |
| Deployment-advantage assessment: what a builder gains vs the cheap baseline (MSP alone) and vs doing nothing | 3 | `docs/deployment_advantage.md`: concrete delta (AUROC + lead time gained over MSP/none) |
| Finalize `RESULTS_REAL.md` (all M1 to M5 numbers, CIs, plots) + reproducibility check (pinned env, seeds, one-command rerun of headline) | 5 | `RESULTS_REAL.md` complete; headline figure regenerates from cache via one command |
| README + `proof_of_concept.md` reconciliation (claims match measured numbers, bound the scope) | 3 | updated docs; every claim traceable to a number |
| Write the verdict | 5 | `docs/decision.md`: CONTINUE or TERMINATE with evidence against each criterion |

**Week 4 deliverables**
- Complete `RESULTS_REAL.md` covering M1 to M5 with CIs and reproducible headline.
- `docs/deployment_advantage.md` and `docs/decision.md`.

---

## End of Week 4: Continue or terminate?

The decision is mechanical, mapped to the four termination criteria. **Terminate if any
single criterion below is met.** Continue only if all four pass.

| Termination criterion | Measured by | Terminate if |
|---|---|---|
| AUROC below target | Final robomimic failure-prediction AUROC + 95% CI (M2) | AUROC < 0.75 **or** CI lower bound <= 0.60 on the mandatory robomimic testbed |
| Excessive false alarms | False-alarm rate on successful episodes at the M3 operating point, and whether lead time survives at that rate | Cannot reach >= 60% of failures flagged >= 1.0s early without > 10% false alarms on successes (M3 fails) |
| No deployment advantage | Delta over MSP-alone (and over no monitor) in AUROC + lead time, and overhead (M5) | Best detector does not beat MSP by a CI-separated margin where MSP applies, **or** feature-based methods give no lead-time gain over the cheapest baseline, **or** overhead >= 15% / >= 2 ms makes runtime use impractical |
| Existing OSS already solves it | `docs/landscape.md` + `docs/deployment_advantage.md` | An existing maintained library already wraps live robot policies and predicts failure with comparable numbers (i.e. the gap we identified does not actually exist) |

### Decision template (`docs/decision.md`)

State, with the actual measured number beside each:

1. Final M2 AUROC + CI (robomimic, pooled and worst task) and openpilot sweep result.
2. M3 lead time at fixed false-alarm rate.
3. M4 ECE; M5 overhead.
4. Deployment delta vs MSP and vs nothing.
5. Landscape verdict: does a comparable OSS tool exist?
6. **CONTINUE** (all four criteria pass) or **TERMINATE** (any one fails), with the
   single most decisive number called out.

A clean negative result (M2 fails, OOD does not track robot failure) is a successful
outcome of this plan: it ends the idea in 30 days on evidence, not on vibes, and the
negative result is itself publishable.

---

## Risk register (front-loaded by design)

| Risk | Lands in week | Mitigation |
|---|---|---|
| robomimic / mujoco install friction | 2 | scheduled in Week 2 day 1; budget already allocates 3h |
| OOD does not track robot failure (core idea is wrong) | 2 | mid-point gate kills it at 40 hours, not 80 |
| MSP/Energy undefined for continuous control narrows coverage | 2 to 3 | feature-based methods carry robomimic; stated in PoC, not discovered late |
| Jetson unavailable for M5 | 4 | report mewtwo overhead; Jetson is a nice-to-have, not a gate |
| An OSS tool already does this | 1 | landscape scan is a Week 1 task, before heavy build |
