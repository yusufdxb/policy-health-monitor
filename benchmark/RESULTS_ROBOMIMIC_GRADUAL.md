# Robomimic gradual-onset induction cross-check (lift, BC-RNN-GMM)

Second OOD-induction mechanism for the same policy, feature tap, detector set and split discipline as `RESULTS_ROBOMIMIC.md`. Failure is induced by a Gaussian noise ramp on the policy's observation vector, zero at frame 0 of every episode by construction and rising linearly from a random per-episode onset, rather than by perturbing the cube's initial placement. Pre-registered at `docs/preregistration/2026-08-27-gradual-induction-cross-check.md` before the run.

Arms. A calibration-fit: 60 nominal episodes, the only pool any detector is fit on. B nominal-eval: 60 disjoint nominal episodes, which also fix the frame-level alarm threshold at the 0.95 quantile (~5% healthy FPR). C1 high-dose ramp: 70 episodes at sigma 3.05x nominal per-channel std, failure rate 0.94. C2 dose-matched ramp: 70 episodes at sigma 1.0x, selected by a pilot on a disjoint seed block using the success flag only, failure rate 0.41. D positive control: 70 episodes under the original cube-perturbation mechanism (half-extent 0.18), failure rate 0.40. Onset ~ U[0, 15) frames, ramp length 15 frames.

Exposure control. Nominal episodes terminate on success at 35 to 91 frames (median 41) while failures run the full 400-frame horizon, so a max-over-all-frames episode statistic gives failures roughly ten times as many chances to cross threshold. Every AUROC below is reported twice: `full` is the legacy max over all frames and is exposure-confounded; `k=30` is a max over the first 30 frames only, below the shortest nominal episode, and is the number that carries a claim.

Dose audit. Over the first 30 frames, which is the length-matched comparison, the failed episodes in the dose-matched arm C2 received a mean injected sigma fraction of 0.528 (n=29) and the succeeded ones 0.498 (n=41), a relative gap of 0.055. The pre-registered blocker is a relative gap above 0.10. Audited over the whole episode instead the gap is 0.231, but that number is a duration artifact rather than a dose mismatch: a successful episode ends after roughly 40 frames while its ramp is still climbing, whereas a failed one runs to the horizon at full amplitude, so the whole-episode mean measures how long an episode lasted as much as how hard it was hit. The injection schedule itself is identical across the two groups by construction.

## Contrast 1: the legacy shape, nominal-eval against a perturbed arm

This is what `RESULTS_ROBOMIMIC.md` reports. It asks whether a perturbed and failing episode looks different from a healthy one, which conflates 'this episode is corrupted' with 'this episode is going to fail'.

The C1 high-dose column is the contrast named in kill criterion 3 of the registration. Read it with the floor effect in mind: C1 failed in 66 of 70 episodes, so only 4 negatives back the whole column.

| Detector | gradual C1, full | gradual C1, k=30 | gradual C2, full | gradual C2, k=30 | cube D, full | cube D, k=30 |
|---|---|---|---|---|---|---|
| phm_rolling_spread | 0.157 [0.097, 0.228] | 0.156 [0.096, 0.227] | 0.543 [0.419, 0.662] | 0.489 [0.371, 0.613] | 0.430 [0.286, 0.585] | 0.375 [0.239, 0.518] |
| hotelling_t2 | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] |
| hotelling_t2_masked | 0.999 [0.996, 1.000] | 0.219 [0.136, 0.305] | 0.737 [0.619, 0.851] | 0.431 [0.310, 0.557] | 0.624 [0.483, 0.759] | 0.381 [0.240, 0.525] |
| mahalanobis | 1.000 [1.000, 1.000] | 0.979 [0.952, 0.998] | 0.933 [0.887, 0.970] | 0.736 [0.642, 0.823] | 0.974 [0.950, 0.993] | 0.901 [0.851, 0.947] |
| relative_mahalanobis | 1.000 [1.000, 1.000] | 0.981 [0.954, 0.998] | 0.936 [0.892, 0.971] | 0.744 [0.650, 0.831] | 0.979 [0.956, 0.995] | 0.906 [0.852, 0.950] |
| knn | 1.000 [1.000, 1.000] | 0.984 [0.964, 0.998] | 0.946 [0.906, 0.979] | 0.613 [0.503, 0.722] | 0.979 [0.955, 0.996] | 0.891 [0.834, 0.943] |
| pca_residual | 1.000 [1.000, 1.000] | 0.983 [0.957, 0.999] | 0.944 [0.902, 0.976] | 0.705 [0.604, 0.805] | 0.973 [0.947, 0.993] | 0.897 [0.843, 0.946] |
| rnd | 1.000 [1.000, 1.000] | 0.985 [0.963, 0.999] | 0.930 [0.884, 0.968] | 0.729 [0.640, 0.813] | 0.981 [0.961, 0.996] | 0.918 [0.870, 0.959] |
| temporal_rnd | 1.000 [1.000, 1.000] | 0.902 [0.848, 0.947] | 0.954 [0.916, 0.984] | 0.520 [0.404, 0.646] | 0.970 [0.936, 0.994] | 0.899 [0.841, 0.949] |

## Contrast 2 (primary): dose-matched, within one arm

Failed against succeeded episodes drawn from the SAME arm. Every episode here got the identical treatment at the identical amplitude, so the perturbation is held fixed and outcome is the only thing that varies. This is the contrast the 2026-06-16 design did not have, and it is where the pre-registered claim lives (primary cell: `phm_rolling_spread`, arm C2, k=30).

| Detector | C2 dose-matched, k=30 | C2 dose-matched, full | D cube within-arm, k=30 | D cube within-arm, full |
|---|---|---|---|---|
| phm_rolling_spread | 0.545 [0.406, 0.682] | 0.591 [0.458, 0.718] | 0.458 [0.313, 0.604] | 0.461 [0.309, 0.617] |
| hotelling_t2 | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] |
| hotelling_t2_masked | 0.511 [0.374, 0.646] | 0.701 [0.569, 0.826] | 0.455 [0.310, 0.605] | 0.641 [0.502, 0.775] |
| mahalanobis | 0.519 [0.383, 0.653] | 0.830 [0.727, 0.922] | 0.757 [0.640, 0.863] | 0.936 [0.872, 0.983] |
| relative_mahalanobis | 0.505 [0.363, 0.637] | 0.840 [0.731, 0.928] | 0.768 [0.658, 0.874] | 0.948 [0.889, 0.990] |
| knn | 0.437 [0.306, 0.563] | 0.862 [0.760, 0.945] | 0.735 [0.615, 0.849] | 0.948 [0.886, 0.991] |
| pca_residual | 0.505 [0.373, 0.639] | 0.858 [0.756, 0.939] | 0.749 [0.632, 0.858] | 0.934 [0.866, 0.983] |
| rnd | 0.467 [0.335, 0.602] | 0.824 [0.722, 0.916] | 0.798 [0.685, 0.899] | 0.955 [0.901, 0.992] |
| temporal_rnd | 0.581 [0.438, 0.725] | 0.882 [0.790, 0.959] | 0.758 [0.634, 0.871] | 0.928 [0.851, 0.984] |

## Alarm-timing diagnostics on the gradual arm C2

`frac_alarm_at_reset` is meaningful here in a way it is not under the cube mechanism, because the gradual arm is a no-op at frame 0 by construction. `frac_alarm_after_onset` is the fraction of first-alarms landing at or after that episode's own noise onset; a detector tracking the injected drift should be near 1.0.

| Detector | % alarms at reset (<=1) | % alarms after onset | alarm frame (median) | lead time mean (median) | % failures flagged >=10 frames early | never-flagged failures |
|---|---|---|---|---|---|---|
| phm_rolling_spread | 0.00 | 1.00 | 19.0 | 367.6 (380.0) | 0.45 | 16/29 |
| hotelling_t2 | 1.00 | 0.07 | 0.0 | 399.0 (399.0) | 1.00 | 0/29 |
| hotelling_t2_masked | 0.00 | 1.00 | 71.0 | 296.9 (328.0) | 0.69 | 9/29 |
| mahalanobis | 0.00 | 0.93 | 21.0 | 379.2 (378.0) | 1.00 | 0/29 |
| relative_mahalanobis | 0.00 | 0.93 | 14.0 | 382.0 (385.0) | 1.00 | 0/29 |
| knn | 0.00 | 1.00 | 20.0 | 378.5 (379.0) | 1.00 | 0/29 |
| pca_residual | 0.00 | 0.93 | 21.0 | 378.7 (378.0) | 1.00 | 0/29 |
| rnd | 0.00 | 0.93 | 14.0 | 380.8 (385.0) | 1.00 | 0/29 |
| temporal_rnd | 0.00 | 0.48 | 4.0 | 389.3 (395.0) | 1.00 | 0/29 |

Ceiling: an onset oracle that alarms exactly at each episode's true noise-onset frame achieves a mean lead time of 392.8 frames (median 393.0) over 29 failures. Lead time under this definition is terminal frame minus first-alarm frame, so a detector that fires early in a horizon-length episode scores a large number almost mechanically; the oracle is what that number looks like with perfect knowledge of the perturbation.


## Verdict against the pre-registered kill criteria

1. Primary. `phm_rolling_spread` on the C2 dose-matched contrast at k=30: AUROC 0.545 [0.406, 0.682]. The pre-registered kill criterion fires when that CI contains 0.5: **FIRED, the claim is dead**.

2. Secondary sweep. Detectors whose C2 dose-matched k=30 CI lower bound exceeds 0.55: none. With none surviving, the registered conclusion stands: neither induction mechanism supports a pre-failure early-warning claim on this policy, and the Week-2 gate PASS recorded on 2026-06-16 is withdrawn as a mechanism artifact.

3. Confound verdict. The registered contrast is nominal-eval against the C1-failed episodes. Detectors scoring AUROC at or above 0.95 there while their C2 dose-matched k=30 CI contains 0.5: mahalanobis, relative_mahalanobis, knn, pca_residual, rnd, temporal_rnd. For these, the gradual mechanism's headline AUROC is a perturbation-presence confound, structurally the same error as the reset confound it was built to remove. Every static feature-space detector in the registered set is in this list; the two that are not, `phm_rolling_spread` and `hotelling_t2`, are excluded because they never reached 0.95 on the cross-arm contrast in the first place, not because they survived the dose-matched one.

   Applying the same rule to the C2 cross-arm contrast instead of C1 narrows the list to temporal_rnd, because the C2 arm's lower dose pulls most detectors below 0.95 on the cross-arm contrast too. The registered C1 wording above is the one that carries the verdict.

4. Pipeline validity. Positive control, arm D under the original cube-perturbation mechanism through this same scoring code: mahalanobis 0.974 [0.950, 0.993]; knn 0.979 [0.955, 0.996]. Pre-registered requirement is AUROC at or above 0.95 for both: **PASS**.


## Deviations from the pre-registration

Deviations 1 and 2 were introduced during a pre-run shakedown at n=6 per arm, before any evaluation-arm result was seen. What is checkable from the repository is the ordering: both are committed in the branch history at commits dated before the timestamps on this run's cache and outputs, so neither could have been written in response to the numbers below. The shakedown itself left no artifact, so the n=6 detail rests on the author's record rather than on evidence in the repository. Deviation 3 is an arm that was registered and not run.

1. The dose audit is computed over the first 30 frames as well as over the whole episode. The registration specified the whole-episode integral only, which the shakedown showed is confounded with episode duration in exactly the direction that would fake a dose mismatch.

2. A ninth detector, `hotelling_t2_masked`, is reported alongside the eight registered ones. `hotelling_t2_window` emits the same saturated constant at frame 0 for every stream regardless of content, because a one-sample window has exactly zero variance and the score is an absolute log-ratio against the ID variance. Under a max-over-frames episode statistic that constant is the episode score for every episode, which pins the detector at AUROC exactly 0.500 and `frac_alarm_at_reset` exactly 1.00 by construction rather than by measurement. That is the explanation for the unresolved 0.500-against-0.982 discrepancy noted in `RESULTS_ROBOMIMIC.md`: it is an aggregation artifact, not a property of the policy or of either induction mechanism. The masked variant NaNs the first 20 frames, where the detector is undefined. The unmodified detector is still reported, and the masked variant is exploratory and excluded from the pre-registered eight-detector sweep in criterion 2.

3. Arm F, the replication of the dose-matched arm at seed offset +100000, was registered and has not been run. Every number in this file therefore comes from a single seed block. The runner supports the arm (`--seed-offset 100000 --dose-sigma 1.0`, which skips the pilot and reuses the selected dose), so this is an unpaid cost rather than a design change. The primary result is a null whose CI already spans chance, and a replication would tighten rather than reverse it, but until arm F is run the null is single-block.


## Reproducing this run

The rollout cache under `benchmark/real/cache/` and the model-zoo checkpoint under `benchmark/real/models/` are gitignored: they are large regenerable binaries, not source. The full run, which collects every arm from scratch and writes both this file and `benchmark/results_robomimic_gradual.json`, is

```
python benchmark/real/robomimic_eval_gradual.py
```
with the defaults recorded in this run's metadata: seed offset 0, 60 calibration, 60 nominal-eval and 70 induced episodes per arm, high-dose sigma 3.05, and the dose-matched sigma left to the pilot, which selected 1.0. Seed blocks are derived from the offset, so the episode set is fixed by that one number. Rollout collection is the expensive step; it writes `benchmark/real/cache/arms_gradual_seed0.npz`, after which `--use-cache` reruns scoring and rewrites both artifacts in seconds. Every detector is deterministic and the bootstrap is seeded, so a cached rerun reproduces both artifacts byte for byte.

