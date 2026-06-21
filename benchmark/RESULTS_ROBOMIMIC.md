# Robomimic real-policy failure-prediction benchmark (lift, BC-RNN-GMM)

Pretrained `lift_ph_low_dim_epoch_1000_succ_100.pth` (robomimic model zoo, BC_RNN_GMM, 2-layer LSTM hidden=400). Feature = LSTM per-step hidden output (400-dim), captured via read-only forward hook, never resetting recurrent state mid-episode. CPU rollout collection (system torch 2.11.0+cu128, robomimic installed from GitHub master 0.5.0, robosuite==1.4.1 pinned for compatibility with this pre-composite-controller checkpoint; see blockers section below).

Calibration-fit: 60 nominal episodes (cube spawn half_extent=0.03). Nominal-eval: 60 disjoint nominal episodes. Induced-failure eval: 70 episodes with cube spawn half_extent=0.18 (6x nominal), empirical failure rate 0.44 (31/70 failed). All three splits use disjoint seed ranges; detectors fit ONLY on calibration-fit features.

Lead-time operating point: per-detector threshold fixed at the (1 - 0.05) quantile of frame-level scores pooled over all nominal-eval frames (~5% healthy-calibrated FPR). Lead time = terminal-frame index minus first-alarm frame index in the failed episode (positive = warned before the episode ended without success); `n/a` if the detector never crossed threshold in that episode.

**Caveat on lead time with this OOD-induction mechanism.** The induced-failure condition perturbs the cube's initial placement (`half_extent=0.18` vs `0.03` nominal), which is itself a distribution shift present from frame 0 of the episode, not a precursor that builds up gradually before failure. A detector with a large `lead_time_mean` that fires at `alarm_frame_median` near 0 is reporting **"this episode looks OOD from the start"**, not **"I foresaw the failure N frames before it happened."** The `frac_alarm_at_reset` column reports the fraction of first-alarms occurring at frame <=1; high values there mean the lead-time number should NOT be read as a gradual pre-failure early-warning result for this detector under this induction mechanism, even though the arithmetic is correct.

## Episode-level failure-prediction metrics

| Detector | AUROC (95% CI) | AUPR (95% CI) | FPR@95TPR | Lead-time mean (median) frames | Alarm frame (median) | % alarms at reset (<=1) | % failures flagged >=10 frames early | Never-flagged failures |
|---|---|---|---|---|---|---|---|---|
| phm_rolling_spread | 0.510 [0.376, 0.658] | 0.374 [0.251, 0.520] | 0.960 | 365.0 (377.0) | 22.0 | 0.00 | 0.45 | 17/31 |
| hotelling_t2 | 0.500 [0.500, 0.500] | 0.238 [0.238, 0.238] | 1.000 | 399.0 (399.0) | 0.0 | 1.00 | 1.00 | 0/31 |
| mahalanobis | 0.965 [0.936, 0.987] | 0.907 [0.833, 0.965] | 0.162 | 398.8 (399.0) | 0.0 | 1.00 | 1.00 | 0/31 |
| relative_mahalanobis | 0.968 [0.939, 0.990] | 0.900 [0.806, 0.970] | 0.172 | 398.0 (398.0) | 1.0 | 1.00 | 1.00 | 0/31 |
| knn | 0.986 [0.968, 0.999] | 0.959 [0.905, 0.997] | 0.030 | 398.9 (399.0) | 0.0 | 0.97 | 1.00 | 0/31 |
| pca_residual | 0.972 [0.948, 0.992] | 0.923 [0.858, 0.975] | 0.162 | 398.8 (399.0) | 0.0 | 1.00 | 1.00 | 0/31 |
| rnd | 0.982 [0.963, 0.995] | 0.949 [0.891, 0.987] | 0.091 | 398.9 (399.0) | 0.0 | 1.00 | 1.00 | 0/31 |
| temporal_rnd | 0.971 [0.945, 0.991] | 0.920 [0.847, 0.973] | 0.172 | 397.9 (398.0) | 1.0 | 0.87 | 1.00 | 0/31 |

## Gate check (30_day_validation_plan.md Week 2 mid-point gate)

Gate: AUROC >= 0.70 with bootstrap 95% CI lower bound > 0.55 for at least one second-order detector (Hotelling-T2, PHM rolling-spread, PCA-residual, temporal-RND).

- phm_rolling_spread: AUROC 0.510 [0.376, 0.658] -> FAIL
- hotelling_t2: AUROC 0.500 [0.500, 0.500] -> FAIL
- pca_residual: AUROC 0.972 [0.948, 0.992] -> PASS
- temporal_rnd: AUROC 0.971 [0.945, 0.991] -> PASS

**Gate verdict: PASS**

**Important nuance on this PASS.** `pca_residual` and `temporal_rnd` clear the AUROC gate, but per the table above both have `frac_alarm_at_reset` >= 0.87, meaning their AUROC is driven almost entirely by detecting the perturbed initial cube position at episode start, not by a gradually building pre-failure signal. `phm_rolling_spread` is the only detector in this run whose alarms are NOT reset-driven (`frac_alarm_at_reset=0.00`, median alarm at frame 22) and it produces the only result in this report that supports a genuine early-warning claim (45% of failures flagged >=10 frames early, mean lead time 365 frames among those flagged) -- but it FAILS the AUROC gate (0.510 [0.376, 0.658], CI spans chance). `hotelling_t2` is flat chance (0.500 [0.500, 0.500]) and contributes nothing here, in sharp contrast to its 0.982 AUROC / +3.0 frame lead time on the phi-2 hidden-state benchmark in RESULTS_REAL.md. Read together: this run passes the letter of the Week 2 AUROC gate via detectors that are mostly detecting "OOD-at-reset," while the one detector that behaves like a genuine second-order pre-failure collapse detector (PHM rolling-spread) does not clear AUROC significance here. This is reported as-is, not reconciled into a single clean win.


## Headline

Naive best mean lead time (unadjusted): **hotelling_t2** at 399.0 frames (median 399.0), AUROC 0.500 [0.500, 0.500] (100% of its alarms fire at frame<=1, i.e. driven by the OOD-at-reset confound, not gradual pre-failure detection).

Best mean lead time AFTER excluding reset-confounded detectors (`frac_alarm_at_reset < 0.5`): **phm_rolling_spread** at 365.0 frames (median 377.0), AUROC 0.510 [0.376, 0.658], 45% of failures flagged >=10 frames early. This is the number that supports a genuine early-warning claim under this induction mechanism.


## Blockers encountered and resolved

- `robomimic==0.3.0` (PyPI) has an unguarded `import mujoco_py` in `env_robosuite.py`; the legacy `mujoco_py` package is not installed and was not targeted (modern `mujoco` 3.9.0 is used instead). Fixed by installing robomimic from GitHub master (0.5.0), whose `env_robosuite.py` guards the `mujoco_py` import in try/except.

- The checkpoint's saved config predates robomimic's transformer-policy support and is missing the `algo.transformer` key; robomimic 0.5.0's BC algo factory unconditionally reads `algo.transformer.enabled`, and `update_config()`'s backward-compat patch does not add this key. Fixed in `load_policy_and_env()` by injecting `{"enabled": False}` into the loaded config dict before `policy_from_checkpoint` -- this restores the schema default and does not alter model weights or behavior.

- robosuite 1.5.x introduced composite controllers as a breaking change; the old single-string `OSC_POSE` controller spec in this checkpoint's env_kwargs is not auto-registered under 1.5.x, raising `AssertionError: OSC_POSE controller is specified, but not imported or loaded`. Fixed by pinning `robosuite==1.4.1` (last pre-composite-controller release).

- GPU not used for rollout collection: the RTX 5070 was already running a live Phoenix PPO training job (`phoenix-stand-v3-dr-long`, ~9.5h elapsed, 6.4 GB VRAM, 69% util) at benchmark start. CPU-only rollout collection (robomimic lift BC-RNN is small; measured throughput below) avoided contending with that job, consistent with the task's CPU-fallback guidance.

