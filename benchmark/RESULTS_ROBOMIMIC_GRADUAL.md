# Robomimic gradual-noise-ramp failure-prediction benchmark (lift, BC-RNN-GMM)

Second OOD-induction mechanism for the same policy/feature/detector stack as `RESULTS_ROBOMIMIC.md`. Failure is induced by ramping Gaussian noise on the policy's observation vector (robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos, object) starting from a random per-episode onset frame and rising linearly to full sigma, rather than by perturbing the cube's initial position. Every episode (including failure-eval) is nominal at frame 0 by construction. Noise sigma scale = 3.05x the per-key, per-dimension std measured over nominal rollouts; onset_frame ~ U[0, 15) frames, ramp length = 15 frames to full noise, held thereafter.

Calibration-fit: 60 nominal episodes, zero injected noise. Nominal-eval: 60 disjoint nominal episodes, zero injected noise. Gradual-noise-ramp eval: 70 episodes with the ramp mechanism above, empirical failure rate 0.94 (66/70 failed). All three splits use disjoint seed ranges; detectors fit ONLY on calibration-fit features.

Lead-time operating point: identical to `RESULTS_ROBOMIMIC.md` -- per-detector threshold fixed at the (1 - 0.05) quantile of frame-level scores pooled over all nominal-eval frames (~5% healthy-calibrated FPR).

**New diagnostic for this mechanism: `frac_alarm_after_onset`.** Fraction of first-alarms (among failed episodes that were ever flagged) occurring at or after that episode's own noise-onset frame. A detector legitimately tracking the injected drift should have this near 1.0; a value well below 1.0 means the detector is alarming on something unrelated to the drift (or on episode structure that happens to correlate with eventual failure), not on the gradual OOD signal this mechanism is designed to test.

## Episode-level failure-prediction metrics (gradual-noise-ramp)

| Detector | AUROC (95% CI) | AUPR (95% CI) | FPR@95TPR | Lead-time mean (median) frames | Alarm frame (median) | % alarms at reset (<=1) | % alarms after onset | % failures flagged >=10 frames early | Never-flagged failures |
|---|---|---|---|---|---|---|---|---|---|
| phm_rolling_spread | 0.196 [0.129, 0.274] | 0.363 [0.341, 0.398] | 1.000 | 380.0 (380.0) | 19.0 | 0.00 | 1.00 | 0.09 | 60/66 |
| hotelling_t2 | 0.500 [0.500, 0.500] | 0.508 [0.508, 0.508] | 1.000 | 399.0 (399.0) | 0.0 | 1.00 | 0.05 | 1.00 | 0/66 |
| mahalanobis | 0.997 [0.991, 1.000] | 0.997 [0.991, 1.000] | 0.016 | 387.3 (388.0) | 11.0 | 0.00 | 0.98 | 1.00 | 0/66 |
| relative_mahalanobis | 0.990 [0.969, 1.000] | 0.986 [0.957, 1.000] | 0.016 | 387.4 (388.0) | 11.0 | 0.00 | 0.97 | 1.00 | 0/66 |
| knn | 0.988 [0.962, 1.000] | 0.980 [0.938, 1.000] | 0.016 | 385.9 (389.0) | 10.0 | 0.00 | 0.94 | 1.00 | 0/66 |
| pca_residual | 0.996 [0.989, 1.000] | 0.996 [0.988, 1.000] | 0.016 | 387.2 (388.0) | 11.0 | 0.00 | 0.94 | 1.00 | 0/66 |
| rnd | 0.993 [0.978, 1.000] | 0.992 [0.974, 1.000] | 0.016 | 387.6 (388.0) | 11.0 | 0.00 | 0.95 | 1.00 | 0/66 |
| temporal_rnd | 0.997 [0.989, 1.000] | 0.997 [0.989, 1.000] | 0.016 | 392.0 (396.0) | 3.0 | 0.00 | 0.50 | 1.00 | 0/66 |

## Headline (gradual mechanism only)

Best AUROC under gradual induction: **mahalanobis** at 0.997 [0.991, 1.000], frac_alarm_at_reset=0.00.

Best non-reset-confounded lead time under gradual induction: **temporal_rnd** at 392.0 frames (median 396.0), AUROC 0.997 [0.989, 1.000], 100% of failures flagged >=10 frames early.

