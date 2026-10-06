# Reliability Benchmark: PHM OOD detector vs canonical baselines

Detector under test: the PHM internal-feature OOD score (`phm_core::rolling_spread` + `phm_core::calibrate_threshold`), the windowed trace of the policy hidden-state covariance. Lower spread = more OOD; the harness negates it to the common higher-is-OOD convention before scoring (`phm_bench::phm_scores`).

Baselines (reimplemented from supercombo-blindspot `src/baselines.py`): Mahalanobis (Lee et al. 2018), Relative Mahalanobis (Ren et al. 2021), KNN k=50 (Sun et al. 2022). RND (Burda et al. 2019) is the 4th method, in two forms: a closed-form ridge predictor and a gradient-trained MLP predictor (Adam, 300 full-batch steps).

## Not-applicable baselines (regression / embedding setting)

| Baseline | Applies? | Reason |
|---|---|---|
| MSP | No | No softmax classification head over a closed label set; the stream is raw policy embeddings. |
| Energy | No | No logits to logsumexp; there is no classifier head on the embedding. |
| ViM | No | Requires a classifier weight matrix + logits; neither exists for an embedding stream. |

These three are N/A here for the same structural reason as in supercombo-blindspot (`src/baselines.py:60-107`): there are no classifier logits, only an internal feature vector.

Metrics are threshold-free (AUROC, AUPR, FPR@95TPR) with stratified-bootstrap 95% CIs (1000 resamples). Latency is the median per-frame wall-clock cost (fit + score over the full evaluation stream, divided by frame count) over 30 repeats (3 for the gradient-trained RND), single-threaded C++ on the machine that generated this file. AUROC, AUPR and FPR@95 follow the sklearn definitions and are unit-tested against hand-computed fixtures (`benchmark/test/test_metrics.cpp`).

## Scenario: collapse OOD (frozen low-variance embedding)

- OOD mode: `collapse`  dim=64, n_id=600, n_ood=600, window=20, in_dist_scale=1.0, ood_scale=0.02, ood_shift=4.0, ar_rho=0.85, seed=42
- Mean rolling-spread (window=20): ID=52.9203, OOD=0.0241

| Detector | AUROC (95% CI) | AUPR (95% CI) | FPR@95 (95% CI) | Latency (us/frame, median) |
|---|---|---|---|---|
| PHM rolling-spread | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.54 |
| Mahalanobis | 0.075 [0.055, 0.095] | 0.322 [0.318, 0.327] | 0.932 [0.912, 0.952] | 0.56 |
| Relative Mahalanobis | 0.117 [0.092, 0.146] | 0.331 [0.327, 0.339] | 0.890 [0.863, 0.913] | 1.67 |
| KNN (k=50, L2-normalized) | 0.419 [0.382, 0.459] | 0.417 [0.404, 0.433] | 0.628 [0.588, 0.667] | 5.23 |
| KNN (k=50, unnormalized) | 0.030 [0.018, 0.044] | 0.313 [0.311, 0.316] | 0.972 [0.958, 0.985] | 5.13 |
| RND (closed-form) | 0.026 [0.015, 0.040] | 0.312 [0.310, 0.315] | 0.980 [0.967, 0.990] | 3.68 |
| RND (MLP, gradient-trained) | 0.029 [0.017, 0.043] | 0.313 [0.311, 0.316] | 0.975 [0.962, 0.985] | 380.53 |

## Scenario: shift OOD (mean-shifted embedding region)

- OOD mode: `shift`  dim=64, n_id=600, n_ood=600, window=20, in_dist_scale=1.0, ood_scale=0.02, ood_shift=4.0, ar_rho=0.85, seed=42
- Mean rolling-spread (window=20): ID=52.9203, OOD=26.1626

| Detector | AUROC (95% CI) | AUPR (95% CI) | FPR@95 (95% CI) | Latency (us/frame, median) |
|---|---|---|---|---|
| PHM rolling-spread | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.52 |
| Mahalanobis | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.46 |
| Relative Mahalanobis | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.37 |
| KNN (k=50, L2-normalized) | 0.095 [0.076, 0.115] | 0.323 [0.320, 0.328] | 0.968 [0.955, 0.983] | 4.39 |
| KNN (k=50, unnormalized) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 4.67 |
| RND (closed-form) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 3.68 |
| RND (MLP, gradient-trained) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 381.33 |

## Headline

The two scenarios separate the two failure families. On the **collapse** scenario the PHM rolling-spread detector scores AUROC 1.000 (FPR@95 0.000) while every location-based baseline scores AUROC 0.03 to 0.42. This is not a harness bug: the collapsed cluster freezes around a real ID anchor point that sits INSIDE the ID cloud (anchor-to-ID-mean distance 7.75 vs median ID radius 9.51, evaluation stream seed 43), so a distance / density score assigns it a LOWER (more-in-distribution) value than typical spread-out ID frames (Mahalanobis median ID 122 vs collapsed-OOD 83). A collapse is a SECOND-order anomaly (variance drops, location does not move), and first-order location detectors are structurally blind to it. On the **shift** scenario the location detectors (Mahalanobis, RMD, unnormalized KNN, both RND forms) and the PHM detector all separate the streams, because the shifted region's within-window spread also differs.

## Notes

- The PHM rolling-spread detector targets the collapse / frozen-embedding failure (second-order: variance drops). It is location-invariant by construction (watches the trace of the windowed covariance, not absolute position).
- Mahalanobis / RMD / KNN target location shift (first-order: the embedding moves to a new region). On a pure collapse with no mean shift they are at-or-below chance; on a shift they are strong. The two scenarios make this contrast explicit.
- L2-normalized KNN (Sun et al. 2022 default, the supercombo-blindspot default) projects onto the unit sphere and discards the radial magnitude. It is at-or-below chance on BOTH scenarios here because the synthetic shift lives in magnitude; the unnormalized variant recovers it. Both are reported so this is visible rather than buried.
- RND captures the shift (closed-form ridge predictor and gradient-trained MLP predictor agree) but NOT the collapse, for the same in-cloud-anchor reason as the distance baselines: the predictor reproduces the target well on the in-distribution anchor.
- Latency: the closed-form detectors cost 0.46 to 5.23 us/frame (fit + score amortised over the stream). The gradient-trained RND costs 381 us/frame because each fit trains an MLP for 300 Adam steps; it is reported for correctness corroboration, not as a latency contender.
- Both scenarios are synthetic: these numbers bound the detectors' structural coverage, not their accuracy on a real policy.
