# Real-embedding benchmark (microsoft/phi-2)

Per-token last-layer hidden states from phi-2 (2.7B, fp16, RTX 5070). ID = coherent generation; collapse = degenerate repetition (objective n-gram label, per-frame); shift = out-of-distribution prompt (whole stream OOD). Fit on healthy(seed S), ID test = healthy(seed S+1) so the fit set is never scored. Streams scored separately. Lead-time at the healthy-calibrated FPR~5% operating point (frames; positive = warns before onset).

Seeds: [0, 1, 2, 3, 4] (n=5)

| Detector | Condition | AUROC (mean +/- std) | Lead-time (mean frames) | n |
|---|---|---|---|---|
| PHM rolling-spread | collapse | 1.000 +/- 0.000 | -16.0 | 5 |
| PHM rolling-spread | shift | 0.898 +/- 0.149 | n/a | 5 |
| Mahalanobis | collapse | 0.236 +/- 0.086 | -58.7 | 5 |
| Mahalanobis | shift | 0.593 +/- 0.218 | n/a | 5 |
| Relative Mahalanobis | collapse | 0.621 +/- 0.112 | -13.8 | 5 |
| Relative Mahalanobis | shift | 0.597 +/- 0.365 | n/a | 5 |
| KNN (k=50, L2-normalized) | collapse | 0.337 +/- 0.132 | -34.2 | 5 |
| KNN (k=50, L2-normalized) | shift | 0.852 +/- 0.145 | n/a | 5 |
| KNN (k=50, unnormalized) | collapse | 0.159 +/- 0.094 | -35.0 | 5 |
| KNN (k=50, unnormalized) | shift | 0.584 +/- 0.213 | n/a | 5 |
| RND (numpy) | collapse | 0.468 +/- 0.096 | -42.4 | 5 |
| RND (numpy) | shift | 0.546 +/- 0.140 | n/a | 5 |
| PCA-residual (k=8) | collapse | 0.221 +/- 0.094 | -48.2 | 5 |
| PCA-residual (k=8) | shift | 0.573 +/- 0.181 | n/a | 5 |
| Hotelling-T2 (windowed) | collapse | 0.982 +/- 0.000 | 3.0 | 5 |
| Hotelling-T2 (windowed) | shift | 0.892 +/- 0.118 | n/a | 5 |
| Temporal-RND (w=8) | collapse | 0.338 +/- 0.090 | -51.8 | 5 |
| Temporal-RND (w=8) | shift | 0.719 +/- 0.116 | n/a | 5 |
