# benchmark/lib/labeling.py
"""Objective degeneration labels for a generated token stream.

Failure = degenerate repetition, detected from the token ids alone (NOT from any
OOD score), so the ground truth is independent of every detector under test.

Definitions:
- ngram_repetition_rate(tokens, n, window): per-position fraction of n-grams in
  the trailing `window` that have already appeared earlier in that same window.
  High rate = the model is looping.
- failure_labels(tokens, n, window, threshold): label 1 where the smoothed
  repetition rate >= threshold; onset = first failing index (or -1 if none).
"""
from __future__ import annotations

from collections import Counter

import numpy as np


def ngram_repetition_rate(tokens, n: int = 3, window: int = 20) -> np.ndarray:
    toks = list(tokens)
    T = len(toks)
    rate = np.zeros(T, dtype=np.float64)
    for i in range(T):
        lo = max(0, i - window + 1)
        win = toks[lo : i + 1]
        grams = [tuple(win[j : j + n]) for j in range(len(win) - n + 1)]
        if not grams:
            rate[i] = 0.0
            continue
        counts = Counter(grams)
        # fraction of n-gram slots occupied by a gram that appears more than once
        repeating_slots = sum(cnt for cnt in counts.values() if cnt > 1)
        rate[i] = repeating_slots / len(grams)
    return rate


def failure_labels(tokens, n: int = 3, window: int = 20, threshold: float = 0.5):
    rate = ngram_repetition_rate(tokens, n=n, window=window)
    labels = (rate >= threshold).astype(np.int64)
    onset_idx = np.argmax(labels) if labels.any() else -1
    onset = int(onset_idx) if labels.any() else -1
    return labels, onset
