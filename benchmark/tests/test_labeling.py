# benchmark/tests/test_labeling.py
"""Tests for the objective degeneration-labeling criterion.

The label is computed from generated token ids only, independent of any OOD
detector, so no detector grades its own homework.
"""
import numpy as np
from lib.labeling import ngram_repetition_rate, failure_labels


def test_repetition_rate_all_repeat_is_one():
    # token 7 repeated -> every 1-gram in the window already seen -> rate 1.0
    toks = [7] * 30
    rate = ngram_repetition_rate(toks, n=1, window=10)
    assert rate[-1] == 1.0


def test_repetition_rate_all_unique_is_zero():
    toks = list(range(30))
    rate = ngram_repetition_rate(toks, n=1, window=10)
    # first token of a window can't repeat anything before it within the window
    assert rate[-1] == 0.0


def test_failure_labels_marks_repeat_tail_and_onset():
    # 20 unique tokens, then a degenerate loop of a single token
    toks = list(range(20)) + [99] * 20
    labels, onset = failure_labels(toks, n=3, window=10, threshold=0.5)
    assert labels.shape == (40,)
    assert labels[:20].sum() == 0           # coherent prefix is healthy
    assert labels[30:].all()                # deep in the loop is failure
    assert 20 <= onset <= 32                 # onset lands at the transition


def test_failure_labels_all_healthy_onset_is_minus_one():
    toks = list(range(40))
    labels, onset = failure_labels(toks, n=3, window=10, threshold=0.5)
    assert labels.sum() == 0
    assert onset == -1
