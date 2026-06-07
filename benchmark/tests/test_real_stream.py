# benchmark/tests/test_real_stream.py
"""Loader tests against the committed real-embedding fixture (no torch, no GPU)."""
import numpy as np
import pytest
from lib.real_stream import FIXTURE, load_real_stream

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="real-embedding fixture not present"
)


def test_load_returns_conditions_with_aligned_shapes():
    streams = load_real_stream(FIXTURE)
    assert set(streams) >= {"healthy", "collapse", "shift"}
    for s in streams.values():
        assert s.feats.ndim == 2
        assert s.feats.shape[0] == s.labels.shape[0]
        assert s.feats.dtype == np.float32
        assert isinstance(s.onset, int)


def test_collapse_has_more_failure_frames_than_healthy():
    streams = load_real_stream(FIXTURE)
    assert streams["collapse"].labels.sum() >= streams["healthy"].labels.sum()
