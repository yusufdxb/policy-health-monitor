"""Generator property tests: ID spread > OOD spread for collapse; shape config."""

import numpy as np

from lib.generator import StreamSpec, generate_stream, rolling_spread_trace


def test_collapse_id_spread_greater_than_ood_spread():
    spec = StreamSpec(dim=64, n_id=300, n_ood=300, ood_mode="collapse", seed=1)
    id_frames, ood_frames = generate_stream(spec)
    s_id = rolling_spread_trace(id_frames, window=20)
    s_ood = rolling_spread_trace(ood_frames, window=20)
    # The collapse OOD stream must have markedly lower rolling spread.
    assert s_id > s_ood
    assert s_ood < 0.1 * s_id


def test_shapes_configurable():
    spec = StreamSpec(dim=32, n_id=120, n_ood=80, ood_mode="shift", seed=2)
    id_frames, ood_frames = generate_stream(spec)
    assert id_frames.shape == (120, 32)
    assert ood_frames.shape == (80, 32)


def test_shift_moves_the_mean():
    spec = StreamSpec(dim=48, n_id=400, n_ood=400, ood_mode="shift",
                      ood_shift=4.0, seed=3)
    id_frames, ood_frames = generate_stream(spec)
    # The shifted OOD stream sits far from the ID mean in L2.
    id_center = id_frames.mean(axis=0)
    ood_center = ood_frames.mean(axis=0)
    sep = np.linalg.norm(ood_center - id_center)
    id_spread = np.sqrt(rolling_spread_trace(id_frames, 20))
    assert sep > id_spread  # shift exceeds the within-ID scale


def test_unknown_ood_mode_raises():
    spec = StreamSpec(ood_mode="bogus")
    try:
        generate_stream(spec)
    except ValueError as e:
        assert "bogus" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown ood_mode")


def test_deterministic_under_seed():
    a, b = generate_stream(StreamSpec(seed=7))
    c, d = generate_stream(StreamSpec(seed=7))
    assert np.array_equal(a, c)
    assert np.array_equal(b, d)
