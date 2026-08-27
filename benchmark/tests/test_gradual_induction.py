"""Tests for the gradual-onset OOD induction mechanism and its analysis controls.

The defining property of this mechanism is that it is a no-op at episode reset:
that is the whole reason it can be used to cross-check the reset-confounded
initial-state mechanism. The first three tests pin exactly that, at the level of
the ramp schedule and at the level of the injected observation.
"""

import numpy as np
from lib.gradual_induction import (
    dose_integral,
    dose_match_report,
    episode_max,
    inject_noise,
    oracle_lead_times,
    ramp_fraction,
)

KEYS = ["a", "b"]


def _std():
    return {"a": np.array([1.0, 2.0, 4.0]), "b": np.array([0.5])}


# --- the no-op-at-reset property ------------------------------------------


def test_ramp_is_exactly_zero_at_frame_zero_for_every_onset():
    # frac_alarm_at_reset is only a meaningful discriminator if frame 0 carries
    # zero injected signal for every episode in the arm, including onset=0.
    for onset in range(0, 16):
        assert ramp_fraction(0, onset, ramp_len=15) == 0.0


def test_ramp_is_zero_strictly_before_onset_and_one_after_full_ramp():
    onset, ramp_len = 7, 15
    assert [ramp_fraction(t, onset, ramp_len) for t in range(onset)] == [0.0] * onset
    assert ramp_fraction(onset, onset, ramp_len) == 0.0
    assert ramp_fraction(onset + ramp_len, onset, ramp_len) == 1.0
    assert ramp_fraction(onset + 5 * ramp_len, onset, ramp_len) == 1.0


def test_ramp_is_monotonic_nondecreasing():
    vals = [ramp_fraction(t, onset=3, ramp_len=15) for t in range(60)]
    assert all(b >= a for a, b in zip(vals, vals[1:], strict=False))


def test_inject_noise_is_a_bit_identical_noop_at_zero_fraction():
    rng = np.random.default_rng(0)
    obs = {"a": np.array([1.0, 2.0, 3.0]), "b": np.array([4.0])}
    out = inject_noise(obs, _std(), KEYS, sigma_scale=3.0, frac=0.0, rng=rng)
    for k in KEYS:
        assert np.array_equal(out[k], obs[k])
    # rng must not have been consumed, so a downstream draw is unaffected by
    # whether a zero-noise frame happened to pass through the injector.
    assert rng.normal() == np.random.default_rng(0).normal()


def test_inject_noise_perturbs_once_the_ramp_is_active():
    rng = np.random.default_rng(0)
    obs = {"a": np.zeros(3), "b": np.zeros(1)}
    out = inject_noise(obs, _std(), KEYS, sigma_scale=3.0, frac=1.0, rng=rng)
    assert not np.array_equal(out["a"], obs["a"])
    assert out["a"].shape == obs["a"].shape


def test_inject_noise_scales_each_channel_by_its_own_std():
    # Channel 2 of key "a" has 4x the std of channel 0, so its injected noise
    # should have roughly 4x the spread. Checked over many draws.
    rng = np.random.default_rng(1)
    obs = {"a": np.zeros(3), "b": np.zeros(1)}
    draws = np.stack(
        [inject_noise(obs, _std(), KEYS, 3.0, 1.0, rng)["a"] for _ in range(4000)]
    )
    ratio = draws[:, 2].std() / draws[:, 0].std()
    assert 3.6 < ratio < 4.4, ratio


def test_inject_noise_leaves_unlisted_keys_alone():
    rng = np.random.default_rng(0)
    obs = {"a": np.zeros(3), "b": np.zeros(1), "untouched": np.ones(2)}
    out = inject_noise(obs, _std(), KEYS, 3.0, 1.0, rng)
    assert np.array_equal(out["untouched"], obs["untouched"])


# --- the exposure control --------------------------------------------------


def test_episode_max_full_window_matches_plain_max():
    s = np.array([0.1, 5.0, 0.2])
    assert episode_max(s) == 5.0


def test_episode_max_fixed_window_ignores_later_frames():
    # This is the exposure control: a long failed episode must not be credited
    # for chances to trip that a short successful episode never got.
    s = np.array([0.1, 0.2, 0.3, 99.0])
    assert episode_max(s, k=3) == 0.3
    assert episode_max(s, k=None) == 99.0


def test_episode_max_handles_short_streams_and_nans():
    assert np.isnan(episode_max(np.zeros(0)))
    assert np.isnan(episode_max(np.array([np.nan, np.nan]), k=2))
    assert episode_max(np.array([np.nan, 2.0]), k=2) == 2.0
    # k longer than the stream is not an error, it just uses what is there.
    assert episode_max(np.array([1.0, 3.0]), k=30) == 3.0


# --- the dose audit --------------------------------------------------------


def test_dose_integral_is_the_mean_active_fraction():
    assert dose_integral([0.0, 0.0, 1.0, 1.0]) == 0.5
    assert dose_integral([]) == 0.0


def test_dose_match_report_flags_a_mismatched_arm():
    eps = [
        {"success": False, "sigma_trace": np.ones(10)},
        {"success": True, "sigma_trace": np.full(10, 0.5)},
    ]
    r = dose_match_report(eps)
    assert r["n_failed"] == 1 and r["n_succeeded"] == 1
    assert abs(r["rel_gap"] - 0.5) < 1e-12


def test_dose_match_report_passes_a_matched_arm():
    eps = [
        {"success": False, "sigma_trace": np.full(10, 0.80)},
        {"success": True, "sigma_trace": np.full(10, 0.78)},
    ]
    assert dose_match_report(eps)["rel_gap"] < 0.10


def test_dose_match_report_is_nan_when_a_group_is_empty():
    eps = [{"success": False, "sigma_trace": np.ones(5)}]
    assert np.isnan(dose_match_report(eps)["rel_gap"])


# --- the oracle ceiling ----------------------------------------------------


def test_oracle_lead_times_measure_terminal_minus_onset():
    eps = [
        {"success": False, "steps": 400, "onset_frame": 10},
        {"success": True, "steps": 40, "onset_frame": 3},
        {"success": False, "steps": 400, "onset_frame": 0},
    ]
    assert oracle_lead_times(eps) == [389, 399]


def test_oracle_lead_times_skip_episodes_without_an_onset():
    eps = [{"success": False, "steps": 100, "onset_frame": None}]
    assert oracle_lead_times(eps) == []
