"""Gradual-onset OOD induction: mechanism and analysis primitives.

The initial-state induction mechanism used by the robomimic benchmark (widening
the cube spawn box) is out of distribution from frame 0, so a detector can score
a high AUROC purely by classifying the episode's starting state. This module
implements the second mechanism built to remove that confound: a Gaussian noise
ramp on the policy's observation vector that is a *no-op at reset* and grows
during the episode.

Everything here is pure numpy with no robomimic, robosuite or torch dependency,
so it is unit-testable without the simulator. The simulator-facing runner lives
in ``benchmark/real/robomimic_eval_gradual.py``.

Two properties matter and are enforced by tests:

1. ``ramp_fraction`` returns exactly 0.0 at frame 0 for every non-negative onset,
   so ``frac_alarm_at_reset`` becomes a meaningful discriminator rather than a
   giveaway.
2. ``inject_noise`` leaves the observation bit-identical when the ramp fraction
   is zero, so a zero-noise prefix is not merely small but absent.

The analysis primitives cover the two controls the original design lacked: a
fixed-window episode statistic that removes the episode-length exposure
confound, and a dose-match audit over the realized injected noise.
"""

from __future__ import annotations

import numpy as np


def ramp_fraction(t: int, onset: int, ramp_len: int) -> float:
    """Fraction (0..1) of full noise sigma active at frame ``t``.

    Zero strictly before ``onset``, rising linearly over ``ramp_len`` frames,
    held at 1.0 thereafter. With ``onset >= 0`` this is 0.0 at t=0 unless
    ``onset == 0 and ramp_len == 0``, which the runner never configures.
    """
    if t < onset:
        return 0.0
    prog = (t - onset) / max(ramp_len, 1)
    return float(np.clip(prog, 0.0, 1.0))


def inject_noise(obs, obs_std, keys, sigma_scale: float, frac: float, rng):
    """Return ``obs`` with Gaussian noise added to ``keys``, scaled per channel.

    Noise sigma for key k is ``sigma_scale * frac * obs_std[k]``, so each channel
    is corrupted relative to its own natural spread rather than by a shared
    absolute amount. When ``frac <= 0`` the input mapping is returned unchanged
    and ``rng`` is not consumed, which keeps the zero-noise prefix exactly
    identical to an uninjected rollout.
    """
    if frac <= 0.0:
        return obs
    out = dict(obs)
    for k in keys:
        if k not in out:
            continue
        sigma = sigma_scale * frac * np.asarray(obs_std[k], dtype=np.float64)
        out[k] = np.asarray(obs[k], dtype=np.float64) + rng.normal(0.0, sigma)
    return out


def episode_max(scores: np.ndarray, k: int | None = None) -> float:
    """Episode-level statistic: max detector score over the first ``k`` frames.

    ``k=None`` reproduces the legacy max-over-all-frames statistic. A finite
    ``k`` below the shortest nominal episode length equalises the number of
    chances every episode gets to cross threshold, which is the control the
    original design lacked: failed episodes run to the 400-frame horizon while
    successful ones terminate in roughly 35 to 50 frames, so a max over all
    frames gives failures about ten times as many opportunities to trip.

    Returns NaN for an empty stream, and for a stream with no finite score in the
    window.
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.size == 0:
        return float("nan")
    if k is not None:
        s = s[:k]
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("nan")
    return float(np.max(s))


def mask_warmup(scores: np.ndarray, n: int) -> np.ndarray:
    """NaN out the first ``n`` frames of a score stream.

    Some windowed detectors emit a degenerate score while their trailing window
    is still filling. ``hotelling_t2_window`` is the concrete case: at frame 0
    the window holds one sample, its within-window variance is exactly 0, and
    the absolute log-ratio against the ID variance saturates at the same large
    constant for EVERY stream regardless of content. Under a max-over-frames
    episode statistic that constant is the episode score for every episode, so
    the detector returns AUROC exactly 0.500 and frac_alarm_at_reset exactly
    1.00 by construction rather than by measurement. Masking the warm-up frames
    lets the detector be evaluated on the frames where it is actually defined.
    """
    s = np.asarray(scores, dtype=np.float64).copy()
    s[: max(n, 0)] = np.nan
    return s


def dose_integral(sigma_trace, k: int | None = None) -> float:
    """Mean injected sigma fraction over an episode, i.e. the realized dose.

    Used to audit whether the failed and succeeded episodes inside one
    perturbation arm actually received comparable corruption. If they did not,
    a within-arm outcome contrast is measuring dose, not prediction.

    ``k`` restricts the average to the first ``k`` frames. That restriction
    matters: averaging over the whole episode conflates dose with duration,
    because a successful episode ends after roughly 40 frames while its ramp is
    still climbing, whereas a failed episode runs to the horizon at full
    amplitude. The two groups then differ in mean dose even though the injection
    schedule was identical. Auditing over a fixed prefix compares like with like.
    """
    tr = np.asarray(sigma_trace, dtype=np.float64)
    if k is not None:
        tr = tr[:k]
    if tr.size == 0:
        return 0.0
    return float(np.mean(tr))


def dose_match_report(episodes, k: int | None = None) -> dict:
    """Realized dose by outcome group for one perturbation arm.

    Returns per-group mean dose and episode counts plus ``rel_gap``, the relative
    difference in mean dose between failed and succeeded episodes. The
    pre-registered blocker is ``rel_gap > 0.10``: above that the arm is not
    dose-matched and its within-arm contrast cannot be read as a prediction
    result. ``rel_gap`` is NaN when either group is empty.
    """
    failed = [dose_integral(e.get("sigma_trace", []), k) for e in episodes if not e["success"]]
    succ = [dose_integral(e.get("sigma_trace", []), k) for e in episodes if e["success"]]
    m_f = float(np.mean(failed)) if failed else float("nan")
    m_s = float(np.mean(succ)) if succ else float("nan")
    if not failed or not succ or max(abs(m_f), abs(m_s)) == 0.0:
        gap = float("nan")
    else:
        gap = abs(m_f - m_s) / max(abs(m_f), abs(m_s))
    return {
        "n_failed": len(failed),
        "n_succeeded": len(succ),
        "mean_dose_failed": m_f,
        "mean_dose_succeeded": m_s,
        "rel_gap": gap,
    }


def oracle_lead_times(episodes) -> list[int]:
    """Lead time an onset oracle would achieve on each failed episode.

    The oracle alarms exactly at that episode's true injected-noise onset frame,
    so its lead time is ``terminal_frame - onset_frame``. This is the ceiling on
    what perfect knowledge of the perturbation buys under this mechanism, and it
    bounds how much of a detector's lead time is attributable to the drift at all.
    """
    out = []
    for e in episodes:
        if e["success"]:
            continue
        onset = e.get("onset_frame")
        if onset is None:
            continue
        out.append(int(max(e["steps"] - 1 - onset, 0)))
    return out
