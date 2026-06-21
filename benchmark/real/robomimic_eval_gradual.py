"""Second OOD-induction mechanism for the robomimic failure-prediction benchmark:
progressive observation-noise drift (gradual, builds DURING the episode).

This is a sibling to robomimic_eval.py, reusing every other component unchanged
(policy, feature hook, detector set, calibrate/nominal-eval/failure-eval split
discipline, bootstrap-CI metrics, lead-time + frac_alarm_at_reset diagnostics).
The only thing that changes is HOW failure is induced.

robomimic_eval.py's mechanism (cube initial-position perturbation, half_extent
0.18 vs 0.03) is OOD from frame 0 of the episode: static feature-space OOD
detectors (knn/rnd/mahalanobis/pca/temporal_rnd) score AUROC 0.96-0.99 but
~100% of their alarms fire at the reset frame (frac_alarm_at_reset~=1.0). They
are detecting "this episode started weird," not "this episode is about to fail."
The only detector with a genuine pre-failure signature in that run
(phm_rolling_spread: frac_alarm_at_reset=0.00, 45% of failures flagged
>=10 frames early) is at chance on AUROC (0.510).

This script induces failure differently: the episode starts at nominal (zero
injected noise at frame 0, observation distribution matches calib-fit exactly)
and Gaussian noise on the policy's observation vector ramps up linearly from a
random onset frame to the episode horizon. The injected-noise condition is
therefore IN-DISTRIBUTION at reset by construction; any OOD signal a detector
picks up must come from the accumulating drift, not from a different starting
state. This directly tests whether the reset-confounded detectors lose their
edge and whether the genuine precursor detectors (phm_rolling_spread,
hotelling_t2) recover positive AUROC with real (non-reset) lead time.

Noise injection point: added directly to the four raw obs dict values robomimic
exposes to RolloutPolicy.__call__ (robot0_eef_pos, robot0_eef_quat,
robot0_gripper_qpos, object) -- i.e. simulating sensor-degradation noise on
exactly what the policy "sees," before any internal normalization. Per-key
noise sigma is scaled relative to that key's per-dimension std measured over
the calibration-fit nominal rollouts (NOISE_SIGMA_SCALE x that std at full
ramp), so the same relative corruption applies to position, orientation,
gripper, and object-pose channels despite their different natural ranges.

Ramp schedule per episode: onset_frame ~ Uniform[0, RAMP_ONSET_MAX_FRAMES)
(random per-episode so the OOD-onset frame is not perfectly aligned across
the failure population -- a fixed onset would let any window-based detector
"cheat" by keying on absolute time rather than the drift itself). After onset,
injected sigma rises linearly from 0 to NOISE_SIGMA_SCALE over
RAMP_LENGTH_FRAMES frames, then holds at NOISE_SIGMA_SCALE for any remaining
frames. Ramp timing is in ABSOLUTE frames, not a fraction of HORIZON=400,
because nominal Lift episodes succeed in ~35-50 frames -- far short of the
horizon -- so a horizon-relative ramp never builds up before the episode
would have already ended (confirmed empirically: an early pilot with
onset~U[0,200) and ramp_len=160 produced 0% failures, 20/20 successes, because
every episode finished before the ramp had any effect). Frame 0 of every
episode (including failure-eval) has zero injected noise by construction
(onset_frame >= 0 and ramp starts at 0), so the episode
is nominal at reset.

DELIBERATELY NOT REUSING robomimic_eval.py's CLI entrypoint: that script's
`main()` hardwires the cube-perturbation mechanism end to end (collection,
caching, scoring, markdown writer all reference half_extent). Re-deriving a
parallel collection + scoring path here, importing every shared primitive
(load_policy_and_env, register_feature_hook, make_sampler, DETECTORS, the rnd
scorer factory, rolling_spread_score, episode_aggregate, compute_lead_times,
metrics.*) from robomimic_eval.py so detector logic and metric math are
identical, not re-implemented.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmark"))
sys.path.insert(0, str(REPO_ROOT / "benchmark" / "lib"))
sys.path.insert(0, str(REPO_ROOT / "benchmark" / "real"))

import metrics  # noqa: E402
import robomimic_eval as base  # noqa: E402

CACHE_DIR = REPO_ROOT / "benchmark" / "real" / "cache"
RESULTS_PATH = REPO_ROOT / "benchmark" / "RESULTS_ROBOMIMIC_GRADUAL.md"

OBS_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
HORIZON = base.HORIZON
TARGET_FPR = base.TARGET_FPR
WINDOW = base.WINDOW

NOISE_SIGMA_SCALE = 3.0       # multiplier on per-key nominal std at full ramp
RAMP_ONSET_MAX_FRAMES = 15    # onset_frame ~ U[0, this), in absolute frames
RAMP_LENGTH_FRAMES = 15       # frames from onset to full-noise, in absolute frames
# NOTE: nominal Lift episodes succeed in ~35-50 frames (measured: median ~41),
# far short of HORIZON=400. Ramp timing must therefore be scaled to typical
# episode length, not to the 400-frame horizon, or the noise never builds up
# before the episode would have already succeeded -- this was confirmed by a
# pilot run with onset~U[0,0.5*400) and ramp_len=0.4*400 that produced a 0%
# failure rate (20/20 succeeded) because every episode finished before the
# ramp had any effect.


def _log(msg: str) -> None:
    print(f"[robomimic_eval_gradual] {msg}", flush=True)


def measure_obs_std(policy, env, raw_env, seeds: list[int]) -> dict[str, np.ndarray]:
    """Per-key, per-dimension std of nominal observations, measured over a few
    nominal rollouts. Used to scale injected noise to each channel's natural
    range so e.g. quaternion components and object xyz get comparably-relative
    corruption rather than the same absolute sigma."""
    raw_env.placement_initializer = base.make_sampler(raw_env, base.NOMINAL_HALF_EXTENT)
    buf: dict[str, list[np.ndarray]] = {k: [] for k in OBS_KEYS}
    for seed in seeds:
        np.random.seed(seed)
        obs = env.reset()
        policy.start_episode()
        for t in range(HORIZON):
            for k in OBS_KEYS:
                buf[k].append(np.asarray(obs[k], dtype=np.float64).copy())
            ac = policy(ob=obs)
            obs, r, done, info = env.step(ac)
            if env.is_success()["task"] or done:
                break
    std = {k: np.std(np.stack(v, axis=0), axis=0) for k, v in buf.items() if v}
    for k, s in std.items():
        s[s < 1e-6] = 1e-6  # guard zero-variance dims (e.g. constant quat component)
    return std


def ramp_sigma_at(t: int, onset: int, ramp_len: int) -> float:
    """Fraction (0..1) of full noise sigma active at frame t."""
    if t < onset:
        return 0.0
    prog = (t - onset) / max(ramp_len, 1)
    return float(np.clip(prog, 0.0, 1.0))


def run_episode_gradual(policy, env, get_and_clear, seed: int, obs_std: dict[str, np.ndarray],
                        rng: np.random.Generator) -> dict:
    np.random.seed(seed)
    obs = env.reset()
    policy.start_episode()
    success = False
    steps = 0
    onset = int(rng.uniform(0, RAMP_ONSET_MAX_FRAMES))
    ramp_len = RAMP_LENGTH_FRAMES
    sigma_trace = np.zeros(HORIZON, dtype=np.float64)
    for t in range(HORIZON):
        frac = ramp_sigma_at(t, onset, ramp_len)
        sigma_trace[t] = frac
        if frac > 0:
            noisy_obs = dict(obs)
            for k in OBS_KEYS:
                if k not in noisy_obs:
                    continue
                sigma = NOISE_SIGMA_SCALE * frac * obs_std[k]
                noisy_obs[k] = np.asarray(obs[k], dtype=np.float64) + rng.normal(0.0, sigma)
            ac = policy(ob=noisy_obs)
        else:
            ac = policy(ob=obs)
        obs, r, done, info = env.step(ac)
        steps = t + 1
        if env.is_success()["task"]:
            success = True
            break
        if done:
            break
    feats = get_and_clear()
    return {
        "features": feats,
        "success": success,
        "steps": steps,
        "seed": seed,
        "onset_frame": onset,
        "sigma_trace": sigma_trace[:steps],
    }


def collect_rollouts_gradual(policy, env, raw_env, seeds: list[int], get_and_clear,
                             obs_std: dict[str, np.ndarray], label: str, base_seed: int) -> list[dict]:
    raw_env.placement_initializer = base.make_sampler(raw_env, base.NOMINAL_HALF_EXTENT)
    out = []
    t0 = time.time()
    rng = np.random.default_rng(base_seed)
    for i, seed in enumerate(seeds):
        ep = run_episode_gradual(policy, env, get_and_clear, seed, obs_std, rng)
        out.append(ep)
        dt = time.time() - t0
        rate = (i + 1) / dt if dt > 0 else 0.0
        _log(f"{label} seed={seed} success={ep['success']} steps={ep['steps']} "
             f"onset={ep['onset_frame']} ({i + 1}/{len(seeds)}, {rate:.2f} ep/s, {dt:.1f}s elapsed)")
    return out


def main():
    global NOISE_SIGMA_SCALE
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-calib", type=int, default=60)
    parser.add_argument("--n-nominal-eval", type=int, default=60)
    parser.add_argument("--n-induced-eval", type=int, default=70)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--noise-sigma-scale", type=float, default=NOISE_SIGMA_SCALE)
    args = parser.parse_args()

    NOISE_SIGMA_SCALE = args.noise_sigma_scale

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"rollouts_gradual_seed{args.seed_offset}_sig{NOISE_SIGMA_SCALE}.npz"

    if args.use_cache and cache_file.exists():
        _log(f"loading cached rollouts from {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        calib_eps = list(data["calib_eps"])
        nominal_eval_eps = list(data["nominal_eval_eps"])
        induced_eval_eps = list(data["induced_eval_eps"])
    else:
        _log("loading policy + env")
        policy, env = base.load_policy_and_env()
        raw_env = env.env
        handle, get_and_clear = base.register_feature_hook(policy)

        offs = args.seed_offset
        calib_seeds = list(range(offs + 0, offs + args.n_calib))
        nominal_eval_seeds = list(range(offs + 10_000, offs + 10_000 + args.n_nominal_eval))
        induced_eval_seeds = list(range(offs + 20_000, offs + 20_000 + args.n_induced_eval))

        _log("measuring nominal obs std for noise scaling (10 short rollouts)")
        obs_std = measure_obs_std(policy, env, raw_env, list(range(offs + 90_000, offs + 90_010)))
        for k, s in obs_std.items():
            _log(f"  obs_std[{k}] = {np.round(s, 4)}")

        _log(f"collecting {len(calib_seeds)} nominal calibration-fit episodes (zero noise)")
        calib_eps = base.collect_rollouts(policy, env, raw_env, base.NOMINAL_HALF_EXTENT,
                                          calib_seeds, get_and_clear, "calib-fit")
        for e in calib_eps:
            e["onset_frame"] = None

        _log(f"collecting {len(nominal_eval_seeds)} nominal eval episodes (zero noise)")
        nominal_eval_eps = base.collect_rollouts(policy, env, raw_env, base.NOMINAL_HALF_EXTENT,
                                                  nominal_eval_seeds, get_and_clear, "nominal-eval")
        for e in nominal_eval_eps:
            e["onset_frame"] = None

        _log(f"collecting {len(induced_eval_seeds)} gradual-noise-ramp eval episodes "
             f"(sigma_scale={NOISE_SIGMA_SCALE}, onset~U[0,{RAMP_ONSET_MAX_FRAMES}), "
             f"ramp_len={RAMP_LENGTH_FRAMES} frames)")
        induced_eval_eps = collect_rollouts_gradual(policy, env, raw_env, induced_eval_seeds,
                                                     get_and_clear, obs_std, "induced-eval-gradual",
                                                     base_seed=offs + 555)

        handle.remove()

        np.savez(cache_file, calib_eps=np.array(calib_eps, dtype=object),
                 nominal_eval_eps=np.array(nominal_eval_eps, dtype=object),
                 induced_eval_eps=np.array(induced_eval_eps, dtype=object))
        _log(f"cached rollouts to {cache_file}")

    n_calib_succ = sum(1 for e in calib_eps if e["success"])
    n_nom_succ = sum(1 for e in nominal_eval_eps if e["success"])
    n_ind_succ = sum(1 for e in induced_eval_eps if e["success"])
    _log(f"calib-fit success: {n_calib_succ}/{len(calib_eps)}")
    _log(f"nominal-eval success: {n_nom_succ}/{len(nominal_eval_eps)}")
    failure_rate = 1 - n_ind_succ / max(len(induced_eval_eps), 1)
    _log(f"induced-eval-gradual success: {n_ind_succ}/{len(induced_eval_eps)} "
         f"(failure rate {failure_rate:.2f})")

    calib_seeds_set = {e["seed"] for e in calib_eps}
    nominal_eval_seeds_set = {e["seed"] for e in nominal_eval_eps}
    induced_eval_seeds_set = {e["seed"] for e in induced_eval_eps}
    assert calib_seeds_set.isdisjoint(nominal_eval_seeds_set)
    assert calib_seeds_set.isdisjoint(induced_eval_seeds_set)
    assert nominal_eval_seeds_set.isdisjoint(induced_eval_seeds_set)
    _log("disjoint calib/nominal-eval/induced-eval seed sets: OK")

    fid = np.concatenate([e["features"] for e in calib_eps if e["features"].shape[0] > 0], axis=0)
    _log(f"calibration-fit feature pool shape {fid.shape}")

    score_fns = dict(base.DETECTORS)
    score_fns["rnd"] = base.make_rnd_scorer(window=1)
    score_fns["temporal_rnd"] = base.make_rnd_scorer(window=8)
    score_fns["phm_rolling_spread"] = base.rolling_spread_score

    all_detector_scores: dict[str, dict] = {}

    for name, fn in score_fns.items():
        _log(f"scoring detector={name}")
        nom_scores = base.fit_and_score(name, fn, fid, nominal_eval_eps)
        ind_scores = base.fit_and_score(name, fn, fid, induced_eval_eps)

        nom_agg = base.episode_aggregate(nom_scores, "max")
        ind_agg = base.episode_aggregate(ind_scores, "max")

        fail_labels = np.array(
            [0] * len(nominal_eval_eps) + [0 if e["success"] else 1 for e in induced_eval_eps]
        )
        scores_pool = np.concatenate([nom_agg, ind_agg])

        valid = np.isfinite(scores_pool)
        scores_v = scores_pool[valid]
        labels_v = fail_labels[valid]

        auroc_mean, auroc_lo, auroc_hi = metrics.bootstrap_ci(metrics.auroc, scores_v, labels_v)
        aupr_mean, aupr_lo, aupr_hi = metrics.bootstrap_ci(metrics.aupr, scores_v, labels_v)
        fpr95 = metrics.fpr_at_tpr(scores_v, labels_v, 0.95)

        nom_frame_scores = np.concatenate([s for s in nom_scores if s.size > 0])
        nom_frame_scores = nom_frame_scores[np.isfinite(nom_frame_scores)]
        thr = float(np.quantile(nom_frame_scores, 1.0 - TARGET_FPR))
        achieved_frame_fpr = float(np.mean(nom_frame_scores >= thr))

        lead_times, alarm_frames = base.compute_lead_times(ind_scores, induced_eval_eps, thr)
        lt_valid = [lt for lt in lead_times if lt is not None]
        af_valid = [af for af in alarm_frames if af is not None]
        n_failed = sum(1 for e in induced_eval_eps if not e["success"])
        frac_flagged_early_10 = (
            sum(1 for lt in lt_valid if lt >= 10) / n_failed if n_failed else float("nan")
        )
        never_flagged = sum(1 for lt in lead_times if lt is None)
        frac_alarm_at_reset = (
            sum(1 for af in af_valid if af <= 1) / len(af_valid) if af_valid else float("nan")
        )

        # gradual-mechanism-specific diagnostic: was the alarm frame before or
        # after this episode's noise-onset frame? An alarm firing before onset
        # would mean the detector is NOT responding to the injected drift at
        # all (false precursor / unrelated signal); firing after onset but
        # before terminal is the genuine "tracked the drift" case.
        failed_eps_with_alarms = [
            (ep, af) for ep, af in zip(
                [e for e in induced_eval_eps if not e["success"]],
                alarm_frames, strict=True
            ) if af is not None
        ]
        frac_alarm_after_onset = (
            sum(1 for ep, af in failed_eps_with_alarms if af >= (ep.get("onset_frame") or 0))
            / len(failed_eps_with_alarms) if failed_eps_with_alarms else float("nan")
        )

        all_detector_scores[name] = {
            "auroc": {"mean": auroc_mean, "lo": auroc_lo, "hi": auroc_hi},
            "aupr": {"mean": aupr_mean, "lo": aupr_lo, "hi": aupr_hi},
            "fpr95": fpr95,
            "threshold_at_5pct_frame_fpr": thr,
            "achieved_frame_fpr": achieved_frame_fpr,
            "lead_times": lead_times,
            "lead_time_mean": float(np.mean(lt_valid)) if lt_valid else None,
            "lead_time_median": float(np.median(lt_valid)) if lt_valid else None,
            "alarm_frame_median": float(np.median(af_valid)) if af_valid else None,
            "frac_alarm_at_reset": frac_alarm_at_reset,
            "frac_alarm_after_onset": frac_alarm_after_onset,
            "frac_failures_flagged_ge10_early": frac_flagged_early_10,
            "n_failed_episodes": n_failed,
            "n_never_flagged": never_flagged,
        }
        _log(f"  {name}: AUROC {auroc_mean:.3f} [{auroc_lo:.3f},{auroc_hi:.3f}]  "
             f"lead_time_mean={all_detector_scores[name]['lead_time_mean']}  "
             f"alarm_frame_median={all_detector_scores[name]['alarm_frame_median']}  "
             f"frac_alarm_at_reset={frac_alarm_at_reset:.2f}  "
             f"frac_alarm_after_onset={frac_alarm_after_onset:.2f}  "
             f"%>=10early={frac_flagged_early_10:.2f}")

    write_results_md(all_detector_scores, len(calib_eps), len(nominal_eval_eps),
                      len(induced_eval_eps), n_ind_succ, args)
    _log(f"wrote {RESULTS_PATH}")
    return all_detector_scores, failure_rate


def write_results_md(results: dict, n_calib: int, n_nom: int, n_ind: int, n_ind_succ: int, args) -> None:
    failure_rate = 1 - n_ind_succ / max(n_ind, 1)
    lines = []
    lines.append("# Robomimic gradual-noise-ramp failure-prediction benchmark (lift, BC-RNN-GMM)\n")
    lines.append(
        "Second OOD-induction mechanism for the same policy/feature/detector stack as "
        "`RESULTS_ROBOMIMIC.md`. Failure is induced by ramping Gaussian noise on the "
        "policy's observation vector (robot0_eef_pos, robot0_eef_quat, "
        "robot0_gripper_qpos, object) starting from a random per-episode onset frame "
        "and rising linearly to full sigma, rather than by perturbing the cube's "
        "initial position. Every episode (including failure-eval) is nominal at "
        f"frame 0 by construction. Noise sigma scale = {args.noise_sigma_scale}x the "
        "per-key, per-dimension std measured over nominal rollouts; onset_frame ~ "
        f"U[0, {RAMP_ONSET_MAX_FRAMES}) frames, ramp length = {RAMP_LENGTH_FRAMES} "
        "frames to full noise, held thereafter.\n"
    )
    lines.append(
        f"Calibration-fit: {n_calib} nominal episodes, zero injected noise. "
        f"Nominal-eval: {n_nom} disjoint nominal episodes, zero injected noise. "
        f"Gradual-noise-ramp eval: {n_ind} episodes with the ramp mechanism above, "
        f"empirical failure rate {failure_rate:.2f} ({n_ind - n_ind_succ}/{n_ind} "
        "failed). All three splits use disjoint seed ranges; detectors fit ONLY on "
        "calibration-fit features.\n"
    )
    lines.append(
        f"Lead-time operating point: identical to `RESULTS_ROBOMIMIC.md` -- "
        f"per-detector threshold fixed at the (1 - {TARGET_FPR:.2f}) quantile of "
        "frame-level scores pooled over all nominal-eval frames "
        f"(~{int(TARGET_FPR*100)}% healthy-calibrated FPR).\n"
    )
    lines.append(
        "**New diagnostic for this mechanism: `frac_alarm_after_onset`.** Fraction of "
        "first-alarms (among failed episodes that were ever flagged) occurring at or "
        "after that episode's own noise-onset frame. A detector legitimately tracking "
        "the injected drift should have this near 1.0; a value well below 1.0 means "
        "the detector is alarming on something unrelated to the drift (or on episode "
        "structure that happens to correlate with eventual failure), not on the "
        "gradual OOD signal this mechanism is designed to test.\n"
    )

    lines.append("## Episode-level failure-prediction metrics (gradual-noise-ramp)\n")
    lines.append(
        "| Detector | AUROC (95% CI) | AUPR (95% CI) | FPR@95TPR | Lead-time mean "
        "(median) frames | Alarm frame (median) | % alarms at reset (<=1) | % alarms "
        "after onset | % failures flagged >=10 frames early | Never-flagged failures |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    order = ["phm_rolling_spread", "hotelling_t2", "mahalanobis", "relative_mahalanobis",
             "knn", "pca_residual", "rnd", "temporal_rnd"]
    order = [o for o in order if o in results] + [o for o in results if o not in order]
    for name in order:
        r = results[name]
        auroc = r["auroc"]
        aupr = r["aupr"]
        lt_mean = r["lead_time_mean"]
        lt_median = r["lead_time_median"]
        lt_str = f"{lt_mean:.1f} ({lt_median:.1f})" if lt_mean is not None else "n/a"
        af_median = r.get("alarm_frame_median")
        af_str = f"{af_median:.1f}" if af_median is not None else "n/a"
        far = r.get("frac_alarm_at_reset")
        far_str = f"{far:.2f}" if far is not None and far == far else "n/a"
        fao = r.get("frac_alarm_after_onset")
        fao_str = f"{fao:.2f}" if fao is not None and fao == fao else "n/a"
        lines.append(
            f"| {name} | {auroc['mean']:.3f} [{auroc['lo']:.3f}, {auroc['hi']:.3f}] "
            f"| {aupr['mean']:.3f} [{aupr['lo']:.3f}, {aupr['hi']:.3f}] "
            f"| {r['fpr95']:.3f} | {lt_str} | {af_str} | {far_str} | {fao_str} "
            f"| {r['frac_failures_flagged_ge10_early']:.2f} "
            f"| {r['n_never_flagged']}/{r['n_failed_episodes']} |"
        )

    lines.append("\n## Headline (gradual mechanism only)\n")
    gradual_candidates = [
        (n, r) for n, r in results.items()
        if r["lead_time_mean"] is not None
        and r.get("frac_alarm_at_reset") is not None
        and r["frac_alarm_at_reset"] == r["frac_alarm_at_reset"]
        and r["frac_alarm_at_reset"] < 0.5
    ]
    best_gradual = max(gradual_candidates, key=lambda kv: kv[1]["lead_time_mean"], default=None)
    best_auroc = max(results.items(), key=lambda kv: kv[1]["auroc"]["mean"], default=None)
    if best_auroc:
        n, r = best_auroc
        lines.append(
            f"Best AUROC under gradual induction: **{n}** at {r['auroc']['mean']:.3f} "
            f"[{r['auroc']['lo']:.3f}, {r['auroc']['hi']:.3f}], "
            f"frac_alarm_at_reset={r['frac_alarm_at_reset']:.2f}.\n"
        )
    if best_gradual:
        n, r = best_gradual
        lines.append(
            f"Best non-reset-confounded lead time under gradual induction: **{n}** at "
            f"{r['lead_time_mean']:.1f} frames (median {r['lead_time_median']:.1f}), "
            f"AUROC {r['auroc']['mean']:.3f} [{r['auroc']['lo']:.3f}, {r['auroc']['hi']:.3f}], "
            f"{r['frac_failures_flagged_ge10_early']:.0%} of failures flagged "
            ">=10 frames early.\n"
        )
    else:
        lines.append(
            "No detector achieved usable lead time without frac_alarm_at_reset>=0.5 "
            "under the gradual mechanism either.\n"
        )

    RESULTS_PATH.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
