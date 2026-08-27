"""Second OOD-induction mechanism for the robomimic failure-prediction benchmark:
progressive observation-noise drift that builds DURING the episode.

Sibling to robomimic_eval.py. Policy, feature tap, detector set, split
discipline and metric math are imported from it unchanged; the only thing that
differs is HOW failure is induced, plus the two extra controls this run adds.

Why a second mechanism exists
-----------------------------
robomimic_eval.py induces failure by widening the cube spawn box (half-extent
0.18 against 0.03 nominal). That perturbation is present at frame 0, so a
detector can score a high AUROC purely by recognising a weird starting state.
In the 2026-06-16 run the static feature-space detectors scored AUROC 0.96 to
0.99 with `frac_alarm_at_reset` at or above 0.87: they were classifying the
initial condition, not anticipating the failure. The only detector whose alarms
were not reset-driven (`phm_rolling_spread`, `frac_alarm_at_reset` 0.00) sat at
chance, AUROC 0.510 [0.376, 0.658].

This script induces failure with a Gaussian noise ramp on the policy's
observation vector (robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos,
object). Injected sigma is exactly zero at frame 0 of every episode by
construction, rises linearly from a random per-episode onset, and holds. The
observation distribution at reset therefore matches the calibration-fit pool
exactly, and `frac_alarm_at_reset` becomes a real discriminator instead of a
giveaway.

Ramp timing is in ABSOLUTE frames, not a fraction of HORIZON=400, because
nominal Lift episodes succeed in roughly 35 to 50 frames, far short of the
horizon. A horizon-relative ramp never builds up before the episode would
already have ended: an early pilot with onset ~ U[0, 200) and ramp_len 160
produced 0 failures in 20 episodes for exactly that reason.

The two controls this run adds
------------------------------
1. **Dose-matched within-arm contrast.** Scoring perturbed episodes against
   unperturbed ones asks "is this episode corrupted", which is not the question.
   The primary contrast here is failed against succeeded episodes drawn from a
   SINGLE perturbation arm at one fixed amplitude, so mechanism, amplitude and
   onset distribution are identical across the two groups and outcome is the
   only thing that varies. The realized dose is audited per outcome group; a
   relative gap above 0.10 blocks the contrast. The same within-arm contrast is
   computed for the cube-perturbation mechanism, so the two mechanisms are
   compared on equal terms.
2. **Fixed-window exposure control.** Failed episodes run the full 400-frame
   horizon while successful ones terminate in roughly 35 to 50 frames, so a
   max-over-all-frames episode statistic gives failures about ten times as many
   chances to cross threshold. Every AUROC is therefore computed twice: with the
   legacy max over all frames, and with a max over the first K_WINDOW frames,
   which is below the shortest nominal episode length. The fixed-window number
   is the one that carries a claim.

Pre-registered at docs/preregistration/2026-08-27-gradual-induction-cross-check.md
before any of these runs were executed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmark"))
sys.path.insert(0, str(REPO_ROOT / "benchmark" / "lib"))
sys.path.insert(0, str(REPO_ROOT / "benchmark" / "real"))

import gradual_induction as gi  # noqa: E402
import metrics  # noqa: E402
import robomimic_eval as base  # noqa: E402

CACHE_DIR = REPO_ROOT / "benchmark" / "real" / "cache"
RESULTS_PATH = REPO_ROOT / "benchmark" / "RESULTS_ROBOMIMIC_GRADUAL.md"
JSON_PATH = REPO_ROOT / "benchmark" / "results_robomimic_gradual.json"

OBS_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
HORIZON = base.HORIZON
TARGET_FPR = base.TARGET_FPR
WINDOW = base.WINDOW

RAMP_ONSET_MAX_FRAMES = 15    # onset_frame ~ U[0, this), absolute frames
RAMP_LENGTH_FRAMES = 15       # frames from onset to full noise, absolute frames
K_WINDOW = 30                 # exposure-control window, below min nominal length
HIGH_SIGMA = 3.05             # amplitude of the existing published gradual run
PILOT_SIGMA_GRID = (0.5, 1.0, 1.5, 2.0, 2.5)
PILOT_TARGET_FAILURE_RATE = 0.35
PILOT_N = 20

DETECTOR_ORDER = ["phm_rolling_spread", "hotelling_t2", "mahalanobis",
                  "relative_mahalanobis", "knn", "pca_residual", "rnd", "temporal_rnd"]


def _log(msg: str) -> None:
    print(f"[robomimic_eval_gradual] {msg}", flush=True)


def measure_obs_std(policy, env, raw_env, seeds: list[int]) -> dict[str, np.ndarray]:
    """Per-key, per-dimension std of nominal observations over a few nominal
    rollouts. Injected noise is scaled by this so each channel is corrupted
    relative to its own natural spread rather than by a shared absolute sigma."""
    raw_env.placement_initializer = base.make_sampler(raw_env, base.NOMINAL_HALF_EXTENT)
    buf: dict[str, list[np.ndarray]] = {k: [] for k in OBS_KEYS}
    for seed in seeds:
        np.random.seed(seed)
        obs = env.reset()
        policy.start_episode()
        for _t in range(HORIZON):
            for k in OBS_KEYS:
                buf[k].append(np.asarray(obs[k], dtype=np.float64).copy())
            ac = policy(ob=obs)
            obs, _r, done, _info = env.step(ac)
            if env.is_success()["task"] or done:
                break
    std = {k: np.std(np.stack(v, axis=0), axis=0) for k, v in buf.items() if v}
    for _k, s in std.items():
        s[s < 1e-6] = 1e-6  # guard zero-variance dims, e.g. a constant quat component
    return std


def run_episode_gradual(policy, env, get_and_clear, seed: int,
                        obs_std: dict[str, np.ndarray], rng: np.random.Generator,
                        sigma_scale: float) -> dict:
    np.random.seed(seed)
    obs = env.reset()
    policy.start_episode()
    success = False
    steps = 0
    onset = int(rng.uniform(0, RAMP_ONSET_MAX_FRAMES))
    sigma_trace = np.zeros(HORIZON, dtype=np.float64)
    for t in range(HORIZON):
        frac = gi.ramp_fraction(t, onset, RAMP_LENGTH_FRAMES)
        sigma_trace[t] = frac
        ac = policy(ob=gi.inject_noise(obs, obs_std, OBS_KEYS, sigma_scale, frac, rng))
        obs, _r, done, _info = env.step(ac)
        steps = t + 1
        if env.is_success()["task"]:
            success = True
            break
        if done:
            break
    return {
        "features": get_and_clear(),
        "success": success,
        "steps": steps,
        "seed": seed,
        "onset_frame": onset,
        "sigma_trace": sigma_trace[:steps],
    }


def collect_rollouts_gradual(policy, env, raw_env, seeds: list[int], get_and_clear,
                             obs_std: dict[str, np.ndarray], label: str, base_seed: int,
                             sigma_scale: float, quiet: bool = False) -> list[dict]:
    raw_env.placement_initializer = base.make_sampler(raw_env, base.NOMINAL_HALF_EXTENT)
    out = []
    t0 = time.time()
    rng = np.random.default_rng(base_seed)
    for i, seed in enumerate(seeds):
        ep = run_episode_gradual(policy, env, get_and_clear, seed, obs_std, rng, sigma_scale)
        out.append(ep)
        if not quiet:
            dt = time.time() - t0
            _log(f"{label} seed={seed} success={ep['success']} steps={ep['steps']} "
                 f"onset={ep['onset_frame']} ({i + 1}/{len(seeds)}, "
                 f"{(i + 1) / dt if dt else 0:.2f} ep/s, {dt:.1f}s)")
    return out


def run_pilot(policy, env, raw_env, get_and_clear, obs_std, seed_block: int) -> tuple[float, list]:
    """Select the dose-matched arm's sigma on seeds disjoint from every eval arm.

    Only the episode success flag is inspected. No detector is fit or scored on
    pilot data, which keeps dose selection out of the "chosen after seeing the
    outcome" category. Rule fixed in the pre-registration: smallest sigma whose
    pilot failure rate is at least PILOT_TARGET_FAILURE_RATE, else the largest
    sigma tested.
    """
    seeds = list(range(seed_block, seed_block + PILOT_N))
    table = []
    for sigma in PILOT_SIGMA_GRID:
        eps = collect_rollouts_gradual(policy, env, raw_env, seeds, get_and_clear,
                                       obs_std, f"pilot-sig{sigma}", base_seed=seed_block,
                                       sigma_scale=sigma, quiet=True)
        rate = 1.0 - sum(e["success"] for e in eps) / len(eps)
        table.append({"sigma": sigma, "failure_rate": rate, "n": len(eps)})
        _log(f"pilot sigma={sigma}: failure rate {rate:.2f} ({int(rate * len(eps))}/{len(eps)})")
    for row in table:
        if row["failure_rate"] >= PILOT_TARGET_FAILURE_RATE:
            _log(f"pilot selected sigma={row['sigma']} (first at or above "
                 f"{PILOT_TARGET_FAILURE_RATE} failure rate)")
            return row["sigma"], table
    chosen = PILOT_SIGMA_GRID[-1]
    _log(f"pilot: no sigma reached {PILOT_TARGET_FAILURE_RATE}; falling back to {chosen}")
    return chosen, table


def _auroc_block(scores: np.ndarray, labels: np.ndarray) -> dict:
    valid = np.isfinite(scores)
    s, y = scores[valid], labels[valid]
    if s.size == 0 or len(set(y.tolist())) < 2:
        return {"auroc": None, "lo": None, "hi": None, "aupr": None,
                "fpr95": None, "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}
    au, au_lo, au_hi = metrics.bootstrap_ci(metrics.auroc, s, y)
    ap, _ap_lo, _ap_hi = metrics.bootstrap_ci(metrics.aupr, s, y)
    return {"auroc": au, "lo": au_lo, "hi": au_hi, "aupr": ap,
            "fpr95": metrics.fpr_at_tpr(s, y, 0.95),
            "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}


def outcome_contrast(episodes: list, scores: list[np.ndarray], k: int | None) -> dict:
    """Within-arm contrast: failed (1) against succeeded (0) inside ONE arm.

    Every episode here received the identical treatment, so this isolates
    "will this episode fail" from "has this episode been perturbed".
    """
    agg = np.array([gi.episode_max(s, k) for s in scores], dtype=np.float64)
    lab = np.array([0.0 if e["success"] else 1.0 for e in episodes], dtype=np.float64)
    return _auroc_block(agg, lab)


def cross_arm_contrast(nominal_eps, nominal_scores, arm_eps, arm_scores, k: int | None) -> dict:
    """Legacy contrast: nominal-eval plus the arm's successes as label 0, the
    arm's failures as label 1. This is the shape of the 2026-06-16 table."""
    agg = [gi.episode_max(s, k) for s in nominal_scores]
    lab = [0.0] * len(nominal_scores)
    for e, s in zip(arm_eps, arm_scores, strict=True):
        agg.append(gi.episode_max(s, k))
        lab.append(0.0 if e["success"] else 1.0)
    return _auroc_block(np.asarray(agg, dtype=np.float64), np.asarray(lab, dtype=np.float64))


def alarm_diagnostics(arm_eps, arm_scores, threshold: float) -> dict:
    """frac_alarm_at_reset, frac_alarm_after_onset, lead times, over failures."""
    lead_times, alarm_frames = base.compute_lead_times(arm_scores, arm_eps, threshold)
    failed = [e for e in arm_eps if not e["success"] and e["features"].shape[0] > 0]
    af_valid = [af for af in alarm_frames if af is not None]
    lt_valid = [lt for lt in lead_times if lt is not None]
    n_failed = len(failed)
    pairs = [(e, af) for e, af in zip(failed, alarm_frames, strict=True)
             if af is not None]
    onset_pairs = [(e, af) for e, af in pairs if e.get("onset_frame") is not None]
    return {
        "n_failed": n_failed,
        "n_never_flagged": sum(1 for lt in lead_times if lt is None),
        "lead_time_mean": float(np.mean(lt_valid)) if lt_valid else None,
        "lead_time_median": float(np.median(lt_valid)) if lt_valid else None,
        "alarm_frame_median": float(np.median(af_valid)) if af_valid else None,
        "frac_alarm_at_reset": (sum(1 for af in af_valid if af <= 1) / len(af_valid)
                                if af_valid else None),
        "frac_alarm_after_onset": (sum(1 for e, af in onset_pairs if af >= e["onset_frame"])
                                   / len(onset_pairs) if onset_pairs else None),
        "frac_failures_flagged_ge10_early": (sum(1 for lt in lt_valid if lt >= 10) / n_failed
                                             if n_failed else None),
    }


def build_score_fns():
    fns = dict(base.DETECTORS)
    fns["rnd"] = base.make_rnd_scorer(window=1)
    fns["temporal_rnd"] = base.make_rnd_scorer(window=8)
    fns["phm_rolling_spread"] = base.rolling_spread_score
    return fns


def _fmt(v, nd=3):
    return "n/a" if v is None or (isinstance(v, float) and v != v) else f"{v:.{nd}f}"


def _ci(block):
    if block["auroc"] is None:
        return "n/a"
    return f"{block['auroc']:.3f} [{block['lo']:.3f}, {block['hi']:.3f}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-calib", type=int, default=60)
    parser.add_argument("--n-nominal-eval", type=int, default=60)
    parser.add_argument("--n-induced-eval", type=int, default=70)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--high-sigma", type=float, default=HIGH_SIGMA)
    parser.add_argument("--dose-sigma", type=float, default=None,
                        help="skip the pilot and use this sigma for the dose-matched arm")
    parser.add_argument("--use-cache", action="store_true")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"arms_gradual_seed{args.seed_offset}.npz"

    if args.use_cache and cache_file.exists():
        _log(f"loading cached arms from {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        arms = {k: list(data[k]) for k in ("calib", "nominal", "high", "dose", "cube")}
        dose_sigma = float(data["dose_sigma"])
        pilot_table = json.loads(str(data["pilot_table"]))
    else:
        _log("loading policy + env")
        policy, env = base.load_policy_and_env()
        raw_env = env.env
        handle, get_and_clear = base.register_feature_hook(policy)

        offs = args.seed_offset
        calib_seeds = list(range(offs, offs + args.n_calib))
        nom_seeds = list(range(offs + 10_000, offs + 10_000 + args.n_nominal_eval))
        ind_seeds = list(range(offs + 20_000, offs + 20_000 + args.n_induced_eval))

        _log("measuring nominal obs std for noise scaling (10 rollouts)")
        obs_std = measure_obs_std(policy, env, raw_env, list(range(offs + 90_000, offs + 90_010)))
        for k, s in obs_std.items():
            _log(f"  obs_std[{k}] = {np.round(s, 4)}")

        if args.dose_sigma is None:
            _log("running dose pilot on a disjoint seed block (success flag only)")
            dose_sigma, pilot_table = run_pilot(policy, env, raw_env, get_and_clear,
                                                obs_std, offs + 80_000)
        else:
            dose_sigma, pilot_table = args.dose_sigma, []
            _log(f"dose sigma supplied on the command line: {dose_sigma}")

        _log(f"arm A calibration-fit: {len(calib_seeds)} nominal episodes")
        calib = base.collect_rollouts(policy, env, raw_env, base.NOMINAL_HALF_EXTENT,
                                      calib_seeds, get_and_clear, "calib-fit")
        _log(f"arm B nominal-eval: {len(nom_seeds)} nominal episodes")
        nominal = base.collect_rollouts(policy, env, raw_env, base.NOMINAL_HALF_EXTENT,
                                        nom_seeds, get_and_clear, "nominal-eval")
        for e in calib + nominal:
            e["onset_frame"] = None
            e["sigma_trace"] = np.zeros(0)

        _log(f"arm C1 high-dose ramp: sigma={args.high_sigma}")
        high = collect_rollouts_gradual(policy, env, raw_env, ind_seeds, get_and_clear,
                                        obs_std, "high-ramp", offs + 555, args.high_sigma)
        _log(f"arm C2 dose-matched ramp: sigma={dose_sigma}")
        dose = collect_rollouts_gradual(policy, env, raw_env, ind_seeds, get_and_clear,
                                        obs_std, "dose-ramp", offs + 556, dose_sigma)
        _log(f"arm D positive control: cube half_extent={base.INDUCED_HALF_EXTENT}")
        cube = base.collect_rollouts(policy, env, raw_env, base.INDUCED_HALF_EXTENT,
                                     ind_seeds, get_and_clear, "cube-perturb")
        for e in cube:
            e["onset_frame"] = None
            e["sigma_trace"] = np.zeros(0)

        handle.remove()
        arms = {"calib": calib, "nominal": nominal, "high": high, "dose": dose, "cube": cube}
        np.savez(cache_file, dose_sigma=dose_sigma, pilot_table=json.dumps(pilot_table),
                 **{k: np.array(v, dtype=object) for k, v in arms.items()})
        _log(f"cached arms to {cache_file}")

    for name, eps in arms.items():
        n_s = sum(1 for e in eps if e["success"])
        _log(f"arm {name}: {n_s}/{len(eps)} succeeded, failure rate "
             f"{1 - n_s / max(len(eps), 1):.2f}")

    seed_sets = {k: {e["seed"] for e in v} for k, v in arms.items()}
    assert seed_sets["calib"].isdisjoint(seed_sets["nominal"])
    assert seed_sets["calib"].isdisjoint(seed_sets["high"])
    assert seed_sets["nominal"].isdisjoint(seed_sets["high"])
    _log("seed-block disjointness calib / nominal-eval / induced: OK")

    dose_audit = {k: gi.dose_match_report(arms[k]) for k in ("high", "dose")}
    for k, r in dose_audit.items():
        _log(f"dose audit {k}: failed mean dose {r['mean_dose_failed']:.3f} "
             f"(n={r['n_failed']}), succeeded {r['mean_dose_succeeded']:.3f} "
             f"(n={r['n_succeeded']}), rel_gap {r['rel_gap']:.3f}")

    oracle = {k: gi.oracle_lead_times(arms[k]) for k in ("high", "dose")}
    for k, v in oracle.items():
        if v:
            _log(f"onset-oracle lead time, arm {k}: mean {np.mean(v):.1f} frames "
                 f"(median {np.median(v):.1f}) over {len(v)} failures")

    fid = np.concatenate([e["features"] for e in arms["calib"]
                          if e["features"].shape[0] > 0], axis=0)
    _log(f"calibration-fit feature pool shape {fid.shape}")

    nominal_lengths = [e["steps"] for e in arms["nominal"]]
    _log(f"nominal-eval episode length: min {min(nominal_lengths)}, "
         f"median {int(np.median(nominal_lengths))}, max {max(nominal_lengths)}; "
         f"exposure-control window K={K_WINDOW}")

    results: dict = {}
    for name, fn in build_score_fns().items():
        _log(f"scoring detector={name}")
        sc = {k: base.fit_and_score(name, fn, fid, arms[k])
              for k in ("nominal", "high", "dose", "cube")}

        nom_frames = np.concatenate([s for s in sc["nominal"] if s.size > 0])
        nom_frames = nom_frames[np.isfinite(nom_frames)]
        thr = float(np.quantile(nom_frames, 1.0 - TARGET_FPR))

        r = {
            "threshold_at_5pct_frame_fpr": thr,
            "achieved_frame_fpr": float(np.mean(nom_frames >= thr)),
            "cross_arm": {
                arm: {"full": cross_arm_contrast(arms["nominal"], sc["nominal"],
                                                 arms[arm], sc[arm], None),
                      "k": cross_arm_contrast(arms["nominal"], sc["nominal"],
                                              arms[arm], sc[arm], K_WINDOW)}
                for arm in ("high", "dose", "cube")
            },
            "within_arm": {
                arm: {"full": outcome_contrast(arms[arm], sc[arm], None),
                      "k": outcome_contrast(arms[arm], sc[arm], K_WINDOW)}
                for arm in ("high", "dose", "cube")
            },
            "alarms": {arm: alarm_diagnostics(arms[arm], sc[arm], thr)
                       for arm in ("high", "dose", "cube")},
        }
        results[name] = r
        _log(f"  {name}: cross-arm dose AUROC {_ci(r['cross_arm']['dose']['full'])} | "
             f"WITHIN-arm dose k={K_WINDOW} {_ci(r['within_arm']['dose']['k'])} | "
             f"cube positive control {_ci(r['cross_arm']['cube']['full'])} | "
             f"frac_alarm_at_reset dose {_fmt(r['alarms']['dose']['frac_alarm_at_reset'], 2)}")

    meta = {
        "dose_sigma": dose_sigma,
        "high_sigma": args.high_sigma,
        "pilot_table": pilot_table,
        "dose_audit": dose_audit,
        "oracle_lead": {k: {"mean": float(np.mean(v)) if v else None,
                            "median": float(np.median(v)) if v else None, "n": len(v)}
                        for k, v in oracle.items()},
        "arm_sizes": {k: len(v) for k, v in arms.items()},
        "arm_failure_rates": {k: 1 - sum(e["success"] for e in v) / len(v)
                              for k, v in arms.items()},
        "nominal_len": {"min": int(min(nominal_lengths)),
                        "median": float(np.median(nominal_lengths)),
                        "max": int(max(nominal_lengths))},
        "k_window": K_WINDOW,
        "seed_offset": args.seed_offset,
    }
    JSON_PATH.write_text(json.dumps({"meta": meta, "detectors": results}, indent=2))
    _log(f"wrote {JSON_PATH}")
    write_results_md(results, meta)
    _log(f"wrote {RESULTS_PATH}")
    return results, meta


def write_results_md(results: dict, meta: dict) -> None:
    K = meta["k_window"]
    order = [d for d in DETECTOR_ORDER if d in results] + \
            [d for d in results if d not in DETECTOR_ORDER]
    L = []
    L.append("# Robomimic gradual-onset induction cross-check (lift, BC-RNN-GMM)\n")
    L.append(
        "Second OOD-induction mechanism for the same policy, feature tap, detector set "
        "and split discipline as `RESULTS_ROBOMIMIC.md`. Failure is induced by a "
        "Gaussian noise ramp on the policy's observation vector, zero at frame 0 of "
        "every episode by construction and rising linearly from a random per-episode "
        "onset, rather than by perturbing the cube's initial placement. Pre-registered "
        "at `docs/preregistration/2026-08-27-gradual-induction-cross-check.md` before "
        "the run.\n"
    )
    L.append(
        f"Arms. A calibration-fit: {meta['arm_sizes']['calib']} nominal episodes, the "
        f"only pool any detector is fit on. B nominal-eval: "
        f"{meta['arm_sizes']['nominal']} disjoint nominal episodes, which also fix the "
        f"frame-level alarm threshold at the {1 - TARGET_FPR:.2f} quantile "
        f"(~{int(TARGET_FPR * 100)}% healthy FPR). C1 high-dose ramp: "
        f"{meta['arm_sizes']['high']} episodes at sigma {meta['high_sigma']}x nominal "
        f"per-channel std, failure rate {meta['arm_failure_rates']['high']:.2f}. "
        f"C2 dose-matched ramp: {meta['arm_sizes']['dose']} episodes at sigma "
        f"{meta['dose_sigma']}x, selected by a pilot on a disjoint seed block using the "
        f"success flag only, failure rate {meta['arm_failure_rates']['dose']:.2f}. "
        f"D positive control: {meta['arm_sizes']['cube']} episodes under the original "
        f"cube-perturbation mechanism (half-extent {base.INDUCED_HALF_EXTENT}), failure "
        f"rate {meta['arm_failure_rates']['cube']:.2f}. Onset ~ U[0, "
        f"{RAMP_ONSET_MAX_FRAMES}) frames, ramp length {RAMP_LENGTH_FRAMES} frames.\n"
    )
    L.append(
        f"Exposure control. Nominal episodes terminate on success at "
        f"{meta['nominal_len']['min']} to {meta['nominal_len']['max']} frames "
        f"(median {meta['nominal_len']['median']:.0f}) while failures run the full "
        f"{HORIZON}-frame horizon, so a max-over-all-frames episode statistic gives "
        f"failures roughly ten times as many chances to cross threshold. Every AUROC "
        f"below is reported twice: `full` is the legacy max over all frames and is "
        f"exposure-confounded; `k={K}` is a max over the first {K} frames only, below "
        f"the shortest nominal episode, and is the number that carries a claim.\n"
    )
    da = meta["dose_audit"]
    L.append(
        "Dose audit. Within the dose-matched arm C2 the failed episodes received a mean "
        f"injected sigma fraction of {da['dose']['mean_dose_failed']:.3f} "
        f"(n={da['dose']['n_failed']}) and the succeeded ones "
        f"{da['dose']['mean_dose_succeeded']:.3f} (n={da['dose']['n_succeeded']}), a "
        f"relative gap of {da['dose']['rel_gap']:.3f}. The pre-registered blocker is a "
        "relative gap above 0.10.\n"
    )

    L.append("## Contrast 1: the legacy shape, nominal-eval against a perturbed arm\n")
    L.append(
        "This is what `RESULTS_ROBOMIMIC.md` reports. It asks whether a perturbed and "
        "failing episode looks different from a healthy one, which conflates "
        "'this episode is corrupted' with 'this episode is going to fail'.\n"
    )
    L.append(f"| Detector | gradual C2, full | gradual C2, k={K} "
             f"| cube D, full | cube D, k={K} |")
    L.append("|---|---|---|---|---|")
    for d in order:
        c = results[d]["cross_arm"]
        L.append(f"| {d} | {_ci(c['dose']['full'])} | {_ci(c['dose']['k'])} "
                 f"| {_ci(c['cube']['full'])} | {_ci(c['cube']['k'])} |")

    L.append("\n## Contrast 2 (primary): dose-matched, within one arm\n")
    L.append(
        "Failed against succeeded episodes drawn from the SAME arm. Every episode here "
        "got the identical treatment at the identical amplitude, so the perturbation is "
        "held fixed and outcome is the only thing that varies. This is the contrast the "
        "2026-06-16 design did not have, and it is where the pre-registered claim lives "
        f"(primary cell: `phm_rolling_spread`, arm C2, k={K}).\n"
    )
    L.append(f"| Detector | C2 dose-matched, k={K} | C2 dose-matched, full | "
             f"D cube within-arm, k={K} | D cube within-arm, full |")
    L.append("|---|---|---|---|---|")
    for d in order:
        w = results[d]["within_arm"]
        L.append(f"| {d} | {_ci(w['dose']['k'])} | {_ci(w['dose']['full'])} "
                 f"| {_ci(w['cube']['k'])} | {_ci(w['cube']['full'])} |")

    L.append("\n## Alarm-timing diagnostics on the gradual arm C2\n")
    L.append(
        "`frac_alarm_at_reset` is meaningful here in a way it is not under the cube "
        "mechanism, because the gradual arm is a no-op at frame 0 by construction. "
        "`frac_alarm_after_onset` is the fraction of first-alarms landing at or after "
        "that episode's own noise onset; a detector tracking the injected drift should "
        "be near 1.0.\n"
    )
    L.append("| Detector | % alarms at reset (<=1) | % alarms after onset | "
             "alarm frame (median) | lead time mean (median) | % failures flagged >=10 "
             "frames early | never-flagged failures |")
    L.append("|---|---|---|---|---|---|---|")
    for d in order:
        a = results[d]["alarms"]["dose"]
        lt = (f"{a['lead_time_mean']:.1f} ({a['lead_time_median']:.1f})"
              if a["lead_time_mean"] is not None else "n/a")
        L.append(f"| {d} | {_fmt(a['frac_alarm_at_reset'], 2)} "
                 f"| {_fmt(a['frac_alarm_after_onset'], 2)} "
                 f"| {_fmt(a['alarm_frame_median'], 1)} | {lt} "
                 f"| {_fmt(a['frac_failures_flagged_ge10_early'], 2)} "
                 f"| {a['n_never_flagged']}/{a['n_failed']} |")
    ol = meta["oracle_lead"]["dose"]
    if ol["mean"] is not None:
        L.append(
            f"\nCeiling: an onset oracle that alarms exactly at each episode's true "
            f"noise-onset frame achieves a mean lead time of {ol['mean']:.1f} frames "
            f"(median {ol['median']:.1f}) over {ol['n']} failures. Lead time under this "
            f"definition is terminal frame minus first-alarm frame, so a detector that "
            f"fires early in a horizon-length episode scores a large number almost "
            f"mechanically; the oracle is what that number looks like with perfect "
            f"knowledge of the perturbation.\n"
        )

    L.append("\n## Verdict against the pre-registered kill criteria\n")
    prim = results.get("phm_rolling_spread", {}).get("within_arm", {}).get("dose", {}).get("k")
    if prim and prim["auroc"] is not None:
        dead = prim["lo"] <= 0.5 <= prim["hi"]
        L.append(
            f"1. Primary. `phm_rolling_spread` on the C2 dose-matched contrast at k={K}: "
            f"AUROC {_ci(prim)}. The pre-registered kill criterion fires when that CI "
            f"contains 0.5: **{'FIRED, the claim is dead' if dead else 'not fired'}**.\n"
        )
    surv = [d for d in order
            if results[d]["within_arm"]["dose"]["k"]["auroc"] is not None
            and results[d]["within_arm"]["dose"]["k"]["lo"] > 0.55]
    L.append(
        f"2. Secondary sweep. Detectors whose C2 dose-matched k={K} CI lower bound "
        f"exceeds 0.55: {', '.join(surv) if surv else 'none'}. "
        + ("With none surviving, the registered conclusion stands: neither induction "
           "mechanism supports a pre-failure early-warning claim on this policy, and "
           "the Week-2 gate PASS recorded on 2026-06-16 is withdrawn as a mechanism "
           "artifact.\n" if not surv else
           "These survive uncorrected and must be Holm-corrected across the eight "
           "detectors before being called significant.\n")
    )
    conf = [d for d in order
            if results[d]["cross_arm"]["dose"]["full"]["auroc"] is not None
            and results[d]["cross_arm"]["dose"]["full"]["auroc"] >= 0.95
            and results[d]["within_arm"]["dose"]["k"]["auroc"] is not None
            and results[d]["within_arm"]["dose"]["k"]["lo"] <= 0.5
            <= results[d]["within_arm"]["dose"]["k"]["hi"]]
    L.append(
        f"3. Confound verdict. Detectors scoring AUROC at or above 0.95 on the legacy "
        f"cross-arm contrast while their dose-matched CI contains 0.5: "
        f"{', '.join(conf) if conf else 'none'}. "
        + ("For these, the gradual mechanism's headline AUROC is a perturbation-presence "
           "confound, structurally the same error as the reset confound it was built to "
           "remove.\n" if conf else "\n")
    )
    pc = [(d, results[d]["cross_arm"]["cube"]["full"]) for d in ("mahalanobis", "knn")
          if d in results]
    ok = all(b["auroc"] is not None and b["auroc"] >= 0.95 for _d, b in pc)
    L.append(
        "4. Pipeline validity. Positive control, arm D under the original "
        "cube-perturbation mechanism through this same scoring code: "
        + "; ".join(f"{d} {_ci(b)}" for d, b in pc)
        + f". Pre-registered requirement is AUROC at or above 0.95 for both: "
          f"**{'PASS' if ok else 'FAIL, every result above is withheld'}**.\n"
    )

    RESULTS_PATH.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
