"""Run TRACK D end-to-end: train the stand-in policy, sweep alpha, plot.

Produces vla_monitor_demo/alpha_sweep.png and prints the measured lead-time.

Usage:
    /usr/bin/python3 vla_monitor_demo/run_demo.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from harness import SweepResult, run_sweep, train_policy

HERE = Path(__file__).resolve().parent
PNG = HERE / "alpha_sweep.png"
CSV = HERE / "alpha_sweep.csv"


def _try_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except Exception:
        return False


def plot_matplotlib(res: SweepResult, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax_left = plt.subplots(figsize=(8, 5), dpi=150)

    # Left axis: normalized output error and the monitor fired-fraction.
    ax_left.plot(
        res.alphas, res.output_error, "o-", color="#d63a3a", lw=1.8,
        label="normalized output error",
    )
    ax_left.plot(
        res.alphas, res.fired_fraction, "s-", color="#2b6cb0", lw=1.8,
        label="monitor fired fraction (frames below threshold)",
    )
    ax_left.axhline(
        res.output_degradation_level, color="#d63a3a", lw=0.8, ls=":",
        label=f"output degradation level ({res.output_degradation_level:.2f})",
    )
    ax_left.axhline(
        res.monitor_fire_fraction, color="#2b6cb0", lw=0.8, ls=":",
        label=(
            f"monitor tripwire ({res.monitor_fire_fraction:.2f}, "
            f"clean FPR={res.clean_fpr:.3f})"
        ),
    )
    ax_left.set_xlabel("alpha (0 = clean input, 1 = full distribution shift)")
    ax_left.set_ylabel("normalized output error / monitor fired fraction (0..1)")
    ax_left.set_ylim(-0.03, 1.05)

    # Right axis: the raw OOD score (mean rolling spread) and the threshold line.
    ax_right = ax_left.twinx()
    ax_right.plot(
        res.alphas, res.ood_score, "^--", color="#2f855a", lw=1.4, alpha=0.8,
        label="OOD score (mean rolling spread of hidden features)",
    )
    ax_right.axhline(
        res.threshold, color="#2f855a", lw=1.0, ls="--",
        label=f"calibrated OOD threshold ({res.threshold:.4f})",
    )
    ax_right.set_ylabel("OOD score: mean rolling spread of policy embedding")

    # Vertical markers for the two firing alphas.
    if not np.isnan(res.monitor_fires_at):
        ax_left.axvline(
            res.monitor_fires_at, color="#2b6cb0", lw=1.6,
            label=f"MONITOR fires @ alpha={res.monitor_fires_at:.2f}",
        )
    if not np.isnan(res.output_collapses_at):
        ax_left.axvline(
            res.output_collapses_at, color="#d63a3a", lw=1.6,
            label=f"OUTPUT collapses @ alpha={res.output_collapses_at:.2f}",
        )

    title = (
        "TRACK D: PHM internal-feature monitor vs stand-in policy output collapse\n"
        f"measured lead-time = {res.lead_time:+.2f} alpha "
        "(positive = monitor fires earlier)"
    )
    ax_left.set_title(title, fontsize=10)

    # Merge legends from both axes.
    h1, l1 = ax_left.get_legend_handles_labels()
    h2, l2 = ax_right.get_legend_handles_labels()
    ax_left.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def write_csv(res: SweepResult, out: Path) -> None:
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "alpha", "normalized_output_error", "ood_score_mean_rolling_spread",
            "monitor_fired_fraction", "raw_output_error",
        ])
        for i, a in enumerate(res.alphas):
            w.writerow([
                f"{a:.4f}", f"{res.output_error[i]:.6f}", f"{res.ood_score[i]:.6f}",
                f"{res.fired_fraction[i]:.6f}", f"{res.raw_output_error[i]:.6f}",
            ])


def build_result() -> SweepResult:
    policy, train_mse = train_policy()
    print(f"[train] stand-in policy trained, final MSE = {train_mse:.6f}")
    res = run_sweep(policy)
    return res


def main() -> int:
    res = build_result()
    write_csv(res, CSV)

    have_mpl = _try_matplotlib()
    if have_mpl:
        plot_matplotlib(res, PNG)
        print(f"[plot] matplotlib available: wrote {PNG}")
    else:
        print("[plot] matplotlib NOT available: wrote CSV only at", CSV)

    print("---")
    print(f"calibrated OOD threshold (p=1.0): {res.threshold:.6f}")
    print(f"clean false-positive rate (frame-flag on clean batch): {res.clean_fpr:.4f}")
    print(f"monitor tripwire (frame-flag fraction): {res.monitor_fire_fraction:.3f}")
    print(f"output degradation level (norm error): {res.output_degradation_level:.3f}")
    print(f"monitor fires at alpha    = {res.monitor_fires_at}")
    print(f"output collapses at alpha = {res.output_collapses_at}")
    print(f"MEASURED LEAD-TIME (alpha) = {res.lead_time:+.4f}")
    if res.lead_time > 0:
        print("HEADLINE: monitor fires BEFORE output collapse (positive lead-time).")
    elif res.lead_time == 0:
        print("HEADLINE: monitor and output cross at the SAME alpha (zero lead-time).")
    else:
        print("HEADLINE: monitor fires AFTER output collapse (NEGATIVE lead-time).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
