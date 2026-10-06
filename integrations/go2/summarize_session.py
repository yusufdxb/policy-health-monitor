#!/usr/bin/env python3
"""Reduce an onboard_session.sh evidence directory to the numbers it supports.

Reads only the recorded files (JSONL from phm_go2_probe.py record, the embedder
logs, calibrate.json) and prints one JSON object. Stdlib only, so it runs on the
robot or on a workstation after copying the session directory back.

All times are the Jetson's clock: verdict/health header stamps and the embedder's
ROS log stamps come from the same host.
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

STATES = {0: "OK", 1: "DEGRADED", 2: "INTERVENE", 3: "STOP"}
_FAULT_RE = re.compile(r"\[(\d+\.\d+)\] \[phoenix_shadow_embedder\]: FAULT INJECTED")


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _fault_time(log: Path) -> float | None:
    if not log.exists():
        return None
    m = _FAULT_RE.search(log.read_text())
    return float(m.group(1)) if m else None


def _health_summary(rows: list[dict]) -> dict:
    health = [r for r in rows if r["topic"] == "/phm/health"]
    states = collections.Counter(STATES[r["state"]] for r in health)
    reasons = collections.Counter(
        f'{STATES[r["state"]]} {r["source"]}' for r in health if r["state"] != 0
    )
    return {
        "health_msgs": len(health),
        "state_counts": dict(states),
        "non_ok_fraction": round(1 - states.get("OK", 0) / len(health), 4) if health else None,
        "non_ok_by_source": dict(reasons.most_common(10)),
    }


def _ood_summary(rows: list[dict]) -> dict:
    ood = [r for r in rows if r["topic"] == "/phm/verdicts" and r["source"] == "phm_ood_cpp"]
    emb = [r for r in rows if r["topic"] == "/policy/embedding"]
    spreads = sorted(r["spread"] for r in emb if r["spread"] is not None)
    span = emb[-1]["stamp"] - emb[0]["stamp"] if len(emb) > 1 else 0.0
    return {
        "ood_verdicts": len(ood),
        "ood_violating": sum(r["violating"] for r in ood),
        "ood_violating_fraction": (
            round(sum(r["violating"] for r in ood) / len(ood), 4) if ood else None
        ),
        "embeddings": len(emb),
        "embedding_rate_hz": round((len(emb) - 1) / span, 2) if span > 0 else None,
        "spread_min": spreads[0] if spreads else None,
        "spread_median": spreads[len(spreads) // 2] if spreads else None,
    }


def _first_after(rows: list[dict], t0: float, pred) -> dict | None:
    for r in sorted(rows, key=lambda r: r["stamp"]):
        if r["stamp"] >= t0 and pred(r):
            return {
                "latency_s": round(r["stamp"] - t0, 3),
                "source": r.get("source"),
                "reason": r.get("reason"),
                "state": STATES.get(r.get("state")) if "state" in r else None,
            }
    return None


# Pre-fault rows inside the first SETTLE_SEC of a trial are excluded from its
# false-alarm count: a trial starts right after the previous trial's induced
# fault, whose STOP verdicts are still clearing.
SETTLE_SEC = 5.0


def _fault_summary(rows: list[dict], t_fault: float | None, verdict_sources: tuple) -> dict:
    out: dict = {"fault_time": t_fault}
    if t_fault is None:
        out["error"] = "no FAULT INJECTED line in the embedder log"
        return out
    emb = [r["stamp"] for r in rows if r["topic"] == "/policy/embedding"]
    settle = (min(emb) if emb else t_fault) + SETTLE_SEC
    before = [r for r in rows if settle <= r["stamp"] < t_fault]
    out["before_fault"] = {**_health_summary(before), **_ood_summary(before)}
    out["first_violating_verdict"] = _first_after(
        rows,
        t_fault,
        lambda r: r["topic"] == "/phm/verdicts" and r["violating"]
        and r["source"].startswith(verdict_sources),
    )
    for name, level in (("first_health_degraded_or_worse", 1),
                        ("first_health_intervene_or_worse", 2),
                        ("first_health_stop", 3)):
        out[name] = _first_after(
            rows, t_fault, lambda r, lv=level: r["topic"] == "/phm/health" and r["state"] >= lv
        )
    after = [r for r in rows if r["stamp"] >= t_fault]
    out["after_fault"] = _health_summary(after)
    return out


_FAULT_SOURCES = {
    "freeze": ("phm_ood_cpp",),
    "stop": ("dead:/policy/embedding", "freq:/policy/embedding", "phm_ood_cpp"),
}


def _aggregate(trials: list[dict]) -> dict:
    """Min / median / max of each detection latency across trials."""
    out: dict = {"n_trials": len(trials)}
    for key in ("first_violating_verdict", "first_health_degraded_or_worse",
                "first_health_intervene_or_worse", "first_health_stop"):
        lat = sorted(t[key]["latency_s"] for t in trials if t.get(key))
        out[key] = {
            "detected": f"{len(lat)}/{len(trials)}",
            "min_s": lat[0] if lat else None,
            "median_s": lat[len(lat) // 2] if lat else None,
            "max_s": lat[-1] if lat else None,
        }
    pre = [t["before_fault"] for t in trials if "before_fault" in t]
    out["pre_fault_health_msgs"] = sum(p["health_msgs"] for p in pre)
    out["pre_fault_non_ok_msgs"] = sum(
        p["health_msgs"] - p["state_counts"].get("OK", 0) for p in pre
    )
    return out


def main() -> int:
    d = Path(sys.argv[1])
    summary: dict = {"session": d.name}
    if (d / "session.txt").exists():
        summary["session_txt"] = d.joinpath("session.txt").read_text().splitlines()
    if (d / "calibrate.json").exists():
        summary["calibration"] = json.loads((d / "calibrate.json").read_text().splitlines()[-1])
    if (d / "cpu_nominal.csv").exists():
        summary["cpu_nominal"] = [
            line for line in (d / "cpu_nominal.csv").read_text().splitlines()
            if not line.startswith("#")
        ]
    nominal = _load(d / "nominal.jsonl")
    summary["nominal"] = {**_health_summary(nominal), **_ood_summary(nominal)}
    for fault, sources in _FAULT_SOURCES.items():
        trials = []
        for log in sorted(d.glob(f"embedder_{fault}_*.log")):
            n = log.stem.rsplit("_", 1)[-1]
            trial = _fault_summary(_load(d / f"fault_{fault}_{n}.jsonl"), _fault_time(log), sources)
            trials.append({"trial": int(n), **trial})
        summary[f"fault_{fault}"] = {"aggregate": _aggregate(trials), "trials": trials}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
