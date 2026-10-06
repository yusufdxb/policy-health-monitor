#!/usr/bin/env python3
"""Calibrate PHM on live policy latents and record PHM's output on the robot.

  calibrate  collect ``--seconds`` of /policy/embedding, compute the rolling-spread
             threshold at ``--percentile`` with phm_core.calibration, and write the
             .npz the phm_ood node loads (key ``threshold``). Also saves the raw
             latents so the threshold can be recomputed offline.
  record     write every /phm/health, /phm/verdicts and /policy/embedding arrival
             (embedding as stamp + spread, not the vector) to JSONL for ``--seconds``.

Read-only on the ROS graph: this script creates no publishers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from phm_core.calibration import calibrate_threshold, rolling_spread

_BE = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=100,
    durability=DurabilityPolicy.VOLATILE,
)
_REL = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=100,
    durability=DurabilityPolicy.VOLATILE,
)


def _spin_for(node: Node, seconds: float) -> None:
    end = time.monotonic() + seconds
    while rclpy.ok() and time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.05)


def calibrate(args: argparse.Namespace) -> int:
    from phm_msgs.msg import PolicyEmbedding  # noqa: PLC0415

    node = Node("phm_go2_calibrate")
    frames: list[np.ndarray] = []
    node.create_subscription(
        PolicyEmbedding,
        "/policy/embedding",
        lambda m: frames.append(np.asarray(m.embedding, dtype=np.float64)),
        _BE,
    )
    _spin_for(node, args.seconds)
    node.destroy_node()
    if len(frames) < 2 * args.window:
        print(f"only {len(frames)} frames in {args.seconds}s; need >= {2 * args.window}")
        return 1
    hidden = np.vstack(frames)
    spreads = rolling_spread(hidden, args.window)
    valid = spreads[~np.isnan(spreads)]
    threshold = calibrate_threshold(valid, args.percentile)
    np.savez(
        args.out,
        threshold=threshold,
        window=args.window,
        percentile=args.percentile,
        n_frames=len(frames),
        seconds=args.seconds,
        latents=hidden.astype(np.float32),
    )
    print(json.dumps({
        "out": args.out,
        "n_frames": len(frames),
        "rate_hz": round(len(frames) / args.seconds, 2),
        "window": args.window,
        "percentile": args.percentile,
        "threshold": threshold,
        "spread_min": float(valid.min()),
        "spread_p50": float(np.median(valid)),
        "spread_max": float(valid.max()),
    }))
    return 0


def record(args: argparse.Namespace) -> int:
    from phm_msgs.msg import DetectorVerdict, PolicyEmbedding, PolicyHealthStatus  # noqa: PLC0415

    node = Node("phm_go2_record")
    out = open(args.out, "w", buffering=1)  # noqa: SIM115
    window: list[np.ndarray] = []

    def stamp(m) -> float:
        return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9

    def on_health(m) -> None:
        out.write(json.dumps({
            "t": time.time(), "topic": "/phm/health", "stamp": stamp(m), "state": int(m.state),
            "score": float(m.score), "source": m.source, "reason": m.reason,
            "action": int(m.suggested_action),
        }) + "\n")

    def on_verdict(m) -> None:
        out.write(json.dumps({
            "t": time.time(), "topic": "/phm/verdicts", "stamp": stamp(m), "source": m.source,
            "score": float(m.score), "violating": bool(m.violating), "reason": m.reason,
        }) + "\n")

    def on_emb(m) -> None:
        window.append(np.asarray(m.embedding, dtype=np.float64))
        if len(window) > args.window:
            window.pop(0)
        spread = (
            float(np.var(np.vstack(window), axis=0).sum())
            if len(window) == args.window else None
        )
        out.write(json.dumps({
            "t": time.time(), "topic": "/policy/embedding", "stamp": stamp(m),
            "dim": int(m.dim), "spread": spread,
        }) + "\n")

    node.create_subscription(PolicyHealthStatus, "/phm/health", on_health, _REL)
    node.create_subscription(DetectorVerdict, "/phm/verdicts", on_verdict, _REL)
    node.create_subscription(PolicyEmbedding, "/policy/embedding", on_emb, _BE)
    _spin_for(node, args.seconds)
    node.destroy_node()
    out.close()
    print(f"wrote {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--seconds", type=float, default=60.0)
    c.add_argument("--window", type=int, default=30)
    c.add_argument("--percentile", type=float, default=1.0)
    c.add_argument("--out", required=True)
    r = sub.add_parser("record")
    r.add_argument("--seconds", type=float, required=True)
    r.add_argument("--window", type=int, default=30)
    r.add_argument("--out", required=True)
    args = ap.parse_args()
    rclpy.init()
    try:
        return calibrate(args) if args.cmd == "calibrate" else record(args)
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
