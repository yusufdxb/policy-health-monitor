# benchmark/lib/real_stream.py
"""Load real phi-2 hidden-state streams written by scripts/extract_real_embeddings.py.

Pure numpy. No torch. Consumes the .npz produced out-of-process.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
FIXTURE = DATA / "fixture_real_seed0.npz"


@dataclass(frozen=True)
class RealStream:
    cond: str
    feats: np.ndarray   # [T, hidden] float32
    labels: np.ndarray  # [T] int {0,1}
    onset: int


def load_real_stream(path) -> dict[str, RealStream]:
    z = np.load(path, allow_pickle=False)
    conds = [c for c in z.files if c.endswith("__feats")]
    out: dict[str, RealStream] = {}
    for key in conds:
        cond = key[: -len("__feats")]
        out[cond] = RealStream(
            cond=cond,
            feats=z[f"{cond}__feats"].astype(np.float32),
            labels=z[f"{cond}__labels"].astype(np.int64),
            onset=int(z[f"{cond}__onset"]),
        )
    return out


def seed_paths(seeds) -> list[Path]:
    return [DATA / f"real_embeddings_seed{s}.npz" for s in seeds]
