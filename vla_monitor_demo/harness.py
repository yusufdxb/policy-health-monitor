"""TRACK D harness: PHM internal-feature OOD monitor fires before output collapse.

WHAT THIS IS
============
A "monitor fires before output collapse" demo. We train a small stand-in policy
in-script (seconds), then sweep an input-perturbation strength alpha from 0.0 to
1.0. At each alpha we measure two things over a batch:

  (a) the policy OUTPUT error, normalized 0..1, against the clean-input output;
  (b) the PHM OOD score, computed as ``rolling_spread`` on the policy's tapped
      hidden-layer features (the "policy embedding").

The OOD threshold is calibrated on the alpha=0 (clean) features using
``calibrate_threshold``. The headline question, mirroring the phantom-braking
alpha sweep, is whether the monitor crosses its threshold at a LOWER alpha than
the output crosses a degradation level: positive lead-time = early warning.

STAND-IN POLICY CAVEAT
======================
The policy here is a stand-in: a small in-script-trained MLP, NOT a VLA. lerobot
/SmolVLA is uninstallable in this environment (rerun-sdk / datasets / opencv
dependency pins are unresolvable, verified across two attempts). A SmolVLA / Octo
swap is pending a clean lerobot install. The VALUE of the demo is the monitor
methodology, not the specific learned model: any model with a tappable hidden
layer drops into the same harness.

REUSED MATH (not duplicated)
============================
The OOD math (``rolling_spread``, ``calibrate_threshold``) is imported from
``phm_core.calibration`` (this repo, src/phm_core/phm_core/calibration.py),
which itself ports it byte-faithfully from phantom-braking. We add that package
dir to ``sys.path`` and import it; we do NOT re-implement the math here.

  phm_core.calibration.rolling_spread       <- e6_detector.py:16-23
  phm_core.calibration.calibrate_threshold  <- e6_detector.py:26-31

The alpha-sweep + calibrate-on-clean + fires-at-alpha methodology mirrors
phantom-braking src/e6_detector.py:69-84 (``evaluate_on_e4``), where the
detector's per-alpha fired-fraction is read off an input-perturbation sweep.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# --- Reuse the OOD math from phm_core (do NOT duplicate it) ------------------
# phm_core.calibration lives at src/phm_core/phm_core/calibration.py. We add the
# package's parent dir to sys.path so `import phm_core.calibration` resolves.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_PHM_CORE_DIR = _REPO_ROOT / "src" / "phm_core"
if str(_PHM_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(_PHM_CORE_DIR))

from phm_core.calibration import (  # noqa: E402  (path inserted above)
    calibrate_threshold,
    rolling_spread,
)


# ---------------------------------------------------------------------------
# Stand-in policy: a small MLP with a tappable hidden layer.
# ---------------------------------------------------------------------------
class StandInPolicy(nn.Module):
    """Small MLP regressor. NOT a VLA: a stand-in with a tappable hidden layer.

    Input: a synthetic 16-D observation vector (think "flattened control state").
    Output: a 2-D continuous action (think "twist": vx, yaw).
    The penultimate layer (``trunk`` output, ``hidden_dim`` wide) is the "policy
    embedding" the PHM monitor watches. ``forward`` returns (action, hidden).
    """

    def __init__(self, in_dim: int = 16, hidden_dim: int = 48, out_dim: int = 2):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor):
        hidden = self.trunk(x)          # (B, hidden_dim) -> the policy embedding
        action = self.head(hidden)      # (B, out_dim)
        return action, hidden


# ---------------------------------------------------------------------------
# Synthetic control/regression task.
# ---------------------------------------------------------------------------
def _true_control(x: np.ndarray) -> np.ndarray:
    """Ground-truth control law the stand-in policy learns to imitate.

    A smooth nonlinear map from a 16-D observation to a 2-D action. Deterministic
    so the policy has a real, meaningful target (not noise fitting).

    The law reads a ROBUST low-dimensional statistic: the MEAN of the first 8
    dims and the mean of the last 8 dims. Averaging cancels zero-mean per-dim
    noise, so the trained output is locally robust to small input perturbations.
    This is deliberate: a realistic policy whose action stays plausible under
    mild shift, which is exactly the regime where an early-warning monitor earns
    its keep (the embedding goes off-manifold before the action visibly breaks).
    """
    a = np.tanh(x[..., :8].mean(axis=-1, keepdims=True) * 2.0)       # (.,1)
    b = np.sin(x[..., 8:].mean(axis=-1, keepdims=True) * 2.0)        # (.,1)
    return np.concatenate([a, b], axis=-1).astype(np.float32)


def make_dataset(n: int, in_dim: int, rng: np.random.Generator):
    """Draw in-distribution observations from a standard normal and label them."""
    x = rng.standard_normal((n, in_dim)).astype(np.float32)
    y = _true_control(x)
    return x, y


def train_policy(
    in_dim: int = 16,
    hidden_dim: int = 48,
    out_dim: int = 2,
    n_train: int = 8192,
    epochs: int = 350,
    lr: float = 1e-2,
    seed: int = 0,
) -> tuple[StandInPolicy, float]:
    """Train the stand-in policy in-script (seconds) on the synthetic task.

    Returns the trained policy and the final training MSE. The policy is small
    enough that this finishes in well under a second on CPU.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x_np, y_np = make_dataset(n_train, in_dim, rng)
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)

    policy = StandInPolicy(in_dim, hidden_dim, out_dim)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    policy.train()
    final_loss = float("nan")
    for _ in range(epochs):
        opt.zero_grad()
        pred, _ = policy(x)
        loss = loss_fn(pred, y)
        final_loss = float(loss.detach())
        loss.backward()
        opt.step()
    policy.eval()
    return policy, final_loss


# ---------------------------------------------------------------------------
# Input perturbation: additive gaussian noise + a fixed feature shift.
# ---------------------------------------------------------------------------
def perturb_inputs(
    x: np.ndarray, alpha: float, rng: np.random.Generator, shift_scale: float = 3.0
) -> np.ndarray:
    """Apply an alpha-scaled input distribution shift.

    Two stacked effects, both scaled by alpha (0 = clean, 1 = full shift):
      - additive gaussian noise (std = alpha * 1.0),
      - a deterministic mean shift along a fixed direction (alpha * shift_scale),
        which pushes inputs off the standard-normal training manifold.
    At alpha=0 this returns x unchanged.
    """
    if alpha == 0.0:
        return x.copy()
    in_dim = x.shape[-1]
    noise = rng.standard_normal(x.shape).astype(np.float32) * (alpha * 1.0)
    direction = np.ones(in_dim, dtype=np.float32) / np.sqrt(in_dim)
    shift = (alpha * shift_scale) * direction
    return (x + noise + shift).astype(np.float32)


# ---------------------------------------------------------------------------
# Alpha sweep.
# ---------------------------------------------------------------------------
@dataclass
class SweepResult:
    alphas: np.ndarray
    output_error: np.ndarray          # normalized 0..1
    ood_score: np.ndarray             # mean rolling_spread of hidden features
    fired_fraction: np.ndarray        # fraction of frames below the threshold
    threshold: float
    window: int
    clean_fpr: float                  # frame-flag rate on the clean batch
    output_degradation_level: float   # normalized-error level = output collapse
    monitor_fire_fraction: float      # frame-flag level = monitor tripwire
    monitor_fires_at: float           # alpha (nan if never)
    output_collapses_at: float        # alpha (nan if never)
    lead_time: float                  # output_collapses_at - monitor_fires_at
    raw_output_error: np.ndarray = field(default=None)  # un-normalized MSE


def _hidden_for_alpha(
    policy: StandInPolicy, x_clean: np.ndarray, alpha: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Run the policy on alpha-perturbed inputs; return (actions, hidden) numpy."""
    x_p = perturb_inputs(x_clean, alpha, rng)
    with torch.no_grad():
        act, hid = policy(torch.from_numpy(x_p))
    return act.numpy(), hid.numpy()


def run_sweep(
    policy: StandInPolicy,
    *,
    n_eval: int = 600,
    in_dim: int = 16,
    n_alphas: int = 21,
    window: int = 30,
    percentile: float = 1.0,
    output_degradation_level: float = 0.5,
    monitor_fire_fraction: float = 0.05,
    seed: int = 7,
) -> SweepResult:
    """Sweep alpha 0..1 and measure output error and the PHM OOD score.

    OOD score per alpha: we treat the batch of hidden vectors as a sequence
    (T = n_eval frames, D = hidden_dim) and compute ``rolling_spread`` over it,
    exactly as phantom-braking treats a drive's hidden_state sequence
    (e6_detector.py:69-84). The OOD SCORE plotted is the mean rolling spread.
    The FIRED FRACTION is the fraction of windows whose spread is BELOW the
    threshold (collapse = low spread), matching ``evaluate_on_e4``
    (e6_detector.py:69-84).

    The threshold is calibrated on the alpha=0 (clean) hidden features via
    ``calibrate_threshold`` at the given percentile (1st percentile, identical to
    phantom-braking e6_detector.py:26-31).

    Two distinct, intentionally different decision levels:
      - the MONITOR fires when the frame-flag fraction first exceeds
        ``monitor_fire_fraction`` (default 0.05). This is a SENSITIVE tripwire,
        set just above the clean false-positive floor (~0.01 here), which is the
        whole point of an early-warning monitor: trip on the first statistically
        real departure from the calibrated clean distribution.
      - the OUTPUT collapses when the normalized output error first exceeds
        ``output_degradation_level`` (default 0.5), i.e. the policy action is
        halfway to its worst-case error. This is a "policy is broken" level.
    The lead-time is the alpha gap between these two. They are NOT the same level
    on purpose: a sensitive monitor vs a real failure. We report whatever gap the
    measurement gives, including zero or negative.
    """
    rng = np.random.default_rng(seed)
    x_clean, _ = make_dataset(n_eval, in_dim, rng)

    # Clean-input reference output (alpha = 0), used to normalize output error.
    act_clean, hid_clean = _hidden_for_alpha(policy, x_clean, 0.0, np.random.default_rng(seed))

    # Calibrate the OOD threshold on the clean hidden features.
    clean_spread = rolling_spread(hid_clean, window)
    threshold = calibrate_threshold(clean_spread, percentile)
    clean_valid = clean_spread[~np.isnan(clean_spread)]
    clean_fpr = (
        float(np.mean(clean_valid < threshold)) if clean_valid.size else float("nan")
    )

    alphas = np.linspace(0.0, 1.0, n_alphas)
    raw_err = np.zeros(n_alphas)
    ood = np.zeros(n_alphas)
    fired = np.zeros(n_alphas)

    for i, a in enumerate(alphas):
        # Fresh rng per alpha so perturbation noise is reproducible and
        # independent of sweep order.
        a_rng = np.random.default_rng(1000 + i)
        act, hid = _hidden_for_alpha(policy, x_clean, float(a), a_rng)

        # (a) Output error vs the clean-input output (mean L2 over the batch).
        raw_err[i] = float(np.mean(np.linalg.norm(act - act_clean, axis=-1)))

        # (b) OOD score = mean rolling spread of the hidden features.
        spread = rolling_spread(hid, window)
        valid = spread[~np.isnan(spread)]
        ood[i] = float(np.mean(valid)) if valid.size else float("nan")
        fired[i] = float(np.mean(valid < threshold)) if valid.size else float("nan")

    # Normalize output error to 0..1 by its own sweep max (robust if monotone).
    err_max = float(raw_err.max()) if raw_err.max() > 0 else 1.0
    output_error = raw_err / err_max

    monitor_fires_at = _first_crossing_up(alphas, fired, monitor_fire_fraction)
    output_collapses_at = _first_crossing_up(
        alphas, output_error, output_degradation_level
    )
    lead_time = float(output_collapses_at - monitor_fires_at)

    return SweepResult(
        alphas=alphas,
        output_error=output_error,
        ood_score=ood,
        fired_fraction=fired,
        threshold=threshold,
        window=window,
        clean_fpr=clean_fpr,
        output_degradation_level=output_degradation_level,
        monitor_fire_fraction=monitor_fire_fraction,
        monitor_fires_at=monitor_fires_at,
        output_collapses_at=output_collapses_at,
        lead_time=lead_time,
        raw_output_error=raw_err,
    )


def _first_crossing_up(x: np.ndarray, y: np.ndarray, level: float) -> float:
    """Smallest x where y first rises to or above ``level``.

    Returns NaN if y never reaches the level. Used identically for the monitor
    (fired_fraction crossing its fire fraction) and the output (normalized error
    crossing the degradation level), so the lead-time is an apples-to-apples
    alpha gap.
    """
    above = y >= level
    if not above.any():
        return float("nan")
    return float(x[int(np.argmax(above))])
