"""Out-of-process torch RND worker (run with /usr/bin/python3, which has torch).

Reads a single .npz (keys: id, test) and writes per-frame RND novelty scores
(higher = more OOD) to an output .npz (key: scores). A fixed random MLP target
network and a separately-initialised MLP predictor are created; the predictor
is trained by Adam to match the target on the ID features, then the squared
per-sample prediction error on the test features is the novelty score.

Burda et al. ICLR 2019 (arXiv:1810.12894). This is the gradient-trained
corroboration of lib/rnd.py's closed-form numpy RND.

Usage:
    /usr/bin/python3 rnd_torch_worker.py <in.npz> <out.npz> [epochs] [seed]
"""

import sys

import numpy as np
import torch
import torch.nn as nn


def main() -> int:
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    torch.manual_seed(seed)
    np.random.seed(seed)

    data = np.load(in_path)
    x_id = torch.tensor(data["id"], dtype=torch.float32)
    x_test = torch.tensor(data["test"], dtype=torch.float32)
    d = x_id.shape[1]
    proj = 128

    # Standardise on ID stats.
    mu = x_id.mean(0, keepdim=True)
    sd = x_id.std(0, keepdim=True).clamp_min(1e-6)
    x_id_n = (x_id - mu) / sd
    x_test_n = (x_test - mu) / sd

    def mlp() -> nn.Module:
        return nn.Sequential(
            nn.Linear(d, proj), nn.ReLU(),
            nn.Linear(proj, proj),
        )

    target = mlp()
    for p in target.parameters():
        p.requires_grad_(False)

    predictor = mlp()
    opt = torch.optim.Adam(predictor.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    with torch.no_grad():
        t_id = target(x_id_n)
    for _ in range(epochs):
        opt.zero_grad()
        pred = predictor(x_id_n)
        loss = loss_fn(pred, t_id)
        loss.backward()
        opt.step()

    with torch.no_grad():
        t_test = target(x_test_n)
        p_test = predictor(x_test_n)
        scores = ((t_test - p_test) ** 2).mean(dim=1).cpu().numpy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    final_mse = float(loss.detach())
    np.savez(out_path, scores=scores.astype(np.float64))
    print(f"rnd_torch_worker: device={device} epochs={epochs} "
          f"final_train_mse={final_mse:.6e} n_test={len(scores)}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
