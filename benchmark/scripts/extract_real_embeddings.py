# benchmark/scripts/extract_real_embeddings.py
"""Extract REAL per-token hidden-state streams from microsoft/phi-2.

Runs on system python (torch+transformers). For each (condition, seed) it runs a
deterministic generation, captures the last-layer hidden state of each newly
generated token, computes the objective failure label from the token ids, and
writes an .npz the numpy-only benchmark can consume.

Conditions:
  healthy  : coherent prompt, low-temperature sampling -> coherent generation
  collapse : repetition-prone prompt, greedy decoding   -> degenerate loop
  shift    : out-of-distribution / garbage-token prompt  -> off-region embeddings

Usage:
  /usr/bin/python3 benchmark/scripts/extract_real_embeddings.py --seeds 0 1 2 3 4
  /usr/bin/python3 benchmark/scripts/extract_real_embeddings.py --smoke   # tiny, writes fixture
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # for `import lib.labeling`
from lib.labeling import failure_labels  # noqa: E402

MODEL_ID = "microsoft/phi-2"
DATA = HERE.parent / "data"

PROMPTS = {
    "healthy": "The history of the Roman empire began when",
    "collapse": "List the number one. one one one one one",
    "shift": "qwx zzt æø ☃ 7f3a !! ;;; 中文混 garbledtokens xkcdq",
}
DECODE = {  # (greedy, temperature)
    "healthy": (False, 0.8),
    "collapse": (True, 1.0),
    "shift": (False, 1.2),
}


def generate_stream(model, tok, prompt, max_new, greedy, temperature, seed):
    torch.manual_seed(seed)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out_ids = ids.input_ids
    n_prompt = out_ids.shape[1]
    hs = []
    with torch.no_grad():
        for _ in range(max_new):
            o = model(out_ids, output_hidden_states=True)
            hs.append(o.hidden_states[-1][0, -1, :].float().cpu().numpy())
            logits = o.logits[0, -1, :]
            if greedy:
                nxt = torch.argmax(logits).view(1, 1)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1).view(1, 1)
            out_ids = torch.cat([out_ids, nxt], dim=1)
    feats = np.asarray(hs, dtype=np.float32)            # [max_new, hidden]
    gen_tokens = out_ids[0, n_prompt:].tolist()
    return feats, gen_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run (max_new=40, seed 0) that writes the committed CI fixture")
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, output_hidden_states=True
    ).to("cuda").eval()

    seeds = [0] if args.smoke else args.seeds
    max_new = 40 if args.smoke else args.max_new

    for seed in seeds:
        per_cond = {}
        for cond, prompt in PROMPTS.items():
            greedy, temp = DECODE[cond]
            feats, gen_tokens = generate_stream(
                model, tok, prompt, max_new, greedy, temp, seed
            )
            labels, onset = failure_labels(gen_tokens, n=3, window=20, threshold=0.5)
            per_cond[cond] = (feats, labels, onset)

        if args.smoke:
            out = DATA / "fixture_real_seed0.npz"
        else:
            out = DATA / f"real_embeddings_seed{seed}.npz"
        save = {"model_id": MODEL_ID, "max_new": max_new, "seed": seed,
                "conditions": list(PROMPTS.keys())}
        for cond, (feats, labels, onset) in per_cond.items():
            save[f"{cond}__feats"] = feats
            save[f"{cond}__labels"] = labels
            save[f"{cond}__onset"] = np.int64(onset)
        np.savez_compressed(out, **save)
        print(f"wrote {out}  " + "  ".join(
            f"{c}:{per_cond[c][0].shape} onset={per_cond[c][2]}" for c in PROMPTS))


if __name__ == "__main__":
    main()
