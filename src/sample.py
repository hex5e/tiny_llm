"""Load the best checkpoint and generate text. Run: `python sample.py`.

Reconstructs the GPT from the config saved alongside the weights, so it stays in
sync with whatever you trained -- no need to keep hyperparameters in two places.
"""
from __future__ import annotations

import argparse

import torch

from utils.data import CKPT_DIR
from utils.model import GPT

CKPT_PATH = CKPT_DIR / "gpt.pt"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=500, help="how many tokens to generate")
    ap.add_argument("--prompt", type=str, default="", help="optional text prompt")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    device = pick_device()
    torch.manual_seed(args.seed)

    ckpt = torch.load(CKPT_PATH, map_location=device)
    model = GPT(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {CKPT_PATH} | iter {ckpt['iter']} | val {ckpt['val_loss']:.4f}")

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")

    if args.prompt:
        ids = enc.encode(args.prompt)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
    else:
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)

    out_ids = model.generate(idx, max_new_tokens=args.tokens)[0].tolist()
    print("\n--- generated ---")
    print(enc.decode(out_ids))


if __name__ == "__main__":
    main()
