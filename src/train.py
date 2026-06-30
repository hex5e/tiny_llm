from __future__ import annotations

import time

import torch

from utils.data import CKPT_DIR, DEFAULT_TOKENS, get_batch, load_tokens
from utils.model import GPT

# ----------------------------- config ---------------------------------------
TOKENS_PATH = DEFAULT_TOKENS
VOCAB_SIZE = 50257       # MUST match the tokenizer used in tokenize_data.py
N_EMBD = 128
N_HEAD = 4
N_LAYER = 4
BLOCK_SIZE = 128
BATCH_SIZE = 32
DROPOUT = 0.2            # high on purpose: 1M tokens overfits fast
LR = 3e-4
WEIGHT_DECAY = 0.1
MAX_ITERS = 5000
EVAL_INTERVAL = 250
EVAL_ITERS = 200
PATIENCE = 6            # stop if val loss hasn't improved in this many evals
SEED = 1337

# Set to your machine's PHYSICAL core count (not logical/hyperthreaded -- over-
# subscription slows matmuls). None = leave torch's default.
NUM_THREADS: int | None = None

CKPT_PATH = CKPT_DIR / "gpt.pt"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def estimate_loss(model, splits, device):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(data, BLOCK_SIZE, BATCH_SIZE, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    torch.manual_seed(SEED)
    device = pick_device()
    if NUM_THREADS is not None and device == "cpu":
        torch.set_num_threads(NUM_THREADS)

    train_data, val_data = load_tokens(TOKENS_PATH)
    splits = {"train": train_data, "val": val_data}

    model = GPT(VOCAB_SIZE, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    print(f"device       : {device} (threads={torch.get_num_threads()})")
    print(f"train tokens : {len(train_data):,}   val tokens: {len(val_data):,}")
    print(f"params       : {model.num_params():,}")
    print(f"config       : L={N_LAYER} H={N_HEAD} d={N_EMBD} blk={BLOCK_SIZE} "
          f"batch={BATCH_SIZE} dropout={DROPOUT}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_iter = 0
    no_improve = 0
    t0 = time.time()

    for it in range(MAX_ITERS):
        if it % EVAL_INTERVAL == 0 or it == MAX_ITERS - 1:
            losses = estimate_loss(model, splits, device)
            dt = time.time() - t0
            flag = ""
            if losses["val"] < best_val - 1e-4:
                best_val, best_iter, no_improve = losses["val"], it, 0
                # checkpoint the BEST model, plus enough config to rebuild it
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": dict(vocab_size=VOCAB_SIZE, n_embd=N_EMBD,
                                       n_head=N_HEAD, n_layer=N_LAYER,
                                       block_size=BLOCK_SIZE, dropout=DROPOUT),
                        "iter": it, "val_loss": best_val,
                    },
                    CKPT_PATH,
                )
                flag = "  <- saved best"
            else:
                no_improve += 1
            print(f"iter {it:5d} | train {losses['train']:.4f} | "
                  f"val {losses['val']:.4f} | {dt:6.1f}s{flag}")
            # early stop: val has U-turned and stayed up -> we're memorizing
            if no_improve >= PATIENCE:
                print(f"\nearly stop: no val improvement for {PATIENCE} evals.")
                break

        x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"\nbest val {best_val:.4f} at iter {best_iter}. checkpoint: {CKPT_PATH}")

    # quick sample from the in-memory model (use sample.py for the saved best)
    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    out_ids = model.generate(start, max_new_tokens=200)[0].tolist()
    try:
        import tiktoken
        print("\n--- sample ---")
        print(tiktoken.get_encoding("gpt2").decode(out_ids))
    except Exception as e:  # noqa: BLE001
        print(f"\nsample ids: {out_ids}\n(skip decode: {e})")


if __name__ == "__main__":
    main()
