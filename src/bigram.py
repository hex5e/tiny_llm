from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.data import DEFAULT_TOKENS, get_batch, load_tokens

# ----------------------------- config ---------------------------------------
TOKENS_PATH = DEFAULT_TOKENS  # data/tinystories_tokens.npy (resolved in data.py)
VOCAB_SIZE = 50257        # GPT-2 BPE vocab; MUST match the tokenizer in tokenize_data.py
N_EMBD = 64               # bottleneck width. Set == VOCAB_SIZE for a pure lookup
                          # bigram (warning: 50257**2 ~= 2.5B params, ~10GB fp32).
BLOCK_SIZE = 64           # throughput knob only for a bigram (it ignores all but
                          # the current token); becomes real context in the GPT.
BATCH_SIZE = 32
MAX_ITERS = 5000
EVAL_INTERVAL = 500
EVAL_ITERS = 200
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 1337


class BigramLanguageModel(nn.Module):
    """Token-only next-token predictor.

    n_embd == vocab_size -> textbook lookup bigram (any transition matrix).
    n_embd  < vocab_size -> rank-constrained bigram: logit matrix factored as
    (vocab x n_embd) @ (n_embd x vocab). Same order-1 Markov assumption, far
    fewer params, and exactly the embedding + lm_head skeleton the GPT reuses.
    """

    def __init__(self, vocab_size: int, n_embd: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        x = self.token_embedding(idx)        # (B, T, n_embd)
        logits = self.lm_head(x)             # (B, T, vocab_size)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


@torch.no_grad()
def estimate_loss(model, splits):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    torch.manual_seed(SEED)
    train_data, val_data = load_tokens(TOKENS_PATH)
    splits = {"train": train_data, "val": val_data}
    print(f"train tokens : {len(train_data):,}")
    print(f"val tokens   : {len(val_data):,}")

    model = BigramLanguageModel(VOCAB_SIZE, N_EMBD).to(DEVICE)
    print(f"device       : {DEVICE}")
    print(f"params       : {sum(p.numel() for p in model.parameters()):,}")
    print(f"random-guess loss ~= ln({VOCAB_SIZE}) = "
          f"{torch.log(torch.tensor(float(VOCAB_SIZE))):.4f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    for it in range(MAX_ITERS):
        if it % EVAL_INTERVAL == 0 or it == MAX_ITERS - 1:
            losses = estimate_loss(model, splits)
            print(f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}")
        x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    start = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    out_ids = model.generate(start, max_new_tokens=200)[0].tolist()
    print("\n--- sample (token ids) ---")
    print(out_ids)
    try:
        import tiktoken
        print("\n--- sample (decoded) ---")
        print(tiktoken.get_encoding("gpt2").decode(out_ids))
    except Exception as e:  # noqa: BLE001
        print(f"\n(skip decode: {e})")


if __name__ == "__main__":
    main()
