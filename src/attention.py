"""Step 3 -- the mathematical trick behind self-attention.

Run this file directly (`python attention.py`). It builds intuition in four
stages, printing as it goes, then exercises the real causal Head from model.py.
Nothing here is imported by training; it exists purely to make the mechanism
legible before it's buried inside a stack of blocks.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from utils.model import Head

torch.manual_seed(1337)

B, T, C = 1, 5, 2  # tiny: 1 batch, 5 positions, 2 channels
x = torch.randn(B, T, C)
print("x (the sequence of token vectors):")
print(x[0])

# ---------------------------------------------------------------------------
# Stage 1: the goal. We want each position t to be a function of all positions
# <= t (it can see the past and itself, never the future). Start with the
# simplest such function: the average of all previous tokens.
# ---------------------------------------------------------------------------
xbow = torch.zeros(B, T, C)  # "bag of words" = running mean
for b in range(B):
    for t in range(T):
        xprev = x[b, : t + 1]        # (t+1, C) -- everything up to and incl. t
        xbow[b, t] = xprev.mean(0)
print("\nStage 1 -- averaged by an explicit loop:")
print(xbow[0])

# ---------------------------------------------------------------------------
# Stage 2: the trick. That loop is just a matrix multiply by a lower-triangular
# matrix of normalized weights. A @ x averages, and the triangular shape is what
# enforces causality (row t has zeros in the future columns).
# ---------------------------------------------------------------------------
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)   # each row sums to 1 -> a weighted average
print("\nStage 2 -- the weight matrix (row t averages cols 0..t):")
print(wei)
xbow2 = wei @ x                         # (T,T) @ (B,T,C) -> (B,T,C) via broadcast
print("matches the loop:", torch.allclose(xbow, xbow2))

# ---------------------------------------------------------------------------
# Stage 3: softmax form. Instead of hardcoding equal weights, start from zeros,
# mask the future to -inf, and softmax. With all-zero affinities this reproduces
# the uniform average -- but now the weights are free to become *data-dependent*.
# ---------------------------------------------------------------------------
tril = torch.tril(torch.ones(T, T))
aff = torch.zeros(T, T)
aff = aff.masked_fill(tril == 0, float("-inf"))
aff = F.softmax(aff, dim=-1)
print("\nStage 3 -- softmax of a masked all-zero affinity matrix:")
print(aff)
print("still the uniform average:", torch.allclose(aff @ x, xbow2))

# ---------------------------------------------------------------------------
# Stage 4: self-attention. The affinities stop being uniform. Each token emits a
# query and a key; affinity_{t,s} = q_t . k_s. Now position t pulls harder on the
# past tokens it finds relevant. The real Head does exactly this (plus values,
# scaling, and dropout). Note the output rows now differ from the flat average.
# ---------------------------------------------------------------------------
n_embd = 32
head = Head(n_embd=n_embd, head_size=16, block_size=T, dropout=0.0)
head.eval()
xb = torch.randn(B, T, n_embd)
out = head(xb)
print(f"\nStage 4 -- real Head output shape: {tuple(out.shape)} "
      f"(B, T, head_size); each row is a data-dependent mix of past values.")
