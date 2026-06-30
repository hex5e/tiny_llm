from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

# ----------------------------------------------------------------------------
# Path resolution. Everything is resolved relative to THIS file, so scripts work
# no matter the current working directory (run from project root, from src/, or
# via an IDE -- all fine).
#   <project_root>/
#       src/        <- this file lives here
#       data/       <- *.npy token files
#       checkpoints/<- saved models
# ----------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
CKPT_DIR = PROJECT_ROOT / "checkpoints"
DEFAULT_TOKENS = DATA_DIR / "tinystories_tokens.npy"


def load_tokens(
    path: str | Path | None = None,
    val_frac: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the .npy token stream and split into (train, val) by position.

    `path` defaults to data/tinystories_tokens.npy. The split is positional (the
    last `val_frac` of the stream) rather than random so no training context
    window leaks across the boundary into the validation set.

    The file is uint16/uint32 (from tokenize_data.py). torch has no native uint16
    tensor and embedding indices must be int64, so we cast up once here.
    """
    path = DEFAULT_TOKENS if path is None else Path(path)
    data = np.load(path)
    data = torch.from_numpy(data.astype(np.int64))
    n_train = int(len(data) * (1.0 - val_frac))
    return data[:n_train], data[n_train:]


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of contiguous chunks.

    Returns (x, y) each of shape (batch_size, block_size), where y is x shifted
    right by one token: y[:, t] is the target that follows x[:, t]. So each row
    yields `block_size` independent (input, next-token) training pairs.
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)

if __name__ == "__main__":
    print(f"SRC_DIR: {SRC_DIR}")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"CKPT_DIR: {CKPT_DIR}")
    print(f"DEFAULT_TOKENS: {DEFAULT_TOKENS}")