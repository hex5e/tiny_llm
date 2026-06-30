from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tiktoken

EOT = "<|endoftext|>"
INPUT_FILE_PATH = "../data/tinystories_roughly_1m_tokens.txt"
OUTPUT_FILE_PATH = "../data/tinystories_tokens.npy"
ENCODING = "gpt2"


def tokenize_file(input_path: Path, output_path: Path, encoding_name: str = "gpt2") -> np.ndarray:
    enc = tiktoken.get_encoding(encoding_name)

    text = input_path.read_text(encoding="utf-8")

    # allowed_special lets the literal "<|endoftext|>" map to its single id;
    # without this, tiktoken raises rather than silently splitting it.
    ids = enc.encode(text, allowed_special={EOT})

    # GPT-2's vocab (50257) fits in uint16; pick a wider dtype automatically if
    # you swap to a larger encoding like cl100k_base or o200k_base.
    dtype = np.uint16 if enc.max_token_value < 2**16 else np.uint32
    arr = np.array(ids, dtype=dtype)
    np.save(output_path, arr)

    n_docs = ids.count(enc.eot_token) if hasattr(enc, "eot_token") else text.count(EOT)
    print(f"Read         : {input_path}")
    print(f"Encoding     : {encoding_name}")
    print(f"Characters   : {len(text):,}")
    print(f"Documents    : {n_docs:,}")
    print(f"Tokens       : {len(ids):,}")
    print(f"Unique tokens: {len(set(ids)):,}")
    print(f"Saved        : {output_path.with_suffix('.npy')}  (shape={arr.shape}, dtype={arr.dtype})")
    return arr


if __name__ == "__main__":

    tokenize_file(Path(INPUT_FILE_PATH), Path(OUTPUT_FILE_PATH), ENCODING)