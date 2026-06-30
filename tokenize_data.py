"""
tokenize_data.py

Tokenize a raw TinyStories-style text corpus into a flat array of token IDs.

The corpus uses the GPT-2 special token <|endoftext|> as a document separator,
so we tokenize with tiktoken's GPT-2 BPE encoding and emit that single token id
(50256) at each boundary instead of splitting the literal string into ordinary
sub-word pieces.

Output: a 1-D numpy array of token ids saved as .npy next to the input file.
Run:    python tokenize_data.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tiktoken

EOT = "<|endoftext|>"


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
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Tokenize a raw text corpus into token ids.")
    parser.add_argument("--input", type=Path, default=here / "tinystories_roughly_1m_tokens.txt")
    parser.add_argument("--output", type=Path, default=here / "tinystories_tokens.npy")
    parser.add_argument("--encoding", default="gpt2", help="tiktoken encoding name (gpt2, cl100k_base, o200k_base)")
    args = parser.parse_args()

    tokenize_file(args.input, args.output, args.encoding)