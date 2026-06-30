# tiny_llm
Simple LLM trained on the Hugging Face TinyStories dataset

Run order:

```
uv run python src/gather_data.py
uv run python src/tokenize_data.py
uv run python src/bigram.py — your ~5–6 loss baseline
uv run python src/attention.py — see the masked-matmul trick
uv run python src/train.py — trains, early-stops, saves checkpoints/gpt.pt
uv run python src/sample.py --tokens 500 --prompt "Once upon a time"
```