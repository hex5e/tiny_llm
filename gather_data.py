from datasets import load_dataset

ds = load_dataset("roneneldan/TinyStories", split="train")

texts = []
word_count = 0

for row in ds:
    text = row["text"]
    texts.append(text)
    word_count += len(text.split())

    # crude approximation: words are not the same as tokens,
    # but this gets you into the right ballpark
    if word_count >= 750_000:
        break

corpus = "\n\n".join(texts)

with open("tinystories_roughly_1m_tokens.txt", "w", encoding="utf-8") as f:
    f.write(corpus)