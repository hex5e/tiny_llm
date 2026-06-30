# from datasets import load_dataset
# ds = load_dataset("roneneldan/TinyStories", split="train")
from pathlib import Path


def _read_local(path: Path):
    txt = path.read_text(encoding="utf-8")
    parts = [p.strip() for p in txt.split("\n\n") if p.strip()]
    if not parts:
        parts = [p.strip() for p in txt.splitlines() if p.strip()]
    return parts

class LocalDataset:
    def __init__(self, rows):
        self._rows = rows
        self.column_names = list(rows[0].keys()) if rows else ["text"]
        self.features = None

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self._rows[idx]
        return self._rows[idx]

    def select(self, indices):
        return LocalDataset([self._rows[i] for i in indices])

if __name__ == "__main__":
    
    local_path = Path("TinyStories-train.txt")
    parts = _read_local(local_path)
    ds_rows = [{"text": p} for p in parts]
    ds = LocalDataset(ds_rows)	

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