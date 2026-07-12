"""Character-level tokenizer and batching for the two corpora used in the
pretrain -> adapt experiment.

Corpus A (pretraining) is the tiny Shakespeare dataset, in English. Corpus B
(adaptation) is Don Quijote, in Spanish, which has a visibly different
character set (accented vowels, inverted punctuation) and a different
style. The vocabulary is built from the *union* of both texts up front, so
a single MiniGPT can be pretrained on A and later adapted to B without
resizing its embedding table or output head: characters that only occur in
Quijote (n, accented letters, ¿, ¡) are present in the vocabulary from the
start, with embeddings that are essentially untrained until the model sees
them during the adaptation stage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class CharTokenizer:
    chars: list[str]

    def __post_init__(self):
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"chars": self.chars}))

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        return cls(chars=json.loads(Path(path).read_text())["chars"])

    @classmethod
    def build(cls, *texts: str) -> "CharTokenizer":
        seen: set[str] = set()
        for text in texts:
            seen.update(text)
        return cls(chars=sorted(seen))


def load_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def encode_split(text: str, tokenizer: CharTokenizer, val_fraction: float = 0.1):
    """Encode `text` and split it into contiguous train/val tensors.

    Contiguous (not random) so validation always covers a held-out stretch
    of the actual corpus, and so the split is deterministic and reused
    identically across every method compared in the experiment.
    """
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n_val = max(1, int(len(data) * val_fraction))
    return data[:-n_val], data[-n_val:]


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, generator: torch.Generator | None = None):
    """Sample `batch_size` random windows of length `block_size` from `data`,
    each paired with the same window shifted one character to the right --
    the next-character prediction target."""
    max_start = len(data) - block_size - 1
    starts = torch.randint(0, max_start, (batch_size,), generator=generator)
    x = torch.stack([data[s : s + block_size] for s in starts])
    y = torch.stack([data[s + 1 : s + block_size + 1] for s in starts])
    return x, y
