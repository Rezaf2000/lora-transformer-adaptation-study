"""Download the two corpora used in the experiment.

Corpus A: the tiny Shakespeare dataset (Karpathy's char-rnn), English.
Corpus B: Don Quijote de la Mancha, Spanish, from Project Gutenberg (public
domain). The Gutenberg boilerplate header and footer (licensing text, not
part of the novel) are stripped so the model trains on prose, not legalese.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
QUIJOTE_URL = "https://www.gutenberg.org/cache/epub/2000/pg2000.txt"

QUIJOTE_START_MARKER = "*** START OF THE PROJECT GUTENBERG"
QUIJOTE_END_MARKER = "*** END OF THE PROJECT GUTENBERG"


def fetch(url: str) -> str:
    with urllib.request.urlopen(url) as response:
        return response.read().decode("utf-8")


def strip_gutenberg_boilerplate(text: str) -> str:
    start = text.find(QUIJOTE_START_MARKER)
    if start != -1:
        start = text.find("\n", start) + 1
        text = text[start:]
    end = text.find(QUIJOTE_END_MARKER)
    if end != -1:
        text = text[:end]
    return text.strip()


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    shakespeare_path = DATA_DIR / "shakespeare.txt"
    print(f"downloading corpus A (Shakespeare) -> {shakespeare_path}")
    shakespeare_text = fetch(SHAKESPEARE_URL)
    shakespeare_path.write_text(shakespeare_text, encoding="utf-8")
    print(f"  {len(shakespeare_text):,} characters")

    quijote_path = DATA_DIR / "quijote.txt"
    print(f"downloading corpus B (Don Quijote) -> {quijote_path}")
    quijote_raw = fetch(QUIJOTE_URL)
    quijote_text = strip_gutenberg_boilerplate(quijote_raw)
    quijote_path.write_text(quijote_text, encoding="utf-8")
    print(f"  {len(quijote_text):,} characters (from {len(quijote_raw):,} raw)")


if __name__ == "__main__":
    main()
