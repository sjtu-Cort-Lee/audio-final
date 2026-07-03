from __future__ import annotations

import json
import re
from pathlib import Path


_KEEP_RE = re.compile(r"[^a-z' ]+")
_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = text.lower()
    text = _KEEP_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def build_ctc_vocab(texts: list[str], vocab_path: str | Path) -> dict[str, int]:
    chars: set[str] = set()
    for text in texts:
        chars.update(normalize_text(text))

    chars.discard(" ")
    vocab_tokens = sorted(chars)
    vocab = {token: idx for idx, token in enumerate(vocab_tokens)}
    vocab["|"] = len(vocab)
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)

    path = Path(vocab_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2, sort_keys=True)
    return vocab


def text_to_ctc_text(text: str) -> str:
    return normalize_text(text).replace(" ", "|")
