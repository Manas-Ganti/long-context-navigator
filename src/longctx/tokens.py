"""Token counting.

The ceiling is only meaningful if every count in the system comes from ONE
counter. This module is that counter. The default is deterministic and
model-free (word / punctuation pieces), so the tests and the generator run
without downloading anything; "hf:<repo>" swaps in a real model tokenizer so
counts match what the policy is charged in its prompt.
"""

from __future__ import annotations

import re
from typing import Protocol

_PIECE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class Tokenizer(Protocol):
    name: str

    def count(self, text: str) -> int: ...

    def truncate(self, text: str, max_tokens: int) -> str: ...


class WhitespaceTokenizer:
    """Counts words and punctuation marks as one token each. Deterministic."""

    name = "whitespace"

    def count(self, text: str) -> int:
        return len(_PIECE.findall(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        matches = list(_PIECE.finditer(text))
        if len(matches) <= max_tokens:
            return text
        return text[: matches[max_tokens - 1].end()]


class HFTokenizer:
    """Counts with a Hugging Face tokenizer (no special tokens)."""

    def __init__(self, repo: str):
        from transformers import AutoTokenizer  # lazy: optional dependency

        self.name = f"hf:{repo}"
        self._tok = AutoTokenizer.from_pretrained(repo)

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False))

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        ids = self._tok.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text
        return self._tok.decode(ids[:max_tokens])


def get_tokenizer(spec: str) -> Tokenizer:
    if spec == "whitespace":
        return WhitespaceTokenizer()
    if spec.startswith("hf:"):
        return HFTokenizer(spec[3:])
    raise ValueError(f"unknown tokenizer spec {spec!r}; use 'whitespace' or 'hf:<repo>'")
