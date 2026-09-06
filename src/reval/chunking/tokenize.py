"""Tokenisation for the fixed-token chunker.

"256 tokens" is not a well-defined chunk size until you say whose tokens. The
two honest options:

- **word** (default): a regex word tokenizer. No dependencies, deterministic,
  identical on every machine, and — importantly — *independent of the encoder*,
  so a BM25 run and a dense run can be scored on literally the same chunks.
  A chunker whose boundaries moved when you swapped encoders would confound
  "which encoder is better" with "which chunking did each encoder get".
- **hf**: the encoder's own subword tokenizer, which is what actually
  determines whether a chunk fits in the model's context window.

We default to `word` and record the choice in the fingerprint. English text runs
roughly 1.3 subword tokens per word for BERT-family vocabularies, so a
256-*word* chunk is ~330 subword tokens — comfortably inside the 512-token limit
of the encoders we use, which is the constraint that actually matters.
"""

from __future__ import annotations

import re
from typing import Protocol

#: Words, numbers, and standalone punctuation. Keeps offsets exact.
_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class Tokenizer(Protocol):
    """Anything that can split text into tokens with character offsets."""

    name: str

    def spans(self, text: str) -> list[tuple[int, int]]:
        """Return ``(char_start, char_end)`` for each token, in order."""
        ...


class WordTokenizer:
    """Regex word tokenizer. Deterministic and dependency-free."""

    name = "word"

    def spans(self, text: str) -> list[tuple[int, int]]:
        return [m.span() for m in _WORD_RE.finditer(text)]

    def count(self, text: str) -> int:
        return len(self.spans(text))


class HuggingFaceTokenizer:
    """The encoder's own subword tokenizer, via ``transformers``.

    Only available with the ``[dense]`` extra. Uses a fast tokenizer so that
    character offsets are exact rather than reconstructed.
    """

    def __init__(self, model_name: str):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ImportError(
                "the 'hf' tokenizer needs the [dense] extra: uv pip install -e '.[dense]'"
            ) from exc

        self._tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if not self._tok.is_fast:  # pragma: no cover - model-specific
            raise ValueError(
                f"{model_name} has no fast tokenizer, so character offsets would be "
                "approximate; chunk boundaries must be exact"
            )
        self.name = f"hf:{model_name}"

    def spans(self, text: str) -> list[tuple[int, int]]:
        enc = self._tok(text, add_special_tokens=False, return_offsets_mapping=True)
        # Zero-width offsets are special/control tokens; they would create empty
        # chunk boundaries.
        return [(s, e) for s, e in enc["offset_mapping"] if e > s]


def get_tokenizer(spec: str) -> Tokenizer:
    """Resolve a tokenizer from a config string: ``word`` or ``hf:<model>``."""
    if spec == "word":
        return WordTokenizer()
    if spec.startswith("hf:"):
        return HuggingFaceTokenizer(spec[3:])
    raise ValueError(f"unknown tokenizer {spec!r}; expected 'word' or 'hf:<model-name>'")


#: Sentence boundaries: terminal punctuation followed by whitespace and a
#: capital or digit. Deliberately simple and dependency-free — it is used only
#: by the semantic chunker, where a missed boundary means a slightly longer
#: chunk rather than a wrong one.
_SENTENCE_RE = re.compile(r"(?<=[.!?])[\s\n]+(?=[A-Z0-9\"'(\[])")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Split into sentence ``(start, end)`` spans covering the whole string."""
    if not text.strip():
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _SENTENCE_RE.finditer(text):
        end = m.start()
        if end > start:
            spans.append((start, end))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans
