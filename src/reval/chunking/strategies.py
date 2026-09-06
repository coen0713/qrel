"""The four chunking strategies from PLAN.md M2.

Fixed-token, recursive-character, semantic, and parent-document. They exist to
be *compared*, and the comparison is one of the things the contamination
experiment re-runs under a synthetic eval set to see whether the ranking of
strategies survives.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from reval.chunking.base import BaseChunker
from reval.chunking.tokenize import get_tokenizer, sentence_spans
from reval.types import Chunk, Document


class FixedTokenChunker(BaseChunker):
    """Fixed-size token windows with a fixed stride.

    The simplest strategy and the one most RAG tutorials use, which makes it the
    right baseline: if a more sophisticated chunker cannot beat this, it is not
    earning its complexity.
    """

    name = "fixed"

    def __init__(self, chunk_tokens: int = 256, overlap_tokens: int = 32, tokenizer: str = "word"):
        if overlap_tokens >= chunk_tokens:
            # stride = chunk - overlap would be <= 0: the window never advances.
            raise ValueError(
                f"overlap_tokens ({overlap_tokens}) must be < chunk_tokens ({chunk_tokens})"
            )
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer_spec = tokenizer
        self._tokenizer = get_tokenizer(tokenizer)

    def config(self) -> dict[str, object]:
        return {
            "chunk_tokens": self.chunk_tokens,
            "overlap_tokens": self.overlap_tokens,
            "tokenizer": self.tokenizer_spec,
        }

    def split(self, text: str) -> list[tuple[str, int, int]]:
        spans = self._tokenizer.spans(text)
        if not spans:
            return []

        stride = self.chunk_tokens - self.overlap_tokens
        out: list[tuple[str, int, int]] = []
        for start_tok in range(0, len(spans), stride):
            window = spans[start_tok : start_tok + self.chunk_tokens]
            if not window:
                break
            start, end = window[0][0], window[-1][1]
            out.append((text[start:end], start, end))
            # Stop once the window has consumed the tail, rather than emitting
            # ever-shorter trailing chunks that are all suffixes of each other.
            if start_tok + self.chunk_tokens >= len(spans):
                break
        return out


class RecursiveCharacterChunker(BaseChunker):
    """Split on the largest natural boundary that fits, recursing downward.

    Tries paragraph breaks, then line breaks, then sentence ends, then spaces,
    then raw characters. This is the LangChain-style default, included because
    it is what a large share of production RAG systems actually run, so a
    benchmark without it is not measuring the thing people deploy.
    """

    name = "recursive"

    DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", " ", "")

    def __init__(
        self,
        chunk_chars: int = 1200,
        overlap_chars: int = 150,
        separators: Sequence[str] = DEFAULT_SEPARATORS,
    ):
        if overlap_chars >= chunk_chars:
            raise ValueError(
                f"overlap_chars ({overlap_chars}) must be < chunk_chars ({chunk_chars})"
            )
        if not separators:
            raise ValueError("separators must not be empty")
        self.chunk_chars = chunk_chars
        self.overlap_chars = overlap_chars
        self.separators = tuple(separators)

    def config(self) -> dict[str, object]:
        return {
            "chunk_chars": self.chunk_chars,
            "overlap_chars": self.overlap_chars,
            "separators": list(self.separators),
        }

    def split(self, text: str) -> list[tuple[str, int, int]]:
        if not text:
            return []
        pieces = self._recurse(text, 0, list(self.separators))
        return self._merge(pieces, text)

    def _recurse(self, text: str, offset: int, separators: list[str]) -> list[tuple[int, int]]:
        """Break ``text`` into spans no longer than ``chunk_chars`` where possible."""
        if len(text) <= self.chunk_chars:
            return [(offset, offset + len(text))]
        if not separators:
            # Hard character split: nothing left to respect.
            return [
                (offset + i, offset + min(i + self.chunk_chars, len(text)))
                for i in range(0, len(text), self.chunk_chars)
            ]

        sep, rest = separators[0], separators[1:]
        if sep == "":
            return [
                (offset + i, offset + min(i + self.chunk_chars, len(text)))
                for i in range(0, len(text), self.chunk_chars)
            ]

        parts: list[tuple[int, int]] = []
        cursor = 0
        for piece in text.split(sep):
            start = cursor
            end = cursor + len(piece)
            if end > start:
                if end - start > self.chunk_chars:
                    parts.extend(self._recurse(text[start:end], offset + start, rest))
                else:
                    parts.append((offset + start, offset + end))
            cursor = end + len(sep)
        return parts

    def _merge(self, spans: list[tuple[int, int]], text: str) -> list[tuple[str, int, int]]:
        """Greedily pack adjacent spans up to ``chunk_chars``, then add overlap.

        Without this, splitting on "\\n\\n" would emit one chunk per paragraph,
        including one-line paragraphs — chunks far below the target size, which
        would make the size parameter meaningless.
        """
        if not spans:
            return []

        packed: list[tuple[int, int]] = []
        cur_start, cur_end = spans[0]
        for start, end in spans[1:]:
            if end - cur_start <= self.chunk_chars:
                cur_end = end
            else:
                packed.append((cur_start, cur_end))
                cur_start, cur_end = start, end
        packed.append((cur_start, cur_end))

        out: list[tuple[str, int, int]] = []
        for i, (start, end) in enumerate(packed):
            # Overlap is applied by extending backwards, so chunk i shares its
            # first `overlap_chars` with the tail of chunk i-1.
            s = max(0, start - self.overlap_chars) if i > 0 else start
            out.append((text[s:end], s, end))
        return out


class SemanticChunker(BaseChunker):
    """Place boundaries where consecutive sentences stop being similar.

    Embeds each sentence, takes the cosine similarity of each adjacent pair, and
    cuts wherever that similarity falls into the bottom ``breakpoint_percentile``
    of the document's own distribution. A per-document percentile rather than a
    global threshold, because absolute cosine values are not comparable across
    documents of different length and register — a fixed 0.7 would shred one
    document and never cut another.

    Needs an encoder, so this chunker is only constructible with the ``[dense]``
    extra. It is also the one place where chunk boundaries depend on a model,
    which is worth remembering when reading a chunker comparison: swapping the
    encoder changes this chunker's output and nothing else's.
    """

    name = "semantic"

    def __init__(
        self,
        encoder,
        breakpoint_percentile: float = 25.0,
        min_chunk_chars: int = 200,
        max_chunk_chars: int = 2000,
    ):
        if not 0 < breakpoint_percentile < 100:
            raise ValueError("breakpoint_percentile must be in (0, 100)")
        self.encoder = encoder
        self.breakpoint_percentile = breakpoint_percentile
        self.min_chunk_chars = min_chunk_chars
        self.max_chunk_chars = max_chunk_chars

    def config(self) -> dict[str, object]:
        return {
            "breakpoint_percentile": self.breakpoint_percentile,
            "min_chunk_chars": self.min_chunk_chars,
            "max_chunk_chars": self.max_chunk_chars,
            # The encoder is part of this chunker's identity, unlike the others.
            "encoder": getattr(self.encoder, "identity", str(self.encoder)),
        }

    def split(self, text: str) -> list[tuple[str, int, int]]:
        spans = sentence_spans(text)
        if len(spans) <= 1:
            return [(text, 0, len(text))] if text else []

        sentences = [text[s:e] for s, e in spans]
        vectors = self.encoder.encode(sentences)
        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit = vectors / norms
        similarities = np.sum(unit[:-1] * unit[1:], axis=1)

        cut_at = float(np.percentile(similarities, self.breakpoint_percentile))

        out: list[tuple[str, int, int]] = []
        start = spans[0][0]
        for i, (_, end) in enumerate(spans[:-1]):
            too_long = end - start >= self.max_chunk_chars
            long_enough = end - start >= self.min_chunk_chars
            if too_long or (similarities[i] <= cut_at and long_enough):
                out.append((text[start:end], start, end))
                start = spans[i + 1][0]

        final_end = spans[-1][1]
        if final_end > start:
            out.append((text[start:final_end], start, final_end))
        return out


class ParentDocumentChunker(BaseChunker):
    """Retrieve on small child chunks, return the larger parent window.

    The idea: small chunks match queries precisely, large chunks give a
    generator enough context. Retrieval scores the child; the caller reads
    ``parent_text``.

    Worth being precise about what this does and does not change *for this
    harness*: because our metrics are computed after chunk→document
    aggregation, and the parent is a window of the same document, parent-document
    retrieval scores identically to its child chunker at the document level. It
    is included because it is a strategy people deploy and because
    ``parent_text`` is what a downstream RAG system would actually consume — but
    if it shows a document-level difference from its child chunker, that is a
    bug in the harness, not a finding. ``test_parent_matches_child_at_document_level``
    pins that.
    """

    name = "parent"

    def __init__(self, child: BaseChunker, parent_chars: int = 3000):
        self.child = child
        self.parent_chars = parent_chars

    def config(self) -> dict[str, object]:
        return {
            "parent_chars": self.parent_chars,
            "child": {"name": self.child.name, **self.child.config()},
        }

    def split(self, text: str) -> list[tuple[str, int, int]]:
        return self.child.split(text)

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.full_text
        children = self.child.chunk(doc)
        out: list[Chunk] = []
        for c in children:
            # Centre a parent window on the child span.
            centre = (c.char_start + c.char_end) // 2
            half = self.parent_chars // 2
            p_start = max(0, centre - half)
            p_end = min(len(text), p_start + self.parent_chars)
            p_start = max(0, p_end - self.parent_chars)
            out.append(
                Chunk(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    ordinal=c.ordinal,
                    text=c.text,
                    char_start=c.char_start,
                    char_end=c.char_end,
                    parent_text=text[p_start:p_end],
                )
            )
        return out
