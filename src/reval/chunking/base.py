"""The ``Chunker`` interface.

Every chunker is a deterministic, fingerprintable function from a document to a
list of chunks. Both properties are load-bearing:

- **Deterministic**, because chunk boundaries feed the embedding cache key. A
  chunker that moved a boundary by one character between runs would invalidate
  the cache and, worse, silently change what the retriever was scored on.
- **Fingerprintable**, because a run manifest has to record *which* chunker
  produced the index, at what settings. "Fixed-token chunking" is not a
  reproducible description; ``fixed(size=256,overlap=32,tok=word)`` is.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Iterable

from reval.types import Chunk, Corpus, Document


class BaseChunker(ABC):
    """Base class for chunkers.

    Subclasses implement :meth:`split`, which returns ``(text, start, end)``
    spans over a document's ``full_text``. The base class turns those into
    :class:`~reval.types.Chunk` objects with deterministic ids, so no subclass
    has to get id construction right.
    """

    #: Short stable name used in config files and run tags.
    name: str = "base"

    @abstractmethod
    def config(self) -> dict[str, object]:
        """The chunker's settings, as a JSON-serialisable dict.

        Everything that affects the output must appear here. Anything omitted
        is invisible to the fingerprint, and therefore to the cache and the
        manifest.
        """

    @abstractmethod
    def split(self, text: str) -> list[tuple[str, int, int]]:
        """Split ``text`` into ``(chunk_text, char_start, char_end)`` spans."""

    def fingerprint(self) -> str:
        """Stable 16-hex-char digest of name plus config."""
        payload = json.dumps(
            {"name": self.name, **self.config()}, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def describe(self) -> str:
        """Human-readable one-liner for tables and run tags."""
        params = ",".join(f"{k}={v}" for k, v in sorted(self.config().items()))
        return f"{self.name}({params})"

    def chunk(self, doc: Document) -> list[Chunk]:
        """Chunk one document."""
        text = doc.full_text
        chunks: list[Chunk] = []
        for ordinal, (chunk_text, start, end) in enumerate(self.split(text)):
            if not chunk_text.strip():
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{ordinal}",
                    doc_id=doc.doc_id,
                    ordinal=ordinal,
                    text=chunk_text,
                    char_start=start,
                    char_end=end,
                )
            )
        if not chunks:
            # An empty document must still be indexable, or the corpus the
            # retriever sees silently differs from the corpus the qrels judge.
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#0",
                    doc_id=doc.doc_id,
                    ordinal=0,
                    text=text,
                    char_start=0,
                    char_end=len(text),
                )
            )
        return chunks

    def chunk_corpus(self, corpus: Corpus) -> list[Chunk]:
        """Chunk a whole corpus, in sorted document order.

        Sorted, not dict order: the resulting list feeds index construction and
        an embedding cache, and a stable order makes both diffable between runs.
        """
        out: list[Chunk] = []
        for doc_id in sorted(corpus):
            out.extend(self.chunk(corpus[doc_id]))
        return out


def chunk_to_doc_map(chunks: Iterable[Chunk]) -> dict[str, str]:
    """Build the ``chunk_id -> doc_id`` mapping the aggregation step needs."""
    return {c.chunk_id: c.doc_id for c in chunks}
