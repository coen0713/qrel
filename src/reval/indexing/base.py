"""The ``Retriever`` protocol.

One interface, two implementations (BM25 sparse, Qdrant dense). Everything
downstream — scoring, contamination analysis, the experiment grid — is written
against this and never against a concrete backend, which is what lets the
BM25-vs-dense comparison be an apples-to-apples one: identical chunks, identical
aggregation, identical metrics, identical tie-breaking.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from reval.types import Chunk, Queries, Run


@runtime_checkable
class Retriever(Protocol):
    """Indexes chunks, then answers queries with scored chunk ids."""

    #: Short identifier for run tags and result tables, e.g. ``bm25`` or
    #: ``dense:bge-small``.
    name: str

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Build the index. Must be idempotent for a given chunk list."""
        ...

    def search(self, queries: Queries, top_k: int) -> Run:
        """Return ``{query_id: {chunk_id: score}}``, at most ``top_k`` per query.

        Chunk ids, not document ids: collapsing to documents is an explicit
        later step with a named policy (:mod:`reval.metrics.aggregation`), and a
        retriever that did it internally would hide that decision.
        """
        ...


class IndexNotBuiltError(RuntimeError):
    """Raised when ``search`` is called before ``index``."""
