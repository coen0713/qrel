"""reval — a retrieval evaluation harness with contamination detection."""

from __future__ import annotations

__version__ = "0.1.0"

from reval.types import (
    Chunk,
    Corpus,
    Dataset,
    DocId,
    Document,
    Qrels,
    Queries,
    QueryId,
    Run,
    ScoredDoc,
)

__all__ = [
    "Chunk",
    "Corpus",
    "Dataset",
    "DocId",
    "Document",
    "Qrels",
    "Queries",
    "QueryId",
    "Run",
    "ScoredDoc",
    "__version__",
]
