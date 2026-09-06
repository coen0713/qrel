"""Retrieval backends behind one Retriever protocol."""

from __future__ import annotations

from reval.indexing.base import IndexNotBuiltError, Retriever
from reval.indexing.bm25 import BM25Retriever

__all__ = ["BM25Retriever", "IndexNotBuiltError", "Retriever"]


def __getattr__(name: str):
    # DenseRetriever pulls in torch via the encoder, so importing this package
    # must not cost that unless dense retrieval is actually used.
    if name == "DenseRetriever":
        from reval.indexing.dense import DenseRetriever

        return DenseRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
