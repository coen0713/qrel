"""Chunking strategies. Deterministic and fingerprintable, because chunk
boundaries feed the embedding cache key and the run manifest."""

from __future__ import annotations

from reval.chunking.base import BaseChunker, chunk_to_doc_map
from reval.chunking.strategies import (
    FixedTokenChunker,
    ParentDocumentChunker,
    RecursiveCharacterChunker,
    SemanticChunker,
)
from reval.chunking.tokenize import WordTokenizer, get_tokenizer, sentence_spans

__all__ = [
    "BaseChunker",
    "FixedTokenChunker",
    "ParentDocumentChunker",
    "RecursiveCharacterChunker",
    "SemanticChunker",
    "WordTokenizer",
    "chunk_to_doc_map",
    "get_tokenizer",
    "sentence_spans",
]
