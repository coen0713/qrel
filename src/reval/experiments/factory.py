"""Config -> object construction.

Kept separate from both the config models and the runner so that a config can be
validated without importing torch, and so the runner reads as a pipeline rather
than a switch statement.
"""

from __future__ import annotations

from reval.chunking.base import BaseChunker
from reval.chunking.strategies import (
    FixedTokenChunker,
    ParentDocumentChunker,
    RecursiveCharacterChunker,
    SemanticChunker,
)
from reval.embedding.cache import CachedEncoder
from reval.embedding.encoder import KNOWN_ENCODERS, Encoder, EncoderConfig
from reval.experiments.config import ChunkerConfig, EncoderConfigModel, ExperimentConfig


def build_encoder(cfg: EncoderConfigModel) -> Encoder:
    """Resolve an encoder by short name or Hugging Face id."""
    if cfg.name in KNOWN_ENCODERS:
        base = KNOWN_ENCODERS[cfg.name]
        ec = EncoderConfig(
            model_name=base.model_name,
            revision=cfg.revision or base.revision,
            normalize=cfg.normalize,
            batch_size=cfg.batch_size,
            device=cfg.device,
            max_seq_length=cfg.max_seq_length,
            # Prefixes come from the registry, never from user config: they are
            # a property of the model, and getting them wrong is a silent
            # several-point recall loss that looks like a bad model.
            query_prefix=base.query_prefix,
            passage_prefix=base.passage_prefix,
        )
    else:
        ec = EncoderConfig(
            model_name=cfg.name,
            revision=cfg.revision,
            normalize=cfg.normalize,
            batch_size=cfg.batch_size,
            device=cfg.device,
            max_seq_length=cfg.max_seq_length,
        )
    return Encoder(config=ec)


def build_cached_encoder(cfg: EncoderConfigModel, use_cache: bool = True) -> CachedEncoder:
    return CachedEncoder(build_encoder(cfg), enabled=use_cache)


def build_chunker(cfg: ChunkerConfig, encoder=None) -> BaseChunker:
    """Construct a chunker. ``encoder`` is required only for ``semantic``."""
    if cfg.kind == "fixed":
        return FixedTokenChunker(
            chunk_tokens=cfg.chunk_tokens,
            overlap_tokens=cfg.overlap_tokens,
            tokenizer=cfg.tokenizer,
        )
    if cfg.kind == "recursive":
        return RecursiveCharacterChunker(
            chunk_chars=cfg.chunk_chars,
            overlap_chars=cfg.overlap_chars,
        )
    if cfg.kind == "semantic":
        if encoder is None:
            raise ValueError("the semantic chunker needs an encoder")
        return SemanticChunker(
            encoder=encoder,
            breakpoint_percentile=cfg.breakpoint_percentile,
            min_chunk_chars=cfg.min_chunk_chars,
            max_chunk_chars=cfg.max_chunk_chars,
        )
    if cfg.kind == "parent":
        child_cfg = cfg.model_copy(update={"kind": cfg.child_kind})
        return ParentDocumentChunker(
            child=build_chunker(child_cfg, encoder=encoder),
            parent_chars=cfg.parent_chars,
        )
    raise ValueError(f"unknown chunker kind: {cfg.kind!r}")


def build_retriever(config: ExperimentConfig, encoder=None):
    """Construct the retriever named by ``config.retriever.kind``."""
    rc = config.retriever
    if rc.kind == "bm25":
        from reval.indexing.bm25 import BM25Retriever

        return BM25Retriever(
            k1=rc.k1, b=rc.b, method=rc.method, stopwords=rc.stopwords, stemmer=rc.stemmer
        )
    if rc.kind == "dense":
        from reval.indexing.dense import DenseRetriever

        if encoder is None:
            encoder = build_cached_encoder(config.encoder)
        return DenseRetriever(
            encoder=encoder,
            collection=f"{config.dataset}-{config.chunker.kind}-{config.config_hash()}",
            mode=rc.qdrant_mode,
            url=rc.qdrant_url,
            name=f"dense:{config.encoder.name}",
        )
    raise ValueError(f"unknown retriever kind: {rc.kind!r}")
