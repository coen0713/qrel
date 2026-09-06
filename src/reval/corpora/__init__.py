"""Corpus loaders. BEIR in, canonical ``Dataset`` out."""

from __future__ import annotations

from reval.corpora.beir import (
    CorpusStats,
    dataset_names,
    dataset_path,
    download,
    file_sha256,
    is_downloaded,
    load,
    spec,
    stats,
)

__all__ = [
    "CorpusStats",
    "dataset_names",
    "dataset_path",
    "download",
    "file_sha256",
    "is_downloaded",
    "load",
    "spec",
    "stats",
]
