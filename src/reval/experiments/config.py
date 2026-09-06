"""Experiment configuration: Pydantic models over YAML.

Configs are hashable and serialisable, and the hash is what a run manifest
records. Two runs with the same config hash claim to be the same experiment; if
they produce different numbers, something outside the config is leaking in, and
that is a bug worth finding.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    """Reject unknown fields.

    A typo'd key that silently does nothing is the worst kind of config bug: the
    run succeeds, the setting you thought you changed did not change, and the
    number you get is for a different experiment than the one you think you ran.
    """

    model_config = ConfigDict(extra="forbid")


class ChunkerConfig(StrictModel):
    kind: Literal["fixed", "recursive", "semantic", "parent"] = "fixed"

    # fixed
    chunk_tokens: int = 256
    overlap_tokens: int = 32
    tokenizer: str = "word"

    # recursive
    chunk_chars: int = 1200
    overlap_chars: int = 150

    # semantic
    breakpoint_percentile: float = 25.0
    min_chunk_chars: int = 200
    max_chunk_chars: int = 2000

    # parent
    parent_chars: int = 3000
    child_kind: Literal["fixed", "recursive"] = "fixed"

    @field_validator("overlap_tokens")
    @classmethod
    def _overlap_fits(cls, v: int, info) -> int:
        size = info.data.get("chunk_tokens", 256)
        if v >= size:
            raise ValueError(f"overlap_tokens ({v}) must be < chunk_tokens ({size})")
        return v


class EncoderConfigModel(StrictModel):
    name: str = "minilm"
    revision: str | None = "main"
    normalize: bool = True
    batch_size: int = 64
    device: str | None = None
    max_seq_length: int | None = None


class RetrieverConfig(StrictModel):
    kind: Literal["bm25", "dense"] = "bm25"

    # bm25 — Anserini's BEIR defaults, not bm25s's library defaults.
    k1: float = 0.9
    b: float = 0.4
    method: str = "lucene"
    stopwords: str = "en"
    stemmer: str | None = "english"

    # dense
    qdrant_mode: Literal["embedded", "server"] = "embedded"
    qdrant_url: str = "http://localhost:6333"


class EvalConfig(StrictModel):
    ks: list[int] = Field(default_factory=lambda: [1, 5, 10, 20, 100])
    #: Retrieval depth. Must be >= max(ks) or the deepest cutoff is truncated
    #: by the retriever rather than by the metric.
    top_k: int = 100
    aggregation: Literal["max", "sum", "mean", "first"] = "max"
    tie_break: Literal["trec", "docid_asc"] = "trec"
    rel_threshold: int = 1

    @field_validator("top_k")
    @classmethod
    def _deep_enough(cls, v: int, info) -> int:
        ks = info.data.get("ks") or [10]
        if v < max(ks):
            raise ValueError(
                f"top_k ({v}) is shallower than the largest cutoff ({max(ks)}); "
                "the metric would be measuring the retrieval depth, not the retriever"
            )
        return v


class BootstrapConfig(StrictModel):
    n_resamples: int = 10_000
    confidence: float = 0.95


class ExperimentConfig(StrictModel):
    """One experiment: a dataset, a chunker, a retriever, and how to score it."""

    name: str = "baseline"
    dataset: str = "scifact"
    split: str | None = None
    #: Seeds everything downstream: chunker boundaries, bootstrap resampling,
    #: negative sampling, synthetic query sampling. PLAN.md §5 rule 3.
    seed: int = 0

    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    encoder: EncoderConfigModel = Field(default_factory=EncoderConfigModel)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)

    #: Optional path to a synthetic query set (M3). When set, the run is scored
    #: against these queries instead of the dataset's human-authored ones.
    synthetic_queries: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(self.model_dump(mode="json"), sort_keys=True), encoding="utf-8"
        )
        return path

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def config_hash(self) -> str:
        """Stable 16-hex digest of the whole config."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()[:16]

    def run_tag(self) -> str:
        """Filesystem- and TREC-safe run name."""
        r = self.retriever.kind
        if r == "dense":
            r = f"dense-{self.encoder.name}"
        return f"{self.dataset}.{self.chunker.kind}.{r}.{self.config_hash()}"


def load_configs(paths: list[str | Path]) -> list[ExperimentConfig]:
    return [ExperimentConfig.from_yaml(p) for p in paths]
