"""The experiment runner.

One pipeline, run identically for every configuration:

    load corpus -> chunk -> index -> search -> aggregate chunks to documents
                -> score -> bootstrap -> write run file + manifest + parquet

Each stage is a plain function so the pipeline can be entered at any point (the
contamination experiment in M3 reuses `search` onwards with a different query
set), and so nothing is only reachable through the CLI.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from reval.chunking.base import chunk_to_doc_map
from reval.corpora import beir
from reval.experiments.config import ExperimentConfig
from reval.experiments.factory import build_cached_encoder, build_chunker, build_retriever
from reval.experiments.manifest import Manifest
from reval.metrics import aggregate_to_documents, estimate_all, evaluate, mean_scores
from reval.metrics.aggregation import deduplicate_by_document
from reval.paths import results_dir, runs_dir
from reval.runfile import write_run
from reval.types import Chunk, Dataset, Queries, Run


@dataclass(slots=True)
class RunResult:
    """Everything one experiment produced."""

    config: ExperimentConfig
    dataset_name: str
    run_tag: str
    #: Document-level run, after chunk aggregation.
    doc_run: Run
    #: Raw chunk-level run, kept for the chunk-overlap analysis (M3c).
    chunk_run: Run
    per_query: dict[str, dict[str, float]]
    means: dict[str, float]
    estimates: dict[str, object]
    #: recall@k computed after chunk->document deduplication, for the
    #: double-counting measurement. Same numbers unless chunks overlap.
    dedup_per_query: dict[str, dict[str, float]] = field(default_factory=dict)
    n_chunks: int = 0
    manifest: Manifest | None = None
    run_path: Path | None = None
    timings: dict[str, float] = field(default_factory=dict)


class Timer:
    """Wall-clock timings per stage, for the manifest."""

    def __init__(self) -> None:
        self.spans: dict[str, float] = {}
        self._start: float | None = None
        self._label: str | None = None

    def __call__(self, label: str) -> Timer:
        self._label = label
        return self

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        if self._label is not None and self._start is not None:
            self.spans[self._label] = round(time.perf_counter() - self._start, 3)
        self._label, self._start = None, None


def load_dataset(config: ExperimentConfig) -> Dataset:
    return beir.load(config.dataset, split=config.split)


def build_chunks(config: ExperimentConfig, dataset: Dataset, encoder=None) -> list[Chunk]:
    chunker = build_chunker(config.chunker, encoder=encoder)
    return chunker.chunk_corpus(dataset.corpus)


def run_experiment(
    config: ExperimentConfig,
    dataset: Dataset | None = None,
    queries: Queries | None = None,
    qrels: dict | None = None,
    write_outputs: bool = True,
    use_cache: bool = True,
) -> RunResult:
    """Execute one experiment end to end.

    Args:
        config: the experiment.
        dataset: preloaded dataset, to avoid re-reading the corpus across a grid.
        queries: override the query set (the M3 synthetic eval sets use this).
        qrels: override the judgments; must be given whenever ``queries`` is.
        write_outputs: write run file, manifest and parquet.
        use_cache: disable to measure cold-cache cost.
    """
    timer = Timer()

    with timer("load"):
        ds = dataset if dataset is not None else load_dataset(config)

    eval_queries = queries if queries is not None else ds.queries
    eval_qrels = qrels if qrels is not None else ds.qrels
    if (queries is None) != (qrels is None):
        raise ValueError("queries and qrels must be overridden together, or not at all")

    # An encoder is needed for dense retrieval and for the semantic chunker;
    # constructing it lazily keeps BM25-only runs free of torch entirely.
    needs_encoder = config.retriever.kind == "dense" or config.chunker.kind == "semantic"
    encoder = build_cached_encoder(config.encoder, use_cache=use_cache) if needs_encoder else None

    with timer("chunk"):
        chunks = build_chunks(config, ds, encoder=encoder)
    chunk_map = chunk_to_doc_map(chunks)

    retriever = build_retriever(config, encoder=encoder)
    with timer("index"):
        retriever.index(chunks)

    with timer("search"):
        chunk_run = retriever.search(eval_queries, top_k=config.eval.top_k)

    with timer("score"):
        doc_run = aggregate_to_documents(chunk_run, chunk_map, policy=config.eval.aggregation)
        per_query = evaluate(
            doc_run,
            eval_qrels,
            ks=config.eval.ks,
            tie_break=config.eval.tie_break,
            rel_threshold=config.eval.rel_threshold,
        )
        means = mean_scores(per_query)

        # Chunk-overlap double counting (M3c): score again after collapsing the
        # chunk run to one chunk per document. With non-overlapping chunkers
        # this is identical, which is itself the control.
        dedup_run = aggregate_to_documents(
            deduplicate_by_document(chunk_run, chunk_map), chunk_map, policy=config.eval.aggregation
        )
        dedup_per_query = evaluate(
            dedup_run,
            eval_qrels,
            ks=config.eval.ks,
            tie_break=config.eval.tie_break,
            rel_threshold=config.eval.rel_threshold,
        )

    with timer("bootstrap"):
        estimates = estimate_all(
            per_query,
            n_resamples=config.bootstrap.n_resamples,
            confidence=config.bootstrap.confidence,
            seed=config.seed,
        )

    cache_stats = {}
    if encoder is not None and getattr(encoder, "last_stats", None) is not None:
        st = encoder.last_stats
        cache_stats = {"requested": st.requested, "hits": st.hits, "misses": st.misses}

    result = RunResult(
        config=config,
        dataset_name=ds.name,
        run_tag=config.run_tag(),
        doc_run=doc_run,
        chunk_run=chunk_run,
        per_query=per_query,
        means=means,
        estimates=estimates,
        dedup_per_query=dedup_per_query,
        n_chunks=len(chunks),
        timings=timer.spans,
    )

    if write_outputs:
        result.run_path = write_run(
            doc_run,
            runs_dir() / f"{config.run_tag()}.trec",
            run_name=config.run_tag(),
            k=config.eval.top_k,
            tie_break=config.eval.tie_break,
        )
        manifest = Manifest.build(
            config,
            ds,
            encoder=(
                {"model": encoder.config.model_name, "revision": encoder.resolved_revision}
                if encoder is not None
                else {}
            ),
            chunker=config.chunker.model_dump(mode="json"),
            retriever=config.retriever.model_dump(mode="json"),
            counts={
                "documents": ds.n_docs,
                "chunks": len(chunks),
                "queries": len(eval_queries),
            },
            cache=cache_stats,
            timing_seconds=timer.spans,
        )
        manifest.write(results_dir() / "manifests" / f"{config.run_tag()}.json")
        result.manifest = manifest

    if hasattr(retriever, "close"):
        retriever.close()
    if encoder is not None:
        encoder.close()

    return result
