"""Chunk → document aggregation.

You retrieve chunks. Qrels judge documents. Something has to collapse one into
the other, and PLAN.md M1 is emphatic that it must never be implicit: an
aggregation buried inside a retriever loop is a policy decision nobody can find,
review, or vary.

So it lives here, as a named function with a named policy, applied as an
explicit step in the pipeline.

The default is **max**: a document's score is the score of its best-matching
chunk. That is the standard choice, and it is the one that treats a document as
relevant if *any* part of it answers the query — which is what a document-level
relevance judgment means.

The alternatives are implemented because their differences are measurable, and
because ``sum`` in particular is a trap: it rewards long documents that split
into many chunks, so a chunk-size comparison scored under ``sum`` would conflate
"better chunking" with "more chunks". That is a version of the same
measurement bug this project is about.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from reval.types import ChunkId, DocId, Run

AggregationPolicy = Literal["max", "sum", "mean", "first"]


def aggregate_to_documents(
    chunk_run: Run,
    chunk_to_doc: Mapping[ChunkId, DocId],
    policy: AggregationPolicy = "max",
) -> Run:
    """Collapse a chunk-level run into a document-level run.

    Args:
        chunk_run: ``{query_id: {chunk_id: score}}`` as returned by a retriever.
        chunk_to_doc: chunk id -> owning document id.
        policy:
            ``max``   document score = best chunk score (default, standard).
            ``sum``   sum of chunk scores; biases toward documents with more
                      chunks, so never use it to compare chunking strategies.
            ``mean``  mean of retrieved chunk scores; note this is the mean over
                      *retrieved* chunks only, so a document with one strong
                      chunk in the top-k beats one with a strong chunk and a
                      weak one.
            ``first`` score of the highest-ranked chunk. Identical to ``max``
                      for any retriever whose scores decrease with rank, which
                      is all of ours; kept as a distinct name because that
                      equivalence is an assumption, not a guarantee.

    Returns:
        ``{query_id: {doc_id: score}}``, ready for :func:`reval.metrics.evaluate`.

    Raises:
        KeyError: if a retrieved chunk has no document mapping. Silently
            dropping it would quietly lower recall with no visible cause.
    """
    out: Run = {}
    for qid, chunk_scores in chunk_run.items():
        doc_scores: dict[DocId, float] = {}
        counts: dict[DocId, int] = {}

        # Rank order matters for `first`; for the others it is irrelevant, and
        # sorting once is cheaper than branching per policy.
        ordered = sorted(chunk_scores.items(), key=lambda kv: kv[1], reverse=True)

        for chunk_id, score in ordered:
            try:
                doc_id = chunk_to_doc[chunk_id]
            except KeyError:
                raise KeyError(
                    f"chunk {chunk_id!r} (query {qid!r}) has no document mapping; "
                    "the index and the chunk table are out of sync"
                ) from None

            if policy == "max":
                if score > doc_scores.get(doc_id, float("-inf")):
                    doc_scores[doc_id] = score
            elif policy == "first":
                doc_scores.setdefault(doc_id, score)
            elif policy == "sum":
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score
            elif policy == "mean":
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score
                counts[doc_id] = counts.get(doc_id, 0) + 1
            else:  # pragma: no cover - guarded by the Literal type
                raise ValueError(f"unknown aggregation policy: {policy!r}")

        if policy == "mean":
            doc_scores = {d: s / counts[d] for d, s in doc_scores.items()}

        out[qid] = doc_scores
    return out


def deduplicate_by_document(run: Run, chunk_to_doc: Mapping[ChunkId, DocId]) -> Run:
    """Keep only the best-scoring chunk per document, still keyed by chunk id.

    This is the measurement behind PLAN.md M3(c). With overlapping chunks the
    same text sits in several index entries, so a raw top-k contains fewer
    distinct pieces of information than its length suggests, and recall@k gets
    multiple shots at one document. Scoring a run before and after this call
    gives the size of that inflation directly.
    """
    out: Run = {}
    for qid, chunk_scores in run.items():
        best: dict[DocId, tuple[ChunkId, float]] = {}
        for chunk_id, score in chunk_scores.items():
            doc_id = chunk_to_doc[chunk_id]
            if doc_id not in best or score > best[doc_id][1]:
                best[doc_id] = (chunk_id, score)
        out[qid] = {chunk_id: score for chunk_id, score in best.values()}
    return out


def count_distinct_documents(
    run: Run, chunk_to_doc: Mapping[ChunkId, DocId], k: int
) -> dict[str, int]:
    """Distinct documents present in each query's top-k chunks.

    A diagnostic for chunk-overlap double-counting: if the mean of this is well
    below k, top-k is showing the user the same document repeatedly.
    """
    from reval.metrics.ranking import rank_docs

    return {
        qid: len({chunk_to_doc[c] for c, _ in rank_docs(scores, k=k)})
        for qid, scores in run.items()
    }
