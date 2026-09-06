"""Turning scores into a ranking, deterministically.

This module exists because tie-breaking is not a detail. Retrieval scores tie
constantly — BM25 gives identical scores to documents sharing the same matched
terms, and any float32 dot product ties at some precision. If ties break by
whatever order a dict happened to iterate in, two runs of the same experiment
produce different numbers, and you will waste a day on it.

The default policy matches ``trec_eval``, which is what ``pytrec_eval`` wraps:
sort by score descending, then break ties by **descending** document id, string
comparison. That reversed direction is not a typo and not our choice; it is
``comp_sim_docno`` in trec_eval's ``form_res_rels.c``. We match it so that our
metrics and pytrec_eval's rank identical documents at identical positions, which
is a precondition for the 1e-6 parity test meaning anything.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from reval.types import DocId, QueryId, Run, ScoredDoc

TieBreak = Literal["trec", "docid_asc"]


def rank_docs(
    scores: dict[DocId, float],
    k: int | None = None,
    tie_break: TieBreak = "trec",
) -> list[tuple[DocId, float]]:
    """Rank ``{doc_id: score}`` into a deterministic ordered list.

    Args:
        scores: unranked doc -> score mapping.
        k: keep only the top ``k``; ``None`` keeps everything.
        tie_break: ``"trec"`` (score desc, doc_id desc) reproduces trec_eval.
            ``"docid_asc"`` (score desc, doc_id asc) is the intuitive
            alternative, available so the effect of the choice is measurable
            rather than assumed.

    Returns:
        ``[(doc_id, score), ...]`` in rank order, best first.
    """
    if tie_break == "trec":
        # Python sorts ascending and is stable, so to get "score desc, docid
        # desc" we sort by docid descending first, then by score descending.
        # Doing it in one key with a negated string is not possible, hence two
        # passes; both are O(n log n) and n is top-k sized in practice.
        items = sorted(scores.items(), key=lambda kv: kv[0], reverse=True)
        items.sort(key=lambda kv: kv[1], reverse=True)
    elif tie_break == "docid_asc":
        items = sorted(scores.items(), key=lambda kv: kv[0])
        items.sort(key=lambda kv: kv[1], reverse=True)
    else:  # pragma: no cover - guarded by the Literal type
        raise ValueError(f"unknown tie_break policy: {tie_break!r}")

    return items[:k] if k is not None else items


def rank_run(
    run: Run,
    k: int | None = None,
    tie_break: TieBreak = "trec",
) -> dict[QueryId, list[tuple[DocId, float]]]:
    """Apply :func:`rank_docs` to every query in a run."""
    return {qid: rank_docs(scores, k=k, tie_break=tie_break) for qid, scores in run.items()}


def truncate_run(run: Run, k: int, tie_break: TieBreak = "trec") -> Run:
    """Cut a run down to its top ``k`` per query, preserving the mapping shape.

    Used to compute cut-off variants of measures that ``pytrec_eval`` only
    offers uncut (``recip_rank``), so that MRR@k can be checked against the
    oracle at all.
    """
    return {qid: dict(rank_docs(scores, k=k, tie_break=tie_break)) for qid, scores in run.items()}


def to_scored_docs(
    run: Run,
    k: int | None = None,
    tie_break: TieBreak = "trec",
) -> Iterable[ScoredDoc]:
    """Flatten a run into ranked :class:`ScoredDoc` records, queries in id order."""
    for qid in sorted(run):
        for rank, (doc_id, score) in enumerate(rank_docs(run[qid], k=k, tie_break=tie_break), 1):
            yield ScoredDoc(query_id=qid, doc_id=doc_id, rank=rank, score=score)
