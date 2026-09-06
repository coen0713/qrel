"""recall@k, MRR@k and nDCG@k, implemented from scratch.

Semantics here are not guessed. They were determined empirically against
``pytrec_eval`` (see ``notes/decisions.md`` D10) and are pinned by
``tests/test_metrics_parity.py`` at 1e-6:

- **nDCG uses linear gain** (``gain = grade``), not the exponential
  ``2^grade - 1`` that much of the literature assumes. trec_eval's ``ndcg_cut``
  is linear, and since trec_eval is the reference the field scores against,
  matching it matters more than matching any particular paper.
- **IDCG is truncated at k.** Computing the ideal DCG over *all* relevant
  documents makes nDCG@k unreachable by construction whenever a query has more
  than k relevant documents — on NFCorpus, with ~38 relevant per query, that
  would depress every nDCG@10 by a factor that varies per query. This is the
  bug PLAN.md M1 warns about, and it is the single easiest way to publish a
  wrong number that still looks plausible.
- **Unjudged documents count as non-relevant.** BEIR judgments are sparse, so
  this systematically penalises retrievers that surface good-but-unjudged
  results. We inherit the convention rather than inventing our own, and name it
  as a threat to validity instead of trying to fix it.
"""

from __future__ import annotations

from math import log2

from reval.metrics.ranking import TieBreak, rank_docs
from reval.types import DocId, Qrels, QueryId, Run

#: Default relevance grade at or above which a document counts as relevant.
DEFAULT_REL_THRESHOLD = 1


def recall_at_k(
    ranked: list[tuple[DocId, float]],
    relevant: set[DocId],
    k: int,
) -> float:
    """``|relevant ∩ retrieved@k| / |relevant|``.

    Returns 0.0 when the query has no relevant documents. Such queries are
    dropped at load time (see :class:`reval.types.Dataset`), so this is a
    guard rather than a policy.
    """
    if not relevant:
        return 0.0
    hits = sum(1 for doc_id, _ in ranked[:k] if doc_id in relevant)
    return hits / len(relevant)


def reciprocal_rank_at_k(
    ranked: list[tuple[DocId, float]],
    relevant: set[DocId],
    k: int,
) -> float:
    """Reciprocal rank of the first relevant document, 0 if none within k."""
    for rank, (doc_id, _) in enumerate(ranked[:k], 1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def dcg_at_k(gains: list[float], k: int) -> float:
    """Discounted cumulative gain over an already-ordered gain list."""
    return sum(g / log2(rank + 1) for rank, g in enumerate(gains[:k], 1))


def ndcg_at_k(
    ranked: list[tuple[DocId, float]],
    grades: dict[DocId, int],
    k: int,
) -> float:
    """Normalised DCG at k, with linear gains and IDCG truncated at k."""
    # Negative grades are clamped to 0; some qrels files use -1 as a sentinel
    # for "junk", and a negative gain would make DCG non-monotonic in rank.
    gains = [float(max(grades.get(doc_id, 0), 0)) for doc_id, _ in ranked[:k]]
    ideal_gains = sorted((float(max(g, 0)) for g in grades.values()), reverse=True)

    idcg = dcg_at_k(ideal_gains, k)
    if idcg == 0.0:
        # No positively-graded document exists, so there is no ideal ranking to
        # normalise against. 0/0 is 0 here, not an error.
        return 0.0
    return dcg_at_k(gains, k) / idcg


def evaluate(
    run: Run,
    qrels: Qrels,
    ks: list[int] | tuple[int, ...] = (10,),
    tie_break: TieBreak = "trec",
    rel_threshold: int = DEFAULT_REL_THRESHOLD,
    complete: bool = False,
) -> dict[str, dict[QueryId, float]]:
    """Score a run, returning per-query values for every measure and cutoff.

    Args:
        run: unranked ``{query_id: {doc_id: score}}``.
        qrels: judgments.
        ks: rank cutoffs to report.
        tie_break: ranking policy; see :mod:`reval.metrics.ranking`.
        rel_threshold: minimum grade counting as relevant for recall and MRR.
            nDCG always uses the full graded scale.
        complete: if True, queries present in ``qrels`` but absent from ``run``
            score 0 instead of being skipped. Defaults to False, matching
            ``pytrec_eval``, so that the parity test compares like with like.
            Set it True when a retriever may legitimately return nothing for a
            query and you want that counted as the failure it is.

    Returns:
        ``{"recall@10": {qid: value, ...}, "mrr@10": {...}, "ndcg@10": {...}}``.

    Per-query values, not means: every downstream CI and paired significance
    test resamples over queries, so aggregating here would throw away exactly
    the information the inference needs.
    """
    ks = list(ks)
    if any(k < 1 for k in ks):
        raise ValueError(f"cutoffs must be >= 1, got {ks}")

    qids = set(qrels) if complete else set(qrels) & set(run)
    max_k = max(ks)

    results: dict[str, dict[QueryId, float]] = {}
    for k in ks:
        results[f"recall@{k}"] = {}
        results[f"mrr@{k}"] = {}
        results[f"ndcg@{k}"] = {}

    for qid in sorted(qids):
        grades = qrels.get(qid, {})
        relevant = {d for d, g in grades.items() if g >= rel_threshold}
        # Rank once to the deepest cutoff, then slice; ranking per cutoff would
        # be the same order recomputed max(ks) times.
        ranked = rank_docs(run.get(qid, {}), k=max_k, tie_break=tie_break)
        for k in ks:
            results[f"recall@{k}"][qid] = recall_at_k(ranked, relevant, k)
            results[f"mrr@{k}"][qid] = reciprocal_rank_at_k(ranked, relevant, k)
            results[f"ndcg@{k}"][qid] = ndcg_at_k(ranked, grades, k)

    return results


def mean_scores(per_query: dict[str, dict[QueryId, float]]) -> dict[str, float]:
    """Macro-average each measure over queries.

    Macro, not micro: every query weighs the same regardless of how many
    judgments it has. Micro-averaging on NFCorpus, where judgment counts range
    from a handful to hundreds, would let a few heavily-judged queries decide
    the result.
    """
    return {
        measure: (sum(vals.values()) / len(vals) if vals else 0.0)
        for measure, vals in per_query.items()
    }
