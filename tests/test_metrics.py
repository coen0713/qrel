"""Metric semantics on hand-computed examples.

The parity test proves we agree with trec_eval. These prove we understand *why*,
by pinning the specific behaviours that a plausible-looking wrong implementation
would get wrong.
"""

from __future__ import annotations

from math import log2

import pytest

from reval.metrics import evaluate, mean_scores, ndcg_at_k, recall_at_k, reciprocal_rank_at_k


def ranked(*doc_ids):
    """Build a ranked list with strictly decreasing scores (no ties)."""
    return [(d, float(100 - i)) for i, d in enumerate(doc_ids)]


# ---- recall ---------------------------------------------------------------


def test_recall_is_over_all_relevant_not_over_k():
    # The classic confusion with precision@k: the denominator is |relevant|.
    assert recall_at_k(ranked("a", "b"), {"a", "b", "c", "d"}, k=10) == pytest.approx(0.5)


def test_recall_with_no_relevant_documents_is_zero_not_nan():
    assert recall_at_k(ranked("a"), set(), k=10) == 0.0


def test_recall_ignores_documents_past_k():
    assert recall_at_k(ranked("x", "y", "a"), {"a"}, k=2) == 0.0
    assert recall_at_k(ranked("x", "y", "a"), {"a"}, k=3) == 1.0


# ---- MRR ------------------------------------------------------------------


def test_mrr_uses_the_first_relevant_document_only():
    assert reciprocal_rank_at_k(ranked("x", "a", "b"), {"a", "b"}, k=10) == pytest.approx(0.5)


def test_mrr_is_zero_when_nothing_relevant_is_within_k():
    # Not undefined, not None: 0. A retriever that misses entirely scores 0.
    assert reciprocal_rank_at_k(ranked("x", "y", "a"), {"a"}, k=2) == 0.0


def test_mrr_rank_one_is_one():
    assert reciprocal_rank_at_k(ranked("a"), {"a"}, k=1) == 1.0


# ---- nDCG -----------------------------------------------------------------


def test_ndcg_uses_linear_gain_not_exponential():
    # Determined empirically against pytrec_eval; see notes/decisions.md D10.
    # A grade-1 doc ranked above a grade-2 doc separates the two conventions.
    grades = {"d1": 2, "d2": 1}
    got = ndcg_at_k(ranked("d2", "d1"), grades, k=2)

    linear = (1 / log2(2) + 2 / log2(3)) / (2 / log2(2) + 1 / log2(3))
    exponential = (1 / log2(2) + 3 / log2(3)) / (3 / log2(2) + 1 / log2(3))

    assert got == pytest.approx(linear, abs=1e-12)
    assert got != pytest.approx(exponential, abs=1e-6)


def test_ndcg_truncates_idcg_at_k():
    """The bug PLAN.md M1 warns about, pinned.

    Three relevant documents, k=2, and the retriever returns two of them
    perfectly. The correct answer is 1.0: within a two-document budget it did
    the best anything could do. Computing IDCG over all three relevant docs
    gives ~0.765 and makes nDCG@2 unreachable by construction — which on
    NFCorpus, at ~38 relevant documents per query, would depress every score.
    """
    grades = {"d1": 1, "d2": 1, "d3": 1}
    assert ndcg_at_k(ranked("d1", "d2"), grades, k=2) == pytest.approx(1.0)

    idcg_over_all = 1 / log2(2) + 1 / log2(3) + 1 / log2(4)
    wrong = (1 / log2(2) + 1 / log2(3)) / idcg_over_all
    assert wrong == pytest.approx(0.7653606, abs=1e-6)
    assert ndcg_at_k(ranked("d1", "d2"), grades, k=2) != pytest.approx(wrong, abs=1e-6)


def test_ndcg_perfect_ranking_is_one():
    grades = {"a": 2, "b": 1, "c": 1}
    assert ndcg_at_k(ranked("a", "b", "c"), grades, k=3) == pytest.approx(1.0)


def test_ndcg_with_only_grade_zero_judgments_is_zero():
    assert ndcg_at_k(ranked("a"), {"a": 0}, k=10) == 0.0


def test_ndcg_clamps_negative_grades():
    # A negative gain would make DCG non-monotonic in rank.
    assert ndcg_at_k(ranked("a", "b"), {"a": -1, "b": 1}, k=2) == pytest.approx(
        (1 / log2(3)) / (1 / log2(2))
    )


def test_ndcg_treats_unjudged_documents_as_zero_gain():
    grades = {"a": 1}
    assert ndcg_at_k(ranked("unjudged", "a"), grades, k=2) == pytest.approx(1 / log2(3))


# ---- evaluate -------------------------------------------------------------


def test_evaluate_emits_every_measure_at_every_cutoff():
    qrels = {"q1": {"a": 1}}
    run = {"q1": {"a": 1.0}}
    got = evaluate(run, qrels, ks=[1, 5])
    assert set(got) == {"recall@1", "mrr@1", "ndcg@1", "recall@5", "mrr@5", "ndcg@5"}


def test_evaluate_skips_unrun_queries_by_default():
    # Matches pytrec_eval, which is what makes the parity comparison like-for-like.
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    run = {"q1": {"a": 1.0}}
    assert set(evaluate(run, qrels, ks=[1])["recall@1"]) == {"q1"}


def test_evaluate_complete_scores_missing_queries_as_zero():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    run = {"q1": {"a": 1.0}}
    got = evaluate(run, qrels, ks=[1], complete=True)["recall@1"]
    assert got == {"q1": 1.0, "q2": 0.0}


def test_evaluate_respects_the_relevance_threshold():
    qrels = {"q1": {"a": 1, "b": 2}}
    run = {"q1": {"a": 5.0}}
    assert evaluate(run, qrels, ks=[10])["recall@10"]["q1"] == pytest.approx(0.5)
    strict = evaluate(run, qrels, ks=[10], rel_threshold=2)["recall@10"]["q1"]
    assert strict == pytest.approx(0.0)


def test_evaluate_rejects_nonsense_cutoffs():
    with pytest.raises(ValueError, match="cutoffs must be >= 1"):
        evaluate({}, {}, ks=[0])


def test_mean_scores_macro_averages():
    per_query = {"recall@10": {"q1": 1.0, "q2": 0.0, "q3": 0.5}}
    assert mean_scores(per_query)["recall@10"] == pytest.approx(0.5)


def test_mean_scores_of_nothing_is_zero_not_a_crash():
    assert mean_scores({"recall@10": {}})["recall@10"] == 0.0
