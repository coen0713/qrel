"""Hard-negative mining and the false-negative trap.

The trap PLAN.md M4 names: BEIR's judgments are sparse, so a top-ranked unjudged
document is not "known non-relevant" — it is the document the retriever thought
was best. Mining those as negatives selects precisely for unlabelled positives.
These tests pin the behaviours that make that visible rather than silent.
"""

from __future__ import annotations

import numpy as np
import pytest

from reval.negatives.mine import (
    MiningReport,
    confusion_profile,
    difficulty_stratified_queries,
    filter_false_negatives,
    mine,
)
from reval.types import Document

CORPUS = {
    "gold": Document("gold", "", "vitamin d supplementation reduces respiratory infection"),
    "near": Document("near", "", "vitamin d supplementation and respiratory outcomes in adults"),
    "far": Document("far", "", "quarterly bond yields and municipal financing structures"),
    "other": Document("other", "", "unrelated content about geological surveys"),
}
QUERIES = {"q1": "does vitamin d reduce respiratory infection"}
QRELS = {"q1": {"gold": 1}}
RUN = {"q1": {"gold": 0.9, "near": 0.8, "far": 0.4, "other": 0.2}}


class StubScorer:
    """Deterministic cross-encoder stand-in: 'near' looks relevant, others don't."""

    identity = "stub-cross-encoder"

    def __init__(self):
        self.calls = 0

    def score(self, pairs, show_progress=False):
        self.calls += 1
        return np.array(
            [5.0 if "respiratory" in doc else -5.0 for _, doc in pairs], dtype=np.float32
        )


# ---- mining ----------------------------------------------------------------


def test_known_positives_are_excluded():
    report = mine(RUN, QRELS, depth=10, per_query=5)
    assert "gold" not in [n.doc_id for n in report.negatives["q1"]]


def test_negatives_come_back_in_rank_order():
    report = mine(RUN, QRELS, depth=10, per_query=5)
    ranks = [n.rank for n in report.negatives["q1"]]
    assert ranks == sorted(ranks)
    assert report.negatives["q1"][0].doc_id == "near"


def test_per_query_limit_is_respected():
    report = mine(RUN, QRELS, depth=10, per_query=2)
    assert len(report.negatives["q1"]) == 2


def test_depth_limits_how_far_down_we_look():
    # depth=2 sees only {gold, near}; gold is a positive, so one negative.
    report = mine(RUN, QRELS, depth=2, per_query=10)
    assert [n.doc_id for n in report.negatives["q1"]] == ["near"]


def test_grade_zero_documents_are_legitimate_hard_negatives():
    """A judged non-relevant document is the best kind of hard negative.

    A human confirmed it is wrong, so unlike an unjudged document it carries no
    false-negative risk. Excluding it would throw away the only negatives we
    actually know are negative.
    """
    qrels = {"q1": {"gold": 1, "near": 0}}
    report = mine(RUN, qrels, depth=10, per_query=5)
    assert "near" in [n.doc_id for n in report.negatives["q1"]]


def test_retrieval_score_and_rank_are_carried_through():
    n = mine(RUN, QRELS, depth=10, per_query=1).negatives["q1"][0]
    assert n.retrieval_score == pytest.approx(0.8)
    assert n.rank == 2  # gold is rank 1


def test_every_query_appears_even_with_no_negatives():
    run = {"q1": {"gold": 1.0}, "q2": {}}
    report = mine(run, {"q1": {"gold": 1}}, depth=10, per_query=5)
    assert set(report.negatives) == {"q1", "q2"}
    assert report.negatives["q1"] == []


# ---- the false-negative filter ---------------------------------------------


def test_filtering_flags_candidates_that_look_relevant():
    report = mine(RUN, QRELS, depth=10, per_query=5)
    filtered = filter_false_negatives(report, QUERIES, CORPUS, StubScorer(), threshold=0.0)
    flagged = {n.doc_id for n in filtered.all_negatives if n.suspected_positive}
    assert flagged == {"near"}


def test_filtering_does_not_delete_anything():
    """The raw-vs-filtered gap is the result; a pruned set cannot produce it."""
    report = mine(RUN, QRELS, depth=10, per_query=5)
    filtered = filter_false_negatives(report, QUERIES, CORPUS, StubScorer(), threshold=0.0)
    assert filtered.n_candidates == report.n_candidates
    assert filtered.n_suspected == 1
    assert len(filtered.usable_only()["q1"]) == report.n_candidates - 1


def test_suspected_fraction_is_the_headline_number():
    report = mine(RUN, QRELS, depth=10, per_query=5)
    filtered = filter_false_negatives(report, QUERIES, CORPUS, StubScorer(), threshold=0.0)
    # 3 candidates (near, far, other); one flagged.
    assert filtered.suspected_fraction == pytest.approx(1 / 3)


def test_cross_encoder_scores_are_recorded_on_every_candidate():
    report = mine(RUN, QRELS, depth=10, per_query=5)
    filtered = filter_false_negatives(report, QUERIES, CORPUS, StubScorer(), threshold=0.0)
    assert all(n.cross_encoder_score is not None for n in filtered.all_negatives)


def test_a_higher_threshold_flags_fewer_candidates():
    report = mine(RUN, QRELS, depth=10, per_query=5)
    loose = filter_false_negatives(report, QUERIES, CORPUS, StubScorer(), threshold=-10.0)
    strict = filter_false_negatives(report, QUERIES, CORPUS, StubScorer(), threshold=100.0)
    assert loose.n_suspected == 3
    assert strict.n_suspected == 0


def test_scores_are_matched_to_the_right_candidate():
    """The alignment bug that would be invisible: right count, wrong pairing.

    Scores are computed in one batch and mapped back by position. If that
    mapping slipped, every candidate would still get a plausible score and the
    filter would flag the wrong documents with no error anywhere.
    """
    run = {"q1": {"gold": 0.9, "near": 0.8, "far": 0.7, "other": 0.6}}
    report = mine(run, QRELS, depth=10, per_query=5)
    filtered = filter_false_negatives(report, QUERIES, CORPUS, StubScorer(), threshold=0.0)
    by_doc = {n.doc_id: n.cross_encoder_score for n in filtered.all_negatives}
    assert by_doc["near"] == pytest.approx(5.0)
    assert by_doc["far"] == pytest.approx(-5.0)
    assert by_doc["other"] == pytest.approx(-5.0)


def test_filtering_an_empty_report_is_a_no_op():
    empty = MiningReport(method="bm25", negatives={"q1": []})
    scorer = StubScorer()
    assert filter_false_negatives(empty, QUERIES, CORPUS, scorer, 0.0).n_candidates == 0
    assert scorer.calls == 0


def test_documents_missing_from_the_corpus_are_left_unscored():
    run = {"q1": {"gold": 0.9, "ghost": 0.8}}
    report = mine(run, QRELS, depth=10, per_query=5)
    filtered = filter_false_negatives(report, QUERIES, CORPUS, StubScorer(), threshold=0.0)
    ghost = next(n for n in filtered.all_negatives if n.doc_id == "ghost")
    assert ghost.cross_encoder_score is None
    assert not ghost.suspected_positive


# ---- difficulty and diagnosis ----------------------------------------------


def test_difficulty_is_the_rank_of_the_first_relevant_document():
    assert difficulty_stratified_queries(RUN, QRELS)["q1"] == 1


def test_a_buried_gold_document_scores_as_harder():
    buried = {"q1": {"near": 0.9, "far": 0.8, "other": 0.7, "gold": 0.1}}
    assert difficulty_stratified_queries(buried, QRELS)["q1"] == 4


def test_a_query_whose_gold_is_never_retrieved_scores_zero():
    assert difficulty_stratified_queries({"q1": {"far": 0.9}}, QRELS)["q1"] == 0


def test_confusion_profile_compares_negatives_against_gold():
    report = mine(RUN, QRELS, depth=10, per_query=5)
    profile = confusion_profile(report, QUERIES, QRELS, CORPUS)
    assert profile["n_positives"] == 1
    assert profile["n_negatives"] == 3
    # The gold document shares more query vocabulary than the average negative,
    # which is what a healthy retriever should look like.
    assert profile["mean_positive_containment"] > profile["mean_negative_containment"]
    assert profile["gap"] < 0


def test_confusion_profile_survives_an_empty_report():
    empty = MiningReport(method="bm25", negatives={})
    profile = confusion_profile(empty, QUERIES, QRELS, CORPUS)
    assert profile["gap"] == 0.0
