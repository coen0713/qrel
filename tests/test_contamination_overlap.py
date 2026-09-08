"""Lexical overlap detectors — PLAN.md M3(a), the headline measurement.

Written before the implementation, per CLAUDE.md. These detectors produce the
contamination scores the whole experiment is stratified by, so a plausible-
looking wrong implementation here would not fail loudly — it would just produce
a gradient that means something other than what the writeup claims.

Every expected value below is hand-computed from the definition, not copied from
a run.
"""

from __future__ import annotations

import pytest

from reval.contamination.overlap import (
    OverlapScores,
    assign_deciles,
    char_ngram_jaccard,
    normalized_lcs,
    score_pair,
    score_query_set,
    token_containment,
)
from reval.types import Document

# ---- character n-gram Jaccard ---------------------------------------------


def test_identical_strings_have_jaccard_one():
    assert char_ngram_jaccard("the quick brown fox", "the quick brown fox") == 1.0


def test_disjoint_strings_have_jaccard_zero():
    assert char_ngram_jaccard("aaaaaaaa", "bbbbbbbb") == 0.0


def test_jaccard_is_symmetric():
    a, b = "retrieval evaluation harness", "evaluation of retrieval systems"
    assert char_ngram_jaccard(a, b) == char_ngram_jaccard(b, a)


def test_jaccard_hand_computed():
    # n=5 over "abcdef" -> {abcde, bcdef}; over "abcdeX" -> {abcde, bcdeX}.
    # Intersection {abcde} = 1, union {abcde, bcdef, bcdeX} = 3.
    assert char_ngram_jaccard("abcdef", "abcdeX", n=5) == pytest.approx(1 / 3)


def test_jaccard_is_bounded():
    for a, b in [("hello world", "world hello"), ("x", "y"), ("abc", "abcabc")]:
        assert 0.0 <= char_ngram_jaccard(a, b) <= 1.0


def test_strings_shorter_than_n_are_handled():
    # Must not silently produce an empty gram set and divide by zero.
    assert char_ngram_jaccard("ab", "ab", n=5) == 1.0
    assert char_ngram_jaccard("ab", "cd", n=5) == 0.0


def test_empty_strings_are_zero_not_a_crash():
    assert char_ngram_jaccard("", "abc") == 0.0
    assert char_ngram_jaccard("", "") == 0.0


def test_jaccard_is_case_insensitive():
    # A query that differs only in capitalisation is not less contaminated.
    assert char_ngram_jaccard("Hello World", "hello world") == 1.0


# ---- normalized longest common substring ----------------------------------


def test_lcs_of_fully_contained_query_is_one():
    assert normalized_lcs("hello", "xxxhelloyyy") == 1.0


def test_lcs_of_disjoint_strings_is_zero():
    assert normalized_lcs("abc", "xyz") == 0.0


def test_lcs_hand_computed():
    # Longest common substring of "hello world" and "hello there" is "hello "
    # (6 chars, including the trailing space). Query length is 11.
    assert normalized_lcs("hello world", "hello there") == pytest.approx(6 / 11)


def test_lcs_normalises_by_query_length_not_document_length():
    """The point of the normalisation: a long gold passage must not dilute it.

    If it were normalised by document length, every score against a 1,500-char
    SciFact abstract would be near zero and the detector would be useless.
    """
    query = "immune response"
    short_doc = "the immune response is"
    long_doc = "the immune response is " + ("padding text " * 200)
    assert normalized_lcs(query, short_doc) == normalized_lcs(query, long_doc) == 1.0


def test_lcs_of_empty_query_is_zero():
    assert normalized_lcs("", "anything") == 0.0


def test_lcs_is_case_insensitive():
    assert normalized_lcs("Immune Response", "the immune response is") == 1.0


# ---- token containment ----------------------------------------------------


def test_token_containment_hand_computed():
    # {the, cat, sat} ∩ {a, cat, sat, here} = {cat, sat} -> 2/3
    assert token_containment("the cat sat", "a cat sat here") == pytest.approx(2 / 3)


def test_full_containment_is_one():
    assert token_containment("cat sat", "the cat sat on the mat") == 1.0


def test_no_containment_is_zero():
    assert token_containment("dog", "the cat sat") == 0.0


def test_token_containment_ignores_repetition():
    # A set measure: repeating a term in the query must not change the fraction.
    assert token_containment("cat cat cat", "the cat") == 1.0


def test_token_containment_of_empty_query_is_zero():
    assert token_containment("", "anything") == 0.0


def test_token_containment_is_case_insensitive():
    assert token_containment("Cat SAT", "the cat sat") == 1.0


# ---- the composite ---------------------------------------------------------


def test_score_pair_returns_all_three_detectors():
    s = score_pair("the immune response", "studies of the immune response in mice")
    assert isinstance(s, OverlapScores)
    for value in (s.jaccard, s.lcs_ratio, s.containment):
        assert 0.0 <= value <= 1.0


def test_a_verbatim_query_scores_at_the_ceiling_on_every_detector():
    """The pathological case the experiment is built to detect.

    A query copied verbatim out of its gold passage should max out all three
    detectors. If any of them fails to, that detector cannot do its job.
    """
    passage = "Vitamin D supplementation reduced the incidence of acute respiratory infection."
    query = "Vitamin D supplementation reduced the incidence of acute respiratory infection"
    s = score_pair(query, passage)
    assert s.lcs_ratio == pytest.approx(1.0)
    assert s.containment == pytest.approx(1.0)
    assert s.jaccard > 0.9


def test_a_paraphrase_scores_far_below_a_verbatim_copy():
    passage = "Vitamin D supplementation reduced the incidence of acute respiratory infection."
    verbatim = "Vitamin D supplementation reduced the incidence of acute respiratory infection"
    paraphrase = "Does taking cholecalciferol lower how often people catch colds?"
    assert score_pair(paraphrase, passage).jaccard < score_pair(verbatim, passage).jaccard
    assert score_pair(paraphrase, passage).containment < 0.4


def test_score_query_set_uses_the_gold_document():
    corpus = {
        "d1": Document("d1", "", "the immune response in mice"),
        "d2": Document("d2", "", "completely unrelated financial text"),
    }
    queries = {"q1": "immune response"}
    qrels = {"q1": {"d1": 1}}
    scores = score_query_set(queries, qrels, corpus)
    assert set(scores) == {"q1"}
    assert scores["q1"].containment == 1.0


def test_score_query_set_takes_the_maximum_over_multiple_gold_documents():
    """Contamination is a property of the easiest route to a correct answer.

    If any one gold document is a near-copy of the query, the query is
    contaminated regardless of how distinct the other gold documents are.
    """
    corpus = {
        "d1": Document("d1", "", "nothing alike at all"),
        "d2": Document("d2", "", "immune response verbatim here"),
    }
    queries = {"q1": "immune response"}
    qrels = {"q1": {"d1": 1, "d2": 1}}
    assert score_query_set(queries, qrels, corpus)["q1"].containment == 1.0


def test_score_query_set_ignores_non_relevant_judgments():
    corpus = {
        "d1": Document("d1", "", "immune response exact match"),
        "d2": Document("d2", "", "unrelated"),
    }
    queries = {"q1": "immune response"}
    qrels = {"q1": {"d1": 0, "d2": 1}}  # d1 judged NON-relevant
    assert score_query_set(queries, qrels, corpus)["q1"].containment == 0.0


def test_score_query_set_skips_queries_with_no_gold_document():
    corpus = {"d1": Document("d1", "", "text")}
    scores = score_query_set({"q1": "a", "q2": "b"}, {"q1": {"d1": 1}, "q2": {}}, corpus)
    assert set(scores) == {"q1"}


def test_missing_gold_document_does_not_crash():
    # A qrel pointing at an absent doc is a data problem, not a reason to die
    # halfway through scoring 500 queries.
    scores = score_query_set({"q1": "a"}, {"q1": {"ghost": 1}}, {})
    assert scores == {} or scores["q1"].containment == 0.0


# ---- decile stratification -------------------------------------------------


def test_deciles_split_into_ten_roughly_equal_bins():
    scores = {f"q{i}": i / 100 for i in range(100)}
    deciles = assign_deciles(scores, n_bins=10)
    counts: dict[int, int] = {}
    for d in deciles.values():
        counts[d] = counts.get(d, 0) + 1
    assert set(counts) == set(range(10))
    assert all(8 <= c <= 12 for c in counts.values())


def test_decile_zero_is_the_lowest_overlap():
    scores = {"low": 0.01, "mid": 0.5, "high": 0.99}
    deciles = assign_deciles(scores, n_bins=3)
    assert deciles["low"] < deciles["high"]
    assert deciles["low"] == 0


def test_deciles_are_deterministic():
    scores = {f"q{i}": (i * 37 % 101) / 101 for i in range(200)}
    assert assign_deciles(scores) == assign_deciles(scores)


def test_deciles_handle_heavy_ties_without_crashing():
    # Every score identical: quantile boundaries collapse. Must still return a
    # valid bin per query rather than raise.
    scores = {f"q{i}": 0.5 for i in range(50)}
    deciles = assign_deciles(scores)
    assert len(deciles) == 50
    assert all(0 <= d <= 9 for d in deciles.values())


def test_empty_input_gives_empty_deciles():
    assert assign_deciles({}) == {}
