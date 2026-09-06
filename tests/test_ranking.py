"""Tie-breaking must be deterministic, or nothing reproduces."""

from __future__ import annotations

import random

from reval.metrics.ranking import rank_docs, rank_run, truncate_run


def test_ranks_by_score_descending():
    scores = {"a": 0.1, "b": 0.9, "c": 0.5}
    assert [d for d, _ in rank_docs(scores)] == ["b", "c", "a"]


def test_trec_policy_breaks_ties_by_descending_docid():
    # This is trec_eval's `comp_sim_docno`: strcmp(ptr2->docno, ptr1->docno).
    # Reversed on purpose; see reval.metrics.ranking module docstring.
    scores = {"doc1": 1.0, "doc2": 1.0, "doc3": 1.0}
    assert [d for d, _ in rank_docs(scores, tie_break="trec")] == ["doc3", "doc2", "doc1"]


def test_docid_asc_policy_is_the_mirror_image():
    scores = {"doc1": 1.0, "doc2": 1.0, "doc3": 1.0}
    assert [d for d, _ in rank_docs(scores, tie_break="docid_asc")] == ["doc1", "doc2", "doc3"]


def test_ties_only_break_within_equal_scores():
    scores = {"z": 2.0, "a": 1.0, "b": 1.0}
    assert [d for d, _ in rank_docs(scores, tie_break="trec")] == ["z", "b", "a"]


def test_ranking_is_invariant_to_dict_insertion_order():
    # The failure this guards against: relying on dict iteration order, which
    # differs between an index built in one pass and one rebuilt from cache.
    base = {f"d{i}": float(i % 3) for i in range(50)}
    expected = rank_docs(base)
    for seed in range(5):
        items = list(base.items())
        random.Random(seed).shuffle(items)
        assert rank_docs(dict(items)) == expected


def test_k_truncates():
    scores = {"a": 3.0, "b": 2.0, "c": 1.0}
    assert [d for d, _ in rank_docs(scores, k=2)] == ["a", "b"]
    assert rank_docs(scores, k=0) == []


def test_rank_run_covers_every_query():
    run = {"q1": {"a": 1.0}, "q2": {"b": 2.0, "c": 3.0}}
    ranked = rank_run(run)
    assert set(ranked) == {"q1", "q2"}
    assert [d for d, _ in ranked["q2"]] == ["c", "b"]


def test_truncate_run_keeps_mapping_shape():
    run = {"q1": {"a": 3.0, "b": 2.0, "c": 1.0}}
    out = truncate_run(run, k=2)
    assert out == {"q1": {"a": 3.0, "b": 2.0}}
