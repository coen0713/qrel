"""Near-duplicate detection — PLAN.md M3(b).

Duplicates inflate recall by giving the retriever several valid targets for one
information need. MinHash + LSH finds them without the O(n²) comparison that
57,638 FiQA documents would otherwise require.

Written before the implementation. The property that matters is that the
estimate tracks true Jaccard: a MinHash that is merely *self-consistent* would
pass a naive test while reporting meaningless similarities.
"""

from __future__ import annotations

import pytest

from reval.contamination.nearduplicate import (
    MinHasher,
    degenerate_documents,
    exact_jaccard,
    find_near_duplicates,
    shingles,
)
from reval.types import Document


def doc(doc_id: str, text: str) -> Document:
    return Document(doc_id, "", text)


# ---- shingles --------------------------------------------------------------


def test_shingles_are_contiguous_word_windows():
    assert shingles("a b c d", k=2) == {"a b", "b c", "c d"}


def test_short_text_yields_one_shingle():
    assert shingles("only two", k=5) == {"only two"}


def test_shingles_of_empty_text_are_empty():
    assert shingles("") == set()


def test_shingles_are_case_insensitive():
    assert shingles("The Cat", k=2) == shingles("the cat", k=2)


# ---- MinHash estimation ----------------------------------------------------


def test_identical_documents_estimate_one():
    h = MinHasher(num_perm=128, seed=0)
    text = "the study found that vitamin d reduced respiratory infection rates"
    assert h.estimate(h.signature(text), h.signature(text)) == 1.0


def test_disjoint_documents_estimate_near_zero():
    h = MinHasher(num_perm=128, seed=0)
    a = h.signature("alpha beta gamma delta epsilon zeta eta theta")
    b = h.signature("one two three four five six seven eight")
    assert h.estimate(a, b) < 0.05


def test_estimate_tracks_true_jaccard():
    """The property that makes the detector meaningful, not just consistent.

    A MinHash that returned a stable but arbitrary number would pass an
    identical/disjoint test. This checks the estimate is actually close to the
    exact Jaccard it is approximating.
    """
    h = MinHasher(num_perm=256, seed=0)
    base = [f"word{i}" for i in range(60)]

    for overlap in (10, 30, 50):
        a_text = " ".join(base[:50])
        b_text = " ".join(base[50 - overlap : 110 - overlap][:50])
        truth = exact_jaccard(shingles(a_text), shingles(b_text))
        est = h.estimate(h.signature(a_text), h.signature(b_text))
        # 256 permutations gives a standard error around 1/sqrt(256) ≈ 0.06.
        assert est == pytest.approx(truth, abs=0.12), f"overlap={overlap}"


def test_more_permutations_reduce_estimation_error():
    base = " ".join(f"word{i}" for i in range(50))
    variant = " ".join(f"word{i}" for i in range(10, 60))
    truth = exact_jaccard(shingles(base), shingles(variant))

    coarse = MinHasher(num_perm=16, seed=0)
    fine = MinHasher(num_perm=512, seed=0)
    err_coarse = abs(coarse.estimate(coarse.signature(base), coarse.signature(variant)) - truth)
    err_fine = abs(fine.estimate(fine.signature(base), fine.signature(variant)) - truth)
    assert err_fine <= err_coarse


def test_signatures_are_deterministic_for_a_seed():
    a = MinHasher(num_perm=64, seed=7).signature("some document text here")
    b = MinHasher(num_perm=64, seed=7).signature("some document text here")
    assert list(a) == list(b)


def test_different_seeds_give_different_permutations():
    a = MinHasher(num_perm=64, seed=1).signature("some document text here")
    b = MinHasher(num_perm=64, seed=2).signature("some document text here")
    assert list(a) != list(b)


def test_signature_length_matches_num_perm():
    assert len(MinHasher(num_perm=32, seed=0).signature("text goes here")) == 32


# ---- exact Jaccard (the oracle for the above) ------------------------------


def test_exact_jaccard_hand_computed():
    assert exact_jaccard({"a", "b", "c"}, {"b", "c", "d"}) == pytest.approx(2 / 4)


def test_exact_jaccard_of_two_empty_sets_is_zero():
    assert exact_jaccard(set(), set()) == 0.0


# ---- LSH over a corpus -----------------------------------------------------


def test_finds_an_exact_duplicate_pair():
    text = "the immune system responds to vitamin d supplementation in measurable ways"
    corpus = {
        "d1": doc("d1", text),
        "d2": doc("d2", text),
        "d3": doc("d3", "completely different content about financial markets and bonds"),
    }
    pairs = find_near_duplicates(corpus, threshold=0.8, seed=0)
    found = {tuple(sorted((p.doc_a, p.doc_b))) for p in pairs}
    assert ("d1", "d2") in found
    assert not any("d3" in p for p in found)


def test_finds_a_near_duplicate_that_is_not_identical():
    base = " ".join(f"token{i}" for i in range(60))
    corpus = {
        "d1": doc("d1", base),
        "d2": doc("d2", base + " with a short tail appended"),
        "d3": doc("d3", " ".join(f"other{i}" for i in range(60))),
    }
    pairs = find_near_duplicates(corpus, threshold=0.7, seed=0)
    assert {tuple(sorted((p.doc_a, p.doc_b))) for p in pairs} == {("d1", "d2")}


def test_reports_the_estimated_similarity():
    text = "identical text for both documents in this corpus fixture here"
    pairs = find_near_duplicates({"a": doc("a", text), "b": doc("b", text)}, threshold=0.8, seed=0)
    assert len(pairs) == 1
    assert pairs[0].similarity == pytest.approx(1.0, abs=0.01)


def test_a_corpus_with_no_duplicates_returns_nothing():
    corpus = {
        f"d{i}": doc(f"d{i}", " ".join(f"unique{i}word{j}" for j in range(40))) for i in range(10)
    }
    assert find_near_duplicates(corpus, threshold=0.8, seed=0) == []


def test_each_pair_is_reported_once():
    text = "repeated content across three separate documents in the corpus fixture"
    corpus = {k: doc(k, text) for k in ("a", "b", "c")}
    pairs = find_near_duplicates(corpus, threshold=0.8, seed=0)
    keys = [tuple(sorted((p.doc_a, p.doc_b))) for p in pairs]
    assert len(keys) == len(set(keys))
    assert set(keys) == {("a", "b"), ("a", "c"), ("b", "c")}


def test_a_higher_threshold_finds_no_more_pairs():
    base = " ".join(f"tok{i}" for i in range(50))
    corpus = {
        "d1": doc("d1", base),
        "d2": doc("d2", " ".join(f"tok{i}" for i in range(15, 65))),
    }
    loose = find_near_duplicates(corpus, threshold=0.3, seed=0)
    strict = find_near_duplicates(corpus, threshold=0.95, seed=0)
    assert len(strict) <= len(loose)


def test_results_are_deterministic():
    text = "some shared content between these two documents for the test"
    corpus = {"a": doc("a", text), "b": doc("b", text + " tail"), "c": doc("c", "unrelated words")}
    first = find_near_duplicates(corpus, threshold=0.5, seed=0)
    second = find_near_duplicates(corpus, threshold=0.5, seed=0)
    assert [(p.doc_a, p.doc_b) for p in first] == [(p.doc_a, p.doc_b) for p in second]


def test_empty_and_single_document_corpora_are_handled():
    assert find_near_duplicates({}, seed=0) == []
    assert find_near_duplicates({"a": doc("a", "text")}, seed=0) == []


# ---- degenerate documents --------------------------------------------------


def test_empty_documents_are_not_reported_as_duplicates_of_each_other():
    """Found on real data: FiQA contains 38 empty documents.

    They collide trivially, forming one 38-member "cluster" that accounted for
    703 of 890 reported pairs and buried every real duplicate. "Near duplicate"
    is not a meaningful claim about two empty strings.
    """
    corpus = {f"e{i}": doc(f"e{i}", "") for i in range(5)}
    corpus["real"] = doc("real", " ".join(f"word{i}" for i in range(40)))
    assert find_near_duplicates(corpus, threshold=0.8, seed=0) == []


def test_very_short_documents_are_excluded():
    corpus = {"a": doc("a", "thanks"), "b": doc("b", "thanks"), "c": doc("c", "thanks")}
    assert find_near_duplicates(corpus, threshold=0.8, seed=0) == []


def test_real_duplicates_still_found_alongside_degenerate_ones():
    text = " ".join(f"token{i}" for i in range(40))
    corpus = {
        "empty1": doc("empty1", ""),
        "empty2": doc("empty2", ""),
        "dup1": doc("dup1", text),
        "dup2": doc("dup2", text),
    }
    pairs = find_near_duplicates(corpus, threshold=0.8, seed=0)
    assert {tuple(sorted((p.doc_a, p.doc_b))) for p in pairs} == {("dup1", "dup2")}


def test_degenerate_documents_are_reported_separately():
    corpus = {
        "empty": doc("empty", ""),
        "tiny": doc("tiny", "two words"),
        "fine": doc("fine", " ".join(f"w{i}" for i in range(20))),
    }
    assert degenerate_documents(corpus) == ["empty", "tiny"]


def test_min_words_threshold_is_adjustable():
    corpus = {"a": doc("a", "one two three"), "b": doc("b", " ".join(f"w{i}" for i in range(20)))}
    assert degenerate_documents(corpus, min_words=2) == []
    assert degenerate_documents(corpus, min_words=10) == ["a"]
