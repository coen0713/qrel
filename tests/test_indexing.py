"""Retriever backends behind one protocol."""

from __future__ import annotations

import pytest

from reval.indexing import BM25Retriever, IndexNotBuiltError, Retriever
from reval.types import Chunk

DOCS = {
    "c1": "the cat sat on the mat and purred",
    "c2": "dogs run fast across the open field",
    "c3": "feline behaviour: a cat will purr when content",
    "c4": "financial markets closed lower on bond yields",
}


def chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id=cid,
            doc_id=cid.replace("c", "d"),
            ordinal=0,
            text=t,
            char_start=0,
            char_end=len(t),
        )
        for cid, t in DOCS.items()
    ]


def test_bm25_satisfies_the_retriever_protocol():
    assert isinstance(BM25Retriever(), Retriever)


def test_search_before_index_is_a_clear_error():
    with pytest.raises(IndexNotBuiltError, match="call index"):
        BM25Retriever().search({"q1": "cat"}, top_k=3)


def test_indexing_nothing_is_rejected():
    with pytest.raises(ValueError, match="empty chunk list"):
        BM25Retriever().index([])


def test_bm25_ranks_lexically_matching_chunks_first():
    r = BM25Retriever()
    r.index(chunks())
    hits = r.search({"q1": "cat purr"}, top_k=4)["q1"]
    top = max(hits, key=hits.get)
    assert top in {"c1", "c3"}
    assert "c4" not in hits  # no shared terms at all


def test_bm25_returns_chunk_ids_not_document_ids():
    # Collapsing to documents is an explicit later step with a named policy;
    # a retriever that did it internally would hide that decision.
    r = BM25Retriever()
    r.index(chunks())
    assert set(r.search({"q1": "cat"}, top_k=4)["q1"]) <= set(DOCS)


def test_top_k_larger_than_the_corpus_is_clamped():
    # bm25s raises rather than returning what it has; smoke runs and small
    # fixtures would otherwise be impossible.
    r = BM25Retriever()
    r.index(chunks())
    hits = r.search({"q1": "cat"}, top_k=1000)["q1"]
    assert 0 < len(hits) <= len(DOCS)


def test_zero_score_hits_are_dropped():
    """A zero BM25 score means no query term matched.

    Keeping those would pad the run with documents whose only claim to a rank is
    the tie-break policy, which inflates recall@100 with noise.
    """
    r = BM25Retriever()
    r.index(chunks())
    hits = r.search({"q1": "cat"}, top_k=4)["q1"]
    assert all(s > 0 for s in hits.values())
    assert "c2" not in hits


def test_every_query_appears_in_the_run_even_when_nothing_matches():
    r = BM25Retriever()
    r.index(chunks())
    run = r.search({"q1": "cat", "q2": "zzzz nonexistentterm"}, top_k=3)
    assert set(run) == {"q1", "q2"}
    assert run["q2"] == {}


def test_bm25_is_deterministic():
    r = BM25Retriever()
    r.index(chunks())
    assert r.search({"q1": "cat purr"}, top_k=4) == r.search({"q1": "cat purr"}, top_k=4)


def test_stemming_matches_morphological_variants():
    # BEIR's published BM25 numbers use Anserini with stemming; without it our
    # sparse baseline would be weaker than the literature's for a reason
    # unrelated to the comparison being made.
    r = BM25Retriever(stemmer="english")
    r.index(chunks())
    assert r.search({"q1": "purring"}, top_k=4)["q1"], "stemmer should match 'purred'/'purr'"


def test_config_records_the_beir_parameters():
    cfg = BM25Retriever().config()
    # Anserini's BEIR defaults, not bm25s's library defaults of 1.5 / 0.75.
    assert (cfg["k1"], cfg["b"]) == (0.9, 0.4)


def test_empty_query_set_returns_an_empty_run():
    r = BM25Retriever()
    r.index(chunks())
    assert r.search({}, top_k=5) == {}
