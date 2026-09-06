"""Chunk -> document aggregation must be explicit and correct."""

from __future__ import annotations

import pytest

from reval.metrics.aggregation import (
    aggregate_to_documents,
    count_distinct_documents,
    deduplicate_by_document,
)

# Two documents; docA split into three chunks, docB into one.
CHUNK_TO_DOC = {"docA#0": "docA", "docA#1": "docA", "docA#2": "docA", "docB#0": "docB"}


def test_max_takes_the_best_chunk():
    run = {"q1": {"docA#0": 0.2, "docA#1": 0.9, "docB#0": 0.5}}
    assert aggregate_to_documents(run, CHUNK_TO_DOC, "max") == {"q1": {"docA": 0.9, "docB": 0.5}}


def test_sum_rewards_documents_with_more_retrieved_chunks():
    """Why `sum` must never be used to compare chunking strategies.

    docB has a single strong chunk; docA has three mediocre ones. Under `max`,
    docB wins, which is the honest answer to "which document best matches?".
    Under `sum`, docA wins purely by having been cut into more pieces — so a
    chunk-size sweep scored under `sum` would measure chunk count, not quality.
    That is the same class of bug this whole project is about.
    """
    run = {"q1": {"docA#0": 0.4, "docA#1": 0.4, "docA#2": 0.4, "docB#0": 0.9}}

    by_max = aggregate_to_documents(run, CHUNK_TO_DOC, "max")["q1"]
    assert by_max["docB"] > by_max["docA"]

    by_sum = aggregate_to_documents(run, CHUNK_TO_DOC, "sum")["q1"]
    assert by_sum["docA"] == pytest.approx(1.2)
    assert by_sum["docA"] > by_sum["docB"]


def test_mean_averages_over_retrieved_chunks_only():
    run = {"q1": {"docA#0": 0.2, "docA#1": 0.8, "docB#0": 0.5}}
    got = aggregate_to_documents(run, CHUNK_TO_DOC, "mean")["q1"]
    assert got["docA"] == pytest.approx(0.5)
    assert got["docB"] == pytest.approx(0.5)


def test_first_equals_max_for_rank_ordered_scores():
    # Documented as an assumption rather than a guarantee; this pins it.
    run = {"q1": {"docA#0": 0.2, "docA#1": 0.9, "docB#0": 0.5}}
    assert aggregate_to_documents(run, CHUNK_TO_DOC, "first") == aggregate_to_documents(
        run, CHUNK_TO_DOC, "max"
    )


def test_unmapped_chunk_is_a_loud_error():
    # Silently dropping it would lower recall with no visible cause.
    run = {"q1": {"ghost#0": 1.0}}
    with pytest.raises(KeyError, match="has no document mapping"):
        aggregate_to_documents(run, CHUNK_TO_DOC, "max")


def test_aggregation_covers_every_query():
    run = {"q1": {"docA#0": 1.0}, "q2": {}}
    got = aggregate_to_documents(run, CHUNK_TO_DOC, "max")
    assert set(got) == {"q1", "q2"}
    assert got["q2"] == {}


def test_deduplicate_keeps_one_chunk_per_document():
    run = {"q1": {"docA#0": 0.2, "docA#1": 0.9, "docA#2": 0.5, "docB#0": 0.7}}
    got = deduplicate_by_document(run, CHUNK_TO_DOC)["q1"]
    assert got == {"docA#1": 0.9, "docB#0": 0.7}


def test_deduplication_shrinks_the_candidate_pool():
    """The M3(c) measurement: top-k holds fewer distinct documents than it looks.

    Four chunks in the top 4, but only two distinct documents. A recall@4 scored
    on the raw run got four shots at two documents.
    """
    run = {"q1": {"docA#0": 0.9, "docA#1": 0.8, "docA#2": 0.7, "docB#0": 0.6}}
    assert count_distinct_documents(run, CHUNK_TO_DOC, k=4) == {"q1": 2}
    assert len(deduplicate_by_document(run, CHUNK_TO_DOC)["q1"]) == 2


def test_distinct_document_count_respects_k():
    run = {"q1": {"docA#0": 0.9, "docA#1": 0.8, "docB#0": 0.1}}
    assert count_distinct_documents(run, CHUNK_TO_DOC, k=2) == {"q1": 1}
    assert count_distinct_documents(run, CHUNK_TO_DOC, k=3) == {"q1": 2}
