"""M0 acceptance, as a test: our loaders must reproduce the BEIR paper.

Marked ``slow``/``network`` because it needs the corpora on disk. Run the fast
suite with ``pytest -m "not slow and not network"``.
"""

from __future__ import annotations

import json

import pytest

from reval.corpora import beir

DATASETS = beir.dataset_names()


def test_registry_covers_exactly_the_three_planned_datasets():
    # PLAN.md §6: more than three datasets is out of scope. This test is the
    # scope contract, not a triviality.
    assert DATASETS == ["fiqa", "nfcorpus", "scifact"]


def test_unknown_dataset_names_are_rejected():
    with pytest.raises(KeyError, match="unknown dataset"):
        beir.spec("msmarco")


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.parametrize("name", DATASETS)
def test_stats_match_published_beir_numbers(name):
    ds = beir.load(name)
    st = beir.stats(ds)
    assert st.n_docs == st.expected_n_docs
    assert st.n_queries == st.expected_n_queries
    assert round(st.mean_relevant_per_query, 1) == pytest.approx(
        st.expected_avg_rel_per_query, abs=0.05
    )
    assert st.all_match


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.parametrize("name", DATASETS)
def test_every_judged_doc_exists_in_the_corpus(name):
    # A qrel pointing at a missing doc means an unreachable relevant document,
    # which caps recall below 1.0 for reasons unrelated to the retriever.
    ds = beir.load(name)
    missing = {
        doc_id
        for rels in ds.qrels.values()
        for doc_id, grade in rels.items()
        if grade >= 1 and doc_id not in ds.corpus
    }
    assert not missing, f"{len(missing)} relevant doc ids absent from the corpus"


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.parametrize("name", DATASETS)
def test_every_judged_query_has_text(name):
    ds = beir.load(name)
    assert all(ds.queries.get(qid, "").strip() for qid in ds.qrels)


@pytest.mark.slow
@pytest.mark.network
def test_checksums_are_recorded_for_the_manifest(name="scifact"):
    ds = beir.load(name)
    assert set(ds.checksums) == {"corpus.jsonl", "queries.jsonl", "qrels/test.tsv"}
    assert all(len(v) == 64 for v in ds.checksums.values())


@pytest.mark.slow
@pytest.mark.network
def test_loader_agrees_with_a_naive_line_count(name="scifact"):
    # Independent path to the same number: if our jsonl parser silently dropped
    # a malformed record, this catches it.
    path = beir.dataset_path(name) / "corpus.jsonl"
    with open(path, encoding="utf-8") as fh:
        raw_ids = {json.loads(line)["_id"] for line in fh if line.strip()}
    assert len(raw_ids) == beir.load(name).n_docs
