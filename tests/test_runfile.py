"""TREC run files must round-trip exactly, including score bits."""

from __future__ import annotations

import math

import pytest

from reval.runfile import (
    merge_runs,
    read_qrels,
    read_run,
    read_scored_docs,
    write_qrels,
    write_run,
)


@pytest.fixture
def run():
    return {
        "q1": {"d1": 0.9, "d2": 0.5, "d3": 0.5},
        "q2": {"d4": 1.25, "d1": -0.3},
    }


def test_trec_format_columns(tmp_path, run):
    path = write_run(run, tmp_path / "r.trec", run_name="baseline")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    first = lines[0].split()
    assert len(first) == 6
    qid, q0, doc_id, rank, _score, name = first
    assert (qid, q0, doc_id, rank, name) == ("q1", "Q0", "d1", "1", "baseline")
    # Ranks restart per query and are 1-based.
    q2_ranks = [ln.split()[3] for ln in lines if ln.startswith("q2 ")]
    assert q2_ranks == ["1", "2"]


def test_roundtrip_is_exact(tmp_path, run):
    path = write_run(run, tmp_path / "r.trec", run_name="baseline")
    assert read_run(path) == run


def test_roundtrip_preserves_float_precision(tmp_path):
    # %f formatting would quantise these to equality and silently change the
    # tie structure of the run, which changes every metric downstream.
    run = {"q1": {"a": 0.1 + 0.2, "b": 0.30000000000000004, "c": math.nextafter(0.3, 1.0)}}
    path = write_run(run, tmp_path / "r.trec", run_name="x")
    assert read_run(path) == run


def test_gzip_roundtrip(tmp_path, run):
    path = write_run(run, tmp_path / "r.trec.gz", run_name="baseline")
    assert path.suffix == ".gz"
    assert read_run(path) == run


def test_tied_docs_written_in_trec_order(tmp_path, run):
    path = write_run(run, tmp_path / "r.trec", run_name="x")
    docs = [sd.doc_id for sd in read_scored_docs(path) if sd.query_id == "q1"]
    # d2 and d3 tie at 0.5; trec policy puts the higher doc id first.
    assert docs == ["d1", "d3", "d2"]


def test_depth_limit(tmp_path):
    run = {"q1": {f"d{i}": float(100 - i) for i in range(100)}}
    path = write_run(run, tmp_path / "r.trec", run_name="x", k=10)
    assert len(read_scored_docs(path)) == 10


def test_malformed_line_is_loud(tmp_path):
    p = tmp_path / "bad.trec"
    p.write_text("q1 Q0 d1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected >=5 fields"):
        read_run(p)


def test_comments_and_blank_lines_are_skipped(tmp_path):
    p = tmp_path / "c.trec"
    p.write_text("# a comment\n\nq1 Q0 d1 1 1.0 x\n", encoding="utf-8")
    assert read_run(p) == {"q1": {"d1": 1.0}}


def test_qrels_roundtrip(tmp_path):
    qrels = {"q1": {"d1": 1, "d2": 0}, "q2": {"d3": 2}}
    path = write_qrels(qrels, tmp_path / "q.tsv")
    assert read_qrels(path) == qrels


def test_merge_runs_keeps_max_score():
    a = {"q1": {"d1": 0.2, "d2": 0.9}}
    b = {"q1": {"d1": 0.7}, "q2": {"d3": 0.1}}
    assert merge_runs([a, b]) == {"q1": {"d1": 0.7, "d2": 0.9}, "q2": {"d3": 0.1}}
