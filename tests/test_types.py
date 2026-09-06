"""The canonical types are the waist of the hourglass; their edge cases matter."""

from __future__ import annotations

from reval.types import Dataset, Document


def test_full_text_joins_title_and_body():
    d = Document("d1", "A Title", "Some body text.")
    assert d.full_text == "A Title Some body text."


def test_full_text_handles_missing_title():
    # BEIR corpora contain title-less documents (FiQA is entirely title-less).
    # A naive f"{title} {text}" would prepend a space and shift every offset
    # our chunkers record.
    assert Document("d1", "", "body").full_text == "body"
    assert Document("d1", "   ", "body").full_text == "body"


def test_full_text_handles_missing_body():
    assert Document("d1", "Title", "").full_text == "Title"


def test_dataset_drops_queries_with_no_judgments():
    # Averaging a metric over unjudged queries silently dilutes it toward zero.
    ds = Dataset(
        name="toy",
        corpus={"d1": Document("d1", "", "x")},
        queries={"q1": "has judgments", "q2": "has none"},
        qrels={"q1": {"d1": 1}},
    )
    assert set(ds.queries) == {"q1"}
    assert ds.n_queries == 1


def test_dataset_drops_queries_whose_judgment_set_is_empty():
    ds = Dataset(
        name="toy",
        corpus={},
        queries={"q1": "a", "q2": "b"},
        qrels={"q1": {"d1": 1}, "q2": {}},
    )
    assert set(ds.qrels) == {"q1"}


def test_relevant_respects_the_grade_threshold():
    ds = Dataset(
        name="toy",
        corpus={},
        queries={"q1": "a"},
        qrels={"q1": {"d1": 0, "d2": 1, "d3": 2}},
    )
    # Grade 0 is an explicit "judged non-relevant" and must not count.
    assert ds.relevant("q1") == {"d2", "d3"}
    assert ds.relevant("q1", threshold=2) == {"d3"}


def test_relevant_of_unknown_query_is_empty():
    ds = Dataset(name="toy", corpus={}, queries={}, qrels={})
    assert ds.relevant("nope") == set()
