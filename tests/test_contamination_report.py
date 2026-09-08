"""Verdict logic for the pre-registered hypotheses.

The verdicts are where a pre-registration either does its job or becomes
decoration. These pin the thresholds to what `notes/preregistration.md` §4
states, so that softening one requires editing a test that says what it is for.
"""

from __future__ import annotations

import json

import pytest

from reval.contamination.experiment import ConditionScores, ContaminationReport
from reval.contamination.rehydrate import load_reports
from reval.contamination.report import h1_verdict, kendall_tau, separated
from reval.metrics.bootstrap import Estimate


def est(mean: float, lo: float, hi: float) -> Estimate:
    return Estimate(mean=mean, lo=lo, hi=hi, n=100)


# ---- H1 thresholds ---------------------------------------------------------


def test_h1_confirmed_needs_five_points_and_significance():
    assert "CONFIRMED" in h1_verdict(19.16, significant=True)
    assert "CONFIRMED" in h1_verdict(5.0, significant=True)


def test_h1_between_two_and_five_points_is_weak_not_confirmed():
    verdict = h1_verdict(3.5, significant=True)
    assert "weak" in verdict
    assert "CONFIRMED" not in verdict


def test_h1_is_null_below_two_points_however_significant():
    """Statistical significance is not the criterion; effect size is too.

    With 500 queries a 1-point difference can clear p<0.05 and still be
    uninteresting. The pre-registration sets a magnitude floor for exactly this.
    """
    assert "NULL" in h1_verdict(1.5, significant=True)


def test_h1_is_null_when_the_ci_spans_zero():
    assert "NULL" in h1_verdict(19.0, significant=False)


def test_h1_reports_the_opposite_direction_rather_than_calling_it_null():
    # Pre-registered as a real possible outcome, not a bug.
    assert "OPPOSITE" in h1_verdict(-9.78, significant=True)


# ---- CI separation ---------------------------------------------------------


def test_separated_detects_disjoint_intervals():
    assert separated(est(0.90, 0.88, 0.93), est(0.70, 0.66, 0.74))


def test_touching_intervals_are_not_separated():
    """The real case from the run: 74.1 vs 74.1 exactly.

    Semantic chunking under synth-b/bm25 came within a hair of separating from
    the others. Treating a touching boundary as separation would have turned a
    null into a claimed finding.
    """
    assert not separated(est(0.778, 0.741, 0.815), est(0.701, 0.660, 0.741))


def test_overlapping_intervals_are_not_separated():
    assert not separated(est(0.806, 0.761, 0.849), est(0.799, 0.752, 0.842))


def test_separation_is_symmetric():
    a, b = est(0.9, 0.88, 0.93), est(0.7, 0.66, 0.74)
    assert separated(a, b) == separated(b, a)


# ---- Kendall tau -----------------------------------------------------------


def test_identical_orderings_give_tau_one():
    assert kendall_tau(["a", "b", "c"], ["a", "b", "c"]) == 1.0


def test_reversed_ordering_gives_tau_minus_one():
    assert kendall_tau(["a", "b", "c"], ["c", "b", "a"]) == -1.0


def test_one_swap_of_four_items():
    # 6 pairs, one discordant -> (5-1)/6
    assert kendall_tau(["a", "b", "c", "d"], ["b", "a", "c", "d"]) == pytest.approx(4 / 6)


def test_tau_of_a_single_item_is_one():
    assert kendall_tau(["a"], ["a"]) == 1.0


# ---- rehydration -----------------------------------------------------------


def test_saved_results_rehydrate(tmp_path):
    """A result you can only see by recomputing it is one nobody will check."""
    payload = {
        "config_hash": "abc",
        "config": {"name": "toy", "dataset": "scifact"},
        "git": {"sha": "deadbeef", "dirty": False},
        "reports": [
            {
                "chunker": "fixed",
                "retriever": "bm25",
                "n_chunks": 100,
                "conditions": [
                    {
                        "name": "human",
                        "n_queries": 300,
                        "mean_query_chars": 90.0,
                        "overlap_means": {"jaccard_5gram": 0.036},
                        "estimates": {
                            "recall@10": {"mean": 0.806, "lo": 0.761, "hi": 0.849, "n": 300}
                        },
                    }
                ],
                "contrasts": {
                    "synth-a": {
                        "mean_diff": 0.19,
                        "lo": 0.149,
                        "hi": 0.237,
                        "p_value": 0.0001,
                        "n": 792,
                    }
                },
                "overlap_correlation": {"synth-a": [0.077, 0.0, 0.134]},
                "deciles": {
                    "synth-a": [
                        {
                            "decile": 0,
                            "n": 50,
                            "mean_overlap": 0.05,
                            "estimates": {"recall@10": {"mean": 0.98, "lo": 0.94, "hi": 1.0}},
                        }
                    ]
                },
            }
        ],
    }
    path = tmp_path / "r.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    reports, back = load_reports(path)
    assert len(reports) == 1
    r = reports[0]
    assert isinstance(r, ContaminationReport)
    assert r.chunker == "fixed" and r.retriever == "bm25"
    assert r.condition("human").estimates["recall@10"].mean == pytest.approx(0.806)
    assert r.contrasts["synth-a"].mean_diff == pytest.approx(0.19)
    assert r.contrasts["synth-a"].significant
    assert r.deciles["synth-a"][0].n == 50
    assert back["config_hash"] == "abc"


def test_rehydrated_conditions_carry_no_per_query_vectors():
    """The intended boundary: rendering works, re-bootstrapping does not.

    Only aggregates are persisted, so anything wanting new inference has to go
    back to the raw run rather than silently resampling a summary.
    """
    c = ConditionScores(name="x", n_queries=1, per_query={}, estimates={})
    assert c.per_query == {}
