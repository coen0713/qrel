"""The M3 experiment machinery.

The statistics here decide what the writeup claims, so the properties that
matter are the ones that would produce a plausible-but-wrong headline: using the
wrong bootstrap between conditions, stratifying on values instead of ranks, or
letting the index differ between conditions.
"""

from __future__ import annotations

import numpy as np
import pytest

from reval.contamination import experiment as exp
from reval.contamination.experiment import Condition, ConditionScores
from reval.metrics.bootstrap import Estimate
from reval.types import Dataset, Document


def make_dataset() -> Dataset:
    corpus = {
        f"d{i}": Document(f"d{i}", "", f"passage {i} about topic {i % 5} with filler words")
        for i in range(20)
    }
    return Dataset(
        name="toy",
        corpus=corpus,
        queries={f"h{i}": f"human query {i}" for i in range(5)},
        qrels={f"h{i}": {f"d{i}": 1} for i in range(5)},
    )


def scores_from(per_query: dict[str, float], name: str) -> ConditionScores:
    return ConditionScores(
        name=name,
        n_queries=len(per_query),
        per_query={"recall@10": per_query},
        estimates={
            "recall@10": Estimate(
                mean=float(np.mean(list(per_query.values()))), lo=0.0, hi=1.0, n=len(per_query)
            )
        },
    )


# ---- conditions ------------------------------------------------------------


def test_human_is_always_the_first_condition():
    ds = make_dataset()
    conds = exp.build_conditions(ds, {"synth-a": ({"s1": "q"}, {"s1": {"d1": 1}})})
    assert conds[0].name == "human"
    assert conds[0].queries == ds.queries


def test_every_synthetic_set_becomes_a_condition():
    ds = make_dataset()
    conds = exp.build_conditions(
        ds,
        {
            "synth-a": ({"a1": "q"}, {"a1": {"d1": 1}}),
            "synth-b": ({"b1": "q"}, {"b1": {"d2": 1}}),
        },
    )
    assert [c.name for c in conds] == ["human", "synth-a", "synth-b"]


def test_conditions_are_ordered_deterministically():
    ds = make_dataset()
    sets = {"synth-b": ({"b": "q"}, {"b": {"d1": 1}}), "synth-a": ({"a": "q"}, {"a": {"d1": 1}})}
    assert [c.name for c in exp.build_conditions(ds, sets)] == ["human", "synth-a", "synth-b"]


# ---- the contrast uses the UNPAIRED bootstrap ------------------------------


def test_contrast_handles_different_sized_conditions():
    """The reason the unpaired bootstrap exists.

    SciFact has 300 human queries against ~492 synthetic ones. A paired test
    would raise on the length mismatch; worse, if the sizes ever coincided it
    would silently pair unrelated queries by position.
    """
    a = scores_from({f"s{i}": 1.0 for i in range(40)}, "synth")
    b = scores_from({f"h{i}": 0.5 for i in range(25)}, "human")
    cmp = exp.contrast(a, b, "recall@10", n_resamples=1000)
    assert cmp.mean_diff == pytest.approx(0.5)
    assert cmp.n == 65


def test_contrast_reports_no_difference_between_equivalent_conditions():
    rng = np.random.default_rng(0)
    a = scores_from({f"s{i}": float(v) for i, v in enumerate(rng.random(200))}, "synth")
    b = scores_from({f"h{i}": float(v) for i, v in enumerate(rng.random(200))}, "human")
    assert not exp.contrast(a, b, "recall@10", n_resamples=2000, seed=0).significant


def test_contrast_is_reproducible():
    a = scores_from({f"s{i}": i / 50 for i in range(50)}, "synth")
    b = scores_from({f"h{i}": i / 40 for i in range(40)}, "human")
    first = exp.contrast(a, b, "recall@10", n_resamples=1000, seed=3)
    second = exp.contrast(a, b, "recall@10", n_resamples=1000, seed=3)
    assert (first.mean_diff, first.lo, first.hi) == (second.mean_diff, second.lo, second.hi)


# ---- Spearman with a bootstrap CI ------------------------------------------


def test_perfect_monotone_relationship_gives_rho_one():
    x = {f"q{i}": float(i) for i in range(50)}
    y = {f"q{i}": float(i) * 3 + 1 for i in range(50)}
    rho, lo, hi = exp.spearman_with_ci(x, y, n_resamples=500)
    assert rho == pytest.approx(1.0)
    assert lo > 0.9


def test_spearman_is_rank_based_not_value_based():
    # A monotone but wildly non-linear transform must not change rho.
    x = {f"q{i}": float(i) for i in range(40)}
    linear = {f"q{i}": float(i) for i in range(40)}
    exponential = {f"q{i}": float(np.exp(i / 5)) for i in range(40)}
    a, *_ = exp.spearman_with_ci(x, linear, n_resamples=200)
    b, *_ = exp.spearman_with_ci(x, exponential, n_resamples=200)
    assert a == pytest.approx(b)


def test_inverse_relationship_gives_negative_rho():
    x = {f"q{i}": float(i) for i in range(40)}
    y = {f"q{i}": float(-i) for i in range(40)}
    rho, _, hi = exp.spearman_with_ci(x, y, n_resamples=500)
    assert rho == pytest.approx(-1.0)
    assert hi < -0.9


def test_unrelated_series_have_a_ci_spanning_zero():
    rng = np.random.default_rng(0)
    x = {f"q{i}": float(v) for i, v in enumerate(rng.random(300))}
    y = {f"q{i}": float(v) for i, v in enumerate(rng.random(300))}
    rho, lo, hi = exp.spearman_with_ci(x, y, n_resamples=1000, seed=0)
    assert lo < 0 < hi


def test_a_constant_series_gives_rho_zero_not_nan():
    """The saturation case, which really happens.

    On synth-a, recall@10 is 100.0 in nine of ten deciles. A correlation against
    a constant is undefined; returning NaN would propagate into the report as a
    blank cell rather than an interpretable zero.
    """
    x = {f"q{i}": float(i) for i in range(30)}
    y = {f"q{i}": 1.0 for i in range(30)}
    rho, lo, hi = exp.spearman_with_ci(x, y, n_resamples=200)
    assert rho == 0.0
    assert not np.isnan(lo) and not np.isnan(hi)


def test_spearman_uses_only_shared_queries():
    x = {"a": 1.0, "b": 2.0, "c": 3.0, "extra": 9.0}
    y = {"a": 1.0, "b": 2.0, "c": 3.0}
    assert exp.spearman_with_ci(x, y, n_resamples=200)[0] == pytest.approx(1.0)


def test_too_few_points_returns_zeros_rather_than_raising():
    assert exp.spearman_with_ci({"a": 1.0}, {"a": 1.0}) == (0.0, 0.0, 0.0)


def test_rankdata_averages_ties():
    # Matches scipy.stats.rankdata's default: [1, 2.5, 2.5, 4].
    assert list(exp._rankdata(np.array([10.0, 20.0, 20.0, 30.0]))) == [1.0, 2.5, 2.5, 4.0]


# ---- stratification --------------------------------------------------------


def test_strata_partition_the_queries():
    ds = make_dataset()
    queries = {f"s{i}": f"passage {i} about topic" for i in range(20)}
    qrels = {f"s{i}": {f"d{i}": 1} for i in range(20)}
    cond = Condition(name="synth", queries=queries, qrels=qrels)
    scores = scores_from({f"s{i}": float(i % 2) for i in range(20)}, "synth")

    rows = exp.stratify_by_overlap(cond, scores, ds, n_bins=4, n_resamples=200)
    assert sum(r.n for r in rows) == 20
    assert [r.decile for r in rows] == sorted(r.decile for r in rows)


def test_strata_are_ordered_by_increasing_overlap():
    ds = make_dataset()
    # Query 0 copies its passage verbatim; later ones share less and less.
    queries = {
        "s0": ds.corpus["d0"].full_text,
        "s1": "passage 1 about topic",
        "s2": "about topic",
        "s3": "unrelated financial content entirely",
    }
    qrels = {f"s{i}": {f"d{i}": 1} for i in range(4)}
    cond = Condition(name="synth", queries=queries, qrels=qrels)
    scores = scores_from({f"s{i}": 1.0 for i in range(4)}, "synth")

    rows = exp.stratify_by_overlap(cond, scores, ds, n_bins=4, n_resamples=100)
    overlaps = [r.mean_overlap for r in rows]
    assert overlaps == sorted(overlaps), "strata must be ordered by increasing overlap"


def test_stratifying_an_unscorable_condition_returns_nothing():
    ds = make_dataset()
    cond = Condition(name="synth", queries={"s0": "q"}, qrels={"s0": {}})
    assert exp.stratify_by_overlap(cond, scores_from({"s0": 1.0}, "synth"), ds) == []
