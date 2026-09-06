"""Bootstrap CIs and the paired test.

These back PLAN.md §5 rule 1 ("nothing is compared without a confidence
interval"). If the interval is wrong, the rule is worse than useless — it
launders noise into a claim.
"""

from __future__ import annotations

import numpy as np
import pytest

from reval.metrics.bootstrap import bootstrap_ci, estimate_all, paired_bootstrap


def test_point_estimate_is_the_plain_mean():
    scores = [0.0, 0.5, 1.0, 0.25]
    assert bootstrap_ci(scores, n_resamples=1000).mean == pytest.approx(np.mean(scores))


def test_interval_brackets_the_mean():
    est = bootstrap_ci([0.1, 0.5, 0.9, 0.3, 0.7] * 20, n_resamples=2000)
    assert est.lo <= est.mean <= est.hi


def test_same_seed_reproduces_the_interval_exactly():
    # PLAN.md §5 rule 3. An interval that moves between runs is itself noise.
    scores = list(np.linspace(0, 1, 100))
    a = bootstrap_ci(scores, n_resamples=2000, seed=42)
    b = bootstrap_ci(scores, n_resamples=2000, seed=42)
    assert (a.lo, a.hi) == (b.lo, b.hi)


def test_different_seeds_give_close_but_distinct_intervals():
    scores = list(np.linspace(0, 1, 100))
    a = bootstrap_ci(scores, n_resamples=2000, seed=1)
    b = bootstrap_ci(scores, n_resamples=2000, seed=2)
    assert a.lo != b.lo
    assert abs(a.lo - b.lo) < 0.05


def test_dict_input_is_ordered_by_query_id_for_reproducibility():
    forward = {"q1": 0.1, "q2": 0.9, "q3": 0.5}
    shuffled = {"q3": 0.5, "q1": 0.1, "q2": 0.9}
    assert bootstrap_ci(forward, n_resamples=500, seed=3) == bootstrap_ci(
        shuffled, n_resamples=500, seed=3
    )


def test_zero_variance_gives_a_degenerate_interval():
    est = bootstrap_ci([0.5] * 30, n_resamples=500)
    assert est.lo == pytest.approx(0.5)
    assert est.hi == pytest.approx(0.5)


def test_wider_spread_gives_a_wider_interval():
    tight = bootstrap_ci([0.5, 0.51, 0.49] * 30, n_resamples=2000, seed=0)
    wide = bootstrap_ci([0.0, 1.0, 0.5] * 30, n_resamples=2000, seed=0)
    assert (wide.hi - wide.lo) > (tight.hi - tight.lo)


def test_more_queries_gives_a_narrower_interval():
    rng = np.random.default_rng(0)
    small = bootstrap_ci(list(rng.random(30)), n_resamples=2000, seed=0)
    large = bootstrap_ci(list(rng.random(600)), n_resamples=2000, seed=0)
    assert (large.hi - large.lo) < (small.hi - small.lo)


def test_empty_and_single_inputs_do_not_crash():
    assert bootstrap_ci([]).n == 0
    single = bootstrap_ci([0.7])
    assert (single.mean, single.lo, single.hi) == (0.7, 0.7, 0.7)


# ---- paired comparison ----------------------------------------------------


def test_paired_difference_is_the_mean_of_differences():
    a = {"q1": 0.9, "q2": 0.5}
    b = {"q1": 0.6, "q2": 0.4}
    cmp = paired_bootstrap(a, b, n_resamples=1000)
    assert cmp.mean_diff == pytest.approx(0.2)


def test_pairing_is_by_query_id_not_position():
    a = {"q1": 1.0, "q2": 0.0}
    b = {"q2": 0.0, "q1": 1.0}
    assert paired_bootstrap(a, b, n_resamples=500).mean_diff == pytest.approx(0.0)


def test_identical_systems_are_not_significant():
    scores = {f"q{i}": float(i % 5) / 4 for i in range(200)}
    cmp = paired_bootstrap(scores, scores, n_resamples=2000)
    assert cmp.mean_diff == pytest.approx(0.0)
    assert not cmp.significant
    assert cmp.p_value == pytest.approx(1.0)


def test_a_consistent_uniform_improvement_is_significant():
    a = {f"q{i}": 0.5 + 0.1 for i in range(200)}
    b = {f"q{i}": 0.5 for i in range(200)}
    cmp = paired_bootstrap(a, b, n_resamples=2000)
    assert cmp.significant
    assert cmp.lo > 0


def test_pairing_detects_what_marginal_intervals_would_miss():
    """Why the paired test, not overlapping CIs.

    Query difficulty dominates the variance: some queries every system gets
    right, some none do. System A beats B on every single query by a small,
    consistent margin. The marginal intervals overlap heavily because they both
    span the whole difficulty range — but the paired difference is unambiguous.
    Comparing marginal CIs here would report "no difference" for a system that
    is better on 100% of queries.
    """
    rng = np.random.default_rng(0)
    difficulty = rng.random(300)
    b = {f"q{i}": float(d) for i, d in enumerate(difficulty)}
    a = {f"q{i}": float(min(d + 0.03, 1.0)) for i, d in enumerate(difficulty)}

    ci_a = bootstrap_ci(a, n_resamples=2000, seed=0)
    ci_b = bootstrap_ci(b, n_resamples=2000, seed=0)
    overlap = ci_a.lo < ci_b.hi and ci_b.lo < ci_a.hi
    assert overlap, "fixture no longer demonstrates the overlap trap"

    cmp = paired_bootstrap(a, b, n_resamples=2000, seed=0)
    assert cmp.significant
    assert cmp.mean_diff > 0


def test_p_value_is_never_exactly_zero():
    # (count + 1) / (n + 1): 10,000 resamples cannot resolve p = 0.
    a = {f"q{i}": 1.0 for i in range(100)}
    b = {f"q{i}": 0.0 for i in range(100)}
    cmp = paired_bootstrap(a, b, n_resamples=1000)
    assert 0 < cmp.p_value <= 1 / 1001 + 1e-12


def test_direction_flips_sign_symmetrically():
    a = {"q1": 0.9, "q2": 0.8, "q3": 0.7}
    b = {"q1": 0.5, "q2": 0.4, "q3": 0.3}
    fwd = paired_bootstrap(a, b, n_resamples=1000, seed=0)
    rev = paired_bootstrap(b, a, n_resamples=1000, seed=0)
    assert fwd.mean_diff == pytest.approx(-rev.mean_diff)


def test_disjoint_query_sets_are_rejected():
    with pytest.raises(ValueError, match="share no query ids"):
        paired_bootstrap({"q1": 1.0}, {"q2": 1.0})


def test_mismatched_array_lengths_are_rejected():
    with pytest.raises(ValueError, match="equal-length"):
        paired_bootstrap([1.0, 2.0], [1.0])


def test_estimate_all_covers_every_measure():
    per_query = {"recall@10": {"q1": 1.0, "q2": 0.0}, "ndcg@10": {"q1": 0.5, "q2": 0.5}}
    got = estimate_all(per_query, n_resamples=500)
    assert set(got) == {"recall@10", "ndcg@10"}
    assert got["ndcg@10"].mean == pytest.approx(0.5)


def test_estimate_renders_in_percentage_points():
    assert bootstrap_ci([0.5] * 10, n_resamples=200).as_pct().startswith("50.0 [50.0, 50.0]")
