"""Confidence intervals and paired significance tests.

PLAN.md §5 rule 1: nothing is compared without a confidence interval. Retrieval
metrics on 300–650 queries are noisy, and a 1.5-point difference is usually
nothing. This module is what makes that rule enforceable rather than aspirational.

Two procedures, both resampling **over queries**, because queries are the unit
of independent observation — documents within a query are not independent, and
resampling them would understate the variance badly:

- :func:`bootstrap_ci` — a percentile bootstrap CI for a single condition.
- :func:`paired_bootstrap` — for A vs B on the *same* queries. Pairing matters:
  query difficulty varies enormously (some queries every retriever gets right),
  and comparing two independent CIs throws that shared variance away. Two
  overlapping marginal CIs are routinely compatible with a difference that is
  clearly non-zero when paired, so the overlap test is not just weaker, it is
  wrong in a direction that makes you miss real effects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Resamples. PLAN.md specifies 10K; enough that Monte-Carlo error on a 95%
#: percentile bound is small relative to the effects we care about.
DEFAULT_RESAMPLES = 10_000


@dataclass(frozen=True, slots=True)
class Estimate:
    """A point estimate with a bootstrap confidence interval."""

    mean: float
    lo: float
    hi: float
    n: int
    confidence: float = 0.95

    def __str__(self) -> str:
        return f"{self.mean:.4f} [{self.lo:.4f}, {self.hi:.4f}]"

    def as_pct(self) -> str:
        """Rendered in percentage points, which is how the writeup reports."""
        return f"{self.mean * 100:.1f} [{self.lo * 100:.1f}, {self.hi * 100:.1f}]"


@dataclass(frozen=True, slots=True)
class Comparison:
    """A paired A-vs-B difference with a CI and a two-sided p-value."""

    mean_diff: float
    lo: float
    hi: float
    p_value: float
    n: int
    confidence: float = 0.95

    @property
    def significant(self) -> bool:
        """Whether the CI for the difference excludes zero."""
        return self.lo > 0.0 or self.hi < 0.0

    def __str__(self) -> str:
        star = "" if self.significant else " (n.s.)"
        return f"{self.mean_diff:+.4f} [{self.lo:+.4f}, {self.hi:+.4f}] p={self.p_value:.4f}{star}"


def _as_array(scores: dict[str, float] | list[float] | np.ndarray) -> np.ndarray:
    if isinstance(scores, dict):
        # Sort by query id so the array order — and therefore every seeded
        # resample drawn from it — is reproducible across processes.
        return np.array([scores[q] for q in sorted(scores)], dtype=np.float64)
    return np.asarray(scores, dtype=np.float64)


def bootstrap_ci(
    scores: dict[str, float] | list[float] | np.ndarray,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    seed: int = 0,
) -> Estimate:
    """Percentile bootstrap CI for the mean of per-query scores.

    Args:
        scores: per-query metric values. A dict is ordered by query id first.
        n_resamples: bootstrap replicates.
        confidence: two-sided coverage, e.g. 0.95.
        seed: fixed, per PLAN.md §5 rule 3. Two runs of the same analysis must
            produce the same interval, or the interval is itself a source of
            noise in the comparison.
    """
    values = _as_array(scores)
    n = len(values)
    if n == 0:
        return Estimate(mean=0.0, lo=0.0, hi=0.0, n=0, confidence=confidence)
    if n == 1:
        v = float(values[0])
        return Estimate(mean=v, lo=v, hi=v, n=1, confidence=confidence)

    rng = np.random.default_rng(seed)
    # Resample indices in one (n_resamples, n) draw; the memory is fine at our
    # scale (10k x 650 floats) and it is far faster than a Python loop.
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = values[idx].mean(axis=1)

    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return Estimate(
        mean=float(values.mean()), lo=float(lo), hi=float(hi), n=n, confidence=confidence
    )


def paired_bootstrap(
    a: dict[str, float] | list[float] | np.ndarray,
    b: dict[str, float] | list[float] | np.ndarray,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    seed: int = 0,
) -> Comparison:
    """Paired bootstrap for ``mean(a) - mean(b)`` over shared queries.

    Both inputs must cover the same queries; if they are dicts, the intersection
    is used and both are ordered by query id, so the pairing is by query rather
    than by position.

    The p-value is a two-sided bootstrap test of H0: mean difference is zero,
    computed by centring the resampled differences on zero and asking how often
    a centred replicate is at least as extreme as the observed difference. The
    ``(count + 1) / (n + 1)`` form keeps it strictly positive — reporting
    ``p = 0`` from a finite number of resamples would overstate what 10,000
    replicates can resolve.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        shared = sorted(set(a) & set(b))
        if not shared:
            raise ValueError("paired_bootstrap: a and b share no query ids")
        arr_a = np.array([a[q] for q in shared], dtype=np.float64)
        arr_b = np.array([b[q] for q in shared], dtype=np.float64)
    else:
        arr_a, arr_b = _as_array(a), _as_array(b)
        if len(arr_a) != len(arr_b):
            raise ValueError(
                f"paired_bootstrap needs equal-length inputs, got {len(arr_a)} and {len(arr_b)}"
            )

    diffs = arr_a - arr_b
    n = len(diffs)
    observed = float(diffs.mean())
    if n < 2:
        return Comparison(
            mean_diff=observed, lo=observed, hi=observed, p_value=1.0, n=n, confidence=confidence
        )

    rng = np.random.default_rng(seed)
    # One resample of query *indices*, applied to the paired differences. This
    # is what makes it paired: query q contributes (a_q - b_q) as a unit.
    idx = rng.integers(0, n, size=(n_resamples, n))
    resampled = diffs[idx].mean(axis=1)

    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(resampled, [alpha, 1.0 - alpha])

    centred = resampled - observed
    extreme = int(np.sum(np.abs(centred) >= abs(observed)))
    p_value = (extreme + 1) / (n_resamples + 1)

    return Comparison(
        mean_diff=observed,
        lo=float(lo),
        hi=float(hi),
        p_value=float(p_value),
        n=n,
        confidence=confidence,
    )


def unpaired_bootstrap(
    a: dict[str, float] | list[float] | np.ndarray,
    b: dict[str, float] | list[float] | np.ndarray,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    seed: int = 0,
) -> Comparison:
    """Two-sample bootstrap for ``mean(a) - mean(b)`` on *unrelated* groups.

    Required by the M3 contamination experiment. Synthetic and human-authored
    queries are different queries over the same corpus: there is no per-query
    correspondence between the conditions, so :func:`paired_bootstrap` does not
    apply — and with 500 synthetic against 300 human queries it would refuse the
    inputs anyway. Reaching for the paired test here because it is the one we
    already had would be a real statistical error, which is why the
    pre-registration names this distinction before any data existed.

    Each group is resampled independently to its own size, which is what makes
    the interval reflect the variance of two separate samples rather than of one
    set of differences. On genuinely paired data this interval is *wider* than
    the paired one, because it throws away the shared per-query difficulty — so
    use the paired test wherever pairing genuinely exists.
    """
    arr_a, arr_b = _as_array(a), _as_array(b)
    if len(arr_a) == 0 or len(arr_b) == 0:
        raise ValueError("unpaired_bootstrap: both groups must be non-empty")

    observed = float(arr_a.mean() - arr_b.mean())
    n_a, n_b = len(arr_a), len(arr_b)

    rng = np.random.default_rng(seed)
    means_a = arr_a[rng.integers(0, n_a, size=(n_resamples, n_a))].mean(axis=1)
    means_b = arr_b[rng.integers(0, n_b, size=(n_resamples, n_b))].mean(axis=1)
    resampled = means_a - means_b

    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(resampled, [alpha, 1.0 - alpha])

    centred = resampled - observed
    extreme = int(np.sum(np.abs(centred) >= abs(observed)))
    p_value = (extreme + 1) / (n_resamples + 1)

    return Comparison(
        mean_diff=observed,
        lo=float(lo),
        hi=float(hi),
        p_value=float(p_value),
        # Total observations across both groups; the groups differ in size, so a
        # single n is only meaningful as the combined count.
        n=n_a + n_b,
        confidence=confidence,
    )


def estimate_all(
    per_query: dict[str, dict[str, float]],
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Estimate]:
    """Bootstrap every measure in an :func:`reval.metrics.evaluate` result."""
    return {
        measure: bootstrap_ci(scores, n_resamples=n_resamples, confidence=confidence, seed=seed)
        for measure, scores in per_query.items()
    }
