"""The M3 headline experiment.

Scores one index against several eval conditions — human-authored queries, each
synthetic variant, and the synthetic set stratified by lexical-overlap decile —
and reports every number with a bootstrap confidence interval.

The design constraint that shapes this module: **the index is built once and
every condition is searched against it.** If each condition rebuilt its own
index, a difference between conditions could come from the index rather than the
queries, and the entire comparison would be uninterpretable. Holding the corpus,
chunker, retriever, aggregation policy and metric fixed is what makes "where the
queries came from" the only thing that varies.

Statistics follow `notes/preregistration.md` §3: between-condition contrasts are
**unpaired** (different queries, no correspondence), within-condition contrasts
are paired.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from reval.contamination import overlap
from reval.metrics import aggregate_to_documents, estimate_all, evaluate
from reval.metrics.bootstrap import Comparison, Estimate, unpaired_bootstrap
from reval.types import Dataset, Qrels, Queries


@dataclass(slots=True)
class Condition:
    """One eval set to be scored against a shared index."""

    name: str
    queries: Queries
    qrels: Qrels
    #: Free-form provenance carried into the report.
    meta: dict = field(default_factory=dict)


@dataclass(slots=True)
class ConditionScores:
    """Scored results for one condition."""

    name: str
    n_queries: int
    per_query: dict[str, dict[str, float]]
    estimates: dict[str, Estimate]
    overlap_means: dict[str, float] = field(default_factory=dict)
    mean_query_chars: float = 0.0


@dataclass(slots=True)
class DecileRow:
    """One overlap stratum of a synthetic condition."""

    decile: int
    n: int
    mean_overlap: float
    estimates: dict[str, Estimate]


@dataclass(slots=True)
class ContaminationReport:
    """Everything one (chunker, retriever) configuration produced."""

    dataset: str
    chunker: str
    retriever: str
    conditions: list[ConditionScores]
    deciles: dict[str, list[DecileRow]] = field(default_factory=dict)
    contrasts: dict[str, Comparison] = field(default_factory=dict)
    #: Spearman correlation between per-query overlap and per-query score,
    #: with a bootstrap CI. The H2 test.
    overlap_correlation: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    n_chunks: int = 0

    def condition(self, name: str) -> ConditionScores | None:
        return next((c for c in self.conditions if c.name == name), None)


def build_conditions(
    dataset: Dataset,
    synthetic_sets: dict[str, tuple[Queries, Qrels]],
) -> list[Condition]:
    """Human control plus one condition per synthetic variant."""
    conditions = [
        Condition(
            name="human",
            queries=dataset.queries,
            qrels=dataset.qrels,
            meta={"source": "BEIR human-authored", "split": dataset.split},
        )
    ]
    for name, (queries, qrels) in sorted(synthetic_sets.items()):
        conditions.append(
            Condition(name=name, queries=queries, qrels=qrels, meta={"source": "LLM-generated"})
        )
    return conditions


def spearman_with_ci(
    x: dict[str, float],
    y: dict[str, float],
    n_resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Spearman rho between two per-query series, with a bootstrap CI.

    Computed at the *query* level rather than over decile means: ten points
    would give a correlation with almost no power, and averaging first discards
    the within-stratum variation that the CI needs.
    """
    shared = sorted(set(x) & set(y))
    if len(shared) < 3:
        return 0.0, 0.0, 0.0
    xs = np.array([x[q] for q in shared])
    ys = np.array([y[q] for q in shared])

    def rho(a: np.ndarray, b: np.ndarray) -> float:
        # Spearman is Pearson on ranks; ties get average ranks.
        ra, rb = _rankdata(a), _rankdata(b)
        if ra.std() == 0 or rb.std() == 0:
            return 0.0
        return float(np.corrcoef(ra, rb)[0, 1])

    observed = rho(xs, ys)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(shared), size=(n_resamples, len(shared)))
    samples = np.array([rho(xs[i], ys[i]) for i in idx])
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return observed, float(lo), float(hi)


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks, matching scipy.stats.rankdata's default tie handling."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    # Average ranks within tied groups.
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    return ranks


def score_conditions(
    retriever,
    chunk_map: dict[str, str],
    conditions: list[Condition],
    dataset: Dataset,
    ks: list[int],
    top_k: int,
    aggregation: str = "max",
    tie_break: str = "trec",
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> list[ConditionScores]:
    """Search every condition against one already-built index and score it."""
    out: list[ConditionScores] = []
    for cond in conditions:
        chunk_run = retriever.search(cond.queries, top_k=top_k)
        doc_run = aggregate_to_documents(chunk_run, chunk_map, policy=aggregation)
        per_query = evaluate(doc_run, cond.qrels, ks=ks, tie_break=tie_break)

        overlap_scores = overlap.score_query_set(cond.queries, cond.qrels, dataset.corpus)
        overlap_means = {}
        if overlap_scores:
            overlap_means = {
                "jaccard_5gram": float(np.mean([s.jaccard for s in overlap_scores.values()])),
                "lcs_ratio": float(np.mean([s.lcs_ratio for s in overlap_scores.values()])),
                "token_containment": float(
                    np.mean([s.containment for s in overlap_scores.values()])
                ),
            }

        out.append(
            ConditionScores(
                name=cond.name,
                n_queries=len(cond.queries),
                per_query=per_query,
                estimates=estimate_all(
                    per_query, n_resamples=n_resamples, confidence=confidence, seed=seed
                ),
                overlap_means=overlap_means,
                mean_query_chars=(
                    float(np.mean([len(q) for q in cond.queries.values()])) if cond.queries else 0.0
                ),
            )
        )
    return out


def stratify_by_overlap(
    condition: Condition,
    scores: ConditionScores,
    dataset: Dataset,
    n_bins: int = 10,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> list[DecileRow]:
    """Split a condition's queries into overlap deciles and score each stratum.

    Decile boundaries come from this condition's own pooled distribution, as
    pre-registered — not chosen to produce a clean gradient.
    """
    overlap_scores = overlap.score_query_set(condition.queries, condition.qrels, dataset.corpus)
    if not overlap_scores:
        return []
    deciles = overlap.assign_deciles(overlap_scores, n_bins=n_bins)

    rows: list[DecileRow] = []
    for d in range(n_bins):
        members = [q for q, b in deciles.items() if b == d]
        if not members:
            continue
        subset = {
            measure: {q: vals[q] for q in members if q in vals}
            for measure, vals in scores.per_query.items()
        }
        rows.append(
            DecileRow(
                decile=d,
                n=len(members),
                mean_overlap=float(np.mean([overlap_scores[q].primary for q in members])),
                estimates=estimate_all(subset, n_resamples=n_resamples, seed=seed),
            )
        )
    return rows


def contrast(
    a: ConditionScores,
    b: ConditionScores,
    measure: str,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Comparison:
    """Between-condition difference: **unpaired**, per the pre-registration.

    Synthetic and human queries are different queries. There is no per-query
    correspondence, so the paired bootstrap does not apply here even though it
    is the one M1 built and the one already imported everywhere else.
    """
    return unpaired_bootstrap(
        a.per_query[measure],
        b.per_query[measure],
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed,
    )
