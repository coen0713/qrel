"""Metrics, ranking policy, and bootstrap inference.

Validated against ``pytrec_eval`` to 1e-6 by ``tests/test_metrics_parity.py``,
which CI enforces. That test is the reason anything else in this repo is worth
reading.
"""

from __future__ import annotations

from reval.metrics.aggregation import (
    AggregationPolicy,
    aggregate_to_documents,
    count_distinct_documents,
    deduplicate_by_document,
)
from reval.metrics.bootstrap import (
    Comparison,
    Estimate,
    bootstrap_ci,
    estimate_all,
    paired_bootstrap,
    unpaired_bootstrap,
)
from reval.metrics.core import (
    dcg_at_k,
    evaluate,
    mean_scores,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from reval.metrics.ranking import TieBreak, rank_docs, rank_run, truncate_run

__all__ = [
    "AggregationPolicy",
    "Comparison",
    "Estimate",
    "TieBreak",
    "aggregate_to_documents",
    "bootstrap_ci",
    "count_distinct_documents",
    "dcg_at_k",
    "deduplicate_by_document",
    "estimate_all",
    "evaluate",
    "mean_scores",
    "ndcg_at_k",
    "paired_bootstrap",
    "rank_docs",
    "rank_run",
    "recall_at_k",
    "reciprocal_rank_at_k",
    "truncate_run",
    "unpaired_bootstrap",
]
