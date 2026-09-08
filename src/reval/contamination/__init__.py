"""Contamination detection — the core module (PLAN.md M3).

Four leak types. Detectors for the first three; the fourth (encoder pretraining
contamination) is unfixable at this scale and is named in the writeup's
threats-to-validity section instead of pretended away.
"""

from __future__ import annotations

from reval.contamination.nearduplicate import (
    DuplicatePair,
    MinHasher,
    degenerate_documents,
    duplicate_clusters,
    exact_jaccard,
    find_near_duplicates,
    shingles,
)
from reval.contamination.overlap import (
    OverlapScores,
    assign_deciles,
    char_ngram_jaccard,
    decile_boundaries,
    normalized_lcs,
    score_pair,
    score_query_set,
    token_containment,
)

__all__ = [
    "DuplicatePair",
    "MinHasher",
    "OverlapScores",
    "assign_deciles",
    "char_ngram_jaccard",
    "decile_boundaries",
    "degenerate_documents",
    "duplicate_clusters",
    "exact_jaccard",
    "find_near_duplicates",
    "normalized_lcs",
    "score_pair",
    "score_query_set",
    "shingles",
    "token_containment",
]


def __getattr__(name: str):
    # Generation pulls in the anthropic SDK; importing this package must not
    # require the [llm] extra when only the detectors are needed.
    if name in {"QueryGenerator", "SyntheticQuerySet", "sample_passages", "load_prompt"}:
        from reval.contamination import generate

        return getattr(generate, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
