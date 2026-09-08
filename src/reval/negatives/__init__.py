"""Hard-negative mining and false-negative filtering (PLAN.md M4).

In an evaluation harness — not a training one, PLAN.md §6 puts fine-tuning out
of scope — hard negatives build difficulty-stratified eval subsets and diagnose
what a retriever confuses with a correct answer.
"""

from __future__ import annotations

from reval.negatives.mine import (
    HardNegative,
    MiningReport,
    confusion_profile,
    difficulty_stratified_queries,
    filter_false_negatives,
    mine,
)

__all__ = [
    "HardNegative",
    "MiningReport",
    "confusion_profile",
    "difficulty_stratified_queries",
    "filter_false_negatives",
    "mine",
]


def __getattr__(name: str):
    # The cross-encoder pulls in torch; importing this package must not require
    # the [dense] extra when only the mining logic is needed.
    if name in {"CrossEncoderScorer", "calibrate_threshold", "DEFAULT_CROSS_ENCODER"}:
        from reval.negatives import crossencoder

        return getattr(crossencoder, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
