"""Shared fixtures.

The important one is :func:`make_adversarial_run`. A parity test against an
oracle only proves something if the runs it scores contain the cases where
implementations actually diverge. A run of distinct random floats over
fully-judged documents would pass against almost any buggy metric.
"""

from __future__ import annotations

import random

import pytest

from reval.corpora import beir
from reval.types import Qrels, Run

#: Cutoffs the M1 acceptance criterion names.
KS = [1, 5, 10, 20, 100]


def make_adversarial_run(qrels: Qrels, seed: int = 0, depth: int = 120) -> Run:
    """Build a run designed to break a sloppy metric implementation.

    Deliberately includes:

    - **Heavy score ties.** Scores are drawn from a small discrete grid, so
      documents tie constantly and tie-breaking decides what lands inside the
      cutoff. This is the realistic case: BM25 ties whenever two documents match
      the same query terms.
    - **Unjudged documents**, which every standard metric silently treats as
      non-relevant.
    - **Partial recall.** Some relevant documents are omitted entirely, so
      recall@k rarely saturates at 1.0.
    - **Queries with nothing relevant retrieved**, where MRR@k must be exactly 0
      rather than undefined.
    - **Short runs**, shorter than the largest cutoff, so k > |run| is exercised.
    - **Relevant documents ranked deep**, past k=10 but inside k=100, so the
      cutoffs actually differ from one another.
    """
    rng = random.Random(seed)
    run: Run = {}
    all_docs = sorted({d for rels in qrels.values() for d in rels})

    for i, (qid, rels) in enumerate(sorted(qrels.items())):
        scores: dict[str, float] = {}

        # Every 11th query retrieves nothing relevant at all.
        include_relevant = i % 11 != 0
        if include_relevant:
            relevant = sorted(d for d, g in rels.items() if g >= 1)
            # Drop roughly a third of the relevant documents.
            kept = [d for d in relevant if rng.random() > 0.33]
            for doc in kept:
                # Coarse grid => ties. Occasionally bury a relevant doc deep.
                scores[doc] = rng.choice([9.0, 8.0, 7.0, 2.0, 1.0])

        # Pad with distractors, some judged non-relevant, some unjudged.
        n_pad = rng.randint(5, depth)
        for j in range(n_pad):
            doc = rng.choice(all_docs) if rng.random() < 0.4 else f"unjudged-{i}-{j}"
            scores.setdefault(doc, rng.choice([8.0, 7.0, 5.0, 3.0, 1.0, 0.5]))

        run[qid] = scores
    return run


@pytest.fixture(scope="session")
def toy_qrels() -> Qrels:
    """A small graded qrels set, no corpus download required."""
    return {
        "q1": {"d1": 2, "d2": 1, "d3": 0, "d7": 1},
        "q2": {"d4": 1, "d5": 1},
        "q3": {"d6": 1},
        "q4": {"d1": 1, "d2": 2, "d3": 2, "d8": 1, "d9": 1},
        "q5": {"aaa": 1, "zzz": 0},
    }


@pytest.fixture(scope="session")
def toy_run(toy_qrels: Qrels) -> Run:
    return make_adversarial_run(toy_qrels, seed=7)


def load_dataset_or_skip(name: str):
    """Load a BEIR dataset, skipping the test if it has not been downloaded."""
    if not beir.is_downloaded(name):
        pytest.skip(f"{name} not downloaded; run: reval corpus download {name}")
    return beir.load(name, download_if_missing=False)
