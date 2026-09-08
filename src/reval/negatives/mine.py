"""Hard-negative mining — PLAN.md M4.

A hard negative is a document a retriever ranks highly for a query that is not
judged relevant to it. Mining them is routine; the trap PLAN.md names is not.

**The trap.** BEIR's judgments are sparse — SciFact averages 1.1 relevant
documents per query out of 5,183. A top-ranked document with no judgment is not
"known non-relevant", it is *unjudged*, and the retriever ranked it first for a
reason. So mining top-ranked unjudged documents as negatives selects
**precisely** for the documents most likely to be unlabelled positives. The
harder you mine, the more false negatives you collect. That is the opposite of
what mining is supposed to do.

The mitigation is a cross-encoder pass: score each mined candidate, and flag the
ones that look as relevant as a genuine judged positive. The fraction flagged is
itself a result — it measures how contaminated a naive hard-negative set would
be, which is the same class of measurement error as the M3 headline.

In an evaluation harness (not a training one — PLAN.md §6 puts fine-tuning out
of scope) hard negatives serve two purposes: building a difficulty-stratified
eval subset, and diagnosing *what kind* of document a retriever confuses with a
correct answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from reval.metrics.ranking import rank_docs
from reval.types import Corpus, DocId, Qrels, Queries, QueryId, Run

MiningMethod = Literal["bm25", "dense", "filtered"]


@dataclass(frozen=True, slots=True)
class HardNegative:
    """One mined candidate, with everything needed to judge whether it is real."""

    query_id: QueryId
    doc_id: DocId
    rank: int
    retrieval_score: float
    cross_encoder_score: float | None = None
    #: True when the cross-encoder thinks this is a relevant document nobody
    #: judged — i.e. a suspected false negative, not a usable hard negative.
    suspected_positive: bool = False

    @property
    def usable(self) -> bool:
        return not self.suspected_positive


@dataclass(slots=True)
class MiningReport:
    """What a mining run produced, and how much of it was suspect."""

    method: str
    negatives: dict[QueryId, list[HardNegative]] = field(default_factory=dict)
    threshold: float | None = None
    n_queries: int = 0

    @property
    def all_negatives(self) -> list[HardNegative]:
        return [n for v in self.negatives.values() for n in v]

    @property
    def n_candidates(self) -> int:
        return len(self.all_negatives)

    @property
    def n_suspected(self) -> int:
        return sum(1 for n in self.all_negatives if n.suspected_positive)

    @property
    def suspected_fraction(self) -> float:
        """The headline M4 number: how much of a naive negative set is wrong."""
        return self.n_suspected / self.n_candidates if self.n_candidates else 0.0

    def usable_only(self) -> dict[QueryId, list[HardNegative]]:
        return {q: [n for n in v if n.usable] for q, v in self.negatives.items()}


def mine(
    run: Run,
    qrels: Qrels,
    depth: int = 50,
    per_query: int = 10,
    rel_threshold: int = 1,
    method: str = "bm25",
    tie_break: str = "trec",
) -> MiningReport:
    """Mine hard negatives: top-``depth`` retrieved, minus the known positives.

    Args:
        run: a document-level run from any retriever. The retriever it came
            from is what makes these negatives "BM25-hard" or "dense-hard".
        qrels: judgments. Only grades >= ``rel_threshold`` are excluded, so a
            grade-0 (judged non-relevant) document is a *legitimate* hard
            negative — a human confirmed it is wrong, which is precisely the
            signal we want.
        depth: how deep to look.
        per_query: how many negatives to keep per query.
        method: label recorded in the report.
    """
    negatives: dict[QueryId, list[HardNegative]] = {}
    for qid, scores in run.items():
        positives = {d for d, g in qrels.get(qid, {}).items() if g >= rel_threshold}
        kept: list[HardNegative] = []
        for rank, (doc_id, score) in enumerate(rank_docs(scores, k=depth, tie_break=tie_break), 1):
            if doc_id in positives:
                continue
            kept.append(HardNegative(query_id=qid, doc_id=doc_id, rank=rank, retrieval_score=score))
            if len(kept) >= per_query:
                break
        negatives[qid] = kept

    return MiningReport(method=method, negatives=negatives, n_queries=len(run))


def filter_false_negatives(
    report: MiningReport,
    queries: Queries,
    corpus: Corpus,
    scorer,
    threshold: float,
    show_progress: bool = False,
) -> MiningReport:
    """Flag mined candidates that look as relevant as a genuine positive.

    Does not delete them. A filtered negative and the reason it was filtered are
    both data — the raw-versus-filtered gap is the result PLAN.md asks for, and
    it cannot be computed from a set that has already been pruned.
    """
    candidates = report.all_negatives
    if not candidates:
        return report

    pairs = [
        (queries.get(n.query_id, ""), corpus[n.doc_id].full_text)
        for n in candidates
        if n.doc_id in corpus
    ]
    index = [i for i, n in enumerate(candidates) if n.doc_id in corpus]
    scores = scorer.score(pairs, show_progress=show_progress)

    by_position = dict(zip(index, scores, strict=True))
    rescored: dict[QueryId, list[HardNegative]] = {}
    position = 0
    for qid, items in report.negatives.items():
        out: list[HardNegative] = []
        for n in items:
            score = by_position.get(position)
            position += 1
            if score is None:
                out.append(n)
                continue
            out.append(
                HardNegative(
                    query_id=n.query_id,
                    doc_id=n.doc_id,
                    rank=n.rank,
                    retrieval_score=n.retrieval_score,
                    cross_encoder_score=float(score),
                    suspected_positive=bool(score >= threshold),
                )
            )
        rescored[qid] = out

    return MiningReport(
        method=f"{report.method}+filtered",
        negatives=rescored,
        threshold=threshold,
        n_queries=report.n_queries,
    )


def difficulty_stratified_queries(
    run: Run,
    qrels: Qrels,
    rel_threshold: int = 1,
    tie_break: str = "trec",
) -> dict[QueryId, int]:
    """Rank of the first relevant document per query, 0 if none retrieved.

    The difficulty signal for building a stratified eval subset: a query whose
    gold document sits at rank 1 is easy for this retriever; one whose gold sits
    at rank 40, behind 39 hard negatives, is not.
    """
    out: dict[QueryId, int] = {}
    for qid, scores in run.items():
        positives = {d for d, g in qrels.get(qid, {}).items() if g >= rel_threshold}
        first = 0
        for rank, (doc_id, _) in enumerate(rank_docs(scores, tie_break=tie_break), 1):
            if doc_id in positives:
                first = rank
                break
        out[qid] = first
    return out


def confusion_profile(
    report: MiningReport,
    queries: Queries,
    qrels: Qrels,
    corpus: Corpus,
    rel_threshold: int = 1,
) -> dict[str, float]:
    """Characterise what a retriever confuses with a correct answer.

    Compares the query's lexical overlap with its mined negatives against its
    overlap with its actual gold documents. If a retriever's hard negatives look
    *more* like the query than the gold documents do, it is being fooled by
    surface form — which is the same failure mode as the M3 result, seen from
    the retriever's side instead of the eval set's.
    """
    from reval.contamination.overlap import score_pair

    neg_overlap: list[float] = []
    pos_overlap: list[float] = []

    for qid, items in report.negatives.items():
        text = queries.get(qid)
        if not text:
            continue
        for n in items:
            if n.usable and n.doc_id in corpus:
                neg_overlap.append(score_pair(text, corpus[n.doc_id].full_text).containment)
        for doc_id, grade in qrels.get(qid, {}).items():
            if grade >= rel_threshold and doc_id in corpus:
                pos_overlap.append(score_pair(text, corpus[doc_id].full_text).containment)

    return {
        "mean_negative_containment": float(np.mean(neg_overlap)) if neg_overlap else 0.0,
        "mean_positive_containment": float(np.mean(pos_overlap)) if pos_overlap else 0.0,
        "gap": (
            float(np.mean(neg_overlap) - np.mean(pos_overlap))
            if neg_overlap and pos_overlap
            else 0.0
        ),
        "n_negatives": len(neg_overlap),
        "n_positives": len(pos_overlap),
    }
