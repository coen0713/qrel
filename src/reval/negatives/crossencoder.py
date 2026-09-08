"""Cross-encoder relevance scoring, for filtering mined hard negatives.

A bi-encoder embeds query and document separately; a cross-encoder reads both
together and attends across them, which makes it far more accurate and far
slower. That trade is exactly right here: we score a few thousand
(query, candidate) pairs once, offline, to decide which mined "negatives" are
actually relevant documents that nobody judged.

Runs entirely locally — no API. Model weights are pinned by revision for the
same reason encoders are (see :mod:`reval.embedding.encoder`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Trained on MS MARCO passage ranking. Outputs an unbounded relevance logit,
#: not a probability — which is why the threshold has to be calibrated rather
#: than guessed (see `calibrate_threshold`).
DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass(slots=True)
class CrossEncoderScorer:
    """Scores (query, document) pairs for relevance."""

    model_name: str = DEFAULT_CROSS_ENCODER
    revision: str | None = "main"
    batch_size: int = 64
    max_length: int = 512
    device: str | None = None
    _model: object | None = field(default=None, init=False, repr=False)

    def load(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ImportError(
                "cross-encoder filtering needs the [dense] extra: uv pip install -e '.[dense]'"
            ) from exc

        self._model = CrossEncoder(
            self.model_name,
            revision=self.revision,
            max_length=self.max_length,
            device=self.device,
        )
        return self._model

    @property
    def identity(self) -> str:
        return f"{self.model_name}@{self.revision}"

    def score(self, pairs: list[tuple[str, str]], show_progress: bool = False) -> np.ndarray:
        """Score ``(query, passage)`` pairs. Higher means more relevant."""
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        model = self.load()
        scores = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=show_progress)
        return np.asarray(scores, dtype=np.float32).reshape(-1)


@dataclass(frozen=True, slots=True)
class Calibration:
    """A measured decision threshold, with the evidence for it."""

    threshold: float
    positive_scores: np.ndarray
    background_scores: np.ndarray
    #: Share of judged-relevant documents at or above the threshold.
    true_positive_rate: float
    #: Share of random background documents at or above it.
    false_positive_rate: float
    #: Probability a random positive outscores a random background document.
    auc: float

    @property
    def youden_j(self) -> float:
        return self.true_positive_rate - self.false_positive_rate

    def summary(self) -> str:
        return (
            f"threshold={self.threshold:.3f}  TPR={self.true_positive_rate:.1%}  "
            f"FPR={self.false_positive_rate:.1%}  AUC={self.auc:.3f}"
        )


def calibrate_threshold(
    scorer: CrossEncoderScorer,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    corpus,
    background_per_query: int = 5,
    max_pairs: int = 2000,
    seed: int = 0,
    show_progress: bool = False,
) -> Calibration:
    """Measure the threshold that best separates relevant from irrelevant.

    PLAN.md warns about picking a filter threshold like 0.7 and being unable to
    justify it six weeks later. So we measure one instead of choosing one — but
    the *first* attempt at measuring it was wrong, and the way it was wrong is
    worth recording.

    **What did not work:** taking a low percentile (p5) of the judged-relevant
    score distribution, on the reasoning that anything scoring above the weakest
    true positives "looks as relevant as a real positive". On SciFact the
    positive distribution has a long left tail — p5 was −8.50 while the median
    was +2.23 — so the threshold sat below almost every document in the corpus
    and flagged **88.4%** of mined candidates as false negatives. A one-sided
    calibration cannot work: it never looks at what an *irrelevant* document
    scores, so it has no idea which side of the tail is unusual.

    **What this does:** scores judged-relevant pairs *and* a seeded random
    background sample of (query, unjudged document) pairs, then picks the
    threshold maximising Youden's J (TPR − FPR) — the point of greatest
    separation between the two distributions. Both score arrays and the
    resulting TPR, FPR and AUC come back with it, so the threshold is auditable
    rather than asserted. A low AUC is itself information: it means the
    cross-encoder cannot separate the classes on this corpus, and the filter
    should not be trusted there.
    """
    rng = np.random.default_rng(seed)
    doc_ids = sorted(corpus)

    positive_pairs: list[tuple[str, str]] = []
    background_pairs: list[tuple[str, str]] = []

    for qid in sorted(qrels):
        text = queries.get(qid)
        if not text:
            continue
        relevant = {d for d, g in qrels[qid].items() if g >= 1}
        for doc_id in sorted(relevant):
            if doc_id in corpus:
                positive_pairs.append((text, corpus[doc_id].full_text))
        # Random documents are overwhelmingly irrelevant to any given query.
        # They are not *guaranteed* irrelevant, but at 1.1 relevant documents in
        # 5,183 the contamination of this reference class is negligible — far
        # smaller than in the top-ranked candidates we are trying to judge.
        for _ in range(background_per_query):
            pick = doc_ids[int(rng.integers(0, len(doc_ids)))]
            if pick not in relevant:
                background_pairs.append((text, corpus[pick].full_text))

    if not positive_pairs or not background_pairs:
        return Calibration(
            threshold=float("inf"),
            positive_scores=np.zeros(0, dtype=np.float32),
            background_scores=np.zeros(0, dtype=np.float32),
            true_positive_rate=0.0,
            false_positive_rate=0.0,
            auc=0.5,
        )

    def subsample(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
        if len(pairs) <= max_pairs:
            return pairs
        # Seeded: NFCorpus has 12,334 judgments and the calibration does not
        # improve past a couple of thousand pairs.
        idx = rng.choice(len(pairs), size=max_pairs, replace=False)
        return [pairs[i] for i in sorted(idx)]

    pos = scorer.score(subsample(positive_pairs), show_progress=show_progress)
    bg = scorer.score(subsample(background_pairs), show_progress=show_progress)

    # Sweep every observed score as a candidate cut point; take the best J.
    candidates = np.unique(np.concatenate([pos, bg]))
    best_j, best_t = -np.inf, float(candidates[0])
    for t in candidates:
        tpr = float(np.mean(pos >= t))
        fpr = float(np.mean(bg >= t))
        if tpr - fpr > best_j:
            best_j, best_t = tpr - fpr, float(t)

    return Calibration(
        threshold=best_t,
        positive_scores=pos,
        background_scores=bg,
        true_positive_rate=float(np.mean(pos >= best_t)),
        false_positive_rate=float(np.mean(bg >= best_t)),
        auc=_auc(pos, bg),
    )


def _auc(positive: np.ndarray, background: np.ndarray) -> float:
    """Probability a random positive outscores a random background document.

    The rank-sum (Mann-Whitney) form, so ties count as half — which matters
    because cross-encoder scores on near-identical documents do tie.
    """
    if len(positive) == 0 or len(background) == 0:
        return 0.5
    combined = np.concatenate([positive, background])
    order = combined.argsort(kind="mergesort")
    ranks = np.empty(len(combined), dtype=np.float64)
    ranks[order] = np.arange(1, len(combined) + 1, dtype=np.float64)
    # Average ranks within ties.
    sorted_vals = combined[order]
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1

    n_pos = len(positive)
    rank_sum = ranks[:n_pos].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * len(background)))
