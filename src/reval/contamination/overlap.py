"""Query–gold lexical overlap. PLAN.md M3(a) — the headline detector.

Three views of "how much of this query is just copied out of its answer":

- **character 5-gram Jaccard** — symmetric, catches shared phrasing at a
  sub-word level, so it survives inflection and light editing.
- **normalized longest common substring** — catches verbatim spans. Normalised
  by *query* length, not document length: a 1,500-character SciFact abstract
  would otherwise drive every score to near zero and the detector would report
  nothing.
- **token containment** — the fraction of query terms present in the passage.
  This is closest to what BM25 actually scores, which makes it the most direct
  predictor of the lexical retriever's behaviour.

All three are cheap, deterministic, and model-free. That matters: a detector
that needed an embedding model would be measuring contamination with an
instrument that has its own contamination problem.

Everything is case-folded. A query that differs from its passage only in
capitalisation is not less contaminated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from reval.types import Corpus, DocId, Qrels, Queries, QueryId

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

#: Character n-gram size. 5 is the standard choice for near-duplicate work: long
#: enough that common short words do not dominate the set, short enough to
#: survive a word substitution in the middle of a phrase.
DEFAULT_NGRAM = 5


@dataclass(frozen=True, slots=True)
class OverlapScores:
    """Contamination scores for one (query, gold passage) pair."""

    jaccard: float
    lcs_ratio: float
    containment: float

    def as_dict(self) -> dict[str, float]:
        return {
            "jaccard_5gram": self.jaccard,
            "lcs_ratio": self.lcs_ratio,
            "token_containment": self.containment,
        }

    @property
    def primary(self) -> float:
        """The score the experiment stratifies by.

        Character 5-gram Jaccard, fixed in the pre-registration. Named rather
        than passed around as a magic field access, so that changing it is a
        visible, reviewable act.
        """
        return self.jaccard


def _char_ngrams(text: str, n: int) -> set[str]:
    text = text.lower()
    if not text:
        return set()
    if len(text) <= n:
        # Shorter than the window: the whole string is the only gram. Returning
        # an empty set here would make every short query score 0 and silently
        # look uncontaminated.
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def char_ngram_jaccard(a: str, b: str, n: int = DEFAULT_NGRAM) -> float:
    """Jaccard similarity of the two strings' character n-gram sets."""
    ga, gb = _char_ngrams(a, n), _char_ngrams(b, n)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def longest_common_substring(a: str, b: str) -> int:
    """Length of the longest contiguous substring shared by ``a`` and ``b``.

    Rolling two rows of the DP table rather than the full matrix: the gold
    passages here run to a few thousand characters, and the full table would be
    megabytes per pair for a number we discard immediately.
    """
    if not a or not b:
        return 0
    # Iterate over the shorter string in the inner dimension to keep the rows small.
    if len(a) > len(b):
        a, b = b, a

    previous = [0] * (len(a) + 1)
    best = 0
    for cb in b:
        current = [0] * (len(a) + 1)
        for i, ca in enumerate(a, 1):
            if ca == cb:
                run = previous[i - 1] + 1
                current[i] = run
                if run > best:
                    best = run
        previous = current
    return best


def normalized_lcs(query: str, passage: str) -> float:
    """Longest common substring, as a fraction of the query's length.

    1.0 means the entire query appears verbatim inside the passage.
    """
    if not query:
        return 0.0
    return longest_common_substring(query.lower(), passage.lower()) / len(query)


def token_containment(query: str, passage: str) -> float:
    """Fraction of distinct query tokens that appear in the passage."""
    q_tokens = set(_TOKEN_RE.findall(query.lower()))
    if not q_tokens:
        return 0.0
    p_tokens = set(_TOKEN_RE.findall(passage.lower()))
    return len(q_tokens & p_tokens) / len(q_tokens)


def score_pair(query: str, passage: str, n: int = DEFAULT_NGRAM) -> OverlapScores:
    """Run all three detectors on one (query, passage) pair."""
    return OverlapScores(
        jaccard=char_ngram_jaccard(query, passage, n=n),
        lcs_ratio=normalized_lcs(query, passage),
        containment=token_containment(query, passage),
    )


def score_query_set(
    queries: Queries,
    qrels: Qrels,
    corpus: Corpus,
    rel_threshold: int = 1,
    n: int = DEFAULT_NGRAM,
) -> dict[QueryId, OverlapScores]:
    """Score every query against its gold documents, keeping the maximum.

    Maximum, not mean: contamination is a property of the *easiest* route to a
    correct answer. If any single gold document is a near-copy of the query, the
    query is contaminated no matter how distinct the others are, because the
    retriever only has to find that one.

    Queries with no relevant judgment are omitted — there is no gold passage to
    measure against, and inventing a score of 0 would pull the distribution
    toward "clean" for a reason that has nothing to do with overlap.
    """
    out: dict[QueryId, OverlapScores] = {}
    for qid, text in queries.items():
        gold: list[DocId] = [d for d, grade in qrels.get(qid, {}).items() if grade >= rel_threshold]
        best: OverlapScores | None = None
        for doc_id in gold:
            document = corpus.get(doc_id)
            if document is None:
                # A qrel pointing at an absent document is a data problem, but
                # not a reason to abandon a 500-query scoring run.
                continue
            scores = score_pair(text, document.full_text, n=n)
            if best is None or scores.primary > best.primary:
                best = scores
        if best is not None:
            out[qid] = best
    return out


def assign_deciles(
    scores: dict[QueryId, float] | dict[QueryId, OverlapScores],
    n_bins: int = 10,
) -> dict[QueryId, int]:
    """Bin queries into ``n_bins`` equal-count strata by score, 0 = lowest.

    Rank-based rather than value-based. Overlap scores are not uniformly
    distributed — they cluster — so cutting the *value* range into ten equal
    slices would leave most bins empty and put nearly every query in one or two.
    Ranking first guarantees roughly equal counts per stratum, which is what the
    per-decile confidence intervals need to be comparable.

    Ties are broken by query id so the assignment is reproducible.
    """
    if not scores:
        return {}

    values = {
        qid: (s.primary if isinstance(s, OverlapScores) else float(s)) for qid, s in scores.items()
    }
    # Sort by score, then query id: a heavily-tied distribution (every score
    # identical) must still produce a deterministic, valid split rather than
    # collapsing onto one bin or raising.
    order = sorted(values, key=lambda q: (values[q], q))
    n = len(order)

    out: dict[QueryId, int] = {}
    for rank, qid in enumerate(order):
        # Rank-based edges; the min() guards the final element off-by-one.
        out[qid] = min(int(rank * n_bins / n), n_bins - 1)
    return out


def decile_boundaries(
    scores: dict[QueryId, float] | dict[QueryId, OverlapScores], n_bins: int = 10
) -> list[float]:
    """Score value at each decile edge, for reporting what the strata mean."""
    if not scores:
        return []
    values = sorted(
        (s.primary if isinstance(s, OverlapScores) else float(s)) for s in scores.values()
    )
    return [float(np.quantile(values, i / n_bins)) for i in range(n_bins + 1)]
