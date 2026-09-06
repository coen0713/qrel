"""Canonical types for the harness.

Everything upstream (BEIR loaders, chunkers, retrievers) converges on these, and
everything downstream (metrics, contamination, reporting) consumes only these.
Keeping the waist of the hourglass narrow is what makes it possible to swap a
retriever without touching the metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DocId = str
QueryId = str
ChunkId = str

#: doc_id -> Document
Corpus = dict[DocId, "Document"]
#: query_id -> query text
Queries = dict[QueryId, str]
#: query_id -> {doc_id -> graded relevance}. Sparse: absence means "unjudged",
#: which every standard metric treats as non-relevant. See PLAN.md M1 note 3.
Qrels = dict[QueryId, dict[DocId, int]]
#: query_id -> {doc_id -> score}. Unranked on purpose; ranking is a separate,
#: explicit step with a named tie-breaking policy (reval.metrics.ranking).
Run = dict[QueryId, dict[DocId, float]]


@dataclass(frozen=True, slots=True)
class Document:
    """A corpus document as BEIR ships it."""

    doc_id: DocId
    title: str
    text: str

    @property
    def full_text(self) -> str:
        """Title and body joined the way BEIR baselines concatenate them.

        BEIR's own evaluation scripts index ``title + " " + text``, so we match
        that: a retriever that silently dropped titles would look worse than
        published numbers for a reason that has nothing to do with the retriever.
        """
        title = self.title.strip()
        text = self.text.strip()
        if not title:
            return text
        if not text:
            return title
        return f"{title} {text}"


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable unit produced by a chunker.

    ``chunk_id`` is deterministic (``{doc_id}#{ordinal}``) so that indexes,
    caches, and run files line up across processes and machines.

    ``parent_text`` is what a parent-document chunker returns to the caller
    while retrieving on ``text``; it is ``None`` for chunkers that retrieve and
    return the same span.
    """

    chunk_id: ChunkId
    doc_id: DocId
    ordinal: int
    text: str
    char_start: int
    char_end: int
    parent_text: str | None = None


@dataclass(slots=True)
class Dataset:
    """A loaded BEIR-style dataset: corpus, queries, and judgments."""

    name: str
    corpus: Corpus
    queries: Queries
    qrels: Qrels
    split: str = "test"
    #: sha256 of the raw corpus/queries/qrels files, for run manifests.
    checksums: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Queries with no judgments cannot contribute to any metric, and
        # silently averaging over them is a classic way to dilute a score.
        # BEIR ships queries.jsonl for every split, so this filtering matters.
        judged = {qid for qid, rels in self.qrels.items() if rels}
        self.queries = {qid: q for qid, q in self.queries.items() if qid in judged}
        self.qrels = {qid: rels for qid, rels in self.qrels.items() if rels}

    @property
    def n_docs(self) -> int:
        return len(self.corpus)

    @property
    def n_queries(self) -> int:
        return len(self.queries)

    def relevant(self, qid: QueryId, threshold: int = 1) -> set[DocId]:
        """Doc ids judged relevant at or above ``threshold`` for ``qid``."""
        return {d for d, g in self.qrels.get(qid, {}).items() if g >= threshold}


@dataclass(frozen=True, slots=True)
class ScoredDoc:
    """One line of a TREC run file, after ranking."""

    query_id: QueryId
    doc_id: DocId
    rank: int
    score: float
