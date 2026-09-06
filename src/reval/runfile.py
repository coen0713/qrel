"""TREC run-file I/O.

Written from day one, per PLAN.md M0, because the format buys three things for
free: every result is inspectable with ``head``, re-scoreable by trec_eval or
anyone else's tooling, and diffable between two experiment configs.

Format, whitespace-separated:

    query_id Q0 doc_id rank score run_name

The literal ``Q0`` is a vestigial iteration field that trec_eval ignores.
"""

from __future__ import annotations

import gzip
from collections.abc import Iterable
from pathlib import Path

from reval.metrics.ranking import TieBreak, to_scored_docs
from reval.types import Run, ScoredDoc


def write_run(
    run: Run,
    path: str | Path,
    run_name: str,
    k: int | None = 1000,
    tie_break: TieBreak = "trec",
) -> Path:
    """Write ``run`` to ``path`` in TREC format.

    Args:
        run: unranked query -> {doc: score}.
        path: destination; ``.gz`` suffix transparently gzips.
        run_name: the run tag written in column 6. Use the config hash or a
            human label; it is what distinguishes runs when several are scored
            together.
        k: rank depth to write. 1000 is the TREC convention and is deep enough
            for every cutoff we report.
        tie_break: see :mod:`reval.metrics.ranking`.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open

    with opener(path, "wt", encoding="utf-8", newline="\n") as fh:
        for sd in to_scored_docs(run, k=k, tie_break=tie_break):
            # repr-style float formatting keeps the file round-trippable to the
            # bit; %f would silently quantise scores and change tie structure.
            fh.write(f"{sd.query_id} Q0 {sd.doc_id} {sd.rank} {sd.score!r} {run_name}\n")
    return path


def read_run(path: str | Path) -> Run:
    """Read a TREC run file back into an unranked ``Run``.

    Ranks in the file are ignored: they are derivable from the scores plus the
    tie-break policy, and trusting them would let a hand-edited file smuggle in
    an ordering the scores do not support.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    run: Run = {}

    with opener(path, "rt", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(
                    f"{path}:{lineno}: expected >=5 fields, got {len(parts)}: {line!r}"
                )
            qid, _q0, doc_id, _rank, score = parts[:5]
            run.setdefault(qid, {})[doc_id] = float(score)
    return run


def read_scored_docs(path: str | Path) -> list[ScoredDoc]:
    """Read a run file preserving the ranks exactly as written.

    Only for inspecting or diffing a file on disk. Scoring should go through
    :func:`read_run`, which re-derives ranks under an explicit policy.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    out: list[ScoredDoc] = []
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 5:
                continue
            out.append(
                ScoredDoc(
                    query_id=parts[0], doc_id=parts[2], rank=int(parts[3]), score=float(parts[4])
                )
            )
    return out


def write_qrels(qrels: dict[str, dict[str, int]], path: str | Path) -> Path:
    """Write judgments in TREC qrels format: ``query_id 0 doc_id relevance``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for qid in sorted(qrels):
            for doc_id in sorted(qrels[qid]):
                fh.write(f"{qid} 0 {doc_id} {qrels[qid][doc_id]}\n")
    return path


def read_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    """Read a TREC qrels file."""
    qrels: dict[str, dict[str, int]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, _it, doc_id, rel = parts[:4]
            qrels.setdefault(qid, {})[doc_id] = int(rel)
    return qrels


def merge_runs(runs: Iterable[Run]) -> Run:
    """Union several runs, keeping the max score per (query, doc).

    Used when a retriever is sharded over corpus partitions.
    """
    merged: Run = {}
    for run in runs:
        for qid, scores in run.items():
            target = merged.setdefault(qid, {})
            for doc_id, score in scores.items():
                if score > target.get(doc_id, float("-inf")):
                    target[doc_id] = score
    return merged
