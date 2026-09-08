"""Rebuild report objects from a saved experiment JSON.

The grid takes about an hour to run. Re-rendering a result — because the
presentation changed, or because a pre-registered criterion needed to be
evaluated properly rather than left to the reader — must not require re-running
the experiment, both because it is slow and because a result you can only see by
recomputing it is a result nobody will check.

Only the aggregate estimates are stored, not per-query vectors, so a rehydrated
report can be rendered but not re-bootstrapped. That is the intended boundary:
new inference goes back to the raw run.
"""

from __future__ import annotations

import json
from pathlib import Path

from reval.contamination.experiment import (
    ConditionScores,
    ContaminationReport,
    DecileRow,
)
from reval.metrics.bootstrap import Comparison, Estimate


def _estimate(d: dict) -> Estimate:
    return Estimate(mean=d["mean"], lo=d["lo"], hi=d["hi"], n=d.get("n", 0))


def load_reports(path: str | Path) -> tuple[list[ContaminationReport], dict]:
    """Return ``(reports, payload)`` from a saved contamination experiment."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    reports: list[ContaminationReport] = []

    for raw in payload["reports"]:
        conditions = [
            ConditionScores(
                name=c["name"],
                n_queries=c["n_queries"],
                # Per-query vectors are not persisted; rendering does not need
                # them, and anything that does should re-run the experiment.
                per_query={},
                estimates={m: _estimate(e) for m, e in c["estimates"].items()},
                overlap_means=c.get("overlap_means", {}),
                mean_query_chars=c.get("mean_query_chars", 0.0),
            )
            for c in raw["conditions"]
        ]
        report = ContaminationReport(
            dataset=payload["config"]["dataset"],
            chunker=raw["chunker"],
            retriever=raw["retriever"],
            conditions=conditions,
            n_chunks=raw.get("n_chunks", 0),
            contrasts={
                k: Comparison(
                    mean_diff=v["mean_diff"],
                    lo=v["lo"],
                    hi=v["hi"],
                    p_value=v["p_value"],
                    n=v["n"],
                )
                for k, v in raw.get("contrasts", {}).items()
            },
            overlap_correlation={
                k: tuple(v) for k, v in raw.get("overlap_correlation", {}).items()
            },
            deciles={
                k: [
                    DecileRow(
                        decile=row["decile"],
                        n=row["n"],
                        mean_overlap=row["mean_overlap"],
                        estimates={m: _estimate(e) for m, e in row["estimates"].items()},
                    )
                    for row in rows
                ]
                for k, rows in raw.get("deciles", {}).items()
            },
        )
        reports.append(report)

    return reports, payload
