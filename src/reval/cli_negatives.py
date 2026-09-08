"""``reval negatives`` — hard-negative mining and the false-negative filter."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

from reval.corpora import beir
from reval.paths import results_dir

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.command()
def mine(
    dataset: str = typer.Option("scifact", "--dataset"),
    retriever: str = typer.Option("bm25", "--retriever", help="bm25 or dense."),
    chunker: str = typer.Option("fixed", "--chunker"),
    encoder: str = typer.Option("minilm", "--encoder"),
    depth: int = typer.Option(50, "--depth", help="How deep to look for negatives."),
    per_query: int = typer.Option(10, "--per-query", help="Negatives kept per query."),
    background_per_query: int = typer.Option(
        5, "--background", help="Random background documents sampled per query for calibration."
    ),
    no_filter: bool = typer.Option(False, "--no-filter", help="Skip the cross-encoder pass."),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Mine hard negatives and measure how many are actually false negatives.

    The M4 result: BEIR's judgments are sparse, so top-ranked unjudged documents
    are the ones most likely to be unlabelled positives. Mining selects for them
    by construction. This measures how badly.
    """
    from reval.chunking.base import chunk_to_doc_map
    from reval.experiments.config import ExperimentConfig, RetrieverConfig
    from reval.experiments.factory import build_cached_encoder, build_chunker, build_retriever
    from reval.metrics import aggregate_to_documents
    from reval.negatives import (
        confusion_profile,
        difficulty_stratified_queries,
        filter_false_negatives,
    )
    from reval.negatives import mine as mine_negatives
    from reval.negatives.crossencoder import CrossEncoderScorer, calibrate_threshold

    ds = beir.load(dataset)
    cfg = ExperimentConfig(dataset=dataset, seed=seed)
    cfg = cfg.model_copy(
        update={
            "chunker": cfg.chunker.model_copy(update={"kind": chunker}),
            "encoder": cfg.encoder.model_copy(update={"name": encoder}),
            "retriever": RetrieverConfig(kind=retriever),
        }
    )

    needs_encoder = retriever == "dense" or chunker == "semantic"
    enc = build_cached_encoder(cfg.encoder) if needs_encoder else None
    chunks = build_chunker(cfg.chunker, encoder=enc).chunk_corpus(ds.corpus)
    chunk_map = chunk_to_doc_map(chunks)

    console.print(f"[dim]indexing[/dim] {dataset} / {chunker} / {retriever}")
    ret = build_retriever(cfg, encoder=enc)
    ret.index(chunks)
    chunk_run = ret.search(ds.queries, top_k=max(depth, 100))
    doc_run = aggregate_to_documents(chunk_run, chunk_map, policy=cfg.eval.aggregation)

    report = mine_negatives(doc_run, ds.qrels, depth=depth, per_query=per_query, method=retriever)
    console.print(f"mined {report.n_candidates:,} candidates over {report.n_queries:,} queries")

    threshold = None
    filtered = report
    if not no_filter:
        scorer = CrossEncoderScorer()
        console.print(f"[dim]calibrating threshold on true positives[/dim] ({scorer.identity})")
        cal = calibrate_threshold(
            scorer,
            ds.queries,
            ds.qrels,
            ds.corpus,
            background_per_query=background_per_query,
            seed=seed,
            show_progress=True,
        )
        threshold = cal.threshold
        console.print(
            f"  {cal.summary()}   "
            f"({len(cal.positive_scores):,} judged-relevant vs "
            f"{len(cal.background_scores):,} random background pairs)"
        )
        if cal.auc < 0.8:
            console.print(
                "  [yellow]AUC below 0.8: the cross-encoder separates relevant from "
                "irrelevant poorly on this corpus, so treat the filter as weak evidence."
                "[/yellow]"
            )
        console.print("[dim]scoring mined candidates[/dim]")
        filtered = filter_false_negatives(
            report, ds.queries, ds.corpus, scorer, threshold, show_progress=True
        )

    table = Table(
        title=f"hard negatives — {dataset} / {chunker} / {retriever}", header_style="bold"
    )
    table.add_column("statistic")
    table.add_column("value", justify="right")
    table.add_row("candidates mined", f"{filtered.n_candidates:,}")
    table.add_row("queries", f"{filtered.n_queries:,}")
    if threshold is not None:
        table.add_row("cross-encoder threshold", f"{threshold:.3f}")
        table.add_row("  calibration AUC", f"{cal.auc:.3f}")
        table.add_row(
            "[bold]suspected false negatives[/bold]", f"[bold]{filtered.n_suspected:,}[/bold]"
        )
        table.add_row(
            "[bold]as a fraction[/bold]", f"[bold]{filtered.suspected_fraction:.1%}[/bold]"
        )
        table.add_row("usable hard negatives", f"{filtered.n_candidates - filtered.n_suspected:,}")
    console.print(table)

    if threshold is not None:
        # The comparison that makes the fraction interpretable rather than
        # merely alarming. The same threshold applied to *random* documents
        # gives the base rate; the ratio is how strongly mining selects for
        # relevant-looking documents.
        base = cal.false_positive_rate
        mined = filtered.suspected_fraction
        enrichment = mined / base if base > 0 else float("inf")
        console.print(
            f"\n[bold]Clearing the same relevance bar:[/bold] "
            f"{base:.1%} of random corpus documents, but [bold]{mined:.1%}[/bold] of "
            f"top-ranked unjudged ones — a [bold]{enrichment:.0f}x[/bold] enrichment.\n"
            f"[dim]That is the trap PLAN.md M4 names, quantified: with {dataset}'s sparse "
            f"judgments, mining top-ranked unjudged documents selects overwhelmingly for "
            f"unlabelled positives rather than for negatives.[/dim]"
        )

    profile = confusion_profile(filtered, ds.queries, ds.qrels, ds.corpus)
    prof = Table(title="what this retriever confuses with a correct answer", header_style="bold")
    prof.add_column("measure")
    prof.add_column("token containment with the query", justify="right")
    prof.add_row("judged-relevant documents", f"{profile['mean_positive_containment']:.3f}")
    prof.add_row("mined hard negatives", f"{profile['mean_negative_containment']:.3f}")
    prof.add_row("gap (negatives − positives)", f"{profile['gap']:+.3f}")
    console.print(prof)

    difficulty = difficulty_stratified_queries(doc_run, ds.qrels)
    found = [r for r in difficulty.values() if r > 0]
    diff = Table(title="query difficulty (rank of first relevant document)", header_style="bold")
    diff.add_column("band")
    diff.add_column("queries", justify="right")
    diff.add_column("share", justify="right")
    total = max(len(difficulty), 1)
    for label, predicate in (
        ("rank 1 (easy)", lambda r: r == 1),
        ("ranks 2-10", lambda r: 2 <= r <= 10),
        ("ranks 11-100", lambda r: 11 <= r <= 100),
        ("not retrieved (hardest)", lambda r: r == 0),
    ):
        n = sum(1 for r in difficulty.values() if predicate(r))
        diff.add_row(label, f"{n:,}", f"{n / total:.1%}")
    console.print(diff)
    if found:
        console.print(
            f"[dim]median rank of first relevant doc: {sorted(found)[len(found) // 2]}[/dim]"
        )

    out = results_dir() / "negatives" / f"{dataset}.{chunker}.{retriever}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "chunker": chunker,
                "retriever": retriever,
                "encoder": encoder if needs_encoder else None,
                "depth": depth,
                "per_query": per_query,
                "threshold": threshold,
                "background_per_query": background_per_query,
                "calibration_auc": cal.auc if threshold is not None else None,
                "calibration_tpr": cal.true_positive_rate if threshold is not None else None,
                "calibration_fpr": cal.false_positive_rate if threshold is not None else None,
                "n_candidates": filtered.n_candidates,
                "n_suspected": filtered.n_suspected,
                "suspected_fraction": filtered.suspected_fraction,
                "background_rate": cal.false_positive_rate if threshold is not None else None,
                "enrichment": (
                    filtered.suspected_fraction / cal.false_positive_rate
                    if threshold is not None and cal.false_positive_rate > 0
                    else None
                ),
                "confusion_profile": profile,
                "difficulty_bands": {
                    "rank_1": sum(1 for r in difficulty.values() if r == 1),
                    "rank_2_10": sum(1 for r in difficulty.values() if 2 <= r <= 10),
                    "rank_11_100": sum(1 for r in difficulty.values() if 11 <= r <= 100),
                    "not_retrieved": sum(1 for r in difficulty.values() if r == 0),
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    console.print(f"[green]wrote[/green] {out}")

    if hasattr(ret, "close"):
        ret.close()
    if enc is not None:
        enc.close()
