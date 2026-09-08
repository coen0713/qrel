"""``reval contaminate`` — detectors, generation, and the M3 experiment."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from reval.contamination import nearduplicate, overlap
from reval.corpora import beir
from reval.paths import results_dir

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


def synthetic_path(dataset: str, variant: str) -> Path:
    return results_dir() / "synthetic" / f"{dataset}-synth-{variant.lower()}.json"


def cache_path(dataset: str, variant: str) -> Path:
    return results_dir() / "synthetic" / f"{dataset}-synth-{variant.lower()}.jsonl"


@app.command()
def generate(
    dataset: str = typer.Option("scifact", "--dataset", help="Corpus to sample passages from."),
    variant: str = typer.Option("a", "--variant", help="Prompt variant: a or b."),
    n: int = typer.Option(500, "--n", help="Passages to sample."),
    seed: int = typer.Option(0, "--seed", help="Sampling seed."),
    model: str = typer.Option(None, "--model", help="Override the generation model."),
    workers: int = typer.Option(8, "--workers", help="Concurrent requests."),
) -> None:
    """Generate a synthetic eval set the way a typical RAG project would.

    Resumable: results append to a JSONL cache as they arrive, and a re-run
    skips passages already generated. The pre-registration commits to resuming
    an interrupted run rather than re-rolling it.
    """
    from reval.contamination.generate import DEFAULT_MODEL, QueryGenerator, sample_passages

    ds = beir.load(dataset)
    doc_ids = sample_passages(ds.corpus, n=n, seed=seed)
    console.print(f"sampled {len(doc_ids)} passages from {dataset} (seed={seed})")

    gen = QueryGenerator(variant=variant, model=model or DEFAULT_MODEL, max_workers=workers)
    console.print(f"prompt {variant.upper()} sha256[:16]={gen.prompt_digest}  model={gen.model}")

    generated = gen.run(ds.corpus, doc_ids, cache_path=cache_path(dataset, variant))
    query_set = gen.to_query_set(generated, corpus_name=dataset)
    out = query_set.save(synthetic_path(dataset, variant))

    prov = query_set.provenance
    table = Table(title=f"generated {query_set.name}", header_style="bold")
    table.add_column("field")
    table.add_column("value", justify="right")
    table.add_row("passages requested", f"{prov['n_requested']:,}")
    table.add_row("usable queries", f"{prov['n_usable']:,}")
    for reason, count in sorted(prov["excluded"].items()):
        table.add_row(f"  excluded: {reason}", f"{count:,}")
    usage = prov["usage"]
    table.add_row("API calls", f"{usage['calls']:,}")
    table.add_row("input tokens", f"{usage['input_tokens']:,}")
    table.add_row("output tokens", f"{usage['output_tokens']:,}")
    # Claude Opus 5 list pricing, for a rough spend figure in the run log.
    cost = usage["input_tokens"] / 1e6 * 5.0 + usage["output_tokens"] / 1e6 * 25.0
    table.add_row("approx cost (this run)", f"${cost:,.2f}")
    console.print(table)
    console.print(f"[green]wrote[/green] {out}")


@app.command()
def score(
    dataset: str = typer.Option("scifact", "--dataset"),
    synthetic: Path = typer.Option(
        None, "--synthetic", help="Synthetic query set JSON; omit to score the human queries."
    ),
    show: int = typer.Option(5, "--show", help="Example queries to print per extreme."),
) -> None:
    """Score query–gold lexical overlap (detector (a)) for an eval set."""
    ds = beir.load(dataset)
    if synthetic:
        from reval.contamination.generate import SyntheticQuerySet

        qs = SyntheticQuerySet.load(synthetic)
        queries, qrels, label = qs.queries, qs.qrels, qs.name
    else:
        queries, qrels, label = ds.queries, ds.qrels, f"{dataset} (human)"

    scores = overlap.score_query_set(queries, qrels, ds.corpus)
    if not scores:
        console.print("[red]no scorable queries[/red]")
        raise typer.Exit(1)

    import numpy as np

    table = Table(title=f"query–gold overlap: {label}  (n={len(scores)})", header_style="bold")
    table.add_column("detector")
    for pct in ("mean", "p10", "p50", "p90", "max"):
        table.add_column(pct, justify="right")

    for name, getter in (
        ("char 5-gram Jaccard", lambda s: s.jaccard),
        ("normalized LCS", lambda s: s.lcs_ratio),
        ("token containment", lambda s: s.containment),
    ):
        values = np.array([getter(s) for s in scores.values()])
        table.add_row(
            name,
            f"{values.mean():.3f}",
            f"{np.quantile(values, 0.10):.3f}",
            f"{np.quantile(values, 0.50):.3f}",
            f"{np.quantile(values, 0.90):.3f}",
            f"{values.max():.3f}",
        )
    console.print(table)

    if show:
        ranked = sorted(scores, key=lambda q: scores[q].primary, reverse=True)
        for heading, subset in (
            ("highest overlap", ranked[:show]),
            ("lowest overlap", ranked[-show:]),
        ):
            console.print(f"\n[bold]{heading}[/bold]")
            for qid in subset:
                console.print(
                    f"  [dim]J={scores[qid].jaccard:.3f} "
                    f"C={scores[qid].containment:.3f}[/dim]  {queries[qid][:110]}"
                )


@app.command()
def dupes(
    dataset: str = typer.Option("scifact", "--dataset"),
    threshold: float = typer.Option(0.8, "--threshold", help="Minimum estimated Jaccard."),
    num_perm: int = typer.Option(128, "--num-perm"),
    seed: int = typer.Option(0, "--seed"),
    show: int = typer.Option(10, "--show"),
) -> None:
    """Find near-duplicate corpus documents (detector (b))."""
    ds = beir.load(dataset)
    console.print(f"MinHash+LSH over {ds.n_docs:,} documents (threshold={threshold})...")
    pairs = nearduplicate.find_near_duplicates(
        ds.corpus, threshold=threshold, num_perm=num_perm, seed=seed
    )
    clusters = nearduplicate.duplicate_clusters(pairs)

    judged = {d for rels in ds.qrels.values() for d, g in rels.items() if g >= 1}
    affected = [c for c in clusters if c & judged]

    table = Table(title=f"near-duplicates in {dataset}", header_style="bold")
    table.add_column("statistic")
    table.add_column("value", justify="right")
    table.add_row("duplicate pairs", f"{len(pairs):,}")
    table.add_row("duplicate clusters", f"{len(clusters):,}")
    table.add_row("documents involved", f"{sum(len(c) for c in clusters):,}")
    table.add_row("share of corpus", f"{sum(len(c) for c in clusters) / max(ds.n_docs, 1):.2%}")
    # The number that actually matters: a duplicate only distorts a metric when
    # one of the copies is a gold document for some query.
    table.add_row("clusters touching a judged doc", f"{len(affected):,}")
    console.print(table)

    # Reported separately, not as duplicates: "near duplicate" is not a
    # meaningful claim about two empty strings, and they collide trivially.
    degenerate = nearduplicate.degenerate_documents(ds.corpus)
    if degenerate:
        gold_degenerate = sorted(set(degenerate) & judged)
        console.print(
            f"\n[yellow]corpus quality[/yellow]: {len(degenerate):,} documents have "
            f"fewer than {nearduplicate.MIN_WORDS} words "
            f"({len(degenerate) / max(ds.n_docs, 1):.2%} of the corpus); "
            f"they are excluded from duplicate detection."
        )
        if gold_degenerate:
            console.print(
                f"  [red]{len(gold_degenerate)} of them are judged relevant to some "
                f"query[/red] — those queries are unanswerable, which caps recall for a "
                f"reason no retriever can fix: {', '.join(gold_degenerate[:5])}"
            )

    for pair in pairs[:show]:
        console.print(f"\n[dim]sim={pair.similarity:.3f}[/dim]  {pair.doc_a} / {pair.doc_b}")
        console.print(f"  a: {ds.corpus[pair.doc_a].full_text[:100]}")
        console.print(f"  b: {ds.corpus[pair.doc_b].full_text[:100]}")


@app.command()
def experiment(
    config: Path = typer.Option(..., "--config", "-c", exists=True, help="Contamination YAML."),
    measure: str = typer.Option(None, "--measure", help="Override the primary measure."),
    quick: bool = typer.Option(False, "--quick", help="First chunker and retriever only."),
) -> None:
    """Run the M3 headline experiment and print the acceptance table.

    One command, one config file, one table: rows are eval conditions, columns
    are metrics, cells are point estimates with 95% CIs. That is the M3
    acceptance criterion.
    """
    import json as _json

    from reval.chunking.base import chunk_to_doc_map
    from reval.contamination import experiment as exp
    from reval.contamination import report as report_mod
    from reval.contamination.generate import SyntheticQuerySet
    from reval.experiments.config import (
        ContaminationConfig,
        ExperimentConfig,
        RetrieverConfig,
    )
    from reval.experiments.factory import build_cached_encoder, build_chunker, build_retriever
    from reval.experiments.manifest import git_state

    cfg = ContaminationConfig.from_yaml(config)
    primary = measure or cfg.primary_measure
    ds = beir.load(cfg.dataset, split=cfg.split)

    synthetic_sets: dict[str, tuple] = {}
    for path in cfg.synthetic:
        p = Path(path)
        if not p.exists():
            console.print(f"[yellow]skipping missing synthetic set[/yellow] {p}")
            continue
        qs = SyntheticQuerySet.load(p)
        synthetic_sets[qs.name.replace(f"{cfg.dataset}-", "")] = (qs.queries, qs.qrels)

    if not synthetic_sets:
        console.print("[red]no synthetic query sets found — run `reval contaminate generate`[/red]")
        raise typer.Exit(1)

    conditions = exp.build_conditions(ds, synthetic_sets)
    chunkers = cfg.chunkers[:1] if quick else cfg.chunkers
    retrievers = cfg.retrievers[:1] if quick else cfg.retrievers

    console.print(
        f"[bold]{cfg.name}[/bold]  hash={cfg.config_hash()}  "
        f"conditions={[c.name for c in conditions]}"
    )

    reports: list[exp.ContaminationReport] = []
    for chunker_kind in chunkers:
        for retriever_kind in retrievers:
            label = f"{chunker_kind} x {retriever_kind}"
            console.print(f"\n[dim]building index:[/dim] {label}")

            base = ExperimentConfig(
                name=cfg.name,
                dataset=cfg.dataset,
                split=cfg.split,
                seed=cfg.seed,
                encoder=cfg.encoder,
                eval=cfg.eval,
                bootstrap=cfg.bootstrap,
            )
            base = base.model_copy(
                update={
                    "chunker": base.chunker.model_copy(update={"kind": chunker_kind}),
                    "retriever": RetrieverConfig(kind=retriever_kind),
                }
            )

            needs_encoder = retriever_kind == "dense" or chunker_kind == "semantic"
            encoder = build_cached_encoder(cfg.encoder) if needs_encoder else None
            chunker = build_chunker(base.chunker, encoder=encoder)
            chunks = chunker.chunk_corpus(ds.corpus)
            chunk_map = chunk_to_doc_map(chunks)

            retriever = build_retriever(base, encoder=encoder)
            retriever.index(chunks)

            scored = exp.score_conditions(
                retriever,
                chunk_map,
                conditions,
                ds,
                ks=cfg.eval.ks,
                top_k=cfg.eval.top_k,
                aggregation=cfg.eval.aggregation,
                tie_break=cfg.eval.tie_break,
                n_resamples=cfg.bootstrap.n_resamples,
                confidence=cfg.bootstrap.confidence,
                seed=cfg.seed,
            )

            report = exp.ContaminationReport(
                dataset=cfg.dataset,
                chunker=chunker_kind,
                retriever=retriever_kind,
                conditions=scored,
                n_chunks=len(chunks),
            )

            human = report.condition("human")
            for cond, cond_scores in zip(conditions, scored, strict=True):
                if cond.name == "human":
                    continue
                # Between-condition: UNPAIRED (preregistration §3).
                report.contrasts[cond.name] = exp.contrast(
                    cond_scores,
                    human,
                    primary,
                    n_resamples=cfg.bootstrap.n_resamples,
                    confidence=cfg.bootstrap.confidence,
                    seed=cfg.seed,
                )
                report.deciles[cond.name] = exp.stratify_by_overlap(
                    cond,
                    cond_scores,
                    ds,
                    n_bins=cfg.n_overlap_bins,
                    n_resamples=cfg.bootstrap.n_resamples,
                    seed=cfg.seed,
                )
                ov = overlap.score_query_set(cond.queries, cond.qrels, ds.corpus)
                report.overlap_correlation[cond.name] = exp.spearman_with_ci(
                    {q: s.primary for q, s in ov.items()},
                    cond_scores.per_query[primary],
                    n_resamples=min(cfg.bootstrap.n_resamples, 2000),
                    seed=cfg.seed,
                )

            reports.append(report)
            if hasattr(retriever, "close"):
                retriever.close()
            if encoder is not None:
                encoder.close()

    report_mod.render(reports, primary, cfg.dataset)

    out = results_dir() / "contamination"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "config_hash": cfg.config_hash(),
        "config": _json.loads(cfg.canonical_json()),
        "git": git_state(),
        "reports": [
            {
                "chunker": r.chunker,
                "retriever": r.retriever,
                "n_chunks": r.n_chunks,
                "conditions": [
                    {
                        "name": c.name,
                        "n_queries": c.n_queries,
                        "mean_query_chars": c.mean_query_chars,
                        "overlap_means": c.overlap_means,
                        "estimates": {
                            m: {"mean": e.mean, "lo": e.lo, "hi": e.hi, "n": e.n}
                            for m, e in c.estimates.items()
                        },
                    }
                    for c in r.conditions
                ],
                "contrasts": {
                    k: {
                        "mean_diff": v.mean_diff,
                        "lo": v.lo,
                        "hi": v.hi,
                        "p_value": v.p_value,
                        "n": v.n,
                    }
                    for k, v in r.contrasts.items()
                },
                "overlap_correlation": r.overlap_correlation,
                "deciles": {
                    k: [
                        {
                            "decile": d.decile,
                            "n": d.n,
                            "mean_overlap": d.mean_overlap,
                            "estimates": {
                                m: {"mean": e.mean, "lo": e.lo, "hi": e.hi}
                                for m, e in d.estimates.items()
                            },
                        }
                        for d in rows
                    ]
                    for k, rows in r.deciles.items()
                },
            }
            for r in reports
        ],
    }
    path = out / f"{cfg.name}.{cfg.config_hash()}.json"
    path.write_text(_json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    console.print(f"\n[green]wrote[/green] {path}")
