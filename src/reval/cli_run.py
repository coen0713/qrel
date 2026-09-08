"""``reval run`` — execute experiments and print scored results with CIs."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from reval.experiments.config import ExperimentConfig
from reval.experiments.runner import RunResult, load_dataset, run_experiment
from reval.metrics.bootstrap import paired_bootstrap
from reval.paths import configs_dir

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


def _is_contamination(path: Path) -> bool:
    """Whether a config file targets the contamination runner rather than this one."""
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return "retrievers" in data


def _results_table(results: list[RunResult], title: str) -> Table:
    """Point estimate with a 95% CI in every cell. PLAN.md §5 rule 1.

    Measures are rows and runs are columns, not the other way round: a cell
    holding "62.4 [57.1, 67.6]" is ~20 characters, and 15 measures across the
    top overflows any terminal into unreadability. Comparisons are read down a
    column anyway.
    """
    measures = sorted(
        results[0].estimates,
        key=lambda m: (int(m.split("@")[1]), m.split("@")[0]),
    )
    table = Table(title=title, header_style="bold")
    table.add_column("measure", style="bold")
    for r in results:
        table.add_column(r.run_tag.rsplit(".", 1)[0], justify="right")

    for m in measures:
        table.add_row(m, *(r.estimates[m].as_pct() for r in results))

    table.add_section()
    table.add_row("chunks indexed", *(f"{r.n_chunks:,}" for r in results))
    table.add_row(
        "queries scored", *(f"{len(next(iter(r.per_query.values()))):,}" for r in results)
    )
    return table


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    config: list[Path] = typer.Option(
        None, "--config", "-c", exists=True, help="Experiment YAML; repeatable."
    ),
    all_configs: bool = typer.Option(False, "--all", help="Run every YAML in configs/."),
    ks: str = typer.Option(None, "--ks", help="Override cutoffs, e.g. '10,20'."),
    no_write: bool = typer.Option(False, "--no-write", help="Skip run files and manifests."),
) -> None:
    """Run one or more experiments and print a results table."""
    if ctx.invoked_subcommand is not None:
        return

    paths = list(config or [])
    if all_configs:
        # configs/ holds two kinds of file. The contamination experiment has its
        # own runner and its own shape (`retrievers` plural, a list of synthetic
        # sets), so parsing it as a single-retriever ExperimentConfig would fail
        # on an unknown key and take `--all` down with it.
        paths = [p for p in sorted(configs_dir().glob("*.yaml")) if not _is_contamination(p)]
    if not paths:
        raise typer.BadParameter("pass --config <file> at least once, or --all")

    configs = [ExperimentConfig.from_yaml(p) for p in paths]
    if ks:
        cutoffs = [int(x) for x in ks.split(",")]
        configs = [
            c.model_copy(
                update={
                    "eval": c.eval.model_copy(
                        update={"ks": cutoffs, "top_k": max(c.eval.top_k, max(cutoffs))}
                    )
                }
            )
            for c in configs
        ]

    # Load each dataset once even when several configs share it; re-reading
    # FiQA's 57,638 documents per config is pure waste.
    datasets: dict[tuple[str, str | None], object] = {}
    results: list[RunResult] = []

    for cfg in configs:
        key = (cfg.dataset, cfg.split)
        if key not in datasets:
            datasets[key] = load_dataset(cfg)
        console.print(f"[dim]running[/dim] {cfg.run_tag()}")
        results.append(run_experiment(cfg, dataset=datasets[key], write_outputs=not no_write))

    console.print(_results_table(results, "results (mean [95% CI], percentage points)"))

    for r in results:
        if r.manifest is not None:
            for w in r.manifest.warnings():
                console.print(f"[yellow]warning[/yellow] {r.run_tag}: {w}")

    if len(results) == 2:
        _print_comparison(results[0], results[1])
    elif len(results) > 2:
        console.print(
            "[dim]Use `reval compare` for pairwise paired-bootstrap tests across many runs.[/dim]"
        )


def _print_comparison(a: RunResult, b: RunResult) -> None:
    measures = sorted(set(a.per_query) & set(b.per_query))
    table = Table(title="paired bootstrap: A - B", header_style="bold")
    table.add_column("measure")
    table.add_column("A - B (points)", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("p", justify="right")
    table.add_column("verdict")

    for m in measures:
        cmp = paired_bootstrap(
            a.per_query[m],
            b.per_query[m],
            n_resamples=a.config.bootstrap.n_resamples,
            confidence=a.config.bootstrap.confidence,
            seed=a.config.seed,
        )
        verdict = "[green]significant[/green]" if cmp.significant else "[dim]not significant[/dim]"
        table.add_row(
            m,
            f"{cmp.mean_diff * 100:+.2f}",
            f"[{cmp.lo * 100:+.2f}, {cmp.hi * 100:+.2f}]",
            f"{cmp.p_value:.4f}",
            verdict,
        )
    console.print(f"A = {a.run_tag}\nB = {b.run_tag}")
    console.print(table)


@app.command()
def compare(
    a: Path = typer.Option(..., "--a", exists=True, help="Config A."),
    b: Path = typer.Option(..., "--b", exists=True, help="Config B."),
) -> None:
    """Run two configs and report the paired bootstrap difference."""
    cfg_a, cfg_b = ExperimentConfig.from_yaml(a), ExperimentConfig.from_yaml(b)
    res_a = run_experiment(cfg_a)
    res_b = run_experiment(cfg_b)
    console.print(_results_table([res_a, res_b], "results (mean [95% CI], percentage points)"))
    _print_comparison(res_a, res_b)
