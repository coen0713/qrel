"""``reval corpus`` — download and inspect corpora."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from reval.corpora import beir

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.command("list")
def list_datasets() -> None:
    """List supported datasets and whether they are downloaded."""
    table = Table(title="supported corpora", header_style="bold")
    table.add_column("dataset")
    table.add_column("domain")
    table.add_column("split")
    table.add_column("docs", justify="right")
    table.add_column("queries", justify="right")
    table.add_column("local")

    for name in beir.dataset_names():
        sp = beir.spec(name)
        local = "[green]yes[/green]" if beir.is_downloaded(name) else "[dim]no[/dim]"
        table.add_row(sp.name, sp.domain, sp.split, f"{sp.n_docs:,}", f"{sp.n_queries:,}", local)
    console.print(table)


@app.command()
def download(
    datasets: list[str] = typer.Argument(None, help="Datasets to fetch; default is all."),
    force: bool = typer.Option(False, "--force", help="Re-download even if present."),
) -> None:
    """Download BEIR datasets into ./data/beir/."""
    names = datasets or beir.dataset_names()
    for name in names:
        beir.spec(name)  # validate before touching the network
    for name in names:
        path = beir.download(name, force=force)
        console.print(f"[green]ok[/green] {name} -> {path}")


@app.command()
def stats(
    dataset: str = typer.Argument(..., help="Dataset name, e.g. scifact."),
    split: str = typer.Option(None, "--split", help="Override the default split."),
    strict: bool = typer.Option(
        False, "--strict", help="Exit non-zero if the numbers disagree with the BEIR paper."
    ),
) -> None:
    """Print corpus statistics and check them against the published BEIR numbers.

    This is the M0 acceptance criterion. If any row is marked MISMATCH, the
    loader is wrong and nothing downstream can be trusted.
    """
    ds = beir.load(dataset, split=split)
    st = beir.stats(ds)

    table = Table(title=f"{st.name} [{st.split}]", header_style="bold")
    table.add_column("statistic")
    table.add_column("value", justify="right")
    table.add_column("BEIR paper", justify="right")
    table.add_column("check")

    def row(label: str, value: str, expected: str = "", ok: bool | None = None) -> None:
        if ok is None:
            mark = ""
        elif ok:
            mark = "[green]match[/green]"
        else:
            mark = "[red]MISMATCH[/red]"
        table.add_row(label, value, expected, mark)

    row("documents", f"{st.n_docs:,}", f"{st.expected_n_docs:,}", st.docs_match)
    row("queries (judged)", f"{st.n_queries:,}", f"{st.expected_n_queries:,}", st.queries_match)
    row(
        "mean relevant / query",
        f"{st.mean_relevant_per_query:.1f}",
        f"{st.expected_avg_rel_per_query:.1f}",
        st.avg_rel_match,
    )
    row("total judgments", f"{st.n_judgments:,}")
    row("mean judgments / query", f"{st.mean_judgments_per_query:.1f}")
    row("mean doc length (chars)", f"{st.mean_doc_chars:,.0f}")
    row("mean query length (chars)", f"{st.mean_query_chars:,.0f}")
    console.print(table)

    grades = Table(title="relevance grade distribution", header_style="bold")
    grades.add_column("grade", justify="right")
    grades.add_column("judgments", justify="right")
    grades.add_column("share", justify="right")
    for grade, count in st.grade_distribution.items():
        share = count / max(st.n_judgments, 1)
        label = f"{grade}" + (" [dim](judged non-relevant)[/dim]" if grade == 0 else "")
        grades.add_row(label, f"{count:,}", f"{share:6.1%}")
    console.print(grades)

    if st.all_match:
        console.print("[green]All published BEIR statistics reproduced.[/green]")
    else:
        console.print(
            "[red]Loader disagrees with the BEIR paper — fix this before trusting any metric.[/red]"
        )
        if strict:
            raise typer.Exit(code=1)
