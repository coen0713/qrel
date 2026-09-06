"""``reval index`` — build indexes and warm the embedding cache.

The M2 acceptance criterion lives here: running ``index build`` twice in a row
must re-embed zero chunks the second time and finish in under 10 seconds.
"""

from __future__ import annotations

import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from reval.chunking.base import chunk_to_doc_map
from reval.experiments.config import ExperimentConfig
from reval.experiments.factory import build_cached_encoder, build_chunker, build_retriever
from reval.experiments.runner import load_dataset

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.command()
def build(
    config: Path = typer.Option(..., "--config", "-c", exists=True, help="Experiment YAML."),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Bypass the embedding cache (measures cold-start cost)."
    ),
) -> None:
    """Chunk the corpus, embed it, and build the retrieval index."""
    cfg = ExperimentConfig.from_yaml(config)
    started = time.perf_counter()

    console.print(f"[bold]{cfg.name}[/bold]  config_hash={cfg.config_hash()}")

    t0 = time.perf_counter()
    ds = load_dataset(cfg)
    t_load = time.perf_counter() - t0

    needs_encoder = cfg.retriever.kind == "dense" or cfg.chunker.kind == "semantic"
    encoder = build_cached_encoder(cfg.encoder, use_cache=not no_cache) if needs_encoder else None

    t0 = time.perf_counter()
    chunker = build_chunker(cfg.chunker, encoder=encoder)
    chunks = chunker.chunk_corpus(ds.corpus)
    t_chunk = time.perf_counter() - t0
    chunk_map = chunk_to_doc_map(chunks)

    t0 = time.perf_counter()
    retriever = build_retriever(cfg, encoder=encoder)
    retriever.index(chunks)
    t_index = time.perf_counter() - t0

    total = time.perf_counter() - started

    table = Table(title="index build", header_style="bold")
    table.add_column("stage")
    table.add_column("value", justify="right")
    table.add_row("dataset", f"{ds.name} [{ds.split}]")
    table.add_row("documents", f"{ds.n_docs:,}")
    table.add_row("chunker", chunker.describe())
    table.add_row("chunks", f"{len(chunks):,}")
    table.add_row("chunks / document", f"{len(chunks) / max(ds.n_docs, 1):.2f}")
    table.add_row("distinct documents", f"{len(set(chunk_map.values())):,}")
    table.add_row("retriever", getattr(retriever, "name", cfg.retriever.kind))
    if getattr(retriever, "reused", False):
        table.add_row("index", "[green]reused (fingerprint unchanged)[/green]")
    else:
        table.add_row("index", "rebuilt")
    table.add_row("load time", f"{t_load:6.2f}s")
    table.add_row("chunk time", f"{t_chunk:6.2f}s")
    table.add_row("index time", f"{t_index:6.2f}s")
    table.add_row("[bold]total[/bold]", f"[bold]{total:6.2f}s[/bold]")
    console.print(table)

    if encoder is not None:
        st = encoder.last_stats
        reused = getattr(retriever, "reused", False)
        if st.requested:
            console.print(f"embedding cache: {st}")
        elif reused:
            console.print(
                "embedding cache: not consulted — the index was reused, so no text needed embedding"
            )

        # M2 acceptance: a warm re-run re-embeds nothing and finishes in <10s.
        # "Re-embeds nothing" covers both paths that get there: a cache full of
        # hits, and an index reused without consulting the cache at all.
        embedded_nothing = st.misses == 0
        if embedded_nothing and (reused or st.requested):
            if total < 10:
                console.print(
                    f"[green]M2 acceptance PASS — zero chunks re-embedded, "
                    f"{total:.2f}s < 10s.[/green]"
                )
            else:
                console.print(
                    f"[red]M2 acceptance FAIL — zero chunks re-embedded, but "
                    f"{total:.2f}s exceeds the 10s budget.[/red]"
                )
        else:
            console.print(
                f"[dim]Cold or partial cache ({st.misses:,} embedded). "
                "Run again to check the M2 acceptance criterion.[/dim]"
            )

    if hasattr(retriever, "close"):
        retriever.close()
    if encoder is not None:
        encoder.close()


@app.command()
def cache(
    config: Path = typer.Option(..., "--config", "-c", exists=True, help="Experiment YAML."),
) -> None:
    """Report embedding cache size and location for a config's encoder."""
    cfg = ExperimentConfig.from_yaml(config)
    encoder = build_cached_encoder(cfg.encoder)
    c = encoder.cache
    table = Table(title="embedding cache", header_style="bold")
    table.add_column("field")
    table.add_column("value")
    table.add_row("encoder identity", encoder.identity)
    table.add_row("namespace", encoder.namespace)
    table.add_row("path", str(c.root))
    table.add_row("vectors", f"{len(c):,}")
    table.add_row("dimension", str(c.dim))
    size = c.vectors_path.stat().st_size if c.vectors_path.exists() else 0
    table.add_row("blob size", f"{size / 1e6:,.1f} MB")
    console.print(table)
    encoder.close()
