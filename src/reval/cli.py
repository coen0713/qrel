"""``reval`` command line.

Thin: every command parses arguments, calls into a module, and renders. No
experiment logic lives here, so that everything the CLI can do is also callable
from a notebook or a test.
"""

from __future__ import annotations

import typer
from rich.console import Console

from reval import __version__
from reval.cli_contaminate import app as contaminate_app
from reval.cli_corpus import app as corpus_app
from reval.cli_index import app as index_app
from reval.cli_run import app as run_app

app = typer.Typer(
    name="reval",
    help="A retrieval evaluation harness with contamination detection.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

app.add_typer(corpus_app, name="corpus", help="Download and inspect BEIR corpora.")
app.add_typer(index_app, name="index", help="Build indexes and warm the embedding cache.")
app.add_typer(run_app, name="run", help="Run experiments and score them with CIs.")
app.add_typer(
    contaminate_app,
    name="contaminate",
    help="Contamination detectors, synthetic query generation, and the M3 experiment.",
)


@app.command()
def version() -> None:
    """Print the reval version."""
    console.print(f"reval {__version__}")


if __name__ == "__main__":  # pragma: no cover
    app()
