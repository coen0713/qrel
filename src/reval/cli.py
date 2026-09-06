"""``reval`` command line.

Thin: every command parses arguments, calls into a module, and renders. No
experiment logic lives here, so that everything the CLI can do is also callable
from a notebook or a test.
"""

from __future__ import annotations

import typer
from rich.console import Console

from reval import __version__
from reval.cli_corpus import app as corpus_app

app = typer.Typer(
    name="reval",
    help="A retrieval evaluation harness with contamination detection.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

app.add_typer(corpus_app, name="corpus", help="Download and inspect BEIR corpora.")


@app.command()
def version() -> None:
    """Print the reval version."""
    console.print(f"reval {__version__}")


if __name__ == "__main__":  # pragma: no cover
    app()
