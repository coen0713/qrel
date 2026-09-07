"""Repo-local secrets, loaded from a gitignored ``.env``.

Why a file rather than a shell variable: ``ANTHROPIC_API_KEY`` is read by other
Anthropic tooling on the same machine, so exporting it user-wide has effects
beyond this project. Keeping it scoped to the repo means the key is used by the
thing that needs it and nothing else.

Hand-rolled rather than depending on ``python-dotenv``: this is thirty lines,
and the ``[llm]`` extra should pull in an API client, not a config framework.

The file is never read for anything except environment defaults, and an existing
environment variable always wins — so CI, which injects secrets properly, is
never overridden by a stray file on a developer's disk.
"""

from __future__ import annotations

import os
from pathlib import Path

from reval.paths import repo_root

ENV_FILENAME = ".env"


def load_dotenv(path: str | Path | None = None, override: bool = False) -> list[str]:
    """Load ``KEY=value`` pairs from ``.env`` into ``os.environ``.

    Args:
        path: file to read; defaults to ``<repo root>/.env``.
        override: if False (default), variables already set in the environment
            are left alone.

    Returns:
        The names of the variables that were set, so a caller can report what
        was loaded without ever touching the values.
    """
    target = Path(path) if path else repo_root() / ENV_FILENAME
    if not target.exists():
        return []

    loaded: list[str] = []
    for raw in target.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        # `export FOO=bar` is a common habit; accept it.
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        value = value.strip()
        # Strip one layer of matching quotes, which people add out of habit and
        # which would otherwise become part of the key.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not key:
            continue
        if key in os.environ and not override:
            continue
        os.environ[key] = value
        loaded.append(key)
    return loaded


def require(name: str, hint: str = "") -> str:
    """Fetch a required secret, with an error that says how to supply it.

    Never logs or echoes the value — only its absence.
    """
    load_dotenv()
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Put it in {repo_root() / ENV_FILENAME} as\n"
            f"    {name}=...\n"
            f"(that file is gitignored), or export it in your shell."
            + (f"\n{hint}" if hint else "")
        )
    return value


def has(name: str) -> bool:
    """Whether a secret is available, without raising or revealing it."""
    load_dotenv()
    return bool(os.environ.get(name))
