"""Tripwire: no real credential may reach a tracked file.

This exists because the near-miss actually happened — a real key was pasted into
``.env.example``, which is tracked, one ``git add -A`` away from being published
to a public repo and permanently in history. The placeholder in an example file
looks exactly like the place a key belongs, which is the whole problem.

Scanning is over ``git ls-files`` rather than the working tree, so the question
asked is precisely the one that matters: *could this be committed?* An untracked
``.env`` holding the real key is fine and is expected to exist.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from reval.paths import repo_root

#: Credential shapes worth catching. Each needs enough trailing entropy that a
#: documentation placeholder ("sk-ant-...") cannot match.
SECRET_PATTERNS = {
    "Anthropic API key": re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    "OpenAI API key": re.compile(r"sk-proj-[A-Za-z0-9_\-]{20,}"),
    "AWS access key id": re.compile(r"AKIA[0-9A-Z]{16}"),
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),
    "Hugging Face token": re.compile(r"hf_[A-Za-z0-9]{34,}"),
    "private key block": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
}

#: This file necessarily contains the patterns it searches for.
SELF = Path(__file__).name


def tracked_files() -> list[Path]:
    root = repo_root()
    try:
        out = subprocess.run(
            ["git", "ls-files"], capture_output=True, text=True, check=True, cwd=root
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pytest.skip("not a git checkout")
    return [root / line for line in out.splitlines() if line.strip()]


def test_no_tracked_file_contains_a_credential():
    offenders: list[str] = []

    for path in tracked_files():
        if path.name == SELF or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable; not where keys hide in this repo
        for label, pattern in SECRET_PATTERNS.items():
            match = pattern.search(text)
            if match:
                # Report the location and the kind, never the value.
                line = text[: match.start()].count("\n") + 1
                rel = path.relative_to(repo_root())
                offenders.append(f"{rel}:{line} looks like a {label}")

    assert not offenders, (
        "Credential-shaped strings found in tracked files:\n  "
        + "\n  ".join(offenders)
        + "\n\nMove the value into .env (gitignored) and restore the placeholder. "
        "If it was ever committed, rotate the credential — git history is forever."
    )


def test_dotenv_is_ignored_by_git():
    """The whole scheme rests on this one line of .gitignore."""
    result = subprocess.run(
        ["git", "check-ignore", "-q", ".env"], cwd=repo_root(), capture_output=True
    )
    assert result.returncode == 0, ".env is NOT gitignored — real secrets could be committed"


def test_dotenv_example_is_tracked_and_holds_only_a_placeholder():
    example = repo_root() / ".env.example"
    assert example.exists(), ".env.example documents which variables are needed"
    text = example.read_text(encoding="utf-8")
    assert "ANTHROPIC_API_KEY" in text
    for label, pattern in SECRET_PATTERNS.items():
        assert not pattern.search(text), f".env.example contains a real {label}"
