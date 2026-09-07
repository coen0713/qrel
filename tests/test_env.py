"""Repo-local secret loading.

The failure mode worth guarding against is a `.env` on a developer's disk
silently overriding a secret that CI injected properly.
"""

from __future__ import annotations

import pytest

from reval.env import has, load_dotenv, require


def write_env(tmp_path, body: str):
    path = tmp_path / ".env"
    path.write_text(body, encoding="utf-8")
    return path


def test_loads_simple_pairs(tmp_path, monkeypatch):
    monkeypatch.delenv("SOME_KEY", raising=False)
    path = write_env(tmp_path, "SOME_KEY=abc123\n")
    assert load_dotenv(path) == ["SOME_KEY"]
    import os

    assert os.environ["SOME_KEY"] == "abc123"


def test_existing_environment_wins(tmp_path, monkeypatch):
    """CI injects secrets properly; a stray file must not override them."""
    monkeypatch.setenv("SOME_KEY", "from-environment")
    path = write_env(tmp_path, "SOME_KEY=from-file\n")
    assert load_dotenv(path) == []
    import os

    assert os.environ["SOME_KEY"] == "from-environment"


def test_override_is_opt_in(tmp_path, monkeypatch):
    monkeypatch.setenv("SOME_KEY", "from-environment")
    path = write_env(tmp_path, "SOME_KEY=from-file\n")
    load_dotenv(path, override=True)
    import os

    assert os.environ["SOME_KEY"] == "from-file"


def test_comments_and_blank_lines_are_skipped(tmp_path, monkeypatch):
    monkeypatch.delenv("REAL", raising=False)
    path = write_env(tmp_path, "# a comment\n\n   \nREAL=yes\n")
    assert load_dotenv(path) == ["REAL"]


def test_quotes_are_stripped(tmp_path, monkeypatch):
    monkeypatch.delenv("Q1", raising=False)
    monkeypatch.delenv("Q2", raising=False)
    load_dotenv(write_env(tmp_path, "Q1=\"double\"\nQ2='single'\n"))
    import os

    assert os.environ["Q1"] == "double"
    assert os.environ["Q2"] == "single"


def test_export_prefix_is_accepted(tmp_path, monkeypatch):
    monkeypatch.delenv("EXPORTED", raising=False)
    assert load_dotenv(write_env(tmp_path, "export EXPORTED=v\n")) == ["EXPORTED"]


def test_values_containing_equals_survive(tmp_path, monkeypatch):
    # Base64-ish secrets routinely contain '='; partition, not split.
    monkeypatch.delenv("B64", raising=False)
    load_dotenv(write_env(tmp_path, "B64=abc==\n"))
    import os

    assert os.environ["B64"] == "abc=="


def test_missing_file_is_not_an_error(tmp_path):
    assert load_dotenv(tmp_path / "nope.env") == []


def test_require_names_the_variable_and_the_file_but_not_the_value(monkeypatch):
    monkeypatch.delenv("MISSING_SECRET", raising=False)
    with pytest.raises(RuntimeError) as exc:
        require("MISSING_SECRET")
    message = str(exc.value)
    assert "MISSING_SECRET" in message
    assert ".env" in message


def test_require_returns_the_value_when_set(monkeypatch):
    monkeypatch.setenv("PRESENT_SECRET", "v")
    assert require("PRESENT_SECRET") == "v"


def test_has_reports_presence_without_raising(monkeypatch):
    monkeypatch.delenv("MAYBE", raising=False)
    assert has("MAYBE") is False
    monkeypatch.setenv("MAYBE", "x")
    assert has("MAYBE") is True
