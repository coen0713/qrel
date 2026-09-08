"""The writeup is a deliverable, so its links are testable.

A broken figure reference or a link to a renamed config is invisible until
someone reads the document — usually the person you most wanted to impress.
These are cheap and catch it at commit time.
"""

from __future__ import annotations

import re

import pytest

from reval.paths import repo_root

WRITEUP = repo_root() / "WRITEUP.md"
README = repo_root() / "README.md"


@pytest.fixture(scope="module")
def writeup() -> str:
    if not WRITEUP.exists():
        pytest.skip("WRITEUP.md not written yet")
    return WRITEUP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


def local_targets(text: str) -> set[str]:
    """Repo-relative paths referenced by markdown links, images or srcset."""
    targets = set(re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", text))
    targets |= set(re.findall(r'src(?:set)?="([^"]+)"', text))
    return {t for t in targets if not t.startswith(("http://", "https://", "mailto:", "#"))}


@pytest.mark.parametrize("doc", ["writeup", "readme"])
def test_every_local_link_resolves(doc, request):
    text = request.getfixturevalue(doc)
    missing = [t for t in sorted(local_targets(text)) if not (repo_root() / t).exists()]
    assert not missing, f"{doc}: dangling links {missing}"


def test_writeup_shows_all_three_figures(writeup):
    figures = {
        re.sub(r"-(light|dark)\.png$", "", f)
        for f in re.findall(r"results/plots/([a-z0-9-]+\.png)", writeup)
    }
    assert figures == {"fig1-conditions", "fig2-overlap-gradient", "fig3-false-negatives"}


def test_every_figure_ships_both_themes(writeup):
    """A light PNG alone becomes a white slab for a dark-mode reader."""
    referenced = set(re.findall(r"results/plots/([a-z0-9-]+\.png)", writeup))
    for name in referenced:
        counterpart = (
            name.replace("-light.png", "-dark.png")
            if "-light" in name
            else name.replace("-dark.png", "-light.png")
        )
        assert counterpart in referenced, f"{name} has no {counterpart}"


def test_writeup_length_is_within_the_planned_range(writeup):
    """PLAN.md M5 specifies 1,500-2,500 words."""
    prose = re.sub(r"```.*?```", "", writeup, flags=re.S)
    prose = re.sub(r"<picture>.*?</picture>", "", prose, flags=re.S)
    words = len(prose.split())
    assert 1500 <= words <= 2500, f"{words} words is outside PLAN.md M5's 1,500-2,500"


def test_writeup_leads_with_the_finding_not_the_motivation(writeup):
    """PLAN.md M5: 'The claim, in the first paragraph. Not the motivation.'"""
    first = writeup.split("\n\n")[1]
    assert re.search(r"\d+(\.\d+)?\s*(points|%)", first), (
        "the opening paragraph states no measured quantity"
    )


def test_writeup_reports_confidence_intervals(writeup):
    # PLAN.md §5 rule 1: nothing compared without one.
    assert len(re.findall(r"\[[+-]?\d+\.\d+,\s*[+-]?\d+\.\d+\]", writeup)) >= 10


def test_writeup_keeps_a_threats_to_validity_section(writeup):
    assert re.search(r"^#+ .*threats to validity", writeup, re.I | re.M)
    for threat in ("pretraining", "sparse judgments", "one corpus"):
        assert threat.lower() in writeup.lower(), f"unstated threat: {threat}"


def test_writeup_documents_reproduction_from_a_clean_clone(writeup):
    assert "git clone" in writeup
    assert re.search(r"^#+ .*reproduc", writeup, re.I | re.M)


def test_readme_has_what_m5_requires(readme):
    """PLAN.md M5: two-sentence what-it-is, headline number, install, worked
    example, and the pytrec_eval parity badge."""
    assert "pytrec_eval parity" in readme or "pytrec__eval" in readme
    assert "uv pip install" in readme
    assert "reval corpus stats scifact" in readme
    assert "recall@10" in readme
    assert "WRITEUP.md" in readme
