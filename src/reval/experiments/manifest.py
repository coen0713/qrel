"""Run manifests.

PLAN.md §5 rule 2: every run is reproducible from its manifest. A run without
one is not a result, so the runner writes one next to every run file.

What it records, and why each field is here rather than "nice to have":

- **config hash** — identifies the experiment.
- **git SHA plus a dirty flag** — a SHA alone is a lie if the tree had
  uncommitted edits, and that is the normal state while developing.
- **model name and resolved revision** — HF repos are mutable; the name is not
  enough (see :mod:`reval.embedding.encoder`).
- **corpus checksums** — proves the data was the data.
- **seed** — everything stochastic derives from it.
- **package versions** — a bm25s or numpy upgrade can move scores.
- **chunk count and cache statistics** — cheap, and they are what makes an
  anomalous re-run obvious at a glance.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from reval.experiments.config import ExperimentConfig

TRACKED_PACKAGES = (
    "qrel",
    "numpy",
    "bm25s",
    "PyStemmer",
    "sentence-transformers",
    "torch",
    "qdrant-client",
    "pytrec-eval-terrier",
)


#: Paths whose contents are *produced by* a run, so changes there say nothing
#: about whether the git SHA describes the code that ran.
#:
#: Without this exclusion the dirty flag is self-referential and useless in a
#: batch: `reval run --all` writes the first run's manifest into `results/`,
#: which makes the tree dirty, so every run after the first reports
#: `dirty: true` — caused entirely by its own predecessor's output.
GENERATED_PATHS = ("results/", "runs/", "data/", "cache/")


def git_state(repo: Path | None = None) -> dict[str, str | bool]:
    """Current commit, and whether the tree's *source* differs from it.

    "Dirty" means the code that ran is not the code at this SHA. Generated
    artifacts are excluded — see :data:`GENERATED_PATHS`.
    """
    cwd = str(repo) if repo else None
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=cwd,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            cwd=cwd,
        ).stdout

        source_changes = [
            line
            for line in status.splitlines()
            if line.strip()
            # Porcelain format is 'XY <path>'; take the path and normalise
            # separators so the prefix test works on Windows too.
            and not line[3:].strip().strip('"').replace("\\", "/").startswith(GENERATED_PATHS)
        ]
        return {"sha": sha, "dirty": bool(source_changes)}
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return {"sha": "unknown", "dirty": True}


def package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for pkg in TRACKED_PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            continue
    return out


@dataclass(slots=True)
class Manifest:
    """Everything needed to re-run an experiment and get the same numbers."""

    run_tag: str
    config_hash: str
    config: dict
    dataset: str
    split: str
    seed: int
    git: dict
    corpus_checksums: dict[str, str]
    encoder: dict[str, str] = field(default_factory=dict)
    chunker: dict[str, object] = field(default_factory=dict)
    retriever: dict[str, object] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    cache: dict[str, float] = field(default_factory=dict)
    timing_seconds: dict[str, float] = field(default_factory=dict)
    packages: dict[str, str] = field(default_factory=package_versions)
    python: str = field(default_factory=lambda: sys.version.split()[0])
    platform_: str = field(default_factory=platform.platform)
    created_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @classmethod
    def build(cls, config: ExperimentConfig, dataset, **extra) -> Manifest:
        return cls(
            run_tag=config.run_tag(),
            config_hash=config.config_hash(),
            config=json.loads(config.canonical_json()),
            dataset=dataset.name,
            split=dataset.split,
            seed=config.seed,
            git=git_state(),
            corpus_checksums=dict(dataset.checksums),
            **extra,
        )

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")
        return path

    @staticmethod
    def read(path: str | Path) -> dict:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def warnings(self) -> list[str]:
        """Reasons this run should not be quoted as a result."""
        out: list[str] = []
        if self.git.get("dirty"):
            out.append("working tree was dirty; the git SHA does not describe the code that ran")
        if self.git.get("sha") == "unknown":
            out.append("git SHA unavailable; run is not traceable to a commit")
        return out
