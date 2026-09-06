"""BEIR dataset loaders.

BEIR ships each dataset as a zip containing ``corpus.jsonl``, ``queries.jsonl``
and ``qrels/{split}.tsv``. We download it once, verify it against the counts
published in the BEIR paper, and convert to the canonical types in
:mod:`reval.types`.

Three datasets, per PLAN.md: SciFact (scientific claim verification, ~1 relevant
doc per query), NFCorpus (medical, ~38 graded judgments per query, so the only
one where nDCG's graded gains really bite), and FiQA (financial opinion, 10x
larger corpus). Different domains, different judgment densities, different
scales — enough to tell a corpus-specific artifact from a real effect, and few
enough to stay honest about "three datasets is not 'in general'".
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

from reval.paths import data_dir
from reval.types import Corpus, Dataset, Document, Qrels, Queries

BEIR_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"


@dataclass(frozen=True, slots=True)
class BeirSpec:
    """A supported dataset plus the numbers the BEIR paper publishes for it.

    The published counts are the M0 acceptance criterion: if our loader
    disagrees with Table 1 of the BEIR paper, we have a parsing bug, and every
    number downstream inherits it.
    """

    name: str
    #: Name of the zip on the UKP mirror, when it differs from ``name``.
    remote: str
    domain: str
    split: str
    n_docs: int
    n_queries: int
    #: Mean judged-relevant docs per query, as published (1 decimal place).
    avg_rel_per_query: float


REGISTRY: dict[str, BeirSpec] = {
    "scifact": BeirSpec(
        name="scifact",
        remote="scifact",
        domain="scientific claim verification",
        split="test",
        n_docs=5183,
        n_queries=300,
        avg_rel_per_query=1.1,
    ),
    "nfcorpus": BeirSpec(
        name="nfcorpus",
        remote="nfcorpus",
        domain="biomedical / nutrition",
        split="test",
        n_docs=3633,
        n_queries=323,
        avg_rel_per_query=38.2,
    ),
    "fiqa": BeirSpec(
        name="fiqa",
        remote="fiqa",
        domain="financial opinion QA",
        split="test",
        n_docs=57638,
        n_queries=648,
        avg_rel_per_query=2.6,
    ),
}


def dataset_names() -> list[str]:
    return sorted(REGISTRY)


def spec(name: str) -> BeirSpec:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown dataset {name!r}; supported: {', '.join(dataset_names())}. "
            "PLAN.md §6 caps this at three datasets on purpose."
        ) from None


def dataset_path(name: str) -> Path:
    return data_dir() / "beir" / name


def is_downloaded(name: str) -> bool:
    root = dataset_path(name)
    return (root / "corpus.jsonl").exists() and (root / "queries.jsonl").exists()


def download(name: str, force: bool = False) -> Path:
    """Fetch and unpack a BEIR dataset. Idempotent unless ``force``."""
    sp = spec(name)
    root = dataset_path(name)
    if is_downloaded(name) and not force:
        return root

    root.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BEIR_BASE_URL}/{sp.remote}.zip"
    zip_path = root.parent / f"{sp.remote}.zip"

    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with (
            open(zip_path, "wb") as fh,
            tqdm(
                total=total, unit="B", unit_scale=True, desc=f"download {name}", leave=False
            ) as bar,
        ):
            for block in resp.iter_content(chunk_size=1 << 16):
                fh.write(block)
                bar.update(len(block))

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root.parent)
    zip_path.unlink()

    # The archive unpacks to a directory named after the dataset; if the mirror
    # ever renames it, fail loudly here rather than at load time.
    extracted = root.parent / sp.remote
    if extracted != root:
        extracted.rename(root)
    if not is_downloaded(name):
        raise RuntimeError(f"unpacked {url} but {root} lacks corpus.jsonl/queries.jsonl")
    return root


def load(name: str, split: str | None = None, download_if_missing: bool = True) -> Dataset:
    """Load a BEIR dataset into canonical types."""
    sp = spec(name)
    split = split or sp.split
    root = dataset_path(name)

    if not is_downloaded(name):
        if not download_if_missing:
            raise FileNotFoundError(f"{name} is not downloaded. Run: reval corpus download {name}")
        root = download(name)

    corpus_path = root / "corpus.jsonl"
    queries_path = root / "queries.jsonl"
    qrels_path = root / "qrels" / f"{split}.tsv"
    if not qrels_path.exists():
        available = sorted(p.stem for p in (root / "qrels").glob("*.tsv"))
        raise FileNotFoundError(
            f"{name} has no {split!r} split; available: {', '.join(available) or 'none'}"
        )

    corpus = _read_corpus(corpus_path)
    queries = _read_queries(queries_path)
    qrels = _read_qrels(qrels_path)

    return Dataset(
        name=name,
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        split=split,
        checksums={
            "corpus.jsonl": file_sha256(corpus_path),
            "queries.jsonl": file_sha256(queries_path),
            f"qrels/{split}.tsv": file_sha256(qrels_path),
        },
    )


def _read_corpus(path: Path) -> Corpus:
    corpus: Corpus = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            doc_id = rec["_id"]
            corpus[doc_id] = Document(
                doc_id=doc_id,
                title=rec.get("title") or "",
                text=rec.get("text") or "",
            )
    return corpus


def _read_queries(path: Path) -> Queries:
    queries: Queries = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            queries[rec["_id"]] = rec.get("text") or ""
    return queries


def _read_qrels(path: Path) -> Qrels:
    qrels: Qrels = {}
    with open(path, encoding="utf-8") as fh:
        header = fh.readline()
        # BEIR's tsv carries a `query-id\tcorpus-id\tscore` header. Some
        # mirrors drop it; detect rather than assume.
        if not header.lower().startswith("query-id"):
            fh.seek(0)
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            qid, doc_id, score = parts[0], parts[1], parts[2]
            rel = int(float(score))
            # Grade 0 is an explicit "judged non-relevant". Keeping it matters:
            # dropping it would make an explicitly-judged miss indistinguishable
            # from an unjudged one, which is a different threat to validity.
            qrels.setdefault(qid, {})[doc_id] = rel
    return qrels


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


@dataclass(slots=True)
class CorpusStats:
    """What ``reval corpus stats`` reports, and what M0 is graded on."""

    name: str
    split: str
    n_docs: int
    n_queries: int
    n_judgments: int
    mean_judgments_per_query: float
    mean_relevant_per_query: float
    grade_distribution: dict[int, int]
    mean_doc_chars: float
    mean_query_chars: float
    #: Published BEIR values, and whether we match them.
    expected_n_docs: int
    expected_n_queries: int
    expected_avg_rel_per_query: float

    @property
    def docs_match(self) -> bool:
        return self.n_docs == self.expected_n_docs

    @property
    def queries_match(self) -> bool:
        return self.n_queries == self.expected_n_queries

    @property
    def avg_rel_match(self) -> bool:
        # Published to one decimal; compare at that precision.
        return abs(round(self.mean_relevant_per_query, 1) - self.expected_avg_rel_per_query) < 0.05

    @property
    def all_match(self) -> bool:
        return self.docs_match and self.queries_match and self.avg_rel_match


def stats(ds: Dataset) -> CorpusStats:
    """Compute corpus statistics and compare them to the published BEIR numbers."""
    sp = spec(ds.name)
    grades: dict[int, int] = {}
    n_judgments = 0
    n_relevant = 0
    for rels in ds.qrels.values():
        for grade in rels.values():
            grades[grade] = grades.get(grade, 0) + 1
            n_judgments += 1
            if grade >= 1:
                n_relevant += 1

    n_q = max(len(ds.qrels), 1)
    return CorpusStats(
        name=ds.name,
        split=ds.split,
        n_docs=ds.n_docs,
        n_queries=ds.n_queries,
        n_judgments=n_judgments,
        mean_judgments_per_query=n_judgments / n_q,
        mean_relevant_per_query=n_relevant / n_q,
        grade_distribution=dict(sorted(grades.items())),
        mean_doc_chars=(sum(len(d.full_text) for d in ds.corpus.values()) / max(ds.n_docs, 1)),
        mean_query_chars=sum(len(q) for q in ds.queries.values()) / max(ds.n_queries, 1),
        expected_n_docs=sp.n_docs,
        expected_n_queries=sp.n_queries,
        expected_avg_rel_per_query=sp.avg_rel_per_query,
    )
