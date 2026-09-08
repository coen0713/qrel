"""Synthetic query generation — the thing under test.

Builds an evaluation set the way a typical RAG project does: sample passages
from the corpus, ask an LLM for one question per passage, and mark the source
passage as the gold answer. That circularity is the object of study, not an
oversight, so this module reproduces the practice faithfully rather than
improving on it.

Three properties the experiment depends on:

- **Resumable.** The pre-registration commits to resuming an interrupted run
  rather than re-rolling it. Results append to JSONL as they arrive and a resume
  skips passages already done, so a crash at query 400 cannot become an
  accidental do-over that quietly selects a luckier sample.
- **Seeded.** Passage sampling is deterministic, and the same passages are used
  for every prompt variant, so an A-vs-B prompt comparison is a comparison of
  prompts and not of passages.
- **Provenanced.** Model id, resolved API model string, prompt file digest,
  seed, and the exclusion counts are all recorded next to the queries. A
  synthetic eval set with no record of how it was made is exactly the artifact
  this project argues against.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from reval.paths import prompts_dir
from reval.types import Corpus, DocId, Qrels, Queries

#: Model used for generation. Recorded in provenance; changing it changes the
#: experiment and must be reported.
DEFAULT_MODEL = "claude-opus-5"

#: Writing one question from one passage is not a hard reasoning task, and the
#: experiment needs 1,000 of them. Low effort keeps the run affordable without
#: touching the model choice, which is the user's to make.
DEFAULT_EFFORT = "low"

#: Pre-registered exclusion filter (notes/preregistration.md §5). A query is
#: dropped only for being empty, over-long, or declined — never for being too
#: easy or too similar to its source, which would be selecting on the dependent
#: variable.
MAX_QUERY_CHARS = 300


@dataclass(frozen=True, slots=True)
class GeneratedQuery:
    """One generated query and where it came from."""

    query_id: str
    text: str
    source_doc_id: DocId
    prompt_variant: str
    #: Set when the model declined or the output failed the filter.
    excluded_reason: str | None = None

    @property
    def usable(self) -> bool:
        return self.excluded_reason is None


@dataclass(slots=True)
class SyntheticQuerySet:
    """A generated evaluation set, plus how it was made."""

    name: str
    queries: Queries
    qrels: Qrels
    provenance: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.queries)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "name": self.name,
                    "queries": self.queries,
                    "qrels": self.qrels,
                    "provenance": self.provenance,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> SyntheticQuerySet:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=data["name"],
            queries=data["queries"],
            qrels=data["qrels"],
            provenance=data.get("provenance", {}),
        )


def load_prompt(variant: str) -> tuple[str, str]:
    """Load a prompt template by variant letter. Returns (template, sha256).

    Comment lines (``#``) are stripped; they document the prompt's intent for a
    reader without becoming part of what the model sees. The digest is over the
    *rendered* template, so a comment edit does not spuriously invalidate the
    provenance of an existing run.
    """
    path = prompts_dir() / f"generate_query_{variant.lower()}.txt"
    if not path.exists():
        available = sorted(p.stem for p in prompts_dir().glob("generate_query_*.txt"))
        raise FileNotFoundError(f"no prompt variant {variant!r}; have: {', '.join(available)}")

    raw = path.read_text(encoding="utf-8")
    template = "\n".join(
        line for line in raw.splitlines() if not line.lstrip().startswith("#")
    ).strip()
    if "{passage}" not in template:
        raise ValueError(f"{path} has no {{passage}} placeholder")
    digest = hashlib.sha256(template.encode("utf-8")).hexdigest()[:16]
    return template, digest


def sample_passages(corpus: Corpus, n: int, seed: int = 0) -> list[DocId]:
    """Sample ``n`` document ids uniformly without replacement, deterministically.

    Sorted before sampling so the result depends on the corpus contents and the
    seed, never on dict iteration order.
    """
    doc_ids = sorted(corpus)
    if n >= len(doc_ids):
        return doc_ids
    return sorted(random.Random(seed).sample(doc_ids, n))


def render(template: str, passage: str) -> str:
    """Substitute the passage. ``replace``, not ``format``.

    A passage containing a brace — chemical formulae and set notation both do —
    would make ``str.format`` raise or silently reinterpret it.
    """
    return template.replace("{passage}", passage)


def _classify(text: str, stop_reason: str | None) -> str | None:
    """Apply the pre-registered exclusion filter. Returns a reason, or None."""
    if stop_reason == "refusal":
        return "model_declined"
    if not text.strip():
        return "empty"
    if len(text) > MAX_QUERY_CHARS:
        return "too_long"
    return None


class QueryGenerator:
    """Generates one query per passage, concurrently and resumably."""

    def __init__(
        self,
        variant: str,
        model: str = DEFAULT_MODEL,
        effort: str = DEFAULT_EFFORT,
        max_workers: int = 8,
        api_key: str | None = None,
    ):
        self.variant = variant
        self.model = model
        self.effort = effort
        self.max_workers = max_workers
        self.template, self.prompt_digest = load_prompt(variant)
        self._api_key = api_key
        self._client = None
        self._lock = threading.Lock()
        self.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depends on extras
                raise ImportError(
                    "query generation needs the [llm] extra: uv pip install -e '.[llm]'"
                ) from exc
            from reval.env import require

            key = self._api_key or require(
                "ANTHROPIC_API_KEY",
                hint="Only `reval contaminate generate` needs it; everything else runs without.",
            )
            self._client = anthropic.Anthropic(api_key=key)
        return self._client

    def _generate_one(self, doc_id: DocId, passage: str) -> GeneratedQuery:
        client = self._get_client()
        response = client.messages.create(
            model=self.model,
            max_tokens=1024,
            output_config={"effort": self.effort},
            messages=[{"role": "user", "content": render(self.template, passage)}],
        )
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        # Models sometimes wrap the question in quotes despite being told not to.
        # Stripping them is formatting, not content selection.
        text = text.strip().strip('"').strip()

        with self._lock:
            self.usage["input_tokens"] += response.usage.input_tokens
            self.usage["output_tokens"] += response.usage.output_tokens
            self.usage["calls"] += 1

        return GeneratedQuery(
            query_id=f"syn{self.variant.lower()}-{doc_id}",
            text=text,
            source_doc_id=doc_id,
            prompt_variant=self.variant,
            excluded_reason=_classify(text, response.stop_reason),
        )

    def run(
        self,
        corpus: Corpus,
        doc_ids: list[DocId],
        cache_path: str | Path,
        progress: bool = True,
    ) -> list[GeneratedQuery]:
        """Generate for every passage, resuming from ``cache_path`` if present."""
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        done: dict[DocId, GeneratedQuery] = {}
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    done[rec["source_doc_id"]] = GeneratedQuery(**rec)

        todo = [d for d in doc_ids if d not in done]
        if progress:
            print(
                f"variant {self.variant}: {len(done)} cached, {len(todo)} to generate",
                flush=True,
            )

        if todo:
            handle = cache_path.open("a", encoding="utf-8")
            try:
                with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                    futures = {
                        pool.submit(self._generate_one, d, corpus[d].full_text): d for d in todo
                    }
                    for i, future in enumerate(as_completed(futures), 1):
                        try:
                            result = future.result()
                        except Exception as exc:  # noqa: BLE001 - one failure must not lose 499
                            doc_id = futures[future]
                            result = GeneratedQuery(
                                query_id=f"syn{self.variant.lower()}-{doc_id}",
                                text="",
                                source_doc_id=doc_id,
                                prompt_variant=self.variant,
                                excluded_reason=f"error:{type(exc).__name__}",
                            )
                        done[result.source_doc_id] = result
                        # Append immediately: a crash must cost one query, not all of them.
                        handle.write(json.dumps(asdict(result), sort_keys=True) + "\n")
                        handle.flush()
                        if progress and i % 25 == 0:
                            print(f"  {i}/{len(todo)}", flush=True)
            finally:
                handle.close()

        return [done[d] for d in doc_ids if d in done]

    def to_query_set(self, generated: list[GeneratedQuery], corpus_name: str) -> SyntheticQuerySet:
        """Assemble usable queries into an eval set with its gold labels."""
        usable = [g for g in generated if g.usable]
        excluded: dict[str, int] = {}
        for g in generated:
            if g.excluded_reason:
                excluded[g.excluded_reason] = excluded.get(g.excluded_reason, 0) + 1

        return SyntheticQuerySet(
            name=f"{corpus_name}-synth-{self.variant.lower()}",
            queries={g.query_id: g.text for g in usable},
            # The circularity under study, stated plainly: the gold label IS the
            # passage the query was generated from.
            qrels={g.query_id: {g.source_doc_id: 1} for g in usable},
            provenance={
                "corpus": corpus_name,
                "prompt_variant": self.variant,
                "prompt_sha256_16": self.prompt_digest,
                "model": self.model,
                "effort": self.effort,
                "generated_utc": datetime.now(UTC).isoformat(),
                "n_requested": len(generated),
                "n_usable": len(usable),
                "excluded": excluded,
                "usage": dict(self.usage),
            },
        )
