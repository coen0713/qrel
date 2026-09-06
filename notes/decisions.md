# Decision log

Every entry is a choice someone could reasonably have made differently. Recorded
at the time, because "why 0.7?" is unanswerable six weeks later.

Format: **what was decided**, the alternatives, why this one, and what would
change the answer.

---

## D1 — Name: `qrel`, package `reval`

**Decided:** repo `qrel`, import path `reval`.

`qrel` is the IR term for a relevance judgment (`query_id 0 doc_id relevance`),
which is precisely the object this project interrogates. The package keeps the
plan's `reval` so the CLI verb (`reval corpus stats`) reads as a verb phrase
rather than as a noun.

**Would change if:** a PyPI collision on `qrel` forced a rename. Checked before
M0 rather than after, per PLAN.md's advice about renaming a repo with stars.

---

## D2 — Tie-breaking: score descending, then **doc_id descending**

**Decided:** default policy `trec`, which sorts by score descending and breaks
ties by document id in *reverse* lexicographic order.

**Alternatives:** doc_id ascending (the intuitive reading of "break ties by
doc_id"); insertion order (i.e. no policy).

**Why:** this is not our aesthetic preference, it is `trec_eval`'s
`comp_sim_docno`, which does `strcmp(ptr2->docno, ptr1->docno)` — note the
reversed argument order. Since the M1 acceptance criterion is agreement with
`pytrec_eval` to 1e-6, and `pytrec_eval` wraps `trec_eval`, any other policy
would put our metrics and the oracle's on differently-ordered lists at the
cutoff boundary and produce spurious disagreements on tied scores. BM25 ties
constantly, so this is not hypothetical.

The alternative is implemented and selectable (`tie_break="docid_asc"`) so the
size of the effect is measurable rather than assumed.

**Would change if:** we stopped using trec_eval as the oracle.

---

## D3 — Run files ignore the rank column on read

**Decided:** `read_run` reconstructs ranks from scores plus the tie-break
policy, discarding the ranks written in the file.

**Why:** the rank column is derived data. Trusting it would let a hand-edited or
third-party file assert an ordering its scores do not support, and the resulting
metric would be unreproducible from the scores alone. `read_scored_docs` keeps
the literal ranks for diffing and inspection, and is deliberately not what the
scorer calls.

---

## D4 — Scores written with `repr`, not fixed-point

**Decided:** `f"{score!r}"` in TREC run files.

**Why:** `%.6f` quantises scores. Two documents 1e-9 apart become exactly equal
on disk, the tie-break policy fires where it should not have, and the run file
no longer reproduces the in-memory run. Costs a few bytes per line; buys exact
round-tripping, which `test_roundtrip_preserves_float_precision` enforces.

---

## D5 — Queries with no judgments are dropped at load time

**Decided:** `Dataset.__post_init__` filters queries to those with a non-empty
qrels entry.

**Why:** BEIR ships `queries.jsonl` covering every split. Scoring the test split
without filtering averages a metric over queries that cannot contribute to it,
diluting every score toward zero by a factor that varies by dataset — a silent,
dataset-specific bias. The M0 acceptance criterion (query counts matching the
BEIR paper) only passes with the filter in place, which is a useful accident:
the published numbers verify our filtering.

---

## D6 — Grade 0 judgments are kept, but excluded from `relevant()`

**Decided:** parse and retain grade-0 rows; treat `grade >= threshold`
(default 1) as relevant.

**Why:** grade 0 means "a human looked and said no", which is genuinely
different information from "nobody looked". Dropping the rows would collapse the
two, and the distinction is exactly the material for the unjudged-documents
threat to validity in the writeup. NFCorpus is the only one of the three with
graded judgments (grades 1 and 2, 4.7% at grade 2), so it is the only dataset
where nDCG's graded gains do any work; SciFact and FiQA are binary.

---

## D7 — `full_text` is `title + " " + text`

**Decided:** match BEIR's own baseline scripts.

**Why:** BEIR's published numbers index the concatenation. Indexing body-only
would make every retriever here look worse than the literature for a reason
having nothing to do with the retriever, and would poison the BM25-vs-dense
comparison in particular (titles are short and high-IDF, so BM25 loses more from
dropping them than a dense encoder does). Empty titles are handled explicitly —
FiQA is entirely title-less, so a naive join would prepend a space to all 57,638
documents and shift every character offset our chunkers record.

---

## D8 — `pytrec-eval-terrier`, not `pytrec_eval`

**Decided:** depend on the `pytrec-eval-terrier` fork.

**Why:** identical API, imports as `pytrec_eval`, but ships prebuilt wheels.
Upstream `pytrec_eval` is an sdist requiring a C++ toolchain, which makes the
single most important test in the repo the hardest one to run — on Windows and
in CI both. The oracle being easy to install is part of the oracle being
credible.

**Would change if:** the fork drifted from upstream's numerics. It wraps the
same `trec_eval` C sources, so a drift would show up as a parity failure, which
is the test we already run.

---

## D9 — Embedded Qdrant by default, docker-compose still shipped

**Decided:** default dense backend is `qdrant-client`'s local mode
(`QdrantClient(path=...)`); `docker-compose.yml` is committed and selectable via
config.

**Why:** same client library and same API surface, so nothing about the
retriever code is a toy — swapping to the server is a config field, not a
rewrite. It also removes Docker from the critical path for reproducing the
result, which matters when the deliverable is "a stranger could reproduce this".

**Would change if:** corpus scale outgrew the local mode. FiQA at 57,638
documents is nowhere near it.
