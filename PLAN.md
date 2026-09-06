# retrieval-eval — Project Plan

A retrieval evaluation harness with contamination detection.

*Working name: `retrieval-eval`. Alternatives if you want more personality: `qrel`, `groundtruth`, `nullhypothesis`. Pick before M0 — renaming a repo after it has stars is annoying.*

---

## 0. The thesis

Everyone builds RAG demos. Almost nobody builds the measurement layer underneath, and the measurement layer is where the interesting bugs live.

The specific bug this project is about: **most RAG evaluation sets are generated from the corpus they evaluate against.** You prompt an LLM to write questions from passage *P*, mark *P* as the gold answer, and measure how often your retriever returns *P*. That is validating a system against the process that produced its own labels — the same failure mode as grading a segmentation model against the thresholding heuristic that made its training masks. The reported number goes up, and it is measuring string overlap rather than retrieval quality.

This project builds the harness that catches that, and then uses the harness to measure how big the effect actually is.

**Done means:** a `pip install -e .`-able package, three BEIR datasets benchmarked with confidence intervals, a contamination module with a validated detection method, and a written result that makes a specific quantitative claim a stranger could reproduce.

**Budget:** 2–3 weeks. Roughly 6 milestones, M0–M5.

---

## 1. The claim the writeup will argue

> Retrieval evaluation sets built by generating queries from corpus passages overstate recall@10 by *N* points relative to human-authored queries on the same corpus. The inflation is concentrated in the top lexical-overlap decile, and [does / does not] change the relative ranking of chunking strategies — meaning practitioners tuning chunk size against a synthetic eval set may be optimizing for the wrong thing.

Fill in *N* and the bracket with what you actually measure. **A null result is a valid outcome** — "inflation is under 2 points, smaller than expected" is a real finding, and the harness stands on its own regardless. Do not tune the experiment until it produces a dramatic number. That would be the exact sin the project is about.

Supporting results to run through the same harness:

- **BM25 vs dense**, per dataset. BM25 beats dense retrievers on several BEIR datasets; a dense-only benchmark with no sparse baseline is not credible.
- **Chunk-overlap inflation.** Overlapping chunks mean the same text appears in multiple index entries, so recall@k gets multiple shots at the same document. Measure recall@k with and without chunk→document deduplication and report the gap.

---

## 2. Stack

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Retrieval ecosystem lives here |
| Env | `uv` | Fast, lockfile, reproducible |
| Vector store | Qdrant via docker-compose | Ties directly to your Contour work |
| Sparse baseline | `bm25s` | Fast, pure-Python, no Java |
| Encoders | `sentence-transformers` | Pin model + revision |
| Metric oracle | `pytrec_eval` | Ground truth for your own metric implementations |
| Corpora | BEIR: SciFact, NFCorpus, FiQA | Small, real qrels, three domains |
| Config | Pydantic + YAML | Hashable, serializable |
| Results | Parquet + DuckDB | Query results without a database server |
| CLI | Typer | |
| Tests | pytest | |
| Plots | matplotlib | Three plots, not thirty |

Datasets are small enough to run on a laptop. SciFact is ~5K docs, NFCorpus ~3.6K, FiQA ~57K. No GPU required, though one helps for encoding.

---

## 3. Repo layout

```
retrieval-eval/
├── src/reval/
│   ├── corpora/          # BEIR loaders → canonical Corpus/Queries/Qrels
│   ├── chunking/         # fixed, recursive, semantic, parent-document
│   ├── indexing/         # Qdrant + BM25 backends behind one Retriever protocol
│   ├── embedding/        # encoder wrappers + content-addressed cache
│   ├── metrics/          # recall@k, MRR@k, nDCG@k + aggregation policy
│   ├── contamination/    # the core module (§6)
│   ├── negatives/        # hard-negative mining
│   ├── experiments/      # runner, config resolution, manifests
│   └── cli.py
├── configs/              # one YAML per experiment
├── tests/
├── runs/                 # TREC-format run files (gitignored)
├── results/              # parquet + plots
├── docker-compose.yml    # Qdrant
└── WRITEUP.md
```

---

## 4. Milestones

Each milestone has an acceptance criterion. Do not advance until it passes.

### M0 — Skeleton and corpus loaders *(1 day)*

Load SciFact, NFCorpus, FiQA into canonical types:

```python
Corpus  = dict[DocId, Document]        # doc_id, title, text
Queries = dict[QueryId, str]
Qrels   = dict[QueryId, dict[DocId, int]]   # graded relevance
```

Write run files in **TREC format** (`query_id Q0 doc_id rank score run_name`) from day one. It makes every result inspectable, re-scoreable by external tools, and diffable.

**Acceptance:** `reval corpus stats scifact` prints doc count, query count, mean judgments per query, and relevance grade distribution, matching the numbers published in the BEIR paper.

### M1 — Metrics, validated against an oracle *(2 days)*

This is the milestone that makes everything after it trustworthy. Implement from scratch:

- **recall@k** — `|relevant ∩ retrieved@k| / |relevant|`
- **MRR@k** — reciprocal rank of first relevant; 0 if none in top-k
- **nDCG@k** — graded gains, IDCG over the ideal ranking **truncated at k** (a common bug is computing IDCG over all relevant docs, which makes nDCG unreachable-by-construction when `|rel| > k`)

Three subtleties to handle explicitly, because they're where implementations silently diverge:

1. **Chunk→document aggregation.** You retrieve chunks; qrels judge documents. Pick a policy (max-score per doc is standard), implement it as an explicit, named, testable step, and document it. Never let it be implicit.
2. **Tie-breaking.** Equal scores must break deterministically (by doc_id) or your numbers won't reproduce.
3. **Unjudged documents.** BEIR qrels are sparse. A retrieved doc with no judgment is treated as non-relevant, which systematically penalizes retrievers that surface good-but-unjudged results. Note it as a threat to validity; don't try to fix it.

**Acceptance:** on the same run files, your metrics match `pytrec_eval` to within `1e-6` across all three datasets at k ∈ {1, 5, 10, 20, 100}. This test is the single highest-credibility artifact in the repo — it proves you are not making your numbers up. Put the assertion in CI and mention it in the README.

Also add **bootstrap confidence intervals** over queries (10K resamples) and a **paired bootstrap test** for A-vs-B comparisons. Point estimates without CIs cannot support the claims in §1.

### M2 — Indexing, chunking, embedding cache *(3 days)*

One `Retriever` protocol, two implementations (Qdrant dense, BM25 sparse). Four chunkers: fixed-token, recursive-character, semantic (embedding-similarity boundaries), parent-document (retrieve child, return parent).

**Content-addressed embedding cache**, keyed on `sha256(model_name + model_revision + chunker_config + chunk_text)`. A re-run re-embeds only what actually changed. This is your Contour reconciliation design applied to a different problem, and it's what makes 20 experiment configs affordable on a laptop.

**Acceptance:** `reval index build --config configs/baseline.yaml` twice in a row; the second run re-embeds zero chunks and completes in under 10 seconds.

### M3 — Contamination detection *(4 days — the core)*

Four distinct leak types. Implement detectors for the first three; document the fourth as out of scope.

**(a) Query–gold lexical overlap.** The headline. For each `(query, gold_doc)` pair compute:
- character 5-gram Jaccard
- longest common substring, normalized by query length
- token-level containment: fraction of query tokens present in the gold passage

Cheap, deterministic, no model required. These are your contamination scores.

**(b) Near-duplicate documents in the index.** MinHash LSH over shingles to find corpus documents that are near-copies of each other. Duplicates inflate recall by giving the retriever multiple valid targets for one information need.

**(c) Chunk-overlap double-counting.** When chunks overlap, top-k contains fewer distinct pieces of information than it appears to. Report deduplicated recall@k alongside raw.

**(d) Encoder pretraining contamination.** BEIR datasets are plausibly in the training data of the encoders you're testing. You cannot resolve this without training your own encoder. **Name it explicitly in the writeup's threats-to-validity section and move on.** Acknowledging a limitation you can't fix is worth more than pretending it isn't there.

**The headline experiment:**

1. Generate a synthetic eval set the way a typical RAG project would: sample passages from SciFact, prompt an LLM for one question per passage, mark the source passage as gold. Target ~500 queries. *Save the generation prompt in the repo* — it's part of the method.
2. Score every synthetic query with detector (a).
3. Evaluate the same retriever on: the synthetic set, the synthetic set stratified by overlap decile, and BEIR's human-authored queries as the clean control.
4. Report recall@10 / nDCG@10 per condition with CIs.
5. Re-rank the four chunking strategies under both eval sets and check whether the ordering changes.

**Acceptance:** a single table with rows = eval condition, columns = metric, cells = point estimate ± 95% CI, generated by one command from a config file.

### M4 — Hard negatives and remaining experiments *(3 days)*

Mine hard negatives three ways: BM25 top-k minus positives, dense top-k minus positives, and a cross-encoder-filtered variant.

**The trap to handle:** unjudged documents mined as "negatives" may actually be relevant — BEIR's sparse judgments guarantee some are. Mining top-ranked unjudged docs as negatives therefore selects *precisely* for likely-relevant false negatives. Use a cross-encoder score threshold to filter, report what fraction gets filtered, and treat the raw-vs-filtered gap as its own small result.

In an eval harness, hard negatives serve two purposes: constructing a difficulty-stratified eval subset, and diagnosing *what kind* of document a retriever confuses with a correct one.

Then run the full grid: 3 datasets × 4 chunkers × 2–3 encoders × {BM25, dense} with CIs throughout.

**Acceptance:** `reval run --all` reproduces every number in the writeup from a cold cache, and the manifest for each run records config hash, git SHA, model revisions, corpus checksums, and seed.

### M5 — Writeup and packaging *(3 days)*

`WRITEUP.md`, 1,500–2,500 words:

1. **The claim, in the first paragraph.** Not the motivation — the finding.
2. Method, tersely. Link to configs rather than describing them.
3. Results: three plots maximum, every number with a CI.
4. **Threats to validity**, honestly. Encoder pretraining contamination, sparse judgments, three datasets is not "in general," single LLM for query generation.
5. Reproduction: exact commands, from `git clone` to the table.

README needs: what it is in two sentences, the headline number, install, one worked example, and the `pytrec_eval` parity badge.

---

## 5. Rigor rules

These are the difference between a project that survives an interview probe and one that doesn't.

- **Nothing is compared without a confidence interval.** Retrieval metrics on 300–500 queries are noisy; a 1.5-point difference is usually nothing.
- **Every run is reproducible from its manifest.** Config hash, git SHA, model name *and revision*, corpus checksum, seed.
- **Fixed seeds everywhere**, including chunker boundaries and bootstrap resampling.
- **Your metrics match a reference implementation** and CI enforces it.
- **The synthetic query generation prompt is in the repo.** It's method, not a detail.
- **Do not tune the experiment toward a bigger effect.** Pre-register the analysis in an issue before running it: what you'll measure, what would count as confirming, what would count as null. Then report what you get.

---

## 6. Explicitly out of scope

Scope discipline is what makes 3 weeks feasible. Do not build:

- **Answer-generation evaluation.** This is a *retrieval* harness. Adding faithfulness/answer-quality doubles the scope and halves the rigor.
- **A web UI.** CLI, a results table, three plots.
- **A custom vector store.** Qdrant exists.
- **Fine-tuning.** Hard negatives are for evaluation and diagnosis here, not training.
- **More than three datasets.** Depth over breadth.
- **Chasing SOTA numbers.** You are measuring measurement, not competing on a leaderboard.

---

## 7. How to drive this with Claude Code

Do not paste this whole document as one prompt. Work milestone by milestone:

1. Start a session with this plan as `PLAN.md` in the repo root, plus a `CLAUDE.md` containing the §5 rigor rules — those need to be in context for every session, not just the first.
2. Prompt per milestone: *"Implement M1 from PLAN.md. Start with the metrics module and its pytrec_eval parity test. Show me the test before the implementation."*
3. **Ask for the test first on M1 and M3.** Those are the milestones where a plausible-looking wrong implementation is the main risk, and a test written before the code is harder to unconsciously fit to a buggy implementation.
4. After each milestone, run the acceptance criterion yourself rather than accepting a report that it passed.
5. Keep `notes/decisions.md` as you go — every choice you make (aggregation policy, tie-breaking, filter thresholds) is interview material, and you will not remember why you picked 0.7 in six weeks.

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| The effect is null | Pre-register; report the null. The harness is the deliverable, the finding is the headline. |
| Scope creep into RAG-answer eval | §6 is a contract with yourself |
| BEIR datasets already in encoder training data | Name it in threats-to-validity; it's unfixable at this scale |
| Synthetic queries are unrealistically easy in a way specific to your prompt | Generate with two different prompts, report both |
| Three weeks becomes eight | M0–M3 is a shippable project on its own. Cut M4 before you cut M5. |

---

## 9. What this buys you

**Resume line (2 lines, matches your existing format):**

> Built a retrieval evaluation harness (recall@k, MRR, nDCG validated against pytrec_eval) with automated eval-set contamination detection, showing that corpus-generated eval queries overstate recall@10 by N points versus human-authored queries on the same index.

**The interview flow it creates:** your Contour work is a Qdrant vector store with MCP retrieval tooling. This project is the measurement layer under exactly that kind of system. An interviewer who asks about one lands naturally in the other, and both stories end at the same place — *how do you know the number is real?* — which is also where your Lockheed and Wodify stories end.

That's four pieces of work telling one coherent story about you, instead of four unrelated things you happened to do.
