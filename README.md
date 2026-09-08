# qrel

[![CI](https://github.com/coen0713/qrel/actions/workflows/ci.yml/badge.svg)](https://github.com/coen0713/qrel/actions/workflows/ci.yml)
[![metrics: pytrec_eval parity](https://img.shields.io/badge/metrics-pytrec__eval%20parity%20%E2%89%A41e--6-brightgreen)](tests/test_metrics_parity.py)

A retrieval evaluation harness with contamination detection. It measures how
well a retriever finds the right documents — and, more to the point, whether
your evaluation set is quietly lying to you about that.

## The headline number

On SciFact, with the corpus, chunker, retriever, aggregation policy and metric
held identical across conditions — only the origin of the queries varies:

| eval set | n | query–gold overlap | recall@10 (BM25) |
|---|---:|---:|---:|
| **human-authored** (BEIR) | 300 | 0.036 | **80.6** [76.1, 84.9] |
| **corpus-generated**, typical prompt | 492 | 0.098 | **99.8** [99.4, 100.0] |
| **corpus-generated**, vocabulary-controlled prompt | 482 | 0.044 | **77.8** [74.1, 81.5] |

Queries generated the way a typical RAG tutorial generates them retrieve their
own source passage **essentially always**: +19.2 recall@10 points over
human-authored queries on the same index (95% CI [+14.9, +23.7], unpaired
bootstrap, p = 0.0001). The pre-registered threshold for calling this confirmed
was +5 points.

Adding one instruction to the generation prompt — *do not reuse the passage's
vocabulary* — removes the entire effect (−2.8 points, CI spans zero).

The subtler half: within the vocabulary-controlled set, retrievability is still
strongly predicted by residual lexical overlap. Sorting those queries into
overlap deciles gives recall@10 of **49.0** in the lowest and **95.8** in the
highest (Spearman ρ = +0.268 [+0.186, +0.344]). The mean effect is gone; the
mechanism is not.

Full analysis, threats to validity and the ceiling-effect caveat:
[`notes/preregistration.md`](notes/preregistration.md) (written and committed
*before* any query was generated — check the git history) and
[`notes/decisions.md`](notes/decisions.md) D19.

> **Status: in progress.** M0–M3 done, every acceptance criterion in
> [PLAN.md](PLAN.md) verified. M4 (hard negatives) and M5 (writeup) remain.

## Why

Most RAG evaluation sets are generated from the corpus they evaluate against.
You prompt an LLM to write questions from passage *P*, mark *P* as the gold
answer, and then measure how often your retriever returns *P*. That validates a
system against the process that produced its own labels. The number goes up, and
what it measures is string overlap rather than retrieval quality.

`qrel` builds the harness that catches that, then uses it to measure how large
the effect actually is.

## Install

```bash
git clone https://github.com/coen0713/qrel && cd qrel
uv venv && uv pip install -e ".[dev]"
```

The core install has no `torch`: BM25 baselines, all metrics, and the whole
contamination module run without it. Add dense retrieval and synthetic query
generation when you need them:

```bash
uv pip install -e ".[dev,dense,llm]"
```

## Worked example

```bash
reval corpus download scifact
reval corpus stats scifact
```

```
                        scifact [test]
  statistic                   value    BEIR paper   check
  documents                   5,183         5,183   match
  queries (judged)              300           300   match
  mean relevant / query         1.1           1.1   match
```

If any row says `MISMATCH`, the loader is wrong and nothing downstream can be
trusted. That is the point of printing the published column next to ours.

## Metric parity

Our `recall@k`, `MRR@k`, and `nDCG@k` are implemented from scratch and checked
against [`pytrec_eval`](https://github.com/cvangysel/pytrec_eval) — the Python
binding for NIST's `trec_eval` — to within `1e-6`, on real run files, across all
three datasets at k ∈ {1, 5, 10, 20, 100}. CI enforces it:

```bash
pytest tests/test_metrics_parity.py
```

This is the highest-credibility artifact in the repo. It is the difference
between "here are my numbers" and "here are my numbers, and here is the proof I
did not make them up."

## Contamination detectors

Three of the four leak types in [PLAN.md](PLAN.md) M3 have detectors; the fourth
is named rather than pretended away.

```bash
reval contaminate score --dataset scifact          # query-gold lexical overlap
reval contaminate dupes --dataset nfcorpus         # MinHash+LSH near-duplicates
reval contaminate experiment -c configs/contamination.yaml   # the whole table
```

Running (b) on the three corpora found real problems, including one the detector
had to be fixed to see:

| corpus | duplicate clusters | touching a judged document |
|---|---:|---:|
| SciFact | 0 | 0 |
| NFCorpus | 41 | **41** |
| FiQA | 73 | 1 |

Every one of NFCorpus's 41 duplicate clusters contains a gold document, which is
the case where duplication actually distorts a metric. FiQA initially reported
890 pairs, but 703 of them came from a single 38-member cluster of *empty*
documents — "near duplicate" is not a meaningful claim about two empty strings.
Excluding them surfaced something better: two of FiQA's empty documents are
judged relevant to a query, making those queries unanswerable and capping recall
for a reason no retriever can fix.

**Not detected:** encoder pretraining contamination. SciFact is plausibly in
`all-MiniLM-L6-v2`'s training data and we cannot resolve that without training an
encoder. It is stated as a threat to validity, which is worth more than
pretending it is not there.

## Design decisions worth knowing

- **Tie-breaking is explicit.** Scores tie constantly. The default policy
  reproduces `trec_eval`: score descending, then document id *descending*.
- **Chunk→document aggregation is a named step.** You retrieve chunks; qrels
  judge documents. Max-score-per-document is the default, and it is a function
  you can swap, not an implicit reduction buried in a loop.
- **Nothing is compared without a confidence interval.** Bootstrap CIs over
  queries, paired bootstrap for A-vs-B.
- **Unjudged documents count as non-relevant**, which systematically penalises
  retrievers that surface good-but-unjudged results. That is a threat to
  validity we name rather than paper over.

See [`notes/decisions.md`](notes/decisions.md) for the full log, and
[`CLAUDE.md`](CLAUDE.md) for the rigor rules the repo is held to.

## License

MIT
