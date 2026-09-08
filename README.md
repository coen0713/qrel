# qrel

[![CI](https://github.com/coen0713/qrel/actions/workflows/ci.yml/badge.svg)](https://github.com/coen0713/qrel/actions/workflows/ci.yml)
[![metrics: pytrec_eval parity](https://img.shields.io/badge/metrics-pytrec__eval%20parity%20%E2%89%A41e--6-brightgreen)](tests/test_metrics_parity.py)

A retrieval evaluation harness with contamination detection. It measures how
well a retriever finds the right documents — and, more to the point, whether
your evaluation set is quietly lying to you about that.

> **Status: in progress.** M0 (corpus loaders), M1 (metrics + `pytrec_eval`
> parity) and M2 (chunking, BM25 + dense retrieval, embedding cache) are done,
> with every acceptance criterion in [PLAN.md](PLAN.md) verified. The headline
> contamination result lands in M3; **the number goes here once it is measured,
> not before.**

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
