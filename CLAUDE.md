# CLAUDE.md — rigor rules for this repo

This project measures measurement. If the harness is sloppy, every number it
produces is worthless. These rules are non-negotiable and apply to every
session, not just the first one.

## The rules

1. **Nothing is compared without a confidence interval.** Retrieval metrics on
   300–650 queries are noisy; a 1.5-point difference is usually nothing. Every
   comparison in a table, plot, or writeup sentence carries a 95% CI, and every
   A-vs-B claim uses the paired bootstrap.

2. **Every run is reproducible from its manifest.** Each run writes a manifest
   recording: config hash, git SHA (with a dirty flag), model name *and*
   revision, corpus checksums, seed, and package versions. A run without a
   manifest is not a result.

3. **Fixed seeds everywhere** — chunker boundaries, bootstrap resampling,
   negative sampling, synthetic query sampling. Seeds live in config, not in
   code.

4. **Metrics match a reference implementation, and CI enforces it.**
   `tests/test_metrics_parity.py` checks our recall@k / MRR@k / nDCG@k against
   `pytrec_eval` to within 1e-6 on real run files. If that test is red, nothing
   else matters. Do not weaken its tolerance; fix the metric.

5. **The synthetic query generation prompt lives in the repo**, under
   `prompts/`. It is method, not a detail. Two prompt variants, both reported.

6. **Do not tune the experiment toward a bigger effect.** The analysis is
   pre-registered in `notes/preregistration.md` before it is run. A null result
   is a valid outcome and gets reported as one. Tuning until the number is
   dramatic is the exact sin this project exists to detect.

## Working conventions

- **Ask for the test before the implementation** on anything in `metrics/` or
  `contamination/`. Those are the places where a plausible-looking wrong
  implementation is the main risk.
- **Record decisions in `notes/decisions.md`** as you make them — aggregation
  policy, tie-breaking direction, filter thresholds, gain functions. Every one
  of these is a choice someone could reasonably have made differently, and
  "why 0.7?" is unanswerable six weeks later.
- **Run the acceptance criterion**, don't report that it passed. Each milestone
  in `PLAN.md` has one.
- **Scope is a contract.** `PLAN.md` §6 lists what this project does not build.
  Answer generation, web UIs, fine-tuning, and a fourth dataset are all out.

## Layout

```
src/reval/
  corpora/        BEIR loaders -> canonical Corpus/Queries/Qrels
  chunking/       fixed, recursive, semantic, parent-document
  indexing/       Qdrant dense + BM25 sparse behind one Retriever protocol
  embedding/      encoder wrappers + content-addressed cache
  metrics/        recall@k, MRR@k, nDCG@k + aggregation policy + bootstrap
  contamination/  overlap detectors, near-dup LSH, synthetic query generation
  negatives/      hard-negative mining
  experiments/    runner, config resolution, manifests
```

## Commands

```bash
uv venv && uv pip install -e ".[dev]"      # core + test deps (no torch)
uv pip install -e ".[dev,dense,llm]"       # everything

pytest -m "not slow and not network"        # fast suite
pytest tests/test_metrics_parity.py         # the one that matters

reval corpus stats scifact                  # M0 acceptance
reval index build --config configs/baseline.yaml
reval run --config configs/baseline.yaml
```
