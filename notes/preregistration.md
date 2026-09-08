# Pre-registration — M3 contamination experiment

**Written and committed before any synthetic query was generated.** The git
history is the evidence: this file lands in its own commit, and the commit that
produces the first synthetic query comes after it. `CLAUDE.md` rule 6 requires
this, because deciding what counts as a finding *after* seeing the data is the
exact failure mode this project exists to detect.

If a decision below turns out to be wrong, it gets **amended in a later commit
with the reason stated**, not silently edited. An amendment after seeing results
is a deviation and will be reported as one in `WRITEUP.md`.

---

## 1. The question

Do retrieval evaluation sets built by generating queries from corpus passages
overstate retrieval quality, relative to human-authored queries on the same
corpus and the same index?

## 2. Design

**Corpus:** SciFact (5,183 documents). One corpus, so the index, the chunker,
the retriever, the aggregation policy and the metric implementation are held
identical across conditions. The only thing that varies is where the queries
came from.

**Conditions:**

| id | queries | gold label | n (planned) |
|---|---|---|---|
| `human` | SciFact's own test queries | SciFact qrels | 300 |
| `synth_a` | LLM-generated from sampled passages, prompt A | the source passage | 500 |
| `synth_b` | LLM-generated from sampled passages, prompt B | the source passage | 500 |

**Retrievers:** BM25 (`k1=0.9, b=0.4`, Porter stemming) and dense
(`all-MiniLM-L6-v2`, pinned revision). Both already validated in M2.

**Chunkers for the ranking-stability test:** fixed-token, recursive-character,
semantic, parent-document — the four from M2.

**Fixed in advance:** seed `0` everywhere (passage sampling, bootstrap
resampling, chunker boundaries); 500 passages sampled uniformly without
replacement from the SciFact corpus; retrieval depth `top_k=100`; cutoffs
k ∈ {1, 5, 10, 20, 100}; max-score chunk→document aggregation; trec
tie-breaking; 10,000 bootstrap resamples at 95%.

**Two prompt variants, both reported.** PLAN.md §8 names "synthetic queries are
unrealistically easy in a way specific to your prompt" as a risk. Prompt A is
written the way a typical RAG tutorial writes one. Prompt B explicitly instructs
the model to avoid reusing the passage's vocabulary. If the effect appears under
A but vanishes under B, the finding is about prompt design, not about corpus
generation as a practice — and that is a different (still reportable) claim.

## 3. Statistics

**The primary contrast is UNPAIRED.** Synthetic and human queries are different
queries, so there is no per-query pairing between conditions and the paired
bootstrap from M1 does not apply. A two-sample bootstrap (resampling each
condition's queries independently, seed 0) gives the CI on the difference of
means. Using the paired test here would be a real error, and it is worth naming
before it is tempting to reach for.

**Within-condition contrasts remain paired.** Chunker A vs chunker B on the same
query set uses the paired bootstrap from M1.

Every number reported carries a 95% CI. Nothing is compared without one.

## 4. Predictions, and what would falsify them

Stated now, in advance. The **primary** outcome is H1; the rest are secondary
and will be labelled as such.

### H1 (primary) — synthetic queries inflate recall@10

Mean recall@10 on `synth_a` exceeds `human` on the same index.

- **Confirmed:** difference ≥ **+5.0 points** and the 95% CI excludes 0.
- **Weak/ambiguous:** CI excludes 0 but the point estimate is between +2.0 and
  +5.0 points.
- **Null:** |difference| < **2.0 points**, or the CI includes 0. *A null is a
  valid, publishable outcome and will be reported as the headline if that is
  what the data says.*
- **Disconfirmed in the opposite direction:** synthetic scores *lower* by ≥ 2.0
  points with a CI excluding 0. This is a genuine possible outcome — human
  SciFact claims are terse and highly lexical, so they may be *easier* than
  generated questions, and I am recording that possibility now rather than
  treating a negative result as a bug.

### H2 (secondary) — inflation concentrates in high lexical overlap

Recall@10 rises monotonically across overlap deciles (detector (a): character
5-gram Jaccard between query and gold passage).

- **Confirmed:** Spearman ρ between decile index and mean recall@10 is > 0 with
  a bootstrap CI excluding 0, and the top decile exceeds the bottom decile by
  ≥ 10 points.
- **Null:** CI on ρ includes 0.

### H3 (secondary) — BM25 inflates more than dense

The H1 difference is larger for BM25 than for the dense retriever, because BM25
scores lexical overlap directly while a dense encoder scores semantic similarity.

- **Confirmed:** BM25's inflation exceeds dense's by ≥ 2.0 points with the CI on
  that difference excluding 0.
- **Null:** CI includes 0.

### H4 (secondary) — chunker ranking is not preserved

The ordering of the four chunkers by recall@10 differs between the human and
synthetic eval sets.

- **Confirmed:** Kendall's τ between the two orderings is < 1.0 **and** at least
  one adjacent pair that is significantly ordered under one eval set is
  significantly ordered the other way under the other.
- **Null:** τ = 1.0, i.e. identical ordering.
- Note in advance: with only four chunkers this test is weak, and a τ < 1 driven
  by two chunkers whose CIs overlap heavily is **not** evidence of a reordering.
  That is why the significance clause is part of the criterion.

## 5. Stopping rules and analysis discipline

- Generation runs **once** per prompt variant at n=500. If a run fails
  mid-way it is resumed, not re-rolled for a better sample.
- Sampled passages are drawn once with seed 0 and **reused across both prompt
  variants**, so A-vs-B is a comparison of prompts and not of passages.
- No passage is excluded after generation except by the pre-stated filter
  below.
- **Pre-stated exclusion filter:** a generated query is dropped if it is empty,
  is longer than 300 characters, or the model declined to answer. Nothing is
  dropped for being too easy, too hard, or too similar to its source — that
  would be selecting on the dependent variable.
- The overlap decile boundaries are computed from the pooled synthetic set, not
  chosen to produce a clean gradient.
- If a result is checked and then the analysis is changed, both the original and
  the revised result appear in `WRITEUP.md`.

## 6. Known threats to validity, recorded in advance

1. **Encoder pretraining contamination.** SciFact is plausibly in
   `all-MiniLM-L6-v2`'s training data. Unfixable without training an encoder;
   named here and in the writeup, per PLAN.md M3(d).
2. **Sparse judgments.** SciFact averages 1.1 relevant documents per query, so a
   retrieved-but-unjudged good document counts as a miss. This penalises the
   `human` condition specifically, because the synthetic condition's gold label
   is by construction complete (exactly one correct passage, known). **This
   biases against H1** — it makes the human condition look worse — so if H1
   confirms, part of the effect may be this rather than contamination. This is
   the most serious threat to the primary claim and will be stated as such.
3. **One corpus, one generating model, two prompts.** Not "in general".
4. **Query-length mismatch.** SciFact claims average 90 characters; generated
   questions may differ systematically. Length will be reported per condition so
   a reader can judge.
5. **The synthetic gold set has exactly one relevant document by construction**,
   while human queries average 1.1. recall@10 is normalised by |relevant|, so
   this is partly absorbed, but not entirely.

## 7. What this experiment does not claim

That synthetic evaluation sets are useless, that any particular RAG system is
mis-tuned, or that the effect size measured on SciFact transfers to other
corpora. The deliverable is the harness and one carefully-bounded measurement.
