"""The test that makes every other number in this repo trustworthy.

Our recall@k, MRR@k and nDCG@k are implemented from scratch. This checks them
against ``pytrec_eval`` — the Python binding for NIST's ``trec_eval``, the
reference implementation the IR field actually uses — to within 1e-6, on real
BEIR qrels, at k ∈ {1, 5, 10, 20, 100}, per query and in aggregate.

Do not relax the tolerance. If this goes red, the metric is wrong, and every
table, plot and claim downstream inherits the error.

Two notes on the comparison:

- ``pytrec_eval`` exposes ``recip_rank`` uncut. To check MRR **@k** against it,
  we truncate the run to depth k first and compare against uncut ``recip_rank``
  on the truncated run, which is the same quantity by definition.
- ``pytrec_eval`` scores only queries present in both qrels and run, so our
  evaluator's default does the same. The ``complete=True`` behaviour (missing
  queries score 0) is checked separately, against a hand-computed expectation.
"""

from __future__ import annotations

import pytest

from conftest import KS, load_dataset_or_skip, make_adversarial_run
from reval.metrics import evaluate
from reval.metrics.ranking import truncate_run
from reval.runfile import read_run, write_run

# The oracle is the only import that can be absent (it is a C++ extension), and
# nothing in `reval` needs it, so the skip guard sits after the imports rather
# than forcing every other import below it.
pytrec_eval = pytest.importorskip("pytrec_eval", reason="pytrec-eval-terrier not installed")

TOLERANCE = 1e-6
DATASETS = ["scifact", "nfcorpus", "fiqa"]


def oracle(qrels, run, measures):
    """Score a run with pytrec_eval, returning {measure: {qid: value}}."""
    ev = pytrec_eval.RelevanceEvaluator(qrels, set(measures))
    per_query = ev.evaluate(run)
    out: dict[str, dict[str, float]] = {m: {} for m in measures}
    for qid, scores in per_query.items():
        for m in measures:
            out[m][qid] = scores[m]
    return out


def assert_parity(ours: dict[str, float], theirs: dict[str, float], label: str):
    assert set(ours) == set(theirs), f"{label}: query sets differ"
    worst_qid, worst = None, 0.0
    for qid in ours:
        delta = abs(ours[qid] - theirs[qid])
        if delta > worst:
            worst_qid, worst = qid, delta
    assert worst <= TOLERANCE, (
        f"{label}: max per-query deviation {worst:.3e} > {TOLERANCE:.0e} "
        f"on query {worst_qid!r} (ours={ours[worst_qid]!r}, pytrec_eval={theirs[worst_qid]!r})"
    )


# --------------------------------------------------------------------------
# The acceptance criterion: real BEIR qrels, all three datasets, all cutoffs.
# --------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("dataset", DATASETS)
def test_recall_matches_pytrec_eval(dataset):
    ds = load_dataset_or_skip(dataset)
    run = make_adversarial_run(ds.qrels, seed=1)
    ours = evaluate(run, ds.qrels, ks=KS)
    theirs = oracle(ds.qrels, run, [f"recall_{k}" for k in KS])
    for k in KS:
        assert_parity(ours[f"recall@{k}"], theirs[f"recall_{k}"], f"{dataset} recall@{k}")


@pytest.mark.slow
@pytest.mark.parametrize("dataset", DATASETS)
def test_ndcg_matches_pytrec_eval(dataset):
    ds = load_dataset_or_skip(dataset)
    run = make_adversarial_run(ds.qrels, seed=2)
    ours = evaluate(run, ds.qrels, ks=KS)
    theirs = oracle(ds.qrels, run, [f"ndcg_cut_{k}" for k in KS])
    for k in KS:
        assert_parity(ours[f"ndcg@{k}"], theirs[f"ndcg_cut_{k}"], f"{dataset} ndcg@{k}")


@pytest.mark.slow
@pytest.mark.parametrize("dataset", DATASETS)
def test_mrr_matches_pytrec_eval(dataset):
    ds = load_dataset_or_skip(dataset)
    run = make_adversarial_run(ds.qrels, seed=3)
    ours = evaluate(run, ds.qrels, ks=KS)
    for k in KS:
        # recip_rank on a depth-k run is MRR@k by definition.
        theirs = oracle(ds.qrels, truncate_run(run, k), ["recip_rank"])
        assert_parity(ours[f"mrr@{k}"], theirs["recip_rank"], f"{dataset} mrr@{k}")


@pytest.mark.slow
@pytest.mark.parametrize("dataset", DATASETS)
def test_parity_survives_a_trec_runfile_roundtrip(dataset, tmp_path):
    """Parity must hold on the artifact we actually ship, not just in memory.

    A run written to disk and read back is what a reviewer would re-score with
    trec_eval, so that is the object worth proving correct.
    """
    ds = load_dataset_or_skip(dataset)
    run = make_adversarial_run(ds.qrels, seed=4)
    path = write_run(run, tmp_path / f"{dataset}.trec", run_name="parity")
    reloaded = read_run(path)

    ours = evaluate(reloaded, ds.qrels, ks=[10])
    theirs = oracle(ds.qrels, reloaded, ["recall_10", "ndcg_cut_10"])
    assert_parity(ours["recall@10"], theirs["recall_10"], f"{dataset} recall@10 (roundtrip)")
    assert_parity(ours["ndcg@10"], theirs["ndcg_cut_10"], f"{dataset} ndcg@10 (roundtrip)")


# --------------------------------------------------------------------------
# Fast parity on toy data, so the property is covered without a download.
# --------------------------------------------------------------------------


def test_parity_on_toy_data(toy_qrels, toy_run):
    ours = evaluate(toy_run, toy_qrels, ks=KS)
    measures = [f"recall_{k}" for k in KS] + [f"ndcg_cut_{k}" for k in KS]
    theirs = oracle(toy_qrels, toy_run, measures)
    for k in KS:
        assert_parity(ours[f"recall@{k}"], theirs[f"recall_{k}"], f"toy recall@{k}")
        assert_parity(ours[f"ndcg@{k}"], theirs[f"ndcg_cut_{k}"], f"toy ndcg@{k}")


@pytest.mark.parametrize("seed", range(12))
def test_parity_holds_across_many_random_runs(toy_qrels, seed):
    """Fuzz the run shape; a single fixture could hide a whole class of bug."""
    run = make_adversarial_run(toy_qrels, seed=seed)
    ours = evaluate(run, toy_qrels, ks=[1, 3, 5, 10])
    measures = [f"recall_{k}" for k in (1, 3, 5, 10)] + [f"ndcg_cut_{k}" for k in (1, 3, 5, 10)]
    theirs = oracle(toy_qrels, run, measures)
    for k in (1, 3, 5, 10):
        assert_parity(ours[f"recall@{k}"], theirs[f"recall_{k}"], f"seed{seed} recall@{k}")
        assert_parity(ours[f"ndcg@{k}"], theirs[f"ndcg_cut_{k}"], f"seed{seed} ndcg@{k}")


def test_tie_break_policy_is_what_makes_parity_hold(toy_qrels):
    """The parity result is load-bearing on the tie-break direction (D2).

    A relevant document tied with a non-relevant one at the cutoff boundary
    lands inside or outside k depending purely on the policy. With trec_eval's
    reversed doc-id ordering we agree with the oracle; with the intuitive
    ascending order we do not. This test exists so that if someone "fixes" the
    tie-break direction, the failure message explains why it was that way.
    """
    qrels = {"q1": {"aaa": 1}}
    run = {"q1": {"aaa": 1.0, "zzz": 1.0}}
    theirs = oracle(qrels, run, ["recall_1"])["recall_1"]["q1"]

    correct = evaluate(run, qrels, ks=[1], tie_break="trec")["recall@1"]["q1"]
    intuitive = evaluate(run, qrels, ks=[1], tie_break="docid_asc")["recall@1"]["q1"]

    assert theirs == pytest.approx(0.0), "trec_eval should rank 'zzz' above 'aaa' on a tie"
    assert correct == pytest.approx(theirs, abs=TOLERANCE)
    assert intuitive == pytest.approx(1.0)
    assert correct != intuitive, "if these agree, the fixture no longer probes tie-breaking"
