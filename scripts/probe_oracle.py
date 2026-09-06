"""Determine pytrec_eval/trec_eval's exact nDCG semantics before implementing it.

Two things we must not guess:
  1. gain function: linear (rel) vs exponential (2^rel - 1)
  2. whether IDCG is truncated at k
"""

import math

import pytrec_eval

L2 = math.log2


def show(title, qrels, run, measures):
    ev = pytrec_eval.RelevanceEvaluator(qrels, measures)
    res = ev.evaluate(run)
    print(f"\n=== {title} ===")
    for qid, m in res.items():
        for k in sorted(m):
            print(f"  {qid} {k:>16} = {m[k]:.10f}")


# ---- Probe 1: gain function -------------------------------------------------
# Rank a grade-1 doc above a grade-2 doc so linear and exponential gains differ.
qrels1 = {"q1": {"d1": 2, "d2": 1}}
run1 = {"q1": {"d2": 3.0, "d1": 2.0}}

dcg_lin = 1 / L2(2) + 2 / L2(3)
idcg_lin = 2 / L2(2) + 1 / L2(3)
dcg_exp = (2**1 - 1) / L2(2) + (2**2 - 1) / L2(3)
idcg_exp = (2**2 - 1) / L2(2) + (2**1 - 1) / L2(3)

print("predicted ndcg_cut_2 with LINEAR gain     :", f"{dcg_lin / idcg_lin:.10f}")
print("predicted ndcg_cut_2 with EXPONENTIAL gain:", f"{dcg_exp / idcg_exp:.10f}")
show("gain function", qrels1, run1, {"ndcg_cut_2"})

# ---- Probe 2: is IDCG truncated at k? ---------------------------------------
# 3 relevant docs, k=2, retriever gets 2 of them perfectly.
qrels2 = {"q1": {"d1": 1, "d2": 1, "d3": 1}}
run2 = {"q1": {"d1": 3.0, "d2": 2.0}}

dcg = 1 / L2(2) + 1 / L2(3)
idcg_trunc = 1 / L2(2) + 1 / L2(3)
idcg_full = 1 / L2(2) + 1 / L2(3) + 1 / L2(4)
print("\npredicted ndcg_cut_2 if IDCG TRUNCATED at k:", f"{dcg / idcg_trunc:.10f}")
print("predicted ndcg_cut_2 if IDCG over ALL rel  :", f"{dcg / idcg_full:.10f}")
show("IDCG truncation", qrels2, run2, {"ndcg_cut_2"})

# ---- Probe 3: measure names and recip_rank behaviour ------------------------
qrels3 = {"q1": {"d1": 1, "d5": 1}}
run3 = {"q1": {"d9": 5.0, "d8": 4.0, "d1": 3.0, "d5": 2.0}}
show(
    "names / recall / recip_rank",
    qrels3,
    run3,
    {"recall_1", "recall_3", "recall_10", "recip_rank", "ndcg_cut_3", "map"},
)
print("  (relevant at ranks 3 and 4; recip_rank should be 1/3 = 0.3333333333)")

# ---- Probe 4: how are grade-0 judgments treated? ----------------------------
qrels4 = {"q1": {"d1": 0, "d2": 1}}
run4 = {"q1": {"d1": 5.0, "d2": 4.0}}
show("grade-0 handling", qrels4, run4, {"recall_2", "recip_rank", "ndcg_cut_2"})
print("  (d1 judged non-relevant; recip_rank should be 1/2 = 0.5)")

# ---- Probe 5: tie-breaking direction ---------------------------------------
# Relevant doc tied with a non-relevant one. If trec_eval breaks ties by
# DESCENDING docid, "zzz" (non-relevant) outranks "aaa" (relevant).
qrels5 = {"q1": {"aaa": 1}}
run5 = {"q1": {"aaa": 1.0, "zzz": 1.0}}
show("tie-break direction", qrels5, run5, {"recip_rank", "recall_1"})
print("  recall_1 == 0.0  => ties break by DESCENDING docid (zzz first)")
print("  recall_1 == 1.0  => ties break by ASCENDING docid (aaa first)")
