"""PAPER2 CV-estimator sensitivity (full-archive CV vs 5-seed subsample CV).

Recomputes, for each of the eleven reconstructed conditions, the full-archive coefficient of
variation CV = sd(gamma)/mean(gamma) and the distribution of CV over five-seed subsamples.
The subsample distribution is estimated with a fixed-seed deterministic sample (B=2000) for
conditions with n>5, and by exhaustive enumeration for n<=5.

Usage: python verify_p2_cv_sensitivity_20260808.py
Outputs ../analyses/p2_cv_sensitivity_20260808.json and prints a table.
"""

import json
import math
import os
import random
import statistics as st
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(BASE, "analyses", "p2_cv_sensitivity_20260808.json")
EXP = r"<ARCHIVE_ROOT>\experiments"

FILES = {
    "DS Self-Eval": "mm_epc_multi_seed_ds_final.json",
    "DS x Qwen (qwen eval)": "mm_epc_multi_seed_final.json",
    "Qwen x DS (official A)": "official_A_qwen_eval_ds_exec.json",
    "DS eval / Qwen exec (official B)": "official_B_deepseek_eval_qwen_exec.json",
    "GPT4o eval / DS exec len500 (official C)": "official_C_gpt4o_eval_ds_exec_len500.json",
    "GPT4o eval / Qwen exec (official D)": "official_D_gpt4o_eval_qwen_exec.json",
    "Ablation max": "mm_epc_ablation_max.json",
    "Ablation no-s0": "mm_epc_ablation_no_s0.json",
    "Qwen37": "mm_epc_qwen37_final.json",
    "GPT4o replication (OLD)": "mm_epc_gpt4o_replication_OLD.json",
    "DS self-eval r30": "mm_epc_ds_selfeval_r30.json",
}


def load_gammas(fname):
    d = json.load(open(os.path.join(EXP, fname), encoding="utf-8"))
    res = None
    if isinstance(d, dict):
        for k in ("results", "repetitions"):
            if k in d and isinstance(d[k], list):
                res = d[k]
                break
        if res is None:
            for v in d.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    res = v
                    break
    elif isinstance(d, list):
        res = d
    g = []
    for r in res:
        if "gamma_TV" in r:
            g.append(float(r["gamma_TV"]))
        elif "gTV" in r:
            g.append(float(r["gTV"]))
    return g


def cv(vals):
    m = st.mean(vals)
    if m == 0.0:
        return float("inf") if st.stdev(vals) > 0 else 0.0
    return st.stdev(vals) / m


def subsample_cv_distribution(g, k=5, b=2000, seed=20260808):
    n = len(g)
    rng = random.Random(seed)
    if n <= k:
        if st.mean(g) == 0.0 and st.stdev(g) == 0.0:
            return [], 1
        return [cv(g)], 0
    if math.comb(n, k) <= 5000:
        idx = list(combinations(range(n), k))
    else:
        idx = [tuple(rng.sample(range(n), k)) for _ in range(b)]
    vals, degenerate = [], 0
    for ix in idx:
        sub = [g[i] for i in ix]
        if st.mean(sub) == 0.0 and st.stdev(sub) == 0.0:
            degenerate += 1
            continue
        vals.append(cv(sub))
    return vals, degenerate


def main():
    rows = []
    for cond, fname in FILES.items():
        g = load_gammas(fname)
        full = cv(g)
        dist, degenerate = subsample_cv_distribution(g)
        rows.append({
            "condition": cond,
            "archive": fname,
            "n": len(g),
            "full_archive_cv": round(full, 4),
            "cv5_subsample": {
                "min": round(min(dist), 4),
                "median": round(st.median(dist), 4),
                "max": round(max(dist), 4),
                "n_nondegenerate_subsets": len(dist),
                "n_all_zero_subsets": degenerate,
            },
        })
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"method": "CV = sd/mean of per-repetition gamma_TV (or gTV); 5-seed subsample distribution over all combinations (n<=15) or 2000 fixed-seed draws (n=30)",
               "rows": rows}, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{'condition':<32}{'n':>4}{'fullCV':>9}{'cv5 min':>9}{'cv5 med':>9}{'cv5 max':>9}")
    for r in rows:
        c5 = r["cv5_subsample"]
        print(f"{r['condition']:<32}{r['n']:>4}{r['full_archive_cv']:>9}{c5['min']:>9}{c5['median']:>9}{c5['max']:>9}")
    print("saved:", OUT)


if __name__ == "__main__":
    main()
