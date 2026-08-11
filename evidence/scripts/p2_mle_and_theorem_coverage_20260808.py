"""PAPER2 Bernoulli-likelihood MLE estimate + theorem coverage simulation.

Part A (MLE closeness): for each of the eleven reconstructed conditions, estimate the logistic
parameter gamma by Bernoulli likelihood on the positive-coupling indicators
Y_i = 1[gamma_i > 0], gamma_MLE = logit(mean(Y)), and report its closeness to the pipeline proxy
(per-repetition mean of the normalized-distance gamma).

Part B (theorem coverage): simulate Bernoulli observations Y ~ Bernoulli(sigmoid(gamma)) and
compare the empirical CV of the logit-MLE estimator with the Proposition (Conditional precision
floor) lower bound 1 / (gamma sqrt(N s(1-s))) across gamma and N, including the gamma->0 boundary
and the degenerate s -> 0.5 limit where the bound diverges and small-N logit estimates are biased.

Deterministic (fixed seed). Outputs ../analyses/p2_mle_theorem_20260808.json.
"""

import json
import math
import os
import random
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(BASE, "analyses", "p2_mle_theorem_20260808.json")
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


def logit(p):
    if p <= 0.0 or p >= 1.0:
        return None
    return math.log(p / (1.0 - p))


def part_a():
    rows = []
    proxy_gammas = []
    mle_gammas = []
    for cond, fname in FILES.items():
        g = load_gammas(fname)
        p_pos = sum(1 for x in g if x > 0) / len(g)
        mle = logit(p_pos)
        proxy = st.mean(g)
        rows.append({
            "condition": cond,
            "n": len(g),
            "proxy_gamma_mean": round(proxy, 4),
            "p_positive_coupling": round(p_pos, 4),
            "gamma_mle_logit": round(mle, 4) if mle is not None else None,
        })
        if mle is not None and proxy > 0:
            proxy_gammas.append(proxy)
            mle_gammas.append(mle)
    n = len(proxy_gammas)
    r = st.correlation(proxy_gammas, mle_gammas) if n >= 2 else None
    return rows, {"n_conditions": n, "pearson_proxy_vs_mle": round(r, 4) if r is not None else None}


def part_b(seed=20260808, sims=5000):
    rng = random.Random(seed)
    out = []
    for gamma in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0]:
        s = 1.0 / (1.0 + math.exp(-gamma))
        for n in [5, 10, 30]:
            cvs = []
            for _ in range(sims):
                y = [1 if rng.random() < s else 0 for _ in range(n)]
                p = sum(y) / n
                if p in (0.0, 1.0):
                    continue
                ghat = math.log(p / (1.0 - p))
                cvs.append(ghat)
            if len(cvs) >= 100:
                emp_cv = st.stdev(cvs) / st.mean(cvs)
            else:
                emp_cv = None
            bound = 1.0 / (gamma * math.sqrt(n * s * (1.0 - s)))
            out.append({
                "gamma": gamma, "s": round(s, 4), "N": n,
                "lower_bound": round(bound, 4),
                "empirical_cv_logit_mle": round(emp_cv, 4) if emp_cv is not None else None,
                "bound_holds": emp_cv is not None and emp_cv >= bound,
                "n_valid_sims": len(cvs),
            })
    return out


def main():
    a_rows, closeness = part_a()
    b_rows = part_b()
    json.dump({
        "part_a_mle_closeness": {"method": "Y_i = 1[gamma_i>0]; gamma_MLE = logit(mean(Y)); proxy = per-repetition mean of normalized-distance gamma",
                                 "rows": a_rows, "summary": closeness},
        "part_b_theorem_coverage": {"method": "Y ~ Bernoulli(sigmoid(gamma)); gamma_hat = logit sample proportion; CV over 5000 simulations vs Proposition lower bound 1/(gamma sqrt(N s(1-s)))",
                                    "rows": b_rows},
    }, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("=== Part A: MLE closeness ===")
    print(f"{'condition':<32}{'proxy':>9}{'p_pos':>9}{'MLE':>9}")
    for r in a_rows:
        print(f"{r['condition']:<32}{r['proxy_gamma_mean']:>9}{r['p_positive_coupling']:>9}{str(r['gamma_mle_logit']):>9}")
    print("closeness:", closeness)
    print("=== Part B: theorem coverage ===")
    print(f"{'gamma':>6}{'N':>4}{'bound':>9}{'empCV':>9}  holds")
    for r in b_rows:
        print(f"{r['gamma']:>6}{r['N']:>4}{r['lower_bound']:>9}{str(r['empirical_cv_logit_mle']):>9}  {r['bound_holds']}")
    print("saved:", OUT)


if __name__ == "__main__":
    main()
