# -*- coding: utf-8 -*-
"""P2-Si: bound-guided sample-size allocation (a deployment decision case).
For each of the 11 conditions, use the conditional precision floor
CV(N) >= 1/(gamma sqrt(N s(gamma)(1-s(gamma)))) to back out the N needed to reach a
target CV, and compare with a naive rule (e.g., "n>=5 seeds is enough").
Shows the bound changing an allocation decision: low-gamma conditions need far more
seeds than the naive rule, high-gamma conditions are fine at small N.
Input: eleven-condition gamma values (P2 tab:recon / p2_cv_sensitivity)."""
import os, json, math, io

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "analyses", "p2_bound_decision_case_20260808.json")

CONDS = {
    "DS Self-Eval":           0.0332,
    "DS x Qwen (qwen eval)":  0.1865,
    "Qwen x DS (official A)": 0.9866,
    "DS eval / Qwen exec (B)":0.8929,
    "GPT4o eval / DS exec (C)":0.7949,
    "GPT4o eval / Qwen exec (D)":0.7238,
    "Ablation max":           1.0384,
    "Ablation no-s0":         0.9790,
    "Qwen37":                 1.0591,
    "GPT4o replication (OLD)":1.1763,
    "DS self-eval r30":       0.9360,
}

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def n_for_target_cv(gamma, cv_target):
    """Invert CV(N) >= 1/(gamma sqrt(N s(1-s))) for N."""
    s = sigmoid(gamma)
    return 1.0 / (gamma ** 2 * cv_target ** 2 * s * (1 - s))

rows = []
for name, gamma in CONDS.items():
    n_10 = n_for_target_cv(gamma, 0.10)
    n_30 = n_for_target_cv(gamma, 0.30)
    rows.append({"condition": name, "gamma": gamma,
                 "N_for_CV0.10": round(n_10, 1),
                 "N_for_CV0.30": round(n_30, 1),
                 "naive_rule_ok": n_30 <= 5,
                 "decision": "small-N adequate" if n_30 <= 5 else ("moderate-N needed" if n_30 <= 30 else "large-N required (>30)" )})
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump({"case": "Bound-guided sample-size allocation for confirmatory evaluation",
           "method": "invert CV(N) >= 1/(gamma sqrt(N s(1-s))) for target CV",
           "rows": rows}, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"{'condition':26s} {'gamma':>7s} {'N@CV0.1':>9s} {'N@CV0.3':>9s}  decision")
for r in rows:
    print(f"{r['condition']:26s} {r['gamma']:7.3f} {r['N_for_CV0.10']:9.1f} {r['N_for_CV0.30']:9.1f}  {r['decision']}")
print("saved", OUT)
