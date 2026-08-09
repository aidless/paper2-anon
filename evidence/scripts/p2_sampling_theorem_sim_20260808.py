# -*- coding: utf-8 -*-
"""P2-A: TTRL multiplicative-update simulation for the empirical coupling proxy gamma-hat.
Model: K strategies, Bradley-Terry preference with strength gamma (P(Y=1)=sigmoid(gamma) for the
favored strategy), multiplicative reweighting (win x1.08, lose x0.96, floor 1e-3). Two domains A and B
share the preference mechanism; gamma-hat_N = ||w_A - w_B|| / ||w_B|| after N rounds.
Outputs E[gamma-hat], Var(gamma-hat) vs (gamma, N), compared with the CR bound 1/(N s(1-s))."""
import json, os, random, math, statistics as st, io

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "p2_sampling_theorem_sim_20260808.json")
K = 11
A_WIN, A_LOSE, FLOOR = 1.08, 0.96, 1e-3
LW, LL = math.log(A_WIN), math.log(A_LOSE)

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def run_trajectory(gamma, N, rng, seed_w0=True):
    """One trajectory: train w_A on domain A and w_B on domain B for N rounds each,
    preferences Bernoulli(sigmoid(gamma)) favoring the incumbent leader among two sampled strategies."""
    wA = [1.0 / K] * K
    wB = [1.0 / K] * K
    p = sigmoid(gamma)
    for _ in range(N):
        for w in (wA, wB):
            i, j = rng.sample(range(K), 2)
            # Y=1: strategy i wins (prob p), else j wins
            if rng.random() < p:
                winner, loser = i, j
            else:
                winner, loser = j, i
            w[winner] *= A_WIN
            w[loser] *= A_LOSE
            w[winner] = max(w[winner], FLOOR); w[loser] = max(w[loser], FLOOR)
            ssum = sum(w)
            for kk in range(K):
                w[kk] /= ssum
    # gamma-hat = ||wA - wB|| / ||wB||
    norm_diff = math.sqrt(sum((a - b) ** 2 for a, b in zip(wA, wB)))
    norm_b = math.sqrt(sum(b * b for b in wB))
    return norm_diff / norm_b if norm_b > 0 else 0.0

results = {"model": "K=11, BT preference P(win)=sigmoid(gamma), multiplicative x1.08/x0.96",
           "gamma_grid": [], "n_grid": [5, 10, 30]}
for gamma in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0]:
    row = {"gamma": gamma, "s": round(sigmoid(gamma), 4)}
    for N in results["n_grid"]:
        M = 2000
        rng = random.Random(12345)
        vals = [run_trajectory(gamma, N, rng) for _ in range(M)]
        mu = st.mean(vals); var = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
        cr = 1.0 / (N * sigmoid(gamma) * (1 - sigmoid(gamma)))
        row[f"N{N}"] = {"E_gamma_hat": round(mu, 4), "Var_gamma_hat": round(var, 6),
                        "CR_bound": round(cr, 4), "Var_over_CR": round(var / cr, 4) if cr else None}
        print(f"gamma={gamma} N={N}: E[gh]={mu:.4f} Var={var:.6f} CR={cr:.4f} Var/CR={var/cr:.4f}", flush=True)
    results["gamma_grid"].append(row)
json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved", OUT)
