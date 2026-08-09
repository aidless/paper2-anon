# -*- coding: utf-8 -*-
"""P2: bound-guided vs n=100 heuristic on Chatbot Arena top-5 recovery.
Ground truth: full-sample win-rate ranking. Heuristic: n=100 battles/model.
Bound-guided: n* = p(1-p)/SE_target^2 battles/model (inverted Bernoulli SE), SE_target=0.03.
Compares total observations and top-5 recovery rate (bootstrap)."""
import os, json, random, statistics as st, zipfile, io

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
_cand = [os.path.join(HERE, "arena_data_cache.zip"), os.path.join(os.path.dirname(HERE), "arena_data_cache.zip")]
ZIP = next((c for c in _cand if os.path.exists(c)), _cand[0])
OUT = os.path.join(BASE, "analyses", "p2_bound_vs_heuristic_20260808.json")

with zipfile.ZipFile(ZIP) as z:
    cache = json.loads(z.read("arena_data_cache.json").decode("utf-8"))
scores = {}
for m, battles in cache["model_battles"].items():
    vals = []
    for b in battles:
        r = b.get("result")
        vals.append(1.0 if r == "win" else (0.0 if r == "loss" else 0.5))
    scores[m] = vals
models = list(scores.keys())

def top5_from(means):
    return tuple(m for m, _ in sorted(means.items(), key=lambda x: -x[1])[:5])

full = top5_from({m: st.mean(v) for m, v in scores.items()})
print("full top-5:", full)

def sample_top5(vals_by_model, n, rng):
    return top5_from({m: st.mean(rng.sample(v, n)) for m, v in vals_by_model.items() if len(v) >= n})

full5set = set(full)

rng = random.Random(7)
HEUR_N = 100
heur_hits = 0
heur_overlap = 0.0
for _ in range(1000):
    q = sample_top5(scores, HEUR_N, rng)
    if q == full:
        heur_hits += 1
    heur_overlap += len(set(q) & full5set) / 5.0
heur_rate = heur_hits / 1000
heur_total = HEUR_N * len(models)

SE_TARGET = 0.03
n_star = {}
for m, v in scores.items():
    p = st.mean(v)
    n_star[m] = max(50, int(p * (1 - p) / SE_TARGET ** 2))
bound_total = sum(n_star.values())
# bound-guided uses per-model n*; to keep comparison fair, use min n* for all? No: use n* capped at available
n_eff = {m: min(n_star[m], len(scores[m])) for m in models}
bound_hits = 0
bound_overlap = 0.0
for _ in range(1000):
    q = top5_from({m: st.mean(rng.sample(scores[m], n_eff[m])) for m in models if len(scores[m]) >= n_eff[m]})
    if q == full:
        bound_hits += 1
    bound_overlap += len(set(q) & full5set) / 5.0
bound_rate = bound_hits / 1000
bound_overlap_rate = bound_overlap / 1000
bound_total_eff = sum(n_eff.values())
heur_overlap_rate = heur_overlap / 1000

print(f"heuristic n=100: total={heur_total} top5_exact={heur_rate:.4f} top5_overlap={heur_overlap_rate:.4f}")
print(f"bound-guided n*: total={bound_total_eff} top5_exact={bound_rate:.4f} top5_overlap={bound_overlap_rate:.4f}")
print(f"total ratio={bound_total_eff/heur_total:.2f}; overlap delta={bound_overlap_rate-heur_overlap_rate:+.4f}")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump({"case": "bound-guided vs n=100 heuristic for Chatbot Arena top-5 recovery",
           "full_top5": list(full), "heur_n": HEUR_N, "heur_total": heur_total, "heur_rate": round(heur_rate, 4), "heur_overlap": round(heur_overlap_rate, 4), "heur_overlap_pct": round(heur_overlap_rate * 100, 1),
           "se_target": SE_TARGET, "n_star": {m: n_eff[m] for m in models},
           "bound_total": bound_total_eff, "bound_rate": round(bound_rate, 4), "bound_overlap": round(bound_overlap_rate, 4),
           "total_ratio": round(bound_total_eff / heur_total, 3)},
          open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved", OUT)
