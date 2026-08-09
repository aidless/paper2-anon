# PAPER2 evidence package (Revision13)

Zero-budget robustness analyses added in Revision13 (2026-08-08):

- scripts/verify_p2_cv_sensitivity_20260808.py — full-archive CV vs five-seed subsample CV for the eleven reconstructed conditions (Table 5). Output: analyses/p2_cv_sensitivity_20260808.json.
- scripts/p2_mle_and_theorem_coverage_20260808.py — Bernoulli-likelihood MLE closeness (Y_i = 1[gamma_i>0]) and simulation coverage of the Proposition (conditional precision floor). Output: analyses/p2_mle_theorem_20260808.json.

Both scripts are deterministic and read the persisted per-repetition archives under F:\Research\experiments (the same sources used for the eleven-condition reconstruction table).
