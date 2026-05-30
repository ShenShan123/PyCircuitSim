# B2 — Test-time seed ensemble + rail voting for SRAM force_ic (Track B, Tier 1)

**Date:** 2026-05-29 • **Branch:** `feat/v6.4.5` • **Verdict: KILL**
**Env:** `CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.
**Driver:** `experiments/v6_4_5_track_b/B2_seed_ensemble.py` → per-seed isolated
`BSIMAR_CHECKPOINT_DIR` via `scripts/eval_v6_4_5_candidate.py` (`--skip inv,ro,opamp`).

## Falsifier (plan B2)

> Promote: ≥ 1 ensemble seed snaps SRAM to rails (r < 0.05) AND inverter 8/8
> holds with ensemble OFF on non-`force_ic` cells.
> Hard kill: no seed snaps any SRAM cell to rails.

## Result — TSMC7 SRAM `force_ic` state-1 (q=VDD, qb=0), rail residual `r = max(|q−VDD|,|qb|)/VDD`

| config | rail_resid | q | qb |
|--------|-----------:|----:|----:|
| stock_s123 | 0.2775 | 0.816 | 0.208 |
| stock_s17 | 0.2858 | 0.815 | 0.214 |
| mono_s123 | 0.2889 | 0.815 | 0.217 |
| mono_s17 | 0.2923 | 0.815 | 0.219 |
| mono_s42 | 0.2992 | 0.815 | 0.224 |
| stock_s7 | 0.2993 | 0.815 | 0.225 |
| mono_s7 | 0.3011 | 0.815 | 0.226 |
| canonical_seed42 | 0.3019 | 0.815 | 0.226 |
| stock_s42 | 0.3090 | 0.815 | 0.232 |

best residual **0.2775** (stock_s123) ≫ 0.05 gate.

## Verdict: **KILL**

Every one of the 9 candidates (8 V6.4.2 Phase-7a seed siblings — stock &
mono, seeds {7,17,42,123} — plus the canonical V6.4.1 seed-42 slot) lands on
the **same** interior attractor: q ≈ 0.815, qb ≈ 0.21. No seed rail-snaps; the
spread in residual is tiny (0.278–0.309). A test-time seed ensemble cannot
help because **there is no rail-snapping basin in any sibling to vote for**.

**Diagnosis sharpened:** q itself nearly rails (0.816 ≈ VDD); the residual is
dominated by **qb stalling at ≈ 0.21 instead of 0**. In state-1 the qb
pull-down NMOS (gate = q ≈ VDD, so nominally ON) cannot sink qb against the
read-bias access transistor pulling up from the VDD bitline — i.e. the NMOS
**on-current / off-leak balance at (Vgs≈VDD, Vds≈0.2) is under-modelled**.
This is a true model-fidelity property shared across the whole DirectNet
family, confirming the V6.4.5 Track-A "true NN attractor" finding via a
fundamentally different probe (8 independent seeds vs warm-start).

→ SRAM rail-snap routes to a model change: B6 (adversarial harvest), B7
(physics skeleton), or B9 (monotone lattice). B4 (`force_ic` continuation) is
still run next as the final solver-path lever.
