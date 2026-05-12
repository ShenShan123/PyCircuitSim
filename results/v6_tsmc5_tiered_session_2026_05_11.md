# V6 — TSMC5 inverter accuracy tiered session

**Date:** 2026-05-11..12 (one continuous session)
**Branch:** `feat/v6` (zero shipping commits this session — all four
attempts reverted)
**Plan doc:** `docs/superpowers/plans/2026-05-11-tsmc5-inverter-tiered-fix.md`
**Goal:** lift TSMC5 DN inverter VTC out of catastrophic 59,286 % NR
blow-up and TSMC5 DN inverter transient below the 15 % gate (was
17.76 %), without regressing the 14 PASSing cells on TSMC7/12/16.

## TL;DR

Four orthogonal levers tried (solver progress watchdog, Vin-bisection,
medium-capacity retrain, TSMC5-only residual head). **All four
reverted.** Structural diagnosis:

> TSMC5 DN inverter VTC's catastrophic failure is a **stable wrong
> fixed point at V_out ≈ 1.3 × VDD** — extrapolated territory that no
> in-distribution training signal or solver tweak can reach. Capacity
> alone shifts the wrong-attractor pattern (TSMC5 → TSMC16) rather
> than removing it.

V6 shipping set unchanged.

## Starting state — V6 shipping (`v6_dn_small_e2_asinh_*`)

| Cell | TSMC5 | TSMC7 | TSMC12 | TSMC16 |
|---|---:|---:|---:|---:|
| DN inverter VTC | **NR_FAIL (59,286 % bounded)** | 9.97 PASS | 6.50 PASS | 5.40 PASS |
| DN inverter tran | **17.76 FAIL** | 10.88 PASS | 3.79 PASS | 8.90 PASS |

14/16 DN cells PASS. Both failures on TSMC5.

## Diagnostic ground (three parallel agents, session start)

1. **Code-path agent** — `_vdd_estimate` (rule 19a) is universal-wide,
   not per-tech; Tier-1A trust-region clamp pins `|ΔV| ≤ VDD` so NR
   oscillates at ceiling; **no global-progress watchdog**; oscillation-
   acceptance averages last 3 snapshots — explains why TSMC5 VTC
   returns smoothed-but-wrong 59,286 % bounded instead of clean NR_FAIL.
2. **Dataset agent** — over `universal_nmos.npz` (12.33 M TSMC rows),
   TSMC5 is **not** under-sampled. Row counts within 1.1× of mean;
   identical T-grid; zero starved cells in `(L, NFIN, T, Vbs-sign)`.
   **Not a coverage problem.**
3. **Web research** — surfaced MSH arclength continuation (DATE 2024),
   ATANSH homotopy (Roychowdhury 2006), Levenberg-Marquardt damping,
   AutoPINN structural Vds transform (Micromachines 2023), KAN paper
   (Novkin 2025): NR convergence dominated by second-derivative
   smoothness, not per-point Id NRMSE.

## Results (every cell, every tier)

| Tier | TSMC5 VTC | TSMC5 tran | TSMC7 VTC | TSMC7 tran | TSMC12 VTC | TSMC12 tran | TSMC16 VTC | TSMC16 tran | Total PASS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V6 baseline | 59,286 | 17.76 | 9.97 | 10.88 | 6.50 | 3.79 | 5.40 | 8.90 | 6/8 DN |
| S1 | 68 (wrong fp) | wall-time | wall-time | wall-time | wall-time | wall-time | wall-time | wall-time | n/a |
| S2 | 59,286 | wall-time | **5.39** | wall-time | **4.55** | wall-time | wall-time | wall-time | n/a |
| M1 | 50.48 | 16.38 | **2.85** | 14.41 | 10.28 | 3.91 | **16.10 FAIL** | 8.99 | 5/8 DN |
| M2 | 59,266 | 15.69 | 5.42 | 14.38 | 6.50 | 3.79 | 5.40 | 8.90 | 6/8 DN |

(Bold = notable change. "wall-time" = the cell could not complete
inside the 40-min verify timeout because the tier's escalation logic
burned per-step budget at TSMC5.)

## Per-tier postmortem

### Tier S1 — Solver progress watchdog + bounded-numeric kill — REVERTED
- Three hunks added to `pycircuitsim/solver.py`: (a) min-residual
  progress watchdog over 20 iters; (b) 5×VDD bounded-numeric kill in
  oscillation-acceptance path; (c) 50-step source-stepping retry in
  `_solve_dc_with_retry` for NN circuits.
- **TSMC5 DN VTC: 59,286 % → 68.22 %** — but failure mode *changed
  shape* from smoothed oscillation to stable wrong fixed point at
  V_out ≈ 0.80–0.91 V (1.3×VDD). 5×VDD threshold too lax for 1.3×VDD.
- **Wall-time exploded.** 50-step source-stepping retry burned uncapped
  per-Vin time. 16-cell verify timed out before TSMC16 VTC.
- **Diagnostic value:** revealed true failure mode (stable wrong
  attractor above rail, not NR oscillation).

### Tier S2 — Bounded-Vout watchdog + Vin-bisection — REVERTED
- Replaced S1's 5×VDD with tighter 1.2×VDD bounded-Vout watchdog at
  sweep-step level. Added Vin-bisection retry (up to 4 levels, 5 s
  per-step cap).
- **TSMC5 DN VTC unchanged at 59,286 %** — wrong attractor exists at
  *every* Vin; bisection has no "known-good" previous solution.
- **TSMC7 VTC 9.97→5.39 %, TSMC12 VTC 6.50→4.55 %** — genuine wins
  where wrong attractor is localized to trip cone.
- **Wall-time exceeded 40-min cap** — 5 s × 130 points × 4 techs.

### Tier M1 — B0-medium asinh DN retrain — REVERTED
- Trained `v6_dn_medium_e2_asinh_{nmos,pmos}` with 340 K params (6×
  V6 small probe), 200 epochs, same recipe.
- **First attempt corrupted** by disk-pressure / process kill at ~80
  min. `torch.save` truncated; missing zip central directory. Re-ran
  without `conda run` (direct python, `PYTHONUNBUFFERED=1`); ~140 min
  total.
- **TSMC5 DN VTC: 59,286 → 50.48 %** (catastrophic → graceful FAIL).
  **TSMC7 DN VTC: 9.97 → 2.85 %** big win.
- **TSMC16 DN VTC: 5.40 PASS → 16.10 FAIL** — hard regression. Capacity
  bump shifted wrong-attractor from TSMC5 to TSMC16, not removed.
- Consistent with KAN paper: NR convergence governed by second-
  derivative smoothness, not Id NRMSE. Medium model has lower training
  NRMSE but different Jacobian shape putting TSMC16 trip cone into
  wrong-attractor basin.

### Tier M2 — TSMC5-only residual head — REVERTED
- 1,844-param residual MLP head on frozen V6 small-probe backbone,
  conditioned on `tech_code ∈ {0,1,2,3}`. Final linear init zero
  (additive identity).
- Trained NMOS+PMOS 30 epochs on TSMC5-only rows (1.9 M / 2.0 M).
  Training val improvement: **0.17 % / 0.31 %** — backbone already
  saturates in-distribution fit; residual has no gradient signal.
- **TSMC5 DN VTC: 59,286 → 59,266 %** (unchanged catastrophic).
  **TSMC5 DN tran: 17.76 → 15.69 %** (+2.07 pp, still marginal FAIL).
- TSMC7 VTC 9.97 → 5.42 % (gating-by-tech-code leakage); TSMC7 tran
  10.88 → 14.38 % (−3.5 pp, violates 0.3 pp gate). TSMC12/16
  byte-identical.
- **Residual cannot fix wrong attractor** — it lives at V_out > VDD,
  out-of-distribution at train time, no gradient signal toward "make
  Id non-zero at V_out > VDD to restore the rail."

## Final structural diagnosis

Four tiers collectively prove:

1. **Not a solver problem.** S1 (watchdog + bounded-numeric kill +
   retry) and S2 (Vin-bisection) both leave wrong fixed point intact.
   Bisection finds the same basin from every starting Vin.
2. **Not a model-capacity problem.** M1 (6× capacity) shape-shifts
   wrong-attractor pattern across techs.
3. **Not an in-distribution training problem.** M2 saturates val
   improvement at 0.17 % while VTC catastrophic value unchanged.
4. **Not a data-coverage problem on standard axes.** TSMC5 row counts
   match peers within 1.1×.

Remaining hypothesis: wrong attractor lives in **extrapolated territory**
(V_out > VDD) where rule 19a rail-restoring is the only "physics" — and
at TSMC5's operating point, not strong enough.

## Recommendations for the next session (out of this plan's scope)

1. **Structural input transform** — AutoPINN-style
   `Vds_new = sign(Vds)·(√(Vds²+γ²)−γ)` at the model input, OR learn
   rule 19a's `g_max`/`x_ref` rail-restoring constants as model
   parameters. Forbids the wrong attractor by construction.
   Retrain-from-scratch.
2. **Rail-overshoot training rows** — sample PyCMG at synthetic
   V_out > VDD points with the true restoring current and add them to
   the training set. Invalidates the 2026-05-10 plan's "no overshoot
   overlay" rule — needs a fresh A/B against V5 `inv_trip` regressions.
3. **MSH arclength continuation** (DATE 2024) — the only solver-side
   lever orthogonal to S1/S2 failures, with a published track record
   on HiZ-node inverter VTCs.

## Artefacts on disk

- `tests/v6_logs/m1b_summary.csv`, `m2_summary.csv` — 16-cell numerics.
- `tests/v6_logs/{m1b_nmos,m1b_pmos,m1b_verify,m2_train_nmos,m2_train_pmos,m2_verify}.log` —
  training + verify transcripts.
- `external_compact_models/bsimar/models/tsmc5_residual.py`,
  `external_compact_models/bsimar/training/tsmc5_residual_train.py`
  — M2 implementation, untracked, kept for reproducibility.
- `docs/superpowers/plans/2026-05-11-tsmc5-inverter-tiered-fix.md`
  — full plan with per-tier Outcomes table and diagnostic subsections.
- V6 shipping checkpoints `v6_dn_small_e2_asinh_{nmos,pmos}*` —
  unchanged on disk; inference path byte-identical.
