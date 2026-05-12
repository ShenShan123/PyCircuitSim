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
reverted.** The session converged on a structural diagnosis:

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

1. **Code-path agent** — identified that `_vdd_estimate` (rule 19a) is
   computed universal-wide, not per-tech; the Tier-1A trust-region
   clamp pins per-iteration `|ΔV| ≤ VDD` so NR oscillates at the
   ceiling; there is **no "no global progress over N iters" watchdog**;
   the oscillation-acceptance path averages the last 3 snapshots and
   accepts them — explains why TSMC5 VTC returns a smoothed-but-wrong
   59,286 % bounded number instead of a clean NR_FAIL.
2. **Dataset agent** — over `universal_nmos.npz` (12.33 M TSMC rows),
   TSMC5 is **not** under-sampled on any off-(Vgs, Vds) axis. Row
   counts within 1.1× of mean; identical T-grid; zero starved cells
   in the joint `(L, NFIN, T, Vbs-sign)` 4-tuple. **Not a coverage
   problem.**
3. **Web research** — surfaced MSH arclength continuation (DATE 2024),
   ATANSH homotopy (Roychowdhury 2006), Levenberg-Marquardt damping,
   AutoPINN structural Vds transform (Micromachines 2023), KAN paper
   (Novkin 2025) finding that NR convergence is dominated by
   second-derivative smoothness rather than per-point Id NRMSE.

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
  the oscillation-acceptance path; (c) 50-step source-stepping retry
  in `_solve_dc_with_retry` for NN circuits.
- **TSMC5 DN VTC: 59,286 % → 68.22 %**, but the value moved because
  the failure mode *changed shape* — from smoothed oscillation around
  a wrong attractor to a stable wrong fixed point at V_out ≈
  0.80–0.91 V (1.3×VDD). The 5×VDD threshold was too lax to catch a
  1.3×VDD attractor.
- **Wall-time exploded.** The 50-step source-stepping retry burned
  uncapped wall-time per Vin step. 16-cell verify timed out before
  reaching TSMC16 VTC + any transient.
- **Diagnostic value:** revealed the true failure mode (stable wrong
  attractor above rail, not NR oscillation).

### Tier S2 — Bounded-Vout watchdog + Vin-bisection — REVERTED
- Replaced S1's 5×VDD threshold with a tighter 1.2×VDD bounded-Vout
  watchdog at the sweep-step level. Added Vin-bisection retry (up to
  4 levels, 5 s wall-time cap per step).
- **TSMC5 DN VTC unchanged at 59,286 %** because the wrong attractor
  exists at *every* Vin in the sweep — bisection has no
  "known-good" previous solution to bootstrap from.
- **TSMC7 VTC 9.97→5.39 %, TSMC12 VTC 6.50→4.55 %** were genuine
  improvements: bisection helped on cells where the wrong attractor
  is localized to the trip cone.
- **Wall-time still exceeded the 40-min cap** because the 5 s × 130
  sweep points × 4 techs of bisection budget = unbounded total.

### Tier M1 — B0-medium asinh DN retrain — REVERTED
- Trained `v6_dn_medium_e2_asinh_{nmos,pmos}` with 340 K params (6×
  the V6 small probe), 200 epochs, same recipe.
- **First attempt corrupted by disk-pressure / process kill** at
  ~80 min in. `torch.save` truncated mid-write, checkpoint zip missing
  central directory. Re-ran cleanly without `conda run` (direct python
  invocation with `PYTHONUNBUFFERED=1`); both runs completed in ~140
  minutes total.
- **TSMC5 DN VTC: 59,286 → 50.48 %** (catastrophic → graceful, still
  FAIL). **TSMC7 DN VTC: 9.97 → 2.85 %** big win.
- **TSMC16 DN VTC: 5.40 PASS → 16.10 FAIL** — hard regression. The
  capacity bump shifted the wrong-attractor problem from TSMC5 to
  TSMC16 rather than removing it.
- Consistent with the KAN paper finding: NR convergence is governed
  by second-derivative smoothness, not Id NRMSE. The medium model has
  lower training NRMSE but a different Jacobian shape that puts
  TSMC16's trip cone into a wrong-attractor basin.

### Tier M2 — TSMC5-only residual head — REVERTED
- 1,844-param residual MLP head on the frozen V6 small-probe backbone,
  conditioned on `tech_code ∈ {0,1,2,3}`. Final linear init to zero so
  residual starts as additive identity.
- Trained NMOS+PMOS for 30 epochs on TSMC5-only rows (1.9 M / 2.0 M).
  Training-space val improvement: **0.17 % / 0.31 %** — backbone
  already saturates the in-distribution fit; the residual has no
  significant gradient signal.
- **TSMC5 DN VTC: 59,286 → 59,266 %** (unchanged catastrophic).
  **TSMC5 DN tran: 17.76 → 15.69 %** (+2.07 pp, still FAIL marginally).
- TSMC7 VTC 9.97 → 5.42 % (unexpected, gating-by-tech-code leakage);
  TSMC7 tran 10.88 → 14.38 % (−3.5 pp regression violating the 0.3 pp
  gate). TSMC12/16 byte-identical ✓.
- **The residual cannot fix the wrong attractor** because the wrong
  attractor lives at V_out > VDD — out-of-distribution at training
  time, no gradient signal points the residual head toward "make Id
  non-zero at V_out > VDD to restore the rail."

## Final structural diagnosis

The four tiers collectively prove a structural property of the
TSMC5 DN inverter:

1. **It is not a solver problem.** S1 (progress watchdog +
   bounded-numeric kill + retry escalation) and S2 (Vin-bisection)
   both leave the wrong fixed point intact. Bisection finds the same
   basin from every starting Vin.
2. **It is not a model-capacity problem.** M1 (6× capacity) only
   shape-shifts the wrong-attractor pattern across techs.
3. **It is not an in-distribution training problem.** M2 (TSMC5-only
   residual head, frozen backbone) saturates the in-distribution val
   improvement at 0.17 % while the VTC catastrophic value is
   unchanged.
4. **It is not a data-coverage problem on standard axes.** TSMC5 row
   counts and joint cell coverage match peers within 1.1×.

The remaining hypothesis: the wrong attractor lives in **extrapolated
territory** (V_out > VDD) where rule 19a's rail-restoring
extrapolation provides the only "physics" — and at TSMC5's specific
operating point, it is not strong enough.

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
