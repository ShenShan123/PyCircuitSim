# Phase 0 — Diagnostic P0-C: cap-swap / gds-swap RO ablation (TSMC7)

**Date:** 2026-06-01  •  **Branch:** `feat/v6.4.6`  •  **Status:** Done
**Env:** `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, `pycircuitsim` conda env.
**Plan:** `docs/plans/2026-06-01-directnet-v6.4.6-ro-sram.md` §4 (row P0-C), §6, §12 Q2.
**Script:** `scripts/v6_4_6_p0c_ablation.py`  •  **Log:** `results/v6_4_6/phase0_logs/p0c_ablation.log`

INSTRUMENTATION-ONLY — the swaps are done by an in-process monkeypatch of
`_MOSFETNNBase._eval`, restored between variants; no shipped file is modified,
no retrain, no checkpoint mutation. Ground truth is the OSDI binary via PyCMG.

## Command

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  stdbuf -oL -eL conda run --no-capture-output -n pycircuitsim \
  python -u scripts/v6_4_6_p0c_ablation.py \
  > results/v6_4_6/phase0_logs/p0c_ablation.log 2>&1
```

## What was run

The TSMC7 DirectNet 5-stage ring-osc transient was run **four ways** and the
period measured each time with `_period_from_wave(mid=VDD/2, SETTLE=0.3 ns)`,
the gate's own estimator. `NN_BATCHED_EVAL=0` was forced for ALL variants so
the per-device `_eval` is the single source of truth the monkeypatch sees; the
baseline period under this flag was validated to be **bit-identical (50.82 ps)**
to the gate's default-flag baseline before trusting the swaps.

| variant | what is substituted at the live bias |
|---------|--------------------------------------|
| baseline | nothing (unmodified NN) |
| cap-swap | all five NN autograd caps (cgg,cgd,cgs,cdg,cdd) → analytic OSDI caps; NN id/gds kept |
| gds-swap | NN gds → analytic OSDI gds, then the SAME `max(\|id\|·0.5,1e-12)` floor re-applied; NN caps/id kept |
| gds-swap-nofloor | NN gds → raw analytic OSDI gds (only the 1e-12 numerical floor); disambiguates whether the floor masks the swap |

Convention alignment (verified in P0-B by finite-difference vs the OSDI
instance): caps injected in NN autograd convention — diagonals cgg/cdd same
sign, off-diagonals cgd/cdg and the source term cgs negated (`NN = −OSDI`). gds
both positive → direct. OSDI is ~1.6 ms/call, so results are memoised on the
bias rounded to 1 mV (94–97 % hit rate; the rounding is identical across
variants so it cannot bias the *comparison*).

## Results

```
NGSPICE ground truth period = 46.64 ps   (gate ≤5% → DirectNet must reach ≤48.97 ps)
```

| variant          | period (ps) | perErr % vs NG | NRMSE % vs NG | R² vs NG |
|------------------|------------:|---------------:|--------------:|---------:|
| baseline         |   **50.82** |       **8.97** |         54.65 |  −0.6692 |
| cap-swap         |       50.82 |           8.97 |         54.65 |  −0.6692 |
| gds-swap         |       50.83 |           8.98 |         64.57 |  −1.3305 |
| gds-swap-nofloor |       50.83 |           8.98 |         62.78 |  −1.2030 |

(OSDI-cache stats — cap-swap 12 810 calls / 219 040 hits; gds-swap 18 382 /
420 058; gds-nofloor 22 167 / 624 843. Wall 498/584/424/373 s.)

## Interpretation

- **cap-swap is BIT-FOR-BIT identical to baseline** (period, NRMSE, R² all
  unchanged to the printed precision). Injecting the exact analytic OSDI caps
  changes nothing. This is the causal confirmation of the D1
  (`NN_SYMMETRIC_CAPS` bit-for-bit) prior and of the P0-B finding that the
  load-bearing caps already track OSDI: **caps are EXONERATED.**
- **gds-swap moves the period by 0.01 ps (8.97 → 8.98 %), i.e. NOT AT ALL** —
  the wrong direction would be *toward* 46.64 ps; it nudged the other way and
  the waveform NRMSE got *worse* (54.65 → 64.57 %). Re-injecting the true OSDI
  current-slope does not close the walk.
- **gds-swap-nofloor (no `|id|·0.5` floor) also leaves the period at 50.83 ps**
  — proving the null is not an artifact of the floor masking the swap. The gds
  the solver consumes is simply **not what sets the RO period**.
- The two surfaces a Jacobian-distillation retrain could bend (gds, caps) are
  **both causally inert for the period**. What remains untouched by both swaps
  is the **`id` value** along the dynamic trip trajectory and the **charge
  state** `qg/qd/qb → qs=-(qg+qd+qb)` fed to the BDF-2 integrator. The 4.2 ps
  must live there (id-value residual integrated over the cycle, or
  charge-reconstruction / numerical-integration error), not in the slope/cap
  Jacobians.

## DECISION (plan §4 decision tree, P0-B/C branch)

> **CAP-SWAP unchanged (bit-identical) ⇒ caps EXONERATED.
> GDS-SWAP (floored AND unfloored) unchanged ⇒ the gds surface is causally
> inert for the RO period.
> ⇒ This is the plan's "RO owner = neither gds nor caps" branch:**
> **the Phase-2 RO Jacobian-distillation lever (gds- OR charge-distillation,
> and the deferred Softplus-cap-head split) is KILLED before any GPU.**

Consequences for V6.4.6:
- **Do NOT build Phase-2 LoRA + Jacobian distillation for the RO gate.** No
  Jacobian surface the NR solver consumes moves the TSMC7 period; the EV of a
  gds- or cap-distillation retrain on the RO gate is **zero** (P0-B's divergent
  gds is real but inert; caps already correct).
- The 4.2 ps is in the **`id`-value / `qs`-reconstruction / BDF-2-truncation**
  bucket — a solver-integration / charge-conservation scope that **no phase in
  the current plan addresses**. Per plan §11 risk row 1, V6.4.6 should ship
  **SRAM-only** (whatever Phase 1 yields) and open a scoped BDF-2/`qs`/dynamic-
  id investigation for V6.4.7; the TSMC7 RO gate stays open and is **not**
  closable by a GPU retrain on this architecture.
- The from-scratch split-head rewrite (deferred in plan §10) is **further
  de-risked into irrelevance for RO**: the P0-C cap ablation exonerates the
  Cgd/Cgg shape it would fix.
