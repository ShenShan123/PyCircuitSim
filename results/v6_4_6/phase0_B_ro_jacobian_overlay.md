# Phase 0 — Diagnostic P0-B: RO-trip Jacobian overlay (TSMC7)

**Date:** 2026-06-01  •  **Branch:** `feat/v6.4.6`  •  **Status:** Done
**Env:** `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, `pycircuitsim` conda env.
**Plan:** `docs/plans/2026-06-01-directnet-v6.4.6-ro-sram.md` §2, §4 (row P0-B), §12 Q2.
**Script:** `scripts/v6_4_6_p0b_ro_overlay.py`  •  **Log:** `results/v6_4_6/phase0_logs/p0b_ro_overlay.log`

INSTRUMENTATION-ONLY — no shipped-behaviour change, no retrain, no checkpoint
mutation. Ground truth is the OSDI binary via PyCMG (`eval_single_point`).

## Command

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  conda run -n pycircuitsim python scripts/v6_4_6_p0b_ro_overlay.py \
  > results/v6_4_6/phase0_logs/p0b_ro_overlay.log 2>&1
```

## What was measured

1. Ran the TSMC7 DirectNet 5-stage ring-osc transient via the gate's own
   `run_directnet_ro(BENCH["TSMC7"], …)`. **DN period reproduced = 50.82 ps**
   (baseline 50.82 ps ✓; NGSPICE 46.64 ps; period error 8.97 %).
2. Selected the **stage-5 inverter that drives n5**: NMOS `Mn5` (d=n5, g=n4,
   s=0, b=0) and PMOS `Mp5` (d=n5, g=n4, s=vdd, b=vdd). Gate = predecessor n4.
3. Took **one full post-settle oscillation cycle**: 26 timesteps,
   346.00 → 396.00 ps (50.00 ps span, = one period).
4. At each timestep reconstructed the absolute terminal bias from the n5/n4
   node waveforms and computed, at the SAME bias:
   - **post-Rule-15** NN `gm, gds, cgg, cgd, cgs, cdg, cdd` via `mosfet._eval({…})`
     (exactly what the NR solver consumes),
   - analytic **OSDI** `gm, gds, …` via `eval_single_point` on a cached
     per-device-type PyCMG instance.

## Sign / convention alignment (verified by finite-difference vs the OSDI instance)

The OSDI `eval_dc` outputs and the NN `result` dict do NOT share a uniform
convention; each surface was reconciled explicitly (FD against the OSDI
instance, `h=1e-4`) so the overlay is apples-to-apples in the **NN convention
the solver consumes**:

| Surface | NN `result[...]` | OSDI raw | Alignment used |
|---------|------------------|----------|----------------|
| `id`  | PyCMG sign (NMOS<0, PMOS>0)        | same                      | direct |
| `gm`  | `-d(id)/dVg` (positive)           | `-d(id)/dVg` (positive)   | **direct** |
| `gds` | `+d(id)/dVd`, then floored to ≥`\|id\|·0.5` | `-d(id)/dVd` (true physical, positive) | **direct** (both positive); floor-binding flagged |
| `cgg` | `+d(qg)/dVg`                      | `+d(qg)/dVg`              | same sign, direct |
| `cdd` | `+d(qd)/dVd`                      | `+d(qd)/dVd`              | same sign, direct |
| `cgd` | `+d(qg)/dVd`                      | `-d(qg)/dVd`              | **NN = −OSDI** → negate OSDI |
| `cdg` | `+d(qd)/dVg`                      | `-d(qd)/dVg`              | **NN = −OSDI** → negate OSDI |
| `cgs` | `+d(qg)/dVs`                      | `-d(qg)/dVs`              | NN = −OSDI; **not NR-load-bearing** (grounded source) |

FD confirmation (NMOS, bias d=0.40 g=0.45 s=0 b=0):
`OSDI cgg=+1.241e-16 = FD d(qg)/dVg`; `OSDI cgd=+1.735e-17 = −FD d(qg)/dVd`;
`OSDI cdg=+5.550e-17 = −FD d(qd)/dVg`; `OSDI cdd=+1.747e-17 = +FD d(qd)/dVd`.
`OSDI gm=−FD d(id)/dVg`, `OSDI gds=−FD d(id)/dVd`.

**gds-floor binding:** the NN's raw autograd `gds` is near-zero/negative in
saturation, so `_floor_gds` (`max(\|id\|·0.5, 1e-12)`) replaces it. Along this
cycle the floor binds on **NMOS 2/26, PMOS 4/26** timesteps; the divergence
below is broader than the floor, i.e. it is not purely a floor artifact.

## Rule-16 overlay — NN (post-Rule-15) vs OSDI, over the 26-point cycle

### NMOS (stage-5, drives n5)

| Surface | MRE % | R² | NRMSE % | MaxErr |
|---------|------:|------:|--------:|-------:|
| **gds** | 417.79 | 0.2352 | **22.94** | 822.1 µS |
| gm  | 88.15 | 0.9664 | 5.56 | 63.91 µS |
| id  | 44.57 | 0.9480 | 5.93 | 17.57 µA |
| cgd | 10.56 | 0.2127 | 39.50 | 145.4 aF |
| cgg | 0.88 | 0.9999 | 0.54 | 2.34 aF |
| cdg | 4.18 | 0.9994 | 1.08 | 1.44 aF |
| cdd | 7.44 | 0.6209 | 27.17 | 70.99 aF |
| cgs | 97.26 | −1.6273 | 58.30 | 116.2 aF (grounded-source, not NR-load-bearing) |

### PMOS (stage-5, drives n5)

| Surface | MRE % | R² | NRMSE % | MaxErr |
|---------|------:|------:|--------:|-------:|
| **gds** | 534.15 | 0.3426 | **20.44** | 453.9 µS |
| gm  | 525.54 | 0.9931 | 2.71 | 30.51 µS |
| id  | 85.78 | 0.9895 | 2.99 | 7.26 µA |
| cgd | 4.49 | 0.9489 | 9.75 | 37.59 aF |
| cgg | 0.55 | 0.9998 | 0.54 | 4.17 aF |
| cdg | 1.51 | 0.9996 | 0.91 | 3.83 aF |
| cdd | 4.35 | 0.9727 | 7.10 | 19.62 aF |
| cgs | 99.43 | −2.2815 | 69.73 | 141.1 aF (grounded-source, not NR-load-bearing) |

> MRE is relative to `\|OSDI\|` with a per-surface scale floor (`1e-3·max\|OSDI\|`)
> so a single near-zero subthreshold true value does not blow up the mean.
> The **NRMSE / R²** columns (ptp- and variance-normalised) are the
> dynamic-range-honest read for a slope NR integrates over a full swing.

## Interpretation

- **`gds` is the dominant divergent surface on BOTH devices.** NRMSE 22.94 %
  (NMOS) / 20.44 % (PMOS), R² 0.24 / 0.34, MaxErr 822 / 454 µS. The
  post-Rule-15 gds NR actually consumes is *systematically wrong* across the
  trip cycle — far outside the ~10 % "already tracks OSDI" bar. The floor binds
  on only 2–4/26 points, so this is a genuine slope error, not a floor artifact.
- **`gm` and `id` track well in dynamic range** (NRMSE 5.6 %/2.7 % gm,
  5.9 %/3.0 % id; R² > 0.94). High MRE is the small-value subthreshold band, not
  the load-bearing on-state — consistent with the already-excellent VTC fit
  (1.21–2.37 %). The *value* fit is fine; the *Vd-slope* is not.
- **Caps are largely exonerated** (the D1 `NN_SYMMETRIC_CAPS` bit-for-bit prior):
  cgg 0.5 %, cdg ~1 %, R² ≈ 1.0 on both devices. The only soft spots are
  **cgd-NMOS (39.5 % NRMSE but MaxErr 145 aF)** and **cdd-NMOS (27 % NRMSE,
  71 aF)** — tiny in absolute terms vs the 0.5 fF stage load and the dominant
  cgg/cdg, so unlikely to own a 4.2 ps walk. PMOS caps all ≤9.75 % NRMSE.
- `cgs` mismatch is expected and **not NR-load-bearing**: both stage devices
  have a grounded/rail-pinned source, so `∂q/∂Vs` is never integrated.

## DECISION (plan §4 decision tree, P0-B branch — SUPERSEDED by the P0-C causal test)

> **P0-B (correlational read):** the post-Rule-15 **gds surface DIVERGES** from
> OSDI (NRMSE ~20–23 %, R² 0.24/0.34, MaxErr 0.45–0.82 mS); the load-bearing
> **caps track OSDI within ~10 %** (cgg/cdg ≈ 1 %, PMOS caps ≤9.75 %). Read
> alone, this would select **Phase 2 = distill the current-Jacobian (gds)** and
> exonerate caps.

> **P0-C (causal test) OVERRIDES this.** A diagnostic overlay shows *which
> surface is wrong*, not *which surface the period depends on*. P0-C swapped the
> exact analytic OSDI gds into the live RO (with AND without the `|id|·0.5`
> floor) and the **period did not move** (50.82 → 50.83 ps, 8.97 → 8.98 %). The
> cap-swap was **bit-for-bit identical** to baseline. So the divergent gds
> surface P0-B found is **not causal for the period** — fixing it (the Phase-2
> gds-distillation lever) would *not* close the 4.2 ps.

**Reconciled verdict:** the gds error is real but **inert for the RO period**;
caps are exonerated (P0-B small + P0-C bit-identical). Per the plan §4 tree this
is the **"both within ~10 % ⇒ RO lever DEAD"** branch in its operational sense
(no Jacobian surface the solver consumes moves the period). The 4.2 ps lives in
the **`id` value itself** (which both swaps leave untouched — VTC is 1.2–2.4 %
but the dynamic trip-trajectory `id` carries a residual integrated over the
cycle) or in **`qs=-(qg+qd+qb)` charge reconstruction / BDF-2 truncation** — a
new scope, not Jacobian distillation. **The Phase-2 RO Jacobian-distillation
lever is KILLED before any GPU.** See `phase0_C_cap_swap_ablation.md` for the
causal numbers and the final DECISION.
