# Phase 0 — Diagnostic P0-H: RO-trip VALUE overlay (TSMC7)

**Date:** 2026-06-02  •  **Branch:** `feat/v6.4.6`  •  **Status:** Done
**Env:** `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, `pycircuitsim` conda env.
**Plan:** `docs/plans/2026-06-01-directnet-v6.4.6-ro-sram.md` §2 (P0-C correction callout), §4 (P0-B/P0-C), §11 risk-1, §12 Q2.
**Script:** `scripts/v6_4_6_p0h_ro_value_overlay.py`  •  **Log:** `results/v6_4_6/phase0_logs/p0h_ro_value_overlay.log`

INSTRUMENTATION-ONLY — no shipped-behaviour change, no retrain, no checkpoint
mutation. Read-only on the frozen V6.4.4 `tsmc7_nmos`/`tsmc7_pmos` checkpoints.
Ground truth is the OSDI binary via PyCMG (`eval_single_point`, CLAUDE.md
Validation rule).

## Why this diagnostic is NEW (not a re-tread of P0-B/P0-C)

P0-B overlaid the model's **Jacobian** (`gm/gds`; cap-derivatives
`cgg/cgd/cdg/cdd`) and P0-C **causally** proved those surfaces are **inert** on
the RO period — swapping the exact analytic OSDI gds/caps into the live TSMC7 RO
moved the period ≤0.01 ps. The code reason (verified): `gds` and the
cap-derivatives are **Jacobian-only** — they enter the NR matrix and a matching
RHS offset and **cancel exactly at the converged NR fixed point**.

What P0-C left open — and **nobody had measured** — is the other half of the
companion model: the **VALUES** the transient stamps actually inject, which do
**NOT** cancel at the fixed point:

- resistive companion current from the **`id` VALUE**:
  `_stamp_mosfet_dc:304`, `i_eq = i_leaving − g_ds·v_ds − …`
- capacitive companion current from the **charge VALUES**:
  `_stamp_mosfet_transient:1772`, `i_g_cap = coeff·charges["qg"] − h_g`
  (and `qd` analogously at `:1773`).

If these VALUES diverge from OSDI along the trip trajectory, they directly shift
the period. This is exactly the bucket P0-C said the 4.18 ps must live in
(id-VALUE + charge-VALUE + BE/Trap/BDF-2 truncation). P0-H closes the
id/qg/qd/qs VALUE half of that bucket.

## What was measured

Identical infrastructure / bias points to P0-B (reused `run_directnet_ro` →
re-parse the gate's own rendered netlist → re-run with all-node waveforms →
select **one full post-settle oscillation cycle**: 26 timesteps, 346.00 → 396.00
ps, span 50.00 ps; DN period reproduced **50.82 ps** = baseline; NG 46.64 ps,
8.97 %). At each of the 26 bias points, for the stage-5 inverter driving n5
(NMOS `Mn5` d=n5 g=n4 s=0 b=0; PMOS `Mp5` d=n5 g=n4 s=vdd b=vdd):

- **NN post-Rule-15 VALUES** `id`, `qg`, `qd` from `mosfet._eval({…})` — the raw
  forward-pass current + charge columns the transient companion stamps (NOT
  their autograd derivatives), and `qs = −(qg+qd+qb)` (Rule-14 reconstruction).
- **analytic OSDI VALUES** `id` (col 0), `qg` (col 4), `qd` (col 5), `qb` (col 7)
  via `eval_single_point`, and `qs = −(qg+qd+qb)` the same way.

## Sign / convention (VALUES — no off-diagonal negation)

Unlike P0-B's cap *derivatives* (where `cgd`/`cdg` needed `NN = −OSDI`), the
VALUES share OSDI's convention **by construction**: the dataset stores
`outputs[k] = result[k]` straight from `eval_single_point`
(`nn_generate.py:657`), so the NN learns OSDI's `id`/`qg`/`qd`/`qb` sign
directly. `id`: NMOS<0 / PMOS>0 conducting → direct. Charges: direct. PMOS is
queried at the **absolute** terminal bias `(vn5, vn4, VDD, VDD)` (same as P0-B);
the NN does its own internal Vs=0 source-shift, identical at train and
inference, so the physical-bias charge state matches.

## Rule-16 overlay — NN (post-Rule-15) VALUE vs OSDI VALUE, 26-point cycle

### NMOS (stage-5, drives n5)

| Surface | MRE % | R² | NRMSE % | MaxErr | OSDI ptp |
|---------|------:|------:|--------:|-------:|---------:|
| id  | 44.57 | 0.9480 | 5.93 | 17.57 µA | 124.1 µA |
| qg  | 4.46 | 0.9994 | **0.96** | **1.99 aC** | 100.1 aC |
| qd  | 3.35 | 0.9996 | **0.75** | **0.78 aC** | 54.04 aC |
| qs  | 8.96 | 0.9992 | **1.19** | **1.08 aC** | 44.41 aC |

### PMOS (stage-5, drives n5)

| Surface | MRE % | R² | NRMSE % | MaxErr | OSDI ptp |
|---------|------:|------:|--------:|-------:|---------:|
| id  | 85.78 | 0.9895 | 2.99 | 7.26 µA | 97.57 µA |
| qg  | 3.25 | 0.9995 | **0.86** | **1.95 aC** | 107.8 aC |
| qd  | 2.99 | 0.9997 | **0.72** | **0.76 aC** | 57.93 aC |
| qs  | 6.52 | 0.9994 | **1.00** | **1.07 aC** | 48.58 aC |

> MRE is relative to `|OSDI|` with a per-surface scale floor (`1e-3·max|OSDI|`)
> so a single near-zero subthreshold true value does not blow up the mean; the
> high `id` MRE is the small-value subthreshold band, not the load-bearing
> on-state (same effect P0-B saw). **NRMSE / R²** (ptp- and variance-normalised)
> are the dynamic-range-honest read for a quantity integrated over a full swing.

### Per-cycle charge-swing context (per-stage Cload ≈ 0.5 fF)

For scale: 0.5 fF over a full 1 V rail swing carries ≈ **500 aC**. The intrinsic
gate/drain charge swing over one cycle is ≈ 100/54 aC (N) and 108/58 aC (P).

| device · surface | OSDI swing | NN swing | MaxErr | MaxErr / OSDI swing |
|------------------|-----------:|---------:|-------:|--------------------:|
| nmos qg | 100.07 aC | 97.99 aC | 1.99 aC | **2.0 %** |
| nmos qd | 54.04 aC | 53.07 aC | 0.78 aC | **1.4 %** |
| pmos qg | 107.82 aC | 105.35 aC | 1.95 aC | **1.8 %** |
| pmos qd | 57.93 aC | 56.75 aC | 0.76 aC | **1.3 %** |

The worst charge-VALUE error over the whole cycle is **≤ 2 aC**, i.e. **≤ 2.0 %
of the per-stage charge swing** and **≤ 0.4 % of the 500 aC the 0.5 fF stage
load moves per 1 V**. The capacitive companion current the BDF-2 integrator
stamps from these charges is essentially exact.

### `id` VALUE — on-state breakdown (the only non-negligible residual)

The full-cycle `id` NRMSE (5.93 %/2.99 %) sits within the ~10 % bar, but the
on-state (|OSDI id| > 10 % of peak, 8/26 pts) is the load-bearing slice:

| device | on-state NRMSE | on-state MaxErr | worst-pt NN vs OSDI |
|--------|---------------:|----------------:|---------------------|
| NMOS | **9.63 %** | 17.57 µA | −72.18 vs −89.75 µA (**−19.6 %** peak pull-down) |
| PMOS | 3.99 % | 6.33 µA | (worst abs pt is an off-state pt; on-state clean) |

The NMOS dynamic **peak pull-down current is under-predicted ~20 %**. Direction
check: less pull-down current → slower n5 discharge → **longer** period →
consistent with DN 50.82 ps > NG 46.64 ps. (DC VTC is 1.21–2.37 %; the residual
is specifically in the high-`|Vds|` saturated trip-trajectory drive current,
which the on-bin VTC fit does not stress.)

## Interpretation

- **Charge VALUES (qg/qd/qs) are essentially EXACT along the trip trajectory:**
  NRMSE 0.7–1.2 %, R² ≥ 0.999, MaxErr ≤ 2 aC = ≤ 2 % of the per-stage charge
  swing on **both** devices. The capacitive companion current the BDF-2
  integrator builds from `coeff·charges["qg"] − h_g` is correct. **A
  charge-value distillation has no error to remove → it would not move the
  period.**
- **`id` VALUE is borderline:** full-cycle NRMSE 5.93 %/2.99 % (within ~10 %),
  but the **NMOS on-state NRMSE is 9.63 %** with a ~20 % under-prediction of the
  dynamic peak pull-down current — directionally consistent with the longer DN
  period. This is the **only** VALUE surface carrying a non-negligible,
  period-relevant residual. It is small (≤ ~10 %) but not zero.
- The 4.18 ps gap is therefore **model-owned, in the NMOS dynamic `id` VALUE.**
  > **Reconciled with P0-G (ran concurrently; this bullet originally mis-attributed
  > the gap to integration truncation).** P0-G's uniform-timestep convergence
  > study is decisive: driving the truncation to zero (tstep→0, both Trapezoidal
  > and Backward-Euler) leaves a common **≈50.4 ps continuum limit — still ~3.7 ps
  > (~8 %) above NG 46.64**. So BE/Trap/BDF-2 integration truncation accounts for
  > only **~0.4 ps** (50.82 → 50.4) of the 4.18 ps gap; the remaining **~3.7 ps is
  > model-owned**. Since the charge VALUES are exact (this study) and the Jacobian
  > is inert (P0-C), the surviving model surface is the **`id` VALUE** — the NMOS
  > on-state residual (9.6 % NRMSE, ~20 % peak pull-down under-prediction,
  > direction-consistent with the longer DN period). The id-value residual looked
  > "≤10 % and bounded" pointwise, but integrated over the discharge it is the
  > dominant cause of the ~3.7 ps model residual — pending the P0-I causal swap.

## Explicit contrast with P0-B / P0-C

| Diagnostic | What it overlaid / swapped | RO-period verdict |
|------------|----------------------------|-------------------|
| **P0-B** (derivative overlay) | NN autograd `gm/gds`, cap-derivs vs OSDI | gds DIVERGES 20–23 % NRMSE; caps track. *Correlational only.* |
| **P0-C** (causal swap) | inject exact OSDI gds / caps into live RO | period moves **≤0.01 ps** → Jacobian surfaces **causally INERT** (cancel at fixed point). |
| **P0-H** (this — VALUE overlay) | NN post-Rule-15 `id/qg/qd/qs` VALUE vs OSDI | charges **EXACT** (≤1.2 % NRMSE); `id` ≤10 % (NMOS on-state 9.6 %, ~20 % peak-current dip). |

P0-H is the missing companion to P0-C: P0-C proved the *Jacobian* (which cancels)
is inert; P0-H measures the *VALUES* (which do not cancel) and finds the charge
VALUES already correct and the id VALUE within ~10 %.

## DECISION (plan §4 tree — the "all VALUES track OSDI within ~10 %" branch)

> **A charge-value LoRA distillation is NOT the V6.4.7 lever.** The qg/qd/qs
> VALUES the capacitive companion stamps are already within ≤1.2 % NRMSE /
> ≤2 aC (≤2 % of the per-stage swing) of analytic OSDI — there is no charge-value
> error to distill out, so bending the charge VALUES toward OSDI cannot move the
> 4.18 ps period. This is **DISTINCT from, and reaches the same "no model-side
> Jacobian lever" conclusion as, the P0-C-killed gds/cap *Jacobian*
> distillation** — but for a different reason: the Jacobian is inert because it
> *cancels at the fixed point*; the charge VALUES are inert because they are
> *already correct*.

> **The 4.18 ps gap is MODEL-owned, in the NMOS dynamic `id` VALUE.** P0-G's
> convergence study (concurrent) shows the tstep→0 continuum limit is ≈50.4 ps —
> BE/Trap/BDF-2 truncation owns only ~0.4 ps; the remaining ~3.7 ps is the model.
> The charge-VALUE half of the P0-C bucket is closed as a clean negative
> (charges exact); the **`id`-VALUE half is the owner** — pointwise ≤10 % but,
> integrated over the discharge, the dominant cause of the ~3.7 ps residual.

Consequences for V6.4.6 / V6.4.7:

- **Do NOT build a charge-value (qg/qd) LoRA distillation for the RO gate.**
  EV is zero — the charge VALUES are already OSDI-accurate. This kills the
  charge-VALUE "distill a VALUE the companion stamps" RO lever (the Jacobian one
  died in P0-C). V6.4.6 already shipped at **9/16** (Phase 1 found SRAM `force_ic`
  is 0/8 honest, not closable at 0 GPU — the probe-hardening was a measurement
  fix, not a gate-close); the TSMC7 RO gate moves to V6.4.7 as a **model-owned
  `id`-VALUE** problem (NOT solver-integration — P0-G drove truncation→0 and ~3.7
  ps of model residual remains; NOT Jacobian — P0-C; NOT charge-value — this study).
- **The one residual model-side lever is the NMOS dynamic-`id` VALUE** (on-state
  NRMSE 9.6 %, ~20 % peak-current under-prediction, directionally consistent
  with the gap). It is **not yet causally confirmed**: P0-H is the correlational
  VALUE overlay; the analogue of P0-C's swap — injecting the exact OSDI `id`
  VALUE into the live RO transient and re-measuring the period — has **not** been
  run and is the natural next 0-GPU step before funding any id-value
  distillation. If that swap closes a meaningful fraction of the 4.18 ps, an
  id-value LoRA (clipped/Huber-on-ln-current against the clean OSDI id, per
  P0-E) becomes the indicated V6.4.7 lever; if it moves the period ≤0.01 ps like
  the gds/cap swaps, the gap is **purely** BE/Trap/BDF-2 truncation and **no**
  model-side distillation is warranted.

**Net P0-H read (reconciled with P0-G):** charge VALUES exonerated as a clean
negative (no charge-value distillation); the **NMOS dynamic `id` VALUE is the
owner** of the ~3.7 ps model residual (P0-G's tstep→0 limit ≈50.4 ps leaves only
~0.4 ps for integration truncation), pending the **P0-I** causal id-VALUE swap to
confirm magnitude before any V6.4.7 id-value distillation.
