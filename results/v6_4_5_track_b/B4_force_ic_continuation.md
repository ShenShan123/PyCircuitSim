# B4 — force_ic continuation with shrinking-λ trust region (Track B, Tier 1)

**Date:** 2026-05-29 • **Branch:** `feat/v6.4.5-track-b` • **Verdict: KILL (code reverted)**
**Env:** `CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.
**Code (reverted):** `pycircuitsim/solver.py:_solve_force_ic_continuation` +
env hook `NN_FORCE_IC_CONTINUATION` in `_solve_newton`. Reverted to HEAD after
the kill; the experiment driver `experiments/v6_4_5_track_b/B4_force_ic_continuation.py`
is retained as the record.

## Mechanism

Replaced the default binary `force_ic` path (hard-pin each `.ic` node with an
ideal VS → solve → remove pins → ONE unconstrained re-solve) with a
continuation in pin stiffness λ. Each `.ic` node is pinned toward its target
by a Thévenin source (ideal VS at target behind `R = 1/λ`); λ shrinks over
`{1e6, 1e4, 1e2, 1e0, 1e-2, 0}`, each stage full-NR and warm-started by the
previous, ending fully unconstrained at λ = 0.

## Falsifier (plan B4)

> Promote: ≥ 1/4 SRAM force_ic cells snap to rails under the continuation.
> Hard kill: no λ schedule moves any cell off the interior attractor.

## Result — 8 cells (4 techs × 2 states), baseline vs continuation

| Tech | state | baseline q / qb | continuation q / qb | snap? |
|------|-------|-----------------|---------------------|:-----:|
| TSMC7 | state1 | 0.815 / 0.226 | 0.814 / 0.228 | no |
| TSMC7 | state0 | 0.226 / 0.815 | 0.228 / 0.814 | no |
| TSMC5 | state1 | 0.702 / 0.163 | 0.702 / 0.163 | no |
| TSMC5 | state0 | 0.163 / 0.702 | 0.163 / 0.702 | no |
| TSMC12 | state1 | 0.866 / 0.192 | 0.866 / 0.192 | no |
| TSMC12 | state0 | 0.192 / 0.866 | 0.192 / 0.866 | no |
| TSMC16 | state1 | 0.865 / 0.199 | 0.864 / 0.200 | no |
| TSMC16 | state0 | 0.199 / 0.865 | 0.200 / 0.864 | no |

`n_snap_cont = 0/8`, `n_snap_base = 0/8`. (VDD: TSMC7 0.75 V.)

## Verdict: **KILL**

The continuation reproduces the baseline attractor bit-for-(near)-bit on
every cell. Gentle λ-release does not keep the solution in the rail basin:
as λ → 0 the soft pin vanishes and the state relaxes to the **same** interior
fixed point (q ≈ 0.82, qb ≈ 0.23 for TSMC7). This proves the rail (q=VDD,
qb=0) is **not a stable DC equilibrium** of the unconstrained NN cross-coupled
pair — no homotopy path ending at λ = 0 can stay there.

Combined with **B2** (8 independent seeds, all land at the same attractor),
the SRAM `force_ic` gate is conclusively a **model-fidelity** property of the
DirectNet I-V surface (under-modelled NMOS on-drive at Vgs≈VDD/Vds≈0.2), not a
solver-path or seed artifact. → SRAM routes to Tier-2 B6/B7 or Tier-3 B9. The
solver change is reverted (default behaviour unchanged).
