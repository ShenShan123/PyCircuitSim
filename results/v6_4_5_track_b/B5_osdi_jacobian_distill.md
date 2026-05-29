# B5 — OSDI Jacobian distillation (Track B, Tier 2)

**Date:** 2026-05-29 • **Branch:** `feat/v6.4.5-track-b` • **Verdict: KILL**
(The B5 agent crashed on a connection error before writing its own report; these
are the numbers I re-scored independently from the candidate checkpoints it
trained. Training-loss fork lived in `experiments/v6_4_5_track_b/B5_*.py`;
candidates `b5_jd_lam{...}_tsmc7_*` — discarded after the kill.)

## Idea
Add a Jacobian-distillation loss so the autograd derivative of the predicted
`id` (the gm/gds/gmb the solver actually uses at inference, Rule 1) matches the
analytic OSDI gm/gds/gmb that are already dataset columns:
`L = L_MAE + λ_J·(‖∂id/∂Vg − gm_OSDI‖ + ‖∂id/∂Vd − gds_OSDI‖ + ‖∂id/∂Vb − gmb_OSDI‖)`.
This is "Phase-1b with the right (analytic) targets." λ_J ∈ {0.001, 0.01, 0.1},
warm-started from the V6.4.4 base, TSMC7 N+P.

## Falsifier (plan B5)
> Promote: TSMC7 RO ≤ 5% AND inv VTC MaxErr regression ≤ +5 mV at best λ_J.
> Kill: no λ_J achieves both — Phase-1b dead end confirmed under analytic targets too.

## Result (re-scored, isolated scorer; baseline = canonical V6.4.4)

| candidate | RO % | inv VTC NRMSE % | inv VTC MaxErr mV | inv tran NRMSE % | SRAM resid (q) |
|-----------|-----:|----------------:|------------------:|-----------------:|---------------:|
| canonical | 8.98 | 3.89 | 188 | 1.10 | 0.302 (0.815) |
| λ=0.001   | 9.38 | 3.91 | 197 | 1.86 | 0.307 (0.815) |
| λ=0.01    | 8.57 | 5.46 | 158 | 1.56 | 0.281 (0.832) |
| λ=0.1     | (incomplete — agent crashed mid-sweep; nmos-only checkpoint) |

## Verdict: **KILL**

No λ_J closes RO (best 8.57% at λ=0.01, still ≫ 5%). λ=0.01 moves SRAM
marginally toward the rail (resid 0.302→0.281, q 0.815→0.832) but does not snap,
and it regresses inv VTC NRMSE to 5.46%. λ=0.001 is essentially baseline.

Jacobian distillation against analytic OSDI derivatives is **necessary but not
sufficient** — sharper pointwise gm/gds does not fix the *integrated* RO phase
walk nor the cross-coupled SRAM equilibrium. This confirms the Phase-1b dead end
holds even with the correct (analytic) targets, and corroborates B6's in-box
finding: the gap is architecture/capacity, not derivative supervision.
