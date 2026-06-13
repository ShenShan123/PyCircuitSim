# V6.4.7 S10 (P4) — control-v2 derivative-fidelity reference (the P4 baseline)

**Date:** 2026-06-14 · seed 42 per tech · v2 data (`--data-suffix v2
--no-apply-filter`) · CPU · `scripts/v6_4_7_deriv_fidelity.py`. This is the
baseline the P4 Sobolev id-derivative arm must strictly beat (rev-3 promotion
criterion). Use the **`fwd_inrail` corridor** keys for A/B — the full-split
`overall`/`deriv_*_nrmse` are leakage-plateau-dominated (hundreds–thousands %)
and not the relevant metric.

## fwd_inrail NRMSE (%) — autograd ∂id/∂V vs OSDI, per device

| tech | dev | gm_fwd | gds_fwd | gmb_fwd | id_value NRMSE | gds sign-agree (SI) |
|------|-----|--------|---------|---------|----------------|---------------------|
| tsmc5  | nmos | 92.2  | 13.3 | 14.3 | 0.018% | 0.923 |
| tsmc5  | pmos | 233.1 | 70.1 | 86.5 | 0.030% | 0.932 |
| tsmc7  | nmos | 44.5  | 6.9  | 9.4  | 0.024% | 0.892 |
| tsmc7  | pmos | 137.0 | 68.7 | 39.5 | 0.024% | 0.906 |
| tsmc12 | nmos | 10.8  | 6.3  | 4.3  | 0.014% | 0.926 |
| tsmc12 | pmos | 25.3  | 20.3 | 11.0 | 0.021% | 0.924 |
| tsmc16 | nmos | 9.2   | 10.8 | 3.8  | 0.050% | 0.928 |
| tsmc16 | pmos | 21.8  | 20.8 | 5.0  | 0.019% | 0.882 |

## Findings (drive P4 design)

- **Strongly device-asymmetric.** NMOS gds_fwd is decent (6.3–13.3 %, *better*
  than the P0-B 20–23 % anchor); **PMOS gds_fwd is the weak channel** (20–70 %,
  worst on tsmc5/tsmc7 at ~70 %). P4's derivative supervision should be
  weighted toward PMOS and the high-Vt/short-tech corners.
- **id VALUE is excellent everywhere** (≤0.05 % NRMSE) — the network nails the
  id surface; it's the *slope* (gds especially) that drifts, exactly the gap
  P4 (autograd-∂id/∂V ↔ OSDI gm/gds/gmb consistency) targets.
- gds **sign agreement** in strong inversion is high (0.88–0.93), so the
  Sobolev term starts from a sign-correct surface — refinement, not repair.
- Off-state id excess is tiny (≤1.4e-4 A max, mean ~1e-6 A) — the unfiltered
  small-current training did NOT inflate OFF leakage.

## P4 implementation basis

Recover `JacobianConsistencyLoss` + `JAC_CHANNELS` + the asinh chain-rule
transform from `git show 930c274` (saved `/tmp/930c274_jac_loss.patch`).
For P4 use **id-channels only**: `("id", in_idx=1, "gm")`, `("id", 0, "gds")`,
`("id", 3, "gmb")`. asinh chain rule: `d(out_phys)/d(in_phys) /
√(scale² + out_phys²)`; gm/gmb targets sign-flipped to match autograd ∂id.
Plan: id-channels only, M-scale, λ-swept, fine-tune from control-v2 first
(needs init-from-checkpoint kwarg), full retrain second; supervise gds in
relative/asinh space, importance-weight the opamp op-point corridor, reuse the
P3 trust-floor mask (don't supervise slopes on sub-floor noise). Promotion:
TSMC7 opamp gain + derivative-fidelity strictly < control-v2 (this table),
inverter held, RO blind-veto on all passing cells.
