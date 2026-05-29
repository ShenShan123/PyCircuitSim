# B8 — Differentiable Mini-Simulator for Test-Time Fine-Tuning (TSMC7 RO)

Track-B Tier-3 lever for the TSMC7 ring-oscillator gate (V6.4.5).

**Thesis.** The RO period is an *integrated* metric that pointwise Id-MAE
training cannot see. Seven prior experiments (B1 cap-symmetry, B3 LoRA,
B5 Jacobian-distill best 8.57 %, B6 harvest, B7 skeleton 11.9 %, plus a
32-recipe retrain best 9.05 %) failed to move the TSMC7 RO error off
**8.98 %** (DN 50.83 ps vs OSDI 46.64 ps; gate ≤ 5 %). B8 optimises the
period **directly** through a differentiable simulation of the actual
oscillator.

Baseline (production scorer `scripts/eval_v6_4_5_candidate.py`, canonical
`tsmc7_dn_medium_*`):

| metric | value |
|---|---|
| ring_osc_period_err | **8.977 %** (DN 50.829 ps / OSDI 46.641 ps) |
| inv_vtc_nrmse | 3.892 % |
| inv_vtc_maxerr_mv | 188.0 mV |
| inv_tran_post_nrmse | 1.099 % |

## Promotion / kill criteria
- **PROMOTE** iff production-scored TSMC7 RO period err ≤ **5 %** AND inverter
  waveform NRMSE ≤ **2 %** on the same fine-tuned weights.
- **KILL** if 200 TTFT steps do not move the (torch / production) RO from
  8.98 % to ≤ **7 %** — then the gate is in cap-shape, not the Id surface.

## Differentiable-sim design (`B8_diff_simulator/ro_torch.py`)
Self-contained; the production scipy solver is never touched.

- **Device** (`TorchDirectNetDevice`): DirectNet forward in torch giving
  `id` + the four terminal charges `(qg, qd, qs, qb)`, differentiable w.r.t.
  the terminal node voltages AND the model weights. Replicates
  `_MOSFETNNBase` bit-for-bit: PMOS source-relative frame, softplus voltage
  clamp + z-score input prep, **asinh** output denorm (TSMC7 norm mode),
  charge conservation `qs = -(qg+qd+qb)`, and the Rule-15 Vds correction on
  `id` (parts a rail-restoring extrapolation, b one-sided turn-on, d
  wrong-sign clamp — part c only shapes `gds`/the Jacobian and does not move
  the converged node voltages, so it is omitted). Verified: torch `id` and
  charges match the production `_MOSFETNNBase._eval` to ≥ 4 sig-figs at four
  operating points for both devices.
- **Topology** (`RingOscTorch`): 5 stages, nodes n1..n5, vdd = 0.75 V held,
  0.5 fF lumped load per node, ring feedback n5 → n1, `.ic = [0, VDD, 0,
  VDD, 0]`. Node KCL: stage k drives node k with input node k-1. Per-node
  current = (current leaving both drains) following the production
  `_stamp_mosfet_dc` "leaving drain" frame (NMOS `-id`, PMOS `-id`). Per-node
  stored charge = drain charge of the local pair + gate charge of the next
  stage (whose gate is this node) + `Cload·V` — exactly the terminals the
  production `_stamp_mosfet_transient` couples to a ring node.
- **Integration**: charge-based companion, **Backward-Euler on step 1,
  Trapezoidal on step 2+** (production schedule). Trap history `i_prev` =
  the previous step's capacitive current. Each output timestep
  (TSTEP = 2 ps, TSTOP = 1.2 ns) solved by a residual-accepted,
  best-iterate, adaptively-damped Newton in torch (dense 5×5, autograd
  Jacobian). A final implicit-function Newton refinement keeps the autograd
  graph so the period gradient flows to the weights.
- **Period**: `soft_period` = differentiable rising-edge midpoint-crossing
  period, mirroring the harness `_period_from_wave`.

## SANITY GATE — make-or-break (PASS)
With the **base canonical TSMC7 weights**, the torch RO must reproduce the
production RO period (≈ 50.8 ps).

| quantity | value |
|---|---|
| torch RO period | **50.828 ps** |
| production DN period | 50.829 ps |
| torch vs production gap | **0.00 %** |
| torch vs OSDI period err | **8.98 %** (reproduces the production gate) |
| post-settle swing | [-0.030, 0.782] V (production: [-0.030, 0.782] V) |

The differentiable simulator is a faithful surrogate of the production
transient — KCL residual at machine precision (1e-14…1e-19) every step. The
make-or-break sanity gate is **cleared**: optimising the torch period is a
valid proxy for the production period.

*(A subtle but key fix: the per-node "current leaving drain" sign for the
PMOS branch must be `-id_phys` — matching `_stamp_mosfet_dc`'s
`i_leaving = -calculate_current` for PMOS. With the wrong sign the ring
charged the wrong way and the Newton stalled / diverged.)*

## TTFT — design + torch-sim trajectory

**Trainable**: a **LoRA delta** (rank 8) on every trunk Linear of the NMOS and
PMOS models (49,984 trainable params). A delta is cheaper than full-weight
tuning and — being zero at init — preserves the inverter while it is small.
The merged (base + delta) plain DirectNet state_dict is saved under
`b8_ttft_tsmc7_{nmos,pmos}_best.pt` (+ base `_norm.npz` copy); canonical
`tsmc7_dn_medium_*` is never touched. Datasets are read-only symlinks.

**Gradient cost**: the unrolled-Newton-with-2nd-order-graph estimator was
infeasible (~22 min / fwd+bwd). Replaced with a **first-order
implicit-function estimate** (final timestep refinement `v* = v_conv -
J_det^{-1}·F`, detached Jacobian — the IFT fixed-point gradient) plus a
**batched per-stage device forward** (one stacked DirectNet call per
device-type over the 5 ring stages). Both are accuracy-neutral (sanity gate
still 50.828 ps, 0.00 % gap).

**Loss lesson**: with `alpha=2.0·waveform_MSE` the MSE term (≈0.32 V², phase-
dominated) swamped the period term (≈0.008) and dragged the period the WRONG
way (50.82 → 50.96 ps over 3 steps). Switched to **pure period loss**
(`alpha=0`): `(period_ps − 46.64)² / 46.64²`, lr=1e-3, Adam, Newton=8,
tstop=0.45 ns / settle=0.20 ns.

**Torch-sim trajectory (alpha=0, lr=1e-3)** — the differentiable period drops
straight to the OSDI target:

| step | torch period (ps) | torch RO err |
|---|---|---|
| 0 | 50.815 | 8.95 % |
| 2 | 50.020 | 7.24 % |
| 3 | 48.631 | 4.27 % |
| **4** | **46.490** | **0.32 %** |
| 5 | 44.389 | 4.83 % (overshoot — lr too high near min) |

**The differentiable sim drives its own period to the OSDI target in 4
steps.** The KILL criterion (≤7 % torch by step 200) is cleared by step 2.
Best saved checkpoint = step 4 (torch 0.32 %).

## Production-scorer quartet — DECISIVE (step-4 checkpoint)

`scripts/eval_v6_4_5_candidate.py --tech TSMC7 --nmos b8_ttft_tsmc7_nmos
--pmos b8_ttft_tsmc7_pmos` (CPU, 1-thread — the real test):

| metric | baseline (canonical) | **B8 TTFT** | B8 gate |
|---|---|---|---|
| **ring_osc_period_err** | 8.977 % | **0.317 %** | ≤ 5 % ✓ |
| ro_dn_period_ps | 50.829 | **46.493** (OSDI 46.641) | |
| **inv_tran_post_nrmse** | 1.099 % | **1.347 %** | ≤ 2 % ✓ |
| inv_vtc_nrmse | 3.892 % | **1.965 %** (improved) | |
| inv_vtc_maxerr_mv | 188.0 | **76.1** (improved) | |
| inv_vtc_dvtrip_mv | −0.16 | −1.43 | |
| sram_rail_snap_resid | 0.302 | 0.899 (degraded — not a B8 gate) | |
| opamp_flat_flag | 0 | 1 (flat — not a B8 gate) | |

**The torch sim and the production sim AGREE**: torch period 46.490 ps
(0.32 %) → production-scored 46.493 ps (0.317 %). Directly optimising the
period through the differentiable sim moved the PRODUCTION-scored RO from
**8.98 % → 0.32 %**, and the inverter did NOT break — the transient NRMSE
stayed at 1.35 % (≤ 2 %) and the DC VTC actually *improved* (3.89 → 1.97 %,
188 → 76 mV). This is the first lever in eight (B1, B3, B5, B6, B7, the
32-recipe retrain) to clear the TSMC7 RO gate.

Side effects (not B8 promotion gates): the 6T-SRAM force_ic rail-snap
residual degraded (0.302 → 0.899) and the Miller-opamp center bias went flat.
The LoRA delta was optimised solely on the RO period; it shifted the Id
surface enough to disturb the SRAM butterfly attractor and the opamp DC bias.
A multi-circuit-regularised TTFT (add SRAM/opamp terms to the loss) is the
obvious follow-up if those gates must be co-held; out of B8's scope.

## VERDICT — **PROMOTE**

PROMOTE criteria (both required):
- TSMC7 production-scored RO period err ≤ 5 %  →  **0.317 %  ✓**
- inverter waveform (transient) NRMSE ≤ 2 % on the SAME weights  →
  **1.347 %  ✓** (DC VTC also improved 3.89 → 1.97 %)

Both met. **B8 PROMOTES.** Candidate stems on disk (NON-canonical, canonical
`tsmc7_dn_medium_*` untouched):
`external_compact_models/bsimar/checkpoints/b8_ttft_tsmc7_{nmos,pmos}_best.pt`
(+ `_norm.npz`).

**Answer to the decisive question**: YES — directly optimising the period
through a differentiable simulation moves the PRODUCTION-scored RO below 5 %
(to 0.32 %) without breaking the inverter. The B8 thesis is confirmed: the RO
period is an integrated metric that pointwise Id-MAE cannot see, but a
differentiable transient exposes it to gradient descent. The torch surrogate
reproduces the production period to 0.00 % on base weights and tracks it
through fine-tuning (torch 0.32 % ↔ production 0.317 %).

