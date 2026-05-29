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

## TTFT — (in progress, updated below)

## Production-scorer quartet — (filled after TTFT)

## VERDICT — (filled at the end)
