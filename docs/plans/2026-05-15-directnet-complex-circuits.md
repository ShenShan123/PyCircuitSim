# DirectNet (LEVEL=73) — Scaling to Complex Circuits, DC + Transient Only

**Date:** 2026-05-15  •  **Status:** Proposal, revised (phase order re-prioritized)  •  **Baseline:** V6.3.1 (DirectNet per-tech, TSMC5/7/12/16, shipping)

> **Revision note (2026-05-15).** This plan was first drafted against the V6.2.1
> baseline. It is re-issued against **V6.3.1** with a new phase order that
> prioritizes *simple, effective, accuracy-first* levers. The driving change:
> the V6.3.1 sprint already shipped the `reverse_vds` overlay and a re-centered
> `inv_trip` overlay, and **diagnosed the remaining inverter error as gain
> amplification at the trip (a gm/gds-fidelity problem), not a data-coverage
> gap.** Since opamp open-loop gain is the *same* failure mode amplified
> further, gm/gds fidelity is now the critical path for *every* complex circuit
> — so the training-side fidelity lever is promoted to Phase 1, ahead of the
> solver/perf work. Retrain is cheap (~2-3 h for 8 medium cells) and is treated
> as a first-class tool, not a barrier.

## Scope

This plan addresses what **DirectNet — the shipping LEVEL=73 single-shot MLP compact model** — needs in order to drive larger circuits than a single inverter (opamps, ring oscillators, SRAM latches, switched-cap blocks) under **`.op`, `.dc`, and `.tran` only**. AC, S-parameter, harmonic-balance, and RF-mixed-signal paths are explicitly out of scope.

**LEVEL=74 BSIMAR is out of scope** at this phase. Rule 18 stands. Any line in this plan that touches `pycircuitsim/models/mosfet_nn.py` or the `bsimar` package's data / loss / training layers must be benchmarked against DirectNet only.

The proposal is the synthesis of a four-agent investigation (architecture, solver/NR, training+data, external prior art) run 2026-05-15, refiltered to DirectNet's single-shot MLP architecture and re-prioritized against the V6.3.1 outcome. File:line references were re-confirmed against live source on 2026-05-15.

## Pre-conditions / load-bearing rules

These cannot be retrained away. Any phase below must preserve them:

- **Rule 1** — autograd-derived `gm/gds/gmb` only at inference; never consume the predicted conductance heads. DirectNet trains them as direct supervision (`external_compact_models/bsimar/models/direct_net.py:26`) but only the autograd Jacobian feeds NR (`pycircuitsim/models/mosfet_directnet.py`, `pycircuitsim/models/mosfet_nn.py:288-319`).
- **Rule 5** — `gds = max(gds, |id|·0.5, 1e-12)`; never `abs(gds)`. Preserve `gm/gmb` signs.
- **Rule 10 — AMENDED by this plan (see Phase 1b).** Until Phase 1, `MAELoss` with per-target LDS weights is the sole loss. **Phase 1b adds one bounded derivative-consistency (Sobolev) term**; the amendment is scoped, validated against the V6.3.1 baseline before adoption, and recorded in CLAUDE.md. `DirectLoss / ChargeConsistencyLoss / SignConsistencyLoss / BoundaryLoss` and the Vov-LDS / subthreshold-LDS axes stay deleted.
- **Rule 14** — `qs = -(qg+qd+qb)` is enforced analytically by the simulator at every transient timestep regardless of DirectNet's `qs` head.
- **Rule 15** — four-part Vds correction in `_MOSFETNNBase._apply_vds_correction()` (`mosfet_nn.py:370-465`), including the V6.2 sign-fix rail-restoring extrapolation (a), exponential clamp (b/c) and sign enforcement (d). DirectNet relies on this for rail behaviour; do not retrain around it.
- **Rule 17** — ASAP7 stays out of scope.
- **Rule 18** — BSIMAR remains parked. This plan is DirectNet-only.
- **Rule 19** — `unknown_code_id` derived from `num_tech_codes`. The DirectNet constructor at `direct_net.py:38` defaults `unknown_code_id=17` — correct only for the universal vocab. Per-tech trainings override it via `train_directnet`; do not regress.
- **Rule 20** — Rule 15(a)'s sign convention is load-bearing. Re-validate on every per-tech checkpoint after any retrain.

## Diagnosis: what blocks DirectNet on bigger circuits

DirectNet today produces a 13-dim output `[id, gm, gds, gmb, qg, qd, qs, qb, cgg, cgd, cgs, cdg, cdd]` from a single 6-layer SiLU MLP with a `tech_embedding` concatenated to the 7-D input (`direct_net.py:29-83`); the output is one `nn.Linear(hidden_dim, 13)` at `direct_net.py:55`. It ships per-tech for TSMC5/7/12/16. Inverter metrics at V6.3.1: VTC MaxErr 66.4 / 65.8 / 78.3 / 45.4 mV, transient post-startup MaxErr 39.5 / 50.3 / 58.2 / 55.3 mV (TSMC5/7/12/16). The blockers below are what stops it from generalizing.

### D0 — gm/gds fidelity at the trip caps inverter *and* opamp accuracy *(V6.3.1 outcome — the dominant blocker)*

The V6.3.1 sprint ran three dataset revisions of the `inv_trip` overlay; trip error moved around but never dropped below ~45 mV. The CHANGELOG conclusion: this is **not a coverage gap**. Inverter gain ≈ −15 to −30 at the trip multiplies DirectNet's residual `Id` error (~0.05 % test-split NRMSE) ~20× into `Vout`. The fix needs a **gm/gds-fidelity lever**, not more `inv_trip` samples. This is the *same* mechanism that sets opamp open-loop gain (gain there is −40 to −60 dB-equivalent) — so D0 is the critical path for the entire plan, not just the deferred inverter gate. Addressed by **Phase 1**.

### D1 — Per-iter NN cost scales linearly with device count *(solver agent)*

`_stamp_mosfet_dc` / `_stamp_mosfet_transient` (`solver.py:115`, `:1320`) call `_eval` once per NN MOSFET per NR iter (`mosfet_nn.py:251-328`). DirectNet's forward is cheap, but a 30-device opamp at 200 timesteps × 15 NR iters is **90 000 forward+autograd calls** with no batching — ~4-5 s per `.tran` sub-step from PyTorch eager overhead alone. Batching collapses the workload into one stacked forward per NR iter. Addressed by **Phase 5**.

### D2 — Asymmetric C-stamps under BDF-2 *(solver agent)*

`cgd = ∂qg/∂Vd` vs `cdg = ∂qd/∂Vg` are independent MLP outputs (`solver.py:1370-1377`). The LDS loss bounds each column but does not couple them; trained asymmetry can drift ~5-15 %. Under BDF-2 (`solver.py:1344`) asymmetry seeds artificial damping/growth in oscillating transients — silent on the inverter, loud on a ring oscillator. Addressed by **Phase 2**.

### D3 — *(Largely stale at V6.3.1.)* Wrong-sign branch and gm/gmb

The original plan flagged the reverse-Vds `id` clamp as leaving `gm/gmb` un-fixed. Live source already handles this: `mosfet_nn.py:434-435` scales `gm/gmb` by `f_id` (= 0 on the reverse branch) and `:445-446` zeroes `gm/gmb` again in the wrong-sign clamp. **What remains:** no *saturation floor* exists for `gm/gmb` on the *conducting* branch (Rule 5 floors `gds` only). Reduced scope folded into **Phase 2b**.

### D4 — Trust-region cap is too coarse for oscillators *(solver agent)*

`solver.py:556`, `:1161` clamp `|ΔV| ≤ VDD` per iter. Fine for an inverter VTC; for ring-osc nodes NR can walk a full VDD per step → DirectNet hallucinates outside training, and the averaged-solution acceptance at `solver.py:683`, `:1292` *succeeds* at a non-physical fixed point. Latches hit the same path through the saddle where `det(J)=0`. Addressed by **Phase 6**.

### D5 — *(Partially shipped at V6.3.)* Operating-region overlays miss new circuit classes *(training agent)*

V6.3 shipped the `reverse_vds` corridor (sample-class 10, `nn_generate.py:65`, `_reverse_vds_points` at `:491`) and re-centered `_inv_trip_points` (`:445`) on VDD/2. **Still missing:** saturation curvature (diff pairs), Miller region (ring osc), bistable anchors (SRAM), off-state (SC hold). The `id > 1e-15` filter (`dataset.py:39`) still discards off-state rows. Remaining scope addressed by **Phase 4**.

### D6 — Uniform 80/10/10 validation hides regressions *(training agent)*

`load_and_split_bsimar` (`dataset.py:55`) does a uniform random split. NRMSE on this split is uncorrelated with opamp gain or RO period. The V6.1 → V6.2 inverter regression (TSMC7 transient 13.49 % while all val metrics were green) is the existence proof — and that bug shipped *in DirectNet*. Addressed by **Phase 1a** (validation harness first).

### D7 — Cross-target gradient imbalance from asinh normalization *(training agent)*

After asinh+zscore each column is O(1), but the 13 targets span 4 orders of magnitude physically. A 1 % error on `qg` carries the same gradient as 1 % on `id`. For circuit fidelity, `id` matters most — its autograd slope *is* the inference-time `gm/gds`. `train_directnet` already accepts `column_weights` (`trainer.py:151-174`), default unpinned. Addressed by **Phase 1c**.

### D8 — Sampling grid undersamples curved regions *(training agent)*

`_sample_hybrid_grid_voltages` (`nn_generate.py:281`) uses `np.linspace` on Vgs/Vds. The `Vgs ≈ Vth` subthreshold knee and the `Vds ≈ V_dsat` saturation knee carry most of the physics and are undersampled ~10× vs flat saturation. The 5-point `Vbs` ladder cannot teach body-bias sensitivity for opamp tail nodes or SRAM read-disturb. Addressed by **Phase 4**.

## Re-prioritized phase plan

Ordering principle: **accuracy-first, simple-first.** Phase 1 attacks the documented dominant blocker (D0) with the cheapest effective lever and is gated on a retrain that is hours, not weeks. Phases 2-3 are no-retrain enablers (solver correctness + the benchmark harness that makes everything downstream measurable). Phase 4 is the larger data lever. Phases 5-6 are solver perf/convergence — necessary to *run* big circuits but accuracy-neutral, so they follow the accuracy work. Phases 7-8 are structural model changes, highest effort and lowest certainty, gated on everything prior.

Every phase is validated against the existing TSMC5/7/12/16 inverter gate (`tests/verify_nn_dc_tran.py --tech TSMC5,TSMC7,TSMC12,TSMC16 --inverter-only`, metrics via `scripts/eval_v6_3_1_inverter.py`) before measuring on new circuits.

### Phase 1 — gm/gds-fidelity training lift *(retrain — the V6.3.1 open-gate closer)*

**Goal:** raise autograd-`gm/gds` fidelity at the inverter trip and in saturation — the documented V6.3.1 cap (D0) and the critical-path quantity for opamp gain. All sub-steps are ≤1-day code changes; one 8-cell medium retrain (~2-3 h, `scripts/train_v6_3_1_parallel.sh`) validates the stack.

- **1a — Validation harness FIRST.** Before any retrain, add DC/Tran-aware slices to `_per_tech_report` (`trainer.py:111`) and drive early-stop on a weighted slice sum, not raw uniform-split val loss (`trainer.py:226`) — this is the D6 fix. Three deterministic slices:
  - **Gain slice** — `Vgs ∈ [Vth+0.05, Vth+0.20]`, `Vds ∈ [0.5·VDD, VDD]` → median `|gm/gds|` MRE (opamp open-loop gain proxy).
  - **Switching slice** — `Vgs = V_M ± 30 mV` at three `Vds` levels → median `id` and autograd-`gm` MRE (predicts inverter trip error and RO period).
  - **Off-state slice** — `Vgs < 0.3·Vth`, full `Vds` range → median `|id|` (predicts SRAM hold leakage / SC droop).
- **1b — Bounded Sobolev derivative-consistency term (Rule 10 amendment).** Add one term to `MAELoss`:
  `L += λ · w_trip · ( |∂id/∂Vgs − gm_label| + |∂id/∂Vds − gds_label| )`
  where the derivatives are `torch.autograd.grad` of the *predicted `id`* w.r.t. the voltage inputs (the exact quantity NR consumes — Rule 1), `gm_label/gds_label` are the existing training targets, and `w_trip` up-weights rows with `sample_class ∈ {inv_trip, reverse_vds}` (the `sample_class` column already in every `.npz`, `nn_generate.py:52-71`). This directly trains the inference-time Jacobian rather than the dead predicted heads.
  - **This amends Rule 10.** It is *not* a blanket revival of the deleted `SlopeMatchLoss` (B2): that lever was deleted on 2026-05-03 for *process* reasons — never validated against a v4 baseline, corrupted by B3's `id_idx` bug — not because it was proven harmful. Phase 1b is scoped (one term, autograd-of-`id` only, trip-weighted), guarded behind a `--sobolev-weight` CLI flag (default 0), and **must win a bake-off vs the V6.3.1 retrain recipe before becoming default**. If it loses, it is reverted and recorded as a dead end; Rule 10 is restored verbatim.
  - On adoption, update CLAUDE.md Rule 10 to read "`MAELoss` with per-target LDS weights **plus one bounded trip-weighted Sobolev term**" and add a dead-end note distinguishing it from `SlopeMatchLoss`.
- **1c — Pin `column_weights` to emphasize `id`.** `train_directnet` already accepts the argument (`trainer.py:151-174`). Pin `id` (column 0) high; `gm/gds` head columns get a *modest* lift only — they do **not** feed NR (Rule 1), so the original plan's symmetric `[2,2,2,…]` pin was misdirected. Default pin: `id`=3, `gm`/`gds` head=1.5, rest=1.
- **1d — Quantile LDS bins.** `strategy="quantile"` in `compute_lds_weights_per_target` (`bni_mae.py:29-50`). Equal sample mass per bin up-weights rare deep-saturation / off-state rows. One-line change.
- **1e — `gds` asinh-scale floor.** Add `gds` to `_OUTPUT_ASINH_SCALE_MIN` (`normalize.py:59`, currently `{"gmb": 1e-5, "qb": 1e-15}`) so deep-saturation rows don't collapse the asinh scale and erase the gain-defining numerator.
- **Success gate:** TSMC5/7/12/16 inverter VTC MaxErr ≤ V6.3.1 (66.4 / 65.8 / 78.3 / 45.4 mV) **and** transient post-startup ≤ V6.3.1 (39.5 / 50.3 / 58.2 / 55.3 mV) — i.e. no regression — with the three phase-1a slice metrics reported and target bands recorded. Stretch: VTC trends toward the deferred ≤25 mV gate. If 1b loses its bake-off, ship 1c-1e only.

### Phase 2 — Solver correctness fixes *(no retrain — independent of Phase 1, can run in parallel)*

Touches `solver.py` / `mosfet_nn.py` only — disjoint from Phase 1's `bsimar` package edits, so the two phases parallelize.

- **2a — Symmetrize C-stamps.** In `_stamp_mosfet_transient` (`solver.py:1370-1415`) replace `cgd, cdg` with `c_sym = ½(cgd+cdg)` stamped both ways; same for `cgs/csg`, `cds/csd`. Lifted from Berkeley BSIM-NN's symmetrized-capacitance convention. Gate behind `NN_SYMMETRIC_CAPS=1`, default-off until phase-3 benchmarks accept.
- **2b — Conducting-branch `gm/gmb` saturation floor.** D3's reverse/wrong-sign handling is *already* in live source (`mosfet_nn.py:434-435`, `:445-446`). The only gap: no saturation floor on the *conducting* branch. Mirror Rule 5 — enforce `gm ≥ 0` (NMOS) / sign-correct `gm`, and a small `|gm|` floor, on the conducting branch only; leave the reverse branch's zeros untouched.
- **Success gate:** TSMC5/7/12/16 inverter transient NRMSE within ±5 % of V6.3.1 (1.22-1.51 %); with `NN_SYMMETRIC_CAPS=1` the phase-3 ring oscillator reaches steady oscillation without dt-halve runaway.

### Phase 3 — Benchmark harness *(no retrain, no model change — promoted from last to third)*

The original plan buried these at Phase 8, yet every complex-circuit claim in Phases 4-8 is unmeasurable without them. Building the netlists + NGSPICE references is pure test infra and has no dependency on the model — so it lands early.

- **3a — 5-stage CMOS ring oscillator** (`.tran`).
- **3b — Two-stage Miller opamp** (`.op` + `.dc` transfer).
- **3c — 6T SRAM read SNM** (`.dc` butterfly, `force_ic`).
- **3d — Switched-cap unit cell** (`.tran` with PULSE clock).
- Each follows the `tests/verify_*.py` pattern with an NGSPICE reference netlist under `tests/references/`, reusing `TechProfile` / `ALL_TECHS` from `tests/common/`. **Never use simplified/self-defined equations as reference — NGSPICE only.**
- **Success gate:** all four reference netlists run clean in NGSPICE and the harness reports DirectNet-vs-NGSPICE metrics (period, gain, SNM, charge-transfer) even where DirectNet currently fails — the harness must *measure* the gap before any phase tries to close it.

### Phase 4 — Data overlays + non-uniform sampling *(retrain — the larger data lever)*

Runs after Phase 1 establishes the validation harness and the column_weights/Sobolev baseline, and after Phase 3 gives circuits to measure against.

- **4a — Extend `SAMPLE_CLASS_NAMES`** (`nn_generate.py:52`) with `diff_pair_sat`, `ring_osc_trip`, `bistable_static`, `switched_cap_offstate` (`reverse_vds` already shipped V6.3 — do **not** re-add). Each gets a generator alongside `_inv_trip_points` (`:445`) / `_reverse_vds_points` (`:491`), promoted through `BinSpec`, `generate_one_bin` (`:790`), `enumerate_bins`. CLI: replace `--enable-inv-trip` with `--overlays inv_trip,reverse_vds,diff_pair_sat,…` (comma list, per-class budget caps).
  - `diff_pair_sat` — `Vgs ∈ [Vth, Vth+3·Vov]`, `Vds ∈ [0.3, 1.0]·VDD`, `Vbs ∈ {0, ±0.25·VDD}`.
  - `ring_osc_trip` — V_M band extended to `Vds ∈ [0.1, 1.5]·VDD`, both polarities (Miller region).
  - `bistable_static` — anchors `(Vgs, Vds) ∈ {(VDD, V_low), (0, V_high)}` plus a fine ΔVds grid near the latch trip.
  - `switched_cap_offstate` — `Vgs < Vth`, full `Vds` range.
- **4b — Sinh-spaced Vgs/Vds.** In `_sample_hybrid_grid_voltages` (`nn_generate.py:281`) replace `np.linspace` with per-bin sinh grids centred on `Vth` (Vgs) and `V_ov` (Vds). Same row budget, ~3× density on curved regions.
- **4c — Vbs LHS draw.** Resurrect `_sample_vbs_lhs_voltages` (`nn_generate.py:562`) gated on `--overlays`; replace the 5-point ladder for body-biased opamp / SRAM read-disturb data.
- **4d — NFIN interpolation grid.** Add NFIN ∈ {4, 6, 8, 12} alongside `{2, 3, 5, 10, 15, 20, 24}` so current-mirror NFIN ratios are interpolation, not extrapolation.
- **4e — Off-state gate.** Make `apply_filter` (`dataset.py:63`, threshold `dataset.py:39`) opt-out via `--keep-offstate` so SC / SRAM hold-leakage rows survive ingestion.
- **Success gate:** all Phase-1 gates hold; the three phase-1a slice metrics improve monotonically vs the Phase-1 baseline; phase-3 RO/opamp/SRAM/SC metrics improve.

### Phase 5 — Batched NN forward + Jacobian *(solver — performance)*

Accuracy-neutral, so it follows the accuracy work — but it *is* a prerequisite for the opamp wall-time DoD criterion and for tolerable iteration speed when debugging Phase 6.

- In `_stamp_mosfet_dc` (`solver.py:115`) and `_stamp_mosfet_transient` (`:1320`): (1) collect `(Vds, Vgs, Vbs, NFIN, L, T, tech_code)` rows for every NN MOSFET; (2) one `directnet_model(stacked_x, tech_codes=…)` call; (3) `autograd.grad` over the stacked input → block-diagonal Jacobian; (4) unpack and stamp.
- Add `_is_nn_mosfet()` alongside `_is_mosfet()` (`solver.py:94`); R/C/BSIM-CMG stay on the per-device path. DirectNet's tech embedding is already vectorized (`direct_net.py:81`).
- **Success gate:** 30-device opamp DC OP wall-time drops ≥ 5×; TSMC5/7/12/16 inverter DC + transient bit-identical to the Phase-1 baseline.

### Phase 6 — NR convergence upgrades *(solver — for RO / SRAM)*

- **6a — Levenberg-Marquardt damping** alongside the rail-cap heuristic (`solver.py:556`, `:1161`). When `‖F(x)‖` does not decrease, add `λ·I` to the MNA Jacobian, scale λ by 10 until accepted (Nielsen rule), shrink by 3 on acceptance.
- **6b — Residual-norm acceptance test.** Add `‖rhs − Av‖∞` as an or-gate so a stalled iterate with small `Δv` but large residual is rejected — and so the averaged-solution acceptance (`solver.py:683`, `:1292`) cannot lock a non-physical fixed point.
- **6c — Pseudo-transient DC continuation** as a fallback in `_solve_dc_with_retry` (`simulation.py`). Infrastructure exists (`solver._add_pseudo_capacitors`, `:1017`); expose it as a DC fallback ladder for circuits with no DC equilibrium (ring oscillators).
- **Success gate:** SRAM 6T `force_ic` DC converges to the seeded rail ≥ 90 % across NFIN corners; 5-stage RO transient period within ±5 % of NGSPICE.

### Phase 7 — Soft physics constraints *(model code — gated on Phases 1-6)*

Both shape the network, not the loss — so they sit cleanly inside the (post-Phase-1b) loss rule.

- **7a — Monotonicity penalty** on `id` w.r.t. `Vgs` (on-state) and `gds` w.r.t. `Vds`, as a residual-construction or weight-clipping layer inside `DirectNet` (`direct_net.py`).
- **7b — Spectral normalization on the `gds` output path** (Lipschitz bound) — bounds autograd-`gds` away from zero in saturation, shrinking the discontinuity Rule 5's hard floor must cover. The hard floor stays.
- Each constraint guarded by an opt-in train-CLI flag; bake-off vs the Phase-4 recipe before adoption.
- **Success gate:** Phase-4 gates hold; RO period and SRAM SNM improve.

### Phase 8 — Per-target output heads *(optional — lowest priority)*

Only if Phases 1-7 leave a measurable gap on the phase-3 opamp gain benchmark.

- Split the single `nn.Linear(hidden_dim, 13)` (`direct_net.py:55`) into per-group heads with their own 1-2 layer trunks: `[id, gm, gds, gmb]`, `[qg, qd, qs, qb]`, `[cgg, cgd, cgs, cdg, cdd]`. Keep single-shot semantics — no AR loop, no cross-head conditioning.
- Ship as a separate model size key (e.g. `medium-split`) without retiring `medium` until benchmarks accept.
- **Success gate:** opamp DC gain MRE improves ≥ 20 % vs the Phase-7 baseline without regressing inverter metrics.

## Suggested order of attack (TL;DR)

| # | Phase | Layer | Retrain? | Why it's where it is | Touches |
|---|---|---|---|---|---|
| 1 | gm/gds-fidelity lift (validation slices + Sobolev + column_weights + quantile LDS + gds floor) | training | yes (~2-3 h) | Attacks D0, the documented dominant blocker; simplest effective lever; on the opamp-gain critical path | `trainer.py:111,151-174,226`, `bni_mae.py:29-50`, `normalize.py:59`, `direct_net.py` (autograd hook) |
| 2 | Symmetrize C-stamps; conducting-branch `gm/gmb` floor | solver | no | Correctness; silent BDF-2 bug; parallels Phase 1 | `solver.py:1370-1415`, `mosfet_nn.py:434-446` |
| 3 | RO / opamp / SRAM SNM / SC benchmark harness | tests | no | Makes Phases 4-8 measurable; pure infra, no model dep | new files under `tests/` + `tests/references/` |
| 4 | New overlays + sinh sampling + LHS Vbs + NFIN grid + `--keep-offstate` | data | yes | Larger accuracy knob; needs Phase 1 harness + Phase 3 circuits | `nn_generate.py:52,281,445,491,562,790`, `dataset.py:39,63` |
| 5 | Batched NN forward + Jacobian | solver | no | Perf only (accuracy-neutral); unblocks opamp wall-time + Phase-6 iteration | `solver.py:94,115,1320` |
| 6 | LM damping + residual-norm test + pseudo-transient DC | solver | no | Convergence for RO / SRAM | `solver.py:556,683,1017,1161,1292`, `simulation.py` |
| 7 | Monotonicity penalty + spectral norm on `gds` head | model | yes | Convergence-friendly Jacobian; gated on 1-6 | `direct_net.py` |
| 8 | Per-target output heads (optional) | model | yes | Closes opamp gain gap if 1-7 leave one | `direct_net.py:55` |

## What we are explicitly NOT doing

- **No BSIMAR (LEVEL=74) work.** Rule 18 stands. Do not touch `transformer.py` or `mosfet_bsimar.py`. Solver and `_MOSFETNNBase` changes are shared infrastructure validated against DirectNet only.
- No AC, harmonic-balance, S-parameter, NQS-RF, or noise validation. DC + Transient only.
- No re-adding `DirectLoss / ChargeConsistencyLoss / SignConsistencyLoss / BoundaryLoss` or Vov-LDS / subthreshold-LDS axes. **Phase 1b adds exactly one** bounded Sobolev term — nothing else from the deleted loss family returns.
- No structural `apply_id_gate` resurrection (deleted 2026-05-03; Rule 15 subsumes it).
- No `torch.clamp` on voltage inputs (Rule 4, smooth softplus only).
- No ASAP7 work (Rule 17). No universal-vocab retrain — per-tech is the shipping convention.

## Risks and dead-end records

- **Risk: Phase 1b Sobolev term repeats the SlopeMatchLoss failure.** Mitigation: `--sobolev-weight` defaults to 0; the term must win a head-to-head bake-off against the plain V6.3.1 retrain recipe (same 8 cells, same seeds) before becoming default. If it loses, revert it, restore Rule 10 verbatim, and log the dead end. Distinguishing fact vs B2: B2 was deleted *unvalidated and corrupted*, never proven harmful — Phase 1b is the validated retry.
- **Risk: C-stamp symmetrization (2a) breaks inverter transient parity.** Mitigation: `NN_SYMMETRIC_CAPS=1` env gate, default-off until phase-3 benchmarks accept.
- **Risk: phase 4 overlays cost weeks of compute.** Mitigation: per-overlay budget caps; Phases 1-3 deliver value before any Phase-4 retrain.
- **Risk: phase 7 monotonicity / spectral-norm constraints cost inverter accuracy.** Mitigation: opt-in CLI flags; bake-off vs the Phase-4 recipe before default.
- **Risk: phase 8 per-target heads regress inverter parity.** Mitigation: ship as a separate size key; do not retire `medium` until benchmarks accept.
- **Dead-end to avoid (V5/V6 history):** heads-only conductance losses, structural Vds gates, hard `torch.clamp` — all reverted in CHANGELOG. Do not re-propose.
- **Dead-end to avoid (V6.1 → V6.2):** trusting uniform-split val NRMSE as the early-stop signal. Phase 1a exists to make that bug uncatchable.
- **Dead-end to avoid (V6.3.1):** treating the inverter trip error as a coverage gap. Three `inv_trip` revisions proved it is gain amplification — Phase 1b/1c (fidelity), not more samples, is the lever.
- **Dead-end to avoid (Rule 15(a) sign):** the V6.2 sign-fix is shipping for all four techs. Do not flip it without Rule-20 re-validation per checkpoint.

## External prior art (DirectNet, DC + Transient relevant only)

- **BSIM-NN, Tung & Hu, IEEE TED 2023** — production NN compact model from Berkeley BDMC. Symmetrized capacitance convention (Phase 2a); explicit derivative training (precedent for the Phase 1b Sobolev term).
- **Sobolev Training for Neural Networks, NeurIPS 2017** — training a network to match target *derivatives* alongside values; the formal basis for Phase 1b.
- **BSIM-CMG111.2.1-NN-assist, BDMC Jan 2025** — NN corrections on top of analytical BSIM-CMG; a defensible fallback if a phase stalls.
- **Scalable Monotonic Neural Networks, ICLR 2024** — soft monotonicity via weight clipping / residual constructions. Input for Phase 7a.
- **Input-Convex Lipschitz layers, arXiv:2401.07494, 2024** — bounds the autograd Jacobian per head. Input for Phase 7b.
- **Physics-Enhanced NN Compact Model preprint, May 2025** — first external NN compact model to pass SRAM SNM and LDO DC. Establishes phase-3c as the right complex-circuit gate.

## Definition of done

DirectNet is "complex-circuit ready" when:

1. All four phase-3 benchmarks pass against NGSPICE within their stated tolerances (RO period ±5 %, opamp DC gain ±10 %, SRAM SNM both curves positive + `force_ic` ≥ 90 %, SC charge transfer ±5 %) on TSMC5/7/12/16.
2. TSMC5/7/12/16 inverter VTC and transient match or beat the V6.3.1 numbers (VTC MaxErr 66.4 / 65.8 / 78.3 / 45.4 mV; transient post-startup 39.5 / 50.3 / 58.2 / 55.3 mV).
3. Wall-time on the two-stage opamp DC OP is within 5× of BSIM-CMG LEVEL=72 on the same hardware.
4. `docs/CHANGELOG.md` gets a `V7.0 — DirectNet complex-circuit support` entry with per-tech metrics and the four benchmark numbers; if Phase 1b ships, CLAUDE.md Rule 10 is amended and the bake-off result recorded.
5. CLAUDE.md status section updated; Rule 18 (BSIMAR park) stays untouched.
