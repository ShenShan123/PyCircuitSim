# NN Compact Model v5 — Inverter Transient Improvement Plan

**Date:** 2026-04-21
**Revision:** v1.1 (after adversarial review — see §10)
**Branch target:** `feat/bsimar-v5`
**Author:** Claude (Opus 4.7) + subagent team (inference / training / architecture / reviewer)
**Status:** PROPOSAL — awaiting approval before implementation

---

## 1. Executive Summary

Shipping-state (commit `381bbfc`) is:

| Tech  | VDD   | DirectNet inv. transient | BSIMAR inv. transient | Inv. VTC (AR) |
|-------|-------|--------------------------|-----------------------|---------------|
| TSMC5 | 0.65V | 17.20 % (marginal FAIL)  | 12.13 % PASS          | 13.96 %       |
| TSMC7 | 0.75V | 8.87  % PASS             | 9.14  % PASS          | 19.15 % (FAIL)|
| TSMC12| 0.80V | 11.65 % PASS             | 6.78  % PASS          | 4.10  % PASS  |
| TSMC16| 0.80V | 10.59 % PASS             | 7.51  % PASS          | 3.40  % PASS  |

Plus single-device NMOS pulse 8/8 PASS (avg ~1.3 % BSIMAR, ~2.3 % DirectNet).
The two remaining problems:

- **TSMC5 DirectNet transient** sits just above the 15 % threshold.
- **TSMC5 + TSMC7 VTC** are 13–19 % — DC accuracy at low-VDD technologies is
  the weakest link and propagates into all feedback-circuit errors.

The v3 post-mortem (`v3_wider_voltage_retrain_report_2026_04_21.md`) ruled out
widening the training voltage box at fixed capacity. The v4 fix report
(`v4_inverter_fix_report_2026_04_19.md`) ruled out adding sign/boundary losses
*without* structural changes.

**v5's thesis:** stop asking the loss function to enforce physical boundary
conditions — **bake them into the network forward pass**, and free the loss to
fit the interior of the operating range. This is the third fix listed in
`v4_inverter_fix_report_2026_04_19.md` as "Priority 2: structural model
change", which has not yet been attempted.

---

## 2. Root-Cause Synthesis (distilled from the 6 v4 reports)

1. **Flat-zero NN extrapolation past the training `|Vds|` range** creates a
   false KCL equilibrium — fixed at inference by the rail-restoring quadratic
   ramp in `_apply_vds_correction` (`mosfet_directnet.py:498-514`).
   **v5 keeps this unchanged.**

2. **Non-zero NN `Id` at `Vds=0`** breaks the inverter rail state — currently
   patched at inference with the one-sided `1-exp(-|Vds|/VT)` factor plus
   hard sign enforcement (`mosfet_directnet.py:520-559`). The patch is
   mathematically sound but creates a hard zero-gradient region at `Vds=0`
   and a discontinuous sign clip, both of which produce Jacobian "cliffs"
   that NR damping just barely absorbs. This is the dominant source of
   residual inverter error.

3. **Wrong-sign subthreshold Id in the BSIMAR Transformer** (reported 2026-04-15)
   was partly masked by the rail-fix but is still visible: BSIMAR VTC at TSMC7
   is 19 % and at TSMC5 is 14 % because the Transformer's cutoff-region Id has
   the wrong sign around `Vds ≈ 0`. Inference-side sign enforcement zeroes
   the prediction; zero current at the true off-state is wrong by the physical
   subthreshold leakage (tens of pA at 16nm).

4. **`gds` floor asymmetry** between the 4-output and 13-output eval paths
   (0.02·|id| vs 0.5·|id|, see `_eval_autograd4` vs `_eval_hybrid13`).
   DirectNet hits the 0.02 path in 4-output mode and the 0.5 path in 13-output.
   Observable as a ~10× inverter-gain discrepancy at rail states; not fatal
   but couples into TSMC5 marginal failures.

5. **Sign/Boundary losses during training hurt the inverter** — confirmed
   by the 2026-04-19 experiment (1/8 PASS, massive regression). They improve
   average NRMSE at the expense of the tiny-Id linear-region balance the
   inverter rail needs. **Conclusion:** these constraints must be structural,
   not loss-based.

6. **Capacity-vs-box tradeoff** — the 5.15 M BSIMAR and the 1.13 M DirectNet
   are saturated on current 2.0× VDD data. Widening data (v3) without
   widening the model is strictly worse. If we want more coverage we also
   need more params, at roughly 10 M for 3.0× VDD per the v3 post-mortem.

---

## 3. v5 Goals & Success Metrics

| Metric                                     | v4 shipping | v5 target | Acceptance |
|--------------------------------------------|-------------|-----------|------------|
| DirectNet inverter transient, worst tech   | 17.20 % (TSMC5) | < 10 % | all 4 techs < 15 % |
| BSIMAR inverter transient, worst tech      | 12.13 % (TSMC5) | < 8  % | all 4 techs < 10 % |
| BSIMAR inverter VTC, worst tech (AR)       | 19.15 % (TSMC7) | < 8  % | all 4 techs < 10 % |
| NMOS pulse, all techs                      | ≤ 4.81 %    | no regression | all 4 techs < 5 % |
| Per-device NRMSE (test split, phys space)  | 0.25–0.27 % | ≤ 0.30 %   | no regression |
| BSIM-CMG regression tests                  | all PASS    | all PASS | zero regression |
| Training wall-clock per model on A100      | ~6–12 hr    | ≤ 12 hr  | production CPUs still usable |
| **Transient charge conservation** (max `\|Σq(t+dt)−Σq(t) − Δt·Σi\|` / VDD·Cgg per step) — new metric, see §4.3 | unmeasured | < 1 % | all 4 techs < 1 % on inverter transient |

**Non-goals:**
- ASAP7 support (still needs separate embedding-vocab expansion; filed).
- SRAM bitcell transient (separate phase).
- `torch.compile` perf — deferred until accuracy lands.

---

## 4. The v5 Change Set (five structural levers)

### 4.1 Hard Id-rail gating inside the forward pass *(load-bearing change)*

**Change:** multiply the network's raw `id` physical-space output by
`tanh(Vds/VT_arch)` **at the boundary between the forward pass and the
simulator output** (after denormalization, before autograd for gm/gds/gmb),
with `VT_arch` a **fixed** per-tech scalar set to `VT_arch = 0.04 V`
(= 1.5× thermal voltage at 300 K). `VT_arch` is non-trainable by default
(see §7 risk table for why we don't learn it) and lives as a buffer on
`_MOSFETNNBase`.

**What the gate does fix (structurally):**
- `Id(Vds=0) = 0` exactly, regardless of NN output. This eliminates the
  rail-state KCL residual that the v4 inference patch only approximates.
- Autograd through `tanh` gives a C∞ Jacobian, so `gds = dId/dVds` in the
  linear region comes out of the chain rule — no hard sign clip, no
  discontinuity, no need for the `|id_raw|·exp(-|Vds|/VT)/VT` product-rule
  patch in `_apply_vds_correction`.

**What the gate does NOT fix (called out explicitly per adversarial review):**
- The **BSIMAR wrong-sign subthreshold bug is a Vgs-side training failure**
  (`exp((Vgs-Vth)/nφt)` sign error at Vgs ≈ 0), **not a Vds antisymmetry
  issue**. `tanh(Vds/VT_arch)` only fixes the Vds-driven rail; it does
  nothing about the Vgs-side. §4.3 and §4.4 target the Vgs side instead.
- Physical `Id` is **not** odd in `Vds` when `Vbs ≠ 0` (body effect makes
  source and drain inequivalent). We are deliberately imposing an approximate
  symmetry that is exact for `Vbs = 0` and within a few % for the body-bias
  ranges seen in inverter operation. For any circuit where that approximation
  fails (e.g. SRAM read with stacked access transistors), v5 needs the
  inference rail-fix as a safety net — §4.5 keeps it.

**AR-conditioning subtlety (BSIMAR only, per adversarial review):** the
Transformer feeds `id_pred` back as conditioning for steps 1..7 (gm, gds,
gmb, charges). If we gate `id_pred` *before* it enters the AR token
embedding, downstream heads lose the "raw current magnitude" signal at
rail states and have to re-infer it from context. That is the bug-prone
path.

**Resolution:** keep **two named outputs** from the id head:
- `id_raw` — the un-gated physical Id (what the AR token sees, what gm/gds
  downstream heads are conditioned on during both training and inference).
- `id_gated = id_raw * tanh(Vds / VT_arch)` — what the simulator consumes
  and what the training loss for the `id` column supervises.

Training loss on columns `gm, gds, gmb, q*, c**` continues to use `id_raw`
as context, and the PyCMG targets for those columns already match the
un-gated physical current (they were computed by the ground-truth BSIM-CMG
at the operating point — no gating in PyCMG). Supervising `id_gated`
against the PyCMG `id` target is still correct because `tanh ≈ 1` at any
sampled `|Vds| ≥ 0.1 V` and the dense Vds=0 points in §4.4 now train the
tanh to do its job.

**Where to insert (DirectNet v5):** refactor `direct_net.py` so
`DirectNetV5` exposes `self.trunk: nn.Sequential`, `self.head_id: nn.Linear`,
`self.head_rest: nn.Linear`. Forward returns a dict
`{'id_gated': id_raw * tanh(Vds / VT_arch), 'id_raw': id_raw, 'rest': ...}`.
The training loss picks the correct one per column; the simulator consumes
`id_gated`. 12-output layout (see §4.2) keeps the cap block in `head_rest`.

**Where to insert (BSIMARV5):** the Transformer's position-0 output (id
head) produces `id_raw`. The AR token for the next step is embedded from
`id_raw` (no gating). A separate "gated-id head" applies the tanh gate
*in parallel* with the AR unroll and is the value fed to the simulator.
This costs one extra scalar head at the encoder output; the AR loop is
unchanged.

**Cost:** ~60 LOC per model (slightly more than v1.0 plan because of the
dual-head split), zero extra parameters (VT_arch is a buffer), no training
cost change. Re-training required for both models.

### 4.2 `qs` wired as analytic residual in the head, not post-hoc

**Change:** drop `qs` from the direct output vector in both models; the head
emits 12 outputs, and `qs = -(qg + qd + qb)` is computed inside `forward()`.

**Why:** the simulator already does this post-inference (Rule #17), so the
NN currently spends capacity on a quantity that gets thrown away. Removing
it frees that capacity for the remaining 12 targets. BSIMAR also loses one
AR step (8 → 7), which slightly reduces the exposure-bias surface area.

**Cost:** ~20 LOC per model, training data columns re-mapped in dataset
loader. The reordered BSIMAR_COLUMN_ORDER becomes
`[qg, qb, qd, id, gm, gds, gmb, cgg, cgd, cgs, cdg, cdd]` (12 entries).

### 4.3 Charge-consistency in **normalized-space only**, plus validation-time diagnostic

*(Revised after adversarial review — v1.0 proposed autograd through asinh
denorm, which is the exact path the v3 "N4" postmortem ruled out.)*

**Change:** replace the v1.0 proposal (autograd `∂q_phys/∂V_phys` during
training) with two separate mechanisms that avoid the asinh chain-rule trap:

1. **Normalized-space charge-consistency penalty (training):** compute
   `∂q_norm/∂V_norm` via autograd and compare against a **pre-computed
   normalized cap target** `C_norm = C_phys_target × in_std /
   (out_std × sqrt(asinh_scale² + q_phys_target²))`. Both sides live on
   the flat normalized surface, so the `sqrt(s²+y²)` cosh term that
   sank N4 never enters the loss. The target normalization happens once
   at dataset load time, not per-batch.
   - Weight: 0.05 on top of MAE+LDS (half the v1.0 value — we're less
     confident, so tune conservatively).
   - Active for the last 30 epochs of the TF cosine schedule (not the
     full run — this is a fine-tune for the already-trained cap surface).
   - 25 % batch sub-sampling, same as v1.0.

2. **Charge-conservation validation diagnostic (non-training):** a new
   `tests/verify_nn_charge_conservation.py` measures, at every transient
   timestep, `Σ_terminal (q(t+dt) - q(t)) - Δt·(Σ i_terminal)` per device
   and reports the max violation over the transient window. This is an
   acceptance metric added to §3. It catches any residual charge-
   conservation error regardless of whether §4.3.1 trains well.

**Why this is different from N4 (the dead end):** N4 (filed in the v3
postmortem as "Known-infeasible explored options") computed the
consistency residual in physical space after asinh denorm, which
multiplies the gradient by `cosh(asinh(q/s))` = `sqrt(1 + (q/s)²)` per
sample. That factor vanishes for small q and blows up for large q,
turning the loss into a non-uniform regulariser that fought asinh
normalization. v5.4.3.1 stays in normalized space, where the cap target
has already absorbed the chain-rule term.

**Why we also keep the direct cap supervision:** the 5 cap columns are
still regression targets with standard MAE+LDS weight; §4.3.1 is an
*additional* term, not a replacement. If §4.3.1 causes training
instability, we can drop it without affecting anything else.

**Expected cost:** ~1.15× current wall-clock (not 1.3× per v1.0 —
smaller weight + fewer active epochs + no cosh explosion). If training
crosses the 12 hr Sprint B gate, drop §4.3.1 and keep only the diagnostic.

### 4.4 Inverter-aware sampling, NOT wider box

**Change:** regenerate training data with `voltage_box_factor=2.0` (unchanged)
but with a new **per-tech inverter-slice overlay**. Per the adversarial-review
feedback, overlay is done in a way that plays well with LDS:

- **Perturbed trajectory, not on-locus.** For each tech/variant/(L, NFIN),
  compute the DC inverter trajectory `f_inv(Vg)` from PyCMG, then draw
  2 000 samples from a Gaussian tube around it:
  `(Vg, Vd + ε_Vd, Vbs)` with `ε_Vd ~ N(0, (0.05·VDD)²)`. This gives the NN
  signal off the exact locus so it generalises to the nearby states NR
  actually visits.
- **Rail-tail samples** (1 000 per bin): `{Vg ≈ 0 ∪ Vg ≈ VDD} × Vd ∈
  [0, 0.2·VDD]` — the pinned-rail region where VTC convergence happens.
- **LDS bypass for overlay rows.** Overlay samples carry a
  `is_overlay=True` flag loaded alongside the data. The LDS weight for
  those rows is multiplied by a fixed priority factor (default 2.0)
  instead of going through per-target bin-density normalization. Without
  this, LDS would exactly down-weight the overlay rows we added (the
  reviewer's point). Code change is a ~20-line tweak in
  `bsimar/training/trainer.py`.

**Why:** the v3 post-mortem's forward guidance — "sample densely in the
plausible NR trajectory, sparsely elsewhere". Uniform LHS at 2× VDD is
fine for single-device DC but underweights the rail tails that dominate
inverter error. The perturbation + LDS-bypass are the corrections that
turn this from an LDS-collision (original proposal, flagged by reviewer)
into a targeted density boost.

**Cost:** 5 000 extra samples × 954 bins = 4.8 M extra rows = ~40 % bigger
dataset (~4 GB). Generation time ~45 min (parallel with 12 workers). No
net training-time increase because we raise batch size proportionally.

### 4.5 Unify and simplify the inference layer

**Change:** three tidy-ups to `_MOSFETNNBase`:
1. Unify `gds` floor to `max(|id| * 0.5, 1e-12)` across all eval paths (remove
   the `0.02` coefficient from `_eval_autograd4`).
2. Delete the one-sided `f_id` factor, the symmetric `gds_linear` term, and
   the hard sign clip from `_apply_vds_correction`. Keep only the rail-
   restoring quadratic. The tanh gate (4.1) replaces all three at the model
   level.
3. Replace the quadratic rail ramp with a `softplus`-based ramp so the
   transition into full restoring conductance is C∞ (currently C¹ at
   `|Vds| = VDD_train`). Minor improvement for TSMC12/16 where the operating
   point sits right at the boundary.

**Cost:** ~50 LOC deleted, ~15 added. Requires the v5 checkpoints to be in
place (this is a v5-only code path; v4 checkpoints keep the old correction).

---

## 5. Implementation Plan (3 sprints, ~2 weeks)

### Sprint A — §4.1 *alone* (structural forward + minimal training)

*Ordering corrected per adversarial review R5: prove §4.1 in isolation
before layering §4.3 and §4.4 on top.*

| # | Task | Owner | Est. | Verifies |
|---|------|-------|------|----------|
| A1 | Add `DirectNetV5` with dual-head Id (`id_raw` + `id_gated`) + 12-output layout | impl | 1 d | unit test: `Id_gated(Vds=0) = 0` exactly; autograd `gds(Vds=0) = id_raw/VT_arch` |
| A2 | Add `BSIMARV5` with `id_raw` AR token + parallel `id_gated` head, 7-step AR (qs dropped) | impl | 1.5 d | unit test: AR loop still returns 12 outputs; `Id_gated(Vds=0)=0` |
| A3 | Update `BSIMARNormalizer` + `OUTPUT_COLUMNS_V5` for 12 outputs; keep v4 loader compat | impl | 0.5 d | dataset loader test |
| A4 | Train DirectNetV5 NMOS + PMOS on **existing v4 data** (no §4.4 overlay, no §4.3 consistency) | impl | 1 d each (parallel) | test NRMSE ≤ 0.30 %; inverter VTC + transient regression gate |
| A5 | Train BSIMARV5 NMOS + PMOS on existing v4 data, same TF+AR schedule | impl | 1.5 d each | phys-best NRMSE ≤ 0.30 %; inverter VTC + transient regression gate |

**Gate A (must pass to proceed to Sprint B):**
- All four v5-gate-only checkpoints beat their v4 equivalents on inverter
  transient by ≥ 2 percentage points on at least 3 of 4 techs, AND
- No per-device NRMSE regression > 0.05 % absolute, AND
- `tests/verify_bsimcmg_*` zero regression.

If Gate A fails, stop and diagnose §4.1 in isolation rather than piling on.

### Sprint B — §4.3 + §4.4 on top of §4.1

| # | Task | Owner | Est. | Verifies |
|---|------|-------|------|----------|
| B1 | Implement `--charge-consistency-weight` in normalized space only (§4.3.1), with pre-computed `C_norm` targets | impl | 1 d | loss unit test: zero when predictions match targets in normalized space |
| B2 | Implement `is_overlay` flag + LDS bypass + perturbed trajectory (§4.4) in `PyCMG/scripts/generate_nn_data.py` and `trainer.py` | impl | 1 d | sample density plot; LDS weight inspection shows overlay rows at priority 2.0 |
| B3 | Generate v5 training data with inverter overlay (4 TSMC techs × 21 variants) | impl | 1 d | dataset file present; `is_overlay` flag roundtrips |
| B4 | Re-train all 4 v5 checkpoints with §4.3 + §4.4 enabled | impl | 1.5 d each (parallel) | acceptance metrics in §3 |
| B5 | New `tests/verify_nn_charge_conservation.py` — transient-time KCL at each dt | impl | 1 d | < 1 % on all 4 techs |
| B6 | Run full L1 + L1+ NN test battery on final v5 checkpoints | impl | 0.5 d | all §3 acceptance metrics met |

**Gate B:** all §3 metrics met including new charge-conservation metric. If
§4.3 causes training instability or OOM, drop it and ship §4.1 + §4.4 only.

### Sprint B — training + validation

| # | Task | Owner | Est. | Verifies |
|---|------|-------|------|----------|
| B1 | Generate v5 training data with inverter overlay (4 TSMC techs × 21 variants) | impl | 1 d | dataset file present, LHS coverage + overlay plots match spec |
| B2 | Train DirectNetV5 NMOS + PMOS (800 epochs, charge-consistency weight 0.1 last 20 %) | impl | 1 d each (parallel) | test NRMSE ≤ 0.30 %; gds autograd vs target caps disagreement < 5 % |
| B3 | Train BSIMARV5 NMOS + PMOS (150 TF + 5 AR epochs) | impl | 1.5 d each (A100+Blackwell parallel) | phys-best NRMSE ≤ 0.30 %; AR val gap ≤ 15 % of TF val |
| B4 | Port `_MOSFETNNBase` to the 12-output layout and the simplified `_apply_vds_correction` | impl | 0.5 d | unit test: `Id`/`gds` continuity at `|Vds|=VDD_train` within `1e-9` |
| B5 | Update `parser.py` auto-resolution to prefer `v5_*` checkpoints, fall back to `v4_*` with deprecation warning | impl | 0.5 d | existing netlists parse |
| B6 | Run L1 + L1+ NN tests: `verify_nn_dc.py`, `verify_nn_tran_v4.py`, `verify_nn_dc_tran.py` with v5 checkpoints | impl | 0.5 d | acceptance metrics in §3 |

**Gate:** Sprint B merges only if all §3 acceptance metrics are met. If any
single-metric fails, we fix forward before proceeding to Sprint C.

### Sprint C — production hardening

| # | Task | Owner | Est. | Verifies |
|---|------|-------|------|----------|
| C1 | Delete v4-only `_apply_vds_correction` branches once v5 is default | impl | 0.5 d | `rg 'sign_weight|boundary_weight'` returns nothing |
| C2 | Update CLAUDE.md rules #19/#20 with v5 facts; remove obsolete rule #21 | impl | 0.25 d | doc review |
| C3 | Archive v4 checkpoints under `external_compact_models/bsimar/checkpoints/legacy_v4/`; add a CLI flag to force the v4 path for backward-compat testing | impl | 0.5 d | `--compat v4` runs |
| C4 | Write `results/v5_release_report_*.md` with before/after tables | impl | 0.5 d | this document's template |

---

## 6. Explicit Non-Changes (deliberate)

To keep the change set small and the blame-line short:

- **Do NOT touch the voltage box factor.** 2.0× stays. v3 settled this.
- **Do NOT reintroduce `SignConsistencyLoss` / `BoundaryLoss`.** The tanh
  gate replaces both at the architecture level. Enabling them *alongside*
  the gate would over-penalise the already-structurally-zero rail.
- **Do NOT widen the model.** 1.13 M / 5.15 M stays. Capacity isn't the
  bottleneck — boundary correctness is.
- **Do NOT change the inference rail-restoring quadratic ramp shape**
  (§4.5 item 3 is a softplus smoothing, not a new ramp). The quadratic
  is known to work; we only smooth the join.
- **Do NOT add `torch.compile` or batch-eval acceleration yet.** Correctness
  first.
- **Do NOT retrain on ASAP7.** Tech-code vocab is a separate problem.

---

## 7. Risks & Mitigations

| Risk | Likelihood | Blast radius | Mitigation |
|------|:----------:|--------------|------------|
| Tanh gate slows convergence because gradient at `Vds=0` is exactly the unscaled id-magnitude | Med | Training time 1.2× | `VT_arch` is trainable and initialised at 0.04 V; observed early-epoch loss vs. v4 is the gate |
| Charge-consistency loss causes OOM on A100 40 GB | Med | Need bigger batch on Blackwell | Pre-emptively sub-sample 25 % of batch for the consistency term, turn on only last 20 epochs |
| Removing `qs` breaks simulator code that expects 13 outputs | Low | All NN tests fail | Test on Sprint A (A4) before training; keep a compatibility shim in `_MOSFETNNBase` |
| V5 inverter VTC still fails on TSMC7 | Med | Sprint B gate fails | Fallback: per-tech fine-tune for TSMC5 + TSMC7 only (Priority 4 from 2026-04-15 report); 4 extra 1–3 hr runs |
| AR finetune OOM reproduces (seen in v3) | High | No AR-best checkpoint | Already mitigated by `num_workers=8` DataLoader fix (commit `fc35daf`); keep AR batch ≤ 4096 |
| Charge consistency loss conflicts with asinh denorm chain rule | Low | Loss landscape distorted | Autograd is computed on physical-space outputs after denorm; mathematically clean |
| Regression in NMOS pulse | Low | Test suite fails | Included in Sprint B gate; v4 checkpoints remain shippable |

---

## 8. Open Decisions for User

Before kicking off Sprint A, please confirm (or push back on) the following:

1. **Scope of v5 models.** Four checkpoints (DirectNet {N,P}MOS + BSIMAR {N,P}MOS)
   — same as v4. Is ASAP7 in or out? Plan assumes out.
2. **Hardware budget.** Sprint B is ~15 hr on A100 + ~15 hr on Blackwell
   running in parallel. Confirm availability.
3. **Acceptance bar for TSMC5.** §3 targets `< 15 %` DirectNet transient on
   all 4 techs. TSMC5 is the hardest; willing to accept 14–15 % or should
   we target `< 10 %` (which may require per-tech fine-tune in Sprint C)?
4. **Compat window.** §Sprint C C3 archives v4 instead of deleting. OK with
   keeping v4 around for one more release cycle?

---

## 9. Appendix: File-by-file change manifest (intent, not code)

| File | Action | Why |
|------|--------|-----|
| `external_compact_models/bsimar/models/direct_net.py` | Add `DirectNetV5` class; keep `DirectNet` alias for v4 load | §4.1 + §4.2 |
| `external_compact_models/bsimar/models/transformer.py` | Add `TransformerEncoderModelV5` with gated Id head, 7-step AR | §4.1 + §4.2 |
| `external_compact_models/bsimar/losses/direct_loss.py` | Make `ChargeConsistencyLoss` available as add-on term in unified CLI | §4.3 |
| `external_compact_models/bsimar/training/trainer.py` | Wire `--charge-consistency-weight` + late-phase activation; 12-output metrics | §4.3 |
| `external_compact_models/bsimar/data/dataset.py`, `normalize.py` | 12-output `OUTPUT_COLUMN_ORDER` + new `BSIMAR_COLUMN_ORDER` | §4.2 |
| `external_compact_models/bsimar/cli/train.py` | Add v5 flags; default `--model-version 5` | §4.2 + §4.3 |
| `external_compact_models/bsimar/config.py` | Add `TECH_CODE_MAP` v5 (unchanged); new `OUTPUT_COLUMNS_V5` | §4.2 |
| `external_compact_models/PyCMG/scripts/generate_nn_data.py` | `--inverter-overlay` flag; per-(tech, L, NFIN) inverter-slice dump | §4.4 |
| `external_compact_models/PyCMG/pycmg/nn_generate.py` | New `inverter_trajectory_samples()` helper (uses PyCMG DC solve) | §4.4 |
| `pycircuitsim/models/mosfet_directnet.py` | Simplify `_apply_vds_correction` to rail-restoring only; unify gds floor; 12-col layout | §4.5 |
| `pycircuitsim/models/mosfet_bsimar.py` | Same simplification; 7-step AR loop driver | §4.5 |
| `pycircuitsim/parser.py` | Prefer `v5_*` checkpoints, fall back to `v4_*` with warning | Sprint B B5 |
| `tests/common/nn.py` | Add v5 checkpoint resolver; keep v4 for compat | B5 |
| `tests/verify_nn_*.py` | No change in test logic; run green on v5 checkpoints | B6 |
| `CLAUDE.md` | Rule updates #19/#20; remove #21 after v5 ships | C2 |
| `results/v5_release_report_*.md` | New (C4) | C4 |

---

**Next action:** await user approval (items in §8). On approval, kick off
Sprint A in a new branch `feat/bsimar-v5`, and spawn a staff-engineer
subagent to review the PR at the end of each sprint.

---

## 10. Adversarial Review Summary (2026-04-21)

A staff-engineer subagent reviewed v1.0 of this plan before committing to
Sprint A. Seven challenges; five led to v1.1 revisions, two were accepted
as documented risks.

| # | Challenge | Verdict | Resolution |
|---|-----------|---------|------------|
| R1 | Tanh gate cascades into BSIMAR AR conditioning (gm/charges downstream of gated id) | **Valid** | §4.1 now uses dual-head: AR sees `id_raw`, simulator consumes `id_gated`. No gating inside AR loop. |
| R2 | "Tanh(-x)=-tanh(x) fixes subthreshold sign bug" is wrong — subthreshold sign is Vgs-driven, not Vds-driven | **Valid, major** | §4.1 rewritten to explicitly disclaim this. Wrong-sign subthreshold is a Vgs-side problem; §4.4 (inverter overlay) and kept inference rail-fix (§4.5) target it instead. |
| R3 | Autograd-based charge consistency in physical space is the exact N4 dead end from v3 postmortem (cosh factor from asinh chain rule) | **Valid, major** | §4.3 rewritten to do consistency in **normalized space only** against a pre-computed normalized cap target; physical-space diagnostic moved to a validation-only metric. |
| R4 | Inverter overlay + LDS = LDS will down-weight the samples we added | **Valid** | §4.4 adds `is_overlay` flag with LDS bypass + priority factor 2.0. Also perturbed samples off the exact locus. |
| R5 | §4.3 + §4.4 + §4.1 together push capacity demand more than isolated | **Partially valid** | Sprint staging updated: §4.1 alone in Sprint A, §4.3 + §4.4 enabled only after Sprint A proves no test-NRMSE regression. |
| R6 | Trainable `VT_arch` may collapse to floor or drift high, masking systemic rail underprediction | **Valid** | §4.1 now uses **fixed** `VT_arch = 0.04 V` buffer, not a trainable parameter. If Sprint B shows evidence we need to tune it, revisit in a follow-up. |
| R7 | §3 metrics are terminal-space; no measurement of transient-time charge conservation violation | **Valid** | New acceptance metric added to §3; new `tests/verify_nn_charge_conservation.py` in Sprint B6. |

Two challenges *not* resolved (documented risks, not blockers):
- **R2 residual:** for `Vbs ≠ 0` operating points the tanh-gated Id is not
  perfectly antisymmetric in Vds; SRAM and stacked-access circuits are the
  risk surface. The kept inference rail-fix (§4.5) is the safety net. First
  bistable-circuit verification that exposes this should trigger a v5.1.
- **R5 residual:** we may still hit capacity saturation when §4.3 and §4.4
  turn on together. Sprint B B2/B3 gate catches this; if it happens, drop
  §4.3 (documented fallback) and only keep §4.1 + §4.4.

Full adversarial-review text is preserved in the agent transcript (not
copied into this plan to keep it focused on the revised decisions).

