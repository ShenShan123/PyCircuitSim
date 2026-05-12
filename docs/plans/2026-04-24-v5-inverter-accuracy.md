# v5 Inverter Accuracy Plan — Trim First, Then Fix the Real Failure Mode

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans`.
> Each step is checkbox-tracked (`- [ ]`).  This plan supersedes
> `results/v5_improvement_plan_2026_04_21.md`, which is closed.

**Date:** 2026-04-24
**Branch target:** `feat/bsimar-v5`
**Status:** PLAN — awaiting approval to start Sprint 0

---

## 0. Why this plan exists (read first)

The closed v5 sprint (E1/E3/D1/E4/E5) ran four retraining/inference experiments and
reverted all four.  The session summary
(`results/v5_session_summary_2026_04_23.md`) and the D1 diagnostic
(`results/v5_d1_tsmc7_nmos_errors/v5_d1_tsmc7_nmos_report.md`) leave us with one
load-bearing finding and one production-ready hypothesis:

1. **The TSMC7 NMOS DC error (14.72 % BSIMAR / 15.79 % DirectNet) is a
   sampling-basis mismatch, not a data-volume problem.**  The verifier sweeps
   uniform `Id-Vgs` at `Vds = VDD/2`; LHS training under-samples that
   high-current saturation plateau by ~16× relative to the region the verifier
   metric weighs.  Per-tech fine-tune (E3) lifted training NRMSE to 0.45 % and
   left inference NMOS DC at 14.74 % — a 29× gap.
2. **The PMOS DC weakness on TSMC12/16 (12-14 % both models) is the same
   class of bug** — wide universal-LHS coverage with the same uniform-sweep
   verifier metric.  Same root cause is highly likely; not yet diagnosed.

In parallel, the production code accumulated complexity that did not pay off:

- `DirectLoss` is imported by the trainer but never instantiated.  Both models
  hard-wire `MAELoss` + LDS.  Dead.
- `SignConsistencyLoss` and `BoundaryLoss` were added in the v4-fix sprint and
  remained in the v4 production training path.  The v5 baseline (with them on)
  still has 14-19 % VTC failures, and the v4-fix report itself documents 1/8
  PASS for sign/boundary alone.  No quantified marginal benefit to keep.
- `compute_lds_weights_per_target` is invoked **three times** (per-target,
  Vov/Vg, subthreshold) and the products multiplied.  The third axis was added
  to fix a wrong-sign subthreshold bug that the rail-restoring extrapolation
  patch (commit `381bbfc`) already neutralized at inference time.  No A/B
  evidence the third axis still helps.
- `BSIMARTransformer` carries seven optional knobs (parallel cap head, AR
  finetune, scheduled sampling, GPT-2 init, token-type embedding, grouped
  inputs, scalar projection).  Five are structural-always-on; the other two
  (scheduled sampling and AR finetune) only matter for the AR rollout path.
- `_eval_autograd4` in `pycircuitsim/models/mosfet_directnet.py` is a dead
  fast-path: every shipping checkpoint is 13-output.
- `TransformerConfig` has six dataclass fields (`w_curr` … `consistency_weight`)
  that no current code path reads.

The v5 plan therefore has **two phases**:

- **Phase A — Trim.**  Delete the dead and unjustified code.  No accuracy
  change expected; we measure to confirm no regression.  This is required
  before Phase B because every change in Phase B has to be A/B tested against
  a clean, comprehensible baseline.
- **Phase B — Fix the real failure mode.**  Address the sampling-basis
  mismatch with a *training-distribution* change, a *shape-aware* loss, and a
  *structural* boundary gate.  Each lever ships behind its own gate so we can
  isolate failures.

Non-goals of this plan, called out so reviewers do not redirect scope:

- ASAP7 retraining (separate vocabulary expansion task).
- SRAM bitcell verification (separate phase).
- `torch.compile` performance work.
- Removing PyCMG as the data oracle (PyCMG is the ground truth).

---

## 1. Real measured baseline (from `results/v5_baseline_2026_04_22.md`)

| Tech   | Model      | NMOS DC       | PMOS DC       | VTC          | Transient    |
|--------|------------|---------------|---------------|--------------|--------------|
| TSMC5  | DirectNet  | 6.20  PASS    | 7.74  PASS    | 10.73 ⚠      | 3.75  PASS   |
| TSMC5  | BSIMAR     | 4.59  PASS    | 5.97  PASS    | 13.96 ⚠      | 12.13 ⚠      |
| TSMC7  | DirectNet  | **15.79** ✖   | 6.53  PASS    | **18.14** ✖  | 6.80  PASS   |
| TSMC7  | BSIMAR     | **14.72** ✖   | 3.06  PASS    | **19.15** ✖  | 9.14  PASS   |
| TSMC12 | DirectNet  | 3.71  PASS    | **12.17** ⚠   | 4.86  PASS   | 4.86  PASS   |
| TSMC12 | BSIMAR     | 9.95  PASS    | **13.72** ⚠   | 4.10  PASS   | 6.78  PASS   |
| TSMC16 | DirectNet  | 3.11  PASS    | **12.40** ⚠   | 5.67  PASS   | 7.86  PASS   |
| TSMC16 | BSIMAR     | 8.96  PASS    | **13.48** ⚠   | 3.40  PASS   | 7.51  PASS   |

(✖ = FAIL 10 % DC threshold; ⚠ = marginal but inside the 15 % transient gate.
All eight transient cells are inside the gate.)

Three failure clusters:

- **F1.** TSMC7 NMOS DC + VTC (both models) — confirmed sampling-basis bug.
- **F2.** PMOS DC at TSMC12/16 (both models) — same class, not yet
  diagnosed; D1-style heatmap should run before any retrain.
- **F3.** BSIMAR TSMC5 transient at 12.13 % — only ~3 pp from the gate; this
  is the lone lever where the §4.1 tanh-gate from the closed v5 plan would
  actually load-bear.

Everything else is cleanly inside its threshold and needs no change.

---

## 2. Acceptance criteria

| Metric                                | Baseline (worst tech) | v5 target           | Hard gate           |
|---------------------------------------|-----------------------|---------------------|---------------------|
| NMOS DC NRMSE, all 4 TSMC techs       | 15.79 % (TSMC7 DN)    | < 8 %               | all 4 ≤ 10 %        |
| PMOS DC NRMSE, all 4 TSMC techs       | 13.72 % (TSMC12 AR)   | < 8 %               | all 4 ≤ 10 %        |
| Inverter VTC NRMSE, all 4 TSMC techs  | 19.15 % (TSMC7 AR)    | < 10 %              | all 4 ≤ 12 %        |
| Inverter transient NRMSE              | 12.13 % (TSMC5 AR)    | < 8 %               | all 8 cells ≤ 15 %  |
| NMOS pulse, all 4 TSMC techs          | ≤ 4.81 %              | no regression       | all 4 ≤ 5 %         |
| Test-split per-device NRMSE (phys)    | 0.22 %–0.27 %         | no regression > 0.05 % abs | ≤ 0.30 %     |
| BSIM-CMG regression suite             | all PASS              | all PASS            | zero regression     |
| Codebase line delta after Phase A     | n/a                   | ≥ −600 net LOC      | not blocking        |

Charge-conservation (the v4.1 plan's new metric) is nice-to-have but not a
gate; transient NRMSE already catches a charge-conservation failure
indirectly.  Keep `tests/diag_d1_tsmc7_nmos_errors.py` and add a PMOS-DC
variant; both are read-only diagnostics, not gates.

---

## 3. Phase A — Trim (no model change, no retrain)

The objective is to reduce the surface area that Phase B has to argue
against.  Each step is a small, reversible PR that lands behind a green
NN regression run on the v4 checkpoints.

### A1.  Delete dead loss code

Files: `external_compact_models/bsimar/losses/direct_loss.py`,
`external_compact_models/bsimar/losses/__init__.py`,
`external_compact_models/bsimar/training/trainer.py`,
`external_compact_models/bsimar/cli/train.py`,
`external_compact_models/bsimar/config.py`.

- [ ] Remove `DirectLoss` class entirely.  No production training path
      instantiates it; both models use `MAELoss` + LDS (proof: `rg
      'DirectLoss\(' external_compact_models/bsimar` returns only the
      definition and the import line).
- [ ] Remove `ChargeConsistencyLoss` (also dead per `rg`).  The v3
      postmortem already concluded the asinh chain rule makes it
      mathematically inert; we keep the lesson, drop the code.
- [ ] Remove the `_forward_4` branch of any remaining 13-vs-4 dispatch.
      All v4 checkpoints are 13-output; the 4-output path has not
      shipped since the universal-NN sprint.
- [ ] Drop `TransformerConfig.{ss_warmup_epochs, ss_max_ratio,
      consistency_weight, curriculum_warmup, w_curr, w_cond, w_charges,
      w_caps, w_zero_bias}` and the `BSIMARConfig`/`TrainConfig` legacy
      aliases.  Keep `DirectNetConfig.{w_id, w_gm, ...}` only if Phase A2
      keeps DirectLoss alive (it does not — see A1).

**Acceptance:** Re-run `tests/verify_bsimar_v4_inverter.py --tech tsmc12`
on a v4 checkpoint; numbers reproduce baseline ±0.05 pp.

### A2.  Remove the unjustified physics constraint losses

Files: `external_compact_models/bsimar/losses/bni_mae.py`,
`external_compact_models/bsimar/training/trainer.py`,
`external_compact_models/bsimar/cli/train.py`.

- [ ] Delete `SignConsistencyLoss` and `BoundaryLoss` classes.  The
      v4-fix report (2026-04-19) showed sign/boundary alone gave 1/8
      PASS, and the v5 baseline (with them on) still fails on TSMC7 DC.
      No A/B evidence they help; they consume training time and obscure
      the loss curve.
- [ ] Drop `--sign-weight` and `--boundary-weight` from
      `bsimar/cli/train.py`.
- [ ] Strip the `sign_loss_fn` / `boundary_loss_fn` parameters from
      `_train_epoch_direct`, `_train_epoch_mae`,
      `_train_epoch_scheduled_mae`, and the matching scoping logic in
      `train_directnet` / `train_transformer`.

**Why we are confident:** The closed v5 plan §4.1 R2 challenge already
established that the wrong-sign subthreshold bug these losses targeted
is **Vgs-side**, not Vds-side.  The current Vds-side rail-fix
(`_apply_vds_correction` step (a)) handles the actual failure mode.

**Acceptance:** v4-style retrain (Phase A4) on the trimmed pipeline
produces test-NRMSE within ±0.05 pp of the v4 number.  If it
regresses, restore one of the two losses and re-run; if both have to be
restored, this trim was wrong and we revert.

### A3.  Reduce LDS-weight stacking from three to one

File: `external_compact_models/bsimar/training/trainer.py`.

- [ ] Replace the three-axis LDS product
      (`per-target × Vov(Vg) × subthreshold`) with **per-target only**.
      Per-target LDS is the paper's recipe and is the only axis with
      published evidence.
- [ ] Delete the `vg_weights_np`/`subthresh_w` blocks in both
      `train_directnet` and `train_transformer`.
- [ ] If the post-trim retrain (A4) regresses TSMC7 NMOS DC by > 1 pp,
      re-enable Vov-LDS only (drop the subthreshold axis permanently).
      The subthreshold axis was added during the wrong-sign-Id sprint and
      is the most suspect.

**Acceptance:** retrain-then-verify (see A4); compare to v4 baseline.

### A4.  Confirm the trim with a "control" retrain

- [ ] Train `v5a_dn_universal_{nmos,pmos}` and
      `v5a_universal_{nmos,pmos}` on the existing
      `universal_{nmos,pmos}.npz` (no data change yet) with the trimmed
      training code (A1+A2+A3).  Default schedule, default architecture.
- [ ] Compare against the v4 baseline table in §1 with
      `tests/verify_bsimar_v4_inverter.py --tech <each>`.
- [ ] **Gate Phase A merge:** every cell in the baseline table must move
      by ≤ 1 pp NRMSE in either direction; no FAIL→PASS or PASS→FAIL on
      any cell.  Net code delta ≥ 600 LOC removed.
- [ ] Land the trim and the control retrain checkpoints together so any
      future bisect can attribute behaviour to one commit.

### A5.  Trim the simulator inference layer

File: `pycircuitsim/models/mosfet_directnet.py`.

- [ ] Delete `_eval_autograd4` and the dispatch on `self._output_dim ==
      13`.  Every shipping v4/v5 checkpoint is 13-output; the autograd-4
      path is dead.
- [ ] Unify the `gds` floor to one constant: `max(|id| * 0.5, 1e-12)`.
      The two paths used different coefficients (0.02 vs 0.5).
- [ ] Keep `_apply_vds_correction` as-is for now (Phase B5 simplifies
      it once the structural tanh-gate ships).

**Acceptance:** `pytest tests/verify_bsimcmg_*.py` zero regression on
the v4 checkpoints; `tests/verify_bsimar_v4_inverter.py` reproduces the
baseline numbers ±0.05 pp.

### A6.  Strip stale CLI flags and dataclass fields

- [ ] Remove `--sign-weight`, `--boundary-weight` from `cli/train.py`.
- [ ] Remove `bsimar/training/finetune.py::finetune_v4` if no Phase B
      step calls it.  (Phase B6 *does* call it for an
      inverter-trajectory mini-finetune, so keep finetune itself; just
      audit its arg list against current trainer signatures.)
- [ ] Drop unused imports surfaced by `pyflakes` after the deletions.

**Acceptance:** `python -m bsimar.cli.train --help` lists no dead flag;
`pyflakes external_compact_models/bsimar/` returns clean.

---

## 4. Phase B — Fix the real failure mode

Phase B has five levers, ordered by *prior probability of fixing the
identified failure modes (F1/F2/F3)* divided by *implementation
risk*.  Each lever ships in its own sprint with its own gate; if the
gate fails, we stop and decide whether to keep the change or revert
before stacking the next lever.

### B1. (Sprint S-DATA) — Replace LHS with a hybrid uniform-grid sampler

**Targets:** F1 (TSMC7 NMOS DC), F2 (PMOS DC TSMC12/16), F3 (BSIMAR
TSMC5 transient marginal).
**Confidence:** highest.  D1 is the smoking gun: the verifier metric is
weighted by `|Id|`, and `|Id|` lives at high-Vgs/high-Vds, where LHS
puts only 3.07 % of the TSMC7-SVT samples.

**Concrete change**, in `external_compact_models/PyCMG/pycmg/nn_generate.py`:

- [ ] **Per-bin distribution = uniform-grid + LHS jitter**, not LHS+anchors.
      For every (tech, variant, L, NFIN, T) bin, generate samples on
      a uniform 2D grid in `(Vgs, Vds)` over the *training box*
      (currently `[0, 2·VDD]` for NMOS), with a small Gaussian jitter
      `ε ~ N(0, 0.05·VDD)` per axis so the grid is not degenerate.  Add
      Vbs as a third coarse axis (5 levels: `[0, ±0.25, ±0.5]·VDD`).
      Total: **30 × 30 × 5 = 4500 grid samples per bin**, replacing the
      current `5000` LHS budget.  Keep the ten anchor points and the
      Vds=0 boundary line samples (they are cheap and proven).
- [ ] **Hot-region densification, structurally:** within
      `Vgs ∈ [0.5·VDD, VDD]` and `Vds ∈ [0.4·VDD, VDD]` (the D1 hot
      box, generalised to per-tech-VDD coordinates) double the grid
      density.  This is the saturation plateau; the verifier's NRMSE is
      dominated here.  Adds ~1500 samples per bin.
- [ ] **Tag every sample's source.**  `npz` adds a `sample_class`
      column with values `{anchor, vds_zero, hot, grid, jitter}`.  This
      is metadata only (no training change yet) and unblocks B2's
      shape loss and B6's inverter overlay.
- [ ] **Storage cap:** total dataset size ≤ 2× the current
      `universal_*.npz` (now ~5 GB → ≤ 10 GB).  If it overflows, drop
      coarse-Vbs from 5 to 3 levels; that is the cheapest knob.

**Wall clock cost:** data generation today is ~45 min on 12 workers.
The ~1.4× sample volume at the same cost-per-sample (PyCMG eval is the
hot path) costs about 60 min.  No training-time change because the
batch size is preserved.

**Gate S-DATA (must pass to merge):**

- A4-style control retrain on the *new* data set with the trimmed
  pipeline produces D1-equivalent heatmap (`tests/diag_d1_*`) on TSMC7
  SVT NMOS where the hot-box mean |rel err| drops from 9 % → ≤ 5 %.  If
  this fails, the data-distribution thesis is wrong and B2/B5 are still
  promising but B1 should be reverted.
- Inverter verifier: TSMC7 NMOS DC ≤ 8 % AND TSMC7 VTC ≤ 12 % AND no
  other-cell regression > 1 pp.

### B2. (Sprint S-LOSS) — Shape-aware loss on Id-Vgs slope

**Targets:** F1 + F2.
**Confidence:** medium-high.  The D1 hot-region error is a *shape*
error (the NN reproduces the |Id| magnitude but mis-curves the
saturation plateau), and a slope penalty is the cheapest direct
counter.

**Change** in `external_compact_models/bsimar/losses/`:

- [ ] Add `class SlopeMatchLoss(nn.Module)` that, on each batch, picks
      a random subsample of size `k ≤ 256` from rows where
      `sample_class == 'grid'` (set by B1) and computes
      ```
      L_slope = mean_i | (∂Id/∂Vgs)_nn  −  (∂Id/∂Vgs)_pycmg | / |Id_pycmg|_i
      ```
      with both derivatives in **normalised** space to avoid the asinh
      chain-rule trap (the same lesson as N4 / v5 §4.3 R3).  Compute
      `(∂Id/∂Vgs)_pycmg` from the analytical `gm` already in the
      training data (it is column 1 in `OUTPUT_COLUMNS`).  Compute
      `(∂Id/∂Vgs)_nn` via `torch.autograd.grad` on the model's
      already-predicted normalised Id with respect to the normalised Vg
      input.  Reuse the same forward graph; no extra forward pass.
- [ ] Add `--slope-weight` flag; default `0.05` (same conservative
      scale as the deferred N4).  Wire it in *only* the trimmed
      `train_directnet` and `train_transformer`.
- [ ] Active for the last 30 % of training epochs only (a fine-tune of
      the converged surface, not a from-scratch term).

**Why this is not the dead-end N4:**  N4 tried `∂q/∂V` consistency in
**physical** space after asinh denorm; the cosh chain-rule term turned
the loss into a non-uniform regulariser.  S-LOSS's slope match is in
**normalised** space, where both sides have already absorbed the
chain-rule terms.  Mathematically clean.

**Gate S-LOSS:**

- TSMC7 NMOS DC ≤ 7 % (improvement over S-DATA only); per-target test
  NRMSE no worse than 0.30 %.  If TSMC7 fails to improve over S-DATA
  alone, drop S-LOSS — the slope penalty was redundant given enough
  uniform-grid data.

### B3. (Sprint S-ARCH-A) — Structural Vds gate on the Id head

**Targets:** F3 (BSIMAR TSMC5 transient), and a smaller win on F1/F2
*if and only if* B1 + B2 leave any inverter-rail tail.

**This is the §4.1 lever from the closed v5 plan, kept on the bench
because it is the only one that addresses rail-state correctness
*structurally* rather than via the inference patch.**

**Change:**

- [ ] In `bsimar/models/direct_net.py` and
      `bsimar/models/transformer.py`, multiply the post-denormalisation
      `id_phys` output by `tanh(Vds_phys / VT_arch)` where `VT_arch =
      0.04 V` (1.5 × kT/q at 300 K) is a **fixed buffer**.  Two named
      outputs: `id_raw` (what the AR loop sees / what the cap heads
      condition on) and `id_gated = id_raw · tanh(Vds/VT_arch)` (what
      the simulator and the supervised loss consume).  Same dual-head
      pattern as the closed v5 plan.
- [ ] Update the BSIMAR AR conditioning so step-1's token sees `id_raw`
      (no gating); the gated value goes to the loss directly.
- [ ] Once the v5b checkpoints exist, simplify `_apply_vds_correction`
      to **rail-restoring extrapolation only** (delete the one-sided
      `f_id`, the `gds_linear` term, the sign clip).  These were
      patches for behaviour the gate now handles structurally.  See B5.

**Cost:** ~80 LOC across two model files + ~50 LOC deleted in
`_apply_vds_correction`; two retrains (DN+AR for each of NMOS+PMOS),
~4 hours of GPU each.

**Gate S-ARCH-A:**

- BSIMAR TSMC5 transient ≤ 8 % AND no transient cell regresses past
  baseline by > 1 pp AND `Id_gated(Vds=0) ≡ 0` to numerical zero
  (unit test in `tests/test_nn_id_gate.py`).
- If the gate fails (Sprint S-ARCH-A retrain misses the transient
  target), keep the *symbolic* gate in inference (it is structurally
  correct) but document the fail and proceed.  Do not roll back unless
  it actively regresses other cells.

### B4. (Sprint S-ARCH-B) — Trim the BSIMAR Transformer

**Targets:** maintainability, training time, *not* accuracy.  This is
inside Phase B because we want to test the trim under the new data and
loss.

**Change** in `bsimar/models/transformer.py`:

- [ ] **Drop the parallel cap head.**  All 13 outputs go through the
      same per-token regression heads as the AR sequence.  Sequence
      length grows from 8 (AR steps) → 13.  For inference, replace the
      Python `for i in range(8)` AR loop with one `for i in range(13)`.
      Eliminates `_parallel_cap_head` and `CAP_START` / `N_CAPS`
      constants; eliminates the two-headed dispatch in `forward()`.
- [ ] **Drop scheduled sampling and AR finetune** *if and only if* a
      retrain from the trimmed code converges to a TF-best ≤ 0.30 %
      test NRMSE.  The closed-v5 `phys-best` checkpoint trick was
      added because TF-best diverged from AR-best by > 15 %; if the
      gated Id (B3) closes that gap, both knobs become inert.  Strip
      `forward_scheduled`, `_train_epoch_scheduled_mae`, and the
      `[N3] AR finetune` block in `train_transformer`.
- [ ] **Optional:** drop `nn.TransformerEncoderLayer` for a 4-layer
      encoder (down from 6) and `d_model=192` (down from 256), reducing
      the model from 5.15 M to ≈ 2.3 M parameters.  This is only
      justified if (a) S-DATA and S-LOSS already met the inverter VTC
      gate and (b) test-NRMSE stays ≤ 0.30 %.  Defer until those
      conditions hold; do not block Phase B on it.

**Why a smaller model now:**  v3's "no capacity headroom" finding was
under v3 data and v3 losses.  Once the training distribution is
uniform-grid (B1) and the loss penalises shape (B2), the model needs
less capacity to interpolate the saturation plateau.  Smaller is faster
and easier to debug.

**Gate S-ARCH-B:** any model trim must keep all §2 acceptance metrics
within their hard gates.  No accuracy-for-speed trade allowed unless
explicitly approved.

### B5. (Sprint S-SOLVER) — Simplify the Vds correction and tighten convergence

**Targets:** robustness, not accuracy directly.  The simplifications
were promised in the closed v5 plan §4.5 but never landed.

**Change** in `pycircuitsim/models/mosfet_directnet.py`:

- [ ] **Once B3 ships**, delete the one-sided `f_id` factor, the
      `gds_linear` product-rule term, and the hard sign clip in
      `_apply_vds_correction`.  Keep only the rail-restoring quadratic
      ramp for `|Vds| > VDD_train`.
- [ ] Replace the quadratic ramp with `softplus`-smoothed quadratic so
      the join at `|Vds| = VDD_train` is C∞ (currently C¹).  Minor; only
      matters for TSMC12/16 whose operating points sit at the boundary.
- [ ] Add a unit test
      `tests/test_nn_vds_correction_continuity.py` that asserts
      `id`, `gds`, `gm`, `gmb` are all continuous and differentiable
      across `|Vds| = VDD_train` (finite-difference derivative within
      `1e-6` of autograd derivative).

**Solver-side cleanups** in `pycircuitsim/solver.py` (no algorithm
change, just audit):

- [ ] Confirm RELTOL=1e-4, VNTOL=1e-7, GMIN=1e-12 are the active
      values (the code has multiple constants); pin them as named
      constants near the top of the file rather than in-line.
- [ ] Tighten the gds floor in `_MOSFETNNBase` to be device-aware:
      `max(|id| * lambda_min, 1e-12)` where `lambda_min = 0.3 V⁻¹` for
      ≥ 16nm techs and `0.5 V⁻¹` for ≤ 7nm techs (matches the
      BSIM-CMG range cited in the rule book).  Currently `0.5`
      everywhere; the looser value over-stabilises the inverter at
      large techs and may mask DC errors.

**Gate S-SOLVER:** all 67 BSIM-CMG regression tests pass and the v5
inverter battery PASSes within ±0.5 pp of the post-Phase-B numbers.
This is purely a refactor; no metric should move.

### B6. (Sprint S-FINETUNE) — REMOVED per §7 decision #3

The user-confirmed constraint is **universal-only checkpoints**.  B6's
per-tech inverter-trajectory finetune would have produced
`v5_universal_*_tsmc7.pt` style outputs and required a parser-level
tech→checkpoint dispatcher; that violates the universal-only rule.

If, after B1+B2+B3, any tech still fails the §2 gate, the failure ships
as a documented limitation in CLAUDE.md.  The
`PyCMG/scripts/generate_tsmc7_overlay.py` script and the `is_overlay`
LDS-bypass design are kept on the bench as future-work seeds, not as
v5-release work.

**Replacement universal-overlay idea (optional, deferred):** if B1+B2
underperform across multiple techs, an inverter-trajectory overlay can
still be added — but it must be **all-tech overlay** (4 techs × 21
variants × ~2k samples ≈ 170k extra rows, ~3 % of the dataset),
trained as part of B1's universal recipe with a global LDS-bypass
flag.  This stays universal.  Not in scope unless B1+B2 close < 50 %
of the residual gap.

---

## 5. Sprint sequencing and time budget

| Sprint    | Phase | Wall-clock | What ships if it passes its gate                |
|-----------|-------|------------|--------------------------------------------------|
| S-TRIM-A  | A     | 1 d code + 0.5 d retrain | A1+A2+A3 + A4 control checkpoints |
| S-TRIM-B  | A     | 0.5 d code | A5 (simulator trim) + A6 (CLI cleanup)           |
| S-DATA    | B     | 0.5 d data + 1 d code + 0.5 d retrain | B1 dataset + retrained models |
| S-LOSS    | B     | 0.5 d code + 0.5 d retrain | B2 slope loss + retrained models     |
| S-ARCH-A  | B     | 1 d code + 0.5 d retrain × 4 | B3 tanh gate + retrained models     |
| S-SOLVER  | B     | 0.5 d code | B5 inference + solver cleanup                    |
| S-ARCH-B  | B     | 0.5 d code + 0.5 d retrain | B4 transformer trim if accuracy holds |
| ~~S-FINETUNE~~ | ~~B~~ | — | **REMOVED** — universal-only constraint (§7 #3) |

Total ≤ 9 days wall-clock if every sprint runs sequentially; 6 days if
S-DATA and S-LOSS run in parallel (they touch different files) and
S-ARCH-A starts as soon as S-DATA's checkpoints converge.

Each sprint commits its own checkpoint with prefix
`v5{a,b,c,d,...}_{dn,ar}_universal_{nmos,pmos}` so a bisect can
attribute behaviour to one sprint.

---

## 6. Risks and mitigations

| Risk | Likelihood | Blast radius | Mitigation |
|------|:---------:|--------------|------------|
| The trim (Phase A) regresses a tech we did not measure | Low | One sprint of debugging | A4 control retrain *requires* every cell within 1 pp of baseline before merge.  If a cell regresses, restore the most recently dropped piece. |
| Uniform-grid sampler (B1) under-samples a region we do not yet know about | Med | New failure mode at v5 ship time | The grid + jitter is strictly more uniform than LHS in the high-current corner.  Re-run D1 on every tech as part of S-DATA's gate; if any tech shows a *new* hot region, add it to B1's hot-region list. |
| Slope loss (B2) destabilises training when paired with asinh | Low | One retrain | Compute the slope penalty in normalised space; cap weight at 0.05 (we already learned 0.1 is too aggressive from N4); active only in the last 30 % of epochs.  Drop if test NRMSE regresses > 0.05 pp. |
| Tanh gate (B3) cascades into AR cap conditioning | Med | One sprint of debugging | Dual-head pattern: AR sees `id_raw`, simulator and loss see `id_gated`.  This is the same R1 mitigation as the closed v5 plan.  Add a unit test that AR step-2 (qb head) sees `id_raw`. |
| Transformer trim (B4) loses an unappreciated capability | Med | Re-add the dropped feature | B4 is opt-in: only ship if all metrics held under the smaller architecture.  Default-off if any cell regresses. |
| The v5 release ships with TSMC7 NMOS DC unimproved | Med | TSMC7 inverter VTC stays in the 14-19 % caveat | Document explicitly; CLAUDE.md keeps the "Known v4 limitation" until v5 closes it. |
| Per-tech finetune (B6) breaks zero-shot generalisation | Med | Need separate checkpoint per tech | B6 ships per-tech checkpoints with a parser-level tech→checkpoint mapping (the closed v5 plan §4.6 already designed this — 40 LOC).  Universal checkpoint stays as fallback. |

---

## 7. Open decisions — RESOLVED 2026-04-24

User-confirmed constraints that this plan now treats as fixed:

1. **Phase A lands first as a standalone release.**  No Phase B sprint
   may start until A1–A6 + A4 control retrain are merged.  Phase A is
   not blocked on Phase B planning.
2. **ASAP7 is always excluded from training and metrics.**  Every
   `--exclude-techs asap7 --num-tech-codes 18` invocation stands;
   §2's hard gates apply only to the four TSMC techs.  v5 releases
   ship with no ASAP7 numbers and no ASAP7 checkpoints — users on
   ASAP7 stay on v4 with the documented warning.
3. **Universal-only checkpoints.**  Exactly one
   `v5_universal_{nmos,pmos}` pair (DirectNet) and one
   `v5_universal_{nmos,pmos}_best.phys.pt` pair (BSIMAR) ship at
   v5 release.  **B6 (per-tech finetune) is therefore deleted from
   this plan** — see §4 below.  If a tech misses the §2 gate after
   B1+B2+B3, the failure is documented as a known limitation, not
   patched with a per-tech checkpoint.
4. (Implicit) Data-regen budget approved.
5. (Implicit) GPU budget approved.

---

## 8. File-by-file change manifest (intent, not code)

| File | Phase | Action | Why |
|------|-------|--------|-----|
| `external_compact_models/bsimar/losses/direct_loss.py` | A1 | DELETE | DirectLoss + ChargeConsistencyLoss never instantiated |
| `external_compact_models/bsimar/losses/bni_mae.py` | A2 | Trim — delete `SignConsistencyLoss`, `BoundaryLoss` | Unjustified; no A/B benefit |
| `external_compact_models/bsimar/losses/__init__.py` | A1+A2 | Update exports | Match deletions |
| `external_compact_models/bsimar/training/trainer.py` | A1+A2+A3 | Trim — drop dead branches and 3-axis LDS to 1-axis | Per §3 |
| `external_compact_models/bsimar/cli/train.py` | A6 | Drop `--sign-weight`, `--boundary-weight`; clean help | Match A2 |
| `external_compact_models/bsimar/config.py` | A1 | Drop dead dataclass fields + legacy aliases | Per §3 |
| `pycircuitsim/models/mosfet_directnet.py` | A5+B5 | Delete `_eval_autograd4`; unify gds floor; **B5** simplify `_apply_vds_correction` after B3 ships | Per §3, §4 |
| `pycircuitsim/solver.py` | B5 | Audit only — pin RELTOL/VNTOL/GMIN constants | Per §4 B5 |
| `external_compact_models/bsimar/models/direct_net.py` | B3+B4 | Add gated Id head; trim if B4 fires | Per §4 B3, B4 |
| `external_compact_models/bsimar/models/transformer.py` | B3+B4 | Add gated Id head; drop parallel cap head if B4 fires | Per §4 B3, B4 |
| `external_compact_models/bsimar/losses/slope_loss.py` | B2 | NEW — `SlopeMatchLoss` in normalised space | Per §4 B2 |
| `external_compact_models/PyCMG/pycmg/nn_generate.py` | B1 | Replace LHS sampler with hybrid uniform-grid + jitter; tag `sample_class` | Per §4 B1 |
| `external_compact_models/PyCMG/scripts/generate_nn_data.py` | B1 | Add `--sampler {grid,lhs}` flag default `grid` | Per §4 B1 |
| `tests/diag_d1_pmos_dc_errors.py` | B1 | NEW — D1 variant for PMOS DC | Per §4 B1 |
| `tests/test_nn_id_gate.py` | B3 | NEW — unit test `Id_gated(Vds=0) = 0` | Per §4 B3 |
| `tests/test_nn_vds_correction_continuity.py` | B5 | NEW — finite-diff vs autograd at boundary | Per §4 B5 |
| `external_compact_models/bsimar/training/finetune.py` | A6 | Audit signature against trimmed `train_transformer` (universal-only, no per-tech use); leave intact for future-work | §7 #3 — universal-only |
| `tests/verify_bsimar_v4_inverter.py` | B-end | Rename to `_v5_` and update default checkpoint prefix | Naming hygiene |
| `CLAUDE.md` | release | Update rule #19/#20; remove "Known v4 limitation" if F1 closes | Documentation |
| `results/v5_release_report_*.md` | release | NEW — before/after table per §1, §2 | Documentation |

---

## 9. What this plan deliberately does **not** propose

- **No new normalisation mode.**  The asinh + zscore normaliser is
  fine.  We learned that twice (signed-log was retired, and the v5 N4
  postmortem confirmed the asinh chain rule is the trap to avoid, not
  the normalisation itself).
- **No widening of the voltage box.**  v3 settled this: more box +
  same model is strictly worse.  Phase B keeps `voltage_box_factor =
  2.0`.
- **No reintroduction of `SignConsistencyLoss` / `BoundaryLoss`.**
  Their job is done structurally by B3's tanh gate.  If B3 is
  rejected, we still do not reintroduce these — they failed
  empirically and there is no new evidence.
- **No `torch.compile` perf work.**  Until accuracy lands; kept as
  follow-up.
- **No charge-consistency loss in physical space.**  N4 dead end;
  documented.
- **No per-tech checkpoints by default.**  Only B6 (last-resort
  finetune) ships per-tech, and only for the tech that needs it.

---

## 10. Definition of Done

- All §2 hard gates met against the v4 baseline in §1.
- Phase A net code delta ≥ 600 LOC removed, no test regression.
- `tests/diag_d1_*` heatmaps regenerated for TSMC5/7/12/16 NMOS *and*
  PMOS; hot-box mean |rel err| ≤ 5 % everywhere or the residual
  documented in CLAUDE.md.
- `tests/verify_bsimar_v4_inverter.py` renamed to `_v5_` and runs
  green on `v5_universal_*` checkpoints across TSMC5/7/12/16.
- `results/v5_release_report_*.md` exists, with before/after tables and
  per-sprint contribution accounting (which lever moved which metric
  by how much).
- `CLAUDE.md` updated: rule list reflects the trimmed code; known
  limitations section reflects the post-v5 state.

---

## 11. Adversarial-review hooks (for the staff-engineer subagent)

The reviewer should challenge at least:

- **R1.** *Is the uniform-grid sampler actually uniform under the
  asinh-transformed loss?*  Asinh compresses high-|Id| samples; the
  loss may still be dominated by the same high-|Id| corner even with
  uniform Vgs samples.  If true, B1 is necessary but not sufficient
  and B2 (slope loss) becomes load-bearing rather than complementary.
- **R2.** *Does the dual-head Id gate (B3) interact with the LDS
  weights computed on the gated Id?*  The LDS bins on `id` after
  reorder; if `id_gated` is what gets binned, near-Vds=0 samples all
  collapse into one bin and dominate the inverse-density weight.
  Decide: bin on `id_raw`, train on `id_gated`.
- **R3.** *Is the slope loss (B2) computable on the BSIMAR Transformer
  without breaking AR teacher-forcing?*  The forward graph at the time
  of the AR loss already detaches step-`t` predictions from the y-side
  inputs.  Confirm `torch.autograd.grad(id_norm.sum(), x_v_norm)` is
  defined under TF and under AR; if not, restrict B2 to DirectNet.
- **R4.** *PMOS DC TSMC12/16 — is it the same hot-region story?*  The
  plan assumes yes.  Run D1 on TSMC12 PMOS first (read-only, ~2 hr) to
  confirm before committing to B1's hot-region list.  If the hot
  region is elsewhere (e.g., low-Vgs/mid-Vds), update the densification
  rule.
- **R5.** *Is the §B6 `is_overlay` LDS bypass implementable without a
  full trainer rewrite?*  The current LDS pipeline is a numpy
  pre-compute; bypassing per-row needs the trainer to read a per-row
  mask.  Estimate ~50 LOC; confirm before promising B6.

If any reviewer-flagged challenge is rated "valid, blocking," the
relevant sprint pauses and the plan's sprint table updates before any
code change.

---

**Next action:** await user approval on §7 open decisions; on
approval, kick off Sprint S-TRIM-A in branch `feat/bsimar-v5`.
