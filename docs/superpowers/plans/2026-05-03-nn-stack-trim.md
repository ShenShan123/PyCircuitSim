# Plan — Trim NN compact-model stack for simplicity

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans`.
> Each step is checkbox-tracked (`- [ ]`).

**Date:** 2026-05-03
**Branch target:** new branch off `feat/bsimar-v5-phase-a` (suggested
`chore/nn-stack-trim`)
**Status:** PLAN — DO NOT EXECUTE without explicit user approval.
**Severity:** Medium — pure simplification; no shipped checkpoint behavior
changes when executed in the order below.

---

## 0. One-paragraph summary

The NN compact-model stack accumulated ~5500 LOC of marginal-value or
dead code across three sprints (v3 medium-tier, v4 tech-code migration,
v5 Phase A/B). Three parallel review agents triaged
`external_compact_models/bsimar/`, `pycircuitsim/models/`, and `tests/`
and produced a deletion list. This plan groups the deletions into two
mechanical PRs (tests cleanup, bsimar Phase B rollback + inference
dedup) and one experimental PR (phys-best A/B). PR-1 and PR-2 are
behavior-preserving for currently shipping v4 checkpoints.

---

## 1. Motivation

* **Phase B B2 (`SlopeMatchLoss`) and B3 (`apply_id_gate`) are unvalidated.**
  No retrain has demonstrated they beat the v4 baseline. B3 has a known
  index-mismatch bug (see `2026-05-03-phys-best-tracker-bug.md` Bug A)
  that corrupted v5b/v5c TF runs.
* **Inference-time `_apply_vds_correction` has 4 stacked corrections.**
  Three are redundant once one of them (rail-restoring extrapolation)
  is in place — but only the rail-restoring step is demonstrably load-bearing.
* **`tests/` has 9 broken or one-shot diagnostic scripts.** Six import
  v3-era APIs that were deleted in the v4 migration and have not been
  ported.
* **Two parallel base classes** (`_MOSFETNNBase`, `_MOSFETBSIMARBase`)
  duplicate ~120 LOC of `__init__`, `_eval`, and denorm logic that
  could live in one parent + a column-index table.

The user's directive: trim what is dead or marginal, keep what is
load-bearing.

---

## 2. Triage (by deletion confidence)

### 2A. Safe-delete now — ~1000 LOC production + ~4500 LOC tests

**Tests (entirely broken or superseded):**

| File | Reason |
|---|---|
| `tests/diag_id_gate_index_mismatch.py` | One-shot reproducer for Bug A; fix documented; regression guarded by `test_nn_id_gate.py` |
| `tests/diag_phys_best_explosion.py` | One-shot reproducer for Bug B; superseded by `test_phys_score_robustness.py` |
| `tests/diag_probe_id_at_vds.py` | Sprint-era ad-hoc probe; covered by `diag_bsimar_kcl_landscape.py` |
| `tests/diag_tsmc7_nmos_coverage.py` | Superseded by `diag_d1_tsmc7_nmos_errors.py` |
| `tests/verify_nn_leave_one_out.py` | Imports deleted v3 symbols (`Normalizer`, `inv_signed_log`, `DirectLoss`) |
| `tests/verify_nn_universal.py` | v3 process-params API; replaced by `verify_nn_dc_tran.py` |
| `tests/verify_nn_universal_v2.py` | Same |
| `tests/verify_nn_multi_tech.py` | Same |
| `tests/verify_nn_tran.py` | Same; v4 successor is `verify_nn_tran_v4.py` |
| `tests/compare_nn_vs_pycmg.py` | v3 API; broken |
| `tests/verify_bsimar_v4_zeroshot.py` | Imports nonexistent `MOSFETDatasetV4`; ASAP7 zero-shot is parked |

**Training pipeline (`external_compact_models/bsimar/`):**

| Item | Files | LOC | Reason |
|---|---|---|---|
| `SlopeMatchLoss` (B2) + wiring | `losses/slope_loss.py`, `losses/__init__.py`, `training/trainer.py` slope-loss branches, `cli/train.py` `--slope-*` flags | ~360 | Never validated; off by default; needs B1-data that does not exist; SDP-kernel toggle is Blackwell-hostile |
| `apply_id_gate` (B3) + wiring | `models/id_gate.py`, trainer `id_gate=` keyword threading, `BSIMARNormStats.id_gate`, simulator-side loader fork in `parser.py` | ~280 | Has known index-mismatch bug (rule 21 / Bug A); equivalent enforcement already exists at inference (`_apply_vds_correction` step b) |
| `forward_scheduled` + AR-finetune phase | `transformer.py:298-347`, `trainer.py:329-378, 964-1023` | ~160 | 5/150 epochs at `ss_ratio=1.0` is a one-line if-statement, not a separate optimizer + loader + tracker + checkpoint variant |

**Inference (`pycircuitsim/models/`):**

| Item | Files | LOC | Reason |
|---|---|---|---|
| Duplicate denorm methods | `mosfet_bsimar.py:173-193` | ~30 | `_denorm_scalar`/`_denorm_derivative` re-implement the asinh path already in parent; subclass should reuse parent + a column-index table |
| gds-floor duplication | `mosfet_directnet.py:346-350, :508`, `mosfet_bsimar.py:303-309` | ~20 (across 4 sites) | Same `max(gds, |id|*0.5, 1e-12)` stamped 4 times; fold into `_floor_gds(id, gds)` helper |
| LEVEL=73 vs LEVEL=74 NN-checkpoint resolver | `parser.py:587-628` (LEVEL=73), `parser.py:632-714` (LEVEL=74) | ~60 | Both blocks: import `CHECKPOINT_DIR`, resolve path with cascade, call `tech_variant_to_code`, emit UNKNOWN warning, build kwargs. Extract `_resolve_nn_checkpoint(level, device_key, tech_key, model_params)` returning `(path, tech_code)` |

### 2B. Delete-after-experiment — ~250 LOC

| Item | Files | LOC | Why deferred |
|---|---|---|---|
| Phys-best checkpoint tracker | `training/trainer.py:840-912, 1014-1023` + `BSIMARNormStats.phys_best_metric` flag + simulator loader fork | ~100 | Two prior bugs (median/mean, id-gate index). A/B one DirectNet run with phys-best vs val-best; if Δ < 5 % NRMSE, delete. Keeps the per-tech print helper. |
| `_apply_vds_correction` steps (b) one-sided f_id and (d) sign clip | `mosfet_directnet.py:484-522` | ~50 | Plan B5 deletion target. Currently the **only** enforcement of Id(Vds=0)=0 — must not delete until v5b structural gate ships *and* passes accuracy gates. Already gated on `self._id_gate_active`. |
| Two parallel base-class collapse | `_MOSFETBSIMARBase` → `_MOSFETNNBase(model_factory, column_indices)` | ~120 | Mechanical refactor; no behavior change. Defer to a separate PR after PR-1/PR-2 land so review surface stays small. |

### 2C. Keep (load-bearing)

* Rail-restoring quadratic ramp `_apply_vds_correction` step (a) — rule 20.
* Per-target LDS weights (`compute_lds_weights_per_target`) — last surviving LDS axis after Phase A.
* `qs = −(qg+qd+qb)` charge-conservation override — rule 17.
* `parallel_caps` + `grouped_inputs` Transformer head — structural, no dead branches.
* Softplus voltage clamp — rule 4.
* asinh normaliser mode — Transformer needs the 14-decade dynamic range.
* `MAELoss`, `DirectNet` MLP — already minimal.
* NMOS / PMOS subclass split — 5 lines each; replacing with a `polarity` ctor arg only obscures the sign convention rule.

---

## 3. Execution sequence (DO NOT START WITHOUT APPROVAL)

Three PRs, ordered to keep each review surface small and each rollback
trivial.

### PR-1 — Tests cleanup (zero production-code risk)

- [ ] Verify each candidate file in §2A (tests block) cannot be
      imported in the current `pycircuitsim` env (i.e. confirm "broken").
- [ ] `git rm` the 11 files listed in §2A tests block.
- [ ] Run `python tests/verify_bsimcmg_op.py && python
      tests/verify_bsimcmg_dc.py && python tests/verify_bsimcmg_tran.py`
      and the v4 NN suite (`verify_nn_dc.py`, `verify_nn_tran_v4.py`)
      to confirm zero regression.
- [ ] Update CLAUDE.md "Testing & Verification" table if any deleted
      script appears there.
- [ ] Commit: `chore(tests): remove broken v3-era and one-shot diagnostic scripts`

### PR-2 — Phase B rollback + inference dedup (no shipped-checkpoint behavior change)

Order matters: training-side first (so simulator changes can verify
checkpoints still load), then simulator-side.

- [ ] **Training-side:**
  - [ ] Delete `external_compact_models/bsimar/losses/slope_loss.py`.
  - [ ] Delete `external_compact_models/bsimar/models/id_gate.py`.
  - [ ] Remove `slope_*` and `id_gate` imports/exports from
        `bsimar/losses/__init__.py` and `bsimar/models/__init__.py`.
  - [ ] In `bsimar/training/trainer.py`: delete `_train_epoch_scheduled_mae`,
        the AR-finetune block at `:964-1023`, all `slope_loss` and
        `id_gate` keyword threading.
  - [ ] In `bsimar/models/transformer.py`: delete `forward_scheduled`
        method.
  - [ ] In `bsimar/cli/train.py`: delete `--slope-weight`,
        `--slope-warmup-frac`, `--no-id-gate` flags.
  - [ ] In `bsimar/data/normalize.py`: delete `BSIMARNormStats.id_gate`
        field (keep `phys_best_metric` until §3 PR-3).
  - [ ] Run a 5-epoch DirectNet training smoke test on the existing
        v4 dataset to confirm the trimmed pipeline still trains.
- [ ] **Simulator-side (`pycircuitsim/`):**
  - [ ] In `parser.py`: extract `_resolve_nn_checkpoint(level,
        device_key, tech_key, model_params)`; collapse LEVEL=73 and
        LEVEL=74 branches.
  - [ ] In `parser.py`: delete the `BSIMARNormStats.id_gate`-aware
        loader fork (the field no longer exists).
  - [ ] In `pycircuitsim/models/mosfet_bsimar.py`: delete
        `_denorm_scalar` and `_denorm_derivative`; pre-compute column
        indices in `__init__` and reuse parent.
  - [ ] Add `_floor_gds(id, gds)` helper to `_MOSFETNNBase`; replace
        the 4 stamp sites in `mosfet_directnet.py` and
        `mosfet_bsimar.py`.
  - [ ] Run the L1 sanity check (`verify_bsimcmg_op` + `verify_bsimcmg_dc`
        + `verify_bsimcmg_tran` + `verify_nn_dc.py` +
        `verify_nn_tran_v4.py`) — all must PASS unchanged.
  - [ ] Run `verify_bsimar_v4_inverter.py` across all 4 TSMC techs;
        NRMSE deltas must be ≤ 0.1 pp from main.
- [ ] Update CLAUDE.md: remove rules 21, B2, B3 from Critical Design
      Rules; remove Phase B references from Status / Future Work.
- [ ] Commit: `refactor(bsimar): remove unvalidated Phase B levers and dedupe inference glue`

### PR-3 — Phys-best A/B + base-class collapse (experimental)

- [ ] Run one DirectNet NMOS training with phys-best tracker disabled
      (use `_best.pt` directly). Compare TSMC{5,7,12,16} NMOS DC NRMSE
      vs main `_best.phys.pt`.
- [ ] If Δ < 5 % NRMSE on all 4 techs: delete the phys-best tracker,
      `phys_best_metric` flag, simulator loader fork, and `_best.phys.pt`
      file naming. Otherwise: keep, document why.
- [ ] Refactor `_MOSFETBSIMARBase` to inherit from
      `_MOSFETNNBase(model_factory=..., column_indices=...)`. Delete
      duplicated `__init__` and `_eval` overrides. Verify L1 sanity.
- [ ] Commit: `refactor(nn): collapse parallel base classes` (and, if
      A/B passed, a separate `chore(bsimar): drop phys-best tracker`).

---

## 4. Rollback

* PR-1: `git revert` the single commit. No code dependencies.
* PR-2: `git revert` the single commit. v4 checkpoints continue to
  load (the deleted Phase B fields were optional). v5b checkpoints
  (which depend on the structural gate) are already discard-only per
  Bug A — no loss.
* PR-3: `git revert` the relevant commit. Phys-best tracker is
  isolated; base-class collapse is mechanical.

---

## 5. Out of scope

* Re-running B1 hybrid-grid data generation.
* Retraining v5b checkpoints with a fixed `apply_id_gate` (decision is
  to delete, not fix).
* SRAM Phase 4, adaptive output timestep, KV-cache encoder — unrelated
  to the trim goal.
* `_eval_cache` two-tier hit-rate audit — flagged but out of scope.

---

## 6. Reference: source-of-triage

* `external_compact_models/bsimar/` reviewed by ce-code-simplicity-reviewer
  agent — full report archived in the originating session.
* `pycircuitsim/models/` and `parser.py` reviewed by a second
  ce-code-simplicity-reviewer agent.
* `tests/` reviewed by a third ce-code-simplicity-reviewer agent.
* CLAUDE.md "v5 Phase B" + Critical Design Rules 19–21.
* `docs/superpowers/plans/2026-05-03-phys-best-tracker-bug.md` (Bug A
  is the load-bearing reason to delete `apply_id_gate` rather than fix
  it).
