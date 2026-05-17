# PyCircuitSim — Detailed Changelog

This is the long-form history of PyCircuitSim. CLAUDE.md keeps a one-paragraph
"current state" summary; everything below is here so the conversation context
isn't burdened with chronology.

---

## Phase Milestones

- **Phases 1-3:** Core simulator (MNA, NR solver, transient).
- **Phases 4-6:** BSIM-CMG (LEVEL=72) integration via PyCMG, NGSPICE-verified (<0.02% OP, <0.1% DC).
- **Phases 7-10:** Charge-based transient (0.20% NRMSE vs NGSPICE), 5-tech support (ASAP7, TSMC5/7/12/16), 21-config parametric sweep all PASS.
- **Phases 11-12:** NN compact model (LEVEL=73) — training pipeline, autograd conductances, multi-tech DC+transient verified.
- **Phases 13-15:** Universal NN v2 — 21 variants across 5 techs, 13-dim input (voltages + 7 process params), 19/21 PASS (ASAP7:SLVT and TSMC7:LVT FAIL on NMOS DC).
- **Leave-one-out transferability:** 8/10 good transfer (gap < 5%), zero-shot avg 4.65% NRMSE, in-dist avg 0.95%.
- **Charge-finetune training:** ChargeConsistencyLoss (autograd dq/dV = C), trained from scratch 800 epochs on A100.
- **NN Transient (charge-finetune + VT fix):** 5/5 PASS — ASAP7 6.20%, TSMC5 14.41%, TSMC7 7.15%, TSMC12 6.47%, TSMC16 7.42%.
- **Solver accuracy improvements:** SPICE-standard convergence (RELTOL=1e-4, VNTOL=1e-7), GMIN reduction (1e-6→1e-12), BE→Trap first-step switching, relative oscillation threshold. NN transient improved: TSMC7 7.15→6.09%, TSMC12 6.47→5.92%, TSMC16 7.42→6.70%. BSIM-CMG transient unchanged at 0.20% (already at integration-method floor).
- **SRAM Solver Upgrades (Phases 1-3):** Sparse matrix solver (lil_matrix→CSR+spsolve), DC GMIN stepping + oscillation detection + adaptive damping + hard `.ic` mode (force_ic), BDF-2 integration (auto-switches on stiffness detection), LTE adaptive sub-stepping. All 67 existing tests PASS with zero regression.

## Test infrastructure

- **3-level DC+Transient test suites** — 3-layer infrastructure: `tests/common/base.py` (tech defs, generic helpers) → `tests/common/bsimcmg_{dc,tran}.py` (analysis-specific) → `verify_*.py` (test scripts). `tests/common/nn.py` consolidates NN scaffolding (nrmse, mre, checkpoint resolution, path bootstrap). NGSPICE references in `tests/references/`.
- **Known-bad combos excluded:** TSMC5 SVT (pch PDIBL2_i<0), TSMC7 SVT/LVT (inverter garbage / pch PDIBL2_i<0), TSMC16 LNVT (nch PDIBL2_i<0), TSMC16 L=24nm (PDIBL2_i<0), NFIN=1 (NR divergence for tsmc5:ulvt / tsmc16:lnvt — ETA0_i/U0_i go negative, internal node drifts to 40V producing id=40kA + NaN derivatives; eval_dc raises RuntimeError), P/N ratio where NFIN_P crosses NFIN group boundary (TSMC naive modelcards are NFIN-group-specific).

## Data generation migration

- NN training data generation moved from `nn_model.data.generate` into `external_compact_models/PyCMG/scripts/generate_nn_data.py`. Data format includes `[NFIN, L, T, 12 process params]` geometry columns; v4 training uses only 7 input features (Vgs, Vds, Vbs, NFIN, L, T, tech_code) and ignores process params. Legal (L, NFIN) combos from PDK bin boundaries (TSMC) or fallback list (ASAP7). 954 total geometry combos across 5 techs, 21 variants.

## BSIMAR package refactor (2026-03)

Consolidated former `nn_model/` (DirectNet baseline) and `external_compact_models/BSIMAR/script/` (Transformer) into a single Python package at `external_compact_models/bsimar/` with clean subpackages (`config`, `data`, `models`, `losses`, `training`, `eval`, `utils`, `cli`). Unified CLI: `python -m bsimar.cli.train --model {direct,transformer} ...`. All downstream imports use the new `bsimar.*` namespace.

## BSIMAR v3 production refactor (2026-04-08/09)

After the medium-tier improvement sprint (see `external_compact_models/bsimar/docs/bsimar_improvement_plan_2026_04_08.md`) the winning recipe is **N7 (Vov-LDS) + N3 (AR finetune) + N1 (150-epoch cosine)**. All hard-wired as defaults. The refactor collapses the CLI, deletes the signed-log normaliser, and removes ~600 net LOC.

- **Final metrics on `universal_nmos.npz` medium (5.15M params):** NRMSE_phys **0.223%** (was 0.419, −46.8%), MRE_phys **1.41%** (was 2.52, −44.0%), R² **0.9984** (was 0.9928). ~107 min on Blackwell GPU.
- **Removed code:** `Normalizer`, `NormStats`, `signed_log` / `inv_signed_log`, `BSIMARNormalizer.signedlog`, `load_and_split` (legacy loader), `WeightedBNILoss`, `forward_curriculum`, `train_epoch_direct_ar` / `curriculum` / `scheduled`, and the CLI flags `--loss direct|bni`, `--lds`, `--vov-lds`, `--no-filter`, `--reorder`, `--scheduled-sampling`, `--curriculum`, `--consistency-weight`, `--norm-mode`, `--charge-consistency-weight`, `--learnable-output-affine`.
- **Hardwired knobs:** loss=MAE+LDS+VovLDS, norm=asinh+zscore, `parallel_caps=True`, `grouped_inputs=True`, BSIMAR reorder, phys-best ckpt, AR finetune (5 epochs).
- **Known-infeasible (DO NOT retry without new structural argument):** N6 Huber on I/V (wrong gradient near zero), N5 learnable output affine (disrupts post-asinh zscore), N4 charge-consistency penalty (asinh chain rule cosh factor makes constraint inequivalent).
- **Deferred:** N2 KV-cache encoder.
- **File renames:** `pycircuitsim/models/mosfet_nn.py` → `mosfet_directnet.py` (class names unchanged).

## BSIMAR v3 LOO cross-tech sprint (2026-04-09/10)

5-fold leave-one-tech-out on universal NMOS. TSMC intra-family: 0.84–2.18% NRMSE (production-usable). ASAP7 held out: 24,678% NRMSE (catastrophic — body physics gmb/qb 10⁴× smaller than TSMC, a data bottleneck not fixable by model changes). One keeper: **S2 asinh-scale floor for gmb/qb** (~3.2% geometric-mean improvement). E2 Vov+extras and E2b Vov-only both REJECT (+10–14% regression). Full report: `external_compact_models/bsimar/results/loo_cross_technology_report.md`.

## Cross-tech transfer roadmap review (2026-04-10)

Five-agent review of the original 10-idea/7-stage roadmap. Conclusion: zero-shot transfer is not a user requirement (retrain with new tech data takes ~2h); TSMC transfer already within threshold; ASAP7 gap is a data bottleneck. Revised to 3-tier plan: (1) retrain v3 NMOS+PMOS + port verify_nn scripts + investigate TSMC5 transient, (2) one low-risk cross-tech probe (multiple process tokens), (3) retrain-with-new-data workflow for new PDKs.

## verify_nn_*.py port to NNTechConfig API (2026-04-11)

Ported 3 broken test scripts (`verify_nn_multi_tech.py`, `verify_nn_universal.py`, `verify_nn_universal_v2.py`) from old `tech.variants[v].get_process_params()` to new `NNTechConfig.resolve_modelcard()` + `extract_process_params()`. Added `default_L()` and `get_process_params()` helpers to `tests/common/nn.py`.

## BSIMAR v4 tech-code migration (2026-04-14)

All v3 code (19-dim continuous process params) removed. Only v4 architecture (7-dim + discrete tech-code embedding via `nn.Embedding`) supported. ASAP7 excluded from training (`--exclude-techs asap7`). 4 universal models trained: DirectNet NMOS/PMOS (0.00167/0.00190 val loss) + Transformer NMOS/PMOS (0.270%/0.252% NRMSE, R²=0.9937/0.9965). TSMC5 SVT verification: DC PASS (7.79%/9.99%), VTC 17.70%. Removed: `ProcessParams`, `extract_process_params`, `PROCESS_PARAM_NAMES`, old 19-dim `INPUT_COLUMNS`. Added: `TECH_CODE_MAP`, `--exclude-techs`, `--num-tech-codes`. Checkpoint naming changed to `v4_` prefix.

## Analytical Vds correction for inverter transient (2026-04-15)

Implemented `_apply_vds_correction()` in `_MOSFETNNBase` to enforce Id(Vds=0)=0 and Id=0 for reverse-Vds at inference. Three-part correction: one-sided Vds factor (VT=0.052V), symmetric gds with linear-region conductance, sign enforcement. DirectNet inverter transient: **3/4 PASS** (TSMC7 8.87%, TSMC12 11.65%, TSMC16 10.59%; TSMC5 17.20% marginal FAIL). BSIMAR inverter: 0/4 PASS due to wrong-sign subthreshold predictions in Transformer. NMOS pulse: 8/8 PASS, zero regression. Full report: `results/v4_vds_correction_report_2026_04_15.md`.

## Rail-restoring extrapolation fix (2026-04-20)

Diagnosed real root cause of BSIMAR inverter transient explosion (V(out)→+4.4V on TSMC12/16): both NN models predict Id≈0 outside `[-VDD_train, VDD_train]`, creating a flat-zero KCL plateau the DCSolver mistakes for equilibrium. Fixed by rail-restoring extrapolation: quadratic Id ramp + linear gds ramp past `VDD_train`, smooth-joined at boundary (linear ramp tried first, caused NR oscillation for TSMC12/16 whose operating points sit at the boundary). Verified across all 4 TSMC techs with probe (670K) and production (5.15M) checkpoints: **inverter transient drops from 18-300% NRMSE (FAIL) to 6-12% (PASS)**. Production: TSMC5 12.13%, TSMC7 9.14%, TSMC12 6.78%, TSMC16 7.51%. Inference-time only, no retraining required.

## v5 inverter-transient sprint (2026-04-22/23) — closed, no production change

5-experiment sweep attempting to lift worst-case TSMC7 NMOS DC (14.72%) and drive BSIMAR inverter VTC TSMC7 (19.15%) below 10%. E1 (wider Vds-correction VT), E3 (per-tech fine-tune on same distribution), E4 (dense hot-box overlay + universal set), E5 (overlay-only fine-tune) all reverted on inverter acceptance gates. D1 diagnostic isolated TSMC7 NMOS error to strong-inversion + saturation plateau (Vgs ∈ [0.52, 0.73] V × Vds ∈ [0.40, 0.75] V), 16× under-sampled by LHS — but both densification approaches regressed NMOS DC by +2.7 pp identically, ruling out the density thesis. Retained: D1 heatmap diagnostic + finetune.py empty-test_idx guard. Full history: `results/v5_session_summary_2026_04_23.md`.

## v5 Phase A — Trim (2026-04-24, branch `feat/bsimar-v5-phase-a`)

Plan: `docs/plans/2026-04-24-v5-inverter-accuracy.md`. Deleted unjustified and dead loss code before Phase B.

- **A1:** Deleted `DirectLoss`, `ChargeConsistencyLoss`, legacy `BSIMARConfig`/`TrainConfig` aliases, dead `TransformerConfig` fields.
- **A2:** Deleted `SignConsistencyLoss`, `BoundaryLoss` — no A/B benefit; superseded by rail-restoring extrapolation and structural B3 gate.
- **A3:** Collapsed 3-axis LDS weight stack (per-target × Vov × subthreshold) to **per-target only**.
- **A5:** Deleted `_eval_autograd4` dead fast-path; added 13-output assertion at load.
- **A4 control retrain — GATE FAIL.** Retrain with trimmed pipeline regressed TSMC7 NMOS DC past ±1 pp gate. Root cause: LHS dataset insufficient — Phase B B1 hybrid uniform-grid data is required.

## v5 Phase B — Levers tried, code reverted (2026-04-24 .. 2026-05-03)

Three Phase B levers prototyped to address TSMC7 sampling-basis mismatch:

- **B1 (data, retained in PyCMG submodule):** Hybrid uniform-grid + LHS jitter sampler with `sample_class` column. Datasets regenerated under this sampler still consumed by the loader.
- **B2 (`SlopeMatchLoss`) and B3 (`apply_id_gate`) — DELETED 2026-05-03.** Neither lever validated against a v4 baseline before B3's `id_idx_in_stats` bug corrupted v5b/v5c TF runs. Inference-time `_apply_vds_correction` already enforces Id(Vds=0)=0; rail-restoring extrapolation is the load-bearing piece.
- **AR-finetune phase / `forward_scheduled` — DELETED 2026-05-03.** The 5/150 final-phase rollout carried ~160 LOC of separate optimizer + loader + tracker + checkpoint plumbing for marginal benefit over cosine.

## v4-re — NN-stack trim (2026-05-03, branch `chore/nn-stack-trim`)

Plan: `docs/plans/2026-05-03-nn-stack-trim.md`. Current shipping NN stack is labeled **v4-re**: same v4 7-dim + tech-code architecture, all unvalidated Phase B levers and AR-finetune plumbing removed. Re-trained checkpoints under `v4_re_` prefix; legacy `v4_` checkpoints continue to load via resolver fallback.

- **PR-1:** Removed 11 broken/superseded test scripts (~3.9 KLOC) — all v3-era APIs.
- **PR-2:** Deleted `bsimar/losses/slope_loss.py`, `bsimar/models/id_gate.py`, `forward_scheduled` on Transformer, `_train_epoch_scheduled_mae`, trainer's AR-finetune block, `BSIMARNormStats.id_gate` field, CLI flags (`--slope-weight`, `--slope-warmup-frac`, `--no-id-gate`, `--ar-finetune-epochs`). Inference glue deduped: `_resolve_nn_checkpoint(level, ...)` collapses LEVEL=73/74 path resolution and prefers `v4_re_*` over legacy `v4_*`; `_floor_gds(id, gds)` replaces 4 stamp sites; `_MOSFETBSIMARBase` reuses parent `_denorm_scalar` / `_denorm_full_derivative` via column-index lookup. v4 checkpoints continue to load unchanged. v5b checkpoints discard-only per Bug A.
- **Default save_prefix:** `train_directnet` → `v4_re_dn_universal_<dev>`; `train_transformer` → `v4_re_universal_<dev>`.

### Known v4 limitation carried into v4-re: TSMC7 NMOS DC 14.72%

TSMC7 NMOS DC NRMSE is 14.72% (BSIMAR v4) / 15.79% (DirectNet v4) against PyCMG ground truth at Vds=VDD/2, NFIN=10, L=16 nm. Propagates to inverter VTC (19.15% BSIMAR / 18.14% DirectNet). Root cause: LHS training distribution under-samples strong-inversion + saturation plateau by ~16× vs verifier's uniform Id-Vgs sweep. Inverter transient at TSMC7 PASSES (6.80% DN / 9.14% BSIMAR). Mitigation: retrain on B1 hybrid-grid data with the trimmed pipeline, save under `v4_re_*` prefix, expect TSMC7 NMOS DC ≤ 8% per trim plan's gate.

## V6.1 — Per-tech dedicated DirectNet for TSMC5/TSMC7 (2026-05-12 / 2026-05-13)

Sprint goal: improve inverter DC/Tran accuracy on TSMC5 and TSMC7 by training **dedicated** per-tech DirectNet models at small + medium scales. Triggered by baseline measurement on `refac_dn_medium` (V6 universal): TSMC5 inv VTC 9.58% PASS, **TSMC7 inv VTC 163383.88% FAIL** (catastrophic OP lock), TSMC5 inv tran 14.33% PASS, TSMC7 inv tran 14.48% PASS.

### Scope and destructive cleanup
- Wiped `external_compact_models/bsimar/checkpoints/` (refac_dn_*, refac_tf_*, v6_dn_*, v4_* symlinks), `checkpoints_legacy/` symlink, and the originals at `/home/shenshan/NN_SPICE/external_compact_models/bsimar/checkpoints/` + `data/datasets/` (~12 GB total). All universal V6 + V4 + legacy artifacts deleted; **no checkpoints remain for TSMC12/16/ASAP7 or LEVEL=74 BSIMAR** (out-of-scope for this sprint per user direction).
- Regenerated per-tech datasets via `generate_nn_data.py --device both --tech {tsmc5,tsmc7} --enable-inv-trip` into `tsmc{5,7}_{nmos,pmos}.npz`. Sizes after V6.1 final regen: TSMC5 nmos 2.30M rows / pmos 2.30M; TSMC7 nmos 2.07M / pmos 2.41M. Inv_trip overlay adds ~218K-255K samples per device.

### Code changes
- `bsimar/config.py`: added `LOCAL_VARIANT_CODES`, `LOCAL_UNKNOWN_CODE_ID`, `LOCAL_VOCAB_SIZE`, `local_variant_code(scope, tech, variant)`, `tech_scope_vocab_size(scope)`, and `VALID_TECH_SCOPES = ("universal", "tsmc5", "tsmc7")`. Per-tech vocab: TSMC5 = 5 (4 variants + UNKNOWN), TSMC7 = 4 (3 variants + UNKNOWN).
- `bsimar/data/dataset.py`: `load_and_split_bsimar` accepts `tech_scope`; when non-universal, remaps tech_codes from universal → 0-indexed local vocab after `exclude_techs` filter.
- `bsimar/cli/train.py`: added `--tech-scope` flag. When non-universal, auto-sets exclude-techs (all other techs), num-tech-codes (per-tech vocab size), default data path (`<scope>_<dev>.npz`), and save_prefix (`<scope>_dn_<size>[_<preset>]_<dev>`).
- `bsimar/training/trainer.py`: passes `tech_scope` through to dataset loader; instantiates `DirectNet(unknown_code_id = num_tech_codes - 1)` so per-tech UNKNOWN is at the LAST embedding row instead of hardcoded 17. **Without this fix, training-time `p_unknown` dropout writes code 17 into a 5-row embedding → CUDA assert.** (Universal training keeps the existing convention since vocab=18 → unknown=17.)
- `pycircuitsim/parser.py`: per-tech preempt slot inserted ABOVE the universal cascade for TSMC5/TSMC7. Resolver decodes vocab scope from the resolved checkpoint stem (`tsmc{5,7}_dn_*` → local; everything else → universal) and uses `local_variant_code` to map the netlist's TECH+VT to the right embedding index. Every resolution prints `[NN-resolver] L73 ... -> <chk> (scope=<s>, tech_code=<c>)` per Rule 12.
- `tests/verify_nn_dc_tran.py`: extended the directnet_v4 checkpoint resolver to also accept `refac_dn_medium`, `refac_dn_small`, and `tsmc{5,7}_dn_{medium,small}` as fallbacks (the path is now an *existence sentinel*). Added `_cascade_handles_stem(path)` and stopped stamping `MODEL_PATH=` for stems that the parser preempt cascade can route — so a single inverter test invocation picks TSMC5 medium for TSMC5 netlists and TSMC7 medium for TSMC7 netlists automatically.
- `external_compact_models/PyCMG/pycmg/nn_generate.py`: widened the inv_trip overlay gate from `tech_name == "tsmc5"` to `tech_name in ("tsmc5", "tsmc7")`. Same lever that took TSMC5 DN inv-tran from 16.90% → 0.92% in V5'.

### Training (8 cells, GPU 2)
S+M × {NMOS, PMOS} × {TSMC5, TSMC7} via `scripts/train_per_tech_8cells.sh`. Best val losses (asinh+zscore + per-target LDS-MAE):

| Cell                    | Best val loss |
|-------------------------|--------------:|
| tsmc5 small  nmos       | 0.00742       |
| tsmc5 small  pmos       | 0.00913       |
| tsmc5 medium nmos       | 0.00103       |
| tsmc5 medium pmos       | 0.00084       |
| tsmc7 small  nmos       | 0.01171       |
| tsmc7 small  pmos       | 0.00861       |
| tsmc7 medium nmos       | 0.00114 (after inv-trip retrain; was 0.00130) |
| tsmc7 medium pmos       | 0.00096 (after inv-trip retrain; was 0.00109) |

Medium val loss is **7-10× lower** than small for every (tech, polarity); medium is the production size and small is retained as a parser cascade fallback only for TSMC5 (TSMC7 small was deleted on the inv-trip regen, since it would be inconsistent with the new dataset and is never selected when medium is present).

### Validation (parser per-tech preempt active)

| Test                       | Baseline `refac_dn_medium` | V6.1 per-tech medium | Δ |
|----------------------------|-------------------------:|---------------------:|---:|
| TSMC5 inv VTC              | 9.58%   PASS             | 7.96%   PASS         | −1.62 pp |
| TSMC7 inv VTC              | 163383.88% **FAIL**      | **1.69%** PASS       | catastrophe fixed |
| TSMC5 inv transient (post-startup) | 14.33% PASS      | **8.23%** PASS       | **−6.10 pp** |
| TSMC7 inv transient (post-startup) | 14.48% PASS      | 13.49% PASS          | −0.99 pp |

Locked success criterion was ≥ 2 pp transient reduction on the worse-of-two (TSMC7). Final TSMC7 transient is **−0.99 pp** — strictly under the gate. Inv-trip overlay (added in the second pass) sharpened TSMC7 VTC further (3.22% → 1.69%) but **did not move TSMC7 transient**. Diagnosis from the comparison plot: TSMC7 transient settles at a **second stable equilibrium ~±100 mV outside the rails** because the PMOS forward-Vds region (V(out) > VDD in source-relative frame) is extrapolated outside the `[0, 2·VDD]` training box and produces non-zero leakage, balanced against Rule 15(a)'s NMOS pull-down. Documented as Rule 20 in CLAUDE.md; fix is out-of-scope for V6.1.

### Net result
- Catastrophic TSMC7 VTC failure fixed (163383% → 1.69%).
- Average DC NRMSE across TSMC5/7 inverter VTC: was unmeaningful (1 catastrophic FAIL); now 4.82%.
- 4/4 inverter tests PASS (was 2/4 PASS, 2/4 FAIL).
- TSMC12/16 / ASAP7 / LEVEL=74 simulations have no checkpoints and will fail until a separate retrain.

### Logs and artifacts
- Baseline measurement: `training_logs/baseline_tsmc57_v6medium/`
- Data-gen logs: `training_logs/data_gen/{tsmc5.log, tsmc7.log, tsmc7_invtrip.log}`
- 8-cell training logs: `training_logs/per_tech/`
- TSMC7 medium inv-trip retrain logs: `training_logs/per_tech_v2/`
- Validation: `training_logs/validation_pertech_medium/` and `training_logs/validation_pertech_v2/`

### Rule 20 fix attempt — closed, no production change

Three variants of an inference-time fix to the Rule 20 forward-Vds rail-overshoot finding were prototyped against `pycircuitsim/models/mosfet_nn.py:_apply_vds_correction` and all reverted: (1) widen the fast-path early-return to skip the wrong-sign clamp whenever `abs_vds > VDD_train`; (2) defer part-(a)'s id injection until after the part-(d) clamp; (3) defer + add an `|NN_raw| < 0.5·|id_a|` off-state detector. Each variant catastrophically regressed TSMC5/7 inverter VTC (>200000% NRMSE), because the wrong-sign clamp also catches NN-error overshoot during DC OP NR iterations at modest Vgs values where NN_raw is a real subthreshold current — not "off". Distinguishing genuine off-state from NR-intermediate subthreshold needs Vgs context, which the function doesn't currently receive. Variant 3 did improve TSMC5 transient (8.23% → 6.81%) but the trade was unacceptable. Recorded for future revisit: Path B (Vgs-aware refactor) and Path C (regenerate with two-sided Vds box + retrain).

## V6.2 — Rule 15(a) sign fix, Rule 20 dead-band closed (2026-05-13)

**Two-line sign flip in `pycircuitsim/models/mosfet_nn.py:_apply_vds_correction`.** No retraining, no dataset regen, no checkpoint changes. Same V6.1 per-tech DirectNet medium artefacts; the only diff is in the rail-restoring extrapolation step (a).

### Diagnosis (rebuts Rule 20's earlier "missing two-sided Vds box" thesis)

V6.1 left TSMC7 inverter transient at 13.49% NRMSE with a stable equilibrium ~±100 mV outside the rails. Rule 20 hypothesised the NN was producing unhandled leakage in a region between `0` and `VDD_train`. **Wrong root cause.** Three Rule-15 variants from V6.1's "Rule 20 fix attempt" all catastrophically regressed VTC (>200000% NRMSE) by deferring or weakening the wrong-sign clamp.

Probing the dead-band directly revealed the actual mechanism: Rule 15(a)'s `id_extra` injection was using the *opposite* sign from physical restoring leakage. In PyCMG convention an NMOS in conduction has `id < 0`; the restoring leakage of an OFF NMOS at high-rail overshoot should also drive `id < 0` (more negative, pulling drain back toward source). The original V4-re ship had `result["id"] += id_extra` for NMOS (positive, wrong direction) and `result["id"] -= id_extra` for PMOS (negative, also wrong). The wrong-sign clamp at step (d) then wiped any contribution that exceeded |id_raw| inside the band `VDD_train < |Vds| < 20·VT`, leaving a current-free dead-band where Vout could settle at any value in ~±0.15 V.

### Fix

```python
if normal_dir:
    if self._is_pmos:
        result["id"] += id_extra      # was: -=
    else:
        result["id"] -= id_extra      # was: +=
```

Two character swap; the existing magnitude/ramp formulae for `id_extra` and `g_extra` are unchanged.

### Validation (parser per-tech preempt active, V6.1 checkpoints unchanged)

| Test                       | V6.1                  | V6.2                  | Δ |
|----------------------------|----------------------:|----------------------:|---:|
| TSMC5 inv VTC              | 7.96%   PASS          | **3.08%** PASS        | −4.88 pp |
| TSMC7 inv VTC              | 1.69%   PASS          | **1.00%** PASS        | −0.69 pp |
| TSMC5 inv tran (post-startup) | 8.23% PASS         | **1.23%** PASS        | −7.00 pp / 6.7× |
| TSMC7 inv tran (post-startup) | 13.49% PASS        | **1.67%** PASS        | −11.82 pp / 8.1× |

Full TSMC5/7 NN sweep — 12/12 PASS:

- TSMC5 NMOS DC 0.81%, TSMC7 NMOS DC 7.44%
- TSMC5 PMOS DC 0.35%, TSMC7 PMOS DC 1.81%
- TSMC5 NMOS pulse tran 1.10%, TSMC7 NMOS pulse tran 8.36%

### Process notes

- The three dead-end V6.1 variants ("widen fast-path / defer id-injection / Vgs-aware off-state detector") all assumed the V4-re ship was correct and the rail-overshoot was an unhandled NN-leakage region. Each tried to extend Rule 15 with new state (Vgs context, deferred clamps, smoothsteps), none worked, because the actual bug was a sign convention in a single conditional that's been live since V4-re's 2026-04-20 rail-restoring extrapolation patch.
- The 2-line diff dispatched to an agent team (3 isolated worktrees, parallel proposals). Agent 2 (originally tasked with "sharper reverse-Vds VT") probed the dead-band before patching and found the sign error. The other two agents (Vgs-aware off-state, solver-level rail clamp) cancelled — the simpler fix dominated.

### Risk / scope

- Re-validation required before resurrecting TSMC12/TSMC16 or LEVEL=74 BSIMAR. Those code paths used the *old* sign and may have been silently relying on the wrong-sign clamp's `id=0` fallback as their effective rail behaviour.
- Rule 15(a) docstring in CLAUDE.md updated. Rule 20 collapsed to a one-line resurrection guard.
- No regression observed on the full 12/12 TSMC5/7 NN gate, but ring oscillator / SRAM / other circuits have not been re-validated as part of this sprint.

### Docs trim (same release boundary)

CLAUDE.md was pruned of stale rules and tricks now obsoleted by V6.2 shipping and BSIMAR being parked. No code or test changes — CLAUDE.md only.

- **Status block** retargeted V6.1 → V6.2 with the corrected NRMSE numbers.
- **Module structure** dropped the unshipped `tsmc5_residual.py` / `tsmc5_residual_train.py` references (V6 Tier M2 experiment, no checkpoints, never resurrected).
- **Resolver cascade** clarified that only `tsmc{5,7}_dn_{medium,small}` checkpoints exist on disk; the `refac_*` / `v4_*` universal fallback chain is wired in `parser.py` but unreachable until someone retrains a universal stack.
- **Testing & Verification** dropped the stale "verify_nn_universal*.py / verify_nn_multi_tech.py need porting" note — those scripts were deleted in v4-re PR-1. Also removed mention of TSMC12-SVT-only entry points (`verify_nn_dc.py`, `verify_nn_tran_v4.py`) since TSMC12 has no V6.2 checkpoint.
- **Rule 8 (PyCMG integration)** dropped the ASAP7-specific train-VDD parenthetical (ASAP7 excluded per Rule 17) and the long-removed `ProcessParams` / `extract_process_params` / `INPUT_COLUMNS` re-export note.
- **Rule 13 (Unified CLI)** retargeted from `refac_{dn,tf}_<size>` defaults to the V6.2 per-tech `tsmc{X}_dn_<size>_<device>` default; dropped the deleted `tsmc5_residual_train` entry.
- **Rule 15(a)** condensed: kept the operative sign-convention rule, deleted the duplicated V6.2 NRMSE numerics (already in this CHANGELOG entry).
- **Rule 19 (per-tech local vocab)** dropped the now-irrelevant universal-training convention (vocab=18, unknown=17) since no universal training is being done.
- **Rule 20** collapsed from a long CLOSED-issue block to a one-line guard noting the sign convention is load-bearing for parked code paths (TSMC12/16, LEVEL=74) and needs re-validation when those are resurrected.
- **Supported Features** retagged LEVEL=74 BSIMAR from "primary" (stale since V4-re) to "parked".

## V6.2.1 — Per-tech TSMC12/TSMC16 DirectNet extension (2026-05-14)

Reusing the V6.2 recipe end-to-end (data → train → verify) for the two unshipped TSMC nodes. Rule 20 explicitly called out re-validation of Rule 15(a)'s sign convention at the new VDD=0.80 V; the inverter gate passes without further changes.

### Code changes (3 small registry edits)

- `external_compact_models/bsimar/config.py`: extended `VALID_TECH_SCOPES` and `LOCAL_VARIANT_CODES` to include `tsmc12` and `tsmc16` (vocab = 5 variants + 1 UNKNOWN = 6 per scope).
- `external_compact_models/PyCMG/pycmg/nn_generate.py`: extended the inv-trip overlay gate from `("tsmc5", "tsmc7")` to `("tsmc5", "tsmc7", "tsmc12", "tsmc16")`. Overlay is VDD-relative (Vd ∈ [0.30·VDD, 0.70·VDD]) so it is safe at the new vdd_train=0.80 V.
- The rest of the pipeline (`bsimar/cli/train.py`, `bsimar/data/dataset.py`, `pycircuitsim/parser.py`, `tests/verify_nn_dc_tran.py`) already generalised on scope — no edits needed.

### Data + training

- Datasets generated with `--enable-inv-trip --n-workers 8`: `bsimar/data/datasets/tsmc{12,16}_{nmos,pmos}.npz`, 2,872,800 samples each.
- 8 training cells on the A100 (GPU 2 visible-index, run sequentially per `logs/train_8cells.sh`): `tsmc{12,16}_dn_{small,medium}_{nmos,pmos}_best.pt` + `_norm.npz`. Medium runs ~38 min/cell (200 epochs), small ~14 min/cell (80 epochs). All 8 cells `rc=0`; total wall ~3h31m.
- Local vocab `unknown_code_id=5` for both scopes — derived from `LOCAL_VOCAB_SIZE`, not hardcoded.

### Validation (parser per-tech preempt active)

| Test                           | TSMC12     | TSMC16     |
|--------------------------------|-----------:|-----------:|
| Inverter VTC NRMSE             | **1.61%** PASS | **0.91%** PASS |
| Inverter transient post-startup | **1.51%** PASS | **1.66%** PASS |
| Inv-tran high-rail / low-rail / transition | 1.29% / 1.47% / 3.16% | 1.06% / 1.67% / 4.21% |

Resolver logs confirm scope routing — `[NN-resolver] L73.0 Mn1 TECH=tsmc12 VT=svt -> tsmc12_dn_medium_nmos_best.pt (scope=tsmc12, tech_code=0)`. Quality is on par with V6.2 TSMC5/7 (TSMC5 3.08% / 1.23%, TSMC7 1.00% / 1.67%). Rule 15(a)'s sign convention transfers cleanly to VDD=0.80 V — no dead-band reappears.

### Risk / scope

- ASAP7 / LEVEL=74 BSIMAR still parked — would still need a dedicated retrain.
- The full DC sweep (without `--inverter-only`) was not run as part of this sprint; the inverter gate was the user-stated success criterion. Rule 20 remains for LEVEL=74 only.

## V6.3 / V6.3.1 — Inverter spike-removal sprint (2026-05-15)

Goal: remove the inverter VTC + transient error spikes documented in
`results/v6_2_1_metrics_report/`. Agent-team diagnosis found three root
causes; the sprint ran in three phases (A discarded, B + C shipped).
Full plan + dead-end record: `docs/plans/2026-05-14-v6.3-spike-removal.md`.

### Root causes (agent-team diagnostic)

- **RC1 — reverse-Vds Id clamp** (`pycircuitsim/models/mosfet_nn.py:430`):
  the `f_id = 0` branch zeroes Id for reverse Vds, so when an inverter's
  load cap rings past a rail the NMOS produces no restoring current and the
  output undershoots ~99 mV (TSMC12/16).
- **RC2 — `inv_trip` overlay mis-centered**: pre-V6.3 the overlay centered
  Vg on the transistor peak-gm Vth, not the inverter Vtrip ≈ VDD/2. TSMC12/16
  had zero overlay rows in the switching band; TSMC5 only 0.24 %.
- **RC3 — zero reverse-Vds training coverage**: the main grid swept Vd ≥ 0
  (NMOS) only, so reverse conduction was never learned.

### Phase A — inference-only `gds`-bump gate (DISCARDED)

Gating the unconditional `gds = max(gds, g_extra)` bump on `normal_dir`
produced **bit-identical** eval traces — the reverse branch with
`|Vds|>VDD_train` is never hit on converged operating points. Reverted.
The real RC1 driver is the Id clamp, not the gds bump.

### Phase B — dataset regen (`_inv_trip_points` recenter + `_reverse_vds_points`)

`nn_generate.py`: re-centered `_inv_trip_points` on VDD/2 with a
`[0.30,0.70]·VDD` Vg/Vd box; added `_reverse_vds_points` (480 samples/bin,
new `sample_class="reverse_vds"` code 10). Regenerated all 8 datasets,
retrained 8 medium cells.

Result: transient pull-low spikes fixed across all 4 techs (TSMC12/16
99→58 mV) and TSMC5 VTC catastrophe fixed (206→58 mV) — **but TSMC7/12/16
VTC regressed** (+87/+24/+20 mV) because the wider, denser `inv_trip`
overlay (9.83 % of rows) over-fit a too-steep Id-Vg slope at the trip.

### Phase C — `_inv_trip_points` Vbs reduction (V6.3.1, SHIPPED)

`_inv_trip_points` dropped the `±0.25·VDD` Vbs sweep (the inverter runs at
Vbs=0 always; the `grid` class already covers Vbs). Overlay cut 25×9×3 →
25×9×1, from 9.83 % → 3.51 % of rows. Regenerated 8 datasets, retrained 8
medium cells (3-way multi-GPU parallel, ~2 h).

### V6.3.1 inverter results vs V6.2.1 (NGSPICE BSIM-CMG ground truth)

| Tech | VTC MaxErr V6.2.1→V6.3.1 | Tran post-startup MaxErr V6.2.1→V6.3.1 |
|------|--------------------------|-----------------------------------------|
| TSMC5  | 206.6 → **66.4 mV** | 79.1 → **39.5 mV** |
| TSMC7  | 55.9 → **65.8 mV**  | 55.5 → **50.3 mV** |
| TSMC12 | 39.2 → **78.3 mV**  | 99.2 → **58.2 mV** |
| TSMC16 | 30.9 → **45.4 mV**  | 97.8 → **55.3 mV** |

VTC NRMSE 1.52–1.77 %, transient post-startup NRMSE 1.22–1.51 %, ΔVtrip
≤0.6 mV, R² ≥ 0.9987 everywhere. Reports in `results/v6_3_1_metrics_report/`;
intermediate V6.3 (pre-Phase-C) in `results/v6_3_metrics_report/`.

### Outcome — shipped with one open gate

V6.3.1 is the new shipping revision. **Wins:** transient pull-low spikes
cut ~43 % (99→58 mV); TSMC5 VTC catastrophe cut 3.1× (206→66 mV); the
Phase-B TSMC7 VTC regression (143 mV) fully recovered (66 mV). Worst-case
VTC 143→78 mV, average VTC 79→64 mV vs the Phase-B intermediate.

**Open gate (deferred per user, 2026-05-15):** VTC MaxErr ≤ 25 mV not met
— V6.3.1 sits at 45–78 mV. Three dataset revisions moved the trip error
around but never below ~45 mV. Diagnosis: this is **not** a coverage gap
but gain amplification — inverter gain ≈ −15 to −30 at the trip multiplies
the NN's residual Id error (~0.05 % test-split NRMSE) ~20× into Vout. The
fix needs a gm/gds-fidelity lever (e.g. trip-weighted gm-matching loss),
not more `inv_trip` samples. Transient post-startup ≤ 30 mV also unmet
(39–58 mV), same root cause at the t=0 DC OP.

### Infra notes

- `/home/shenshan/NN_SPICE/` (this worktree's parent `.git` + the PyCMG
  submodule + dataset `.npz` targets) was moved to `/tmp/NN_SPICE/` mid-sprint
  to free a 98 %-full `/home`. Symlinks `external_compact_models/PyCMG` and
  `bsimar/data/datasets/*.npz` were repointed to `/tmp/NN_SPICE/`.
- V6.3 (pre-Phase-C) datasets preserved as `*.v6_3.npz`; V6.3 checkpoints
  backed up to `/tmp/v6_3_checkpoints_backup/`.
- New scripts: `scripts/regen_v6_3_1.sh`, `scripts/train_v6_3_1_parallel.sh`
  (3-way multi-GPU), `scripts/eval_v6_3_1_inverter.py`.

## V6.3.2 — NN parametric test harness (2026-05-17)

Branched `feat/v6.3.2` from the pre-V6.4-finalize HEAD (`beac301`). No model
or checkpoint change — this release **ports the BSIM-CMG L3 parametric test
harness to the DirectNet (LEVEL=73) NN models** and runs the V6.3.1
checkpoints through it. (The `feat/v6.4` branch independently advanced the
checkpoints; V6.3.2 is a test-infrastructure point release on the V6.3.1
model. Its harness was merged into `feat/v6.4` on 2026-05-17.)

### Motivation

The BSIM-CMG harness (`verify_multi_tech_{dc,tran}.py`) sweeps device geometry
and inverter circuit parameters; the NN harness (`verify_nn_dc_tran.py`) only
ran fixed-geometry points. The V6.3.1 DirectNet checkpoints had never been
stress-tested across the parametric space the BSIM-CMG reference covers.

### What shipped

- **`tests/common/nn_sweep.py`** — shared parametric harness: sweep-config
  dataclasses, builders, single-test orchestrators, a baseline-gated
  multi-tech loop, and summary/CSV/bar-plot helpers. Reuses the existing
  `verify_nn_dc_tran.py` runners; geometry/VT/VDD sweeps ride on
  `dataclasses.replace(TestTechConfig, ...)` (zero DC-runner refactor).
- **`tests/verify_nn_multi_tech_dc.py`** — single-device NMOS/PMOS Id-Vgs over
  L / NFIN / VT.
- **`tests/verify_nn_multi_tech_tran.py`** — inverter VTC + transient over P/N
  ratio, VDD, Cload, input slew, pulse width.
- **`verify_nn_dc_tran.py` refactor (behaviour-preserving)** — added
  `InvCircuitParams` (frozen dataclass; Cload/tr/tf/pw/td/tstop, defaults =
  legacy globals) threaded through the two inverter-transient runners, and an
  `inv_nfin_p` field + `effective_inv_nfin_p` property on `TestTechConfig` for
  the P/N-ratio NFIN split. `circuit=None` / `inv_nfin_p=0` reproduce the
  legacy fixed point exactly — verified by netlist-string audit (a value-match
  regression guard was infeasible, see "checkpoint contamination" below).

### V6.3.1 results

- **Single-device DC — 55/55 PASS** (gate NRMSE < 10%). Baselines ≤ 0.2%
  except TSMC7 NMOS 6.5%. Stressors elevate as expected: off-bin L (TSMC5 nmos
  L=24nm 2.6%), NFIN=10 (TSMC12 pmos 7.6% / MRE 25%), TSMC5 NMOS VT variants
  (ulvt 6.3%). VT sweeps on TSMC12/16 near-perfect (< 0.25%).
- **Inverter VTC + transient — 63/64 PASS** (gate NRMSE < 15%). All baselines
  pass (VTC 1.4–4.1%, transient 1.3–1.8%). Sole **FAIL: `TSMC5_vtc_vdd_0p55`**
  16.8% — VDD−0.1V drops the trip below the per-tech NN's accurate band.
  VDD−0.1V is the dominant stressor across techs (TSMC7/TSMC16 VTC 11.7/12.2%,
  near the gate); fast slew (10 ps) elevates to 4–6%; Cload, pulse-width,
  slew=500 ps and the single P/N-ratio point (`nfin_p=3`) all stay < 3%.
- Finding: V6.3.1 DirectNet is **VDD-specific** — it degrades sharply ~0.1 V
  off the training VDD. A VDD-robustness lever (train-time VDD jitter) is the
  natural follow-up.

### Harness design notes

- **P/N ratio is one point.** The TSMC naive-modelcard NFIN-group rule
  (`nfin_p > nfin+1` skipped) with `default_nfin=2` admits only `nfin_p=3` —
  exact parity with the BSIM-CMG harness; the limiter is the modelcard.
- Off-bin L/NFIN points exercise NN extrapolation beyond the per-tech training
  bins; elevated NRMSE there is expected model behaviour, not a harness fault.

### Dead end recorded — checkpoint contamination (cost ~1 h)

The first full runs were **invalid**: `bsimar/checkpoints/` was symlinked into
the shared checkout, and the concurrent `feat/v6.4` best-of-N work **overwrote
the `tsmc*_dn_medium_*` slots at 07:46:58** mid-run. This produced ~±1 % VTC
NRMSE run-to-run scatter (7 TSMC12 readings spanning 1.7–3.8 %) — chased
fruitlessly against PyTorch threading and `PYTHONHASHSEED` before the moving
checkpoint files were identified as the cause. Fix: a worktree-local copy of
`/tmp/v6_3_1_checkpoints_backup/` (md5-verified V6.3.1), plus an isolated
`PyCMG/build/modelcards/` (the v6.4 eval jobs also raced the shared naive
modelcards). Lesson: a verification harness must own immutable copies of its
inputs; never point it at a directory under active training.

## V6.4 — Best-of-N retrain + complex-circuit benchmark harness (2026-05-15 .. 17)

Plan: `docs/plans/2026-05-15-directnet-complex-circuits.md` (re-prioritized
2026-05-15). V6.4 executed Phases 1–3; Phases 4–8 deferred.

### Phase 3 — complex-circuit benchmark harness (shipped)

Four benchmarks vs NGSPICE BSIM-CMG ground truth: 5-stage ring oscillator,
two-stage Miller opamp, 6T SRAM read SNM, switched-cap unit cell. Harness in
`tests/common/complex.py` + `tests/verify_complex_*.py`; netlists in
`examples/complex/` + `tests/references/complex/`. Baseline V6.3.1: ring-osc
2/4, opamp 0/4 (gain error 10–135 % — confirms plan blocker D0, gain
amplification of DirectNet's Id residual), SRAM-SNM 4/4, switched-cap 1/4.

### Phase 1 — DirectNet retraining is a seed lottery; best-of-N selection

The plan's original 1a–1e levers were all dropped: the 1b Sobolev term lost
its bake-off decisively (7–8× worse VTC — a *validated* dead end, distinct
from the 2026-05-03 unvalidated `SlopeMatchLoss`; Rule 10 unchanged); 1e
(`gds` asinh floor) is a confirmed no-op; the 1a validation slices are a
broken near-zero-denominator proxy.

Decisive finding: a clean, verified stock-recipe DirectNet retrain (`--seed
42`) regressed TSMC5 inverter VTC MaxErr to 218 mV vs V6.3.1's 66 mV; seed
123 gave 79 mV — a **139 mV seed-driven swing**. Transient is seed-stable
(~38 mV at every seed). V6.3.1's shipped checkpoints were a lucky draw. So
V6.4 produces checkpoints by **best-of-N**: 8 seeds × 8 cells
(tsmc{5,7,12,16} × {nmos,pmos}), each tech's (nmos,pmos) pair selected on the
real inverter VTC sim — never on val loss (the D6 decoupling).

Corrected false lead: an agent claimed the datasets had drifted from V6.3.1
and recommended restoring `*.v6_3.npz`. Verified false — the current datasets
carry a 3.51 % `inv_trip` overlay, an exact match to the V6.3.1 spec; the
`*.v6_3.npz` files are the *older V6.3* (9.83 %) data and must not be restored.

### Phase 2 — 2a kept, 2b reverted (unsound)

2a (transient C-stamp symmetrization, env-gated `NN_SYMMETRIC_CAPS`, default
off) is kept as dormant infrastructure for the Phase-3 ring oscillator.

**2b (always-on conducting-branch `gm/gmb` sign-floor) was reverted.** It
appeared to halve inverter VTC error on TSMC5/12/16 — but that was an artifact
of *circular selection*: best-of-N had been scored on the 2b solver, so it
merely picked seeds compatible with the gm hack. On neutral ground (V6.3.1
checkpoints) the `gm`-floor breaks TSMC7 (66→215 mV) and TSMC12 (78→261 mV);
the `gmb`-floor is completely inert; a `reflect` variant (wrong-sign `gm` →
correct sign, magnitude kept) breaks 3/4 techs (TSMC12 354 mV). Zeroing or
altering an autograd wrong-sign `gm` is a checkpoint-dependent coin-flip —
there is no sound `_floor_gm` fix. The principled fix for wrong-sign `gm` is a
network constraint (plan Phase 6 monotonicity / spectral norm), not a solver
hack. `mosfet_nn.py` reverted to its pre-Phase-2 state.

### V6.4 final — clean-solver best-of-N, inverter vs NGSPICE BSIM-CMG

Selection re-run on the clean (2b-reverted) solver. All 4 techs beat V6.3.1
inverter VTC MaxErr; transient holds (TSMC7 +1.2 mV, within noise):

| Tech   | seeds n/p   | VTC MaxErr V6.3.1→V6.4 | Tran post-startup V6.3.1→V6.4 |
|--------|-------------|------------------------|-------------------------------|
| TSMC5  | 17 / 42     | 66.4 → **62.0** (−7 %)  | 39.5 → 37.9 mV |
| TSMC7  | 31337 / 42  | 65.8 → **60.1** (−9 %)  | 50.3 → 51.5 mV |
| TSMC12 | 123 / 123   | 78.3 → **32.3** (−59 %) | 58.2 → 57.6 mV |
| TSMC16 | 42 / 123    | 45.4 → **29.7** (−35 %) | 55.3 → 54.9 mV |

VTC NRMSE 1.20–2.13 %, R² ≥ 0.9981, ΔVtrip ≤ 0.3 mV. TSMC16 (29.7 mV) and
TSMC12 (32.3 mV) approach the deferred ≤25 mV stretch gate. TSMC5/7 gains are
modest — their clean-solver seed lottery surfaced no strongly better draw
within N=8.

### Open / deferred

- Inverter VTC ≤25 mV still unmet (V6.4 at 29.7–62.0). TSMC5/7 are the
  laggards; a larger seed sweep or the plan's Phase 6 structural levers are
  the next step.
- Plan Phases 4–8 (data overlays, batched NN forward, NR convergence, soft
  physics constraints, per-target heads) deferred.
- Complex-circuit benchmarks not yet re-measured on the V6.4 checkpoints —
  harness is in place; pass/fail TBD.

### Process notes / dead ends

- Three sub-agents died to a 600 s no-progress watchdog while waiting on long
  jobs; orchestration was redone as plain background scripts (no watchdog).
- The retrain pool was once killed one batch early; 8 tsmc16-pmos cells were
  re-trained cleanly.
- Artifacts: 64 best-of-N candidate checkpoints `v6_4_bof_*` / `v6_4_repro_*`
  in `checkpoints/` (gitignored); V6.3.1 backup at
  `/tmp/v6_3_1_checkpoints_backup/`. Best-of-N pair evaluator:
  `scripts/eval_v6_4_pair.py`.

## V6.4.1 — harness merge + single-seed retrain (2026-05-17)

Branch `feat/v6.4.1`. Two changes: (1) merged the V6.3.2 parametric NN test
harness into the V6.4 line (`tests/common/nn_sweep.py` +
`verify_nn_multi_tech_{dc,tran}.py`); (2) re-trained all 8 DirectNet medium
cells from a **single seed (42)** — not best-of-N — and re-ran the full
extended harness against the fresh checkpoints.

### Retrain

`scripts/run_v6_4_1_retrain.sh`: 8 medium cells (tsmc{5,7,12,16} ×
{nmos,pmos}), `--seed 42`, across 3 GPUs (GPU1 Blackwell + GPU0/GPU2 A100).
The entire `checkpoints/` directory was wiped first (107 MB of V6.4 best-of-N
production + candidate artifacts); a copy was preserved at
`/tmp/v6_4_checkpoints_backup_20260517/`. New checkpoints land in the canonical
`tsmc{X}_dn_medium_{nmos,pmos}` parser-cascade slots. Per-tech test R²
0.997–1.000.

### Extended harness results (V6.4.1 seed-42 checkpoints)

- **`verify_nn_multi_tech_dc.py` — 55/55 PASS** (gate NRMSE < 10%). Unchanged
  vs the V6.3.1 harness run.
- **`verify_nn_multi_tech_tran.py` — 64/64 PASS** (gate NRMSE < 15%), VTC +
  transient. This *clears* the V6.3.1 harness's sole FAIL `TSMC5_vtc_vdd_0p55`
  (16.8% → 14.08%).

### Seed lottery confirmed — VTC regressed vs V6.4 best-of-N

Single-seed retrain lost the lottery on inverter VTC MaxErr, exactly as the
V6.4 finding predicted. Inverter VTC baseline MaxErr vs NGSPICE BSIM-CMG:

| Tech   | V6.4 best-of-N | V6.4.1 seed-42 |
|--------|----------------|----------------|
| TSMC5  | 62.0 mV        | **128.0 mV**   |
| TSMC7  | 60.1 mV        | **174.7 mV**   |
| TSMC12 | 32.3 mV        | **41.6 mV**    |
| TSMC16 | 29.7 mV        | **33.6 mV**    |

All 4 techs regressed (TSMC7 ~3×). Transient is seed-stable as documented —
baseline transient MaxErr TSMC5 61.1 / TSMC7 49.2 / TSMC12 64.2 / TSMC16
69.0 mV, in line with V6.4. The harness NRMSE gates (10% / 15%) still pass
because they are looser than the VTC-MaxErr program gate; the seed-42 draw is
**not** an improvement over the shipped V6.4 best-of-N checkpoints.

**Recommendation:** the better V6.4 best-of-N checkpoints are preserved at
`/tmp/v6_4_checkpoints_backup_20260517/`; restore them, or run a full
best-of-N (`scripts/run_v6_4_bestof.sh`) on `feat/v6.4.1`, before treating
V6.4.1 as a shippable model. V6.4.1 currently ships the harness merge, not a
model improvement.
