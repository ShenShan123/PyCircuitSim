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
