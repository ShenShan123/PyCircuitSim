# Project: PyCircuitSim

## Overview
Python-based SPICE-like circuit simulator emphasizing educational clarity and modular architecture.
**Primary Goal:** specific support for three compact model families:
- **BSIM-CMG** (LEVEL=72) — PyCMG-wrapped OSDI FinFET model (ground truth).
- **DirectNet** (LEVEL=73) — baseline feed-forward MLP compact model (PyTorch).
- **BSIM-AR Transformer** (LEVEL=74) — autoregressive Transformer compact model (PyTorch).

DirectNet and BSIM-AR share the same data, normalization, and evaluation pipelines
via the unified `bsimar` package at `external_compact_models/bsimar/`. DirectNet
is used as the baseline for comparison against BSIM-AR.

The simulator must support **Operating Point (OP)**, **DC Sweep**, and **Transient Analysis** for all model types.

**Core Principles:**
* Pure Python with clean, readable code
* Complete decoupling: Solver <-> Device Models
* Production-grade compact model integration via PyCMG/OSDI
* Basic HSPICE netlist compatibility

## Architecture

### Module Structure
```
pycircuitsim/
├── __init__.py         # Package initialization, exports public API
├── config.py           # Path configuration (OSDI binary, modelcards)
├── simulation.py       # Simulation orchestration (run_simulation, run_dc_sweep, run_transient)
├── parser.py           # Two-pass netlist parsing, .model directive support
├── circuit.py          # Circuit topology management
├── solver.py           # MNA matrix construction, Newton-Raphson solver
├── logger.py           # HSPICE-like .lis output
├── visualizer.py       # Matplotlib plotting
└── models/
    ├── __init__.py
    ├── base.py               # Component abstract base class
    ├── passive.py            # R, C, V, I sources (including PULSE)
    ├── mosfet_cmg.py         # BSIM-CMG FinFET (LEVEL=72) via PyCMG — MOSFET_CMG + NMOS/PMOS
    ├── mosfet_directnet.py   # DirectNet v4 (LEVEL=73) via PyTorch — tech-code embedding, _MOSFETNNBase + NMOS_NN/PMOS_NN
    └── mosfet_bsimar.py      # BSIMAR v4 Transformer (LEVEL=74) — tech-code embedding, _MOSFETBSIMARBase + NMOS_BSIMAR/PMOS_BSIMAR

external_compact_models/
├── bsimar/             # Unified NN compact model package — importable as `bsimar`
│   ├── __init__.py
│   ├── config.py                   # NNTechConfig + DirectNetConfig + TransformerConfig + TECH_CODE_MAP
│   ├── data/
│   │   ├── normalize.py            # BSIMARNormalizer (asinh / zscore) + BSIMARNormStats
│   │   ├── dataset.py              # MOSFETDataset + load_and_split_bsimar + filter_small_targets
│   │   └── analyze.py              # Dataset analysis script (distribution, outliers, physical constraints)
│   ├── models/
│   │   ├── direct_net.py           # DirectNetV4 MLP with nn.Embedding tech-code (baseline)
│   │   └── transformer.py          # TransformerEncoderModel with nn.Embedding tech-code (parallel_caps + grouped_inputs)
│   ├── losses/
│   │   └── bni_mae.py              # MAELoss + compute_lds_weights_per_target (shared by DirectNet + BSIMAR)
│   ├── training/
│   │   ├── early_stopping.py
│   │   └── trainer.py              # train_directnet, train_transformer, per-epoch helpers
│   ├── eval/
│   │   ├── metrics.py              # compute_physical_metrics, print_metrics
│   │   └── visualization.py        # plot_scatter_comparison, plot_loss_curves
│   ├── utils/seed.py
│   ├── cli/train.py                # Unified CLI: `python -m bsimar.cli.train --model {direct,transformer} ...`
│   ├── checkpoints/                # Saved model weights (.pt + _norm.npz + _config.npz) — gitignored
│   │   └── pretrained/             # Legacy pretrained .pth files from the paper
│   ├── results/                    # Training plots — gitignored
│   ├── docs/                       # Reference paper and ablation notes
│   ├── imgs/                       # README imagery
│   ├── README.md
│   ├── LICENSE
│   └── requirments.txt
│
├── PyCMG/              # BSIM-CMG OSDI wrapper (git submodule)
│   ├── pycmg/          # Python OSDI interface (Model, Instance, tech registry)
│   │   ├── core.py     # Low-level OSDI: OsdiLibrary, OsdiModel, OsdiInstance
│   │   ├── model.py    # Public API: Model, Instance, eval_dc, eval_tran
│   │   ├── parser.py   # Modelcard parsing, SPICE number parsing
│   │   ├── osdi_types.py # OSDI constants, ctypes structures
│   │   └── tech.py     # TECH_REGISTRY, DeviceConfig, TechConfig, resolve_modelcard
│   ├── build/osdi/bsimcmg.osdi             # Compiled OSDI binary
│   └── modelcards/                          # Technology modelcards
│       ├── ASAP7/                           # ASAP7 7nm modelcards
│       └── TSMC{5,7,12,16}/                 # Raw TSMC PDKs (gitignored, IP-protected); naive modelcards regenerated on-the-fly via pycmg.tech.resolve_modelcard into build/modelcards/
main.py                 # CLI entry point (single main entrance)
examples/*.sp           # Example netlists
results/                # Simulation output (.lis, .csv, .png)
tests/
├── __init__.py
├── common/                          # Shared test infrastructure (subpackage)
│   ├── __init__.py
│   ├── base.py                      # PROJECT_ROOT, OSDI_PATH, TechProfile, VtPair, ALL_TECHS, NGSPICE runner, generic orchestration
│   ├── bsimcmg_dc.py                # DC-specific: DCTestConfig, runners, metrics, plots
│   ├── bsimcmg_tran.py               # Transient-specific: TestConfig, runners, metrics, plots
│   └── nn.py                        # NN-specific: nrmse, mre, directnet_checkpoint, transformer_checkpoint, path bootstrap
├── references/                      # NGSPICE reference netlists (ngspice_*.cir)
├── test_phys_score_robustness.py    # Phys-score median-vs-mean regression guard (plan §2B)
└── verify_*.py                      # 3-level test scripts (L1/L2/L3 for DC + transient, plus NN verification)
```

### Key Algorithms
* **MNA (Modified Nodal Analysis)** - Sparse matrix construction (scipy.sparse lil_matrix→CSR+spsolve)
* **Newton-Raphson** - Non-linear circuit solver with SPICE-standard convergence (RELTOL + VNTOL)
* **BE→Trap→BDF-2 Integration** - Backward Euler (step 1), Trapezoidal (default), BDF-2 (auto on stiffness)
* **Source Stepping + GMIN Stepping** - Homotopy methods for convergence; GMIN stepping opt-in for bistable circuits
* **LTE Sub-stepping** - Local truncation error estimation with adaptive internal sub-steps (opt-in via `max_substeps`)
* **Bistable Convergence** - DC oscillation detection, adaptive damping, hard `.ic` mode (`force_ic`)

## Supported Features

### Devices
* Passive: R, C
* Active:
  - NMOS/PMOS Level 72 (BSIM-CMG FinFET via PyCMG/OSDI — ground truth)
  - NMOS/PMOS Level 73 (DirectNet NN compact model via PyTorch — baseline)
  - NMOS/PMOS Level 74 (BSIM-AR Transformer compact model via PyTorch — primary)
* Sources: DC voltage/current, PULSE

Note: the legacy Shichman-Hodges LEVEL=1 model has been removed. Only
LEVEL=72/73/74 are supported.

### Analysis
* `.op` - Operating Point Analysis (Basic DC solution)
* `.dc` - DC Sweep Analysis
* `.tran` - Transient Analysis

### Directives
* `.model` - MOSFET model definitions (LEVEL=72, LEVEL=73, or LEVEL=74)
* `.include` - External library files
* `.ic` - Initial conditions (critical for SRAM/bistable circuits)

## Validation Strategy

* **Test Case:** An inverter circuit must be used to verify functionality.
* **Analysis:** The inverter must successfully pass **Transient Analysis**.
* **Ground Truth:** All simulation results must be verified against **NGSPICE**.
* **Metric:** Waveforms and operating points must match NGSPICE within reasonable numerical tolerance.

## Status

All phases (1-15) are complete. Key milestones:
- **Phases 1-3:** Core simulator (MNA, NR solver, transient)
- **Phases 4-6:** BSIM-CMG (LEVEL=72) integration via PyCMG, NGSPICE-verified (<0.02% OP, <0.1% DC)
- **Phases 7-10:** Charge-based transient (0.20% NRMSE vs NGSPICE), 5-tech support (ASAP7, TSMC5/7/12/16), 21-config parametric sweep all PASS
- **Phases 11-12:** NN compact model (LEVEL=73) — training pipeline, autograd conductances, multi-tech DC+transient verified
- **Phases 13-15:** Universal NN v2 — 21 variants across 5 techs, 13-dim input (voltages + 7 process params), 19/21 PASS (ASAP7:SLVT and TSMC7:LVT FAIL on NMOS DC)
- **Leave-one-out transferability** — 8/10 good transfer (gap < 5%), zero-shot avg 4.65% NRMSE, in-dist avg 0.95%
- **Charge-finetune training** — ChargeConsistencyLoss (autograd dq/dV = C), trained from scratch 800 epochs on A100
- **NN Transient (charge-finetune + VT fix)** — 5/5 PASS: ASAP7 6.20%, TSMC5 14.41%, TSMC7 7.15%, TSMC12 6.47%, TSMC16 7.42%
- **Solver accuracy improvements** — SPICE-standard convergence (RELTOL=1e-4, VNTOL=1e-7), GMIN reduction (1e-6→1e-12), BE→Trap first-step switching, relative oscillation threshold. NN transient improved: TSMC7 7.15→6.09%, TSMC12 6.47→5.92%, TSMC16 7.42→6.70%. BSIM-CMG transient unchanged at 0.20% (already at integration-method floor).
- **SRAM Solver Upgrades (Phases 1-3)** — Sparse matrix solver (scipy.sparse lil_matrix→CSR+spsolve), DC GMIN stepping + oscillation detection + adaptive damping + hard `.ic` mode (force_ic), BDF-2 integration (auto-switches on stiffness detection), LTE adaptive sub-stepping as constructor param. All 67 existing tests PASS with zero regression.
- **3-level DC+Transient test suites** — 3-layer infrastructure: `tests/common/base.py` (tech defs, generic helpers) -> `tests/common/bsimcmg_{dc,tran}.py` (analysis-specific) -> `verify_*.py` (test scripts). `tests/common/nn.py` consolidates the previously-duplicated NN scaffolding (nrmse, mre, checkpoint resolution, path bootstrap). NGSPICE reference netlists live in `tests/references/`. L1 regressions (OP, DC, transient) all pass against the refactored layout.
  - **Known-bad combos excluded:** TSMC5 SVT (pch PDIBL2_i<0), TSMC7 SVT/LVT (inverter garbage / pch PDIBL2_i<0), TSMC16 LNVT (nch PDIBL2_i<0), TSMC16 L=24nm (PDIBL2_i<0), NFIN=1 (NR divergence for tsmc5:ulvt / tsmc16:lnvt — ETA0_i/U0_i go negative, internal node drifts to 40V producing id=40kA + NaN derivatives; eval_dc now raises RuntimeError), P/N ratio where NFIN_P crosses NFIN group boundary (TSMC naive modelcards are NFIN-group-specific).
- **PyCMG data generation migration** — NN training data generation moved from the old `nn_model.data.generate` into `external_compact_models/PyCMG/scripts/generate_nn_data.py`. Data format includes `[NFIN, L, T, 12 process params]` geometry columns; v4 training uses only 7 input features (Vgs, Vds, Vbs, NFIN, L, T, tech_code) and ignores process params. Legal (L, NFIN) combos come from PDK bin boundaries (TSMC) or fallback list (ASAP7). 954 total geometry combos across 5 techs, 21 variants.
- **BSIMAR package refactor** — Consolidated the former `nn_model/` (DirectNet baseline) and `external_compact_models/BSIMAR/script/` (Transformer) into a single Python package at `external_compact_models/bsimar/` with clean subpackages (`config`, `data`, `models`, `losses`, `training`, `eval`, `utils`, `cli`). Unified CLI: `python -m bsimar.cli.train --model {direct,transformer} ...`. All downstream imports (pycircuitsim parser, mosfet_directnet, mosfet_bsimar) use the new `bsimar.*` namespace.
- **BSIMAR v3 production refactor (2026-04-08/09)** — After the medium-tier improvement sprint (see `external_compact_models/bsimar/docs/bsimar_improvement_plan_2026_04_08.md`) the winning recipe is **N7 (Vov-LDS) + N3 (AR finetune) + N1 (150-epoch cosine)**. All three are hard-wired as defaults. The refactor collapses the CLI (removes every experimental flag that either failed to beat baseline or was structural-always-on), deletes the signed-log normaliser entirely, and removes ~600 net lines of dead code.
  - **Final metrics on `universal_nmos.npz` medium (5.15M params)**:
    NRMSE_phys **0.223 %** (was 0.419, −46.8 %) /
    MRE_phys **1.41 %** (was 2.52, −44.0 %) /
    R² **0.9984** (was 0.9928). Wall-clock ~107 min on Blackwell GPU.
  - **Removed code**: `Normalizer` / `NormStats` / `signed_log` / `inv_signed_log`, `BSIMARNormalizer.signedlog` mode, `load_and_split` (legacy loader), `WeightedBNILoss`, `forward_curriculum`, `train_epoch_direct_ar` / `curriculum` / `scheduled` helpers, and all of `--loss direct` / `--loss bni` / `--lds` / `--vov-lds` / `--no-filter` / `--reorder` / `--scheduled-sampling` / `--curriculum` / `--consistency-weight` / `--norm-mode` / `--charge-consistency-weight` / `--learnable-output-affine` CLI flags.
  - **Hardwired knobs** (inside `train_transformer`): loss=MAE+LDS+VovLDS, norm=asinh+zscore, `parallel_caps=True`, `grouped_inputs=True`, BSIMAR column reorder, phys-best checkpoint tracker, AR finetune phase (default 5 epochs).
  - **Known-infeasible explored options** (DO NOT retry without new structural argument): N6 Huber on I/V block (wrong gradient shape near zero), N5 learnable output affine (disrupts post-asinh zscore), N4 charge-consistency penalty (asinh chain rule has a `cosh(asinh(q/s))` factor that makes the constraint inequivalent to matching targets). Full postmortems in the plan file.
  - **Deferred**: N2 KV-cache encoder. Empirical evidence that 5 AR-finetune epochs suffices means the ~10 min it would save on a 107 min run is not worth the ~200 LOC bit-exact rewrite of `nn.TransformerEncoderLayer`.
  - **File renames**: `pycircuitsim/models/mosfet_nn.py` → `mosfet_directnet.py` (the class names `NMOS_NN` / `PMOS_NN` / `_MOSFETNNBase` are unchanged).
  - **Checkpoint incompatibility**: existing v2 Transformer and legacy DirectNet checkpoints do NOT load with the refactored loaders. Superseded by v4 tech-code migration.
- **BSIMAR package refactor** (2026-03) — Consolidated the former `nn_model/` (DirectNet baseline) and `external_compact_models/BSIMAR/script/` (Transformer) into a single Python package at `external_compact_models/bsimar/`.
- **BSIMAR v4 tech-code migration (2026-04-14)** — All v3 code (19-dim continuous process params) removed. Only v4 architecture (7-dim + discrete tech-code embedding via `nn.Embedding`) is supported. DirectNet (`DirectNetV4`) and Transformer both use tech-code embedding for technology identity instead of 12 continuous process parameters. ASAP7 excluded from training (`--exclude-techs asap7`). 4 universal models trained: DirectNet NMOS/PMOS (0.00167/0.00190 val loss) + Transformer NMOS/PMOS (0.270%/0.252% NRMSE, R²=0.9937/0.9965). TSMC5 SVT verification: DC PASS (7.79%/9.99%), VTC 17.70%. Removed: `ProcessParams`, `extract_process_params`, `PROCESS_PARAM_NAMES`, old 19-dim `INPUT_COLUMNS`. Added: `TECH_CODE_MAP`, `--exclude-techs`, `--num-tech-codes` CLI flags. Checkpoint naming changed to `v4_` prefix.
- **Analytical Vds correction for inverter transient (2026-04-15)** — Implemented `_apply_vds_correction()` in `_MOSFETNNBase` to enforce Id(Vds=0)=0 and Id=0 for reverse-Vds at inference time. Three-part correction: one-sided Vds factor (VT=0.052V), symmetric gds with linear-region conductance, sign enforcement (NMOS id≤0, PMOS id≥0). DirectNet inverter transient: **3/4 PASS** (TSMC7 8.87%, TSMC12 11.65%, TSMC16 10.59%; TSMC5 17.20% marginal FAIL). BSIMAR inverter: 0/4 PASS due to wrong-sign subthreshold predictions in Transformer (requires retraining). NMOS pulse: 8/8 PASS, zero regression. Full report: `results/v4_vds_correction_report_2026_04_15.md`.
- **Rail-restoring extrapolation fixes BSIMAR inverter transient (2026-04-20)** — Diagnosed the real root cause of BSIMAR inverter transient explosion (V(out)→+4.4V on TSMC12/16): both NN models (BSIMAR and DirectNet) predict Id≈0 outside the `[-VDD_train, VDD_train]` training range, creating a flat-zero KCL plateau the DCSolver mistakes for an equilibrium. The earlier sign+boundary loss hypothesis was incomplete. Fixed by adding rail-restoring extrapolation to `_apply_vds_correction()`: quadratic Id ramp + linear gds ramp past `VDD_train`, smooth-joined at the boundary (a linear ramp was tried first and caused NR oscillation for TSMC12/16 whose operating points sit at the boundary). Verified on probe (670K) and production (5.15M) checkpoints across all 4 TSMC techs: **inverter transient drops from 18-300% NRMSE (FAIL) to 6-12% NRMSE (PASS)** on TSMC5/7/12/16. Production: TSMC5 12.13%, TSMC7 9.14%, TSMC12 6.78%, TSMC16 7.51%. The fix is inference-time only — no retraining required to ship. Diagnostic + multi-tech verify infrastructure added under `tests/diag_bsimar_kcl_landscape.py` and `tests/verify_bsimar_v4_inverter.py --tech <tsmc5|tsmc7|tsmc12|tsmc16>`.
- **v5 inverter-transient sprint (2026-04-22/23) — closed, no production change.** Five-experiment sweep attempting to lift the worst-case TSMC7 NMOS DC (14.72 %) and drive BSIMAR inverter VTC TSMC7 (19.15 %) below 10 %. E1 (wider Vds-correction VT), E3 (per-tech fine-tune on same distribution), E4 (dense hot-box overlay + universal set), E5 (overlay-only fine-tune) all reverted on inverter acceptance gates. D1 diagnostic (`results/v5_d1_tsmc7_nmos_errors/`) isolated the TSMC7 NMOS error to the strong-inversion + saturation plateau (Vgs ∈ [0.52, 0.73] V × Vds ∈ [0.40, 0.75] V) and showed it is 16× under-sampled by LHS, but both densification approaches regressed NMOS DC by identically +2.7 pp, ruling out the straight density thesis. Retained shipping changes: (a) `tests/diag_d1_tsmc7_nmos_errors.py` heatmap diagnostic, reusable for any tech/device accuracy investigation; (b) `external_compact_models/bsimar/training/finetune.py` edge-case patch to guard empty `test_idx` when all rows belong to `finetune_techs`. Full history: `results/v5_session_summary_2026_04_23.md` (concise sprint postmortem).
- **v5 Phase A — Trim (2026-04-24, branch `feat/bsimar-v5-phase-a`).** Deleted all unjustified and dead loss code before Phase B work. Plan: `docs/superpowers/plans/2026-04-24-v5-inverter-accuracy.md`.
  - **A1:** Deleted `DirectLoss`, `ChargeConsistencyLoss` (never instantiated in production), legacy `BSIMARConfig`/`TrainConfig` aliases, and dead `TransformerConfig` fields (`ss_warmup_epochs`, `w_curr`, etc.).
  - **A2:** Deleted `SignConsistencyLoss`, `BoundaryLoss` — no A/B benefit; superseded by rail-restoring extrapolation at inference and the structural B3 gate.
  - **A3:** Collapsed 3-axis LDS weight stack (per-target × Vov × subthreshold) to **per-target only** — the only axis with published evidence; the other two were tied to the wrong-sign-Id sprint.
  - **A5:** Deleted `_eval_autograd4` dead fast-path in `mosfet_directnet.py`; added 13-output assertion at load time.
  - **A4 control retrain — GATE FAIL.** Retrain with trimmed pipeline (A1–A3) on existing `universal_*.npz` regressed TSMC7 NMOS DC past the ±1 pp gate. Root cause confirmed: the LHS dataset that Phase A trains on is insufficient — Phase B B1's hybrid uniform-grid data is required. Phase A code changes are correct and retained; the gate failure is a data issue, not a code regression.
- **v5 Phase B — Levers tried, code reverted (2026-04-24..2026-05-03).** Three Phase B levers were prototyped to address the confirmed TSMC7 sampling-basis mismatch:
  - **B1 (data, retained in PyCMG submodule):** hybrid uniform-grid + LHS jitter sampler with `sample_class` column. Datasets regenerated under this sampler are still consumed by the loader.
  - **B2 (`SlopeMatchLoss`) and B3 (`apply_id_gate`) — DELETED 2026-05-03** under `docs/superpowers/plans/2026-05-03-nn-stack-trim.md`. Neither lever was validated against a v4 baseline before B3's `id_idx_in_stats` bug corrupted the v5b/v5c TF runs. Inference-time `_apply_vds_correction` (rule 19) already enforces Id(Vds=0)=0; rail-restoring extrapolation (rule 19a) is the load-bearing piece.
  - **AR-finetune phase / `forward_scheduled` — DELETED 2026-05-03** in the same trim. The 5/150 final-phase rollout was carrying ~160 LOC of separate optimizer + loader + tracker + checkpoint plumbing for a marginal benefit over the cosine schedule.
- **v4-re — NN-stack trim (2026-05-03, branch `chore/nn-stack-trim`).** Plan: `docs/superpowers/plans/2026-05-03-nn-stack-trim.md`. The current shipping NN stack is now labeled **v4-re** (v4 reissue): same v4 7-dim + tech-code architecture, but with all unvalidated Phase B levers and AR-finetune plumbing removed. Re-trained checkpoints land under the `v4_re_` prefix; legacy `v4_` checkpoints continue to load via the resolver fallback.
  - **PR-1:** Removed 11 broken/superseded test scripts (~3.9 KLOC) — all v3-era APIs (`Normalizer`, `inv_signed_log`, `DirectLoss`, `MOSFETDatasetV4`, `PROCESS_PARAM_NAMES`).
  - **PR-2:** Deleted `bsimar/losses/slope_loss.py`, `bsimar/models/id_gate.py`, `forward_scheduled` on the Transformer, `_train_epoch_scheduled_mae`, the trainer's AR-finetune block, `BSIMARNormStats.id_gate` field, and the matching CLI flags (`--slope-weight`, `--slope-warmup-frac`, `--no-id-gate`, `--ar-finetune-epochs`). Inference glue deduped: `_resolve_nn_checkpoint(level, ...)` collapses LEVEL=73/74 path resolution and prefers `v4_re_*` over legacy `v4_*`; `_floor_gds(id, gds)` replaces 4 stamp sites; `_MOSFETBSIMARBase` reuses parent `_denorm_scalar` / `_denorm_full_derivative` via column-index lookup. v4 checkpoints continue to load unchanged (the deleted Phase B fields were optional). v5b checkpoints are discard-only per Bug A.
  - **Default save_prefix:** `train_directnet` → `v4_re_dn_universal_<dev>`; `train_transformer` → `v4_re_universal_<dev>`. Override with `--exp-name`.

### Known v4 limitation (carried into v4-re until retrain): TSMC7 NMOS DC 14.72 %
TSMC7 NMOS DC NRMSE is 14.72 % (BSIMAR v4) / 15.79 % (DirectNet v4) against PyCMG ground truth at Vds=VDD/2, NFIN=10, L=16 nm. This propagates to inverter VTC (19.15 % BSIMAR / 18.14 % DirectNet). Root cause confirmed by D1 diagnostic: the LHS training distribution under-samples the strong-inversion + saturation plateau by ~16× relative to the verifier's uniform Id-Vgs sweep. Inverter transient at TSMC7 PASSES (6.80 % DN / 9.14 % BSIMAR). VTC should be reported with the 14-19 % NRMSE caveat. The mitigation lives in v4-re: retrain on B1 hybrid-grid data with the trimmed pipeline, save under `v4_re_*` prefix, expect TSMC7 NMOS DC ≤ 8 % per the trim plan's gate.

### Future Work
- [ ] **v4-re retrain on B1 hybrid-grid data** — Infrastructure committed in PyCMG. Acceptance gate: TSMC7 NMOS DC ≤ 8 %, VTC ≤ 12 %, all transients ≤ 15 %. Saves under `v4_re_dn_universal_*` / `v4_re_universal_*`.
- [ ] **PMOS DC TSMC12/16** — 12-14 % NRMSE (both models); same class of sampling-basis bug as TSMC7 NMOS. Run D1-style heatmap diagnostic on TSMC12 PMOS before committing to B1 hot-region list.
- [ ] **PR-3 of the trim plan: phys-best A/B + base-class collapse** — A/B one DirectNet run with phys-best disabled; if Δ < 5 % NRMSE, delete the tracker. Then collapse `_MOSFETBSIMARBase` into `_MOSFETNNBase(model_factory, column_indices)`.
- [ ] **Improve TSMC5 inverter transient** — DirectNet 17.20% NRMSE (just above 15% threshold). Likely also a B1 data densification target.
- [ ] **SRAM Validation (Phase 4)** — 6T bitcell DC+transient, 8-cell column, 64-bit array benchmark vs NGSPICE
- [ ] **Adaptive Output Timestep** — Variable-length output array with true adaptive dt (full adaptive requires output grid changes)
- [ ] ~~**BSIMAR v4 per-tech runs**~~ — **DEAD**: v5 E3 showed training-space NRMSE 0.454 % on TSMC7 → inference-space NMOS DC 14.74 % (29× gap). Per-tech fine-tuning alone does not close the gap.
- [ ] **N2 KV-cache encoder** (filed) — Only worth doing if 500-epoch training becomes routine. ~200 LOC bit-exact rewrite of `nn.TransformerEncoderLayer` attention.

---

## Setup

### Environment
```bash
# Create conda environment
conda create -n pycircuitsim python=3.10 -y
conda activate pycircuitsim

# Install dependencies
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# Install PyTorch (CPU; for GPU training use the CUDA variant)
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch

# Initialize PyCMG submodule
git submodule update --init --recursive
```

### Prerequisites
- **NGSPICE 45.2+**: `/usr/local/ngspice-45.2/bin/ngspice` (for verification tests)
- **OpenVAF 23.5.0+**: `/usr/local/bin/openvaf` (for OSDI compilation)
- **BSIM-CMG OSDI binary**: Pre-compiled at `external_compact_models/PyCMG/build/osdi/bsimcmg.osdi`

## Quick Start

### Basic Simulation
Create a netlist (`.sp` file). Examples in `examples/`.

**BSIM-CMG Geometric Parameters:** `L` (channel length), `NFIN` (fin count), `TFIN`/`HFIN`/`FPITCH` (optional, uses modelcard defaults).

### NN Model (LEVEL=73 DirectNet baseline + LEVEL=74 BSIM-AR Transformer) — current revision: **v4-re**
```bash
# Generate universal data (PyCMG is the canonical data generator)
# Uses PDK-defined (L, NFIN) bins (TSMC bin boundaries, ASAP7 fallback list)
# and extracts process parameters on-the-fly from modelcards (per-bin accurate).
# 954 total geometry combos across 5 techs and 21 variants.
conda run -n pycircuitsim python external_compact_models/PyCMG/scripts/generate_nn_data.py \
    --device both --universal

# Train DirectNet v4-re (MAE + per-target LDS, hard-wired recipe).
# ASAP7 excluded from training; 18 tech-codes for 4 TSMC techs x variants.
# Saves to `v4_re_dn_universal_<dev>_best.pt` by default.
conda run -n pycircuitsim python -u -m bsimar.cli.train \
    --model direct --device-type nmos \
    --exclude-techs asap7 --num-tech-codes 18 \
    --epochs 800 --hidden 384 --layers 6 --patience 150 --batch-size 2048 --cuda
conda run -n pycircuitsim python -u -m bsimar.cli.train \
    --model direct --device-type pmos \
    --exclude-techs asap7 --num-tech-codes 18 \
    --epochs 800 --hidden 384 --layers 6 --patience 150 --batch-size 2048 --cuda

# Train BSIMAR v4-re Transformer (production recipe).
# Hard-wired: MAE + per-target LDS, asinh+zscore, parallel_caps,
# grouped_inputs, BSIMAR reorder, phys-best ckpt, nn.Embedding tech-code.
# Saves to `v4_re_universal_<dev>_best.pt` (+ `.ar.pt`, `.phys.pt`).
conda run -n pycircuitsim python -u -m bsimar.cli.train \
    --model transformer --device-type nmos \
    --exclude-techs asap7 --num-tech-codes 18 --cuda
conda run -n pycircuitsim python -u -m bsimar.cli.train \
    --model transformer --device-type pmos \
    --exclude-techs asap7 --num-tech-codes 18 --cuda

# Note: `bsimar` is importable because consumers add
#       `external_compact_models/` to sys.path. Checkpoints live under
#       `external_compact_models/bsimar/checkpoints/` for both models.
```
Checkpoints (under `external_compact_models/bsimar/checkpoints/`):
- **v4-re DirectNet**: `v4_re_dn_universal_{nmos,pmos}_best.pt` + `_norm.npz` (target: post-trim retrain).
- **v4-re Transformer**: `v4_re_universal_{nmos,pmos}_best.phys.pt` + `_best.pt` + `_best.ar.pt` + `_norm.npz` + `_config.npz` (**simulator prefers `_best.phys.pt`** when norm.npz declares `phys_best_metric == "median"`; otherwise falls back to `_best.pt`).
- **Legacy v4** (`v4_dn_universal_*`, `v4_universal_*`): still on disk; resolver picks them only when no `v4_re_*` exists. Kept for reproducibility while v4-re retrains land.

Netlist usage:
- LEVEL=73 (DirectNet): `.model nmos_nn NMOS (LEVEL=73 TECH=tsmc5 VT=lvt)` with `L=16n NFIN=10`.
- LEVEL=74 (BSIMAR v4 Transformer): `.model nmos_ar NMOS (LEVEL=74 TECH=tsmc5 VT=lvt)` with `L=16n NFIN=10`.
- Parser auto-resolves tech-code from TECH+VT and prefers the universal checkpoint when available.

### Output Files
Results organized in `results/<circuit_name>/<analysis_type>/`:
- `*_simulation.lis` - Detailed iteration log (HSPICE-like)
- `*_dc_sweep.csv` / `*_transient.csv` - Waveform data (node voltages + device currents)

## Testing & Verification

All tests require: `conda activate pycircuitsim`

Shared test infrastructure lives in the `tests/common/` subpackage:
- `tests/common/base.py` — project paths, `TechProfile`, `VtPair`, `ALL_TECHS`, NGSPICE subprocess runner, generic orchestration.
- `tests/common/bsimcmg_dc.py` — DC-specific runners, metrics, plots.
- `tests/common/bsimcmg_tran.py` — transient-specific runners, metrics, plots.
- `tests/common/nn.py` — NN-specific helpers (`nrmse`, `mre`, checkpoint resolution, `bsimar` + `pycmg` sys.path bootstrap).
- `tests/references/` — tracked NGSPICE reference netlists (ngspice_*.cir).

**BSIM-CMG DC Verification (3-level, shared infra in `tests/common/base.py` + `tests/common/bsimcmg_dc.py`):**

| Level | Script | Tests | What it tests |
|-------|--------|-------|---------------|
| 1 | `verify_bsimcmg_dc.py` | 2 | NMOS + PMOS Id-Vgs (ASAP7, <1% NRMSE) |
| 2 | `verify_bsimcmg_dc_comprehensive.py` | 67 | VT/L/NFIN sweeps, NMOS+PMOS, 5 techs |
| 3 | `verify_multi_tech_dc.py` | 44 | Inverter VTC + parametric (VT, L, NFIN, P/N) |

**BSIM-CMG Transient Verification (3-level, shared infra in `tests/common/base.py` + `tests/common/bsimcmg_tran.py`):**

| Level | Script | Tests | What it tests |
|-------|--------|-------|---------------|
| 1 | `verify_bsimcmg_tran.py` | 1 | Inverter pulse (ASAP7, <5% NRMSE) |
| 2 | `verify_bsimcmg_tran_comprehensive.py` | 37 | VT/L/NFIN sweeps, 5 techs |
| 3 | `verify_multi_tech_tran.py` | 72 | Multi-tech + parametric (P/N, VDD, Cload, slew, pw) |

**NN v4 DC Verification (aligned with BSIM-CMG 3-level structure):**

| Level | Script | Tests | What it tests |
|-------|--------|-------|---------------|
| 1 | `verify_nn_dc.py` | ~6 | NMOS+PMOS DC + inverter VTC (TSMC12 SVT, LEVEL=73/74) |
| 1+ | `verify_nn_dc_tran.py --dc-only` | ~12 | NMOS DC across all 4 TSMC techs |
| 1+ | `verify_nn_dc_tran.py --pmos-only` | ~12 | PMOS DC across all 4 TSMC techs |
| 1+ | `verify_nn_dc_tran.py --inverter-only` | ~12 | Inverter VTC across all 4 TSMC techs |

**NN v4 Transient Verification:**

| Level | Script | Tests | What it tests |
|-------|--------|-------|---------------|
| 1 | `verify_nn_tran_v4.py` | ~4 | NMOS pulse + inverter tran (TSMC12 SVT, LEVEL=73/74) |
| 1+ | `verify_nn_dc_tran.py --tran-only` | ~8 | NMOS pulse across all 4 TSMC techs |

**Other:**

| Test Suite | Script | What it tests |
|-----------|--------|---------------|
| OP Verification | `verify_bsimcmg_op.py` | NMOS, PMOS, Inverter OP vs NGSPICE (<0.02%) |
| NN Leave-One-Out | `verify_nn_leave_one_out.py` | Zero-shot transferability experiment |

Note: the `verify_nn_*.py` scripts above need porting to the v4 tech-code
API (`TECH_CODE_MAP` lookup instead of the old `extract_process_params`).
v4 checkpoints are available; the test scripts need updating to pass
tech-code integers instead of process-param vectors.
This is a tracked follow-up.

**Quick Sanity Check:**
```bash
python tests/verify_bsimcmg_op.py && python tests/verify_bsimcmg_dc.py && python tests/verify_bsimcmg_tran.py
```

---

## Development Guidelines

### Coding Standards
- Type hints required for all function signatures
- Clear variable names (e.g., `v_gate`, `i_drain`, not `a`, `b`)
- Docstrings for complex algorithms
- Voltage clamping: Vgs +/- 5V, Vds +/- 10V

### Separation Principle
- **Solver** (`solver.py`) builds MNA matrix, executes Newton-Raphson (no device equations)
- **Device Models** (`models/`) calculate current/conductances from voltages (no matrix operations)
- **Simulation** (`simulation.py`) orchestrates the workflow (parse -> solve -> visualize)
- All devices inherit from `Component` base class

### Key Numerical Techniques
- **Sparse MNA solver**: `scipy.sparse.lil_matrix` for assembly, CSR + `spsolve` for linear solve. O(n) memory, O(n·log n) solve.
- **SPICE-standard convergence**: `|ΔV| < VNTOL + RELTOL × max(|V_old|, |V_new|)` (RELTOL=1e-4, VNTOL=1e-7)
- **GMIN conductance** (1e-12 S) prevents singular matrices. DC GMIN stepping (opt-in via `use_gmin_stepping=True`): schedule [1e-6, 1e-8, 1e-10, 1e-12] for bistable circuits.
- **BE→Trap→BDF-2 switching**: Backward Euler (step 1), Trapezoidal (step 2+), BDF-2 (auto on stiffness, NR>20 iters). One-way switch: once BDF-2 activated, stays on BDF-2.
- Source stepping (20 steps) improves convergence
- Adaptive damping with supply-relative thresholds and stuck-counter detection
- **DC oscillation detection**: tracks last 5 NR voltage snapshots; accepts averaged solution if variance < 10× tolerance
- **Hard `.ic` mode** (`force_ic=True`): stamps `.ic` nodes as temporary voltage source constraints, then re-solves unconstrained. Ensures correct bistable state for SRAM latches.
- Two-stage analysis: DC OP -> DC sweep/transient
- Voltage-source-constrained nodes exempt from damping
- **LTE sub-stepping** (opt-in via `max_substeps` constructor param, default 1=disabled): estimates local truncation error from output-grid curvature, uses internal sub-steps when LTE exceeds `lte_safety_factor` threshold.

### Entry Points
- **CLI**: `main.py` - Command-line interface (argparse, error handling)
- **API**: `pycircuitsim.simulation.run_simulation()` - Programmatic access
- **Module**: `pycircuitsim` - Package exports (Circuit, Parser, Visualizer, run_simulation)

## Environment & Tools
* **Conda Environment**: `pycircuitsim` in `/home/shenshan/.conda/envs/pycircuitsim`
* **PyTorch:** 2.10.0 (CPU, installed via pip in pycircuitsim env)
* **OpenVAF Compiler:** `/usr/local/bin/openvaf`
* **NGSPICE Simulator:** `/usr/local/ngspice-45.2/bin/ngspice`
* **Build System:** CMake / Make
* **Python Bindings:** PyBind11

---

## Critical Design Rules

These rules were learned from bugs. Violating them causes NR divergence or wrong results.

### Sign Convention for Device Models

When integrating new compact models, follow this checklist:

1. **Use terminal current `id`, NOT channel current `ids`** — `ids = id - is ~ 2*id` (2x error)
2. **NMOS `calculate_current()`**: return `-result["id"]` (positive = current leaving drain)
3. **PMOS `calculate_current()`**: return `result["id"]` (positive = current into drain)
4. **Solver stamping** uses unified "current leaving drain" convention. All VCCS conductances (g_ds, g_m, g_mb) must have full 4-entry matrix stamps (drain,ctrl+, drain,ctrl-, source,ctrl-, source,ctrl+). An incomplete stamp (e.g., only drain,bulk for gmb) breaks the Jacobian symmetry and degrades NR convergence.
   ```python
   i_leaving = -i_ds if is_pmos else i_ds
   i_eq = i_leaving - g_ds * v_ds - g_m * v_gs - g_mb * v_bs
   rhs[d_idx] -= i_eq  # Same for NMOS and PMOS
   rhs[s_idx] += i_eq
   ```
5. **Conductance signs**: `max(gds, 1e-12)` floor (do NOT use `abs(gds)` — it turns large negative gds into large positive, causing NR divergence). Preserve gm/gmb signs.
6. **Update `_is_mosfet()`** in `solver.py` when adding new device types
7. **Test both NMOS and PMOS** against NGSPICE: single OP, DC sweep, inverter VTC, inverter transient

### NN Model Rules (LEVEL=73 DirectNet v4 + LEVEL=74 BSIMAR v4)

Both NN compact models share the same data pipeline and the same
inference-time rules. DirectNet is the baseline (single-shot MLP);
BSIMAR is the primary model (autoregressive Transformer with
parallel cap head). Both use `nn.Embedding` for discrete tech-code
identity (7-dim input: Vgs, Vds, Vbs, NFIN, L, T, tech_code)
instead of 19-dim continuous process parameters.

1. **Jacobian consistency is mandatory** — gm/gds/gmb MUST be `torch.autograd.grad(id, V)`, never independent predictions. Without this, NR diverges in multi-device circuits. This holds for both LEVEL=73 and LEVEL=74.
2. **PMOS source-relative frame** — Shift all voltages by -Vs before NN eval (`v_d_nn = v_d - v_s`). Training uses Vs=0; in CMOS, PMOS Vs=VDD.
3. **Training range covers NR overshoot** — Margin of +/-VDD beyond operating range, not just +/-0.1V.
4. **Voltage clamping** — Use softplus-based smooth clamp (NOT `torch.clamp`) to the training-domain `input_min`/`input_max`. Hard clamp creates zero-gradient cliffs at boundaries that stall NR convergence. Smooth clamp margin = 5% of per-dimension training range.
5. **gds floor** — Use physics-based floor `gds = max(gds, |id|*0.5, 1e-12)`, NOT `max(gds, 1e-12)`. NN autograd gds ≈ 0 in saturation (NN learns flat Id-Vds). Without the floor, inverter gain → ∞ and NR diverges. At FinFET 16nm, BSIM-CMG lambda=0.3-1.2 V⁻¹. The floor only affects the NR Jacobian, not the converged solution.
6. **Normalisation — asinh (Transformer) or zscore (DirectNet)**. The Transformer uses asinh + z-score on outputs (`y_norm = (asinh(y/s_k) - m)/std`, `s_k` = per-target geometric-mean scale); DirectNet uses plain z-score on outputs. Both use z-score on the 6 continuous inputs (Vgs, Vds, Vbs, NFIN, L, T); the tech-code integer is passed directly to `nn.Embedding` and is not normalised. (Post-2026-05-03: `train_directnet` now passes `norm_mode="zscore"` explicitly to `load_and_split_bsimar`; previously it relied on the loader default. Existing v4/v5a/v5b/v5c DirectNet checkpoints stay valid because their training is end-to-end consistent under their training norm_mode.)
7. **Chain-rule denormalisation** — Simulator consumers must apply the right chain rule per normaliser:
   - **zscore** (DirectNet): `dy_phys/dv_phys = dy_norm/dv_norm * out_std / in_std` (linear).
   - **asinh** (Transformer): `dy_phys/dv_phys = dy_norm/dv_norm * out_std * sqrt(asinh_scale² + y_phys²) / in_std`.
8. **TSMC asymmetric L** — NMOS L=16nm, PMOS L=20nm; NNTechConfig uses `L_nmos`/`L_pmos`.
9. **ASAP7 modelcard name mapping** — Parser auto-maps netlist names to `nmos_rvt`/`pmos_rvt`.
10. **PyCMG integration** — `bsimar/config.py` re-exports `NNTechConfig` and `TECH_CONFIGS` from PyCMG's `pycmg.nn_config`, plus `TECH_CODE_MAP` (maps `"tech:vt"` strings to integer codes) and `OUTPUT_COLUMNS`. Process params (`ProcessParams`, `extract_process_params`, `INPUT_COLUMNS`) are no longer re-exported -- v4 uses discrete tech-code embedding instead. Training VDD may differ from PyCMG (e.g., ASAP7: train=0.7V, PyCMG=0.9V). Backward-compat alias `TechConfig = NNTechConfig` exists for test files.
10. **Data generation validates PyCMG output** — `eval_single_point` rejects NaN/Inf and `|id| > 1A`. PyCMG `eval_dc` raises `RuntimeError` on internal-node convergence failure. Default NFIN range is `[2, 3, 5, 10, 15, 20, 24]` (NFIN=1 excluded due to OSDI convergence failures).
11. **Loss layer (per model)** — Both DirectNet and Transformer use `bsimar.losses.MAELoss` with **per-target LDS weights only** (3-axis stack collapsed to 1-axis in v5 Phase A). Hard-wired inside `train_directnet` / `train_transformer`. Deleted code (DO NOT re-add): `DirectLoss`, `ChargeConsistencyLoss`, `SignConsistencyLoss`, `BoundaryLoss`, `SlopeMatchLoss`, Vov-LDS and subthreshold-LDS axes. The structural Vds gate (`apply_id_gate`) and slope-match loss were prototyped under v5 Phase B and deleted on 2026-05-03 (`docs/superpowers/plans/2026-05-03-nn-stack-trim.md`) — Bug A corrupted their checkpoints and rule 19's inference-time correction already enforces Id(Vds=0)=0.
12. **BSIMAR output ordering** — The Transformer output is in `BSIMAR_COLUMN_ORDER` (`qg, qb, qd, qs, id, gm, gds, gmb, cgg, cgd, cgs, cdg, cdd`), not `OUTPUT_COLUMN_ORDER`. Consumer code (`mosfet_bsimar.py`) takes autograd derivatives at the right column indices.
13. **Parallel cap head** — The Transformer emits the 5 capacitance outputs in parallel from the gmb hidden state, not as sequential AR steps. The AR loop runs only 8 steps (charges + currents/conds). `parallel_caps` and `grouped_inputs` are structural and not configurable.
14. **(removed)** — The AR-finetune phase / `forward_scheduled` was deleted on 2026-05-03; the cosine TF schedule alone produces the production checkpoint. The phys-best checkpoint tracker still selects the best model across the schedule.
15. **Unified CLI** — Training goes through `python -m bsimar.cli.train --model {direct,transformer} ...`. Both models read the same `.npz` produced by `external_compact_models/PyCMG/scripts/generate_nn_data.py` and write checkpoints under `external_compact_models/bsimar/checkpoints/`.
16. **Checkpoint files** — Current revision is **v4-re**. DirectNet: `v4_re_dn_universal_{nmos,pmos}_best.pt` + `_norm.npz`. Transformer: `v4_re_universal_{nmos,pmos}_best.pt` (TF val-best), `_best.ar.pt` (AR val-best), `_best.phys.pt` (physical-space val-best), `_norm.npz` (BSIMARNormStats asinh), `_config.npz` (architecture). The simulator's `_resolve_nn_checkpoint` cascade is **v4_re_universal > v4_universal > per-tech > bare**, so legacy `v4_*` checkpoints keep working until v4-re replacements ship. **Post-2026-05-03 phys-best fix**: `_best.phys.pt` is only trustworthy when the trainer's `BSIMARNormStats.phys_best_metric == "median"`. Pre-fix files were corrupted by the mean-aggregated phys-score (plan §2B) and have been renamed to `*best.phys.bug.pt`; the simulator loader checks the flag and falls back to `_best.pt` for legacy norm.npz files lacking the key. v5b/v5c TF Transformer runs additionally hit the now-deleted `apply_id_gate` index-mismatch bug — discard-only.
17. **Charge conservation enforcement** — Simulator always computes `qs = -(qg + qd + qb)` analytically, even for 13-output models that directly predict `qs`. This guarantees Kirchhoff current conservation at every transient timestep.
18. **ASAP7 tech-code exclusion** — ASAP7 tech codes (18-21) exceed the v4 training vocabulary (18 codes, indices 0-17). Running ASAP7 with v4 universal models will crash with an embedding index-out-of-range error. ASAP7 requires separate fine-tuning.
19. **Analytical Vds correction** — `_MOSFETNNBase._apply_vds_correction()` enforces Id(Vds=0)=0 and Id=0 for reverse-direction Vds at inference time. Four-part correction (the order matters):
   - (a) **Rail-restoring extrapolation** when `|Vds| > VDD_train` (where `VDD_train = self._vdd_estimate`, derived from training norm stats): adds a *quadratic* Id ramp `½·g_max·overshoot² / x_ref` (with `g_max=1mS`, `x_ref=½·VDD_train`) and a *linear* gds ramp `g_max·overshoot / x_ref`. Both start at zero with zero slope at the boundary so NR sees a smooth join (a linear/discontinuous ramp causes NR oscillation when operating points sit at the boundary, e.g. TSMC12/16 where `_vdd_estimate≈VDD`). Must run BEFORE the fast-path early-return.
   - (b) one-sided `1-exp(-|Vds|/VT)` with VDD-proportional `VT = max(0.06·VDD, 0.026)V` for Id/gm/gmb,
   - (c) symmetric gds with linear-region conductance `|Id_raw|·exp(-|Vds|/VT)/VT`,
   - (d) sign enforcement (NMOS id≤0, PMOS id≥0).
   
   The rail-restoring step (a) is what fixed the BSIMAR transient bug: the trained NN extrapolates flat-near-zero outside `[-VDD_train, VDD_train]`, creating a false KCL plateau the DCSolver mistakes for an equilibrium (inverter transient OP locking at V(out)=4.4V instead of VDD). Step (a) replicates PyCMG's restoring leakage/impact-ionization physics so NR converges to the true rail.
20. **BSIMAR inverter circuits** — With rule 19 step (a), both BSIMAR (LEVEL=74) and DirectNet (LEVEL=73) work on inverter and feedback circuits. Verified across all 4 TSMC techs with both probe (670K) and production (5.15M) checkpoints — transients drop from 18-300% NRMSE (FAIL/explosion) to 6-12% NRMSE (PASS). The structural bug was extrapolation past the training Vds range, not subthreshold sign errors. The rail-restoring fix unblocks production checkpoints without retraining.

---

## References
- **ngspice** - Physics equation verification
- **Xyce** - Architecture patterns for device/solver separation
- **BSIM-CMG** - FinFET compact model (LEVEL=72), integrated via PyCMG
- **ASAP7** - https://github.com/The-OpenROAD-Project/asap7_pdk_r1p7.git
- **PyCMG** - https://github.com/ShenShan123/PyCMG.git

## Project Structure Notes

### Important Path References
- **PyCMG Location**: `external_compact_models/PyCMG/` (git submodule, 21 device variants)
- **PyCMG Submodule**: `external_compact_models/PyCMG/` (git submodule)
- **BSIM-CMG OSDI Binary**: `build/osdi/bsimcmg.osdi` (relative to PyCMG root)
- **Modelcards**: `modelcards/` (relative to PyCMG root; ASAP7: `ASAP7/*.pm` committed; TSMC: raw PDK `TSMC{5,7,12,16}/cln*.l` is gitignored/IP-protected and naive modelcards are regenerated on-the-fly via `pycmg.tech.resolve_modelcard` into `build/modelcards/`)
- **PyCMG Test Helpers**: `tests/helpers.py` (relative to PyCMG root; was `pycmg/testing.py`)
- **Results Output**: `results/<circuit_name>/<analysis_type>/` (`.lis`, `.csv`, `.png`)
- **Examples**: `examples/` (13 netlists)
- **Test Results**: `tests/verify_*_results/` (generated, not tracked in git)

## Other Tips
* **Start every complex task in plan mode:**
    * Pour your energy into the plan for 1-shot the implementation.
    * The moment something goes sideways, just switch back to plan mode and re-plan. Don't keep pushing.
    * Enter plan mode for verification steps, not just for the build.
* **Update CLAUDE.md:**
    * After every correction, update your CLAUDE.md so you don't make that mistake again.
* **Never be lazy:**
    * Never be lazy in writing the code and running tests.
    * Do NOT use any simplified equations or self-defined CMG models as reference, ALWAYS use simulation results as ground truth for comparison.
* Use subagents.
    * Use a second agent to review the plan as a staff engineer.
    * If you want to try multiple solutions, use multiple subagents, git commit to different branches. Roll back and to the main branch and create new branch when the subagent find it's a dead end.
* Enable the "Explanatory" or "Learning" output style in /config to explain the *why* behind its changes.
