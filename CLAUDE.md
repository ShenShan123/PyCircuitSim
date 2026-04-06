# Project: PyCircuitSim

## Overview
Python-based SPICE-like circuit simulator emphasizing educational clarity and modular architecture.
**Primary Goal:** specific support for **PyCMG-wrapped CMG models** (LEVEL=72) and **NN-based compact models** (LEVEL=73).
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
    ├── base.py         # Component abstract base class
    ├── passive.py      # R, C, V, I sources (including PULSE)
    ├── mosfet_cmg.py   # BSIM-CMG FinFET model (LEVEL=72) via PyCMG — MOSFET_CMG base + NMOS/PMOS subclasses
    └── mosfet_nn.py    # NN-based compact model (LEVEL=73) via PyTorch — _MOSFETNNBase + NMOS/PMOS subclasses

external_compact_models/
├── BSIMAR/             # Unified NN-based compact model package (DirectNet baseline + BSIM-AR Transformer)
│   ├── bsimar/         # Python package — importable as `bsimar`
│   │   ├── __init__.py
│   │   ├── config.py               # NNTechConfig + ProcessParams + DirectNetConfig + TransformerConfig, re-exports from pycmg.nn_config
│   │   ├── data/
│   │   │   ├── normalize.py        # Normalizer (signed-log) + BSIMARNormalizer (zscore/signedlog) + signed_log helpers
│   │   │   ├── dataset.py          # MOSFETDataset + load_and_split{_bsimar} + filter_small_targets
│   │   │   └── analyze.py          # Dataset analysis script (distribution, outliers, physical constraints)
│   │   ├── models/
│   │   │   ├── direct_net.py       # DirectNet MLP (baseline used for comparison)
│   │   │   └── transformer.py      # TransformerEncoderModel (primary, autoregressive)
│   │   ├── losses/
│   │   │   ├── direct_loss.py      # DirectLoss + ChargeConsistencyLoss
│   │   │   └── bni_mae.py          # WeightedBNILoss + MAELoss + compute_lds_weights_per_target
│   │   ├── training/
│   │   │   ├── early_stopping.py
│   │   │   └── trainer.py          # train_directnet, train_transformer, per-epoch helpers
│   │   ├── eval/
│   │   │   ├── metrics.py          # compute_physical_metrics, print_metrics
│   │   │   └── visualization.py    # plot_scatter_comparison, plot_loss_curves
│   │   ├── utils/seed.py
│   │   └── cli/train.py            # Unified CLI: `python -m bsimar.cli.train --model {direct,transformer} ...`
│   ├── checkpoints/    # Saved model weights for both architectures (.pt + _norm.npz + _config.npz) — gitignored
│   ├── results/        # Training plots (scatter, loss curves) — gitignored
│   ├── docs/           # Reference paper and ablation notes
│   ├── imgs/           # README imagery
│   ├── README.md
│   └── LICENSE
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
│       └── TSMC{5,7,12,16}/naive/           # Pre-baked TSMC naive modelcards
main.py                 # CLI entry point (single main entrance)
examples/*.sp           # Example netlists
results/                # Simulation output (.lis, .csv, .png)
tests/
├── test_common.py                   # Shared infrastructure: TechProfile, VtPair, ALL_TECHS, generic helpers
├── bsimcmg_tran_common.py           # Transient-specific: TestConfig, runners, metrics, plots
├── bsimcmg_dc_common.py             # DC-specific: DCTestConfig, runners, metrics, plots
└── verify_*.py                      # 3-level test scripts (L1/L2/L3 for DC + transient)
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
  - NMOS/PMOS Level 72 (BSIM-CMG FinFET via PyCMG)
  - NMOS/PMOS Level 73 (NN-based compact model via PyTorch)
* Sources: DC voltage/current, PULSE

### Analysis
* `.op` - Operating Point Analysis (Basic DC solution)
* `.dc` - DC Sweep Analysis
* `.tran` - Transient Analysis

### Directives
* `.model` - MOSFET model definitions (LEVEL=72 or LEVEL=73)
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
- **3-level DC+Transient test suites** — 3-layer infrastructure: `test_common.py` (tech defs, generic helpers) -> `bsimcmg_{dc,tran}_common.py` (analysis-specific) -> `verify_*.py` (test scripts). All 223 tests PASS (0 FAIL, 0 ERROR): DC 113 (L1:2 + L2:67 + L3:44), Transient 110 (L1:1 + L2:37 + L3:72).
  - **Known-bad combos excluded:** TSMC5 SVT (pch PDIBL2_i<0), TSMC7 SVT/LVT (inverter garbage / pch PDIBL2_i<0), TSMC16 LNVT (nch PDIBL2_i<0), TSMC16 L=24nm (PDIBL2_i<0), NFIN=1 (NR divergence for tsmc5:ulvt / tsmc16:lnvt — ETA0_i/U0_i go negative, internal node drifts to 40V producing id=40kA + NaN derivatives; eval_dc now raises RuntimeError), P/N ratio where NFIN_P crosses NFIN group boundary (TSMC naive modelcards are NFIN-group-specific).
- **PyCMG data generation migration** — NN training data generation moved from the old `nn_model.data.generate` into `external_compact_models/PyCMG/scripts/generate_nn_data.py`. Geometry format changed to 15-col `[NFIN, L, T, 12 process params]` (was 14-col). NN input dimension is now 19 features (was 18). Legal (L, NFIN) combos come from PDK bin boundaries (TSMC) or fallback list (ASAP7); process params extracted on-the-fly from modelcards (per-bin accurate). 954 total geometry combos across 5 techs, 21 variants. Existing checkpoints require retraining with the new data format.
- **BSIMAR package refactor** — Consolidated the former `nn_model/` (DirectNet baseline) and `external_compact_models/BSIMAR/script/` (Transformer) into a single Python package at `external_compact_models/BSIMAR/bsimar/` with clean subpackages (`config`, `data`, `models`, `losses`, `training`, `eval`, `utils`, `cli`). DirectNet is now explicitly a baseline for comparison against the BSIM-AR Transformer. Unified CLI: `python -m bsimar.cli.train --model {direct,transformer} ...`. All downstream imports (pycircuitsim parser, mosfet_nn, mosfet_bsimar, tests/verify_nn_*) updated to the new `bsimar.*` namespace.
- **BSIM-AR Quick Smoke Test (NMOS, 50 epochs, 67K params)** — End-to-end pipeline verified on universal_nmos.npz (951K samples, input_dim=18). Three runs trained successfully without crashes:
  - **zscore + MAE+LDS:** best val loss 0.0606 @ epoch 32, AVG NRMSE 1.43%, MRE 12.79%, R2 0.968
  - **zscore + MAE (no LDS):** best val loss 0.0613 @ epoch 41, identical test metrics (shared checkpoint path)
  - **signedlog + MAE:** best val loss 0.1032 @ epoch 28, AVG NRMSE 7.14%, MRE 170.44%, **R2 -5.99 (collapsed)**
  - **Key finding:** zscore dominates signedlog in physical-space metrics on every target. Signedlog R2_norm=0.91 looks fine but `inv_signed_log` denormalization amplifies AR-accumulated errors catastrophically (gds: R2_norm=0.81 → R2_phys=-74.09). This is a fundamental mismatch between log-compressed normalization and AR token-by-token error accumulation.
  - **Bottleneck:** ~170-190s/epoch, dominated by sequential 13-step autoregressive validation on 60-95K samples (cannot be parallelized with current `forward()` impl).
  - **Bug found:** concurrent runs sharing the same `--exp-name` overwrite each other's `_best.pt` checkpoints. Always use distinct exp-names for parallel experiments.

### Future Work
- [ ] **Improve TSMC5 Transient** — 14.41% NRMSE (close to 15% threshold); try denser mid-supply data (`--n-dense-mid 30`) + retrain
- [ ] **SRAM Validation (Phase 4)** — 6T bitcell DC+transient, 8-cell column, 64-bit array benchmark vs NGSPICE
- [ ] **Adaptive Output Timestep** — Variable-length output array with true adaptive dt (full adaptive requires output grid changes)
- [ ] **BSIM-AR — Default to zscore+MAE+LDS** — Document this as the recommended config in BSIMAR/README.md; deprecate signedlog mode for AR training (it remains valid for non-AR DirectNet).
- [ ] **BSIM-AR — Speed up AR validation** — Implement KV-cache for the Transformer encoder so the 13 sequential steps reuse cached attention. Expected ~5-10x speedup on validation/test (currently 170-190s/epoch).
- [ ] **BSIM-AR — Larger production model** — After the smoke test confirms the pipeline, retrain with paper-scale config (d_model=256, layers=6, ff=1024, ~110K params) for 500 epochs to get apples-to-apples comparison vs paper Table.
- [ ] **BSIM-AR — Auto-disambiguate exp-names** — Add a guard in `main.py` that aborts (or appends a timestamp) if `<exp_name>_best.pt` already exists, to prevent silent checkpoint clobbering across parallel runs.
- [ ] **BSIM-AR — Validate on PMOS + per-tech** — Smoke test only covered universal NMOS; need PMOS run + at least one per-tech run to confirm the pipeline handles all data shapes.

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

### NN Model (LEVEL=73 DirectNet baseline + LEVEL=74 BSIM-AR Transformer)
```bash
# Generate universal data (PyCMG is the canonical data generator)
# Uses PDK-defined (L, NFIN) bins (TSMC bin boundaries, ASAP7 fallback list)
# and extracts process parameters on-the-fly from modelcards (per-bin accurate).
# 954 total geometry combos across 5 techs and 21 variants.
conda run -n pycircuitsim python external_compact_models/PyCMG/scripts/generate_nn_data.py \
    --device both --universal

# Train DirectNet (baseline) via unified CLI
conda run -n pycircuitsim python -u -m bsimar.cli.train \
    --model direct --device-type nmos --universal --mode direct13 \
    --epochs 800 --hidden 384 --layers 6 --patience 150 --batch-size 2048 --cuda
conda run -n pycircuitsim python -u -m bsimar.cli.train \
    --model direct --device-type pmos --universal --mode direct13 \
    --epochs 800 --hidden 384 --layers 6 --patience 150 --batch-size 2048 --cuda

# Optional: charge-finetune for transient accuracy (autograd dq/dV = C, ~5-10x slower)
conda run -n pycircuitsim python -u -m bsimar.cli.train \
    --model direct --device-type nmos --universal --mode charge-finetune \
    --epochs 800 --hidden 384 --layers 6 --patience 150 --batch-size 2048 --cuda --resume none

# Train BSIM-AR Transformer (primary, autoregressive) — paper recommended config
conda run -n pycircuitsim python -u -m bsimar.cli.train \
    --model transformer --device-type nmos --universal --loss mae --lds --cuda

# Note: `bsimar` is importable because consumers add
#       `external_compact_models/BSIMAR` to sys.path. Checkpoints live under
#       `external_compact_models/BSIMAR/checkpoints/` for both models.
```
Checkpoints (under `external_compact_models/BSIMAR/checkpoints/`):
- DirectNet: `universal_{nmos,pmos}_best.pt` + `_norm.npz` (universal); `{tech}_{nmos,pmos}_best.pt` (per-tech).
- Transformer: `ar_universal_{nmos,pmos}_best.pt` + `_norm.npz` + `_config.npz`.

Netlist usage:
- LEVEL=73 (DirectNet): `.model nmos_nn NMOS (LEVEL=73 TECH=tsmc5 VT=lvt)` with `L=16n NFIN=10`.
- LEVEL=74 (BSIM-AR Transformer): `.model nmos_ar NMOS (LEVEL=74 TECH=tsmc5 VT=lvt)` with `L=16n NFIN=10`.
- Parser auto-resolves process params from TECH+VT and prefers universal checkpoint when available.
- Direct process params: `.model nmos_nn NMOS (LEVEL=73 PHIG=4.41 U0=0.033 VSAT=65370 EOT=1.06e-9 ETA0=0.005 CIT=-9.81e-4 RDSW=15)`.

### Output Files
Results organized in `results/<circuit_name>/<analysis_type>/`:
- `*_simulation.lis` - Detailed iteration log (HSPICE-like)
- `*_dc_sweep.csv` / `*_transient.csv` - Waveform data (node voltages + device currents)

## Testing & Verification

All tests require: `conda activate pycircuitsim`

**BSIM-CMG DC Verification (3-level, shared infra in `test_common.py` + `bsimcmg_dc_common.py`):**

| Level | Script | Tests | What it tests |
|-------|--------|-------|---------------|
| 1 | `verify_bsimcmg_dc.py` | 2 | NMOS + PMOS Id-Vgs (ASAP7, <1% NRMSE) |
| 2 | `verify_bsimcmg_dc_comprehensive.py` | 67 | VT/L/NFIN sweeps, NMOS+PMOS, 5 techs |
| 3 | `verify_multi_tech_dc.py` | 44 | Inverter VTC + parametric (VT, L, NFIN, P/N) |

**BSIM-CMG Transient Verification (3-level, shared infra in `test_common.py` + `bsimcmg_tran_common.py`):**

| Level | Script | Tests | What it tests |
|-------|--------|-------|---------------|
| 1 | `verify_bsimcmg_tran.py` | 1 | Inverter pulse (ASAP7, <5% NRMSE) |
| 2 | `verify_bsimcmg_tran_comprehensive.py` | 37 | VT/L/NFIN sweeps, 5 techs |
| 3 | `verify_multi_tech_tran.py` | 72 | Multi-tech + parametric (P/N, VDD, Cload, slew, pw) |

**Other:**

| Test Suite | Script | What it tests |
|-----------|--------|---------------|
| OP Verification | `verify_bsimcmg_op.py` | NMOS, PMOS, Inverter OP vs NGSPICE (<0.02%) |
| NN Multi-Tech | `verify_nn_multi_tech.py` | NMOS/PMOS DC + Inverter VTC per tech (<10%/15%) |
| NN Universal v2 | `verify_nn_universal_v2.py` | 21 variants x 3 tests (63 tests) |
| NN Transient | `verify_nn_tran.py` | NN vs NGSPICE transient per tech (<15%) |
| NN Leave-One-Out | `verify_nn_leave_one_out.py` | Zero-shot transferability experiment |

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
4. **Solver stamping** uses unified "current leaving drain" convention:
   ```python
   i_leaving = -i_ds if is_pmos else i_ds
   i_eq = i_leaving - g_ds * v_ds - g_m * v_gs - g_mb * v_bs
   rhs[d_idx] -= i_eq  # Same for NMOS and PMOS
   rhs[s_idx] += i_eq
   ```
5. **Conductance signs**: `abs(gds)` always (can be negative at extremes), but preserve gm/gmb signs
6. **Update `_is_mosfet()`** in `solver.py` when adding new device types
7. **Test both NMOS and PMOS** against NGSPICE: single OP, DC sweep, inverter VTC, inverter transient

### NN Model Rules (LEVEL=73)

1. **Jacobian consistency is mandatory** — gm/gds MUST be `torch.autograd.grad(id, V)`, never independent predictions. Without this, NR diverges in multi-device circuits.
2. **PMOS source-relative frame** — Shift all voltages by -Vs before NN eval (`v_d_nn = v_d - v_s`). Training uses Vs=0; in CMOS, PMOS Vs=VDD.
3. **Training range covers NR overshoot** — Margin of +/-VDD beyond operating range, not just +/-0.1V
4. **Voltage clamping** — Clip inputs to training range to prevent extrapolation garbage
5. **Signed-log normalization** — `sign(x) * log10(|x|/floor)` preserves sign across 14-decade range
6. **TSMC asymmetric L** — NMOS L=16nm, PMOS L=20nm; NNTechConfig uses `L_nmos`/`L_pmos`
7. **ASAP7 modelcard name mapping** — Parser auto-maps netlist names to `nmos_rvt`/`pmos_rvt`
8. **PyCMG integration** — `bsimar/config.py` re-exports the NN config from PyCMG's `pycmg.nn_config` (TECH_CONFIGS, ProcessParams, extract_process_params, OUTPUT_COLUMNS, INPUT_COLUMNS). Training VDD may differ from PyCMG (e.g., ASAP7: train=0.7V, PyCMG=0.9V). Backward-compat alias `TechConfig = NNTechConfig` exists for test files.
9. **Data generation validates PyCMG output** — `eval_single_point` rejects NaN/Inf and `|id| > 1A`. PyCMG `eval_dc` raises `RuntimeError` on internal-node convergence failure. Default NFIN range is `[2, 3, 5, 10, 15, 20, 24]` (NFIN=1 excluded due to OSDI convergence failures).

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
- **Modelcards**: `modelcards/` (relative to PyCMG root; ASAP7: `ASAP7/`, TSMC: `TSMC{5,7,12,16}/naive/`)
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
