# PyCircuitSim

<div align="center">

**Pure Python Circuit Simulator with Compact-Model Research Bench**

A clean, readable SPICE-like circuit simulator with three coexisting
compact-model families that share the same solver:

- **BSIM-CMG** (LEVEL=72) via PyCMG/OSDI — production-grade FinFET ground truth.
- **DirectNet** (LEVEL=73) — PyTorch MLP baseline.
- **BSIM-AR Transformer** (LEVEL=74) — autoregressive compact model that
  predicts I-V, Q-V, and C-V curves token-by-token.

DirectNet and BSIM-AR live side-by-side in the unified `bsimar` package
(`external_compact_models/bsimar/`) and share data generation,
normalization, losses, training, metrics, and visualization. DirectNet
acts as the baseline for BSIM-AR research.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Netlist Syntax](#netlist-syntax)
- [Examples](#examples)
- [Python API](#python-api)
- [Output Files](#output-files)
- [NN Compact Models (LEVEL=73 & LEVEL=74)](#nn-compact-models-level73--level74)
- [Verification](#verification)
- [Architecture](#architecture)
- [Algorithms](#algorithms)
- [Development](#development)
- [Limitations](#limitations)
- [References](#references)

---

## Features

### Supported Components

| Component | Symbol | Description |
|-----------|--------|-------------|
| Resistor | `R` | Linear resistance |
| Capacitor | `C` | Linear capacitance |
| Voltage Source | `V` | DC or PULSE waveform |
| Current Source | `I` | DC current source |
| NMOS/PMOS Level 72 | `M` | BSIM-CMG FinFET via PyCMG/OSDI |
| NMOS/PMOS Level 73 | `M` | DirectNet (MLP) compact model via PyTorch — baseline |
| NMOS/PMOS Level 74 | `M` | BSIM-AR Transformer compact model via PyTorch — primary |

The legacy Shichman-Hodges `LEVEL=1` model has been removed. Only
LEVEL=72/73/74 are supported.

### Supported Analyses

- **DC Operating Point** (`op`) - Single-point bias calculation
- **DC Sweep** (`.dc`) - Parameter sweep analysis
- **Transient Analysis** (`.tran`) - Time-domain simulation

### Supported Directives

- `.model` - MOSFET model definitions (LEVEL=72, LEVEL=73, or LEVEL=74)
- `.include` - Include external library files
- `.ic` - Set initial node voltages

---

## Installation

### Prerequisites

- Python 3.10+
- Conda (recommended for environment management)

### Setting Up the Environment

```bash
# Clone the repository with submodules
git clone --recurse-submodules https://github.com/ShenShan123/PyCircuitSim.git
cd PyCircuitSim

# Create and activate the conda environment
conda create -n pycircuitsim python=3.10
conda activate pycircuitsim

# Install Python dependencies (requirements.txt pins all solver + NN deps)
pip install -r requirements.txt

# PyTorch is required for LEVEL=73 / LEVEL=74 inference and for training
# the bsimar compact models. Install CPU or CUDA build as appropriate:
pip install torch
```

### BSIM-CMG Setup

To use BSIM-CMG FinFET models (LEVEL=72), the PyCMG submodule and a compiled OSDI binary are required:

```bash
# Initialize the PyCMG submodule (if not cloned with --recurse-submodules)
git submodule update --init --recursive

# Verify the OSDI binary exists
ls external_compact_models/PyCMG/build/osdi/bsimcmg.osdi
```

Technology modelcards live under `external_compact_models/PyCMG/modelcards/`:

- `ASAP7/` — committed ASAP7 7nm predictive PDK.
- `TSMC{5,7,12,16}/cln*.l` — raw TSMC PDK files. These are IP-protected and
  gitignored; supply them yourself to run any TSMC verification. Naive
  modelcards used by NGSPICE tests are regenerated on-the-fly from the raw
  PDK via `pycmg.tech.resolve_modelcard` and cached under
  `external_compact_models/PyCMG/build/modelcards/`.

### NN Compact Model Setup (LEVEL=73 / LEVEL=74)

The `bsimar` package at `external_compact_models/bsimar/` is importable
once `external_compact_models/` is on `sys.path`. The `pycircuitsim`
parser and the test harness add this automatically. Checkpoints live at
`external_compact_models/bsimar/checkpoints/` and are resolved by the
parser at netlist load time. See [NN Compact Models](#nn-compact-models-level73--level74)
for training and inference details.

---

## Quick Start

### Running a Simulation

```bash
# Activate the environment
conda activate pycircuitsim

# Run a BSIM-CMG (LEVEL=72) inverter transient simulation
python main.py examples/bsimcmg_inverter_tran.sp

# Run a BSIM-CMG NMOS DC sweep
python main.py examples/bsimcmg_nmos_dc.sp

# Run a DirectNet (LEVEL=73) NMOS DC sweep (requires a trained checkpoint)
python main.py examples/nn_nmos_dc.sp

# Run a BSIM-AR (LEVEL=74) inverter DC sweep
python main.py examples/bsimar_inverter_dc.sp

# Specify a custom output directory
python main.py examples/bsimcmg_inverter_tran.sp -o my_results

# Enable verbose logging (shows Newton-Raphson iterations)
python main.py examples/bsimcmg_inverter_tran.sp -v
```

### CLI Options

```
usage: main.py [-h] [-o OUTPUT_DIR] [-v] netlist

positional arguments:
  netlist               Path to the HSPICE-format netlist file

options:
  -o, --output DIR      Output directory for results (default: results)
  -v, --verbose         Enable verbose logging output
```

### Output Location

Results are saved to `results/<circuit_name>/<analysis_type>/` by default:

```
results/
└── bsimcmg_inverter_tran/
    └── tran/
        ├── bsimcmg_inverter_tran_simulation.lis   # Iteration log
        ├── bsimcmg_inverter_tran_transient.csv     # Waveform data
        └── bsimcmg_inverter_tran_transient.png     # Plot
```

---

## Netlist Syntax

PyCircuitSim supports HSPICE-like netlist format.

### Component Syntax

```
* Resistor: R<name> <node+> <node-> <value>
R1 1 2 1k
R2 2 0 10k

* Capacitor: C<name> <node+> <node-> <value>
C1 2 0 100p

* Voltage Source: V<name> <node+> <node-> <value>
Vdd 1 0 3.3

* Current Source: I<name> <node+> <node-> <value>
Ibias 1 0 1m

* MOSFET: M<name> <drain> <gate> <source> <bulk> <model> L=<len> W=<width>
Mn1 3 2 0 0 NMOS_VTL L=1u W=10u
Mp1 3 2 1 1 PMOS_VTL L=1u W=20u
```

### Value Suffixes

| Suffix | Multiplier | Example |
|--------|-----------|---------|
| `T` | 10^12 | `1T` = 1,000,000,000,000 |
| `G` | 10^9 | `1G` = 1,000,000,000 |
| `M` (uppercase) | 10^6 | `1M` = 1,000,000 |
| `k`, `K` | 10^3 | `1k` = 1,000 |
| `m` (lowercase) | 10^-3 | `1m` = 0.001 |
| `u`, `U` | 10^-6 | `1u` = 0.000001 |
| `n`, `N` | 10^-9 | `1n` = 0.000000001 |
| `p`, `P` | 10^-12 | `1p` = 10^-12 |
| `f`, `F` | 10^-15 | `1f` = 10^-15 |

### Analysis Commands

```spice
* DC Operating Point
.op

* DC Sweep: .dc <source> <start> <stop> <step>
.dc Vin 0 3.3 0.1

* Transient: .tran <tstep> <tstop>
.tran 10p 5n
```

### MOSFET Models

#### Level 72 (BSIM-CMG FinFET)

```spice
* Model declaration — device parameters come from the ASAP7 modelcard
.model nmos1 NMOS (LEVEL=72)
.model pmos1 PMOS (LEVEL=72)

* Instance parameters are specified on the device line
Mn1 out in 0 0 nmos1 L=30n NFIN=10
Mp1 out in vdd vdd pmos1 L=30n NFIN=10
```

**BSIM-CMG geometric parameters:**

| Parameter | Description | Notes |
|-----------|-------------|-------|
| `L` | Channel length | Required (e.g., `30n`) |
| `NFIN` | Number of fins | Required (integer or float) |
| `TFIN` | Fin thickness | Optional (uses modelcard default) |
| `HFIN` | Fin height | Optional (uses modelcard default) |
| `FPITCH` | Fin pitch | Optional (uses modelcard default) |

#### Level 73 (DirectNet — MLP baseline compact model)

```spice
* Auto-resolve process params from a technology + threshold variant
.model nmos_nn NMOS (LEVEL=73 TECH=tsmc5 VT=lvt)
.model pmos_nn PMOS (LEVEL=73 TECH=tsmc5 VT=lvt)

* Alternatively, supply process params directly on the .model line
.model nmos_nn_direct NMOS (LEVEL=73
    PHIG=4.41 U0=0.033 VSAT=65370 EOT=1.06e-9
    ETA0=0.005 CIT=-9.81e-4 RDSW=15)

* Device instances use the same L / NFIN syntax as LEVEL=72
Mn1 out in 0 0 nmos_nn L=16n NFIN=10
Mp1 out in vdd vdd pmos_nn L=16n NFIN=10
```

Checkpoints are resolved automatically:
`external_compact_models/bsimar/checkpoints/universal_{nmos,pmos}_best.pt`
(preferred if present) or per-tech `{tech}_{nmos,pmos}_best.pt`.

#### Level 74 (BSIM-AR — autoregressive Transformer compact model)

```spice
* Same netlist syntax as LEVEL=73, just a different LEVEL
.model nmos_ar NMOS (LEVEL=74 TECH=tsmc5 VT=lvt)
.model pmos_ar PMOS (LEVEL=74 TECH=tsmc5 VT=lvt)

Mn1 out in 0 0 nmos_ar L=16n NFIN=10
Mp1 out in vdd vdd pmos_ar L=16n NFIN=10
```

BSIM-AR checkpoints: `ar_universal_{nmos,pmos}_best.pt` + `_norm.npz` +
`_config.npz` under `external_compact_models/bsimar/checkpoints/`.

Both LEVEL=73 and LEVEL=74 expose autograd-derived conductances (gm, gds,
gmb) so Newton-Raphson stays consistent in multi-device circuits. See
[NN Compact Models](#nn-compact-models-level73--level74) for training,
checkpoint layout, and inference trade-offs.

### PULSE Sources

```spice
* PULSE: V<name> <n+> <n-> PULSE(V1 V2 TD TR TF PW PER)
Vclk 1 0 PULSE(0 3.3 0n 1n 1n 10n 20n)

* Parameters:
* V1  : Initial value (V)
* V2  : Pulsed value (V)
* TD  : Delay time
* TR  : Rise time
* TF  : Fall time
* PW  : Pulse width
* PER : Period
```

### Initial Conditions

```spice
* Set initial node voltage (useful for bistable circuits)
.ic V(out)=0.7
```

---

## Examples

All example netlists live in `examples/`. The files below cover the three
compact-model families plus a passive-only RC reference:

| File | Analysis | Models used | What it demonstrates |
|------|----------|-------------|----------------------|
| `examples/bsimcmg_nmos_dc.sp` | DC sweep | LEVEL=72 | NMOS Id-Vgs against PyCMG/OSDI |
| `examples/bsimcmg_pmos_dc.sp` | DC sweep | LEVEL=72 | PMOS Id-Vgs against PyCMG/OSDI |
| `examples/bsimcmg_inverter_dc.sp` | DC sweep | LEVEL=72 | Inverter VTC |
| `examples/bsimcmg_inverter_tran.sp` | Transient | LEVEL=72 | FinFET inverter pulse response |
| `examples/bsimcmg_inverter_dc_asap7_ref.sp` | DC sweep | LEVEL=72 | ASAP7 reference configuration |
| `examples/nn_nmos_op.sp` | OP | LEVEL=73 | DirectNet single-point NMOS |
| `examples/nn_nmos_dc.sp` | DC sweep | LEVEL=73 | DirectNet NMOS Id-Vgs |
| `examples/nn_inverter_dc.sp` | DC sweep | LEVEL=73 | DirectNet inverter VTC |
| `examples/bsimar_nmos_dc.sp` | DC sweep | LEVEL=74 | BSIM-AR NMOS Id-Vgs |
| `examples/bsimar_inverter_dc.sp` | DC sweep | LEVEL=74 | BSIM-AR inverter VTC |
| `examples/rc_transient.sp` | Transient | passives | Pure RC reference |

### Sample: BSIM-CMG FinFET Inverter Transient (ASAP7 7nm)

```spice
* examples/bsimcmg_inverter_tran.sp
Vdd 1 0 0.7
Vin 2 0 PULSE 0 0.7 0.5n 0.1n 0.1n 0.8n 2n

Mp1 3 2 1 1 pmos1 L=30n NFIN=10
Mn1 3 2 0 0 nmos1 L=30n NFIN=10
Cload 3 0 10f

.ic V(3)=0.7
.model nmos1 NMOS (LEVEL=72)
.model pmos1 PMOS (LEVEL=72)
.tran 10p 5n
.end
```

Run any example with `python main.py examples/<file>.sp`.

---

## Python API

```python
# High-level: parse + solve + plot in one call
from pycircuitsim.simulation import run_simulation
run_simulation('examples/bsimcmg_inverter_tran.sp',
               output_dir='my_results', verbose=True)

# Low-level: drive the solver directly
from pycircuitsim import Parser
from pycircuitsim.solver import DCSolver, TransientSolver

parser = Parser()
parser.parse_file('examples/bsimcmg_inverter_tran.sp')
circuit = parser.circuit

dc_solution = DCSolver(circuit).solve()
time_points, waveforms = TransientSolver(circuit).solve(
    tstep=10e-12, tstop=5e-9, dc_solution=dc_solution,
)
```

`pycircuitsim.simulation` also exposes `run_dc_sweep()` and
`run_transient()` for the full orchestrated workflow.

---

## Output Files

PyCircuitSim generates output files organized by circuit name and analysis type:

```
results/
└── <circuit_name>/
    ├── dc/
    │   ├── <circuit>_simulation.lis      # Detailed iteration log
    │   ├── <circuit>_dc_sweep.csv        # Numerical waveform data
    │   └── <circuit>_dc_sweep.png        # Voltage/current plots
    └── tran/
        ├── <circuit>_simulation.lis
        ├── <circuit>_transient.csv
        └── <circuit>_transient.png
```

### Log Files (.lis)

HSPICE-like detailed logs showing:
- Circuit summary (component count, node count)
- Newton-Raphson iterations per step
- Convergence status and iteration count
- Final node voltages and device currents

### CSV Data Files

Column-oriented waveform data, importable into Excel, MATLAB, or Python:

```csv
Vin (V),V(1),V(2),V(3),i(Vdd),i(Vin)
0.000000,3.300000e+00,0.000000e+00,3.299967e+00,...
0.100000,3.300000e+00,1.000000e-01,3.299934e+00,...
```

---

## NN Compact Models (LEVEL=73 & LEVEL=74)

Both NN compact-model families live in the unified `bsimar` package at
`external_compact_models/bsimar/`. DirectNet is the baseline; BSIM-AR
is the primary Transformer model; they share every layer below the
model architecture itself (data, normalization, losses, training, eval).
See [Architecture](#architecture) for the package layout.

### Data Generation

Data is produced by PyCMG (the ground-truth BSIM-CMG model):

```bash
conda run -n pycircuitsim python \
    external_compact_models/PyCMG/scripts/generate_nn_data.py \
    --device both --universal
```

This walks 954 `(L, NFIN)` combinations across 5 techs and 21 threshold
variants (legal bin boundaries for TSMC, a fallback list for ASAP7),
extracts process parameters on-the-fly from the modelcards, and writes
one `.npz` per device under
`external_compact_models/bsimar/data/datasets/`. Each sample is a
`19`-feature input (`Vd, Vg, Vs, Vb, log2(NFIN), L, T, <12 process params>`)
and a `13`-column output (`id, gm, gds, gmb, qg, qd, qs, qb, cgg, cgd, cgs, cdg, cdd`).

### Training

Training is driven by a single CLI. The `--model` flag picks the architecture;
every other flag is shared between the two:

```bash
# DirectNet (baseline MLP, ~2 s/epoch on a modern GPU)
conda run -n pycircuitsim python -u -m bsimar.cli.train \
    --model direct --device-type nmos --universal --mode direct13 \
    --epochs 800 --hidden 384 --layers 6 --patience 150 --batch-size 2048 --cuda

# DirectNet charge-finetune (autograd dq/dV = C consistency, ~5-10x slower,
# improves transient accuracy)
conda run -n pycircuitsim python -u -m bsimar.cli.train \
    --model direct --device-type nmos --universal --mode charge-finetune \
    --epochs 800 --hidden 384 --layers 6 --patience 150 --batch-size 2048 \
    --cuda --resume none

# BSIM-AR Transformer (paper recommended: zscore + MAE + LDS)
conda run -n pycircuitsim python -u -m bsimar.cli.train \
    --model transformer --device-type nmos --universal \
    --loss mae --lds --cuda
```

Checkpoints land under `external_compact_models/bsimar/checkpoints/`:

| File | Model | Notes |
|------|-------|-------|
| `universal_{nmos,pmos}_best.pt` + `_norm.npz` | DirectNet universal | Preferred by the parser |
| `{tech}_{nmos,pmos}_best.pt` + `_norm.npz` | DirectNet per-tech | Fallback |
| `ar_universal_{nmos,pmos}_best.pt` + `_norm.npz` + `_config.npz` | BSIM-AR universal | Transformer |

### Inference (LEVEL=73 vs LEVEL=74)

Both LEVELs are drop-in replacements for LEVEL=72 on the same netlist.
They share the same sign conventions, the same Jacobian-via-autograd
guarantee for Newton-Raphson consistency, and the same source-relative
voltage frame for PMOS.

Key differences:

| Aspect | LEVEL=73 (DirectNet) | LEVEL=74 (BSIM-AR) |
|--------|----------------------|---------------------|
| Architecture | MLP (SiLU, 5-6 layers) | Transformer encoder with causal mask |
| Forward pass | 1 per device eval | 13 sequential tokens per device eval |
| Training time | Fast | Slower (~5-10x) |
| Inference cost | ~1x | ~13x |
| Role | Baseline | Primary research model |
| Normalization | Signed-log + z-score (default) | z-score (default) or signed-log (optional) |

---

## Verification

All BSIM-CMG results are validated against NGSPICE 45.2 with the
BSIM-CMG OSDI binary. NN compact models (LEVEL=73/74) are validated
against PyCMG/NGSPICE as the ground truth.

### Test Harness Layout

```
tests/
├── __init__.py
├── common/                        # Shared test infrastructure (subpackage)
│   ├── base.py                    # PROJECT_ROOT, OSDI_PATH, TechProfile, VtPair, NGSPICE runner
│   ├── bsimcmg_dc.py              # DC-specific runners, metrics, plots
│   ├── bsimcmg_tran.py            # Transient-specific runners, metrics, plots
│   └── nn.py                      # NN helpers (nrmse, mre, checkpoint resolution, path bootstrap)
├── references/                    # NGSPICE reference netlists (ngspice_*.cir)
└── verify_*.py                    # Flat verification scripts
```

### Running Verification

```bash
conda activate pycircuitsim

# Operating point verification (NMOS, PMOS, Inverter)
python tests/verify_bsimcmg_op.py

# DC sweep verification (Id-Vgs)
python tests/verify_bsimcmg_dc.py

# Transient verification (single baseline config)
python tests/verify_bsimcmg_tran.py

# Comprehensive transient verification (21 parametric configs)
python tests/verify_bsimcmg_tran_comprehensive.py

# Run L1 smoke suite in one line
python tests/verify_bsimcmg_op.py && \
python tests/verify_bsimcmg_dc.py && \
python tests/verify_bsimcmg_tran.py
```

### Verification Results

#### Operating Point

| Test | Metric | Result |
|------|--------|--------|
| NMOS OP (Vgs=0.7V, Vds=0.5V) | Relative error | 0.00% |
| PMOS OP (Vgs=-0.7V, Vds=-0.5V) | Relative error | 0.01% |
| Inverter OP (Vin=0V) | Relative error | 0.00% |
| Inverter OP (Vin=0.7V) | Relative error | 0.00% |

#### DC Sweep

| Test | Metric | Result |
|------|--------|--------|
| NMOS Id-Vgs (Vds=0.5V, Vgs=0-0.7V) | NRMSE | 0.010% |
| PMOS Id-Vgs (Vds=-0.5V, Vgs=0 to -0.7V) | NRMSE | 0.014% |
| Inverter VTC (Vin=0-0.7V) | NRMSE | 0.002% |

#### Transient (Baseline)

| Metric | Value |
|--------|-------|
| NRMSE (post-settling) | 0.20% |
| NRMSE (full-range) | 0.26% |
| Max absolute error | 7.6 mV (1.1% of Vdd) |

#### Comprehensive Transient (21 Configurations)

Sweeps VDD (0.5-0.8 V), Cload (1-100 fF), input slew (10-500 ps),
pulse width (0.2-2.0 ns), NFIN scaling (1-20), and P/N ratio (0.5-2.0).
**All 21 configs PASS (NRMSE < 5%); worst case 0.84% NRMSE / 42 mV at
Cload=1fF.** Representative rows:

| Config | VDD | Cload | NRMSE(%) | MaxErr(mV) |
|--------|-----|-------|----------|------------|
| baseline | 0.70V | 10fF | 0.19 | 7.6 |
| vdd_0p5 | 0.50V | 10fF | 0.14 | 4.7 |
| vdd_0p8 | 0.80V | 10fF | 0.21 | 12.9 |
| cload_1fF (worst) | 0.70V | 1fF | 0.84 | 42.0 |
| cload_100fF | 0.70V | 100fF | 0.02 | 0.9 |
| nfin_20 | 0.70V | 10fF | 0.37 | 20.8 |

#### NN Transient (LEVEL=73, 5 Technologies)

| Tech | VDD | NRMSE(%) | MaxErr(mV) | Status |
|------|-----|----------|------------|--------|
| ASAP7 | 0.70V | 6.29 | 268.3 | PASS |
| TSMC5 | 0.65V | 14.41 | 499.7 | PASS |
| TSMC7 | 0.75V | 6.09 | 396.4 | PASS |
| TSMC12 | 0.80V | 5.92 | 311.7 | PASS |
| TSMC16 | 0.80V | 6.70 | 364.2 | PASS |

### Verification Scripts

| Script | Purpose |
|--------|---------|
| `tests/verify_bsimcmg_op.py` | OP analysis: PyCircuitSim vs NGSPICE for NMOS, PMOS, inverter |
| `tests/verify_bsimcmg_dc.py` | DC sweep L1: Id-Vgs (ASAP7 baseline) |
| `tests/verify_bsimcmg_dc_comprehensive.py` | DC sweep L2: 67-config VT/L/NFIN sweep across 5 techs |
| `tests/verify_multi_tech_dc.py` | DC sweep L3: 44-config inverter VTC + parametric |
| `tests/verify_bsimcmg_tran.py` | Transient L1: single inverter baseline |
| `tests/verify_bsimcmg_tran_comprehensive.py` | Transient L2: 37-config VT/L/NFIN sweep |
| `tests/verify_multi_tech_tran.py` | Transient L3: 72-config multi-tech parametric |
| `tests/verify_nn_tran.py` | NN transient: 5 technologies vs NGSPICE (<15% NRMSE) |
| `tests/verify_nn_universal_v2.py` | NN universal: 21 variants × 3 tests (DC + VTC) |
| `tests/verify_nn_leave_one_out.py` | NN zero-shot transferability experiment |

Each script generates comparison plots and detailed metrics in `tests/verify_*_results/`.

> **Note:** The `verify_nn_*.py` scripts still reference the old
> `tech.variants[v].get_process_params(device)` API that was removed
> when `nn_model.config` was folded into `bsimar.config`. They need to
> be ported to the new `NNTechConfig` API
> (`tech.resolve_modelcard(...)` + `extract_process_params(...)`)
> before they can run end-to-end. The BSIM-CMG scripts above are
> unaffected and exercise the refactored `tests/common/` subpackage.

---

## Architecture

PyCircuitSim follows a clean, modular architecture:

```
pycircuitsim/                       # Python package (simulator core)
├── __init__.py                     # Public API exports
├── config.py                       # Path configuration (OSDI binary, modelcards)
├── simulation.py                   # Simulation orchestration
├── parser.py                       # Netlist parser (HSPICE syntax)
├── circuit.py                      # Circuit topology (nodes, components)
├── solver.py                       # MNA + Newton-Raphson + Transient solvers
├── logger.py                       # HSPICE-like logging (.lis files)
├── visualizer.py                   # Matplotlib plotting
└── models/                         # Device model implementations
    ├── __init__.py
    ├── base.py                     # Component abstract base class
    ├── passive.py                  # R, C, V, I, PULSE sources
    ├── mosfet_cmg.py               # BSIM-CMG FinFET (LEVEL=72) via PyCMG/OSDI
    ├── mosfet_nn.py                # DirectNet (LEVEL=73) via PyTorch
    └── mosfet_bsimar.py            # BSIM-AR Transformer (LEVEL=74) via PyTorch

external_compact_models/            # External compact-model packages
├── bsimar/                         # Unified NN compact model package (DirectNet + BSIM-AR)
│   ├── config.py                   # TECH_CONFIGS + DirectNetConfig + TransformerConfig
│   ├── data/                       # Normalizers, dataset loaders, dataset analysis
│   ├── models/                     # direct_net.py + transformer.py
│   ├── losses/                     # direct_loss.py + bni_mae.py
│   ├── training/                   # train_directnet, train_transformer, EarlyStopping
│   ├── eval/                       # physical-units metrics, plots
│   ├── cli/train.py                # `python -m bsimar.cli.train --model {direct,transformer}`
│   ├── checkpoints/                # Trained weights (gitignored)
│   └── docs/, imgs/, README.md, LICENSE
│
└── PyCMG/                          # BSIM-CMG OSDI wrapper (git submodule)
    ├── pycmg/                      # Python ctypes-based OSDI interface
    ├── build/osdi/bsimcmg.osdi     # Compiled OSDI binary
    ├── modelcards/                 # Technology modelcards (ASAP7 + TSMC5/7/12/16)
    └── scripts/generate_nn_data.py # NN training-data generator

main.py                             # CLI entry point
examples/                           # Example netlists (.sp files)
tests/                              # NGSPICE verification scripts
├── common/                         # Shared test infrastructure
└── references/                     # NGSPICE reference netlists
results/                            # Simulation output (generated at runtime)
```

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `simulation.py` | Orchestrates parse -> solve -> visualize workflow |
| `parser.py` | Two-pass netlist parsing, `.model`/`.include`/`.ic` directives, LEVEL=73/74 process-param resolution via `bsimar.config` |
| `circuit.py` | Stores circuit topology, component list, node mapping |
| `solver.py` | MNA construction, Newton-Raphson iteration, transient stepping |
| `models/mosfet_cmg.py` | BSIM-CMG physics via PyCMG (LEVEL=72) |
| `models/mosfet_nn.py` | DirectNet inference with autograd conductances (LEVEL=73) |
| `models/mosfet_bsimar.py` | BSIM-AR Transformer inference (LEVEL=74), reuses `_MOSFETNNBase` |
| `logger.py` | HSPICE-compatible output formatting |
| `visualizer.py` | Automatic plot generation |
| `bsimar.*` | Training / eval pipeline shared by LEVEL=73 and LEVEL=74 |

### Design Principles

- **Separation of concerns** — solver builds matrices and iterates (no
  device equations); device models compute currents/conductances from
  voltages (no matrix ops); `simulation.py` orchestrates the workflow.
- **Modularity** — every device inherits from `Component` and exposes a
  common interface (`calculate_current()`, `get_conductances()`); new
  devices can be added without touching the solver.
- **Drop-in compact models** — LEVEL=72/73/74 share sign conventions
  and Jacobian-via-autograd contracts so the same netlist runs against
  any of the three.

---

## Algorithms

### Modified Nodal Analysis (MNA)

PyCircuitSim uses MNA to construct circuit equations:

```
[G  B] [v]   [i]
[    ] [ ] = [ ]
[C  D] [j]   [e]
```

- **G**: Conductance matrix (resistive elements + linearized MOSFETs)
- **B, C**: Voltage source incidence matrices
- **v**: Node voltages (unknowns)
- **j**: Voltage source currents (unknowns)
- **i, e**: Known current/voltage excitations

### Newton-Raphson Iteration

For non-linear circuits (MOSFETs):

1. Linearize devices at current operating point (compute gds, gm, gmb)
2. Construct MNA matrix with linearized conductances
3. Solve for voltage update dv
4. Apply adaptive damping: `v_new = v_old + alpha * dv`
5. Repeat until SPICE-standard convergence: `|dv| < VNTOL + RELTOL × max(|V_old|, |V_new|)` for all nodes (RELTOL=1e-4, VNTOL=1e-7)

### Source Stepping

Improves convergence for difficult circuits:

1. Start with all sources at 0V
2. Gradually ramp sources to final values (20 steps)
3. Use previous step's solution as initial guess

### BE→Trapezoidal Integration

For transient analysis with capacitors and MOSFET intrinsic capacitances:

```
Backward Euler (step 1):  i(t+dt) = (C/dt) * [v(t+dt) - v(t)]
Trapezoidal (step 2+):    i(t+dt) = (2C/dt) * [v(t+dt) - v(t)] - i(t)
```

- Backward Euler for first timestep avoids startup ringing (standard SPICE technique)
- 2nd order implicit Trapezoidal integration (A-stable) from step 2 onward
- Converts capacitors to companion conductance + current source
- Also stamps BSIM-CMG/NN intrinsic capacitances (Cgd, Cgs, Cdd) as companion models
- Charge state tracking via `get_charges()`, `init_charge_state()`, `update_charge_state()`
- LTE-based adaptive sub-stepping available (opt-in, `max_substeps > 1`)

### Convergence Aids

- **SPICE-standard GMIN**: Minimum channel conductance (1e-12 S, matching NGSPICE)
- **Gmin stepping**: Exponentially decaying minimum conductance (1e-9 to 1e-12)
- **Pseudo-transient initialization**: Artificial capacitances for startup (auto-scaled to 5x max circuit cap)
- **Adaptive damping**: Oscillation detection with supply-relative threshold
- **Voltage clamping**: Vgs +/-5V, Vds +/-10V to prevent numerical overflow

---

## Development

**Adding a new component:** subclass `pycircuitsim.models.base.Component`,
implement `stamp_conductance()`, `stamp_rhs()`, and `calculate_current()`,
then register the new prefix in `pycircuitsim/parser.py` (and in
`solver._is_mosfet()` if it's a MOSFET-like nonlinear device).

**Coding style:** type hints on all signatures, descriptive variable
names (`v_gate`, `i_drain`), docstrings on public APIs, stdlib →
third-party → local import order.

**Debugging:** `python main.py circuit.sp -v` shows Newton-Raphson
iterations; the `.lis` file records per-step convergence and final
device currents.

---

## Limitations

PyCircuitSim is intentionally simplified for educational use.

**Not supported:** inductors, mutual inductance, `.noise`, `.option`,
`.measure`, `.param`, subcircuits (`.subckt`).

**Known limits:** pure Python is ~10-100× slower than compiled
simulators; tested on circuits with <100 components; strongly
non-linear circuits may need source stepping. For production IC
design, large netlists (>1000 components), or RF/high-frequency
simulation, use ngspice / Xyce / Spectre.

## Future Work

- [ ] Adaptive output timestep
- [ ] Expanded SRAM / ring-oscillator test suite
- [ ] Inductor support, AC small-signal, `.subckt`

## References

- **ngspice** — open-source SPICE simulator ([ngspice.sourceforge.net](http://ngspice.sourceforge.net)). Reference for netlist syntax and device equations.
- **PyCMG** — Python BSIM-CMG OSDI wrapper ([github.com/ShenShan123/PyCMG](https://github.com/ShenShan123/PyCMG)).
- **ASAP7 PDK** — Arizona State Predictive 7nm PDK ([github.com/The-OpenROAD-Project/asap7_pdk_r1p7](https://github.com/The-OpenROAD-Project/asap7_pdk_r1p7)).
- **Xyce** — parallel electronic simulator ([xyce.sandia.gov](https://xyce.sandia.gov)). Architectural patterns for solver-device separation.
- Shichman & Hodges (1968), "Modeling and Simulation of Insulated-Gate Field-Effect Transistor Switching Circuits," *IEEE JSSC*.
- Nagel, L. W. (1975), "SPICE2: A Computer Program to Simulate Semiconductor Circuits," *ERL-M520*, UC Berkeley.

## License

MIT — see [LICENSE](LICENSE). Inspired by ngspice, Xyce, and SPICE2.
Issues / discussions: <https://github.com/ShenShan123/PyCircuitSim>.
