# PyCircuitSim

<div align="center">

**Pure Python Circuit Simulator with Compact-Model Research Bench**

A clean, readable SPICE-like circuit simulator with four coexisting
compact-model families that share the same solver:

- **BSIM-CMG** (LEVEL=72) via PyCMG/OSDI — production-grade FinFET ground truth.
- **DirectNet** (LEVEL=73) — feed-forward MLP; the **production** NN fast path.
- **BSIM-AR Transformer** (LEVEL=74) — autoregressive compact model; the
  validated **higher-fidelity** option (best accuracy, ~30–100× slower).
- **PFN / TabPFN** (LEVEL=75) — TabPFN-v3-style in-context transformer
  (**research**, V6.10.0).

The three NN families live side-by-side in the unified `bsimar` package
(`external_compact_models/bsimar/`) and share data generation,
normalization, losses, training, metrics, and evaluation. BSIM-CMG is the
ground truth all three train against and are gated on.

Six technologies are supported: **ASAP7** and **TSMC5/6/7/12/16**.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Netlist Syntax](#netlist-syntax)
  - [Subcircuits (.subckt)](#subcircuits-subckt-v6120)
- [Examples](#examples)
- [Python API](#python-api)
- [Output Files](#output-files)
- [NN Compact Models (LEVEL=73 / 74 / 75)](#nn-compact-models-level73--74--75)
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
| NMOS/PMOS Level 72 | `M` | BSIM-CMG FinFET via PyCMG/OSDI — ground truth |
| NMOS/PMOS Level 73 | `M` | DirectNet (MLP) via PyTorch — **production** NN |
| NMOS/PMOS Level 74 | `M` | BSIM-AR Transformer via PyTorch — highest fidelity |
| NMOS/PMOS Level 75 | `M` | PFN / TabPFN in-context transformer via PyTorch — research |
| Subcircuit instance | `X` | Hierarchical instance of a `.subckt` definition |

The legacy Shichman-Hodges `LEVEL=1` model has been removed. Only
LEVEL=72/73/74/75 are supported.

### Supported Analyses

- **DC Operating Point** (`op`) - Single-point bias calculation
- **DC Sweep** (`.dc`) - Parameter sweep analysis
- **Transient Analysis** (`.tran`) - Time-domain simulation
- **AC Analysis** (`.ac`) - Small-signal frequency-domain sweep (`dec`/`oct`/`lin`);
  linearizes about the DC OP and solves the complex MNA `Y = G + jωC` including the
  full MOSFET transcapacitance matrix. NGSPICE-validated for LEVEL=72 (machine
  precision) and LEVEL=73 (NN — see `docs/V6.6.6-accuracy-report.md`).

### Supported Directives

- `.model` - MOSFET model definitions (LEVEL=72, 73, 74, or 75)
- `.include` - Include external library files
- `.ic` - Set initial node voltages (top-level or inside a `.subckt` body)
- `.subckt` / `.ends` - Subcircuit definitions with ports and parameters,
  instantiated by `X` lines; flattened at parse time (V6.12.0)
- `uic` on `.tran` - start the transient from the `.ic` state (NGSPICE-style)
- AC stimulus on sources: `AC=<mag> [phase]` on `V`/`I` lines

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

# PyTorch is required for LEVEL=73 / 74 / 75 inference and for training
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
- `TSMC{5,6,7,12,16}/cln*.l` — raw TSMC PDK files. These are IP-protected and
  gitignored; supply them yourself to run any TSMC verification. Naive
  modelcards used by NGSPICE tests are regenerated on-the-fly from the raw
  PDK via `pycmg.tech.resolve_modelcard` and cached under
  `external_compact_models/PyCMG/build/modelcards/`.

### NN Compact Model Setup (LEVEL=73 / 74 / 75)

The `bsimar` package at `external_compact_models/bsimar/` is importable
once `external_compact_models/` is on `sys.path`. The `pycircuitsim`
parser and the test harness add this automatically. Checkpoints live at
`external_compact_models/bsimar/checkpoints/` and are resolved by the
parser at netlist load time. See [NN Compact Models](#nn-compact-models-level73--74--75)
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
        ├── bsimcmg_inverter_tran_simulation.lis    # Run summary log
        ├── bsimcmg_inverter_tran_transient.csv     # Waveform data
        └── bsimcmg_inverter_tran_transient.png     # Waveform plot
```

Artifacts by analysis type:

| Analysis | Subdirectory | Files produced |
|----------|--------------|----------------|
| `.dc` | `dc/` | `_simulation.lis`, `_dc_sweep.csv`, `_dc_sweep.png` |
| `.ac` | `ac/` | `_ac_sweep.csv`, Bode plot |
| `.tran` | `tran/` | `_simulation.lis`, `_transient.csv`, `_transient.png` |
| none (`.op`) | `dc_op/` | `_dc_op_point.txt`, `_dc_op_simulation.lis` |

For programmatic access to waveforms without reading the CSV, drive
`TransientSolver` through the [Python API](#python-api) — it returns the
time vector and per-node arrays directly.

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

* MOSFET: M<name> <drain> <gate> <source> <bulk> <model> L=<len> NFIN=<fins>
* FinFET geometry is L + NFIN (fin count) — there is no W.
Mn1 3 2 0 0 nmos1 L=30n NFIN=10
Mp1 3 2 1 1 pmos1 L=30n NFIN=20

* Subcircuit instance: X<name> <nodes...> <subckt name> [param=val ...]
Xinv in out vdd inv NF=10
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

* AC sweep: .ac <dec|oct|lin> <points> <fstart> <fstop>   (AC stimulus via AC= on a source)
.ac dec 20 1e3 1e12
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

Checkpoints are resolved automatically at parse time:

1. The env pin `PYCIRCUITSIM_NN_CHECKPOINT_DN_{NMOS,PMOS}` is read **first**.
   Since V6.6.6 an absent pinned stem raises — there is no silent fallback.
2. For a TSMC5/6/7/12/16 netlist, the per-tech slot
   `tsmc{X}_dn_{large,medium,small,xl}` preempts (large-first); production
   is the `large` tier.
3. The dormant universal fallback chain is used only if neither applies.

Each resolution is logged so you can confirm what actually loaded:

```
[NN-resolver] L73 nmos_nn TECH=tsmc5 VT=lvt -> tsmc5_dn_large_nmos (scope=tsmc5, tech_code=2)
```

Per-tech models carry a **local embedding vocabulary** (variant count + 1
UNKNOWN slot), so the parser remaps the tech code via
`bsimar.config.local_variant_code(scope, tech, variant)`.

#### Level 74 (BSIM-AR — autoregressive Transformer compact model)

```spice
* Same netlist syntax as LEVEL=73, just a different LEVEL
.model nmos_ar NMOS (LEVEL=74 TECH=tsmc5 VT=lvt)
.model pmos_ar PMOS (LEVEL=74 TECH=tsmc5 VT=lvt)

Mn1 out in 0 0 nmos_ar L=16n NFIN=10
Mp1 out in vdd vdd pmos_ar L=16n NFIN=10
```

BSIM-AR uses per-tech checkpoints `tsmc{X}_tf_{small,medium,large}_{nmos,pmos}`
(plus recipe variants) under `external_compact_models/bsimar/checkpoints/`.
Un-parked in V6.8.0: the best config (`corroft@medium`, 1.9M params) reaches
**15/16 strict** complex-circuit gates, beating DirectNet production — at
~30–100× the CPU inference cost, which is why DirectNet remains the default.

#### Level 75 (PFN — TabPFN-style in-context transformer)

```spice
.model nmos_pfn NMOS (LEVEL=75 TECH=tsmc5 VT=lvt)
.model pmos_pfn PMOS (LEVEL=75 TECH=tsmc5 VT=lvt)

Mn1 out in 0 0 nmos_pfn L=16n NFIN=10
Mp1 out in vdd vdd pmos_pfn L=16n NFIN=10
```

Per-tech checkpoints `tsmc{X}_pfn_{small,medium,large}_{nmos,pmos}`; env pins
`PYCIRCUITSIM_NN_CHECKPOINT_PFN_{NMOS,PMOS}`. Frozen-context buffers live
inside the checkpoint, and the `_config.npz` sidecar is **required** to
rebuild the architecture. Research-tier: clean `small` scores 11/16 strict —
notably the first family with zero OMP-threading flips.

All three NN levels expose autograd-derived conductances (gm, gds, gmb) so
Newton-Raphson stays consistent in multi-device circuits. See
[NN Compact Models](#nn-compact-models-level73--74--75) for training,
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

* Reach a node inside a subcircuit instance with the hierarchical name
.ic V(X1.n1)=0.3
```

### Subcircuits (`.subckt`, V6.12.0)

Definitions take ports and optional parameters with defaults; `X` lines
instantiate them and may override any parameter. Expansion is **flattening
at parse time**, ngspice-style — the solver never sees hierarchy.

```spice
* Definition: .subckt <name> <ports...> [param=default ...]
.subckt inv i o vdd NF=10 VIC=1.0
Mp1 o i vdd vdd pmos1 L=30n NFIN=NF
Mn1 o i 0 0 nmos1 L=30n NFIN={NF*2}
.ic V(o)=VIC
.ends

* Instance: X<id> <nodes...> <name> [param=val ...]
Xinv in out vdd inv NF=20
```

Naming after flattening:

| Item | Becomes | Nested |
|------|---------|--------|
| Internal node `n1` in `X1` | `X1.n1` | `X1.X2.n1` |
| Device `Mp1` in `X1` | `M.X1.Mp1` | `M.X1.X2.Mp1` |
| Port nodes | mapped to the connecting node | — |
| Ground `0` / `GND` | stays global | — |

- **Parameters** resolve as bare names or `{expr}` / `'expr'` arithmetic
  (`+ - * /`, unit suffixes allowed).
- **`.ic` inside a body** is node-remapped *and* parameter-resolved; `uic`
  and `force_ic` consume the result unchanged.
- **`.model` / `.include` inside a body** are hoisted global; nested
  `.subckt` definitions register globally.
- **Loud errors** on unknown subcircuit names, port-count mismatches, and
  recursion (depth > 64) — no silent misbehavior.

Because the device type character is preserved (`M.X1.Mp1`), first-char
dispatch still works, so no circuit or solver changes were needed.

---

## Examples

All example netlists live in `examples/`. The files below cover the three
compact-model families plus a passive-only RC reference:

| File | Analysis | Models used | What it demonstrates |
|------|----------|-------------|----------------------|
| `examples/bsimcmg_nmos_dc.sp` | DC sweep | LEVEL=72 | NMOS Id-Vgs against PyCMG/OSDI |
| `examples/bsimcmg_pmos_dc.sp` | DC sweep | LEVEL=72 | PMOS Id-Vgs against PyCMG/OSDI |
| `examples/bsimcmg_inverter_dc.sp` | OP (no directive) | LEVEL=72 | Inverter bias point; hierarchical `.subckt` form |
| `examples/bsimcmg_inverter_tran.sp` | Transient | LEVEL=72 | FinFET inverter pulse response |
| `examples/bsimcmg_inverter_dc_asap7_ref.sp` | DC sweep | LEVEL=72 | ASAP7 reference configuration |
| `examples/nn_nmos_op.sp` | OP | LEVEL=73 | DirectNet single-point NMOS |
| `examples/nn_nmos_dc.sp` | DC sweep | LEVEL=73 | DirectNet NMOS Id-Vgs |
| `examples/nn_inverter_dc.sp` | DC sweep | LEVEL=73 | DirectNet inverter VTC |
| `examples/bsimar_nmos_dc.sp` | DC sweep | LEVEL=74 | BSIM-AR NMOS Id-Vgs |
| `examples/bsimar_inverter_dc.sp` | DC sweep | LEVEL=74 | BSIM-AR inverter VTC |
| `examples/bsimcmg_cs_amp_ac.sp` | AC sweep | LEVEL=72 | Common-source amp Bode response |
| `examples/rc_lowpass_ac.sp` | AC sweep | passives | Single-pole RC `.ac` reference |
| `examples/rc_transient.sp` | Transient | passives | Pure RC reference |
| `examples/complex/miller_opamp_directnet.sp` | DC | LEVEL=73 | Two-stage Miller opamp |
| `examples/complex/ring_osc_5stage_directnet.sp` | Transient | LEVEL=73 | 5-stage ring oscillator |
| `examples/complex/sram_6t_directnet.sp` | DC | LEVEL=73 | 6T SRAM butterfly / SNM |
| `examples/complex/switchcap_unitcell_directnet.sp` | Transient | LEVEL=73 | Switched-capacitor unit cell |

Since V6.12.0 the inverter, NN, and complex decks are written
**hierarchically** with `.subckt` + `X` instances. Probed nodes are kept at
top level (they are ports), so results and tooling are unchanged.

### Sample: BSIM-CMG FinFET Inverter Transient (ASAP7 7nm)

```spice
* examples/bsimcmg_inverter_tran.sp — hierarchical (.subckt) version
Vdd 1 0 1.0
Vin 2 0 PULSE 0 1.0 0.5n 0.1n 0.1n 0.8n 2n

* Inverter instance: ports (in, out, vdd); node 3 = output stays top-level
Xinv 2 3 1 inv
Cload 3 0 10e-15

* The .ic inside the body is remapped to the connected port node (node 3);
* VIC shows a parameterized initial condition.
.subckt inv i o vdd VIC=1.0
Mp1 o i vdd vdd pmos1 L=30n NFIN=10
Mn1 o i 0 0 nmos1 L=30n NFIN=10
.ic V(o)=VIC
.ends

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

# Low-level: drive the solver directly (this is how you get waveform arrays)
from pycircuitsim import Parser
from pycircuitsim.solver import DCSolver, TransientSolver

parser = Parser()
parser.parse_file('examples/bsimcmg_inverter_tran.sp')
circuit = parser.circuit

# DC operating point -> {node_name: voltage}
dc_solution = DCSolver(circuit).solve()

# Transient: t_stop / dt are constructor args; solve() takes none.
solver = TransientSolver(circuit, t_stop=5e-9, dt=10e-12,
                         initial_guess=dc_solution)
results = solver.solve()

time = results['time']          # numpy array of timepoints
v_out = results['3']            # per-node numpy array, keyed by node name
```

`results` is a dict keyed by node name plus a `'time'` entry; hierarchical
nodes appear under their flattened names (e.g. `'X1.n1'`).

`pycircuitsim.simulation` also exposes `run_dc_sweep()`, `run_transient()`,
and `run_ac_sweep()` for the full orchestrated workflow (parse → solve →
plot → write files).

---

## Output Files

PyCircuitSim generates output files organized by circuit name and analysis type:

```
results/
└── <circuit_name>/
    ├── dc/                              # .dc sweep
    │   ├── <circuit>_simulation.lis      # Detailed iteration log
    │   ├── <circuit>_dc_sweep.csv        # Numerical waveform data
    │   └── <circuit>_dc_sweep.png        # Voltage/current plots
    ├── ac/                              # .ac sweep
    │   ├── <circuit>_ac_sweep.csv        # Magnitude / phase per frequency
    │   └── <circuit>_ac_sweep.png        # Bode plot
    ├── tran/                            # .tran
    │   ├── <circuit>_simulation.lis      # Run summary log
    │   ├── <circuit>_transient.csv       # Time + per-node voltages
    │   └── <circuit>_transient.png       # Waveform plot
    └── dc_op/                           # no analysis directive
        ├── <circuit>_dc_op_point.txt     # Final node voltages
        └── <circuit>_dc_op_simulation.lis
```

> **Note:** the transient `.lis` is a run summary (header, circuit
> summary, final state) rather than a per-timestep iteration log —
> `TransientSolver` takes no `output_file`. Use `debug=True` for
> iteration-level convergence detail.

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

## NN Compact Models (LEVEL=73 / 74 / 75)

All three NN compact-model families live in the unified `bsimar` package at
`external_compact_models/bsimar/` and share every layer below the model
architecture itself (data, normalization, losses, training, eval). See
[Architecture](#architecture) for the package layout.

Models are trained **per technology**, not universally: each
(tech × device) pair gets its own checkpoint with a local embedding
vocabulary. BSIM-CMG (LEVEL=72) is the ground truth for both training
targets and pass/fail gating.

### Data Generation

Data is produced by PyCMG (the ground-truth BSIM-CMG model), one `.npz`
per tech + device:

```bash
conda run -n pycircuitsim python \
    external_compact_models/PyCMG/scripts/generate_nn_data.py \
    --device both --tech tsmc5 --enable-inv-trip --n-workers 8
```

`--tech` accepts `tsmc5`, `tsmc6`, `tsmc7`, `tsmc12`, `tsmc16`, `asap7`, or
`all`. `--enable-inv-trip` adds the inverter-trip overlay; the grid sampler
also carries the reverse-Vds corridor. Datasets land under
`external_compact_models/bsimar/data/datasets/`.

Each sample is a **7-dim** input (`Vgs, Vds, Vbs, NFIN, L, T, tech_code`),
where `tech_code` is a discrete index consumed by `nn.Embedding`, and a
13-column output (`id, gm, gds, gmb, qg, qd, qs, qb, cgg, cgd, cgs, cdg, cdd`).

Note that NFIN=1 is excluded in practice: unstable `(variant, NFIN=1)` bins
fail OSDI convergence and are dropped per-bin during generation, so NFIN≥2
is what actually trains.

### Training

One CLI drives all three architectures; `--model` picks which.
`--tech-scope` auto-sets `--exclude-techs`, `--num-tech-codes`, the default
`--data` path, and the `save_prefix` that the parser's resolver recognizes.

```bash
# DirectNet (LEVEL=73) — production family
conda run -n pycircuitsim python -u -m bsimar.cli.train \
    --model direct --size large \
    --device-type nmos --tech-scope tsmc5 --cuda --overwrite

# BSIM-AR Transformer (LEVEL=74)
conda run -n pycircuitsim python -u -m bsimar.cli.train \
    --model transformer --size medium \
    --device-type nmos --tech-scope tsmc5 --cuda --overwrite

# PFN / TabPFN (LEVEL=75)
conda run -n pycircuitsim python -u -m bsimar.cli.train \
    --model tabpfn --size small \
    --device-type nmos --tech-scope tsmc5 --cuda --overwrite
```

`--size` ∈ `{small, medium, large, xl}`; `xl` (~2.13M params) sits past the
over-fit boundary for DirectNet. Capacity does **not** improve monotonically
— each family peaks at a different tier (see the table below).

### Checkpoints

Checkpoints live in `external_compact_models/bsimar/checkpoints/`, each as
`*_best.pt` plus a `_norm.npz` (and `_config.npz` for BSIM-AR / PFN).

| Stem | Family | Notes |
|------|--------|-------|
| `tsmc{5,6,7,12,16}_dn_{small,medium,large,xl}_{nmos,pmos}` | DirectNet | Production = `large`; resolver is large-first |
| `tsmc{X}_dn_v660clean_large_*` | DirectNet | Clean archive; warm-start base for curriculum runs |
| `tsmc{X}_dn_crit30f_large_*` | DirectNet | Production provenance (crit30 curriculum) |
| `tsmc{X}_tf_{small,medium,large}_{nmos,pmos}` | BSIM-AR | Best config `corroft@medium` |
| `tsmc{X}_pfn_{small,medium,large}_{nmos,pmos}` | PFN | `_config.npz` required to rebuild arch |
| `u716_dn_*` | Universal DirectNet | 18-code vocab, env-pin only |

A completed training run leaves a `*_best.pt.complete` marker — a bare
`_best.pt` may be from a killed run, so gate on the marker.

### Inference: choosing a family

All three LEVELs are drop-in replacements for LEVEL=72 on the same netlist.
They share sign conventions, the Jacobian-via-autograd guarantee for
Newton-Raphson consistency, and the source-relative voltage frame (for
**both** device types — see Critical Design Rules in CLAUDE.md).

| Aspect | LEVEL=73 (DirectNet) | LEVEL=74 (BSIM-AR) | LEVEL=75 (PFN) |
|--------|----------------------|--------------------|----------------|
| Architecture | MLP (SiLU) | Transformer, causal mask | In-context transformer |
| Forward pass | 1 per device eval | 13 sequential tokens | 1 with frozen context |
| Inference cost | ~1× | ~30–100× | ~10× |
| Best capacity tier | `large` | `medium` | `small` |
| Complex gates (strict) | 14/16 | **15/16** | 11/16 |
| Role | **Production** fast path | Highest fidelity | Research |

The practical trade-off: **DirectNet** unless you specifically need
BSIM-AR's extra fidelity and can absorb the AR inference cost.

---

## Verification

All BSIM-CMG results are validated against NGSPICE 45.2 with the
BSIM-CMG OSDI binary. NN compact models (LEVEL=73/74/75) are validated
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

# Subcircuit hierarchy gate (8 checks)
python tests/verify_subckt.py

# Run L1 smoke suite in one line
python tests/verify_bsimcmg_op.py && \
python tests/verify_bsimcmg_dc.py && \
python tests/verify_bsimcmg_tran.py
```

Gates are **CPU-pinned** and honor `NGSPICE_BIN` (use the repo-local ngspice
when `/usr/local/ngspice-45.2` is absent). The high-gain VTC trip has ~±1%
run-to-run scatter under multi-threaded BLAS, so thread pinning is not
optional for reproducible NN results:

```bash
NGSPICE_BIN="$PWD/tools/ngspice-45.2/bin/ngspice" \
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python tests/verify_subckt.py
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

#### Subcircuit Hierarchy (V6.12.0)

`tests/verify_subckt.py` — **8/8 PASS**. Levels 1–3 cover linear
equivalence, an L72 inverter, and a nested buffer:

| Check | Metric | Result |
|-------|--------|--------|
| RC-ladder `.tran`, subckt == flat | max abs dV | 0.000e+00 V |
| RC-lowpass `.ac`, subckt == flat | max abs dMag / dPh | 0.000e+00 |
| Nested subckt + `{expr}` params, DC OP | max abs dV | 0.000e+00 V |
| L72 inverter `.tran`, subckt == flat | max abs dV | 0.000e+00 V |
| L72 inverter subckt vs NGSPICE | NRMSE | 0.187% of VDD |
| L72 nested buffer vs NGSPICE | NRMSE (out / mid) | 0.638% / 0.861% |

Flattening is exactly equivalent, not merely close: the subckt-vs-flat
checks are **bit-identical**.

#### NN Compact Models (LEVEL=73, production)

DirectNet production (`large` tier, crit30 curriculum) device-level baseline
across TSMC5/7/12/16: **inverter 8/8, DC 55/55, transient 64/64**.

On the 16 complex-circuit gates (4 circuits × 4 techs), scored strictly
against NGSPICE: DirectNet **14/16**, BSIM-AR **15/16**, PFN **11/16**.
The one cell no family passes is the TSMC7 opamp, which is reachable only
via the T3 differentiable-DC-solver fine-tune.

### Verification Scripts

| Script | Purpose |
|--------|---------|
| `tests/verify_bsimcmg_op.py` | OP analysis: PyCircuitSim vs NGSPICE for NMOS, PMOS, inverter |
| `tests/verify_bsimcmg_dc.py` | DC sweep L1: Id-Vgs (ASAP7 baseline) |
| `tests/verify_bsimcmg_dc_comprehensive.py` | DC sweep L2: 67-config multi-tech VT/L/NFIN sweep |
| `tests/verify_multi_tech_dc.py` | DC sweep L3: 44-config inverter VTC + parametric |
| `tests/verify_bsimcmg_tran.py` | Transient L1: single inverter baseline |
| `tests/verify_bsimcmg_tran_comprehensive.py` | Transient L2: 37-config VT/L/NFIN sweep |
| `tests/verify_multi_tech_tran.py` | Transient L3: 86-config multi-tech parametric |
| `tests/verify_ac.py` | AC L1 passive RC + L2 BSIM-CMG common-source amp |
| `tests/verify_subckt.py` | Subcircuit hierarchy: equivalence, L72 inverter, nested buffer (8 checks) |

NN compact models (LEVEL=73/74/75):

| Script | Purpose |
|--------|---------|
| `tests/verify_nn_dc_tran.py` | NN device + inverter gate across TSMC techs (`--tech`, `--inverter-only`) |
| `tests/verify_nn_multi_tech_dc.py` | Parametric Id-Vgs over L / NFIN / VT (55 configs) |
| `tests/verify_nn_multi_tech_tran.py` | Parametric inverter over P/N ratio, VDD, Cload, slew, pulse width |
| `tests/verify_nn_ac.py` | NN CS-amp AC: gain / f3db / magnitude NRMSE |
| `tests/verify_nn_lifted_source_dc.py` | Lifted-source canary — guards the source-relative frame |

Complex circuits (4 circuits × 4 techs = 16 gates), scored against NGSPICE
BSIM-CMG ground truth:

| Script | Purpose |
|--------|---------|
| `tests/verify_complex_opamp.py` / `_ac.py` / `_sweep.py` | Two-stage Miller opamp: gain, open-loop AC, parametric |
| `tests/verify_complex_ring_osc.py` / `verify_complex_ringosc_sweep.py` | Ring-oscillator period + parametric mirror |
| `tests/verify_complex_switchcap.py` / `_sweep.py` | Switched-capacitor charge / droop |
| `tests/verify_complex_sram_snm.py` / `verify_complex_sram_sweep.py` | 6T SRAM butterfly positivity + NRMSE tracking |
| `tests/verify_complex_sweep_canaries.py` | Guards single-point ↔ sweep equivalence |

Diagnostics (reusable controls, **not** pass/fail gates — they use
L72-in-PyCircuitSim as reference rather than NGSPICE):

| Script | Purpose |
|--------|---------|
| `tests/diag_l72_complex_control.py` | Proves L72-in-PyCircuitSim matches NGSPICE, isolating NN-surface gaps |
| `tests/diag_l72_switchcap_control.py` / `_uic_control.py` | Same control for the switched-cap cell (incl. `uic`) |
| `tests/diag_nn_jacobian_consistency.py` | FD-vs-autograd Jacobian consistency check |

Each script generates comparison plots and detailed metrics in `tests/verify_*_results/`.

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
    ├── mosfet_nn.py                # Shared _MOSFETNNBase for LEVEL=73/74/75
    ├── mosfet_directnet.py         # DirectNet (LEVEL=73) via PyTorch
    ├── mosfet_bsimar.py            # BSIM-AR Transformer (LEVEL=74) via PyTorch
    └── mosfet_pfn.py               # PFN / TabPFN (LEVEL=75) via PyTorch

external_compact_models/            # External compact-model packages
├── bsimar/                         # Unified NN compact model package (all 3 families)
│   ├── config.py                   # NNTechConfig + TECH_CODE_MAP + local-vocab helpers
│   ├── data/                       # normalize.py, dataset.py
│   ├── models/                     # direct_net.py, transformer.py, tabpfn.py
│   ├── losses/                     # bni_mae.py (MAELoss + per-target LDS weights)
│   ├── training/                   # trainer.py
│   ├── eval/                       # metrics.py, loo_labels.py
│   ├── cli/train.py                # `python -m bsimar.cli.train --model {direct,transformer,tabpfn}`
│   ├── checkpoints/                # Trained weights (gitignored)
│   └── docs/, imgs/, README.md, LICENSE
│
└── PyCMG/                          # BSIM-CMG OSDI wrapper (git submodule)
    ├── pycmg/                      # Python ctypes-based OSDI interface
    ├── build/osdi/bsimcmg.osdi     # Compiled OSDI binary
    ├── modelcards/                 # Technology modelcards (ASAP7 + TSMC5/6/7/12/16)
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
| `parser.py` | Two-pass netlist parsing, `.model`/`.include`/`.ic`/`.subckt` directives, subcircuit flattening, LEVEL=73/74/75 checkpoint + process-param resolution via `bsimar.config` |
| `circuit.py` | Stores circuit topology, component list, node mapping |
| `solver.py` | MNA construction, Newton-Raphson iteration, transient stepping, AC solve |
| `models/mosfet_cmg.py` | BSIM-CMG physics via PyCMG (LEVEL=72) |
| `models/mosfet_nn.py` | Shared NN base: voltage prep, autograd Jacobian, Vds correction |
| `models/mosfet_directnet.py` | DirectNet inference with autograd conductances (LEVEL=73) |
| `models/mosfet_bsimar.py` | BSIM-AR Transformer inference (LEVEL=74) |
| `models/mosfet_pfn.py` | PFN / TabPFN in-context inference (LEVEL=75) |
| `logger.py` | HSPICE-compatible output formatting |
| `visualizer.py` | Automatic plot generation |
| `bsimar.*` | Training / eval pipeline shared by LEVEL=73, 74, and 75 |

### Design Principles

- **Separation of concerns** — solver builds matrices and iterates (no
  device equations); device models compute currents/conductances from
  voltages (no matrix ops); `simulation.py` orchestrates the workflow.
- **Modularity** — every device inherits from `Component` and exposes a
  common interface (`calculate_current()`, `get_conductances()`); new
  devices can be added without touching the solver.
- **Drop-in compact models** — LEVEL=72/73/74/75 share sign conventions
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
`.measure`, global `.param` (subcircuit-scoped parameters *are* supported —
see [Subcircuits](#subcircuits-subckt-v6120)).

**Known limits:** pure Python is ~10-100× slower than compiled
simulators; tested on circuits with <100 components; strongly
non-linear circuits may need source stepping. For production IC
design, large netlists (>1000 components), or RF/high-frequency
simulation, use ngspice / Xyce / Spectre.

## Future Work

- [x] AC small-signal (`.ac`) — LEVEL=72 NGSPICE-exact; LEVEL=73 NN gated across all
  24 capacity checkpoints (V6.5: device CS-amp 13/24, opamp open-loop dynamics validated)
- [x] Subcircuits (`.subckt` / `.ends` / `X` instances) — V6.12.0, 8/8 gate
- [x] Expanded SRAM / ring-oscillator test suite — 16 complex-circuit gates
      plus parametric sweep mirrors
- [ ] Adaptive output timestep
- [ ] Inductor support
- [ ] Global `.param` / `.measure`

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
