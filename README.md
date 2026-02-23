# PyCircuitSim

<div align="center">

**Pure Python Circuit Simulator for Education**

A clean, readable SPICE-like circuit simulator with production-grade BSIM-CMG FinFET model support via PyCMG/OSDI.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Netlist Syntax](#netlist-syntax)
- [Examples](#examples)
- [Python API](#python-api)
- [Output Files](#output-files)
- [Verification](#verification)
- [Architecture](#architecture)
- [Algorithms](#algorithms)
- [Development](#development)
- [Limitations](#limitations)
- [References](#references)

---

## Overview

PyCircuitSim is an open-source, pure Python circuit simulator designed for educational purposes and compact model integration. It provides a clean implementation of SPICE-like simulation with emphasis on code readability and modular architecture, plus production-grade BSIM-CMG FinFET model support via PyCMG/OSDI.

### Key Design Goals

- **Educational Clarity**: Clean, well-documented code that's easy to understand and modify
- **Modular Architecture**: Complete separation between solver engine and device models
- **Compact Model Integration**: BSIM-CMG FinFET models (LEVEL=72) via PyCMG/OSDI
- **HSPICE Compatibility**: Supports standard SPICE netlist syntax

### What It Does

PyCircuitSim simulates electronic circuits using:
- **Modified Nodal Analysis (MNA)** for circuit equations
- **Newton-Raphson iteration** for non-linear components (MOSFETs)
- **Trapezoidal integration** for transient analysis (2nd order)
- **BSIM-CMG compact models** for FinFET device simulation (ASAP7 7nm)

---

## Features

### Supported Components

| Component | Symbol | Description |
|-----------|--------|-------------|
| Resistor | `R` | Linear resistance |
| Capacitor | `C` | Linear capacitance |
| Voltage Source | `V` | DC or PULSE waveform |
| Current Source | `I` | DC current source |
| NMOS/PMOS Level 1 | `M` | Shichman-Hodges MOSFET |
| NMOS/PMOS Level 72 | `M` | BSIM-CMG FinFET via PyCMG/OSDI |

### Supported Analyses

- **DC Operating Point** (`op`) - Single-point bias calculation
- **DC Sweep** (`.dc`) - Parameter sweep analysis
- **Transient Analysis** (`.tran`) - Time-domain simulation

### Supported Directives

- `.model` - MOSFET model definitions (LEVEL=1 or LEVEL=72)
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

# Install Python dependencies
pip install numpy matplotlib
```

### BSIM-CMG Setup (Optional)

To use BSIM-CMG FinFET models (LEVEL=72), the PyCMG submodule and a compiled OSDI binary are required:

```bash
# Initialize the PyCMG submodule (if not cloned with --recurse-submodules)
git submodule update --init --recursive

# Verify the OSDI binary exists
ls external_compact_models/PyCMG/build-deep-verify/osdi/bsimcmg.osdi
```

The ASAP7 7nm modelcards are included in `external_compact_models/PyCMG/tech_model_cards/ASAP7/`.

---

## Quick Start

### Running a Simulation

```bash
# Activate the environment
conda activate pycircuitsim

# Run a Level-1 CMOS inverter DC sweep
python main.py examples/test_cmos_inverter_level1.sp

# Run a BSIM-CMG inverter transient simulation
python main.py examples/bsimcmg_inverter_tran.sp

# Specify a custom output directory
python main.py examples/bsimcmg_inverter_tran.sp -o my_results

# Enable verbose logging (shows Newton-Raphson iterations)
python main.py examples/test_cmos_inverter_level1.sp -v
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

#### Level 1 (Shichman-Hodges)

```spice
.model NMOS_VTL NMOS (
    LEVEL=1
    VTO=0.7      ; Threshold voltage (V)
    KP=110u      ; Transconductance parameter (A/V^2)
    GAMMA=0.4    ; Body effect coefficient (V^0.5)
    LAMBDA=0.02  ; Channel-length modulation (1/V)
)

.model PMOS_VTL PMOS (
    LEVEL=1
    VTO=-0.7
    KP=50u
    GAMMA=0.5
    LAMBDA=0.03
)
```

#### Level 72 (BSIM-CMG FinFET)

```spice
* Model declaration (device parameters come from ASAP7 modelcard)
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

### Level 1: CMOS Inverter DC Sweep

File: `examples/test_cmos_inverter_level1.sp`

```spice
* CMOS Inverter - Level 1 Models
Vdd 1 0 3.3
Vin 2 0 1.65

Mp1 3 2 1 1 PMOS L=1u W=20u
Mn1 3 2 0 0 NMOS L=1u W=10u

.dc Vin 0 3.3 0.1

.end
```

```bash
python main.py examples/test_cmos_inverter_level1.sp
```

### Level 1: Inverter Transient

File: `examples/level1_inverter_tran.sp`

```spice
* CMOS Inverter Transient - Level 1
Vdd 1 0 3.3
Vin 2 0 PULSE(0 3.3 1n 0.1n 0.1n 5n 10n)

Mp1 3 2 1 1 PMOS_VTL L=1u W=20u
Mn1 3 2 0 0 NMOS_VTL L=1u W=10u
Cload 3 0 100f

.ic V(3)=3.3

.model NMOS_VTL NMOS (LEVEL=1 VTO=0.7 KP=110u GAMMA=0.4 LAMBDA=0.02)
.model PMOS_VTL PMOS (LEVEL=1 VTO=-0.7 KP=50u GAMMA=0.5 LAMBDA=0.03)

.tran 100p 50n

.end
```

```bash
python main.py examples/level1_inverter_tran.sp
```

### BSIM-CMG: FinFET Inverter Transient (ASAP7 7nm)

File: `examples/bsimcmg_inverter_tran.sp`

```spice
* BSIM-CMG CMOS Inverter - ASAP7 7nm FinFET
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

```bash
python main.py examples/bsimcmg_inverter_tran.sp
```

### BSIM-CMG: NMOS DC Sweep

File: `examples/bsimcmg_nmos_dc.sp`

```bash
python main.py examples/bsimcmg_nmos_dc.sp
```

### RC Transient

File: `examples/rc_transient.sp`

```bash
python main.py examples/rc_transient.sp
```

---

## Python API

### Basic Usage

```python
from pycircuitsim.simulation import run_simulation

# Run a simulation from a netlist file
run_simulation(
    netlist_path='examples/bsimcmg_inverter_tran.sp',
    output_dir='my_results',
    verbose=True
)
```

### Direct Solver Access

```python
from pycircuitsim import Parser, DCSolver

# Parse netlist
parser = Parser()
parser.parse_file('examples/test_cmos_inverter_level1.sp')
circuit = parser.circuit

# Run DC analysis
solver = DCSolver(circuit)
solution = solver.solve()

# Access node voltages
for node, voltage in solution.items():
    print(f"{node}: {voltage:.4f} V")

# Calculate device currents
for component in circuit.components:
    current = component.calculate_current(solution)
    print(f"{component.name}: {current:.6f} A")
```

### DC Sweep

```python
from pycircuitsim.simulation import run_dc_sweep
from pycircuitsim import Parser

parser = Parser()
parser.parse_file('examples/test_cmos_inverter_level1.sp')
circuit = parser.circuit

# run_dc_sweep returns sweep values and results dict
sweep_values, results = run_dc_sweep(circuit)
```

### Transient Analysis

```python
from pycircuitsim import Parser
from pycircuitsim.solver import DCSolver, TransientSolver

parser = Parser()
parser.parse_file('examples/bsimcmg_inverter_tran.sp')
circuit = parser.circuit

# First solve DC operating point
dc_solver = DCSolver(circuit)
dc_solution = dc_solver.solve()

# Then run transient
tran_solver = TransientSolver(circuit)
time_points, results = tran_solver.solve(
    tstep=10e-12,    # 10 ps
    tstop=5e-9,      # 5 ns
    dc_solution=dc_solution
)
```

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

## Verification

All BSIM-CMG results are validated against NGSPICE 45.2 with BSIM-CMG OSDI. Verification scripts are in `tests/`.

### Running Verification

```bash
conda activate pycircuitsim

# Operating point verification (NMOS, PMOS, Inverter)
python tests/verify_bsimcmg_op.py

# DC sweep verification (Id-Vgs, VTC)
python tests/verify_bsimcmg_dc.py

# Transient verification (single baseline config)
python tests/verify_bsimcmg_tran.py

# Comprehensive transient verification (21 parametric configs)
python tests/verify_bsimcmg_tran_comprehensive.py

# Run all verification scripts
python tests/verify_bsimcmg_op.py && \
python tests/verify_bsimcmg_dc.py && \
python tests/verify_bsimcmg_tran.py && \
python tests/verify_bsimcmg_tran_comprehensive.py
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
| NRMSE (post-settling) | 0.23% |
| NRMSE (full-range) | 0.29% |
| Max absolute error | 9.9 mV (1.4% of Vdd) |

#### Comprehensive Transient (21 Configurations)

The comprehensive suite sweeps VDD, Cload, input slew, pulse width, NFIN scaling, and P/N ratio. All 21 configurations pass (NRMSE < 5%).

| Config | VDD | NFIN_N/P | Cload | tr/tf | pw | NRMSE(%) | MaxErr(mV) | Status |
|--------|-----|----------|-------|-------|----|----------|------------|--------|
| vdd_0p5 | 0.50V | 10/10 | 10fF | 100ps | 0.8ns | 0.17 | 5.2 | PASS |
| vdd_0p6 | 0.60V | 10/10 | 10fF | 100ps | 0.8ns | 0.22 | 10.1 | PASS |
| baseline | 0.70V | 10/10 | 10fF | 100ps | 0.8ns | 0.22 | 9.9 | PASS |
| vdd_0p8 | 0.80V | 10/10 | 10fF | 100ps | 0.8ns | 0.20 | 11.1 | PASS |
| cload_1fF | 0.70V | 10/10 | 1fF | 100ps | 0.8ns | 0.95 | 67.3 | PASS |
| cload_5fF | 0.70V | 10/10 | 5fF | 100ps | 0.8ns | 0.30 | 17.6 | PASS |
| cload_50fF | 0.70V | 10/10 | 50fF | 100ps | 0.8ns | 0.04 | 1.3 | PASS |
| cload_100fF | 0.70V | 10/10 | 100fF | 100ps | 0.8ns | 0.03 | 0.9 | PASS |
| slew_10ps | 0.70V | 10/10 | 10fF | 10ps | 0.8ns | 0.15 | 7.2 | PASS |
| slew_50ps | 0.70V | 10/10 | 10fF | 50ps | 0.8ns | 0.13 | 6.1 | PASS |
| slew_500ps | 0.70V | 10/10 | 10fF | 500ps | 0.8ns | 0.17 | 8.3 | PASS |
| pw_0p2ns | 0.70V | 10/10 | 10fF | 100ps | 0.2ns | 0.31 | 9.9 | PASS |
| pw_0p5ns | 0.70V | 10/10 | 10fF | 100ps | 0.5ns | 0.26 | 9.9 | PASS |
| pw_2p0ns | 0.70V | 10/10 | 10fF | 100ps | 2.0ns | 0.15 | 9.9 | PASS |
| nfin_1 | 0.70V | 1/1 | 10fF | 100ps | 0.8ns | 0.03 | 0.9 | PASS |
| nfin_2 | 0.70V | 2/2 | 10fF | 100ps | 0.8ns | 0.05 | 1.3 | PASS |
| nfin_5 | 0.70V | 5/5 | 10fF | 100ps | 0.8ns | 0.10 | 3.9 | PASS |
| nfin_20 | 0.70V | 20/20 | 10fF | 100ps | 0.8ns | 0.30 | 17.6 | PASS |
| pn_0p5 | 0.70V | 10/5 | 10fF | 100ps | 0.8ns | 0.18 | 8.9 | PASS |
| pn_1p5 | 0.70V | 10/15 | 10fF | 100ps | 0.8ns | 0.25 | 12.0 | PASS |
| pn_2p0 | 0.70V | 10/20 | 10fF | 100ps | 0.8ns | 0.26 | 13.2 | PASS |

### Verification Scripts

| Script | Purpose |
|--------|---------|
| `tests/verify_bsimcmg_op.py` | OP analysis: PyCircuitSim vs NGSPICE for NMOS, PMOS, inverter |
| `tests/verify_bsimcmg_dc.py` | DC sweep: Id-Vgs and VTC curves vs NGSPICE |
| `tests/verify_bsimcmg_tran.py` | Transient: single inverter config vs NGSPICE |
| `tests/verify_bsimcmg_tran_comprehensive.py` | Transient: 21-config parametric sweep vs NGSPICE (6 sweeps) |
| `tests/verify_level1_transient.py` | Level 1 transient validation |

Each script generates comparison plots and detailed metrics in `tests/verify_bsimcmg_*_results/`.

---

## Architecture

PyCircuitSim follows a clean, modular architecture:

```
pycircuitsim/                    # Python package (simulator core)
├── __init__.py                  # Public API exports
├── config.py                    # Path configuration (OSDI binary, modelcards)
├── simulation.py                # Simulation orchestration
├── parser.py                    # Netlist parser (HSPICE syntax)
├── circuit.py                   # Circuit topology (nodes, components)
├── solver.py                    # MNA + Newton-Raphson + Transient solvers
├── logger.py                    # HSPICE-like logging (.lis files)
├── visualizer.py                # Matplotlib plotting
└── models/                      # Device model implementations
    ├── __init__.py
    ├── base.py                  # Component abstract base class
    ├── passive.py               # R, C, V, I, PULSE sources
    ├── mosfet.py                # Level 1 Shichman-Hodges model
    └── mosfet_cmg.py            # BSIM-CMG FinFET model (LEVEL=72)

external_compact_models/         # External compact model binaries
└── PyCMG/                       # BSIM-CMG OSDI wrapper (git submodule)
    ├── pycmg/                   # Python ctypes-based OSDI interface
    ├── build-deep-verify/osdi/  # Compiled OSDI binary (bsimcmg.osdi)
    └── tech_model_cards/ASAP7/  # ASAP7 7nm FinFET modelcards

main.py                          # CLI entry point
examples/                        # Example netlists (.sp files)
tests/                           # NGSPICE verification scripts
results/                         # Simulation output (generated at runtime)
```

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `simulation.py` | Orchestrates parse -> solve -> visualize workflow |
| `parser.py` | Two-pass netlist parsing, `.model`/`.include`/`.ic` directives |
| `circuit.py` | Stores circuit topology, component list, node mapping |
| `solver.py` | MNA construction, Newton-Raphson iteration, transient stepping |
| `models/` | Device physics (I-V equations, conductances, capacitances) |
| `logger.py` | HSPICE-compatible output formatting |
| `visualizer.py` | Automatic plot generation |

### Design Principles

1. **Separation of Concerns**
   - Solver builds matrices and iterates (no device equations)
   - Device models compute current/conductances from voltages (no matrix operations)
   - Simulation orchestrates the workflow (parse -> solve -> visualize)

2. **Modularity**
   - All devices inherit from the `Component` base class
   - Common interface: `calculate_current()`, `get_conductances()`
   - New devices can be added without modifying the solver

3. **Compact Model Integration**
   - BSIM-CMG models are accessed via PyCMG's ctypes-based OSDI interface
   - The `mosfet_cmg.py` module wraps PyCMG's `Model`/`Instance` classes
   - Intrinsic capacitances (Cgd, Cgs, Cdd) are extracted for transient analysis

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
4. Apply damping: `v_new = v_old + alpha * dv` (alpha = 0.5 for |dv| >= 1V)
5. Repeat until convergence (max|dv| < 1 uV)

### Source Stepping

Improves convergence for difficult circuits:

1. Start with all sources at 0V
2. Gradually ramp sources to final values (20 steps)
3. Use previous step's solution as initial guess

### Trapezoidal Integration

For transient analysis with capacitors and MOSFET intrinsic capacitances:

```
i(t+dt) = (2C/dt) * [v(t+dt) - v(t)] - i(t)
```

- 2nd order implicit integration (A-stable)
- Converts capacitors to companion conductance + current source
- Also stamps BSIM-CMG intrinsic capacitances (Cgd, Cgs, Cdd) as companion models
- Charge state tracking via `get_charges()`, `init_charge_state()`, `update_charge_state()`

### Convergence Aids

- **Gmin stepping**: Exponentially decaying minimum conductance (1e-9 to 1e-12)
- **Pseudo-transient initialization**: Artificial capacitances for startup (auto-scaled to 5x max circuit cap)
- **Adaptive damping**: Oscillation detection with automatic damping adjustment
- **Voltage clamping**: Vgs +/-5V, Vds +/-10V to prevent numerical overflow

---

## Development

### Adding New Components

1. Create a component class inheriting from `Component` in `pycircuitsim/models/`:

```python
from pycircuitsim.models.base import Component

class Inductor(Component):
    def __init__(self, name: str, nodes: list, value: float):
        super().__init__(name, nodes, value)

    def stamp_conductance(self, mna_matrix, node_map):
        # Add inductor entries to MNA matrix
        pass

    def stamp_rhs(self, rhs, node_map):
        # Add current to RHS vector
        pass

    def calculate_current(self, voltages: dict) -> float:
        # Calculate branch current from node voltages
        pass
```

2. Register in parser (`pycircuitsim/parser.py`)
3. Add to `_is_mosfet()` or equivalent type helpers in solver if needed
4. Write tests

### Coding Style

- **Type hints**: Required for all function signatures
- **Variable names**: Descriptive (`v_gate`, `i_drain`, not `a`, `b`)
- **Docstrings**: Required for public APIs
- **Import order**: stdlib, third-party, local

### Debugging Tips

Enable verbose logging to see Newton-Raphson convergence:

```bash
python main.py circuit.sp -v
```

Check `.lis` files for iteration counts, convergence status, and device currents.

---

## Limitations

PyCircuitSim is intentionally simplified for educational use:

### Not Supported

- Inductors (L)
- Mutual inductance (transformers)
- AC analysis (.ac)
- Noise analysis (.noise)
- Complex directives (.option, .measure, .param)
- Subcircuits (.subckt)

### Known Limitations

- **Speed**: Pure Python is ~10-100x slower than compiled simulators
- **Scale**: Tested on circuits with <100 components
- **Convergence**: May fail on strongly non-linear circuits without source stepping

### When to Use Other Tools

Consider ngspice, Xyce, or Spectre for:
- Production IC design
- Large-scale circuits (>1000 components)
- High-frequency or RF simulation

---

## Future Work

- [ ] Adaptive timestep control (local truncation error estimates)
- [ ] Expanded test suite (NAND/NOR gates, ring oscillator, SRAM)
- [ ] Inductor support
- [ ] AC small-signal analysis
- [ ] Subcircuit support (.subckt)

---

## References

### Software

- **ngspice** - Open-source SPICE simulator ([ngspice.sourceforge.net](http://ngspice.sourceforge.net)). Reference for netlist syntax and device equations.
- **PyCMG** - Python BSIM-CMG OSDI wrapper ([github.com/ShenShan123/PyCMG](https://github.com/ShenShan123/PyCMG)). Provides ctypes-based OSDI interface for compact models.
- **ASAP7 PDK** - Arizona State Predictive 7nm PDK ([github.com/The-OpenROAD-Project/asap7_pdk_r1p7](https://github.com/The-OpenROAD-Project/asap7_pdk_r1p7)). FinFET modelcards used for BSIM-CMG validation.
- **Xyce** - Parallel electronic simulator ([xyce.sandia.gov](https://xyce.sandia.gov)). Architectural patterns for solver-device separation.

### Academic

- Shichman, H., & Hodges, D. A. (1968). "Modeling and Simulation of Insulated-Gate Field-Effect Transistor Switching Circuits." *IEEE JSSC*.
- Nagel, L. W. (1975). "SPICE2: A Computer Program to Simulate Semiconductor Circuits." *ERL-M520*, UC Berkeley.

---

## License

MIT License - see LICENSE file for details.

---

## Acknowledgments

Developed as an educational tool to demonstrate SPICE-like simulation in pure Python. Inspired by ngspice, Xyce, and the original SPICE2 from UC Berkeley.

---

## Contact

- GitHub Issues: https://github.com/ShenShan123/PyCircuitSim/issues
- Discussions: https://github.com/ShenShan123/PyCircuitSim/discussions
