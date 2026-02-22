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

### Requirements

- Python 3.10 or higher
- NumPy 1.20+
- Matplotlib 3.3+

### Install from Source

```bash
# Clone the repository
git clone https://github.com/ShenShan123/PyCircuitSim.git
cd PyCircuitSim

# Install dependencies (international users: use Tsinghua mirror)
pip install -r requirements.txt
# or in China:
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

### Development Installation

```bash
# Install in editable mode
pip install -e .

# Run tests
pytest tests/
```

---

## Quick Start

### Running Simulations

```bash
# Basic usage
python main.py examples/rc_transient.sp

# Specify output directory
python main.py examples/test_nmos_level1.sp -o my_results

# Enable verbose logging
python main.py examples/test_cmos_inverter_level1.sp -v
```

### Your First Circuit

Create a file `voltage_divider.sp`:

```spice
* Simple Voltage Divider
* Input voltage
Vin 1 0 10

* Resistors
R1 1 2 1k
R2 2 0 1k

* DC sweep: vary Vin from 0 to 10V
.dc Vin 0 10 1

.end
```

Run the simulation:

```bash
python main.py voltage_divider.sp
```

Results are saved to `results/voltage_divider/dc/`:
- `voltage_divider_dc_sweep.csv` - Numerical data
- `voltage_divider_dc_sweep.png` - Plot

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
| `k`, `K` | 10³ | `1k` = 1,000 |
| `u`, `U` | 10⁻⁶ | `1u` = 0.000001 |
| `n`, `N` | 10⁻⁹ | `1n` = 0.000000001 |
| `p`, `P` | 10⁻¹² | `1p` = 0.000000000001 |

### Analysis Commands

```spice
* DC Sweep: .dc <source> <start> <stop> <step>
.dc Vin 0 3.3 0.1

* Transient: .tran <tstep> <tstop>
.tran 1n 100n
```

### MOSFET Models

```spice
* NMOS Level 1 Model
.model NMOS_VTL NMOS (
    LEVEL=1
    VTO=0.7      ; Threshold voltage
    KP=110u      ; Transconductance
    GAMMA=0.4    ; Body effect
    LAMBDA=0.02  ; Channel-length modulation
)

* PMOS Level 1 Model
.model PMOS_VTL PMOS (
    LEVEL=1
    VTO=-0.7
    KP=50u
    GAMMA=0.5
    LAMBDA=0.03
)
```

### PULSE Sources

```spice
* PULSE: V<name> <n+> <n-> PULSE(V1 V2 TD TR TF PW PER)
Vclk 1 0 PULSE(0 3.3 0n 1n 1n 10n 20n)

* Parameters:
* V1  : Initial value
* V2  : Pulsed value
* TD  : Delay time
* TR  : Rise time
* TF  : Fall time
* PW  : Pulse width
* PER : Period
```

---

## Examples

### Example 1: Common-Source Amplifier

```spice
* Common-Source NMOS Amplifier

* Bias and input
Vdd 1 0 5
Vbias 2 0 2.5
Vin 3 0 0 AC 1

* MOSFET
Mn1 1 2 0 0 NMOS_VTL L=1u W=10u

* Load resistor
Rd 1 0 10k

* Input resistor
Rin 3 2 1k

* NMOS Model
.model NMOS_VTL NMOS (
    LEVEL=1
    VTO=1.0
    KP=100u
    GAMMA=0.3
    LAMBDA=0.01
)

* DC sweep of input
.dc Vin 0 3 0.1

.end
```

### Example 2: CMOS Inverter

```spice
* CMOS Inverter - Voltage Transfer Characteristic

* Power supply
Vdd 1 0 3.3

* Input voltage (swept)
Vin 2 0 1.65

* PMOS transistor (drain=3, gate=2, source=1, bulk=1)
Mp1 3 2 1 1 PMOS_VTL L=1u W=20u

* NMOS transistor (drain=3, gate=2, source=0, bulk=0)
Mn1 3 2 0 0 NMOS_VTL L=1u W=10u

* NMOS model
.model NMOS_VTL NMOS (LEVEL=1 VTO=0.7 KP=110u GAMMA=0.4 LAMBDA=0.02)

* PMOS model
.model PMOS_VTL PMOS (LEVEL=1 VTO=-0.7 KP=50u GAMMA=0.5 LAMBDA=0.03)

* DC sweep: Vin from 0 to 3.3V
.dc Vin 0 3.3 0.1

.end
```

### Example 3: RC Circuit (Transient)

```spice
* RC Charging Circuit - Transient Analysis

* Voltage source (step input)
V1 1 0 5

* Resistor and capacitor
R1 1 2 1k
C1 2 0 1n

* Initial condition: capacitor initially uncharged
.ic V(2)=0

* Transient analysis: 1ns step, 10µs total
.tran 1n 10u

.end
```

### Example 4: Ring Oscillator

```spice
* 3-Stage Ring Oscillator

* Power supply
Vdd 1 0 3.3

* Stage 1
Mp1 2 0 1 1 PMOS_VTL L=1u W=10u
Mn1 2 0 3 0 NMOS_VTL L=1u W=5u

* Stage 2
Mp2 4 2 1 1 PMOS_VTL L=1u W=10u
Mn2 4 2 5 0 NMOS_VTL L=1u W=5u

* Stage 3
Mp3 6 4 1 1 PMOS_VTL L=1u W=10u
Mn3 6 4 7 0 NMOS_VTL L=1u W=5u

* Feedback connection
Rfb 6 0 1M

* Load capacitors
C1 2 0 100f
C2 4 0 100f
C3 6 0 100f

* Initial conditions to kickstart oscillation
.ic V(2)=1.65

* MOSFET models
.model NMOS_VTL NMOS (LEVEL=1 VTO=0.7 KP=110u)
.model PMOS_VTL PMOS (LEVEL=1 VTO=-0.7 KP=50u)

* Transient analysis
.tran 100p 50n

.end
```

### Example 5: BSIM-CMG FinFET Inverter (ASAP7 7nm)

```spice
* BSIM-CMG CMOS Inverter - ASAP7 7nm FinFET
* VDD=0.7V, L=30nm, NFIN=10

* Power supply
Vdd 1 0 0.7

* Input pulse
Vin 2 0 PULSE 0 0.7 0.5n 0.1n 0.1n 0.8n 2n

* PMOS (drain=out, gate=in, source=Vdd, bulk=Vdd)
Mp1 3 2 1 1 pmos1 L=30n NFIN=10

* NMOS (drain=out, gate=in, source=GND, bulk=GND)
Mn1 3 2 0 0 nmos1 L=30n NFIN=10

* Load capacitance
Cload 3 0 10f

* Initial condition
.ic V(3)=0.7

* Model definitions (LEVEL=72 = BSIM-CMG via PyCMG)
.model nmos1 NMOS (LEVEL=72)
.model pmos1 PMOS (LEVEL=72)

* Transient analysis
.tran 10p 5n

.end
```

---

## Python API

### Basic Usage

```python
from pycircuitsim.simulation import run_simulation

# Run a simulation
run_simulation(
    netlist_path='circuit.sp',
    output_dir='my_results',
    verbose=True
)
```

### Advanced: Direct API Access

```python
from pycircuitsim import Parser, DCSolver
import matplotlib.pyplot as plt

# Parse netlist
parser = Parser()
parser.parse_file('circuit.sp')
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

### Custom Visualization

```python
from pycircuitsim import Parser, DCSolver, Visualizer
import numpy as np

# Parse and solve
parser = Parser()
parser.parse_file('circuit.sp')
circuit = parser.circuit

# Parameter sweep
source_values = np.linspace(0, 3.3, 100)
results = {'V(in)': source_values}

for v in source_values:
    # Modify source value
    circuit.components[0].value = v

    # Solve
    solver = DCSolver(circuit)
    solution = solver.solve()

    # Store output
    results.setdefault('V(out)', []).append(solution.get('out', 0))

# Plot
visualizer = Visualizer()
visualizer.plot_dc_sweep(
    sweep_values=source_values,
    results=results,
    sweep_variable='Input (V)',
    output_path='custom_plot.png'
)
```

---

## Output Files

PyCircuitSim generates several output files organized by circuit name and analysis type:

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
- Newton-Raphson iterations
- Convergence status
- Final node voltages and device currents

Example:
```
================================================================================
                    PyCircuitSim - DC Sweep Analysis
================================================================================

Circuit Summary:
  Components: 3
  Nodes: 4
  Voltage Sources: 1

--------------------------------------------------------------------------------
Sweep Point 0: Vin = 0.00 V
--------------------------------------------------------------------------------
  Newton-Raphson Iteration 1:
    max_delta = 2.5000e+00 V
  Newton-Raphson Iteration 2:
    max_delta = 1.2500e-01 V
  Newton-Raphson Iteration 3:
    max_delta = 3.1250e-03 V
  Converged in 3 iterations

Final Results:
  V(1) = 0.0000e+00 V
  V(2) = 0.0000e+00 V
  i(V1) = -1.0000e-03 A
```

### CSV Data Files

Column-oriented data suitable for plotting in Excel, MATLAB, or Python:

```csv
Vin (V),V(1),V(2),i(V1),i(Mn1)
0.000000,0.000000e+00,0.000000e+00,-1.000000e-03,-1.000000e-03
0.100000,1.000000e-01,9.900990e-02,-9.900990e-04,-9.900990e-04
...
```

---

## Architecture

PyCircuitSim follows a clean, modular architecture:

```
pycircuitsim/
├── __init__.py         # Package initialization, public API exports
├── config.py           # Path configuration (OSDI binary, modelcards)
├── simulation.py       # Simulation orchestration (high-level workflow)
├── parser.py           # Netlist parser (HSPICE syntax)
├── circuit.py          # Circuit topology (nodes, components)
├── solver.py           # MNA + Newton-Raphson + Transient solvers
├── logger.py           # HSPICE-like logging (.lis files)
├── visualizer.py       # Matplotlib plotting
└── models/
    ├── __init__.py
    ├── base.py         # Component abstract base class
    ├── passive.py      # R, C, V, I, PULSE sources
    ├── mosfet.py       # Level 1 Shichman-Hodges model
    └── mosfet_cmg.py   # BSIM-CMG FinFET model (LEVEL=72) via PyCMG

models/PyCMG/           # BSIM-CMG OSDI wrapper (git submodule)
main.py                 # CLI entry point
examples/*.sp           # Example netlists
results/                # Simulation output
tests/                  # NGSPICE validation scripts
```

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `simulation.py` | Orchestrates parse → solve → visualize workflow |
| `parser.py` | Two-pass netlist parsing, model definitions |
| `circuit.py` | Stores circuit topology, component list |
| `solver.py` | MNA construction, Newton-Raphson iteration |
| `models/` | Device physics (I-V equations, conductances) |
| `logger.py` | HSPICE-compatible output formatting |
| `visualizer.py` | Automatic plot generation |

### Design Principles

1. **Separation of Concerns**
   - Solver doesn't know device physics
   - Device models don't touch matrices
   - Simulation orchestrates but doesn't compute

2. **Modularity**
   - Each component independently implemented
   - Common `Component` base class interface
   - Easy to add new device types

3. **Extensibility**
   - Inherit from `Component` to add devices
   - Implement `stamp_conductance()` and `stamp_rhs()`
   - Solver automatically handles new devices

---

## Algorithms

### Modified Nodal Analysis (MNA)

PyCircuitSim uses MNA to construct circuit equations:

```
[G  C] [v]     [i]
[      ] [ ] = [ ]  (Conductance matrix)
[Cᵀ 0] [j]     [v]     (Current unknowns)
```

- **G**: Conductance matrix (resistive elements)
- **C**: Incidence matrix (voltage sources)
- **v**: Node voltages
- **j**: Source currents

### Newton-Raphson Iteration

For non-linear circuits (MOSFETs):

1. Linearize devices at current operating point
2. Construct MNA matrix
3. Solve for voltage update Δv
4. Apply damping: v_new = v_old + α·Δv (α = 0.5)
5. Repeat until convergence (max|Δv| < 1µV)

### Source Stepping

Improves convergence for difficult circuits:

1. Start with all sources at 0V
2. Gradually ramp sources to final values (20 steps)
3. Use previous step's solution as initial guess
4. Reduces risk of convergence failures

### Trapezoidal Integration

For transient analysis with capacitors:

```
i(t+Δt) = 2C/Δt · [v(t+Δt) - v(t)] - i(t)
```

- 2nd order implicit integration (A-stable)
- Converts capacitors to companion conductance + current source
- Also stamps BSIM-CMG intrinsic capacitances (Cgd, Cgs, Cdd) as companion models

---

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_parser.py
pytest tests/test_solver.py
pytest tests/test_transient.py

# Run with coverage
pytest --cov=pycircuitsim tests/
```

### Adding New Components

1. **Create component class** in `models/`:

```python
from pycircuitsim.models.base import Component

class Inductor(Component):
    def __init__(self, name: str, nodes: list, value: float):
        super().__init__(name, nodes, value)

    def stamp_conductance(self, mna_matrix, node_map):
        # Add inductor conductance to MNA matrix
        pass

    def stamp_rhs(self, rhs, node_map):
        # Add current to RHS vector
        pass

    def calculate_current(self, voltages):
        # Calculate branch current from voltages
        pass
```

2. **Register in parser** (`parser.py`):

```python
def _parse_component(self, line: str):
    if line.startswith('L'):
        self._parse_inductor(line)
    # ...
```

3. **Add tests**:

```python
def test_inductor_dc():
    # Test inductor in DC circuit
    pass
```

### Coding Style

- **Type hints**: Required for all function signatures
- **Variable names**: Descriptive (e.g., `v_gate`, not `vg`)
- **Docstrings**: Required for public APIs
- **Maximum line length**: 100 characters
- **Import order**: stdlib → third-party → local

### Debugging Tips

Enable verbose logging:

```bash
python main.py circuit.sp -v
```

Check `.lis` files for:
- Newton-Raphson convergence status
- Matrix condition numbers
- Device currents at each iteration

Common issues:
- **Singular matrix**: Add minimum conductance (1µS)
- **Divergence**: Try source stepping or voltage clamping
- **Slow convergence**: Reduce damping factor or increase steps

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

## Current Status

### Complete Features

- [x] MNA matrix construction
- [x] Level 1 MOSFET (Shichman-Hodges)
- [x] BSIM-CMG FinFET (LEVEL=72) via PyCMG/OSDI
- [x] Newton-Raphson solver with damping
- [x] Source stepping for convergence
- [x] DC operating point, DC sweep, and transient analysis
- [x] Trapezoidal integration (2nd order) with intrinsic capacitances
- [x] PULSE voltage sources
- [x] HSPICE-like logging (.lis)
- [x] CSV data export and automatic plot generation
- [x] Initial conditions (.ic)
- [x] Python API

### NGSPICE Verification (ASAP7 7nm)

All results validated against NGSPICE 45.2 with BSIM-CMG OSDI:

| Test | Metric | Result |
|------|--------|--------|
| NMOS/PMOS OP | Relative error | < 0.02% |
| DC sweep (Id-Vgs, VTC) | NRMSE | < 0.1% |
| Transient (baseline) | NRMSE (post-settling) | 0.23% |
| Comprehensive (14 configs) | NRMSE (worst case) | 0.95% |

The comprehensive suite sweeps VDD (0.5-0.8V), Cload (1-100fF), input slew (10-500ps), and pulse width (0.2-2.0ns). All 14 configurations pass with NRMSE well under 5%.

### Future Work

- [ ] Inductor support
- [ ] AC small-signal analysis
- [ ] Subcircuit support
- [ ] Adaptive timestep control
- [ ] Expanded test suite (NAND/NOR, ring oscillator, SRAM)

---

## Contributing

Contributions are welcome! Areas of interest:

1. **New device models** (diodes, BJTs, op-amps)
2. **Analysis types** (AC, noise, sensitivity)
3. **Performance** (JIT compilation, GPU acceleration)
4. **Documentation** (tutorials, examples)
5. **Tests** (validation vs ngspice)

### Pull Request Guidelines

- Write clear commit messages (Conventional Commits)
- Add tests for new features
- Update documentation
- Ensure all tests pass: `pytest tests/`

---

## References

### Academic Papers

- **Shichman, H., & Hodges, D. A.** (1968). "Modeling and Simulation of Insulated-Gate Field-Effect Transistor Switching Circuits." *IEEE Journal of Solid-State Circuits*.

- **Nagel, L. W.** (1975). "SPICE2: A Computer Program to Simulate Semiconductor Circuits." *Memorandum No. ERL-M520*, UC Berkeley.

### Software

- **ngspice**: Open-source SPICE simulator
  - Website: http://ngspice.sourceforge.net
  - Reference for netlist syntax and device equations

- **PyCMG**: Python BSIM-CMG OSDI wrapper
  - Repository: https://github.com/ShenShan123/PyCMG
  - Provides ctypes-based OSDI interface for compact models

- **ASAP7 PDK**: Arizona State Predictive 7nm PDK
  - Repository: https://github.com/The-OpenROAD-Project/asap7_pdk_r1p7
  - FinFET modelcards used for BSIM-CMG validation

- **Xyce**: Parallel electronic simulator
  - Website: https://xyce.sandia.gov
  - Architectural patterns for solver-device separation

### Books

- **Rabaey, J. M., Chandrakasan, A., & Nikolic, B.** (2003). *Digital Integrated Circuits: A Design Perspective*. Prentice Hall.

- **Razavi, B.** (2001). *Design of Analog CMOS Integrated Circuits*. McGraw-Hill.

- **Sedra, A. S., & Smith, K. C.** (2014). *Microelectronic Circuits*. Oxford University Press.

---

## License

MIT License - see LICENSE file for details.

---

## Acknowledgments

Developed as an educational tool to demonstrate SPICE-like simulation in pure Python. Inspired by ngspice, Xyce, and the original SPICE2 from UC Berkeley.

**Purpose**: Teaching circuit simulation, numerical methods, and software architecture to students and engineers.

---

## Contact

For questions, issues, or contributions:
- GitHub Issues: https://github.com/ShenShan123/PyCircuitSim/issues
- Discussions: https://github.com/ShenShan123/PyCircuitSim/discussions

---

<div align="center">

**Happy Simulating! 📈⚡**

</div>
