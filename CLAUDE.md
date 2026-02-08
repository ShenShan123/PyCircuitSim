# Project: PyCircuitSim

## Overview
Python-based SPICE-like circuit simulator emphasizing educational clarity and modular architecture. Supports DC/transient analysis with Level 1 MOSFET models.

**Core Principles:**
* Pure Python with clean, readable code
* Complete decoupling: Solver ↔ Device Models
* Basic HSPICE netlist compatibility

## Architecture

### Module Structure
```
pycircuitsim/
├── __init__.py         # Package initialization, exports public API
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
    └── mosfet.py       # Level 1 Shichman-Hodges model

main.py                 # CLI entry point (single main entrance)
examples/*.sp           # Example netlists
results/                # Simulation output (.lis, .csv, .png)
```

### Key Algorithms
* **MNA (Modified Nodal Analysis)** - Circuit equation matrix construction
* **Newton-Raphson** - Non-linear circuit solver
* **Backward Euler** - Capacitor integration for transient analysis
* **Source Stepping** - Two-stage analysis for improved convergence

## Supported Features

### Devices
* Passive: R, C
* Active: NMOS/PMOS (Level 1 Shichman-Hodges)
* Sources: DC voltage/current, PULSE

### Analysis
* `.dc` - DC sweep analysis
* `.tran` - Transient analysis

### Directives
* `.model` - MOSFET model definitions (LEVEL=1 only)
* `.include` - External library files
* `.ic` - Initial conditions (critical for SRAM/bistable circuits)

## Status

### Phase 1: Core Implementation ✅ Complete
- [x] MNA matrix construction
- [x] Level 1 MOSFET model (Shichman-Hodges)
- [x] Newton-Raphson solver
- [x] Transient analysis with capacitors

### Phase 2: Enhancements ✅ Complete
- [x] HSPICE-like logging (.lis files)
- [x] Voltage clamping for numerical stability
- [x] Two-stage DC analysis
- [x] Enhanced visualization

### Phase 3: Production Ready ✅ Complete
- [x] Comprehensive numerical validation
- [x] Clean project structure
- [x] Documentation and examples

---

## Quick Start

### Basic Simulation
Create a netlist (`.sp` file) with your circuit. Examples provided in `examples/` directory.

```bash
# Set PYTHONPATH if not installed
export PYTHONPATH=/path/to/NN_SPICE:$PYTHONPATH

# Run simulation
python main.py examples/rc_transient.sp
python main.py examples/test_nmos_level1.sp

# Custom circuit
python main.py your_circuit.sp
```

### MOSFET Terminal Order
**Important**: Terminals are `drain gate source bulk`

```spice
* NMOS: drain=output, gate=input, source=GND, bulk=GND
Mn1 3 2 0 0 NMOS_VTL L=1u W=10u

* PMOS: drain=output, gate=input, source=Vdd, bulk=Vdd
Mp1 3 2 1 1 PMOS_VTL L=1u W=20u
```

### Python API
```python
from pycircuitsim.simulation import run_simulation

# Run simulation programmatically
run_simulation(
    netlist_path='circuit.sp',
    output_dir='results',
    verbose=True
)
```

### Output Files
Results organized in `results/<circuit_name>/<analysis_type>/`:
- `*_simulation.lis` - Detailed iteration log (HSPICE-like)
- `*_dc_sweep.csv` / `*_transient.csv` - Waveform data (node voltages + device currents)
- `*_dc_sweep.png` / `*_transient.png` - Voltage/current plots

## Development Guidelines

### Coding Standards
- Type hints required for all function signatures
- Clear variable names (e.g., `v_gate`, `i_drain`, not `a`, `b`)
- Docstrings for complex algorithms
- Voltage clamping: Vgs ± 5V, Vds ± 10V

### Separation Principle
- **Solver** (`solver.py`) builds MNA matrix, executes Newton-Raphson (no device equations)
- **Device Models** (`models/`) calculate current/conductances from voltages (no matrix operations)
- **Simulation** (`simulation.py`) orchestrates the workflow (parse → solve → visualize)
- All devices inherit from `Component` base class

### Key Numerical Techniques
- Minimum conductance (1µS) prevents singular matrices
- Source stepping (20 steps) improves convergence
- Damping factor (0.5) for large voltage deltas
- Two-stage analysis: DC OP → DC sweep/transient
- Voltage-source-constrained nodes exempt from damping

### Entry Points
- **CLI**: `main.py` - Command-line interface (argparse, error handling)
- **API**: `pycircuitsim.simulation.run_simulation()` - Programmatic access
- **Module**: `pycircuitsim` - Package exports (Circuit, Parser, Visualizer, run_simulation)

## References
- **ngspice** - Physics equation verification
- **Xyce** - Architecture patterns for device/solver separation
- **Shichman-Hodges Model** - Level 1 MOSFET compact model

## Other Notes
- Use conda environment `pycircuitsim`
- Git commit for every significant change
- Single main entrance: `main.py` at project root
