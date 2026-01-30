# Project: PyCircuitSim

## Overview
Python-based SPICE-like circuit simulator emphasizing educational clarity and modular architecture. Supports DC/transient analysis with Level 1 and BSIM4V5 (45nm) MOSFET models.

**Core Principles:**
* Pure Python with clean, readable code
* Complete decoupling: Solver ↔ Device Models
* Basic HSPICE netlist compatibility

## Architecture

### Module Structure
* `pycircuitsim/parser.py` - Two-pass netlist parsing, `.model` directive support
* `pycircuitsim/circuit.py` - Circuit topology management
* `pycircuitsim/solver.py` - MNA matrix construction, Newton-Raphson solver
* `pycircuitsim/models/base.py` - Abstract `Component` base class
* `pycircuitsim/models/passive.py` - R, C, V, I sources (including PWL/PULSE)
* `pycircuitsim/models/mosfet.py` - Level 1 Shichman-Hodges model
* `pycircuitsim/models/bsim4v5/` - Industry-standard Level 54 compact model
* `pycircuitsim/logger.py` - HSPICE-like .lis output
* `pycircuitsim/visualizer.py` - Matplotlib plotting
* `examples/*.sp` - Example netlists for testing and debugging (DC, transient analysis)
* `results/` - Directory for simulation output files (`.lis`, `.png`)

### FreePDK45 for BSIM4V5 MOSFET Model
* `freePDK45nm_spice/` - Directory for freePDK45nm model card (`.l` files)

### Key Algorithms
* **MNA (Modified Nodal Analysis)** - Circuit equation matrix construction
* **Newton-Raphson** - Non-linear circuit solver
* **Backward Euler** - Capacitor integration for transient analysis
* **Source Stepping** - Two-stage analysis for improved convergence

## Supported Features

### Devices
* Passive: R, C
* Active: NMOS/PMOS (Level 1, BSIM4V5)
* Sources: DC voltage/current, PWL, PULSE

### Analysis
* `.dc` - DC sweep analysis
* `.tran` - Transient analysis

### Directives
* `.model` - MOSFET model definitions (LEVEL=1 or 54)
* `.include` - External library files
* `.ic` - Initial conditions (critical for SRAM/bistable circuits)

## Status

### Phase 1: Core Implementation ✅ Complete
- [x] MNA matrix construction
- [x] Level 1 MOSFET model
- [x] Newton-Raphson solver
- [x] Transient analysis with capacitors

### Phase 2: Enhancements ✅ Complete
- [x] HSPICE-like logging (.lis files)
- [x] Voltage clamping for numerical stability
- [x] Two-stage DC analysis
- [x] Enhanced visualization

### Phase 3: BSIM4V5 Integration 
- [x] C library wrapper with ctypes
- [x] PTM 45nm PDK integration
- [x] Parser enhancements for `.model` directive
- [x] Standalone model verification
- [x] Circuit simulation with Newton-Raphson convergence

### Phase 4: Stabilization & Production Ready 
- [x] Fixed BSIM4V5 Newton-Raphson divergence
- [x] Fixed BSIM4V5 characteristic curve discontinuities
- [x] Comprehensive numerical validation
- [x] Cleaned up debug/test artifacts

### Phase 5: Documentation & Validation 
- [x] Created comprehensive BSIM4V5 integration status document
- [x] Validated current accuracy vs ngspice (0.13% error)
- [x] Documented known limitations and future work
- [x] Production ready for educational and research use

### BSIM4V5 Model Library
The BSIM4V5 model is implemented as a compiled C library with Python ctypes bindings:
- **Library path**: `pycircuitsim/models/bsim4v5/bridge/libbsim4v5.so`
- **Source files**: `pycircuitsim/models/bsim4v5/bridge/*.c`
- **Build**: Run `make` in the bridge directory to compile the library

---

## Quick Start

### Basic Simulation
Create a netlist (`.sp` file) with your circuit. Examples provided in `examples/` directory.

```bash
# RC transient analysis example
python main.py examples/rc_transient.sp

# Custom circuit example
python main.py your_circuit.sp
```

### MOSFET Terminal Order
**Important**: Terminals are `drain gate source bulk`

```spice
* NMOS: drain=output, gate=input, source=GND, bulk=GND
Mn1 3 2 0 0 NMOS_VTL L=45n W=90n

* PMOS: drain=output, gate=input, source=Vdd, bulk=Vdd
Mp1 3 2 1 1 PMOS_VTL L=45n W=180n
```

### Output Files
Results organized in `results/<circuit_name>/<analysis_type>/`:
- `*_simulation.lis` - Detailed iteration log (HSPICE-like)
- `*_dc_sweep.csv` - Waveform data (node voltages + device currents)
- `*_dc_sweep.png` - Voltage/current plots

## Development Guidelines

### Coding Standards
- Type hints required for all function signatures
- Clear variable names (e.g., `v_gate`, `i_drain`, not `a`, `b`)
- Docstrings for complex algorithms
- Voltage clamping: Vgs ± 5V, Vds ± 10V

### Separation Principle
- **Solver** builds MNA matrix, executes Newton-Raphson (no device equations)
- **Device Models** calculate current/conductances from voltages (no matrix operations)
- All devices inherit from `Component` base class

### Key Numerical Techniques
- Minimum conductance (1µS) prevents singular matrices
- Source stepping (20 steps) improves convergence
- Damping factor (0.5) for large voltage deltas
- Two-stage analysis: DC OP → DC sweep/transient
- Voltage-source-constrained nodes exempt from damping

## References
- **ngspice** - Physics equation verification
- **Xyce** - Architecture patterns for device/solver separation
- **BSIM4V5 spec** - Industry-standard compact model

## Other Memoos
- Use conda environment `pycircuitsim`
- Git commit for every significant change