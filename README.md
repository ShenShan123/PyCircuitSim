# PyCircuitSim

Simple Python Circuit Simulator - An open-source, pure Python circuit simulator designed for educational purposes and architectural clarity.

## Features

- **Pure Python Implementation**: Clean, readable codebase with no external dependencies beyond NumPy and Matplotlib
- **HSPICE-Compatible Netlists**: Support for standard SPICE syntax for components and analysis commands
- **Multiple Analyses**:
  - DC Sweep Analysis (`.dc`)
  - Transient Analysis (`.tran`)
  - DC Operating Point
- **Component Support**:
  - Resistors (R)
  - Capacitors (C)
  - DC Voltage Sources (V)
  - DC Current Sources (I)
  - MOSFETs (NMOS/PMOS) with Level 1 Shichman-Hodges model
- **Visualization**: Automatic plot generation for simulation results

## Installation

### Requirements

- Python 3.10 or higher
- NumPy
- Matplotlib

### Install from Source

```bash
git clone <repository-url>
cd NN_SPICE
pip install -r requirements.txt
```

### Install with Tsinghua Mirror (China)

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

## Quick Start

### Basic Usage

Run a simulation from the command line:

```bash
python main.py examples/voltage_divider.sp
```

Specify output directory:

```bash
python main.py examples/inverter.sp -o my_results
```

Enable verbose logging:

```bash
python main.py examples/rc_circuit.sp -v
```

### Python API

```python
from pycircuitsim.main import run_simulation

# Run simulation programmatically
run_simulation(
    netlist_path='circuit.sp',
    output_dir='results',
    verbose=True
)
```

## Netlist Syntax

PyCircuitSim supports HSPICE-like netlist format:

### Components

```
* Resistors: R<name> <n1> <n2> <value>
R1 1 2 1k
R2 2 0 10k

* Capacitors: C<name> <n1> <n2> <value>
C1 2 0 100p

* Voltage Sources: V<name> <n+> <n-> <value>
V1 1 0 3.3

* Current Sources: I<name> <n+> <n-> <value>
I1 1 0 1m

* MOSFETs: M<name> <d> <g> <s> <b> <model> L=<l> W=<w>
M1 3 2 0 0 NMOS L=1u W=10u
```

### Value Suffixes

Supported unit suffixes:
- `k` or `K`: kilo (1e3)
- `u` or `U`: micro (1e-6)
- `n` or `N`: nano (1e-9)
- `p` or `P`: pico (1e-12)

### Analysis Commands

```
* DC Sweep: .dc <source> <start> <stop> <step>
.dc Vin 0 3.3 0.1

* Transient: .tran <tstep> <tstop>
.tran 1n 100n
```

## Examples

### Example 1: Voltage Divider

```spice
* Voltage Divider Circuit
V1 1 0 10
R1 1 2 1k
R2 2 0 1k
.dc V1 0 10 1
.end
```

### Example 2: CMOS Inverter

```spice
* CMOS Inverter
Vdd 1 0 3.3
Vin 2 0 0
Mp1 1 2 3 1 PMOS L=1u W=20u
Mn1 0 2 3 0 NMOS L=1u W=10u
.dc Vin 0 3.3 0.1
.end
```

### Example 3: RC Circuit (Transient)

```spice
* RC Charging Circuit
V1 1 0 5
R1 1 2 1k
C1 2 0 1n
.tran 100n 10u
.end
```

## Output

Simulation results are saved to the specified output directory (default: `results/`):

- `dc_sweep.png` - DC sweep analysis plots
- `transient.png` - Transient analysis plots
- `dc_op_point.txt` - DC operating point results (if no analysis specified)

## Architecture

PyCircuitSim follows a modular, extensible architecture:

```
pycircuitsim/
├── parser.py      # Netlist parser
├── circuit.py     # Circuit topology container
├── solver.py      # DC and transient solvers
├── visualizer.py  # Plot generation
├── main.py        # Simulation orchestration
└── models/
    ├── base.py    # Component base class
    ├── passive.py # R, C, V, I components
    └── mosfet.py  # MOSFET Level 1 model
```

### Key Design Principles

1. **Separation of Concerns**: The solver engine is completely decoupled from device physics
2. **Modularity**: Each component type is implemented independently with a common interface
3. **Extensibility**: New components can be added by inheriting from `Component` base class

## Algorithms

- **Modified Nodal Analysis (MNA)**: Constructs and solves circuit equations
- **Newton-Raphson**: Iterative method for non-linear circuits (MOSFETs)
- **Backward Euler**: Numerical integration for transient analysis

## Testing

Run the test suite:

```bash
pytest tests/
```

Run specific test modules:

```bash
pytest tests/test_parser.py
pytest tests/test_solver.py
pytest tests/test_transient.py
```

## Limitations

- Only supports Level 1 MOSFET model (Shichman-Hodges)
- Does not support inductors
- PWL and PULSE sources are planned but not yet implemented
- Complex directives (.option, .measure, .param) are intentionally ignored for simplicity

## Contributing

Contributions are welcome! The project prioritizes:
- Code clarity and readability
- Educational value
- Clean architecture over optimization

## License

[Specify your license here]

## References

- **ngspice**: Reference for SPICE syntax and MOS Level 1 equations
- **Xyce**: Architectural patterns for solver-device separation
- **Shichman-Hodges Model**: MOSFET Level 1 compact model

## Acknowledgments

Developed as an educational circuit simulator to demonstrate SPICE-like simulation in pure Python.
