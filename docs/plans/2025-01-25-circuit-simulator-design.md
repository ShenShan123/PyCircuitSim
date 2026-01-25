# Circuit Simulator Design Document

**Date:** 2025-01-25
**Project:** PyCircuitSim - Python Circuit Simulator
**Purpose:** Educational, pure-python circuit simulator with HSPICE-like syntax

---

## 1. Architecture Overview

The simulator follows a **layered architecture** with strict separation of concerns:

### Layer 1 - Core Models (`models/`)
- Abstract `Component` base class defining the device interface
- Device implementations: R, C, V_source, I_source, NMOS, PMOS
- Each device owns its physics equations (current, conductance derivatives)

### Layer 2 - Circuit Topology (`circuit.py`)
- `Circuit` class maintains netlist topology
- Maps node names to numerical indices for MNA matrix
- Holds list of all component instances

### Layer 3 - Solver Engine (`solver.py`)
- `DCSolver`: Steady-state analysis using Newton-Raphson
- `TransientSolver`: Time-domain simulation using Backward Euler
- Builds/solves MNA matrix system Ax=b
- **No device physics** - only numerical methods

### Layer 4 - Parser (`parser.py`)
- Reads `.sp` netlist files
- Validates syntax and extracts component definitions
- Creates Component instances and populates Circuit

### Layer 5 - Visualization (`visualizer.py`)
- Plots results using Matplotlib
- Supports DC sweeps and transient waveforms

### Key Design Principle
The Solver only knows about "devices with conductances and currents" - it never contains MOS equations or capacitor formulas. All physics lives in Component subclasses.

---

## 2. Component Interface & MNA Integration

### Abstract Component Interface
```python
class Component(ABC):
    @abstractmethod
    def get_nodes(self) -> List[str]:
        """Return list of node names this component connects to"""

    @abstractmethod
    def stamp_conductance(self, matrix: np.ndarray, node_map: Dict[str, int]) -> None:
        """Add conductance terms to the MNA matrix (G part)"""

    @abstractmethod
    def stamp_rhs(self, rhs: np.ndarray, node_map: Dict[str, int]) -> None:
        """Add current/source terms to the RHS vector (z part)"""

    @abstractmethod
    def calculate_current(self, voltages: Dict[str, float]) -> float:
        """Calculate device current given terminal voltages"""
```

### MNA Matrix Structure
For circuit with N nodes (excluding ground) and M voltage sources:
- Matrix size: (N + M) × (N + M)
- Top-left N×N: Conductance matrix (G)
- Top-right N×M: Voltage source coupling (B)
- Bottom-left M×N: Voltage source coupling (C)
- Bottom-right M×M: Zeros for ideal sources

### Device Stamping Rules
- **Resistor** (R, nodes i-j): Add ±1/R to G[i,i], G[j,j], G[i,j], G[j,i]
- **Voltage Source**: Add +1/-1 to B and C matrices
- **Current Source**: Only stamps RHS vector
- **MOSFET**: Conductance changes each Newton iteration

---

## 3. Solver Algorithms

### DC Solver (Newton-Raphson)

For linear circuits: single matrix solve.

For MOSFET circuits (non-linear):
```
1. Initial voltage guess (0V or previous solution)
2. Build MNA matrix with conductances at current voltages
3. Solve Ax = b for new voltages
4. Check convergence: ||v_new - v_old|| < tolerance
5. If not converged, update guess and repeat
6. Limit iterations (50-100) to catch divergence
```

**Convergence Criteria:**
- Absolute voltage change < 1e-6 V
- Relative voltage change < 1e-6
- Combined: `|Δv| < atol + rtol * |v|`

### Transient Solver (Backward Euler)

Capacitors become time-dependent using Backward Euler discretization:
```
C * dV/dt = I  →  C * (V[t] - V[t-1]) / Δt = I[t]
```

Transforms to equivalent conductance + current source:
- **Equivalent Conductance:** G_eq = C / Δt
- **Equivalent Current:** I_eq = (C / Δt) * V[t-1]

**Time-Stepping Logic:**
1. Start at t=0 with DC solution
2. For each timestep:
   - Update capacitor companion models
   - Run Newton-Raphson to steady-state
   - Store voltages for plotting
   - Advance: t += Δt

### Source Waveforms
- **PWL** (Piece-Wise Linear): Interpolate between time points
- **PULSE**: Periodic transitions with rise/fall times

---

## 4. MOS Level 1 Model (Shichman-Hodges)

### Device Terminals
Drain, Gate, Source, Bulk (4 terminals)

### Operating Regions

1. **Cutoff** (V_gs < V_th): I_ds = 0
2. **Linear** (V_ds < V_ov): Triode region
3. **Saturation** (V_ds ≥ V_ov): Constant current

### Key Equations
```
V_ov = V_gs - V_th  (overdrive voltage)

Linear region (V_ds < V_ov):
I_ds = K * [(V_gs - V_th) * V_ds - 0.5 * V_ds²]

Saturation region (V_ds ≥ V_ov):
I_ds = 0.5 * K * (V_gs - V_th)²

Where K = KP * (W/L)
```

### Parameters (from SPICE)
- `KP`: Transconductance coefficient
- `L`: Channel length
- `W`: Channel width
- `VTO`: Zero-bias threshold voltage
- `GAMMA`: Body effect coefficient (optional initially)

### Conductance Derivatives
For Newton-Raphson, the solver needs:
```
g_ds = ∂I_ds/∂V_ds (output conductance)
g_m = ∂I_ds/∂V_gs (transconductance)
```

### Body Effect (Simplified)
Initial implementation: bulk tied to source (V_sb = 0)
Future extension: V_th = VTO + γ * (√(2|φ_f| + V_sb) - √(2|φ_f|))

---

## 5. Parser & Netlist Format

### Input Format (HSPICE-like)
```
* Comments start with asterisk
Vdd 1 0 3.3              ; DC voltage: node1-node0, 3.3V
Vin 2 0 0                ; Grounded input (starts at 0V)
R1 1 3 10k               ; Resistor: 10kΩ between nodes 1-3
M1 3 2 0 0 NMOS L=1u W=10u  ; MOSFET: drain gate source bulk model
.dc Vin 0 3.3 0.1        ; DC sweep: Vin from 0-3.3V in 0.1V steps
.tran 1n 100n            ; Transient: 1ns step for 100ns
.end
```

### Parser Responsibilities

1. **Tokenize**: Split into lines, ignore comments/blank lines
2. **Component Extraction**: Regex match device types
   - Format: `<name> <n1> <n2> ... <params>`
   - First char determines type (R, C, V, I, M)
3. **Node Mapping**: Assign numerical indices
   - Ground ("0" or "GND") → excluded from matrix
   - Other nodes → 0, 1, 2, ... in order of appearance
4. **Parameter Parsing**: Extract key=value pairs
5. **Unit Conversion**: Handle suffixes (k, u, n, p, etc.)
6. **Analysis Extraction**: Parse `.dc` and `.tran` commands

### Unit Suffixes
```
T = 1e12,  G = 1e9,  Meg = 1e6,  k = 1e3
m = 1e-3, u = 1e-6,  n = 1e-9,  p = 1e-12,  f = 1e-15
```

### Error Handling
- Unknown component types → clear error with line number
- Duplicate component names → warning
- Unconnected nodes → validation error
- Invalid parameters → exception with context

---

## 6. Visualization & Output

### Real-Time Console Logging
```
Parsing netlist: inverter.sp
Found 4 components, 3 nodes
DC Analysis: Sweeping Vin from 0.0V to 3.3V (34 points)
  Point 1/34: V_out = 3.30V (converged in 2 iterations)
  Point 2/34: V_out = 3.29V (converged in 2 iterations)
  ...
Analysis complete. Elapsed time: 0.15s
```

### Logging Levels
- `INFO`: Progress updates, analysis type, sweep points
- `WARNING`: Convergence issues, skipped devices
- `ERROR`: Matrix singular, netlist syntax errors
- `DEBUG`: Matrix values, Newton iterations (verbose mode)

### Matplotlib Visualization

**DC Sweep:**
- X-axis: Sweep variable (input voltage)
- Y-axis: All node voltages or selected probes
- Save to: `results/dc_sweep_<timestamp>.png`

**Transient:**
- X-axis: Time
- Y-axis: Node voltages vs time
- Save to: `results/transient_<timestamp>.png`

**Optional CSV Export:**
- Raw data for external analysis
- Format: `time,v1,v2,v3,...` or `vin,vout`

### Error Recovery
- **Non-convergence**: Log last voltages, continue to next point
- **Singular matrix**: Check floating nodes, report cause
- **NaN/Inf**: Check device parameters, suggest validation

---

## 7. Module Structure

```
pycircuitsim/
├── models/
│   ├── __init__.py
│   ├── base.py          # Component abstract base class
│   ├── passive.py       # R, C, V_source, I_source
│   └── mosfet.py        # NMOS, PMOS (Level 1)
├── parser.py            # Netlist parser
├── circuit.py           # Circuit topology container
├── solver.py            # DC and Transient solvers
├── visualizer.py        # Matplotlib plotting
├── main.py              # CLI entry point
├── tests/
│   ├── test_models.py
│   ├── test_solver.py
│   ├── test_parser.py
│   └── test_circuits/   # Sample netlists
└── results/             # Output plots and data
```

---

## 8. Development Roadmap

1. **Step 1**: Define `Component` base class and MNA matrix construction
2. **Step 2**: Implement R, V, I models and `.dc` single-point solver
3. **Step 3**: Implement MOS Level 1 model + Newton-Raphson
4. **Step 4**: Implement R, C, V, I transient solver (no MOS)
5. **Step 5**: Add MOS to transient solver
6. **Step 6**: Implement Parser and Visualizer

---

## 9. Verification Strategy

**Reference Implementations:**
- **ngspice**: Primary reference for physics correctness
- **Xyce**: Architectural patterns (translated to Python)

**Test-While-Building:**
- Unit tests for each device model
- Integration tests for solver convergence
- Comparison with ngspice output for known circuits

**Example Test Circuits:**
- Voltage divider (linear)
- Inverter chain (non-linear DC)
- RC circuit (transient)
- Ring oscillator (transient + MOS)

---

## 10. Coding Conventions

- **Type Hints**: Required on all function signatures
- **Docstrings**: Required for key classes and algorithms
- **Variable Names**: Use physically meaningful names (`v_gate`, `i_drain`, `conductance`)
- **Error Messages**: Clear exceptions with context

**Dependencies:**
- Python 3.10+
- NumPy (matrix operations)
- Matplotlib (visualization)
- Standard library `logging` module
