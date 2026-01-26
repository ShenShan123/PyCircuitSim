# Project: PyCircuitSim (Simple Python Circuit Simulator)

## 1. Project Overview
This is an open-source, Python-based circuit simulator designed to provide a clean, readable, and architecturally clear core for circuit simulation. The project prioritizes educational value and architectural extensibility over raw simulation speed.

**Core Goals:**
* **Pure Python Implementation**: Logic must be clear and easy to understand.
* **Modular Design**: **Must** implement complete decoupling between the Solver Engine and Device Models.
* **HSPICE Compatibility**: Support basic HSPICE netlist syntax.

## 2. Tech Stack
* **Language**: Python 3.10+
* **Core Logic**: NumPy (for matrix operations and linear algebra)
* **Visualization**: Matplotlib (for plotting waveforms)
* **Logging**: Python built-in logging module (for simulation logs)

## 3. Functional Requirements

### 3.1 Supported Devices
* **Passive Devices**: Resistors (R), Capacitors (C)
* **Active Devices**: MOSFETs (NMOS/PMOS) - **Support Level 1 Compact Model only**
* **Sources**: DC Voltage Source (V), DC Current Source (I), including PWL and PULSE sources for transient analysis.

### 3.2 Analysis Types
1.  **`.dc` (DC Sweep Analysis)**: Sweep input source voltage/current to calculate DC operating points.
2.  **`.tran` (Transient Analysis)**: Simulate circuit behavior over time (requires handling capacitor integration and time stepping).

### 3.3 Keep It Simple
* **Ignore Complex Directives**: ignore complex directives like `.option`, `.measure`, `.probe`, `.param`, and functions.

### 3.4 Input/Output
* **Input**: Parse `.sp` files similar to HSPICE format.
* **Output**:
    * Real-time simulation progress and convergence status via console logging.
    * Detailed simulation logs saved as `.lis` files (HSPICE-like format):
      - Iteration-by-iteration Newton-Raphson convergence data
      - Node voltages, voltage deltas, device currents
      - MOSFET conductances (gm, gds) for each iteration
      - Sweep point information for DC sweep analysis
    * Save waveforms and timing diagrams after simulation completes.
    * DC sweep plots with separate voltage and current subplots for better readability.

## 4. Architecture & Design Guidelines

**These are the core constraints of the project. Generated code must strictly adhere to these rules:**

### 4.1 The Separation Principle
* **Solver Engine**: Responsible for building the MNA (Modified Nodal Analysis) matrix, executing Newton-Raphson iterations, and managing time stepping. **The Solver must NOT contain any physical formulas for specific devices (e.g., MOS equations).**
* **Device Models**: Responsible for calculating current and its derivatives (conductance) based on terminal voltages. All devices must inherit from a common abstract `Device` base class.

### 4.2 Recommended Module Structure
1.  `parser.py`: Reads the netlist and extracts nodes and component parameters.
2.  `circuit.py`: Maintains the circuit topology (node lists, component lists).
3.  `solver.py`: Contains logic for the DC solver and Transient solver.
4.  `models/`:
    * `base.py`: Defines the abstract base class `Component`.
    * `passive.py`: Implements R, C, V\_source, I\_source.
    * `mosfet.py`: Implements **MOS Level 1** model equations.
5.  `logger.py`: Handles detailed simulation logging in HSPICE-like .lis format.
6.  `visualizer.py`: Handles Matplotlib plotting.

### 4.3 Key Algorithms
* **MNA (Modified Nodal Analysis)**: Used to construct the circuit equation matrix $Ax=b$.
* **Newton-Raphson**: Used to solve non-linear circuits (introduced by MOSFETs).
* **Backward Euler**: Used for the discretization of capacitors (numerical integration) during transient analysis.

**Two-Stage Analysis for Non-Linear Circuits:**
* For circuits with MOSFETs, a two-stage approach improves convergence:
  1. **Stage 1**: Compute DC operating point using source stepping
  2. **Stage 2**: Use OP solution as initial guess for DC sweep or transient analysis
* This prevents Newton-Raphson divergence caused by poor initial guesses (all zeros)
* The OP solution provides an excellent starting point close to the actual operating region

## 5. Coding Conventions
* **Type Hinting**: All function signatures must include Python type annotations.
* **Docstrings**: Key classes and complex algorithms must include docstrings explaining the physical meaning.
* **Variable Naming**: Use variable names with clear physical meaning (e.g., `v_gate`, `i_drain`, `conductance`), avoiding generic names like `a`, `b`, `tmp`.
* **Error Handling**: Raise clear exceptions when netlist syntax is invalid or the matrix is singular (unsolvable).
* **Numerical Stability**:
    * MOSFET models implement voltage clamping to prevent numerical overflow
    * Gate-source overdrive voltage clamped to ±5V
    * Drain-source voltage clamped to ±10V
    * This prevents unrealistic current values and improves convergence

## 6. Development Roadmap

**Phase 1: Core Implementation** ✅ Complete
1.  **Step 1**: Define the `Component` base class and the basic MNA matrix construction logic. ✅
2.  **Step 2**: Implement R, V, I models and the `.dc` (single point) solver; verify with linear circuits. ✅
3.  **Step 3**: Implement MOS Level 1 model and Newton-Raphson iteration; verify with non-linear DC sweeps. ✅
4.  **Step 4**: Implement R, C, V, I model without MOS devices and Backward Euler integration; verify `.tran` analysis. ✅
5.  **Step 5**: Implement R, C, V, I model with MOS devices and Backward Euler integration; verify `.tran` analysis. ✅
6.  **Step 6**: Implement Parser and Visualizer to connect the entire workflow. ✅

**Phase 2: Enhancements** ✅ Complete
1.  **Logger System**: Added HSPICE-like .lis file output with iteration-by-iteration details ✅
2.  **Numerical Stability**: Implemented voltage clamping in MOSFET models to prevent overflow ✅
3.  **Two-Stage Analysis**: DC operating point computation before sweep/transient for better convergence ✅
4.  **Enhanced Visualization**: Separate voltage/current subplots for DC sweep analysis ✅
5.  **Complex Test Circuits**: Added 6T SRAM, inverter chain, and 5T op-amp examples ✅

## 7. Example Circuits

The project includes several example netlists in the `examples/` directory:

### Basic Examples
* `inverter.sp` - Simple CMOS inverter with DC sweep
* `rc_circuit.sp` - RC circuit with transient analysis

### Advanced Examples
* `sram_cell.sp` - 6-transistor SRAM cell demonstrating:
  - Cross-coupled inverter pair for bistable storage
  - Access transistors for read/write operations
  - DC sweep of word line voltage

* `inverter_chain.sp` - 5-stage CMOS inverter chain demonstrating:
  - Signal propagation through multiple logic stages
  - Voltage transfer characteristics
  - Signal restoration properties

* `opamp_5t.sp` - 5-transistor operational amplifier demonstrating:
  - Differential pair input stage
  - Active load current mirror
  - Differential amplification behavior

## 8. Reference Implementations (For Logic Verification)
* **ngspice**: Use as the primary reference for checking the correctness of physics equations (especially the implementation of the MOS Level 1 Shichman-Hodges equations) and standard SPICE behaviors.
* **Xyce**: Refer to this for architectural patterns on how to cleanly separate device state from the global solver matrix, although our implementation will remain in pure Python.
* **Note**: While referencing these C/C++ projects, **do not** copy low-level optimizations (like pointer arithmetic or memory pooling). Translate the core logic into clean, readable Python.

## 9. Output Files

### Simulation Log Files (.lis)
PyCircuitSim generates detailed simulation log files in HSPICE-like `.lis` format:

**Location:** `<output_dir>/simulation.lis`

**Contents:**
- **Header**: Analysis type, parameters, timestamp
- **Circuit Summary**: Component count, node count, voltage source count
- **Iteration Details** (for each sweep point or time step):
  - Node voltages for current iteration
  - Voltage deltas (change from previous iteration)
  - Device currents through all components
  - MOSFET conductances (gm, gds) when applicable
- **Convergence Status**: Iteration count, final tolerance
- **Final Results**: Node voltages at each sweep point

### Plot Files
* **DC Sweep**: `dc_sweep.png` - Separate subplots for voltages (top) and currents (bottom)
* **Transient Analysis**: `transient.png` - Time-domain waveforms with auto-scaled time axis

### Example .lis File Excerpt
```
----------------------------------------------------------------------
  Iteration 3:
    Node Voltages:
           1:          3.3 V
           2:          1.65 V
           3:          0.82 V
    Device Currents:
          Mp1:      -1.2e-05 A
          Mn1:       1.2e-05 A
    Conductances:
          Mp1: gm=4.5e-05 S, gds=2.3e-06 S
----------------------------------------------------------------------
Point 5: CONVERGED in 3 iterations
Final tolerance: 3.2e-07
----------------------------------------------------------------------
```

## 10. Example Syntax
```spice
* Inverter Simulation
Vdd 1 0 3.3
Vin 2 0 0
M1 3 2 0 0 NMOS L=1u W=10u
R1 1 3 10k
.dc Vin 0 3.3 0.1
.end
```
