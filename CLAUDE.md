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

### 3.3 Input/Output
* **Input**: Parse `.sp` files similar to HSPICE format.
* **Output**:
    * Real-time simulation progress and convergence status via console logging.
    * Save waveforms and timing diagrams after simulation completes.

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
5.  `visualizer.py`: Handles Matplotlib plotting.

### 4.3 Key Algorithms
* **MNA (Modified Nodal Analysis)**: Used to construct the circuit equation matrix $Ax=b$.
* **Newton-Raphson**: Used to solve non-linear circuits (introduced by MOSFETs).
* **Backward Euler**: Used for the discretization of capacitors (numerical integration) during transient analysis.

## 5. Coding Conventions
* **Type Hinting**: All function signatures must include Python type annotations.
* **Docstrings**: Key classes and complex algorithms must include docstrings explaining the physical meaning.
* **Variable Naming**: Use variable names with clear physical meaning (e.g., `v_gate`, `i_drain`, `conductance`), avoiding generic names like `a`, `b`, `tmp`.
* **Error Handling**: Raise clear exceptions when netlist syntax is invalid or the matrix is singular (unsolvable).

## 6. Development Roadmap
1.  **Step 1**: Define the `Component` base class and the basic MNA matrix construction logic.
2.  **Step 2**: Implement R, V, I models and the `.dc` (single point) solver; verify with linear circuits.
3.  **Step 3**: Implement MOS Level 1 model and Newton-Raphson iteration; verify with non-linear DC sweeps.
4.  **Step 4**: Implement R, C, V, I model without MOS devices and Backward Euler integration; verify `.tran` analysis.
4.  **Step 5**: Implement R, C, V, I model with MOS devices and Backward Euler integration; verify `.tran` analysis.
5.  **Step 6**: Implement Parser and Visualizer to connect the entire workflow.

## 7. Reference Implementations (For Logic Verification)
* **ngspice**: Use as the primary reference for checking the correctness of physics equations (especially the implementation of the MOS Level 1 Shichman-Hodges equations) and standard SPICE behaviors.
* **Xyce**: Refer to this for architectural patterns on how to cleanly separate device state from the global solver matrix, although our implementation will remain in pure Python.
* **Note**: While referencing these C/C++ projects, **do not** copy low-level optimizations (like pointer arithmetic or memory pooling). Translate the core logic into clean, readable Python.

## 8. Example Syntax
```spice
* Inverter Simulation
Vdd 1 0 3.3
Vin 2 0 0
M1 3 2 0 0 NMOS L=1u W=10u
R1 1 3 10k
.dc Vin 0 3.3 0.1
.end
```
