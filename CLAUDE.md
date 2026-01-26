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
    * Detailed simulation logs saved as `<circuit_name>_simulation.lis` (HSPICE-like format):
      - Iteration-by-iteration Newton-Raphson convergence data
      - Node voltages, voltage deltas, device currents
      - MOSFET conductances (gm, gds) for each iteration
      - Sweep point information for DC sweep analysis
    * Waveform data saved as `<circuit_name>_dc_sweep.csv` or `<circuit_name>_transient.csv`:
      - CSV format with all node voltages and device currents
      - Compatible with Excel, Python pandas, MATLAB
      - Uses scientific notation for precision
    * Plot files saved as `<circuit_name>_dc_sweep.png` or `<circuit_name>_transient.png`
    * DC sweep plots have separate voltage/current subplots for better readability
    * All output files organized in subdirectory: `results/<circuit_name>/`

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

PyCircuitSim generates output files organized by circuit name in the `results/` directory:

**Directory Structure:**
```
results/
├── inverter/
│   ├── inverter_simulation.lis      # Detailed simulation log
│   ├── inverter_dc_sweep.png        # DC sweep plot
│   └── inverter_dc_sweep.csv        # Waveform data
├── sram_cell/
│   ├── sram_cell_simulation.lis
│   ├── sram_cell_dc_sweep.png
│   └── sram_cell_dc_sweep.csv
└── ...
```

### File Naming Convention
All output files use the format: `<circuit_name>_<analysis_type>.<ext>`

- **Simulation Log**: `<circuit_name>_simulation.lis`
- **DC Sweep Plot**: `<circuit_name>_dc_sweep.png`
- **DC Sweep Data**: `<circuit_name>_dc_sweep.csv`
- **Transient Plot**: `<circuit_name>_transient.png`
- **Transient Data**: `<circuit_name>_transient.csv` (future)
- **DC OP Point**: `<circuit_name>_dc_op_point.txt`

### Simulation Log Files (.lis)
PyCircuitSim generates detailed simulation log files in HSPICE-like `.lis` format:

**Contents:**
- **Header**: Analysis type, parameters, timestamp
- **Circuit Summary**: Component count, node count, voltage source count
- **Iteration Details** (for each sweep point or time step):
  - Node voltages for current iteration
  - Voltage deltas (change from previous iteration)
  - Device currents through all components (including voltage sources!)
  - MOSFET conductances (gm, gds) when applicable
- **Convergence Status**: Iteration count, final tolerance
- **Final Results**: Node voltages at each sweep point

### CSV Waveform Data
CSV files contain numerical data for analysis in external tools:

**Format:**
```csv
Vin (V), 1, 2, 3, i(Vdd), i(Vin), i(Mp1), i(Mn1), i(Rload)
0.000000, 3.300000e+00, 0.000000e+00, 3.297501e+00, -3.297501e-06, ...
0.100000, 3.300000e+00, 1.000000e-01, 3.297423e+00, -3.297423e-06, ...
...
```

**Columns:**
- First column: Sweep variable (e.g., Vin)
- Remaining columns: Node voltages and device currents
- Uses scientific notation (e.g., `3.300000e+00`)
- All currents labeled as `i(component_name)`

**Usage:**
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV data
df = pd.read_csv('results/inverter/inverter_dc_sweep.csv')

# Plot custom graph
plt.plot(df['Vin (V)'], df['3'], label='Output')
plt.plot(df['Vin (V)'], df['i(Vdd)']*1000, label='Vdd current (mA)')
plt.xlabel('Input Voltage (V)')
plt.ylabel('Output Voltage (V) / Current (mA)')
plt.legend()
plt.grid(True)
```

### Plot Files
* **DC Sweep**: `<circuit_name>_dc_sweep.png` - Separate subplots for voltages (top) and currents (bottom)
* **Transient Analysis**: `<circuit_name>_transient.png` - Time-domain waveforms with auto-scaled time axis

**Plot Features:**
- Top subplot: All node voltages
- Bottom subplot: All device currents
- Automatic detection of voltage vs current signals
- Shared x-axis for easy comparison
- High resolution (150 DPI) for publication quality

### Example .lis File Excerpt
```
----------------------------------------------------------------------
Sweep Point 5: Sweep Value = 0.500
----------------------------------------------------------------------
  Iteration 0:
    Node Voltages:
           1:            3.3 V
           2:          0.05 V
           3:       3.29705 V
    Voltage Changes:
           1:  0.00 V
           2:         0.05 V
           3:      0.00004 V
    Device Currents:
         Mp1: -2.472357e-06 A
         Mn1:  0.000000e+00 A
         Vdd:  0.000000e+00 A
         Vin:  0.000000e+00 A
       Rload:   0.0003297 A
    Device Conductances:
      Mp1: gm=1.018980e-03 S, gds=1.017156e-06 S
      Mn1:
         gds: 0 S
          gm: 0 S
----------------------------------------------------------------------
Point 5: CONVERGED in 1 iterations
Final tolerance: 4.38427e-05
----------------------------------------------------------------------
```

## 10. Example Usage

### Basic Simulation
```bash
# Run DC sweep simulation
python main.py examples/inverter.sp -o results/

# Output files created in results/inverter/:
#   - inverter_simulation.lis      (detailed log)
#   - inverter_dc_sweep.png       (plot)
#   - inverter_dc_sweep.csv        (waveform data)
```

### Correct MOSFET Terminal Order
**IMPORTANT**: The terminal order for MOSFETs is `drain gate source bulk`:

**NMOS** (N-channel):
- Correct: `Mn1 3 2 0 0 NMOS L=1u W=10u`
  - drain=3 (output), gate=2 (input), source=0 (GND), bulk=0 (GND)
- Current flows drain → source when ON

**PMOS** (P-channel):
- Correct: `Mp1 3 2 1 1 PMOS L=1u W=20u`
  - drain=3 (output), gate=2 (input), source=1 (Vdd), bulk=1 (Vdd)
- Current flows source → drain when ON (opposite of NMOS)

**Common Mistake**: Swapping drain/source terminals leads to incorrect circuit behavior!

### Example Netlist (CMOS Inverter)
```spice
* CMOS Inverter with Correct MOSFET Terminal Order
* Demonstrates voltage switching from HIGH to LOW

* Power supply
Vdd 1 0 3.3

* Input voltage source
Vin 2 0 0

* PMOS (pull-up transistor)
* drain=output, gate=input, source=Vdd, bulk=Vdd
Mp1 3 2 1 1 PMOS L=1u W=20u

* NMOS (pull-down transistor)
* drain=output, gate=input, source=GND, bulk=GND
Mn1 3 2 0 0 NMOS L=1u W=10u

* Load resistor (prevents floating output)
Rload 3 0 10000

* DC Sweep: Sweep input from 0V to 3.3V in 0.1V steps
.dc Vin 0 3.3 0.1

.end
```

### Command Line Options
```bash
# Specify custom output directory
python main.py examples/inverter.sp -o my_results

# Enable verbose logging
python main.py examples/inverter.sp -o results --verbose
```

### Accessing Waveform Data
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV data
df = pd.read_csv('results/inverter/inverter_dc_sweep.csv')

# Plot inverter transfer characteristic
plt.figure(figsize=(10, 6))
plt.plot(df['Vin (V)'], df['3'], 'b-', linewidth=2, label='V(out)')
plt.xlabel('Input Voltage (V)')
plt.ylabel('Output Voltage (V)')
plt.title('Inverter DC Transfer Characteristic')
plt.grid(True)
plt.legend()
plt.show()

# Verify KCL at Vin = 0V (first row)
vdd_current = df['i(Vdd)'][0] * 1e6  # Convert to microamps
load_current = df['i(Rload)'][0] * 1e6
print(f"Vdd current: {vdd_current:.2f} µA")
print(f"Load current: {load_current:.2f} µA")
print(f"KCL check: i(Vdd) + i(Mp1) + i(Mn1) + i(Rload) = 0")
```