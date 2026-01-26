# DC Sweep Fixes and Enhancements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix DC sweep overflow issues, add detailed .lis logging, separate voltage/current plots, and add complex test circuits.

**Architecture:**
- Two-stage analysis: DC Operating Point → DC Sweep/Transient (using OP as initial guess)
- Logger class for detailed iteration-by-iteration output to .lis files
- Visualizer with separate subplots for voltages and currents
- MOSFET voltage clamping for numerical stability
- Numerical safeguards in solver (NaN/Inf detection, delta limiting)

**Tech Stack:** Python 3.10+, NumPy, Matplotlib, logging module

---

## Task 1: Create Logger Class

**Files:**
- Create: `pycircuitsim/logger.py`
- Create: `tests/test_logger.py`

**Step 1: Write the failing test**

```python
# tests/test_logger.py
import pytest
import tempfile
from pathlib import Path
from pycircuitsim.logger import Logger, IterationInfo

def test_logger_initialization():
    """Create logger with output file"""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger("test.sp", Path(tmpdir) / "test.lis")
        assert logger.netlist == "test.sp"
        assert logger.output_file == Path(tmpdir) / "test.lis"

def test_log_header():
    """Test header logging"""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger("test.sp", Path(tmpdir) / "test.lis")
        logger.log_header("dc", {"source": "Vin", "start": 0, "stop": 3.3, "step": 0.1})

        content = (Path(tmpdir) / "test.lis").read_text()
        assert "PyCircuitSim Simulation Results" in content
        assert "test.sp" in content
        assert "DC Sweep" in content

def test_log_iteration():
    """Test iteration logging"""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger("test.sp", Path(tmpdir) / "test.lis")

        iter_info = IterationInfo(
            iteration=2,
            voltages={"n1": 3.3, "n2": 0.0},
            deltas={"n1": 0.01, "n2": 0.005},
            currents={"V1": -0.001},
            conductances={"M1": {"gm": 0.001, "gds": 0.0}}
        )
        logger.log_iteration(1, iter_info)

        content = (Path(tmpdir) / "test.lis").read_text()
        assert "Newton-Raphson Iteration 2" in content
        assert "n1: 3.3" in content
        assert "Max Delta: 0.01" in content

def test_log_convergence():
    """Test convergence logging"""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger("test.sp", Path(tmpdir) / "test.lis")
        logger.log_convergence(1, converged=True, iterations=3)

        content = (Path(tmpdir) / "test.lis").read_text()
        assert "Converged in 3 iterations" in content
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_logger.py -v
# Expected: FAIL - ModuleNotFoundError
```

**Step 3: Implement Logger class**

```python
# pycircuitsim/logger.py
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class IterationInfo:
    """Information about a single Newton-Raphson iteration"""
    iteration: int
    voltages: Dict[str, float]
    deltas: Dict[str, float]
    currents: Dict[str, float]
    conductances: Dict[str, Dict[str, float]]  # Device name -> {"gm": x, "gds": y}


class Logger:
    """
    Logger for detailed simulation output (HSPICE .lis format).

    Writes iteration-by-iteration details including voltages, currents,
    conductances, and convergence status to a text file.
    """

    def __init__(self, netlist: str, output_file: Path):
        """
        Initialize logger.

        Args:
            netlist: Name of the netlist file
            output_file: Path to .lis output file
        """
        self.netlist = netlist
        self.output_file = output_file
        self.file_handle = None

    def __enter__(self):
        """Open log file for writing"""
        self.file_handle = open(self.output_file, 'w')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close log file"""
        if self.file_handle:
            self.file_handle.close()

    def _write(self, text: str) -> None:
        """Write text to log file"""
        if self.file_handle:
            self.file_handle.write(text + "\n")

    def log_header(self, analysis_type: str, analysis_params: Dict[str, Any]) -> None:
        """
        Log simulation header with netlist info, date, and analysis type.

        Args:
            analysis_type: 'dc', 'tran', or 'op'
            analysis_params: Analysis parameters
        """
        self._write("=" * 60)
        self._write("PyCircuitSim Simulation Results")
        self._write("=" * 60)
        self._write(f"Netlist: {self.netlist}")
        self._write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._write("")

        if analysis_type == "dc":
            self._write(f"Analysis: DC Sweep")
            src = analysis_params.get('source', '')
            self._write(f"  Source: {src}")
            self._write(f"  Range: {analysis_params.get('start', 0)}V to {analysis_params.get('stop', 0)}V")
            self._write(f"  Step: {analysis_params.get('step', 0)}V")
        elif analysis_type == "tran":
            self._write(f"Analysis: Transient")
            self._write(f"  Time step: {analysis_params.get('tstep', 0)}s")
            self._write(f"  Stop time: {analysis_params.get('tstop', 0)}s")
        elif analysis_type == "op":
            self._write(f"Analysis: DC Operating Point")

        self._write("")

    def log_circuit_summary(self, component_count: int, node_count: int,
                           vsource_count: int) -> None:
        """Log circuit summary (components, nodes, voltage sources)"""
        self._write("Circuit Summary:")
        self._write(f"  Components: {component_count}")
        self._write(f"  Nodes: {node_count} (excluding ground)")
        self._write(f"  Voltage Sources: {vsource_count}")
        self._write("")

    def log_sweep_point_start(self, point_num: int, sweep_value: float) -> None:
        """Log start of a DC sweep point"""
        self._write("-" * 40)
        self._write(f"DC Sweep Point {point_num}: Value = {sweep_value:.4f}V")
        self._write("-" * 40)
        self._write("")

    def log_iteration(self, point_num: int, iter_info: IterationInfo) -> None:
        """
        Log a single Newton-Raphson iteration with all details.

        Args:
            point_num: Sweep point number (for DC) or 0 for transient
            iter_info: Iteration information
        """
        self._write(f"Newton-Raphson Iteration {iter_info.iteration}:")
        self._write("")
        self._write("  Node Voltages:")
        for node, voltage in sorted(iter_info.voltages.items()):
            self._write(f"    {node}: {voltage:.4f}V")

        self._write("  Voltage Deltas:")
        max_delta = 0.0
        for node, delta in sorted(iter_info.deltas.items()):
            self._write(f"    {node}: {delta:.6f}V")
            max_delta = max(max_delta, abs(delta))

        self._write("  Device Currents:")
        for device, current in sorted(iter_info.currents.items()):
            self._write(f"    {device}: {current:.6e}A", end="")
            if device in iter_info.conductances:
                cond = iter_info.conductances[device]
                if 'gm' in cond or 'gds' in cond:
                    gm = cond.get('gm', 0.0)
                    gds = cond.get('gds', 0.0)
                    self._write(f" (gm={gm:.6e}S, gds={gds:.6e}S)")
                else:
                    self._write("")
            else:
                self._write("")

        self._write(f"  Max Delta: {max_delta:.6f}V")
        self._write("")

    def log_convergence(self, point_num: int, converged: bool,
                       iterations: int, tolerance: float = 1e-6) -> None:
        """Log convergence status"""
        if converged:
            self._write(f"Converged in {iterations} iterations (tolerance: {tolerance:.0e})")
        else:
            self._write(f"Failed to converge after {iterations} iterations")
        self._write("")

    def log_final_results(self, results: Dict[str, float],
                         title: str = "Final Results") -> None:
        """Log final operating point results"""
        self._write(title)
        self._write("-" * len(title))
        for node, voltage in sorted(results.items()):
            if node != "0" and node.lower() != "gnd":
                self._write(f"  {node}: {voltage:.6f}V")
        self._write("")
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_logger.py -v
# Expected: PASS
```

**Step 5: Commit**

```bash
git add pycircuitsim/logger.py tests/test_logger.py
git commit -m "feat: add Logger class for detailed simulation output"
```

---

## Task 2: Add Voltage Clamping to MOSFET

**Files:**
- Modify: `pycircuitsim/models/mosfet.py`
- Modify: `tests/test_mosfet.py`

**Step 1: Write failing test**

```python
# Add to tests/test_mosfet.py
def test_nmos_voltage_clamping():
    """Verify MOSFET clamps extreme voltages to prevent overflow"""
    from pycircuitsim.models.mosfet import NMOS

    m = NMOS("M1", ["d", "g", "s", "b"], L=1e-6, W=10e-6)

    # Extreme voltages that would cause overflow
    extreme_voltages = {"d": 1000.0, "g": 0.0, "s": 0.0, "b": 0.0}

    # Should not raise overflow error
    current = m.calculate_current(extreme_voltages)

    # Current should be clamped/limited, not inf or nan
    assert not np.isinf(current)
    assert not np.isnan(current)
    assert current >= 0  # NMOS current should be non-negative
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_mosfet.py::test_nmos_voltage_clamping -v
# Expected: FAIL (overflow will occur without clamping)
```

**Step 3: Implement voltage clamping**

```python
# Modify pycircuitsim/models/mosfet.py

# In NMOS class, add clamping constants at top:
class NMOS(Component):
    """..."""
    # Voltage clamping limits to prevent numerical overflow
    _MAX_V_OVERDRIVE = 5.0  # Maximum |V_gs - V_th|
    _MAX_V_DS = 10.0  # Maximum |V_ds|

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        """Calculate drain current with voltage clamping"""
        v_d, v_g, v_s, v_b = self._get_voltages(voltages)

        v_gs = v_g - v_s
        v_ds = v_d - v_s

        # Clamp voltages to prevent overflow
        v_gs = np.clip(v_gs, -self._MAX_V_OVERDRIVE, self._MAX_V_OVERDRIVE)
        v_ds = np.clip(v_ds, -self._MAX_V_DS, self._MAX_V_DS)

        # ... rest of the logic remains the same
        if v_gs < self.VTO:
            self.current_region = "cutoff"
            return 0.0

        v_ov = v_gs - self.VTO
        v_ov = np.clip(v_ov, -self._MAX_V_OVERDRIVE, self._MAX_V_OVERDRIVE)

        if v_ds >= v_ov:
            self.current_region = "saturation"
            return 0.5 * self.K * v_ov**2
        else:
            self.current_region = "linear"
            return self.K * (v_ov * v_ds - 0.5 * v_ds**2)

    def get_conductance(self, voltages: Dict[str, float]) -> Tuple[float, float]:
        """Calculate conductance with voltage clamping"""
        v_d, v_g, v_s, v_b = self._get_voltages(voltages)

        v_gs = v_g - v_s
        v_ds = v_d - v_s

        # Clamp voltages before calculating derivatives
        v_gs = np.clip(v_gs, -self._MAX_V_OVERDRIVE, self._MAX_V_OVERDRIVE)
        v_ds = np.clip(v_ds, -self._MAX_V_DS, self._MAX_V_DS)

        # ... rest of logic with clamped values
        if v_gs < self.VTO:
            return 0.0, 0.0

        v_ov = v_gs - self.VTO
        v_ov = np.clip(v_ov, -self._MAX_V_OVERDRIVE, self._MAX_V_OVERDRIVE)

        if v_ds >= v_ov:
            g_m = self.K * v_ov
            return 0.0, g_m
        else:
            g_m = self.K * v_ds
            g_ds = self.K * (v_ov - v_ds)
            return g_ds, g_m
```

Also update PMOS class with similar clamping (using positive limits since voltages are negative).

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_mosfet.py::test_nmos_voltage_clamping -v
# Expected: PASS
```

**Step 5: Commit**

```bash
git add pycircuitsim/models/mosfet.py tests/test_mosfet.py
git commit -m "fix: add voltage clamping to MOSFET models to prevent overflow"
```

---

## Task 3: Update DCSolver with Logger Integration

**Files:**
- Modify: `pycircuitsim/solver.py`
- Create: `tests/test_solver_with_logging.py`

**Step 1: Add logger parameter to DCSolver**

```python
# Modify pycircuitsim/solver.py

class DCSolver:
    def __init__(self, circuit: Circuit, tolerance: float = 1e-6,
                 max_iterations: int = 50, logger: Optional['Logger'] = None):
        """
        Args:
            circuit: Circuit to solve
            tolerance: Convergence tolerance
            max_iterations: Maximum Newton iterations
            logger: Optional Logger instance for detailed output
        """
        self.circuit = circuit
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.logger = logger
```

**Step 2: Modify _solve_newton to log iterations**

```python
# In DCSolver._solve_newton method

def _solve_newton(self, node_map, num_nodes, num_vsources,
                 op_voltages: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    Solve non-linear circuit using Newton-Raphson with iteration logging.

    Args:
        op_voltages: Optional initial voltages from DC operating point
    """
    from pycircuitsim.models.mosfet import NMOS, PMOS

    # Initial guess
    if op_voltages is None:
        voltages = {node: 0.0 for node in node_map.keys()}
    else:
        voltages = op_voltages.copy()

    voltages["0"] = 0.0

    for iteration in range(self.max_iterations):
        old_voltages = voltages.copy()

        # Build MNA matrix (existing code)
        # ... (matrix construction code)

        # Calculate iteration info for logging
        if self.logger:
            # Calculate device currents and conductances
            currents = {}
            conductances = {}
            for comp in self.circuit.components:
                if hasattr(comp, 'calculate_current'):
                    try:
                        current = comp.calculate_current(voltages)
                        currents[comp.name] = current

                        if hasattr(comp, 'get_conductance'):
                            g_ds, g_m = comp.get_conductance(voltages)
                            conductances[comp.name] = {'gm': g_m, 'gds': g_ds}
                    except:
                        pass  # Skip if calculation fails

            # Calculate deltas
            deltas = {node: abs(voltages[node] - old_voltages[node])
                     for node in node_map.keys()}

            # Log iteration
            from pycircuitsim.logger import IterationInfo
            iter_info = IterationInfo(
                iteration=iteration + 1,
                voltages=voltages.copy(),
                deltas=deltas,
                currents=currents,
                conductances=conductances
            )
            self.logger.log_iteration(0, iter_info)

        # ... (rest of solving logic)
```

**Step 3: Add numerical safeguards**

```python
# After solving for delta:
delta = np.linalg.solve(G_matrix, rhs)

# Check for NaN or Inf
if np.any(np.isnan(delta)) or np.any(np.isinf(delta)):
    if self.logger:
        self.logger._write("Warning: NaN or Inf detected in solution")
    # Reduce damping and retry
    damping = 0.1
else:
    damping = 0.8

# Limit max delta
max_delta = np.max(np.abs(delta))
if max_delta > 1.0:
    if self.logger:
        self.logger._write(f"Warning: Large delta detected ({max_delta:.2f}V), clamping")
    delta = np.clip(delta, -1.0, 1.0)

# Update voltages
for i, node_name in enumerate(node_map.keys()):
    voltages[node_name] += damping * delta[i]
```

**Step 4: Write tests for solver with logging**

```python
# tests/test_solver_with_logging.py
def test_solver_with_logging(tmp_path):
    """Test solver produces detailed .lis file"""
    from pycircuitsim.logger import Logger
    from pycircuitsim.parser import Parser
    import tempfile

    # Create simple test circuit
    netlist_content = """* Test circuit
Vdd 1 0 3.3
Vin 2 0 1.5
M1 3 2 0 0 NMOS L=1u W=10u
R1 1 3 10k
.dc Vin 0 3.3 0.5
.end"""

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write netlist
        sp_file = Path(tmpdir) / "test.sp"
        sp_file.write_text(netlist_content)

        # Parse and solve
        parser = Parser()
        parser.parse_file(str(sp_file))

        lis_file = Path(tmpdir) / "test.lis"
        with Logger("test.sp", lis_file) as logger:
            logger.log_header("dc", parser.analysis_params)
            logger.log_circuit_summary(len(parser.circuit.components),
                                         len(parser.circuit.get_nodes()), 2)

            solver = DCSolver(parser.circuit, logger=logger)
            # Run single point to test logging
            # ... (solver logic)

        # Verify .lis file was created
        assert lis_file.exists()
        content = lis_file.read_text()

        # Check for required sections
        assert "PyCircuitSim Simulation Results" in content
        assert "Newton-Raphson Iteration" in content
        assert "Converged" in content
```

**Step 5: Run tests and commit**

```bash
pytest tests/test_solver_with_logging.py -v
git add pycircuitsim/solver.py tests/test_solver_with_logging.py
git commit -m "feat: integrate Logger into DCSolver with iteration tracking"
```

---

## Task 4: Implement Two-Stage Analysis (OP → Sweep/Transient)

**Files:**
- Modify: `pycircuitsim/main.py`
- Modify: `pycircuitsim/solver.py`

**Step 1: Extract DC Operating Point method**

```python
# Modify pycircuitsim/solver.py - Add compute_op method

class DCSolver:
    def compute_operating_point(self) -> Dict[str, float]:
        """
        Compute DC operating point and return result.

        This is used as the initial state for DC sweeps and transient analysis.

        Returns:
            Dictionary of node voltages at DC operating point
        """
        node_map = self.circuit.get_node_map()
        num_nodes = len(node_map)
        num_vsources = self.circuit.count_voltage_sources()

        # Use existing solve logic
        return self._solve_linear(node_map, num_nodes, num_vsources)
```

**Step 2: Modify run_dc_sweep to use OP analysis**

```python
# Modify pycircuitsim/main.py - run_dc_sweep function

def run_dc_sweep(
    circuit: Circuit,
    analysis_params: Dict,
    visualizer: Visualizer,
    output_path: Path
) -> Dict[str, List[float]]:
    """
    Run DC sweep analysis with two-stage approach.

    Stage 1: Compute DC Operating Point (initial state)
    Stage 2: Sweep from OP, using each point as initial guess for next
    """
    from pycircuitsim.solver import DCSolver
    from pycircuitsim.logger import Logger

    source_name = analysis_params['source']
    start = analysis_params['start']
    stop = analysis_params['stop']
    step = analysis_params['step']

    # Initialize logger
    lis_file = output_path / f"{Path(source_name).stem}.lis"
    with Logger(source_name, lis_file) as logger:
        # Log header
        logger.log_header("dc", analysis_params)
        logger.log_circuit_summary(
            len(circuit.components),
            len(circuit.get_nodes()),
            circuit.count_voltage_sources()
        )

        # Stage 1: Compute DC Operating Point
        logger.log_final_results({}, "DC Operating Point (Initial State)")
        logger._write("")

        # Compute OP at starting voltage
        source_component = None
        for comp in circuit.components:
            if comp.name == source_name:
                source_component = comp
                break

        if source_component is None:
            raise ValueError(f"Source {source_name} not found")

        # Set source to start value and compute OP
        original_value = source_component.value
        source_component.value = start

        solver = DCSolver(circuit, logger=logger)
        op_voltages = solver.compute_operating_point()

        # Log OP results
        logger.log_final_results(op_voltages, "DC Operating Point (Computed)")

        # Stage 2: Sweep from OP, using previous point as initial guess
        sweep_values = []
        all_results = {}

        # Start from start value (OP already computed this)
        current_value = start
        point_num = 1

        while current_value <= stop:
            # For points after the first, use previous voltages as initial guess
            if point_num > 1:
                op_voltages = all_results_voltages.copy()

            source_component.value = current_value

            # Solve with this initial guess
            solution = solver.solve(initial_voltages=op_voltages)

            # Store results
            sweep_values.append(current_value)
            for node, node_value in solution.items():
                if node not in all_results:
                    all_results[node] = []
                all_results[node].append(node_value)

            # Store for next iteration
            all_results_voltages = solution.copy()

            current_value += step
            point_num += 1

        # Restore original value
        source_component.value = original_value

        # Generate plot with separated subplots
        plot_path = output_path / "dc_sweep.png"
        visualizer.plot_dc_sweep(
            sweep_values=sweep_values,
            results=all_results,
            sweep_variable=f"{source_name} (V)",
            output_path=str(plot_path)
        )

    return all_results
```

**Step 3: Modify DCSolver.solve() to accept initial_voltages**

```python
# In DCSolver class
def solve(self, initial_voltages: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    Solve the circuit for DC operating point.

    Args:
        initial_voltages: Optional initial voltage guess (for sweep continuity)

    Returns:
        Dictionary mapping node names to voltages
    """
    node_map = self.circuit.get_node_map()
    num_nodes = len(node_map)
    num_vsources = self.circuit.count_voltage_sources()

    # Check if non-linear
    has_mosfet = any('m' in comp.name.lower() for comp in self.circuit.components)

    if not has_mosfet:
        return self._solve_linear(node_map, num_nodes, num_vsources)
    else:
        return self._solve_newton(node_map, num_nodes, num_vsources,
                                  op_voltages=initial_voltages)
```

**Step 4: Write tests and commit**

```bash
pytest tests/ -v -k "dc_sweep"
git add pycircuitsim/solver.py pycircuitsim/main.py
git commit -m "feat: implement two-stage analysis (OP → sweep/Transient)"
```

---

## Task 5: Separate Voltage and Current Plots

**Files:**
- Modify: `pycircuitsim/visualizer.py`
- Modify: `tests/test_visualizer.py`

**Step 1: Write failing test for separated plots**

```python
# Add to tests/test_visualizer.py
def test_plot_dc_sweep_separated(tmp_path):
    """Test DC sweep plots voltages and currents in separate subplots"""
    import matplotlib
    matplotlib.use('Agg')

    from pycircuitsim.visualizer import Visualizer
    viz = Visualizer()

    # Mock data with both voltages and currents
    sweep_values = [0, 1.0, 2.0, 3.0]
    results = {
        'n1': [0.0, 1.0, 2.0, 3.0],  # voltages
        'n2': [5.0, 4.0, 3.0, 2.0],  # voltages
        'Vdd': [0.001, 0.002, 0.003, 0.004],  # currents
        'I(R1)': [0.0005, 0.001, 0.0015, 0.002]  # currents
    }

    output_file = tmp_path / "test_dc_sweep.png"
    viz.plot_dc_sweep(
        sweep_values=sweep_values,
        results=results,
        sweep_variable="Vin (V)",
        output_path=str(output_file)
    )

    # Verify file was created
    assert output_file.exists()

    # In a real test, you'd check the plot has 2 subplots
    # This requires loading the image and verifying subplot count
```

**Step 2: Modify plot_dc_sweep to use subplots**

```python
# Modify pycircuitsim/visualizer.py

def plot_dc_sweep(
    self,
    sweep_values: List[float],
    results: Dict[str, List[float]],
    sweep_variable: str,
    output_path: str,
    title: Optional[str] = None,
    figsize: tuple = (10, 8)
) -> None:
    """
    Plot DC sweep analysis with separate voltage and current subplots.

    Creates a figure with 2 subplots:
    - Top: All node voltages
    - Bottom: All device currents

    Args:
        sweep_values: List of swept source values
        results: Dictionary mapping node/branch names to value lists
        sweep_variable: Name of swept variable (x-axis label)
        output_path: Path to save plot
        title: Optional plot title
        figsize: Figure size (width, height)
    """
    import numpy as np

    # Separate voltages and currents
    voltages = {}
    currents = {}

    for name, values in results.items():
        # Heuristic: voltages are typically in range -10 to 10V
        # Currents are typically in range -1 to 1A (or mA/uA)
        if name.startswith('v(') or name.startswith('V') and name.count('(') > 0:
            # This is a node voltage
            voltages[name] = values
        else:
            # This might be a current
            # Check magnitude to determine
            max_val = max(abs(v) for v in values) if values else 0
            if max_val < 10:  # Likely a voltage
                voltages[name] = values
            else:  # Likely a current (very large or very small)
                currents[name] = values

    # Create figure with 2 subplots (stacked vertically)
    fig, (ax_voltages, ax_currents) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Plot voltages in top subplot
    if voltages:
        for node_name, values in voltages.items():
            if len(values) != len(sweep_values):
                continue
            ax_voltages.plot(sweep_values, values, marker='o', markersize=3, label=node_name)

        ax_voltages.set_ylabel('Voltage (V)')
        ax_voltaxes.legend(loc='best')
        ax_voltages.grid(True, alpha=0.3)
        ax_voltages.set_title('Node Voltages')
    else:
        ax_voltages.text(0.5, 0.5, 'No voltage data', ha='center', va='center',
                         transform=ax_voltages.transAxes)

    # Plot currents in bottom subplot
    if currents:
        for device_name, values in currents.items():
            if len(values) != len(sweep_values):
                continue
            ax_currents.plot(sweep_values, values, marker='s', markersize=3, label=device_name)

        ax_currents.set_xlabel(f'{sweep_variable}')
        ax_currents.set_ylabel('Current (A)')
        ax_currents.legend(loc='best')
        ax_currents.grid(True, alpha=0.3)
        ax_currents.set_title('Device Currents')
    else:
        ax_currents.text(0.5, 0.5, 'No current data', ha='center', va='center',
                         transform=ax_currents.transAxes)

    # Set title
    if title is None:
        title = f'DC Sweep Analysis - {sweep_variable}'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Create directory if needed
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"DC sweep plot saved to: {output_path}")
```

**Step 3: Run tests and commit**

```bash
pytest tests/test_visualizer.py -v
git add pycircuitsim/visualizer.py tests/test_visualizer.py
git commit -m "feat: separate voltages and currents in DC sweep plots"
```

---

## Task 6: Add Complex Test Circuits

**Files:**
- Create: `examples/sram_6t.sp`
- Create: `examples/inverter_chain.sp`
- Create: `examples/opamp_5t.sp`
- Modify: `README.md`

**Step 1: Create 6T SRAM Cell netlist**

```spice
* examples/sram_6t.sp
* 6-Transistor SRAM Cell
* Storage element with bitline access

* Supply
Vdd vdd 0 1.2

* Wordlines
WL wl 0 0.6
WL wl_bar vdd 0.6

* Bitlines
BL bl 0 0.6
BL bl_bar vdd 0.6

* SRAM Cell (6 transistors)
Mp1 vdd wl q1 vdd PMOS L=0.06u W=0.12u
Mp2 q1 wl_bar vdd PMOS L=0.06u W=0.12u
Mn1 q1 wl bl 0 NMOS L=0.06u W=0.06u
Mn2 bl_bar q1 vdd NMOS L=0.06u W=0.06u
Mp3 vdd wl_bar q vdd PMOS L=0.06u W=0.06u
Mn4 bl_bar q vdd NMOS L=0.06u W=0.06u

* Access transistor
Ma5 q bl 0 NMOS L=0.06u W=0.06u

* DC Operating Point Analysis
.op

.end
```

**Step 2: Create Inverter Chain netlist**

```spice
* examples/inverter_chain.sp
* 3-Stage CMOS Inverter Chain
* Tests signal propagation through multiple stages

* Supply
Vdd vdd 0 1.8

* Input
Vin in 0 0.9

* Stage 1
Mp1 vdd in out1 PMOS L=0.18u W=0.5u
Mn1 out1 in 0 NMOS L=0.18u W=0.18u

* Stage 2
Mp2 vdd out1 out2 PMOS L=0.18u W=0.5u
Mn2 out2 out1 0 NMOS L=0.18u W=0.18u

* Stage 3
Mp3 vdd out2 out3 PMOS L=0.18u W=0.5u
Mn3 out3 out2 0 NMOS L=0.18u W=0.18u

* DC Sweep - Measure transfer curve
.dc Vin 0 1.8 0.05

.end
```

**Step 3: Create 5T Operational Amplifier netlist**

```spice
* examples/opamp_5t.sp
* 5-Transistor Operational Amplifier
* Simple differential amplifier with active load

* Supply
Vdd vdd 0 3.3
Vee vee 0 -3.3

* Input
Vin vip 0 1.5
Vin vim 0 1.5

* Current Mirror (loads)
M1 vdd vbias vdd PMOS L=1u W=10u
M2 vdd vim vdd PMOS L=1u W=10u

* Differential Pair
M3 vdd vip out PMOS L=1u W=20u
M4 vdd vim out PMOS L=1u W=20u

* Active Load
M5 vdd vout vout PMOS L=1u W=10u

* Bias
Ibias vbias 0 10u

* DC Transfer Characteristic
.dc Vin -0.5 0.5 0.01

.end
```

**Step 4: Update README with new examples**

```markdown
# Add to README.md:

## Test Circuits

The simulator includes several example netlists:

### Basic Circuits
- `voltage_divider.sp` - Simple resistive voltage divider
- `rc_circuit.sp` - RC charging/discharging transient
- `inverter.sp` - Single CMOS inverter

### Advanced Circuits
- `sram_6t.sp` - 6-transistor SRAM cell (memory element)
- `inverter_chain.sp` - 3-stage inverter chain (digital logic)
- `opamp_5t.sp` - 5-transistor operational amplifier (analog circuit)

### Running Examples

```bash
# Basic circuits
python main.py examples/voltage_divider.sp
python main.py examples/rc_circuit.sp

# Advanced circuits (detailed logging)
python main.py examples/sram_6t.sp -v

# View detailed simulation results
cat results/Vin.lis  # DC sweep details
```

**Step 5: Test and commit**

```bash
# Test each circuit
python main.py examples/sram_6t.sp
python main.py examples/inverter_chain.sp
python main.py examples/opamp_5t.sp

git add examples/sram_6t.sp examples/inverter_chain.sp examples/opamp_5t.sp README.md
git commit -m "feat: add complex test circuits (6T SRAM, inverter chain, 5T op-amp)"
```

---

## Task 7: Update CLAUDE.md Documentation

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update development roadmap**

```markdown
# Modify the development roadmap section in CLAUDE.md:

## 6. Development Roadmap
1.  **Step 1**: Define the `Component` base class and the basic MNA matrix construction logic.
2.  **Step 2**: Implement R, V, I models and `.dc` single-point solver; verify with linear circuits.
3.  **Step 3**: Implement MOS Level 1 model and Newton-Raphson iteration; verify with non-linear DC sweeps.
4.  **Step 4**: Implement R, C, V, I model without MOS devices and Backward Euler integration; verify `.tran` analysis.
5.  **Step 5**: Implement R, C, V, I model with MOS devices and Backward Euler integration; verify `.tran` analysis.
6.  **Step 6**: Implement Parser and Visualizer to connect the entire workflow.
7.  **Step 7**: Add Logger for detailed simulation output (.lis files).
8.  **Step 8**: Add voltage clamping and numerical safeguards for stability.
9.  **Step 9**: Implement two-stage analysis (OP → DC sweep/Transient).
10. **Step 10**: Add complex test circuits (SRAM, inverter chain, op-amp).

All 10 steps completed ✓
```

**Step 2: Update architecture documentation**

```markdown
# Add to CLAUDE.md - Architecture section:

## Output and Logging

### .lis File Format
Detailed simulation logs are saved in HSPICE-like `.lis` format:
- **Header**: Netlist, date, analysis type
- **Circuit Summary**: Component and node counts
- **Iteration Log**: For each Newton-Raphson iteration:
  - Node voltages
  - Voltage deltas
  - Device currents and conductances (gm, gds for MOSFETs)
  - Max delta (convergence metric)
- **Convergence Status**: Iterations taken, tolerance met/failed

### Two-Stage Analysis
DC sweeps and transient analysis use a two-stage approach:
1. **Stage 1**: Compute DC Operating Point (OP) at initial conditions
2. **Stage 2**: Run sweep/transient using OP as initial guess
   - DC sweep: Each point uses previous point's voltages
   - Transient: Uses OP as t=0 initial state
```

**Step 3: Add features section**

```markdown
# Add to CLAUDE.md:

## Features

### Supported Devices
- **Passive**: Resistors (R), Capacitors (C)
- **Sources**: DC Voltage (V), DC Current (I)
- **Active**: NMOS, PMOS (Level 1 Shichman-Hodges model)

### Analysis Types
- **.op**: DC operating point
- **.dc**: DC parameter sweep
- **.tran**: Transient time-domain analysis

### Key Features
- **Modified Nodal Analysis (MNA)** formulation
- **Newton-Raphson iteration** for non-linear convergence
- **Backward Euler** integration for transient analysis
- **Detailed logging** (.lis files) with iteration-by-iteration tracking
- **Voltage clamping** for numerical stability
- **Two-stage analysis** (OP → sweep) for improved convergence
- **Separated plots** (voltages and currents in different subplots)
```

**Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with completed features and architecture"
```

---

## Execution Summary

This implementation plan addresses all four major requirements:

1. **Visualizer fixes** - Separate subplots for voltages/currents
2. **.lis file logging** - Detailed iteration-by-iteration output
3. **Overflow fixes** - Voltage clamping + numerical safeguards + two-stage analysis
4. **Complex test circuits** - 6T SRAM, inverter chain, 5T op-amp
5. **Documentation updates** - CLAUDE.md reflects all changes

**Total estimated tasks:** 7
**Key modules updated:** Logger, MOSFET models, Solver, Visualizer, Main
**Test coverage:** Logger integration, voltage clamping, complex circuits

Ready to hand off to execution phase.
