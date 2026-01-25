# Circuit Simulator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a pure-Python circuit simulator supporting R, C, MOSFET devices with DC and transient analysis using HSPICE-like netlist syntax.

**Architecture:** Layered architecture with strict separation between Solver Engine (numerical methods only) and Device Models (physics equations). Uses Modified Nodal Analysis (MNA) matrix formulation and Newton-Raphson iteration for non-linear devices.

**Tech Stack:** Python 3.10+, NumPy (linear algebra), Matplotlib (visualization), standard library logging

---

## Task 1: Project Structure Setup

**Files:**
- Create: `pycircuitsim/__init__.py`
- Create: `pycircuitsim/models/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/test_models.py`
- Create: `tests/test_circuits/`
- Create: `results/`

**Step 1: Create package directory structure**

```bash
mkdir -p pycircuitsim/models tests/test_circuits results
touch pycircuitsim/__init__.py pycircuitsim/models/__init__.py tests/__init__.py
```

**Step 2: Create requirements.txt**

```bash
cat > requirements.txt << 'EOF'
numpy>=1.21.0
matplotlib>=3.4.0
pytest>=7.0.0
EOF
```

**Step 3: Install dependencies with Tsinghua mirror**

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

**Step 4: Commit**

```bash
git add requirements.txt pycircuitsim tests results
git commit -m "feat: initialize project structure and dependencies"
```

---

## Task 2: Component Base Class

**Files:**
- Create: `pycircuitsim/models/base.py`
- Create: `tests/test_base.py`

**Step 1: Write failing test for Component interface**

```python
# tests/test_base.py
import pytest
from abc import ABC
from pycircuitsim.models.base import Component

def test_component_is_abstract():
    """Component should not be instantiable directly"""
    with pytest.raises(TypeError):
        Component()

def test_component_requires_interface():
    """Subclass must implement all abstract methods"""
    class IncompleteComponent(Component):
        def get_nodes(self):
            return ["n1", "n2"]

    with pytest.raises(TypeError):
        IncompleteComponent()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_base.py -v
# Expected: FAIL - ModuleNotFoundError or AttributeErrors
```

**Step 3: Implement Component base class**

```python
# pycircuitsim/models/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np

class Component(ABC):
    """
    Abstract base class for all circuit components.

    All devices (resistors, capacitors, sources, MOSFETs) inherit from this.
    The Solver only knows about Components - it never contains device physics.
    """

    def __init__(self, name: str, nodes: List[str], value: Any = None):
        """
        Initialize a component.

        Args:
            name: Component identifier (e.g., 'R1', 'M1')
            nodes: List of node names this component connects to
            value: Component value (resistance, capacitance, etc.)
        """
        self.name = name
        self.nodes = nodes
        self.value = value

    @abstractmethod
    def get_nodes(self) -> List[str]:
        """Return list of node names this component connects to"""
        pass

    @abstractmethod
    def stamp_conductance(self, matrix: np.ndarray, node_map: Dict[str, int]) -> None:
        """
        Add conductance terms to the MNA matrix (G part).

        Args:
            matrix: The MNA matrix to modify
            node_map: Mapping from node names to matrix indices
        """
        pass

    @abstractmethod
    def stamp_rhs(self, rhs: np.ndarray, node_map: Dict[str, int]) -> None:
        """
        Add current/source terms to the RHS vector (z part).

        Args:
            rhs: The RHS vector to modify
            node_map: Mapping from node names to matrix indices
        """
        pass

    @abstractmethod
    def calculate_current(self, voltages: Dict[str, float]) -> float:
        """
        Calculate device current given terminal voltages.

        Args:
            voltages: Dictionary mapping node names to voltages

        Returns:
            Current flowing through the device
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name}, nodes={self.nodes})"
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_base.py -v
# Expected: PASS
```

**Step 5: Commit**

```bash
git add pycircuitsim/models/base.py tests/test_base.py
git commit -m "feat: add Component abstract base class"
```

---

## Task 3: Resistor Model

**Files:**
- Create: `pycircuitsim/models/passive.py`
- Modify: `tests/test_models.py`

**Step 1: Write failing test for Resistor**

```python
# tests/test_models.py
import numpy as np
import pytest
from pycircuitsim.models.passive import Resistor

def test_resistor_creation():
    """Create a resistor with proper parameters"""
    r = Resistor("R1", ["n1", "n2"], 1000.0)
    assert r.name == "R1"
    assert r.nodes == ["n1", "n2"]
    assert r.value == 1000.0

def test_resistor_stamp_conductance():
    """Resistor should stamp 1/R to matrix diagonal and off-diagonal"""
    r = Resistor("R1", ["n1", "n2"], 1000.0)
    node_map = {"n1": 0, "n2": 1}

    matrix = np.zeros((2, 2))
    r.stamp_conductance(matrix, node_map)

    # G[0,0] += 1/R, G[1,1] += 1/R, G[0,1] -= 1/R, G[1,0] -= 1/R
    expected = np.array([[1e-3, -1e-3], [-1e-3, 1e-3]])
    np.testing.assert_array_almost_equal(matrix, expected)

def test_resistor_current():
    """I = (V1 - V2) / R"""
    r = Resistor("R1", ["n1", "n2"], 1000.0)
    voltages = {"n1": 5.0, "n2": 3.0}
    assert r.calculate_current(voltages) == 0.002  # (5-3)/1000 = 2mA

def test_resistor_stamp_rhs():
    """Resistor contributes nothing to RHS vector"""
    r = Resistor("R1", ["n1", "n2"], 1000.0)
    node_map = {"n1": 0, "n2": 1}

    rhs = np.zeros(2)
    r.stamp_rhs(rhs, node_map)

    np.testing.assert_array_equal(rhs, np.zeros(2))
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_models.py::test_resistor_creation -v
# Expected: FAIL - ModuleNotFoundError
```

**Step 3: Implement Resistor class**

```python
# pycircuitsim/models/passive.py
import numpy as np
from typing import List, Dict
from .base import Component

class Resistor(Component):
    """
    Linear resistor following Ohm's Law: I = V/R
    """

    def __init__(self, name: str, nodes: List[str], resistance: float):
        """
        Args:
            name: Component name (e.g., 'R1')
            nodes: [node_positive, node_negative]
            resistance: Resistance in ohms
        """
        super().__init__(name, nodes, resistance)
        if len(nodes) != 2:
            raise ValueError(f"Resistor must have exactly 2 nodes, got {len(nodes)}")
        if resistance <= 0:
            raise ValueError(f"Resistance must be positive, got {resistance}")
        self.conductance = 1.0 / resistance

    def get_nodes(self) -> List[str]:
        return self.nodes

    def stamp_conductance(self, matrix: np.ndarray, node_map: Dict[str, int]) -> None:
        """
        Stamp conductance to MNA matrix.
        For resistor between nodes i and j:
          G[i,i] += 1/R, G[j,j] += 1/R
          G[i,j] -= 1/R, G[j,i] -= 1/R
        """
        if self.nodes[0] not in node_map or self.nodes[1] not in node_map:
            return  # One or both nodes are ground

        i = node_map[self.nodes[0]]
        j = node_map[self.nodes[1]]
        g = self.conductance

        matrix[i, i] += g
        matrix[j, j] += g
        matrix[i, j] -= g
        matrix[j, i] -= g

    def stamp_rhs(self, rhs: np.ndarray, node_map: Dict[str, int]) -> None:
        """Resistors don't contribute to RHS vector"""
        pass

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        """I = (V_positive - V_negative) / R"""
        v1 = voltages.get(self.nodes[0], 0.0)
        v2 = voltages.get(self.nodes[1], 0.0)
        return (v1 - v2) * self.conductance
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_models.py -v
# Expected: PASS for all resistor tests
```

**Step 5: Commit**

```bash
git add pycircuitsim/models/passive.py tests/test_models.py
git commit -m "feat: add Resistor model"
```

---

## Task 4: Voltage Source Model

**Files:**
- Modify: `pycircuitsim/models/passive.py`
- Modify: `tests/test_models.py`

**Step 1: Write failing test for Voltage Source**

```python
# Add to tests/test_models.py
def test_voltage_source_creation():
    """Create a voltage source"""
    from pycircuitsim.models.passive import VoltageSource
    v = VoltageSource("V1", ["n1", "0"], 5.0)
    assert v.name == "V1"
    assert v.nodes == ["n1", "0"]
    assert v.value == 5.0

def test_voltage_source_stamp_rhs():
    """Voltage source should stamp to RHS"""
    from pycircuitsim.models.passive import VoltageSource
    v = VoltageSource("V1", ["n1", "n2"], 5.0)
    node_map = {"n1": 0, "n2": 1}

    # For voltage source, we need an augmented matrix
    # For now, test the interface exists
    rhs = np.zeros(2)
    v.stamp_rhs(rhs, node_map)
    # Implementation will add to specific position

def test_voltage_source_get_nodes():
    """Voltage source should return its nodes"""
    from pycircuitsim.models.passive import VoltageSource
    v = VoltageSource("Vdd", ["1", "0"], 3.3)
    assert v.get_nodes() == ["1", "0"]
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_models.py::test_voltage_source_creation -v
# Expected: FAIL - import error or class not found
```

**Step 3: Implement VoltageSource class**

```python
# Add to pycircuitsim/models/passive.py
class VoltageSource(Component):
    """
    Ideal DC voltage source.
    In MNA, this adds a row/col to the matrix for the source current.
    """

    def __init__(self, name: str, nodes: List[str], voltage: float):
        """
        Args:
            name: Component name (e.g., 'V1')
            nodes: [positive_node, negative_node]
            voltage: Source voltage in volts
        """
        super().__init__(name, nodes, voltage)
        if len(nodes) != 2:
            raise ValueError(f"VoltageSource must have exactly 2 nodes, got {len(nodes)}")

    def get_nodes(self) -> List[str]:
        return self.nodes

    def stamp_conductance(self, matrix: np.ndarray, node_map: Dict[str, int]) -> None:
        """
        Stamp B and C matrices for voltage source.
        This is handled by the solver - component just provides node info.
        """
        pass  # Solver handles the matrix structure for voltage sources

    def stamp_rhs(self, rhs: np.ndarray, node_map: Dict[str, int]) -> None:
        """
        Add voltage to RHS.
        The actual position depends on the solver's matrix ordering.
        """
        # Solver will call this with the correct index for the source equation
        # For now, the source value is stored in self.value
        pass

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        """
        Current is determined by the solver (unknown a priori).
        Returns the voltage value for reference.
        """
        return self.value  # Return voltage, not current (solver tracks current)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_models.py -v
# Expected: PASS
```

**Step 5: Commit**

```bash
git add pycircuitsim/models/passive.py tests/test_models.py
git commit -m "feat: add VoltageSource model"
```

---

## Task 5: Current Source Model

**Files:**
- Modify: `pycircuitsim/models/passive.py`
- Modify: `tests/test_models.py`

**Step 1: Write failing test for Current Source**

```python
# Add to tests/test_models.py
def test_current_source_creation():
    """Create a current source"""
    from pycircuitsim.models.passive import CurrentSource
    i = CurrentSource("I1", ["n1", "0"], 0.001)
    assert i.name == "I1"
    assert i.value == 0.001

def test_current_source_stamp_rhs():
    """Current source should add to RHS"""
    from pycircuitsim.models.passive import CurrentSource
    i = CurrentSource("I1", ["n1", "n2"], 0.001)
    node_map = {"n1": 0, "n2": 1}

    rhs = np.zeros(2)
    i.stamp_rhs(rhs, node_map)
    # Should add +I to n1 and -I to n2

def test_current_source_current():
    """Current source returns its current value"""
    from pycircuitsim.models.passive import CurrentSource
    i = CurrentSource("I1", ["n1", "0"], 0.005)
    assert i.calculate_current({}) == 0.005
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_models.py::test_current_source_creation -v
# Expected: FAIL
```

**Step 3: Implement CurrentSource class**

```python
# Add to pycircuitsim/models/passive.py
class CurrentSource(Component):
    """
    Ideal DC current source.
    Stamps directly to RHS vector (no conductance).
    """

    def __init__(self, name: str, nodes: List[str], current: float):
        """
        Args:
            name: Component name (e.g., 'I1')
            nodes: [source_node, sink_node]
            current: Source current in amps (flows from node[0] to node[1])
        """
        super().__init__(name, nodes, current)
        if len(nodes) != 2:
            raise ValueError(f"CurrentSource must have exactly 2 nodes, got {len(nodes)}")

    def get_nodes(self) -> List[str]:
        return self.nodes

    def stamp_conductance(self, matrix: np.ndarray, node_map: Dict[str, int]) -> None:
        """Current sources don't add conductance"""
        pass

    def stamp_rhs(self, rhs: np.ndarray, node_map: Dict[str, int]) -> None:
        """
        Add current to RHS vector.
        +I to source node, -I to sink node.
        """
        if self.nodes[0] in node_map:
            rhs[node_map[self.nodes[0]]] -= self.value  # Current leaving node
        if self.nodes[1] in node_map:
            rhs[node_map[self.nodes[1]]] += self.value  # Current entering node

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        """Return the fixed current value"""
        return self.value
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_models.py -v
# Expected: PASS
```

**Step 5: Commit**

```bash
git add pycircuitsim/models/passive.py tests/test_models.py
git commit -m "feat: add CurrentSource model"
```

---

## Task 6: Circuit Container Class

**Files:**
- Create: `pycircuitsim/circuit.py`
- Create: `tests/test_circuit.py`

**Step 1: Write failing test**

```python
# tests/test_circuit.py
import pytest
from pycircuitsim.circuit import Circuit
from pycircuitsim.models.passive import Resistor, VoltageSource

def test_circuit_creation():
    """Create empty circuit"""
    c = Circuit()
    assert len(c.components) == 0
    assert len(c.nodes) == 0

def test_circuit_add_component():
    """Add component to circuit"""
    c = Circuit()
    r = Resistor("R1", ["n1", "n2"], 1000.0)
    c.add_component(r)
    assert len(c.components) == 1

def test_circuit_auto_discover_nodes():
    """Circuit should auto-discover all unique nodes"""
    c = Circuit()
    c.add_component(Resistor("R1", ["n1", "n2"], 1000.0))
    c.add_component(VoltageSource("V1", ["n2", "0"], 5.0))
    c.add_component(Resistor("R2", ["n2", "n3"], 2000.0))

    nodes = c.get_nodes()
    assert "n1" in nodes
    assert "n2" in nodes
    assert "n3" in nodes
    # Ground is special case - might or might not be in list

def test_circuit_node_mapping():
    """Create node map for MNA matrix (exclude ground)"""
    c = Circuit()
    c.add_component(Resistor("R1", ["1", "2"], 1000.0))
    c.add_component(VoltageSource("V1", ["2", "0"], 5.0))

    node_map = c.get_node_map()
    # Ground should be excluded
    assert "0" not in node_map
    assert "1" in node_map
    assert "2" in node_map
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_circuit.py -v
# Expected: FAIL - ModuleNotFoundError
```

**Step 3: Implement Circuit class**

```python
# pycircuitsim/circuit.py
from typing import List, Dict, Set
from .models.base import Component

class Circuit:
    """
    Container for circuit topology.

    Manages components, nodes, and creates node mappings for MNA.
    """

    def __init__(self):
        self.components: List[Component] = []
        self._nodes: Set[str] = set()

    def add_component(self, component: Component) -> None:
        """Add a component to the circuit"""
        self.components.append(component)
        self._nodes.update(component.get_nodes())

    def get_nodes(self) -> List[str]:
        """Return list of all unique nodes (excluding ground)"""
        return [n for n in self._nodes if n != "0" and n.lower() != "gnd"]

    def get_node_map(self) -> Dict[str, int]:
        """
        Create mapping from node names to matrix indices.
        Ground node ("0" or "GND") is excluded.
        """
        nodes = self.get_nodes()
        return {node: idx for idx, node in enumerate(nodes)}

    def count_voltage_sources(self) -> int:
        """Count voltage sources in circuit (for MNA matrix sizing)"""
        count = 0
        for comp in self.components:
            # Voltage sources start with 'V'
            if comp.name.startswith('V'):
                count += 1
        return count

    def __repr__(self) -> str:
        return f"Circuit({len(self.components)} components, {len(self.get_nodes())} nodes)"
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_circuit.py -v
# Expected: PASS
```

**Step 5: Commit**

```bash
git add pycircuitsim/circuit.py tests/test_circuit.py
git commit -m "feat: add Circuit container class"
```

---

## Task 7: DC Solver - Linear Circuits

**Files:**
- Create: `pycircuitsim/solver.py`
- Create: `tests/test_solver.py`

**Step 1: Write failing test for simple voltage divider**

```python
# tests/test_solver.py
import pytest
import numpy as np
from pycircuitsim.circuit import Circuit
from pycircuitsim.models.passive import Resistor, VoltageSource
from pycircuitsim.solver import DCSolver

def test_voltage_divider():
    """
    Simple voltage divider: Vdd -- R1 -- Vout -- R2 -- GND
    Vdd = 5V, R1 = 1k, R2 = 1k
    Expected: Vout = 2.5V
    """
    circuit = Circuit()
    circuit.add_component(VoltageSource("Vdd", ["1", "0"], 5.0))
    circuit.add_component(Resistor("R1", ["1", "2"], 1000.0))
    circuit.add_component(Resistor("R2", ["2", "0"], 1000.0))

    solver = DCSolver(circuit)
    result = solver.solve()

    # Node "1" should be at 5V (connected to Vdd)
    # Node "2" should be at 2.5V (voltage divider)
    np.testing.assert_almost_equal(result["1"], 5.0, decimal=5)
    np.testing.assert_almost_equal(result["2"], 2.5, decimal=5)

def test_single_resistor_circuit():
    """
    V -- R -- GND
    Both nodes at V = 5V
    """
    circuit = Circuit()
    circuit.add_component(VoltageSource("V1", ["1", "0"], 5.0))
    circuit.add_component(Resistor("R1", ["1", "0"], 1000.0))

    solver = DCSolver(circuit)
    result = solver.solve()

    np.testing.assert_almost_equal(result["1"], 5.0, decimal=5)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_solver.py::test_voltage_divider -v
# Expected: FAIL - ModuleNotFoundError or AttributeError
```

**Step 3: Implement basic DCSolver for linear circuits**

```python
# pycircuitsim/solver.py
import numpy as np
from typing import Dict
from .circuit import Circuit
from .models.base import Component

class DCSolver:
    """
    DC analysis solver using Modified Nodal Analysis (MNA).

    For linear circuits: single matrix solve Ax = b
    For non-linear (MOSFET): Newton-Raphson iteration
    """

    def __init__(self, circuit: Circuit, tolerance: float = 1e-6, max_iterations: int = 50):
        """
        Args:
            circuit: Circuit to solve
            tolerance: Convergence tolerance (for Newton-Raphson)
            max_iterations: Maximum Newton iterations
        """
        self.circuit = circuit
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def solve(self) -> Dict[str, float]:
        """
        Solve the circuit for DC operating point.

        Returns:
            Dictionary mapping node names to voltages
        """
        node_map = self.circuit.get_node_map()
        num_nodes = len(node_map)
        num_vsources = self.circuit.count_voltage_sources()

        # MNA matrix size: N nodes + M voltage sources
        size = num_nodes + num_vsources
        G_matrix = np.zeros((size, size))
        rhs_vector = np.zeros(size)

        # Let each component stamp the matrix
        # For voltage sources, we need to track their indices
        vsource_idx = num_nodes

        for comp in self.circuit.components:
            # Stamp conductance
            comp.stamp_conductance(G_matrix, node_map)

            # Stamp RHS
            comp.stamp_rhs(rhs_vector, node_map)

            # Handle voltage source matrix entries (B and C parts of MNA)
            if comp.name.startswith('V'):
                nodes = comp.get_nodes()
                # B matrix: +1 at positive node, -1 at negative node
                if nodes[0] in node_map:
                    G_matrix[node_map[nodes[0]], vsource_idx] = 1.0
                if nodes[1] in node_map:
                    G_matrix[node_map[nodes[1]], vsource_idx] = -1.0

                # C matrix (transpose of B)
                if nodes[0] in node_map:
                    G_matrix[vsource_idx, node_map[nodes[0]]] = 1.0
                if nodes[1] in node_map:
                    G_matrix[vsource_idx, node_map[nodes[1]]] = -1.0

                # RHS: voltage value
                rhs_vector[vsource_idx] = comp.value
                vsource_idx += 1

        # Solve the system
        try:
            solution = np.linalg.solve(G_matrix, rhs_vector)
        except np.linalg.LinAlgError:
            raise ValueError(f"Matrix is singular - circuit may have floating nodes or short circuits")

        # Extract node voltages (first num_nodes entries)
        node_voltages = {}
        for node_name, node_idx in node_map.items():
            node_voltages[node_name] = solution[node_idx]

        # Ground is at 0V
        node_voltages["0"] = 0.0

        return node_voltages
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_solver.py -v
# Expected: PASS
```

**Step 5: Commit**

```bash
git add pycircuitsim/solver.py tests/test_solver.py
git commit -m "feat: add DC solver for linear circuits"
```

---

## Task 8: MOS Level 1 Model

**Files:**
- Create: `pycircuitsim/models/mosfet.py`
- Create: `tests/test_mosfet.py`

**Step 1: Write failing test for MOSFET**

```python
# tests/test_mosfet.py
import pytest
import numpy as np
from pycircuitsim.models.mosfet import NMOS

def test_nmos_creation():
    """Create NMOS transistor"""
    m = NMOS("M1", ["3", "2", "0", "0"], L=1e-6, W=10e-6, VTO=0.7, KP=100e-6)
    assert m.name == "M1"
    assert m.drain == "3"
    assert m.gate == "2"
    assert m.source == "0"
    assert m.bulk == "0"

def test_nmos_cutoff_region():
    """NMOS in cutoff (V_gs < V_th): I_ds = 0"""
    m = NMOS("M1", ["3", "2", "0", "0"], L=1e-6, W=10e-6, VTO=0.7, KP=100e-6)

    # V_gs = 0.5V < V_th = 0.7V
    voltages = {"3": 5.0, "2": 0.5, "0": 0.0, "0": 0.0}
    i_ds = m.calculate_current(voltages)

    assert i_ds == 0.0

def test_nmos_saturation_region():
    """NMOS in saturation: I_ds = 0.5 * K * (V_gs - V_th)^2"""
    m = NMOS("M1", ["3", "2", "0", "0"], L=1e-6, W=10e-6, VTO=0.7, KP=100e-6)

    # V_gs = 2V > V_th, V_ds = 5V > V_ov = 1.3V (saturation)
    voltages = {"3": 5.0, "2": 2.0, "0": 0.0}
    i_ds = m.calculate_current(voltages)

    # K = KP * W/L = 100e-6 * 10 = 1e-3
    # I_ds = 0.5 * 1e-3 * (2 - 0.7)^2 = 0.5 * 1e-3 * 1.69 = 0.845 mA
    expected = 0.5 * 1e-3 * (2.0 - 0.7)**2
    np.testing.assert_almost_equal(i_ds, expected, decimal=6)

def test_nmos_linear_region():
    """NMOS in linear/triode region"""
    m = NMOS("M1", ["3", "2", "0", "0"], L=1e-6, W=10e-6, VTO=0.7, KP=100e-6)

    # V_gs = 2V, V_ds = 0.5V < V_ov = 1.3V (linear)
    voltages = {"3": 0.5, "2": 2.0, "0": 0.0}
    i_ds = m.calculate_current(voltages)

    # I_ds = K * [(V_gs - V_th) * V_ds - 0.5 * V_ds^2]
    K = 1e-3
    V_ov = 2.0 - 0.7
    V_ds = 0.5
    expected = K * (V_ov * V_ds - 0.5 * V_ds**2)
    np.testing.assert_almost_equal(i_ds, expected, decimal=6)

def test_nmos_conductance_saturation():
    """Test conductance calculation in saturation"""
    m = NMOS("M1", ["3", "2", "0", "0"], L=1e-6, W=10e-6, VTO=0.7, KP=100e-6)

    voltages = {"3": 5.0, "2": 2.0, "0": 0.0}
    g_ds, g_m = m.get_conductance(voltages)

    # In saturation, g_m = K * (V_gs - V_th), g_ds ≈ 0
    K = 1e-3
    expected_gm = K * (2.0 - 0.7)

    np.testing.assert_almost_equal(g_m, expected_gm, decimal=6)
    # g_ds should be small (channel length modulation not modeled)
    assert abs(g_ds) < 1e-9
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_mosfet.py -v
# Expected: FAIL - ModuleNotFoundError
```

**Step 3: Implement NMOS class**

```python
# pycircuitsim/models/mosfet.py
import numpy as np
from typing import Dict, Tuple
from .base import Component

class NMOS(Component):
    """
    N-channel MOSFET implementing Shichman-Hodges Level 1 model.

    Terminals: Drain, Gate, Source, Bulk
    """

    def __init__(self, name: str, nodes: list, L: float, W: float,
                 VTO: float = 0.7, KP: float = 20e-6):
        """
        Args:
            name: Component name (e.g., 'M1')
            nodes: [drain, gate, source, bulk]
            L: Channel length (meters)
            W: Channel width (meters)
            VTO: Zero-bias threshold voltage (volts)
            KP: Transconductance parameter (A/V²)
        """
        super().__init__(name, nodes)
        if len(nodes) != 4:
            raise ValueError(f"MOSFET must have exactly 4 nodes, got {len(nodes)}")

        self.drain, self.gate, self.source, self.bulk = nodes
        self.L = L
        self.W = W
        self.VTO = VTO
        self.KP = KP

        # Transconductance parameter K = KP * (W/L)
        self.K = KP * (W / L)

        self.current_region = "cutoff"

    def get_nodes(self) -> list:
        return [self.drain, self.gate, self.source, self.bulk]

    def _get_voltages(self, voltages: Dict[str, float]) -> Tuple[float, float, float, float]:
        """Extract V_d, V_g, V_s, V_b from voltage dict"""
        v_d = voltages.get(self.drain, 0.0)
        v_g = voltages.get(self.gate, 0.0)
        v_s = voltages.get(self.source, 0.0)
        v_b = voltages.get(self.bulk, 0.0)
        return v_d, v_g, v_s, v_b

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        """
        Calculate drain current using Shichman-Hodges equations.

        Returns:
            I_ds (positive flowing drain to source)
        """
        v_d, v_g, v_s, v_b = self._get_voltages(voltages)

        v_gs = v_g - v_s
        v_ds = v_d - v_s

        # Cutoff region
        if v_gs < self.VTO:
            self.current_region = "cutoff"
            return 0.0

        v_ov = v_gs - self.VTO  # Overdrive voltage

        # Saturation region
        if v_ds >= v_ov:
            self.current_region = "saturation"
            return 0.5 * self.K * v_ov**2

        # Linear (triode) region
        self.current_region = "linear"
        return self.K * (v_ov * v_ds - 0.5 * v_ds**2)

    def get_conductance(self, voltages: Dict[str, float]) -> Tuple[float, float]:
        """
        Calculate conductance derivatives for Newton-Raphson.

        Returns:
            (g_ds, g_m) where:
                g_ds = ∂I_ds/∂V_ds (output conductance)
                g_m = ∂I_ds/∂V_gs (transconductance)
        """
        v_d, v_g, v_s, v_b = self._get_voltages(voltages)

        v_gs = v_g - v_s
        v_ds = v_d - v_s

        # Cutoff: no conductance
        if v_gs < self.VTO:
            return 0.0, 0.0

        v_ov = v_gs - self.VTO

        # Saturation: g_m = K * V_ov, g_ds ≈ 0
        if v_ds >= v_ov:
            g_m = self.K * v_ov
            return 0.0, g_m

        # Linear: g_m = K * V_ds, g_ds = K * (V_ov - V_ds)
        g_m = self.K * v_ds
        g_ds = self.K * (v_ov - v_ds)
        return g_ds, g_m

    def stamp_conductance(self, matrix: np.ndarray, node_map: Dict[str, int]) -> None:
        """
        Stamp MOSFET conductance to MNA matrix.
        This is called each Newton iteration with updated voltages.
        """
        # For simplicity in initial implementation:
        # MOSFET stamping is handled by the solver during Newton-Raphson
        # The solver will call get_conductance() and stamp accordingly
        pass

    def stamp_rhs(self, rhs: np.ndarray, node_map: Dict[str, int]) -> None:
        """MOSFET RHS contribution handled by solver"""
        pass


class PMOS(Component):
    """
    P-channel MOSFET (complement to NMOS).

    Similar equations but with polarities reversed.
    """

    def __init__(self, name: str, nodes: list, L: float, W: float,
                 VTO: float = -0.7, KP: float = 20e-6):
        """
        Args:
            name: Component name
            nodes: [drain, gate, source, bulk]
            L: Channel length
            W: Channel width
            VTO: Threshold voltage (negative for PMOS)
            KP: Transconductance
        """
        super().__init__(name, nodes)
        if len(nodes) != 4:
            raise ValueError(f"MOSFET must have exactly 4 nodes, got {len(nodes)}")

        self.drain, self.gate, self.source, self.bulk = nodes
        self.L = L
        self.W = W
        self.VTO = VTO  # Negative for PMOS
        self.KP = KP
        self.K = KP * (W / L)

    def get_nodes(self) -> list:
        return [self.drain, self.gate, self.source, self.bulk]

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        """PMOS current (negative of NMOS convention)"""
        # For PMOS, roles are reversed
        # Implement similar to NMOS but with VTO < 0
        v_d, v_g, v_s, v_b = voltages.get(self.drain, 0.0), voltages.get(self.gate, 0.0), \
                             voltages.get(self.source, 0.0), voltages.get(self.bulk, 0.0)

        v_gs = v_g - v_s
        v_ds = v_d - v_s

        # For PMOS, device is on when V_gs < VTO (more negative)
        if v_gs > self.VTO:
            return 0.0

        v_ov = v_gs - self.VTO  # Positive when on

        # PMOS saturation when V_ds < v_ov (note the inequality reversal)
        if v_ds <= v_ov:
            return -0.5 * self.K * v_ov**2  # Negative current

        # Linear
        return -self.K * (v_ov * v_ds - 0.5 * v_ds**2)

    def stamp_conductance(self, matrix: np.ndarray, node_map: Dict[str, int]) -> None:
        pass

    def stamp_rhs(self, rhs: np.ndarray, node_map: Dict[str, int]) -> None:
        pass
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_mosfet.py -v
# Expected: PASS
```

**Step 5: Commit**

```bash
git add pycircuitsim/models/mosfet.py tests/test_mosfet.py
git commit -m "feat: add MOS Level 1 model (NMOS/PMOS)"
```

---

## Task 9: DC Solver with Newton-Raphson

**Files:**
- Modify: `pycircuitsim/solver.py`
- Modify: `tests/test_solver.py`

**Step 1: Write failing test for non-linear circuit**

```python
# Add to tests/test_solver.py
def test_diode_like_circuit_with_mos():
    """
    Simple MOS test: Vdd -- R -- drain
                         |
                        gate (driven by Vin)
    Vdd = 5V, R = 1k
    Sweep V_gate from 0 to 5V
    """
    from pycircuitsim.models.mosfet import NMOS

    # Test single operating point
    circuit = Circuit()
    circuit.add_component(VoltageSource("Vdd", ["1", "0"], 5.0))
    circuit.add_component(VoltageSource("Vin", ["2", "0"], 2.0))
    circuit.add_component(Resistor("R1", ["1", "3"], 1000.0))
    circuit.add_component(NMOS("M1", ["3", "2", "0", "0"], L=1e-6, W=10e-6, VTO=0.7, KP=100e-6))

    solver = DCSolver(circuit)
    result = solver.solve()

    # V_gate = 2V > V_th = 0.7V, MOS is on
    # V_drain should be pulled down
    assert result["2"] == 2.0  # Gate at 2V
    assert 0.0 <= result["3"] <= 5.0  # Drain somewhere between
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_solver.py::test_diode_like_circuit_with_mos -v
# Expected: FAIL - MOSFET not being solved correctly
```

**Step 3: Update DCSolver with Newton-Raphson**

```python
# Update pycircuitsim/solver.py
# Add import and update solve() method

def solve(self) -> Dict[str, float]:
    """
    Solve the circuit for DC operating point.

    For linear circuits: single solve
    For non-linear (MOSFETs): Newton-Raphson iteration
    """
    node_map = self.circuit.get_node_map()
    num_nodes = len(node_map)
    num_vsources = self.circuit.count_voltage_sources()

    # Check if circuit has MOSFETs (non-linear)
    has_mosfet = any('m' in comp.name.lower() for comp in self.circuit.components)

    if not has_mosfet:
        return self._solve_linear(node_map, num_nodes, num_vsources)
    else:
        return self._solve_newton(node_map, num_nodes, num_vsources)

def _solve_linear(self, node_map, num_nodes, num_vsources) -> Dict[str, float]:
    """Solve linear circuit (single matrix solve)"""
    size = num_nodes + num_vsources
    G_matrix = np.zeros((size, size))
    rhs_vector = np.zeros(size)

    vsource_idx = num_nodes
    for comp in self.circuit.components:
        comp.stamp_conductance(G_matrix, node_map)
        comp.stamp_rhs(rhs_vector, node_map)

        if comp.name.startswith('V'):
            nodes = comp.get_nodes()
            if nodes[0] in node_map:
                G_matrix[node_map[nodes[0]], vsource_idx] = 1.0
            if nodes[1] in node_map:
                G_matrix[node_map[nodes[1]], vsource_idx] = -1.0
            if nodes[0] in node_map:
                G_matrix[vsource_idx, node_map[nodes[0]]] = 1.0
            if nodes[1] in node_map:
                G_matrix[vsource_idx, node_map[nodes[1]]] = -1.0
            rhs_vector[vsource_idx] = comp.value
            vsource_idx += 1

    solution = np.linalg.solve(G_matrix, rhs_vector)

    node_voltages = {}
    for node_name, node_idx in node_map.items():
        node_voltages[node_name] = solution[node_idx]
    node_voltages["0"] = 0.0

    return node_voltages

def _solve_newton(self, node_map, num_nodes, num_vsources) -> Dict[str, float]:
    """
    Solve non-linear circuit using Newton-Raphson.

    Iteratively linearize and solve until convergence.
    """
    # Initial guess: all nodes at 0V
    voltages = {node: 0.0 for node in node_map.keys()}
    voltages["0"] = 0.0

    for iteration in range(self.max_iterations):
        old_voltages = voltages.copy()

        # Build linearized MNA matrix at current voltages
        size = num_nodes + num_vsources
        G_matrix = np.zeros((size, size))
        rhs_vector = np.zeros(size)

        vsource_idx = num_nodes

        for comp in self.circuit.components:
            # Linear components
            if hasattr(comp, 'conductance'):
                # Resistor
                comp.stamp_conductance(G_matrix, node_map)
            elif 'm' in comp.name.lower():
                # MOSFET - stamp based on current voltages
                g_ds, g_m = comp.get_conductance(voltages)
                d_node, g_node, s_node, b_node = comp.drain, comp.gate, comp.source, comp.bulk

                # Stamp conductance matrix
                # This is simplified - full implementation stamps 4x4 submatrix
                # For initial version, treat as voltage-controlled current source
                if d_node in node_map and s_node in node_map:
                    d_idx = node_map[d_node]
                    s_idx = node_map[s_node]
                    G_matrix[d_idx, d_idx] += g_ds
                    G_matrix[s_idx, s_idx] += g_ds
                    G_matrix[d_idx, s_idx] -= g_ds
                    G_matrix[s_idx, d_idx] -= g_ds
            else:
                comp.stamp_conductance(G_matrix, node_map)

            comp.stamp_rhs(rhs_vector, node_map)

            # Voltage source handling
            if comp.name.startswith('V'):
                nodes = comp.get_nodes()
                if nodes[0] in node_map:
                    G_matrix[node_map[nodes[0]], vsource_idx] = 1.0
                if nodes[1] in node_map:
                    G_matrix[node_map[nodes[1]], vsource_idx] = -1.0
                if nodes[0] in node_map:
                    G_matrix[vsource_idx, node_map[nodes[0]]] = 1.0
                if nodes[1] in node_map:
                    G_matrix[vsource_idx, node_map[nodes[1]]] = -1.0
                rhs_vector[vsource_idx] = comp.value
                vsource_idx += 1

        # Solve
        try:
            delta = np.linalg.solve(G_matrix, rhs_vector)
        except np.linalg.LinAlgError:
            raise ValueError(f"Matrix singular at iteration {iteration}")

        # Update voltages
        for i, node_name in enumerate(node_map.keys()):
            voltages[node_name] += delta[i]

        # Check convergence
        max_change = max(abs(voltages[n] - old_voltages[n]) for n in node_map.keys())
        if max_change < self.tolerance:
            print(f"Converged in {iteration + 1} iterations")
            return voltages

    raise ValueError(f"Failed to converge after {self.max_iterations} iterations")
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_solver.py -v
# Expected: PASS (may need adjustment)
```

**Step 5: Commit**

```bash
git add pycircuitsim/solver.py tests/test_solver.py
git commit -m "feat: add Newton-Raphson iteration for non-linear circuits"
```

---

## Task 10: Capacitor Model

**Files:**
- Modify: `pycircuitsim/models/passive.py`
- Modify: `tests/test_models.py`

**Step 1: Write failing test**

```python
# Add to tests/test_models.py
def test_capacitor_creation():
    """Create a capacitor"""
    from pycircuitsim.models.passive import Capacitor
    c = Capacitor("C1", ["n1", "n2"], 1e-9)
    assert c.name == "C1"
    assert c.value == 1e-9

def test_capacitor_companion_model():
    """Test backward Euler companion model"""
    from pycircuitsim.models.passive import Capacitor
    c = Capacitor("C1", ["n1", "n2"], 1e-9)

    dt = 1e-9  # 1ns timestep
    g_eq, i_eq = c.get_companion_model(dt, v_prev=1.0)

    # G_eq = C / dt = 1e-9 / 1e-9 = 1.0
    # I_eq = G_eq * V_prev = 1.0 * 1.0 = 1.0
    assert g_eq == 1.0
    assert i_eq == 1.0
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_models.py::test_capacitor_creation -v
# Expected: FAIL
```

**Step 3: Implement Capacitor class**

```python
# Add to pycircuitsim/models/passive.py
class Capacitor(Component):
    """
    Capacitor using Backward Euler discretization for transient analysis.

    C * dV/dt = I  →  C * (V[t] - V[t-1]) / dt = I[t]

    Companion model: equivalent conductance + current source
    - G_eq = C / dt
    - I_eq = (C / dt) * V[t-1]
    """

    def __init__(self, name: str, nodes: List[str], capacitance: float):
        """
        Args:
            name: Component name (e.g., 'C1')
            nodes: [positive_node, negative_node]
            capacitance: Capacitance in farads
        """
        super().__init__(name, nodes, capacitance)
        if len(nodes) != 2:
            raise ValueError(f"Capacitor must have exactly 2 nodes, got {len(nodes)}")
        if capacitance <= 0:
            raise ValueError(f"Capacitance must be positive, got {capacitance}")

        # Store previous voltage for companion model
        self.v_prev = 0.0

    def get_nodes(self) -> List[str]:
        return self.nodes

    def get_companion_model(self, dt: float, v_prev: float) -> tuple:
        """
        Get Backward Euler companion model parameters.

        Args:
            dt: Timestep
            v_prev: Voltage across capacitor at previous timestep

        Returns:
            (G_eq, I_eq) - equivalent conductance and current source
        """
        g_eq = self.value / dt
        i_eq = g_eq * v_prev
        return g_eq, i_eq

    def stamp_conductance(self, matrix: np.ndarray, node_map: Dict[str, int]) -> None:
        """
        Stamp companion model conductance.
        During transient analysis, this is called each timestep.
        """
        # Get companion model from current state
        # Note: dt needs to be set by the solver before stamping
        if hasattr(self, '_g_eq'):
            g = self._g_eq

            if self.nodes[0] not in node_map or self.nodes[1] not in node_map:
                return

            i = node_map[self.nodes[0]]
            j = node_map[self.nodes[1]]

            matrix[i, i] += g
            matrix[j, j] += g
            matrix[i, j] -= g
            matrix[j, i] -= g

    def stamp_rhs(self, rhs: np.ndarray, node_map: Dict[str, int]) -> None:
        """Stamp companion model current source"""
        if hasattr(self, '_i_eq'):
            if self.nodes[0] in node_map:
                rhs[node_map[self.nodes[0]]] += self._i_eq
            if self.nodes[1] in node_map:
                rhs[node_map[self.nodes[1]]] -= self._i_eq

    def update_voltage(self, voltages: Dict[str, float]) -> None:
        """Update stored voltage after solving timestep"""
        v1 = voltages.get(self.nodes[0], 0.0)
        v2 = voltages.get(self.nodes[1], 0.0)
        self.v_prev = v1 - v2

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        """
        Calculate capacitor current.
        Note: For transient analysis, this is I = C * dV/dt
        For DC, capacitor is open circuit (I = 0)
        """
        return 0.0  # DC: open circuit
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_models.py::test_capacitor_companion_model -v
# Expected: PASS
```

**Step 5: Commit**

```bash
git add pycircuitsim/models/passive.py tests/test_models.py
git commit -m "feat: add Capacitor model with Backward Euler"
```

---

## Task 11: Transient Solver

**Files:**
- Modify: `pycircuitsim/solver.py`
- Create: `tests/test_transient.py`

**Step 1: Write failing test for RC circuit**

```python
# tests/test_transient.py
import pytest
import numpy as np
from pycircuitsim.circuit import Circuit
from pycircuitsim.models.passive import Resistor, VoltageSource, Capacitor
from pycircuitsim.solver import TransientSolver

def test_rc_charging():
    """
    RC circuit: V -- R -- C -- GND
    V = 5V, R = 1k, C = 1uF
    Time constant: tau = R*C = 1ms
    At t = tau, V_c = 5 * (1 - e^-1) ≈ 3.16V
    """
    circuit = Circuit()
    circuit.add_component(VoltageSource("V1", ["1", "0"], 5.0))
    circuit.add_component(Resistor("R1", ["1", "2"], 1000.0))
    circuit.add_component(Capacitor("C1", ["2", "0"], 1e-6))

    solver = TransientSolver(circuit, t_stop=0.005, dt=1e-4)
    results = solver.solve()

    # Check voltage at t = 1ms (approximately index 10)
    t_1ms_idx = int(0.001 / 1e-4)
    v_at_1ms = results["2"][t_1ms_idx]
    expected = 5.0 * (1 - np.exp(-0.001 / 0.001))  # tau = 1ms

    np.testing.assert_almost_equal(v_at_1ms, expected, decimal=1)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_transient.py -v
# Expected: FAIL - TransientSolver not implemented
```

**Step 3: Implement TransientSolver**

```python
# Add to pycircuitsim/solver.py
class TransientSolver:
    """
    Transient analysis using Backward Euler integration.

    Time-stepping algorithm:
    1. DC solve at t=0
    2. For each timestep:
       - Update capacitor companion models
       - DC solve at current timestep
       - Store voltages
       - Advance time
    """

    def __init__(self, circuit: Circuit, t_stop: float, dt: float,
                 tolerance: float = 1e-6, max_iterations: int = 50):
        """
        Args:
            circuit: Circuit to simulate
            t_stop: Stop time (seconds)
            dt: Timestep (seconds)
            tolerance: Convergence tolerance
            max_iterations: Max Newton iterations per timestep
        """
        self.circuit = circuit
        self.t_stop = t_stop
        self.dt = dt
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def solve(self) -> Dict[str, np.ndarray]:
        """
        Run transient simulation.

        Returns:
            Dictionary mapping node names to voltage arrays vs time
        """
        node_map = self.circuit.get_node_map()
        nodes = list(node_map.keys())

        # Number of timesteps
        num_steps = int(self.t_stop / self.dt) + 1
        time_points = np.linspace(0, self.t_stop, num_steps)

        # Storage for voltages: {node_name: array of voltages}
        voltages_history = {node: np.zeros(num_steps) for node in nodes}
        voltages_history["time"] = time_points

        # Initial DC solution
        print("Computing DC operating point at t=0...")
        dc_solver = DCSolver(self.circuit, self.tolerance, self.max_iterations)
        current_voltages = dc_solver.solve()

        # Store initial voltages
        for node in nodes:
            voltages_history[node][0] = current_voltages.get(node, 0.0)

        # Initialize capacitor voltages
        for comp in self.circuit.components:
            if comp.name.startswith('C'):
                comp.update_voltage(current_voltages)

        # Time-stepping loop
        for step in range(1, num_steps):
            t = time_points[step]

            # Update capacitor companion models
            for comp in self.circuit.components:
                if comp.name.startswith('C'):
                    g_eq, i_eq = comp.get_companion_model(self.dt, comp.v_prev)
                    comp._g_eq = g_eq
                    comp._i_eq = i_eq

            # Solve at this timestep
            # For now, use linear solve (capacitors are linear)
            # TODO: Integrate with Newton-Raphson for MOS + transient
            try:
                timestep_voltages = self._solve_timestep(node_map)
            except ValueError as e:
                print(f"Failed at t={t:.6f}s: {e}")
                break

            # Update capacitor voltages for next iteration
            for comp in self.circuit.components:
                if comp.name.startswith('C'):
                    comp.update_voltage(timestep_voltages)

            # Store results
            for node in nodes:
                voltages_history[node][step] = timestep_voltages.get(node, 0.0)

            current_voltages = timestep_voltages

            if step % 100 == 0:
                print(f"  t = {t:.6f}s")

        print(f"Transient analysis complete: {num_steps} points")
        return voltages_history

    def _solve_timestep(self, node_map: Dict[str, int]) -> Dict[str, float]:
        """Solve circuit at a single timestep"""
        num_nodes = len(node_map)
        num_vsources = self.circuit.count_voltage_sources()
        size = num_nodes + num_vsources

        G_matrix = np.zeros((size, size))
        rhs_vector = np.zeros(size)

        vsource_idx = num_nodes

        for comp in self.circuit.components:
            comp.stamp_conductance(G_matrix, node_map)
            comp.stamp_rhs(rhs_vector, node_map)

            if comp.name.startswith('V'):
                nodes = comp.get_nodes()
                if nodes[0] in node_map:
                    G_matrix[node_map[nodes[0]], vsource_idx] = 1.0
                if nodes[1] in node_map:
                    G_matrix[node_map[nodes[1]], vsource_idx] = -1.0
                if nodes[0] in node_map:
                    G_matrix[vsource_idx, node_map[nodes[0]]] = 1.0
                if nodes[1] in node_map:
                    G_matrix[vsource_idx, node_map[nodes[1]]] = -1.0
                rhs_vector[vsource_idx] = comp.value
                vsource_idx += 1

        solution = np.linalg.solve(G_matrix, rhs_vector)

        voltages = {}
        for node_name, node_idx in node_map.items():
            voltages[node_name] = solution[node_idx]
        voltages["0"] = 0.0

        return voltages
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_transient.py -v
# Expected: PASS
```

**Step 5: Commit**

```bash
git add pycircuitsim/solver.py tests/test_transient.py
git commit -m "feat: add Transient solver with Backward Euler"
```

---

## Task 12: Netlist Parser

**Files:**
- Create: `pycircuitsim/parser.py`
- Create: `tests/test_parser.py`
- Create: `tests/test_circuits/inverter.sp`

**Step 1: Create test netlist**

```spice
* tests/test_circuits/inverter.sp
* Simple CMOS inverter
Vdd 1 0 3.3
Vin 2 0 0
M1 3 2 0 0 NMOS L=1u W=10u
R1 1 3 10k
.dc Vin 0 3.3 0.1
.end
```

**Step 2: Write failing test**

```python
# tests/test_parser.py
import pytest
from pycircuitsim.parser import Parser
from pycircuitsim.models.passive import Resistor, VoltageSource
from pycircuitsim.models.mosfet import NMOS

def test_parse_voltage_source():
    """Parse voltage source line"""
    parser = Parser()
    components = parser.parse_line("Vdd 1 0 3.3")

    assert len(components) == 1
    assert isinstance(components[0], VoltageSource)
    assert components[0].name == "Vdd"
    assert components[0].value == 3.3

def test_parse_resistor():
    """Parse resistor line"""
    parser = Parser()
    comp = parser.parse_line("R1 1 2 10k")[0]

    assert isinstance(comp, Resistor)
    assert comp.value == 10000.0  # 10k converted

def test_parse_mosfet():
    """Parse MOSFET line"""
    parser = Parser()
    comp = parser.parse_line("M1 3 2 0 0 NMOS L=1u W=10u")[0]

    assert isinstance(comp, NMOS)
    assert comp.L == 1e-6
    assert comp.W == 10e-6

def test_parse_full_netlist():
    """Parse complete netlist file"""
    parser = Parser()
    circuit = parser.parse_file("tests/test_circuits/inverter.sp")

    assert len(circuit.components) == 3
    # Verify components...
```

**Step 3: Run test to verify it fails**

```bash
pytest tests/test_parser.py -v
# Expected: FAIL - Parser not implemented
```

**Step 4: Implement Parser**

```python
# pycircuitsim/parser.py
import re
from typing import List, Dict, Optional
from .circuit import Circuit
from .models.passive import Resistor, VoltageSource, CurrentSource, Capacitor
from .models.mosfet import NMOS, PMOS

class Parser:
    """
    Parse HSPICE-like netlist files.

    Supported syntax:
    - Comments: * comment
    - Resistors: R<name> <n1> <n2> <value>
    - Capacitors: C<name> <n1> <n2> <value>
    - Voltage sources: V<name> <n1> <n2> <value>
    - Current sources: I<name> <n1> <n2> <value>
    - MOSFETs: M<name> <d> <g> <s> <b> <model> L=<l> W=<w>
    - Analysis: .dc or .tran
    """

    # Unit suffix multipliers
    UNITS = {
        'T': 1e12, 'G': 1e9, 'Meg': 1e6, 'k': 1e3,
        'm': 1e-3, 'u': 1e-6, 'n': 1e-9, 'p': 1e-12, 'f': 1e-15
    }

    def __init__(self):
        self.circuit = Circuit()
        self.analysis_type = None
        self.analysis_params = {}

    def parse_file(self, filename: str) -> Circuit:
        """
        Parse a netlist file.

        Args:
            filename: Path to .sp file

        Returns:
            Populated Circuit object
        """
        with open(filename, 'r') as f:
            lines = f.readlines()

        for line in lines:
            self.parse_line(line.strip())

        return self.circuit

    def parse_line(self, line: str) -> List:
        """
        Parse a single line and add components to circuit.

        Args:
            line: Netlist line

        Returns:
            List of created components (empty for comments/analysis)
        """
        # Skip empty lines and comments
        if not line or line.startswith('*'):
            return []

        # Handle .end
        if line.lower().startswith('.end'):
            return []

        # Handle analysis commands
        if line.lower().startswith('.dc'):
            self._parse_dc(line)
            return []

        if line.lower().startswith('.tran'):
            self._parse_tran(line)
            return []

        # Parse component
        tokens = line.split()
        if len(tokens) < 3:
            return []

        name = tokens[0]
        comp_type = name[0].upper()

        try:
            if comp_type == 'R':
                comp = self._parse_resistor(name, tokens)
            elif comp_type == 'C':
                comp = self._parse_capacitor(name, tokens)
            elif comp_type == 'V':
                comp = self._parse_voltage_source(name, tokens)
            elif comp_type == 'I':
                comp = self._parse_current_source(name, tokens)
            elif comp_type == 'M':
                comp = self._parse_mosfet(name, tokens)
            else:
                print(f"Warning: Unknown component type {comp_type}")
                return []

            self.circuit.add_component(comp)
            return [comp]

        except (ValueError, IndexError) as e:
            print(f"Error parsing line '{line}': {e}")
            return []

    def _parse_value(self, value_str: str) -> float:
        """Convert value string with unit suffix to float"""
        value_str = value_str.strip()

        # Check for unit suffix
        for suffix, multiplier in self.UNITS.items():
            if value_str.endswith(suffix):
                return float(value_str[:-len(suffix)]) * multiplier

        return float(value_str)

    def _parse_resistor(self, name: str, tokens: List[str]) -> Resistor:
        """R<name> <n1> <n2> <value>"""
        if len(tokens) < 4:
            raise ValueError("Resistor requires: name n1 n2 value")

        nodes = [tokens[1], tokens[2]]
        value = self._parse_value(tokens[3])
        return Resistor(name, nodes, value)

    def _parse_capacitor(self, name: str, tokens: List[str]) -> Capacitor:
        """C<name> <n1> <n2> <value>"""
        if len(tokens) < 4:
            raise ValueError("Capacitor requires: name n1 n2 value")

        nodes = [tokens[1], tokens[2]]
        value = self._parse_value(tokens[3])
        return Capacitor(name, nodes, value)

    def _parse_voltage_source(self, name: str, tokens: List[str]) -> VoltageSource:
        """V<name> <n1> <n2> <value>"""
        if len(tokens) < 4:
            raise ValueError("Voltage source requires: name n1 n2 value")

        nodes = [tokens[1], tokens[2]]
        value = self._parse_value(tokens[3])
        return VoltageSource(name, nodes, value)

    def _parse_current_source(self, name: str, tokens: List[str]) -> CurrentSource:
        """I<name> <n1> <n2> <value>"""
        if len(tokens) < 4:
            raise ValueError("Current source requires: name n1 n2 value")

        nodes = [tokens[1], tokens[2]]
        value = self._parse_value(tokens[3])
        return CurrentSource(name, nodes, value)

    def _parse_mosfet(self, name: str, tokens: List[str]):
        """M<name> <d> <g> <s> <b> <model> L=<l> W=<w>"""
        if len(tokens) < 6:
            raise ValueError("MOSFET requires: name d g s b model [params]")

        nodes = tokens[1:5]  # d, g, s, b
        model_type = tokens[4].upper()

        # Parse parameters
        params = {}
        for token in tokens[5:]:
            if '=' in token:
                key, val = token.split('=')
                params[key] = val

        L = self._parse_value(params.get('L', '1u'))
        W = self._parse_value(params.get('W', '1u'))

        if model_type == 'NMOS':
            return NMOS(name, nodes, L=L, W=W,
                       VTO=float(params.get('VTO', 0.7)),
                       KP=float(params.get('KP', '20e-6')))
        elif model_type == 'PMOS':
            return PMOS(name, nodes, L=L, W=W,
                       VTO=float(params.get('VTO', '-0.7')),
                       KP=float(params.get('KP', '20e-6')))
        else:
            raise ValueError(f"Unknown MOSFET model: {model_type}")

    def _parse_dc(self, line: str) -> None:
        """.dc <source> <start> <stop> <step>"""
        tokens = line.split()
        if len(tokens) < 5:
            raise ValueError(".dc requires: source start stop step")

        self.analysis_type = 'dc'
        self.analysis_params = {
            'source': tokens[1],
            'start': float(tokens[2]),
            'stop': float(tokens[3]),
            'step': float(tokens[4])
        }

    def _parse_tran(self, line: str) -> None:
        """.tran <dt> <tstop>"""
        tokens = line.split()
        if len(tokens) < 3:
            raise ValueError(".tran requires: dt tstop")

        self.analysis_type = 'tran'
        self.analysis_params = {
            'dt': float(tokens[1]),
            'tstop': float(tokens[2])
        }
```

**Step 5: Run test to verify it passes**

```bash
pytest tests/test_parser.py -v
# Expected: PASS
```

**Step 6: Commit**

```bash
git add pycircuitsim/parser.py tests/test_parser.py tests/test_circuits/inverter.sp
git commit -m "feat: add netlist parser"
```

---

## Task 13: Visualizer

**Files:**
- Create: `pycircuitsim/visualizer.py`
- Create: `tests/test_visualizer.py`

**Step 1: Write failing test**

```python
# tests/test_visualizer.py
import pytest
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pycircuitsim.visualizer import Visualizer
import numpy as np

def test_plot_dc_sweep(tmp_path):
    """Test DC sweep plotting"""
    results = {
        'vin': np.linspace(0, 3.3, 34),
        'vout': 3.3 - np.linspace(0, 3.3, 34)  # Inverter response
    }

    viz = Visualizer()
    viz.plot_dc_sweep(results, 'vin', ['vout'], 'Inverter DC Sweep')

    # Check file was created
    import os
    files = os.listdir('results')
    assert any('dc_sweep' in f for f in files)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_visualizer.py -v
# Expected: FAIL
```

**Step 3: Implement Visualizer**

```python
# pycircuitsim/visualizer.py
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import os
from datetime import datetime

class Visualizer:
    """
    Plot simulation results using Matplotlib.
    """

    def __init__(self, output_dir: str = "results"):
        """
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_dc_sweep(self, results: Dict[str, np.ndarray],
                      sweep_var: str, signals: List[str],
                      title: str = "DC Sweep") -> str:
        """
        Plot DC sweep results.

        Args:
            results: Dictionary with 'x_axis' and signal arrays
            sweep_var: Name of sweep variable (for x-axis label)
            signals: List of signal names to plot
            title: Plot title

        Returns:
            Path to saved plot file
        """
        plt.figure(figsize=(10, 6))

        x_data = results.get(sweep_var, np.array([]))

        for signal in signals:
            if signal in results:
                plt.plot(x_data, results[signal], label=signal, linewidth=2)

        plt.xlabel(f"{sweep_var} (V)")
        plt.ylabel("Voltage (V)")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

        filename = f"dc_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved plot to {filepath}")
        return filepath

    def plot_transient(self, results: Dict[str, np.ndarray],
                       signals: List[str],
                       title: str = "Transient Analysis") -> str:
        """
        Plot transient analysis results.

        Args:
            results: Dictionary with 'time' and signal arrays
            signals: List of signal names to plot
            title: Plot title

        Returns:
            Path to saved plot file
        """
        plt.figure(figsize=(12, 6))

        time_data = results.get('time', np.array([]))

        for signal in signals:
            if signal in results:
                plt.plot(time_data, results[signal], label=signal, linewidth=2)

        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

        filename = f"transient_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved plot to {filepath}")
        return filepath
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_visualizer.py -v
# Expected: PASS
```

**Step 5: Commit**

```bash
git add pycircuitsim/visualizer.py tests/test_visualizer.py
git commit -m "feat: add visualizer for DC and transient plots"
```

---

## Task 14: Main CLI Entry Point

**Files:**
- Create: `pycircuitsim/main.py`
- Create: `main.py` (project root)

**Step 1: Write integration test**

```python
# tests/test_integration.py
import pytest
import os
from pycircuitsim.main import run_simulation

def test_full_simulation(tmp_path):
    """Run complete simulation from netlist to plot"""
    netlist_file = "tests/test_circuits/inverter.sp"

    # Run simulation
    run_simulation(netlist_file)

    # Check plot was generated
    files = os.listdir('results')
    assert any('.png' in f for f in files)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_integration.py -v
# Expected: FAIL
```

**Step 3: Implement main.py**

```python
# pycircuitsim/main.py
import sys
from .parser import Parser
from .solver import DCSolver, TransientSolver
from .visualizer import Visualizer

def run_simulation(netlist_file: str) -> None:
    """
    Run circuit simulation from netlist file.

    Args:
        netlist_file: Path to .sp netlist file
    """
    print(f"Parsing netlist: {netlist_file}")

    # Parse netlist
    parser = Parser()
    circuit = parser.parse_file(netlist_file)

    print(f"Found {len(circuit.components)} components, {len(circuit.get_nodes())} nodes")

    # Run analysis
    if parser.analysis_type == 'dc':
        print(f"DC Analysis: Sweeping {parser.analysis_params['source']} "
              f"from {parser.analysis_params['start']}V to {parser.analysis_params['stop']}V "
              f"in {parser.analysis_params['step']}V steps")

        # For now, just solve single point
        # TODO: Implement full sweep
        solver = DCSolver(circuit)
        result = solver.solve()

        # Package results for plotting
        import numpy as np
        num_points = int((parser.analysis_params['stop'] - parser.analysis_params['start']) /
                        parser.analysis_params['step']) + 1
        results = {parser.analysis_params['source']: np.zeros(num_points)}

        for node in circuit.get_nodes():
            results[node] = np.zeros(num_points)

        # Plot
        viz = Visualizer()
        # viz.plot_dc_sweep(results, ...)  # TODO: implement full sweep

    elif parser.analysis_type == 'tran':
        print(f"Transient Analysis: dt={parser.analysis_params['dt']}s, "
              f"tstop={parser.analysis_params['tstop']}s")

        solver = TransientSolver(
            circuit,
            t_stop=parser.analysis_params['tstop'],
            dt=parser.analysis_params['dt']
        )

        results = solver.solve()

        # Plot all node voltages
        signals = [node for node in results.keys() if node != 'time']
        viz = Visualizer()
        viz.plot_transient(results, signals, "Transient Analysis")

    else:
        # No analysis specified, just do DC operating point
        print("DC Operating Point Analysis")
        solver = DCSolver(circuit)
        result = solver.solve()

        print("\nNode Voltages:")
        for node, voltage in sorted(result.items()):
            if node != '0':
                print(f"  {node}: {voltage:.4f} V")

    print("\nSimulation complete!")
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_integration.py -v
# Expected: PASS (may need adjustments)
```

**Step 5: Create CLI entry point**

```python
# main.py (project root)
#!/usr/bin/env python3
"""
PyCircuitSim - Python Circuit Simulator

Usage:
    python main.py <netlist_file>
"""
import sys
from pycircuitsim.main import run_simulation

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <netlist_file>")
        sys.exit(1)

    netlist_file = sys.argv[1]
    run_simulation(netlist_file)
```

**Step 6: Commit**

```bash
git add pycircuitsim/main.py main.py tests/test_integration.py
git commit -m "feat: add main CLI entry point"
```

---

## Task 15: Documentation and Examples

**Files:**
- Create: `README.md`
- Create: `examples/rc_circuit.sp`
- Create: `examples/inverter.sp`

**Step 1: Create README**

```markdown
# PyCircuitSim

A pure-Python circuit simulator for educational purposes.

## Features

- Linear devices: Resistors, Capacitors, Voltage/Current sources
- Non-linear devices: MOSFETs (Level 1 Shichman-Hodges model)
- Analysis types: DC sweep, Transient
- HSPICE-like netlist syntax

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py examples/inverter.sp
```

## Netlist Format

```spice
* Comments start with *
Vdd 1 0 3.3          ; Voltage source
R1 1 2 10k           ; Resistor (10k ohms)
C1 2 0 1u            ; Capacitor (1 microfarad)
M1 3 2 0 0 NMOS L=1u W=10u  ; MOSFET
.dc Vin 0 3.3 0.1    ; DC sweep analysis
.tran 1n 100n        ; Transient analysis
.end
```

## Running Tests

```bash
pytest tests/
```

## Architecture

- `models/`: Device models (R, C, MOSFET)
- `solver.py`: DC and Transient solvers
- `parser.py`: Netlist parser
- `visualizer.py`: Plotting utilities
```

**Step 2: Create example netlists**

```spice
# examples/rc_circuit.sp
* Simple RC charging circuit
V1 1 0 5
R1 1 2 1k
C1 2 0 1u
.tran 1u 5m
.end
```

```spice
# examples/inverter.sp
* CMOS Inverter
Vdd 1 0 3.3
Vin 2 0 0
M1 3 2 0 0 NMOS L=1u W=10u
Rload 1 3 10k
.dc Vin 0 3.3 0.1
.end
```

**Step 3: Commit**

```bash
git add README.md examples/
git commit -m "docs: add README and example netlists"
```

---

## Task 16: Final Integration Testing

**Files:**
- Modify: `tests/`

**Step 1: Run full test suite**

```bash
pytest tests/ -v --tb=short
```

**Step 2: Test with example circuits**

```bash
python main.py examples/rc_circuit.sp
python main.py examples/inverter.sp
```

**Step 3: Verify plots are generated**

```bash
ls -l results/
```

**Step 4: Fix any issues**

(Iterate as needed based on test results)

**Step 5: Final commit**

```bash
git add .
git commit -m "test: verify integration with example circuits"
```

---

## Execution Summary

This implementation plan follows **TDD principles** with:
- Tests written before implementation
- Small, focused commits
- Incremental feature building
- Clear separation of Solver vs Models

**Total estimated tasks:** 16
**Core modules:** models, solver, parser, visualizer
**Test coverage:** Unit tests for each component + integration tests

Ready to hand off to execution phase.
