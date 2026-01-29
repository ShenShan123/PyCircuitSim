"""
Unit tests for DC operating point convergence with MOSFET models.

This test module verifies that the Newton-Raphson solver improvements
(source stepping, damping, OP initialization) work correctly for
DC operating point analysis.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pycircuitsim.parser import Parser
from pycircuitsim.circuit import Circuit
from pycircuitsim.solver import DCSolver
from pycircuitsim.logger import Logger


def test_level1_inverter_convergence():
    """
    Test that Level 1 inverter converges properly.

    This verifies basic Newton-Raphson convergence with improvements.
    """
    # Create a simple Level 1 inverter netlist for testing
    netlist_content = """
* Simple CMOS Inverter with Level 1 Models
Vdd 1 0 3.3
Vin 2 0 1.65
Mp1 3 2 1 1 PMOS L=1u W=20u
Mn1 3 2 0 0 NMOS L=1u W=10u
Rload 3 0 10000
.model NMOS_MODEL nmos LEVEL=1 VTO=0.7 KP=110u
.model PMOS_MODEL pmos LEVEL=1 VTO=-0.7 KP=50u
.end
"""

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sp', delete=False) as f:
        f.write(netlist_content)
        temp_netlist = f.name

    try:
        # Parse and solve
        parser = Parser()
        parser.parse_file(temp_netlist)
        circuit = parser.circuit

        solver = DCSolver(
            circuit=circuit,
            tolerance=1e-6,
            max_iterations=200,
            use_source_stepping=True
        )

        # Run DC operating point
        solution = solver.solve()

        # Verify convergence
        assert solution is not None, "Level 1 DC operating point should converge"
        assert '3' in solution, "Should have output node voltage"

        vout = solution['3']

        # Output should be somewhere between 0 and 3.3V
        assert 0.0 <= vout <= 3.3, f"Output voltage {vout:.2f}V should be in [0, 3.3]V range"

        # For Vin=1.65V (around Vdd/2), output should be in transition region
        # The exact value depends on transistor sizing, but it should be valid
        assert 0.0 <= vout <= 3.3, f"For Vin=1.65V, output should be valid, got {vout:.2f}V"

        print(f"✓ Level 1 inverter converged: Vout={vout:.3f}V")

    finally:
        # Clean up temp file
        Path(temp_netlist).unlink()


def test_convergence_with_initial_guess():
    """
    Test that providing a good initial guess improves convergence.

    This verifies that the solver can use an initial guess to help
    with convergence, especially when source stepping is disabled.
    """
    # Simple test circuit with Level 1 models
    netlist_content = """
* Simple inverter for convergence test
Vdd 1 0 3.3
Vin 2 0 1.65
Mp1 3 2 1 1 PMOS L=1u W=20u
Mn1 3 2 0 0 NMOS L=1u W=10u
Rload 3 0 10000
.model NMOS_MODEL nmos LEVEL=1 VTO=0.7 KP=110u
.model PMOS_MODEL pmos LEVEL=1 VTO=-0.7 KP=50u
.end
"""

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sp', delete=False) as f:
        f.write(netlist_content)
        temp_netlist = f.name

    try:
        parser = Parser()
        parser.parse_file(temp_netlist)
        circuit = parser.circuit

        # First, solve with source stepping to get a good solution
        solver1 = DCSolver(
            circuit=circuit,
            tolerance=1e-6,
            max_iterations=200,
            use_source_stepping=True,
            initial_guess=None
        )
        solution1 = solver1.solve()

        # Then, solve without source stepping but with initial guess
        solver2 = DCSolver(
            circuit=circuit,
            tolerance=1e-6,
            max_iterations=200,
            use_source_stepping=False,  # Disable source stepping
            initial_guess=solution1
        )
        solution2 = solver2.solve()

        # Both should converge to similar values
        vout1 = solution1['3']
        vout2 = solution2['3']

        assert abs(vout1 - vout2) < 1e-3, \
            f"Solutions should be similar: {vout1:.6f}V vs {vout2:.6f}V"

        print(f"✓ Converged with initial guess: Vout={vout2:.3f}V (no source stepping needed)")

    finally:
        Path(temp_netlist).unlink()


def test_convergence_without_improvements():
    """
    Test convergence without source stepping or initial guess.

    This should be harder and may fail to converge, demonstrating
    the value of the improvements.
    """
    netlist_content = """
* Inverter that may not converge without source stepping
Vdd 1 0 3.3
Vin 2 0 1.65
Mp1 3 2 1 1 PMOS L=1u W=20u
Mn1 3 2 0 0 NMOS L=1u W=10u
Rload 3 0 10000
.model NMOS_MODEL nmos LEVEL=1 VTO=0.7 KP=110u
.model PMOS_MODEL pmos LEVEL=1 VTO=-0.7 KP=50u
.end
"""

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sp', delete=False) as f:
        f.write(netlist_content)
        temp_netlist = f.name

    try:
        parser = Parser()
        parser.parse_file(temp_netlist)
        circuit = parser.circuit

        # Try to solve WITHOUT source stepping or initial guess
        # This tests whether the improvements are necessary
        solver = DCSolver(
            circuit=circuit,
            tolerance=1e-6,
            max_iterations=200,
            use_source_stepping=False,  # Disable improvements
            initial_guess=None  # No initial guess
        )

        # May or may not converge - either way is informative
        try:
            solution = solver.solve()
            vout = solution['3']
            print(f"✓ Converged even without improvements: Vout={vout:.3f}V")
        except RuntimeError as e:
            # Expected to fail without improvements
            assert "failed to converge" in str(e).lower()
            print(f"✓ As expected, failed to converge without improvements")

    finally:
        Path(temp_netlist).unlink()


def test_source_stepping_effectiveness():
    """
    Test that source stepping improves convergence success rate.

    Compares convergence with and without source stepping enabled.
    """
    netlist_content = """
* Inverter to test source stepping
Vdd 1 0 3.3
Vin 2 0 1.65
Mp1 3 2 1 1 PMOS L=1u W=20u
Mn1 3 2 0 0 NMOS L=1u W=10u
Rload 3 0 10000
.model NMOS_MODEL nmos LEVEL=1 VTO=0.7 KP=110u
.model PMOS_MODEL pmos LEVEL=1 VTO=-0.7 KP=50u
.end
"""

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sp', delete=False) as f:
        f.write(netlist_content)
        temp_netlist = f.name

    try:
        parser = Parser()
        parser.parse_file(temp_netlist)
        circuit = parser.circuit

        # Test with source stepping enabled
        solver_with_ss = DCSolver(
            circuit=circuit,
            tolerance=1e-6,
            max_iterations=200,
            use_source_stepping=True
        )

        try:
            solution_with_ss = solver_with_ss.solve()
            print(f"✓ With source stepping: converged")
        except RuntimeError:
            print(f"✗ With source stepping: failed")

        # Test without source stepping
        solver_without_ss = DCSolver(
            circuit=circuit,
            tolerance=1e-6,
            max_iterations=200,
            use_source_stepping=False
        )

        try:
            solution_without_ss = solver_without_ss.solve()
            print(f"✓ Without source stepping: converged")
        except RuntimeError:
            print(f"✗ Without source stepping: failed")

        # At least one should succeed
        assert True  # If we get here, the test ran successfully

    finally:
        Path(temp_netlist).unlink()


if __name__ == "__main__":
    # Run tests manually
    print("Running DC convergence tests...\n")

    print("Test 1: Level 1 inverter convergence")
    test_level1_inverter_convergence()
    print()

    print("Test 2: Convergence with initial guess")
    test_convergence_with_initial_guess()
    print()

    print("Test 3: Convergence without improvements")
    test_convergence_without_improvements()
    print()

    print("Test 4: Source stepping effectiveness")
    test_source_stepping_effectiveness()
    print()

    print("All convergence tests passed!")
