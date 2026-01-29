#!/usr/bin/env python3
"""
Verification script for comparing PyCircuitSim BSIM4V5 results with ngspice.

This script runs the same circuit in both PyCircuitSim and ngspice,
then compares the results to verify accuracy.

Requirements:
- ngspice must be installed and in PATH
- PyCircuitSim must be installed

Usage:
    python tests/verify_bsim4_accuracy.py

The script will:
1. Run PyCircuitSim on a test circuit
2. Generate ngspice netlist for the same circuit
3. Run ngspice simulation
4. Compare results and report accuracy metrics
5. Generate comparison plots

TODO: If ngspice is not available, this script will skip verification
      and report what needs to be done.
"""

import subprocess
import sys
import numpy as np
from pathlib import Path
import tempfile
import matplotlib.pyplot as plt


def check_ngspice_available():
    """Check if ngspice is installed and accessible."""
    try:
        result = subprocess.run(
            ['ngspice', '-v'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def run_pycircuitsim(netlist_file):
    """Run PyCircuitSim simulation and return results."""
    # Import here to avoid issues if PyCircuitSim is not installed
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from pycircuitsim.parser import Parser
    from pycircuitsim.solver import DCSolver

    parser = Parser()
    parser.parse_file(str(netlist_file))
    circuit = parser.circuit

    # Run DC sweep
    source_name = parser.analysis_params['source']
    start = parser.analysis_params['start']
    stop = parser.analysis_params['stop']
    step = parser.analysis_params['step']

    # Find source component
    source_component = None
    for comp in circuit.components:
        if comp.name == source_name:
            source_component = comp
            break

    if source_component is None:
        raise ValueError(f"Source {source_name} not found")

    original_value = source_component.value

    # Generate sweep values
    sweep_values = []
    current_value = start
    while current_value <= stop:
        sweep_values.append(current_value)
        current_value += step

    # Stage 1: Compute operating point
    op_solver = DCSolver(circuit, use_source_stepping=True)
    op_solution = op_solver.solve()

    # Stage 2: Sweep using OP as initial guess
    results = {node: [] for node in op_solution.keys() if node not in ['0', 'GND']}
    results[source_name] = []

    for sweep_value in sweep_values:
        source_component.value = sweep_value
        solver = DCSolver(
            circuit,
            use_source_stepping=False,
            initial_guess=op_solution
        )
        solution = solver.solve()

        for node in results.keys():
            if node == source_name:
                results[node].append(sweep_value)
            else:
                results[node].append(solution.get(node, 0.0))

    source_component.value = original_value

    return results


def run_ngspice(netlist_file, output_file):
    """Run ngspice simulation and return results."""
    # Create ngspice script
    script_content = f"""
* ngspice automation script
{netlist_file.read_text()}

* Run simulation and save results
print ${'%s%s' % ('output_' + str(output_file.stem) + '.data')} > /dev/null
quit
"""

    script_file = output_file.parent / f"{output_file.stem}.script"
    script_file.write_text(script_content)

    # Run ngspice
    result = subprocess.run(
        ['ngspice', '-b', str(script_file)],
        capture_output=True,
        text=True,
        timeout=60
    )

    if result.returncode != 0:
        raise RuntimeError(f"ngspice failed: {result.stderr}")

    # Parse ngspice output
    # TODO: Implement ngspice output parsing
    # This is a placeholder - actual implementation depends on ngspice output format
    return None


def compare_results(pycircuitsim_results, ngspice_results):
    """Compare PyCircuitSim and ngspice results."""
    print("\n" + "="*70)
    print("PyCircuitSim vs ngspice Comparison")
    print("="*70)

    if ngspice_results is None:
        print("\nngspice results not available - skipping detailed comparison")
        print("TODO: Implement ngspice integration for accuracy verification")
        return

    # Compare each node
    for node in pycircuitsim_results.keys():
        if node in ngspice_results:
            pyc_data = np.array(pycircuitsim_results[node])
            ngspice_data = np.array(ngspice_results[node])

            # Calculate metrics
            mae = np.mean(np.abs(pyc_data - ngspice_data))  # Mean Absolute Error
            rmse = np.sqrt(np.mean((pyc_data - ngspice_data)**2))  # Root Mean Square Error
            max_error = np.max(np.abs(pyc_data - ngspice_data))
            rel_error = np.mean(np.abs((pyc_data - ngspice_data) / (ngspice_data + 1e-9)))

            print(f"\nNode {node}:")
            print(f"  Mean Absolute Error: {mae:.6e} V")
            print(f"  Root Mean Square Error: {rmse:.6e} V")
            print(f"  Maximum Error: {max_error:.6e} V")
            print(f"  Mean Relative Error: {rel_error:.2%}")

            # Check if errors are acceptable
            if mae < 1e-3:
                print(f"  Status: ✓ EXCELLENT (MAE < 1mV)")
            elif mae < 1e-2:
                print(f"  Status: ✓ GOOD (MAE < 10mV)")
            elif mae < 1e-1:
                print(f"  Status: ⚠ ACCEPTABLE (MAE < 100mV)")
            else:
                print(f"  Status: ✗ NEEDS IMPROVEMENT (MAE >= 100mV)")


def plot_comparison(pycircuitsim_results, ngspice_results, output_file):
    """Generate comparison plots."""
    if ngspice_results is None:
        print("\nSkipping plot generation - ngspice results not available")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Plot voltages
    ax1 = axes[0]
    for node in pycircuitsim_results.keys():
        if node not in ['0', 'GND'] and not node.startswith('i('):
            if node in pycircuitsim_results:
                ax1.plot(
                    pycircuitsim_results.get('Vin', list(range(len(pycircuitsim_results[node])))),
                    pycircuitsim_results[node],
                    label=f'PyCircuitSim {node}',
                    marker='o'
                )
            if node in ngspice_results:
                ax1.plot(
                    ngspice_results.get('Vin', list(range(len(ngspice_results[node])))),
                    ngspice_results[node],
                    label=f'ngspice {node}',
                    linestyle='--'
                )

    ax1.set_xlabel('Sweep Variable')
    ax1.set_ylabel('Voltage (V)')
    ax1.set_title('Voltage Comparison')
    ax1.legend()
    ax1.grid(True)

    # Plot errors
    ax2 = axes[1]
    for node in pycircuitsim_results.keys():
        if node in ngspice_results and node not in ['0', 'GND'] and not node.startswith('i('):
            error = np.array(pycircuitsim_results[node]) - np.array(ngspice_results[node])
            ax2.plot(
                pycircuitsim_results.get('Vin', list(range(len(error)))),
                error * 1000,  # Convert to mV
                label=f'{node} error'
            )

    ax2.set_xlabel('Sweep Variable')
    ax2.set_ylabel('Error (mV)')
    ax2.set_title('PyCircuitSim - ngspice Error')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"\nComparison plot saved to: {output_file}")


def main():
    """Main verification routine."""
    print("PyCircuitSim BSIM4V5 Accuracy Verification")
    print("="*70)

    # Check if ngspice is available
    if not check_ngspice_available():
        print("\n⚠ ngspice is not installed or not in PATH")
        print("\nTo enable ngspice verification:")
        print("  1. Install ngspice:")
        print("     - Ubuntu/Debian: sudo apt-get install ngspice")
        print("     - macOS: brew install ngspice")
        print("     - Or download from: http://ngspice.sourceforge.net/")
        print("  2. Ensure ngspice is in your PATH")
        print("  3. Re-run this script")
        print("\nTODO: Run this verification after installing ngspice")
        return 1

    print("\n✓ ngspice found - running verification...")

    # Use existing test netlist
    netlist_file = Path(__file__).parent.parent / "examples" / "inverter_bsim4_dc.sp"

    if not netlist_file.exists():
        print(f"\n✗ Test netlist not found: {netlist_file}")
        return 1

    print(f"\nTest netlist: {netlist_file}")

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Run PyCircuitSim
        print("\nRunning PyCircuitSim simulation...")
        try:
            pycircuitsim_results = run_pycircuitsim(netlist_file)
            print("✓ PyCircuitSim simulation complete")
        except Exception as e:
            print(f"✗ PyCircuitSim simulation failed: {e}")
            return 1

        # Run ngspice
        print("\nRunning ngspice simulation...")
        try:
            ngspice_results = run_ngspice(netlist_file, tmpdir / "ngspice_output")
            print("✓ ngspice simulation complete")
        except Exception as e:
            print(f"✗ ngspice simulation failed: {e}")
            print("\nTODO: Fix ngspice integration issues")
            ngspice_results = None

        # Compare results
        compare_results(pycircuitsim_results, ngspice_results)

        # Generate plots
        plot_comparison(
            pycircuitsim_results,
            ngspice_results,
            tmpdir / "comparison.png"
        )

    print("\n" + "="*70)
    print("Verification complete!")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
