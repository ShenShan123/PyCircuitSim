#!/usr/bin/env python3
"""
Compare PyCircuitSim BSIM4V5 results with ngspice reference
"""
import subprocess
import re
import numpy as np
import pandas as pd
from pathlib import Path

def run_ngspice(netlist_path):
    """Run ngspice and parse output"""
    result = subprocess.run(
        ['ngspice', '-b', str(netlist_path)],
        capture_output=True,
        text=True
    )

    output = result.stdout + result.stderr

    # Parse DC transfer data
    data = []
    in_dc_section = False

    for line in output.split('\n'):
        if 'DC transfer characteristic' in line:
            in_dc_section = True
            continue
        if in_dc_section:
            if line.strip() == '' or 'Total analysis' in line:
                break
            # Match data lines: index v-sweep v(...) current
            match = re.match(r'^\s*(\d+)\s+([+-]?\d+\.?\d*[eE]?[+-]?\d*)\s+([+-]?\d+\.?\d*[eE]?[+-]?\d*)\s+([+-]?\d+\.?\d*[eE]?[+-]?\d*)', line)
            if match:
                idx, vds, v_node, current = match.groups()
                data.append({
                    'Vds': float(vds),
                    'Id_ngspice': float(current)
                })

    return pd.DataFrame(data)

def run_pycircuitsim(netlist_path):
    """Run PyCircuitSim and parse output"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from main import simulate
    from pycircuitsim.parser import NetlistParser
    from pycircuitsim.circuit import Circuit
    from pycircuitsim.solver import DCSolver

    parser = NetlistParser()
    parser.parse_file(netlist_path)

    circuit = Circuit(parser.components, parser.nodes)
    solver = DCSolver(circuit)

    # Get the sweep parameters from parser
    if parser.dc_sweep:
        source_name = parser.dc_sweep['source']
        start = parser.dc_sweep['start']
        stop = parser.dc_sweep['stop']
        step = parser.dc_sweep['step']

        # For single transistor test, find the Vds source
        vds_sources = [c for c in parser.components if c.name.lower() == 'vds']
        if not vds_sources:
            return None

        # Run sweep manually
        results = {'Vds': [], 'Id_pycircuitsim': []}
        num_steps = int((stop - start) / step) + 1

        for i in range(num_steps):
            vds_val = start + i * step

            # Set source value
            for comp in parser.components:
                if comp.name.lower() == 'vds':
                    comp.value = vds_val

            # Solve
            circuit = Circuit(parser.components, parser.nodes)
            solver = DCSolver(circuit)
            solution = solver.solve_dc()

            # Get drain current
            for comp in parser.components:
                if comp.name.lower() == 'vds':
                    # Get current through voltage source
                    if hasattr(comp, 'current'):
                        results['Vds'].append(vds_val)
                        results['Id_pycircuitsim'].append(comp.current)
                    break

        return pd.DataFrame(results)

    return None

def compare_results():
    """Main comparison function"""
    print("=" * 70)
    print("PyCircuitSim vs ngspice BSIM4V5 Validation")
    print("=" * 70)

    # Test NMOS
    print("\n### NMOS Test (L=45nm, W=90nm, Vgs=0.5V) ###")

    # Since PyCircuitSim uses simplified parameters, we'll use standalone C tests
    # instead of full circuit simulation for fair comparison

    # Run ngspice on test_nmos_ngspice.sp
    ngspice_data = run_ngspice('examples/test_nmos_ngspice.sp')

    if ngspice_data is not None and len(ngspice_data) > 0:
        print(f"\nngspice results (Vds, Id):")
        print(f"  At Vds=0.1V: Id = {ngspice_data.iloc[5]['Id_ngspice']*1e6:.2f} µA")
        print(f"  At Vds=0.5V: Id = {ngspice_data.iloc[25]['Id_ngspice']*1e6:.2f} µA")

        # Run PyCircuitSim standalone test
        import subprocess
        result = subprocess.run(
            ['./pycircuitsim/models/bsim4v5/bridge/test_nmos_simple'],
            capture_output=True,
            text=True,
            cwd='/home/shenshan/NN_SPICE'
        )

        print(f"\nPyCircuitSim C model results:")
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'Id=' in line or 'Gm=' in line:
                    print(f"  {line.strip()}")

        print("\n--- Comparison Summary ---")
        print("Note: PyCircuitSim uses simplified BSIM4V5 parameters")
        print("      ngspice uses full freePDK45 model library")
        print("      Expect some differences in absolute values")

    return

if __name__ == '__main__':
    compare_results()
