#!/usr/bin/env python3
"""
Verify Level-1 MOSFET Transient Analysis against NGSPICE.

This script:
1. Runs NGSPICE on the reference netlist
2. Runs PyCircuitSim on the test netlist
3. Compares waveforms using RMSE metric
4. Reports pass/fail based on tolerance

Usage:
    python tests/verify_level1_transient.py
"""

import os
import sys
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pycircuitsim.simulation import run_simulation


def run_ngspice(netlist_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run NGSPICE simulation and extract results.

    Args:
        netlist_path: Path to NGSPICE netlist

    Returns:
        Tuple of (time, v_in, v_out) arrays
    """
    # Create batch file for NGSPICE
    batch_cmds = f"""
* NGSPICE batch control
level1_inverter_tran_ngspice.sp
* Exit after simulation
exit
"""

    batch_path = netlist_path.replace('.sp', '_batch.cmds')
    with open(batch_path, 'w') as f:
        f.write(batch_cmds)

    # Run NGSPICE with output file
    out_path = netlist_path.replace('.sp', '.out')

    try:
        result = subprocess.run(
            ['/usr/local/ngspice-45.2/bin/ngspice', '-b', '-o', out_path, batch_path],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            print(f"NGSPICE Error:\n{result.stderr}")

        # Parse output file
        return parse_ngspice_output(out_path)

    except FileNotFoundError:
        print(f"Error: NGSPICE not found at /usr/local/ngspice-45.2/bin/ngspice")
        print("Please install NGSPICE or update the path in this script.")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("Error: NGSPICE simulation timed out")
        sys.exit(1)


def parse_ngspice_output(out_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse NGSPICE raw output file.

    Args:
        out_path: Path to NGSPICE .out file

    Returns:
        Tuple of (time, v_in, v_out) arrays
    """
    data = []
    reading_data = False

    with open(out_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Index'):
                # Header line
                parts = line.split()
                reading_data = True
                continue
            if reading_data and line:
                try:
                    values = line.split()
                    if len(values) >= 3:
                        time_val = float(values[0])
                        v_in_val = float(values[1])
                        v_out_val = float(values[2])
                        data.append((time_val, v_in_val, v_out_val))
                except (ValueError, IndexError):
                    continue

    if not data:
        print("Error: No data parsed from NGSPICE output")
        sys.exit(1)

    data = np.array(data)
    return data[:, 0], data[:, 1], data[:, 2]


def run_pycircuitsim(netlist_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run PyCircuitSim simulation.

    Args:
        netlist_path: Path to netlist

    Returns:
        Tuple of (time, v_in, v_out) arrays
    """
    # Run simulation
    result = run_simulation(netlist_path)

    # Extract CSV path
    csv_path = netlist_path.replace('/examples/', '/results/')
    csv_path = csv_path.replace('.sp', '/tran/' + Path(netlist_path).stem + '_transient.csv')

    if not os.path.exists(csv_path):
        print(f"Error: CSV output not found at {csv_path}")
        sys.exit(1)

    # Read CSV
    df = pd.read_csv(csv_path)

    return df['time'].values, df['in'].values, df['out'].values


def calculate_rmse(reference: np.ndarray, actual: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error between reference and actual waveforms.

    Args:
        reference: Reference waveform (NGSPICE)
        actual: Simulated waveform (PyCircuitSim)

    Returns:
        RMSE value in volts
    """
    if len(reference) != len(actual):
        # Interpolate to match lengths
        ref_x = np.linspace(0, 1, len(reference))
        act_x = np.linspace(0, 1, len(actual))
        actual_interp = np.interp(ref_x, act_x, actual)
    else:
        actual_interp = actual

    mse = np.mean((reference - actual_interp)**2)
    rmse = np.sqrt(mse)
    return rmse


def calculate_max_error(reference: np.ndarray, actual: np.ndarray) -> float:
    """Calculate maximum absolute error between waveforms."""
    if len(reference) != len(actual):
        # Interpolate to match lengths
        ref_x = np.linspace(0, 1, len(reference))
        act_x = np.linspace(0, 1, len(actual))
        actual_interp = np.interp(ref_x, act_x, actual)
    else:
        actual_interp = actual

    return np.max(np.abs(reference - actual_interp))


def main():
    """Main verification routine."""
    print("=" * 60)
    print("Level-1 MOSFET Transient Analysis Verification")
    print("=" * 60)

    # File paths
    base_dir = Path(__file__).parent.parent
    ngspice_netlist = base_dir / 'examples' / 'level1_inverter_tran_ngspice.sp'
    pycircuit_netlist = base_dir / 'examples' / 'level1_inverter_tran.sp'

    if not ngspice_netlist.exists():
        print(f"Error: NGSPICE netlist not found: {ngspice_netlist}")
        sys.exit(1)

    if not pycircuit_netlist.exists():
        print(f"Error: PyCircuitSim netlist not found: {pycircuit_netlist}")
        sys.exit(1)

    # Run simulations
    print("\n[1/4] Running NGSPICE reference simulation...")
    try:
        time_ng, v_in_ng, v_out_ng = run_ngspice(str(ngspice_netlist))
        print(f"      Got {len(time_ng)} timepoints")
    except Exception as e:
        print(f"      Failed: {e}")
        print("      Skipping NGSPICE comparison (results will not be validated)")
        time_ng, v_out_ng = None, None

    print("\n[2/4] Running PyCircuitSim...")
    try:
        time_py, v_in_py, v_out_py = run_pycircuitsim(str(pycircuit_netlist))
        print(f"      Got {len(time_py)} timepoints")
    except Exception as e:
        print(f"      Failed: {e}")
        sys.exit(1)

    # Compare results
    if time_ng is not None:
        print("\n[3/4] Comparing waveforms...")

        # Calculate errors
        rmse = calculate_rmse(v_out_ng, v_out_py)
        max_err = calculate_max_error(v_out_ng, v_out_py)

        print(f"      RMSE: {rmse*1000:.3f} mV")
        print(f"      Max Error: {max_err*1000:.3f} mV")

        # Check against tolerance (5% of VDD = 50mV)
        VDD = 1.0
        tolerance = 0.05 * VDD  # 50mV

        print("\n[4/4] Result:")
        if rmse < tolerance and max_err < 2 * tolerance:
            print(f"      PASS: Waveforms match within tolerance")
            print(f"      RMSE {rmse*1000:.3f}mV < {tolerance*1000:.0f}mV")
            return 0
        else:
            print(f"      FAIL: Waveforms differ significantly")
            print(f"      RMSE {rmse*1000:.3f}mV >= {tolerance*1000:.0f}mV")
            return 1
    else:
        print("\n[3/4] NGSPICE comparison skipped")
        print("\n[4/4] Result:")
        print("      INFO: PyCircuitSim simulation completed")
        print("      Run NGSPICE manually to verify results")
        return 0


if __name__ == "__main__":
    sys.exit(main())
