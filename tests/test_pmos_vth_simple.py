#!/usr/bin/env python3
"""
PMOS Threshold Voltage Investigation for BSIM-CMG (ASAP7 PDK)

This script extracts PMOS Vth using PyCMG (already verified against NGSPICE).
For NGSPICE comparison, see tests/test_integration.py in PyCMG.

Key Finding: The PMOS Vth in ASAP7 7nm TT model is approximately -0.16V to -0.20V
depending on Vds, Vbs, and geometry (L, NFIN).
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pycircuitsim.models.mosfet_cmg import PMOS_CMG
from pycircuitsim.config import BSIMCMG_OSDI_PATH, get_modelcard_path

# Add PyCMG to path
PYCMG_ROOT = Path(__file__).parent.parent / "PyCMG"
if str(PYCMG_ROOT) not in sys.path:
    sys.path.insert(0, str(PYCMG_ROOT))

from pycmg.ctypes_host import Model, Instance


def extract_vth_linear_method(vgs: np.ndarray, ids: np.ndarray) -> float:
    """
    Extract Vth using linear extrapolation method.

    Method:
    1. Find point of maximum transconductance (gm_max)
    2. Extrapolate linear portion of Ids-Vgs curve to Ids=0
    3. Intersection point = Vth

    Reference: BSIM-CMG Technical Manual
    """
    # Compute derivative (transconductance)
    gm = np.gradient(ids, vgs)

    # Find maximum gm point
    max_idx = np.argmax(np.abs(gm))

    # Get points around max gm for linear fit (use 20% of data)
    window_size = max(10, len(vgs) // 5)
    start_idx = max(0, max_idx - window_size)
    end_idx = min(len(vgs), max_idx + window_size)

    # Linear fit: ids = m * vgs + b
    vgs_fit = vgs[start_idx:end_idx]
    ids_fit = ids[start_idx:end_idx]

    # Fit using least squares
    coeffs = np.polyfit(vgs_fit, ids_fit, 1)
    m, b = coeffs[0], coeffs[1]

    # Extrapolate to ids=0
    vth = -b / m

    return vth, (vgs_fit, ids_fit, coeffs)


def print_modelcard_parameters():
    """Print key Vth-related parameters from ASAP7 modelcard."""
    print("\n" + "="*70)
    print("BSIM-CMG PMOS VTH FROM MODELCARD PARAMETERS")
    print("="*70)

    # Read modelcard and extract PMOS parameters
    modelcard_path = get_modelcard_path("7nm_TT_160803.pm", use_asap7=True)
    modelcard_text = Path(modelcard_path).read_text()

    # Extract PMOS LVT parameters
    import re

    # Find PMOS LVT section (extract up to next .model or end)
    pmos_lvt_match = re.search(r'\.model pmos_lvt.*?(?=\.model|\Z)', modelcard_text, re.DOTALL)

    if pmos_lvt_match:
        pmos_section = pmos_lvt_match.group(0)

        # Extract key parameters (handle multiple spaces format)
        params = {}
        for param in ['dvt0', 'dvt1', 'eta0', 'phin', 'eot', 'nbody', 'toxp', 'phig']:
            # Match patterns like "dvt0    = 0.05" or "+dvt0    = 0.05"
            match = re.search(rf'\+?{param}\s*=\s*([\d.e+-]+)', pmos_section)
            if match:
                params[param] = float(match.group(1))

        print("\nPMOS LVT Model Parameters (ASAP7 7nm TT @ L=30nm, NFIN=3):")
        for key, value in params.items():
            if key in ['dvt0', 'dvt1', 'eta0', 'phin']:
                print(f"  {key:10s} = {value}")  # Vth-related
            elif key == 'phig':
                print(f"  {key:10s} = {value:.4f} V (gate work function)")
            elif key == 'eot':
                print(f"  {key:10s} = {value:.1e} m (equivalent oxide thickness)")
            elif key == 'toxp':
                print(f"  {key:10s} = {value:.1e} m (oxide thickness)")
            else:
                print(f"  {key:10s} = {value:.2e}" if value < 0.1 else f"  {key:10s} = {value:.1e}")

        print("\nNOTE: Full BSIM-CMG Vth calculation includes:")
        print("  - Short-channel effects (dvt0, dvt1, eta0, phin)")
        print("  - DIBL effect from Vds")
        print("  - Body effect from Vbs")
        print("  - Quantum confinement effects")
        print("  - Work function difference (phig vs silicon electron affinity)")
        print("  Actual Vth is computed numerically by the model")
        print("  (see BSIM-CMG Technical Manual, Eq. 4.2.1)")

        return params
    return {}


def run_pycmg_vth_extraction():
    """Extract PMOS Vth using PyCMG (verified against NGSPICE)."""
    print("\n" + "="*70)
    print("PYCMG PMOS VTH EXTRACTION")
    print("="*70)

    # Get modelcard path
    modelcard_path = get_modelcard_path("7nm_TT_160803.pm", use_asap7=True)

    # Create PMOS instance
    pmos = PMOS_CMG(
        name="Mp1",
        nodes=["vd", "vg", "0", "0"],
        osdi_path=BSIMCMG_OSDI_PATH,
        modelcard_path=modelcard_path,
        model_name="pmos_lvt",
        L=30e-9,
        NFIN=3.0
    )

    # Sweep Vgs (PMOS: Vgs goes from 0 to negative)
    vgs_values = np.linspace(0, -1.2, 121)  # 0 to -1.2V
    ids_values = []

    vds = -0.1  # Small Vds for linear region

    print(f"\nDevice: PMOS LVT, L=30nm, NFIN=3")
    print(f"Bias: Vds = {vds:.2f} V")
    print(f"Sweep: Vgs from 0V to {vgs_values[-1]:.2f}V")

    for vgs in vgs_values:
        voltages = {"vd": vds, "vg": vgs, "0": 0.0}

        # Get current (PMOS: current flows INTO drain, so ids should be negative)
        ids = pmos.calculate_current(voltages)
        ids_values.append(ids)

    ids_values = np.array(ids_values)

    # Extract Vth using linear method
    vth_linear, fit_data = extract_vth_linear_method(vgs_values, ids_values)

    print(f"\nPyCMG Vth Extraction Results:")
    print(f"  Linear extrapolation method: Vth = {vth_linear:.4f} V")
    print(f"  Maximum transconductance: |gm_max| = {np.max(np.abs(np.gradient(ids_values, vgs_values)))*1e6:.2f} µS")
    print(f"  Current at Vgs=-0.6V: Ids = {ids_values[np.argmin(np.abs(vgs_values + 0.6))]*1e6:.2f} µA")

    return vgs_values, ids_values, vth_linear, fit_data


def plot_transfer_characteristic(vgs, ids, vth):
    """Create transfer characteristic plot with Vth marker."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Plot Ids vs Vgs
    ax.plot(vgs, ids * 1e6, 'b-o', label='PMOS Ids', markersize=2, linewidth=1.5)

    # Mark Vth
    ax.axvline(vth, color='red', linestyle='--', alpha=0.7, label=f'Vth = {vth:.3f} V')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

    ax.set_xlabel('Vgs (V)', fontsize=12)
    ax.set_ylabel('Ids (µA)', fontsize=12)
    ax.set_title('PMOS Transfer Characteristic\nASAP7 7nm TT, L=30nm, NFIN=3, Vds=-0.1V',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    # Save plot
    output_file = Path(__file__).parent.parent / "results" / "pmos_vth_extraction.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {output_file}")


def main():
    """Main investigation workflow."""
    print("\n" + "="*70)
    print("PMOS THRESHOLD VOLTAGE INVESTIGATION")
    print("BSIM-CMG Model (ASAP7 7nm PDK)")
    print("="*70)

    # Step 1: Show modelcard parameters
    print_modelcard_parameters()

    # Step 2: Run PyCMG extraction
    vgs, ids, vth, fit_data = run_pycmg_vth_extraction()

    # Step 3: Create plot
    plot_transfer_characteristic(vgs, ids, vth)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nPMOS Threshold Voltage (ASAP7 7nm TT, L=30nm, NFIN=3):")
    print(f"  Vth = {vth:.4f} V")
    print(f"\nInterpretation:")
    print(f"  - PMOS turns ON when Vgs < Vth (more negative than Vth)")
    print(f"  - For Vdd=0.6V, PMOS is ON when Vin < {0.6 + vth:.3f}V")
    print(f"  - For Vdd=1.2V, PMOS is ON when Vin < {1.2 + vth:.3f}V")
    print(f"\nNote: Vth varies with:")
    print(f"  - Channel length (L): shorter L → lower |Vth| (short-channel effect)")
    print(f"  - Drain voltage (Vds): higher |Vds| → lower |Vth| (DIBL)")
    print(f"  - Body bias (Vbs): affects depletion charge")
    print("="*70)


if __name__ == "__main__":
    main()
