#!/usr/bin/env python3
"""
Verify BSIM-CMG DC sweep analysis: PyCircuitSim vs NGSPICE.

Tasks 9-11:
  Task 9:  NMOS Id-Vgs sweep (Vds=0.5V, Vgs: 0->0.7V, L=30n, NFIN=10)
  Task 10: PMOS with Rload (Vdd=0.7V, Vg: 0->0.7V, Rload=1k, L=30n, NFIN=10)
  Task 11: Inverter VTC  (Vdd=0.7V, Vin: 0->0.7V, L=30n, NFIN=10)

Acceptance criteria:
  - RMSE normalized to max value < 1%
  - Max relative error < 5%

Key NGSPICE OSDI notes:
  - OSDI devices use N prefix (NOT M).
  - Instance params (L, NFIN) must be BAKED into the modelcard.
  - DEVTYPE=1 for NMOS, DEVTYPE=0 for PMOS (ASAP7 modelcard lacks it).
  - wrdata format: first col = sweep variable, then value columns.
"""
from __future__ import annotations

import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

# -- Project paths -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "PyCMG"))

OSDI_PATH = PROJECT_ROOT / "external_compact_models" / "PyCMG" / "build-deep-verify" / "osdi" / "bsimcmg.osdi"
MODELCARD_PATH = PROJECT_ROOT / "external_compact_models" / "PyCMG" / "tech_model_cards" / "ASAP7" / "7nm_TT_160803.pm"
NGSPICE_BIN = "/usr/local/ngspice-45.2/bin/ngspice"
RESULTS_DIR = PROJECT_ROOT / "tests" / "verify_bsimcmg_dc_results"

# -- PyCMG baking utility ----------------------------------------------------
from pycmg.testing import bake_inst_params

# -- PyCircuitSim imports ----------------------------------------------------
from pycircuitsim.parser import Parser
from pycircuitsim.simulation import run_dc_sweep
from pycircuitsim.visualizer import Visualizer

# -- Test parameters ---------------------------------------------------------
L = 30e-9
NFIN = 10
VDD = 0.7
VGS_START = 0.0
VGS_STOP = 0.7
VGS_STEP = 0.01

NMOS_INST_PARAMS: Dict[str, Any] = {"L": L, "NFIN": float(NFIN), "DEVTYPE": 1}
PMOS_INST_PARAMS: Dict[str, Any] = {"L": L, "NFIN": float(NFIN), "DEVTYPE": 0}

# Acceptance criteria
RMSE_THRESHOLD = 0.01   # 1% of max value
MAX_REL_ERR_THRESHOLD = 0.05  # 5%


# ============================================================================
# Utility functions
# ============================================================================

def run_ngspice_dc_sweep(
    netlist_path: Path,
    runner_path: Path,
    csv_path: Path,
    log_path: Path,
    signals: List[str],
) -> Dict[str, np.ndarray]:
    """Run NGSPICE DC sweep and parse wrdata output.

    Args:
        netlist_path: Path to the NGSPICE netlist (.cir)
        runner_path: Path to the NGSPICE runner script
        csv_path: Path where wrdata will write output
        log_path: Path for NGSPICE log
        signals: List of signal names to extract (e.g., ["i(Vds)", "v(drain)"])

    Returns:
        Dict with 'sweep' key for sweep values and signal names for values.
    """
    # Build runner script
    signal_str = " ".join(signals)
    runner_content = (
        f"* ngspice DC sweep runner\n"
        f".control\n"
        f"osdi {OSDI_PATH}\n"
        f"source {netlist_path}\n"
        f"set filetype=ascii\n"
        f"set wr_vecnames\n"
        f"run\n"
        f"wrdata {csv_path} {signal_str}\n"
        f".endc\n"
        f".end\n"
    )
    runner_path.write_text(runner_content)

    # Run NGSPICE
    res = subprocess.run(
        [NGSPICE_BIN, "-b", "-o", str(log_path), str(runner_path)],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        log_content = log_path.read_text() if log_path.exists() else "(no log)"
        raise RuntimeError(
            f"NGSPICE failed (rc={res.returncode}):\n"
            f"  stderr: {res.stderr[:500]}\n"
            f"  log (last 500): ...{log_content[-500:]}\n"
        )
    if not csv_path.exists():
        log_content = log_path.read_text() if log_path.exists() else "(no log)"
        raise RuntimeError(
            f"NGSPICE produced no output: {csv_path}\n"
            f"  log (last 500): ...{log_content[-500:]}\n"
        )

    # Parse wrdata output
    # Format: header line, then data rows.
    # For 1 signal: col0=sweep_val, col1=signal_val
    # For N signals: col0=sweep, col1=sig1, col2=sweep, col3=sig2, ...
    with csv_path.open() as f:
        lines = f.readlines()

    if not lines:
        raise RuntimeError(f"Empty NGSPICE output: {csv_path}")

    # Data lines (skip header)
    data_rows = []
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        vals = [float(x) for x in stripped.split()]
        data_rows.append(vals)

    data = np.array(data_rows)

    result: Dict[str, np.ndarray] = {}
    if len(signals) == 1:
        result["sweep"] = data[:, 0]
        result[signals[0]] = data[:, 1]
    else:
        # Each signal has sweep + value pair
        for i, sig in enumerate(signals):
            col_base = i * 2
            result[sig] = data[:, col_base + 1]
        result["sweep"] = data[:, 0]

    return result


def run_pycircuitsim_dc_sweep(
    netlist_content: str,
    sweep_node: str,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Run PyCircuitSim DC sweep and return (sweep_values, results).

    Args:
        netlist_content: Netlist as a string.
        sweep_node: Node name whose voltage represents the sweep variable.

    Returns:
        Tuple of (sweep_array, results_dict).
        sweep_array: 1D array of sweep values (from the swept node voltages).
        results_dict: Dict mapping result keys to 1D numpy arrays.
    """
    tmpdir = tempfile.mkdtemp(prefix="pycircuitsim_dc_")
    try:
        netlist_path = Path(tmpdir) / "circuit.sp"
        netlist_path.write_text(netlist_content)

        parser = Parser()
        parser.parse_file(str(netlist_path))
        circuit = parser.circuit

        vis = Visualizer()
        out_path = Path(tmpdir) / "out"
        out_path.mkdir()

        results = run_dc_sweep(
            circuit, parser.analysis_params, vis, out_path, "pysim"
        )

        # Extract actual sweep values from the swept node
        sweep_vals = np.array(results[sweep_node])

        # Convert all result lists to numpy arrays
        np_results: Dict[str, np.ndarray] = {}
        for key, vals in results.items():
            np_results[key] = np.array(vals)

        return sweep_vals, np_results
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def compute_metrics(
    ng_sweep: np.ndarray,
    ng_values: np.ndarray,
    py_sweep: np.ndarray,
    py_values: np.ndarray,
) -> Dict[str, float]:
    """Compute comparison metrics between NGSPICE and PyCircuitSim curves.

    Interpolates PyCircuitSim data onto NGSPICE sweep points for fair comparison.
    Only compares over the overlapping sweep range.

    Returns:
        Dict with keys: rmse, nrmse, max_abs_err, max_rel_err
    """
    # Determine common sweep range
    common_start = max(ng_sweep[0], py_sweep[0])
    common_stop = min(ng_sweep[-1], py_sweep[-1])

    # Filter NGSPICE to common range
    mask = (ng_sweep >= common_start - 1e-10) & (ng_sweep <= common_stop + 1e-10)
    ng_sweep_common = ng_sweep[mask]
    ng_values_common = ng_values[mask]

    # Interpolate PyCircuitSim onto NGSPICE common points
    py_interp = np.interp(ng_sweep_common, py_sweep, py_values)

    diff = np.abs(py_interp - ng_values_common)
    max_val = np.max(np.abs(ng_values_common))

    rmse = np.sqrt(np.mean(diff ** 2))
    nrmse = rmse / max_val if max_val > 0 else float("inf")
    max_abs_err = np.max(diff)

    # Relative error (avoid division by zero)
    denom = np.maximum(np.abs(ng_values_common), 1e-12)
    rel_errs = diff / denom
    # Only consider points where NGSPICE value is significant (> 1% of max)
    significant = np.abs(ng_values_common) > 0.01 * max_val
    if np.any(significant):
        max_rel_err = np.max(rel_errs[significant])
    else:
        max_rel_err = 0.0

    return {
        "rmse": rmse,
        "nrmse": nrmse,
        "max_abs_err": max_abs_err,
        "max_rel_err": max_rel_err,
        "n_common_points": len(ng_sweep_common),
    }


def plot_comparison(
    ng_sweep: np.ndarray,
    ng_values: np.ndarray,
    py_sweep: np.ndarray,
    py_values: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
    metrics: Dict[str, float],
    log_scale: bool = False,
) -> None:
    """Plot NGSPICE vs PyCircuitSim comparison with metrics annotation."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Top: overlay
    ax1 = axes[0]
    ax1.plot(ng_sweep, ng_values, "b-", linewidth=2, label="NGSPICE (reference)")
    ax1.plot(py_sweep, py_values, "r--", linewidth=1.5, label="PyCircuitSim")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    if log_scale and np.any(ng_values > 0):
        ax1.set_yscale("log")
        pos = ng_values[ng_values > 0]
        ax1.set_ylim(bottom=max(1e-12, np.min(pos) * 0.1))

    # Annotation box
    textstr = (
        f"NRMSE: {metrics['nrmse']*100:.4f}%\n"
        f"Max |err|: {metrics['max_abs_err']:.4e}\n"
        f"Max rel err: {metrics['max_rel_err']*100:.2f}%"
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=9,
             verticalalignment="top", bbox=props)

    # Bottom: error
    ax2 = axes[1]
    # Compute error at common points
    common_start = max(ng_sweep[0], py_sweep[0])
    common_stop = min(ng_sweep[-1], py_sweep[-1])
    mask = (ng_sweep >= common_start - 1e-10) & (ng_sweep <= common_stop + 1e-10)
    ng_common = ng_sweep[mask]
    ng_common_vals = ng_values[mask]
    py_interp = np.interp(ng_common, py_sweep, py_values)
    error = py_interp - ng_common_vals
    ax2.plot(ng_common, error, "g-", linewidth=1)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("Error (PySim - NGSPICE)")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="k", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()


# ============================================================================
# Task 9: NMOS Id-Vgs DC Sweep
# ============================================================================

def test_nmos_dc_sweep() -> bool:
    """Verify NMOS Id-Vgs: sweep Vgs 0->0.7V, Vds=0.5V, L=30n, NFIN=10."""
    print("=" * 70)
    print("Task 9: NMOS Id-Vgs DC Sweep Verification")
    print("=" * 70)
    print(f"  L = {L*1e9:.0f} nm, NFIN = {NFIN}, Vds = 0.5 V")
    print(f"  Vgs sweep: {VGS_START} -> {VGS_STOP} V, step = {VGS_STEP} V")
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # -- NGSPICE reference ---------------------------------------------------
    print("  [1/3] Running NGSPICE reference...")
    ng_mc = RESULTS_DIR / "nmos_rvt_baked.lib"
    bake_inst_params(MODELCARD_PATH, ng_mc, "nmos_rvt", NMOS_INST_PARAMS)

    ng_net = RESULTS_DIR / "ngspice_nmos_dc_sweep.cir"
    ng_net.write_text(
        f"* NMOS Id-Vgs DC Sweep\n"
        f'.include "{ng_mc}"\n'
        f".temp 27\n"
        f"Vds d 0 0.5\n"
        f"Vgs g 0 0.0\n"
        f"N1 d g 0 0 nmos_rvt\n"
        f".dc Vgs {VGS_START} {VGS_STOP} {VGS_STEP}\n"
        f".end\n"
    )

    ng_data = run_ngspice_dc_sweep(
        ng_net,
        RESULTS_DIR / "ngspice_nmos_dc_sweep_runner.cir",
        RESULTS_DIR / "ngspice_nmos_dc_sweep.csv",
        RESULTS_DIR / "ngspice_nmos_dc_sweep.log",
        ["i(Vds)"],
    )
    ng_sweep = ng_data["sweep"]
    ng_id = np.abs(ng_data["i(Vds)"])  # NGSPICE: negative = current out of drain
    print(f"    NGSPICE: {len(ng_sweep)} points, Id range: [{ng_id.min():.4e}, {ng_id.max():.4e}] A")

    # -- PyCircuitSim --------------------------------------------------------
    print("  [2/3] Running PyCircuitSim...")
    # Netlist: node 1=drain, node 2=gate
    py_netlist = (
        f"* BSIM-CMG NMOS DC sweep\n"
        f"Vds 1 0 0.5\n"
        f"Vgs 2 0 0.0\n"
        f"Mn1 1 2 0 0 nmos1 L=30n NFIN=10\n"
        f".model nmos1 NMOS (LEVEL=72)\n"
        f".dc Vgs {VGS_START} {VGS_STOP} {VGS_STEP}\n"
        f".end\n"
    )
    py_sweep, py_results = run_pycircuitsim_dc_sweep(py_netlist, sweep_node="2")
    py_id = np.abs(py_results["i(Mn1)"])
    print(f"    PyCircuitSim: {len(py_sweep)} points, Id range: [{py_id.min():.4e}, {py_id.max():.4e}] A")

    # -- Compare -------------------------------------------------------------
    print("  [3/3] Computing metrics...")
    metrics = compute_metrics(ng_sweep, ng_id, py_sweep, py_id)

    print(f"    Common points = {metrics['n_common_points']}")
    print(f"    RMSE        = {metrics['rmse']:.6e} A")
    print(f"    NRMSE       = {metrics['nrmse']*100:.4f}%")
    print(f"    Max |error| = {metrics['max_abs_err']:.6e} A")
    print(f"    Max rel err = {metrics['max_rel_err']*100:.4f}%")

    # Plot
    plot_comparison(
        ng_sweep, ng_id, py_sweep, py_id,
        xlabel="Vgs (V)", ylabel="|Id| (A)",
        title="Task 9: NMOS Id-Vgs (Vds=0.5V, L=30nm, NFIN=10)",
        output_path=RESULTS_DIR / "nmos_id_vgs_comparison.png",
        metrics=metrics,
    )
    # Also plot log scale
    plot_comparison(
        ng_sweep, ng_id, py_sweep, py_id,
        xlabel="Vgs (V)", ylabel="|Id| (A)",
        title="Task 9: NMOS Id-Vgs (log scale)",
        output_path=RESULTS_DIR / "nmos_id_vgs_comparison_log.png",
        metrics=metrics,
        log_scale=True,
    )
    print(f"    Plots saved to: {RESULTS_DIR}/nmos_id_vgs_comparison*.png")

    # Pass/Fail
    nrmse_pass = metrics["nrmse"] < RMSE_THRESHOLD
    maxrel_pass = metrics["max_rel_err"] < MAX_REL_ERR_THRESHOLD
    overall = nrmse_pass and maxrel_pass

    print()
    print(f"    NRMSE  < {RMSE_THRESHOLD*100}%: {'PASS' if nrmse_pass else 'FAIL'} ({metrics['nrmse']*100:.4f}%)")
    print(f"    MaxRel < {MAX_REL_ERR_THRESHOLD*100}%: {'PASS' if maxrel_pass else 'FAIL'} ({metrics['max_rel_err']*100:.4f}%)")
    print(f"    Result: {'PASS' if overall else 'FAIL'}")
    print()

    return overall


# ============================================================================
# Task 10: PMOS DC Sweep (Resistor-loaded)
# ============================================================================

def test_pmos_dc_sweep() -> bool:
    """Verify PMOS V(drain) vs Vgate: Vdd=0.7V, Rload=1k, L=30n, NFIN=10."""
    print("=" * 70)
    print("Task 10: PMOS DC Sweep (Resistor-loaded) Verification")
    print("=" * 70)
    print(f"  L = {L*1e9:.0f} nm, NFIN = {NFIN}, Vdd = {VDD} V, Rload = 1k")
    print(f"  Vgate sweep: {VGS_START} -> {VGS_STOP} V, step = {VGS_STEP} V")
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # -- NGSPICE reference ---------------------------------------------------
    print("  [1/3] Running NGSPICE reference...")
    ng_mc = RESULTS_DIR / "pmos_rvt_baked.lib"
    bake_inst_params(MODELCARD_PATH, ng_mc, "pmos_rvt", PMOS_INST_PARAMS)

    ng_net = RESULTS_DIR / "ngspice_pmos_dc_sweep.cir"
    ng_net.write_text(
        f"* PMOS DC Sweep with Rload\n"
        f'.include "{ng_mc}"\n'
        f".temp 27\n"
        f"Vdd vdd 0 {VDD}\n"
        f"Vg g 0 0.0\n"
        f"N1 drain g vdd vdd pmos_rvt\n"
        f"Rload drain 0 1k\n"
        f".dc Vg {VGS_START} {VGS_STOP} {VGS_STEP}\n"
        f".end\n"
    )

    ng_data = run_ngspice_dc_sweep(
        ng_net,
        RESULTS_DIR / "ngspice_pmos_dc_sweep_runner.cir",
        RESULTS_DIR / "ngspice_pmos_dc_sweep.csv",
        RESULTS_DIR / "ngspice_pmos_dc_sweep.log",
        ["v(drain)"],
    )
    ng_sweep = ng_data["sweep"]
    ng_vdrain = ng_data["v(drain)"]
    print(f"    NGSPICE: {len(ng_sweep)} points, V(drain) range: [{ng_vdrain.min():.6f}, {ng_vdrain.max():.6f}] V")

    # -- PyCircuitSim --------------------------------------------------------
    print("  [2/3] Running PyCircuitSim...")
    # PyCircuitSim netlist: node 1=Vdd, node 2=gate, node 3=drain
    py_netlist = (
        f"* PMOS DC Sweep with Rload\n"
        f"Vdd 1 0 {VDD}\n"
        f"Vg 2 0 0.0\n"
        f"Mp1 3 2 1 1 pmos1 L=30n NFIN=10\n"
        f"Rload 3 0 1k\n"
        f".model pmos1 PMOS (LEVEL=72)\n"
        f".dc Vg {VGS_START} {VGS_STOP} {VGS_STEP}\n"
        f".end\n"
    )
    py_sweep, py_results = run_pycircuitsim_dc_sweep(py_netlist, sweep_node="2")
    py_vdrain = py_results["3"]  # node 3 = drain
    print(f"    PyCircuitSim: {len(py_sweep)} points, V(drain) range: [{py_vdrain.min():.6f}, {py_vdrain.max():.6f}] V")

    # -- Compare -------------------------------------------------------------
    print("  [3/3] Computing metrics...")
    metrics = compute_metrics(ng_sweep, ng_vdrain, py_sweep, py_vdrain)

    print(f"    Common points = {metrics['n_common_points']}")
    print(f"    RMSE        = {metrics['rmse']:.6e} V")
    print(f"    NRMSE       = {metrics['nrmse']*100:.4f}%")
    print(f"    Max |error| = {metrics['max_abs_err']:.6e} V")
    print(f"    Max rel err = {metrics['max_rel_err']*100:.4f}%")

    # Plot
    plot_comparison(
        ng_sweep, ng_vdrain, py_sweep, py_vdrain,
        xlabel="Vgate (V)", ylabel="V(drain) (V)",
        title="Task 10: PMOS V(drain) vs Vgate (Vdd=0.7V, Rload=1k)",
        output_path=RESULTS_DIR / "pmos_vdrain_comparison.png",
        metrics=metrics,
    )
    print(f"    Plot saved to: {RESULTS_DIR}/pmos_vdrain_comparison.png")

    # Pass/Fail
    nrmse_pass = metrics["nrmse"] < RMSE_THRESHOLD
    maxrel_pass = metrics["max_rel_err"] < MAX_REL_ERR_THRESHOLD
    overall = nrmse_pass and maxrel_pass

    print()
    print(f"    NRMSE  < {RMSE_THRESHOLD*100}%: {'PASS' if nrmse_pass else 'FAIL'} ({metrics['nrmse']*100:.4f}%)")
    print(f"    MaxRel < {MAX_REL_ERR_THRESHOLD*100}%: {'PASS' if maxrel_pass else 'FAIL'} ({metrics['max_rel_err']*100:.4f}%)")
    print(f"    Result: {'PASS' if overall else 'FAIL'}")
    print()

    return overall


# ============================================================================
# Task 11: Inverter VTC (Vout vs Vin)
# ============================================================================

def test_inverter_vtc() -> bool:
    """Verify Inverter VTC: Vdd=0.7V, Vin sweep 0->0.7V, L=30n, NFIN=10."""
    print("=" * 70)
    print("Task 11: Inverter VTC Verification")
    print("=" * 70)
    print(f"  L = {L*1e9:.0f} nm, NFIN = {NFIN}, Vdd = {VDD} V")
    print(f"  Vin sweep: {VGS_START} -> {VGS_STOP} V, step = {VGS_STEP} V")
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # -- NGSPICE reference ---------------------------------------------------
    print("  [1/3] Running NGSPICE reference...")
    ng_mc = RESULTS_DIR / "combined_baked.lib"
    bake_inst_params(MODELCARD_PATH, ng_mc, "nmos_rvt", NMOS_INST_PARAMS)
    bake_inst_params(ng_mc, ng_mc, "pmos_rvt", PMOS_INST_PARAMS)

    ng_net = RESULTS_DIR / "ngspice_inverter_vtc.cir"
    ng_net.write_text(
        f"* CMOS Inverter VTC\n"
        f'.include "{ng_mc}"\n'
        f".temp 27\n"
        f"Vdd vdd 0 {VDD}\n"
        f"Vin in 0 0.0\n"
        f"Np out in vdd vdd pmos_rvt\n"
        f"Nn out in 0 0 nmos_rvt\n"
        f".dc Vin {VGS_START} {VGS_STOP} {VGS_STEP}\n"
        f".end\n"
    )

    ng_data = run_ngspice_dc_sweep(
        ng_net,
        RESULTS_DIR / "ngspice_inverter_vtc_runner.cir",
        RESULTS_DIR / "ngspice_inverter_vtc.csv",
        RESULTS_DIR / "ngspice_inverter_vtc.log",
        ["v(out)"],
    )
    ng_sweep = ng_data["sweep"]
    ng_vout = ng_data["v(out)"]
    print(f"    NGSPICE: {len(ng_sweep)} points, V(out) range: [{ng_vout.min():.6f}, {ng_vout.max():.6f}] V")

    # -- PyCircuitSim --------------------------------------------------------
    print("  [2/3] Running PyCircuitSim...")
    # PyCircuitSim: node 1=Vdd, node 2=Vin, node 3=out
    py_netlist = (
        f"* CMOS Inverter VTC\n"
        f"Vdd 1 0 {VDD}\n"
        f"Vin 2 0 0.0\n"
        f"Mp1 3 2 1 1 pmos1 L=30n NFIN=10\n"
        f"Mn1 3 2 0 0 nmos1 L=30n NFIN=10\n"
        f".model nmos1 NMOS (LEVEL=72)\n"
        f".model pmos1 PMOS (LEVEL=72)\n"
        f".dc Vin {VGS_START} {VGS_STOP} {VGS_STEP}\n"
        f".end\n"
    )
    py_sweep, py_results = run_pycircuitsim_dc_sweep(py_netlist, sweep_node="2")
    py_vout = py_results["3"]  # node 3 = out
    print(f"    PyCircuitSim: {len(py_sweep)} points, V(out) range: [{py_vout.min():.6f}, {py_vout.max():.6f}] V")

    # -- Compare -------------------------------------------------------------
    print("  [3/3] Computing metrics...")
    metrics = compute_metrics(ng_sweep, ng_vout, py_sweep, py_vout)

    print(f"    Common points = {metrics['n_common_points']}")
    print(f"    RMSE        = {metrics['rmse']:.6e} V")
    print(f"    NRMSE       = {metrics['nrmse']*100:.4f}%")
    print(f"    Max |error| = {metrics['max_abs_err']:.6e} V")
    print(f"    Max rel err = {metrics['max_rel_err']*100:.4f}%")

    # Plot
    plot_comparison(
        ng_sweep, ng_vout, py_sweep, py_vout,
        xlabel="Vin (V)", ylabel="Vout (V)",
        title="Task 11: CMOS Inverter VTC (Vdd=0.7V, L=30nm, NFIN=10)",
        output_path=RESULTS_DIR / "inverter_vtc_comparison.png",
        metrics=metrics,
    )
    print(f"    Plot saved to: {RESULTS_DIR}/inverter_vtc_comparison.png")

    # Pass/Fail
    nrmse_pass = metrics["nrmse"] < RMSE_THRESHOLD
    maxrel_pass = metrics["max_rel_err"] < MAX_REL_ERR_THRESHOLD
    overall = nrmse_pass and maxrel_pass

    print()
    print(f"    NRMSE  < {RMSE_THRESHOLD*100}%: {'PASS' if nrmse_pass else 'FAIL'} ({metrics['nrmse']*100:.4f}%)")
    print(f"    MaxRel < {MAX_REL_ERR_THRESHOLD*100}%: {'PASS' if maxrel_pass else 'FAIL'} ({metrics['max_rel_err']*100:.4f}%)")
    print(f"    Result: {'PASS' if overall else 'FAIL'}")
    print()

    return overall


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    """Run all DC sweep verification tests and report summary."""
    print()
    print("*" * 70)
    print("  BSIM-CMG DC Sweep Verification: PyCircuitSim vs NGSPICE")
    print(f"  ASAP7 7nm TT, L={L*1e9:.0f}nm, NFIN={NFIN}, Vdd={VDD}V")
    print("*" * 70)
    print()

    results: Dict[str, bool] = {}

    results["Task 9:  NMOS Id-Vgs"] = test_nmos_dc_sweep()
    results["Task 10: PMOS V(drain)"] = test_pmos_dc_sweep()
    results["Task 11: Inverter VTC"] = test_inverter_vtc()

    # -- Summary -------------------------------------------------------------
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")

    n_pass = sum(results.values())
    n_total = len(results)
    print(f"\n  Total: {n_pass}/{n_total} passed")

    if n_pass == n_total:
        print("\n  All DC sweep tests PASSED.")
    else:
        print(f"\n  {n_total - n_pass} test(s) FAILED.")

    print("=" * 70)
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
