#!/usr/bin/env python3
"""
Verify BSIM-CMG OP analysis: PyCircuitSim vs NGSPICE.

Tasks 6-8: Compare NMOS, PMOS, and Inverter operating points.

Uses PyCMG's testing utilities (bake_inst_params, run_ngspice_op) for the
NGSPICE reference, and runs PyCircuitSim programmatically for comparison.

Criteria: drain current / output voltage must match within 1% relative error.

Key NGSPICE OSDI notes:
  - OSDI devices use generic prefix (N), NOT MOSFET prefix (M).
  - Model type in .model block must be 'bsimcmg' (done by bake_inst_params).
  - DEVTYPE must be baked: 1=NMOS, 0=PMOS (ASAP7 modelcard lacks it).
  - Instance params (L, NFIN) cannot go on device line for OSDI.
"""
from __future__ import annotations

import sys
import os
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Tuple

# -- Project paths -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "models" / "PyCMG"))

OSDI_PATH = PROJECT_ROOT / "models" / "PyCMG" / "build-deep-verify" / "osdi" / "bsimcmg.osdi"
MODELCARD_PATH = PROJECT_ROOT / "models" / "PyCMG" / "tech_model_cards" / "ASAP7" / "7nm_TT_160803.pm"
NGSPICE_BIN = "/usr/local/ngspice-45.2/bin/ngspice"
RESULTS_DIR = PROJECT_ROOT / "tests" / "verify_bsimcmg_op_results"

# -- PyCMG testing utilities -------------------------------------------------
from pycmg.testing import bake_inst_params, run_ngspice_op

# -- PyCircuitSim imports ----------------------------------------------------
from pycircuitsim.parser import Parser
from pycircuitsim.solver import DCSolver


# -- Shared test parameters --------------------------------------------------
L = 30e-9        # 30nm channel length
NFIN = 10        # 10 fins
VDD = 0.7        # ASAP7 nominal Vdd
REL_TOL = 0.01   # 1% relative tolerance
ABS_TOL_I = 1e-9 # 1 nA absolute tolerance floor (for near-zero currents)
ABS_TOL_V = 1e-4 # 0.1 mV absolute tolerance floor (for voltages)

# BSIM-CMG DEVTYPE: distinguishes NMOS (1) from PMOS (0).
# ASAP7 modelcard does NOT contain DEVTYPE; PyCMG auto-injects it,
# but NGSPICE OSDI does not. Must bake into modelcard for NGSPICE.
NMOS_INST_PARAMS = {"L": L, "NFIN": float(NFIN), "DEVTYPE": 1}
PMOS_INST_PARAMS = {"L": L, "NFIN": float(NFIN), "DEVTYPE": 0}


def _relative_error(measured: float, reference: float, abs_tol: float) -> float:
    """Compute relative error with absolute tolerance floor."""
    diff = abs(measured - reference)
    denom = max(abs(reference), abs_tol)
    return diff / denom


def _pass_fail(rel_err: float, threshold: float = REL_TOL) -> str:
    """Return PASS/FAIL string."""
    return "PASS" if rel_err <= threshold else "FAIL"


def run_pycircuitsim_op(netlist_content: str) -> Dict[str, float]:
    """Run PyCircuitSim OP analysis on a netlist string."""
    tmpdir = tempfile.mkdtemp(prefix="pycircuitsim_op_")
    try:
        netlist_path = Path(tmpdir) / "circuit.sp"
        netlist_path.write_text(netlist_content)

        parser = Parser()
        parser.parse_file(str(netlist_path))
        circuit = parser.circuit

        initial_guess = circuit.initial_conditions if circuit.initial_conditions else None
        solver = DCSolver(
            circuit,
            initial_guess=initial_guess,
            use_source_stepping=True,
            source_stepping_steps=20,
        )
        solution = solver.solve()
        return solution
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def get_mosfet_current_from_solution(
    netlist_content: str,
    solution: Dict[str, float],
    mosfet_name: str,
) -> float:
    """Extract MOSFET drain current from a solved circuit."""
    tmpdir = tempfile.mkdtemp(prefix="pycircuitsim_cur_")
    try:
        netlist_path = Path(tmpdir) / "circuit.sp"
        netlist_path.write_text(netlist_content)

        parser = Parser()
        parser.parse_file(str(netlist_path))
        circuit = parser.circuit

        for comp in circuit.components:
            if comp.name.lower() == mosfet_name.lower():
                return comp.calculate_current(solution)

        raise ValueError(f"MOSFET '{mosfet_name}' not found in circuit")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_ngspice_custom(runner_path: Path, log_path: Path, csv_path: Path) -> Dict[str, float]:
    """Run NGSPICE with a custom runner script and parse CSV output."""
    res = subprocess.run(
        [NGSPICE_BIN, "-b", "-o", str(log_path), str(runner_path)],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        log_content = ""
        if log_path.exists():
            log_content = log_path.read_text()
        raise RuntimeError(
            f"NGSPICE failed (rc={res.returncode}):\n"
            f"  stdout: {res.stdout[:300]}\n"
            f"  stderr: {res.stderr[:300]}\n"
            f"  log (last 500 chars): ...{log_content[-500:]}\n"
        )

    if not csv_path.exists():
        log_content = log_path.read_text() if log_path.exists() else "(no log)"
        raise RuntimeError(
            f"NGSPICE produced no CSV output: {csv_path}\n"
            f"  log (last 500 chars): ...{log_content[-500:]}\n"
        )

    with csv_path.open() as f:
        csv_lines = f.readlines()
    if not csv_lines:
        raise RuntimeError(f"Empty NGSPICE CSV: {csv_path}")
    headers = csv_lines[0].split()
    values = [float(x) for x in csv_lines[1].split()]
    return {name: values[i] for i, name in enumerate(headers)}


# ============================================================================
# Task 6: NMOS OP
# ============================================================================

def test_nmos_op() -> bool:
    """Verify NMOS OP: single NMOS at Vds=0.5V, Vgs=0.5V."""
    print("=" * 70)
    print("Task 6: NMOS Operating Point Verification")
    print("=" * 70)
    print(f"  L = {L*1e9:.0f} nm, NFIN = {NFIN}")
    print(f"  Vds = 0.5 V, Vgs = 0.5 V")
    print()

    # -- NGSPICE reference (using PyCMG utility) -----------------------------
    print("  [1/2] Running NGSPICE reference...")
    ng_result = run_ngspice_op(
        modelcard=MODELCARD_PATH,
        model_name="nmos_rvt",
        inst_params=NMOS_INST_PARAMS,
        vd=0.5, vg=0.5, vs=0.0, ve=0.0,
        tag="verify_nmos_op",
    )
    ng_id = ng_result["id"]
    ng_gm = ng_result["gm"]
    ng_gds = ng_result["gds"]

    print(f"    NGSPICE: i(Vd) = {ng_id:.6e} A")
    print(f"    NGSPICE: gm = {ng_gm:.6e} S, gds = {ng_gds:.6e} S")

    # -- PyCircuitSim --------------------------------------------------------
    print("  [2/2] Running PyCircuitSim...")

    nmos_netlist = """\
* NMOS OP verification
Vds 1 0 0.5
Vgs 2 0 0.5

Mn1 1 2 0 0 nmos1 L=30n NFIN=10

.model nmos1 NMOS (LEVEL=72)

.op

.end
"""
    solution = run_pycircuitsim_op(nmos_netlist)
    py_id = get_mosfet_current_from_solution(nmos_netlist, solution, "mn1")

    print(f"    PyCircuitSim: I_drain = {py_id:.6e} A")
    print(f"    PyCircuitSim node voltages: {solution}")

    # -- Compare (use magnitude) ---------------------------------------------
    ng_id_mag = abs(ng_id)
    py_id_mag = abs(py_id)

    rel_err = _relative_error(py_id_mag, ng_id_mag, ABS_TOL_I)
    status = _pass_fail(rel_err)

    print()
    print(f"  NGSPICE |Id|   = {ng_id_mag:.6e} A")
    print(f"  PySim   |Id|   = {py_id_mag:.6e} A")
    print(f"  Relative error = {rel_err*100:.4f}%")
    print(f"  Result: {status}")
    print()

    return status == "PASS"


# ============================================================================
# Task 7: PMOS OP
# ============================================================================

def test_pmos_op() -> bool:
    """Verify PMOS OP: PMOS with Vdd=0.7, Vg=0.2, Rload=10k."""
    print("=" * 70)
    print("Task 7: PMOS Operating Point Verification")
    print("=" * 70)
    print(f"  L = {L*1e9:.0f} nm, NFIN = {NFIN}")
    print(f"  Vdd = {VDD} V, Vg = 0.2 V, Rload = 10k")
    print()

    # -- NGSPICE reference (standalone PMOS with voltage sources) -------------
    # First verify raw PMOS current matches PyCMG using run_ngspice_op
    print("  [1/3] Verifying raw PMOS current (voltage-source driven)...")
    ng_raw = run_ngspice_op(
        modelcard=MODELCARD_PATH,
        model_name="pmos_rvt",
        inst_params=PMOS_INST_PARAMS,
        vd=0.0, vg=0.2, vs=0.7, ve=0.7,
        tag="verify_pmos_raw",
    )
    print(f"    NGSPICE raw: id = {ng_raw['id']:.6e} A, gm = {ng_raw['gm']:.6e} S")

    # -- NGSPICE reference (resistor-loaded) ---------------------------------
    print("  [2/3] Running NGSPICE reference (resistor-loaded PMOS)...")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ng_modelcard = RESULTS_DIR / "ng_pmos_rvt.lib"
    bake_inst_params(
        MODELCARD_PATH, ng_modelcard, "pmos_rvt", PMOS_INST_PARAMS,
    )

    ng_netlist = RESULTS_DIR / "ngspice_pmos_op.cir"
    ng_runner = RESULTS_DIR / "ngspice_pmos_op_runner.cir"
    ng_csv = RESULTS_DIR / "ngspice_pmos_op.csv"
    ng_log = RESULTS_DIR / "ngspice_pmos_op.log"

    ng_netlist.write_text(
        f'* PMOS OP with load resistor\n'
        f'.include "{ng_modelcard}"\n'
        f'.temp 27\n'
        f'Vdd vdd 0 {VDD}\n'
        f'Vg g 0 0.2\n'
        f'N1 drain g vdd vdd pmos_rvt\n'
        f'Rload drain 0 10k\n'
        f'.op\n'
        f'.end\n'
    )

    ng_runner.write_text(
        f'* ngspice runner for PMOS OP\n'
        f'.control\n'
        f'osdi {OSDI_PATH}\n'
        f'source {ng_netlist}\n'
        f'set filetype=ascii\n'
        f'set wr_vecnames\n'
        f'run\n'
        f'wrdata {ng_csv} v(drain) v(g) v(vdd)\n'
        f'.endc\n'
        f'.end\n'
    )

    ng_data = run_ngspice_custom(ng_runner, ng_log, ng_csv)
    ng_v_drain = ng_data["v(drain)"]
    ng_id_mag = abs(ng_v_drain / 10e3)

    print(f"    NGSPICE: V(drain) = {ng_v_drain:.6f} V")
    print(f"    NGSPICE: |Id| = V(drain)/10k = {ng_id_mag:.6e} A")

    # -- PyCircuitSim --------------------------------------------------------
    print("  [3/3] Running PyCircuitSim...")

    pmos_netlist = (
        f'* PMOS OP with load resistor\n'
        f'Vdd 1 0 {VDD}\n'
        f'Vg 2 0 0.2\n'
        f'\n'
        f'Mp1 3 2 1 1 pmos1 L=30n NFIN=10\n'
        f'\n'
        f'Rload 3 0 10k\n'
        f'\n'
        f'.model pmos1 PMOS (LEVEL=72)\n'
        f'\n'
        f'.op\n'
        f'\n'
        f'.end\n'
    )
    solution = run_pycircuitsim_op(pmos_netlist)
    py_v_drain = solution.get("3", 0.0)
    py_id_mag = abs(py_v_drain / 10e3)

    print(f"    PyCircuitSim: V(drain) = {py_v_drain:.6f} V")
    print(f"    PyCircuitSim: |Id| = V(drain)/10k = {py_id_mag:.6e} A")
    print(f"    PyCircuitSim node voltages: {solution}")

    # -- Compare -------------------------------------------------------------
    rel_err_v = _relative_error(py_v_drain, ng_v_drain, ABS_TOL_V)
    rel_err_i = _relative_error(py_id_mag, ng_id_mag, ABS_TOL_I)
    status_v = _pass_fail(rel_err_v)
    status_i = _pass_fail(rel_err_i)
    overall = "PASS" if (status_v == "PASS" and status_i == "PASS") else "FAIL"

    print()
    print(f"  V(drain) comparison:")
    print(f"    NGSPICE = {ng_v_drain:.6f} V, PySim = {py_v_drain:.6f} V")
    print(f"    Relative error = {rel_err_v*100:.4f}% -> {status_v}")
    print(f"  |Id| comparison:")
    print(f"    NGSPICE = {ng_id_mag:.6e} A, PySim = {py_id_mag:.6e} A")
    print(f"    Relative error = {rel_err_i*100:.4f}% -> {status_i}")
    print(f"  Result: {overall}")
    print()

    return overall == "PASS"


# ============================================================================
# Task 8: Inverter OP
# ============================================================================

def _run_ngspice_inverter_op(vin: float) -> float:
    """Run NGSPICE inverter OP and return V(out).

    Bakes BOTH nmos_rvt and pmos_rvt with DEVTYPE into a single modelcard.
    Uses N-prefix device lines for OSDI compatibility.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Bake NMOS: target nmos_rvt, set DEVTYPE=1
    ng_nmos_mc = RESULTS_DIR / f"ng_nmos_rvt_inv_{vin:.1f}.lib"
    bake_inst_params(
        MODELCARD_PATH, ng_nmos_mc, "nmos_rvt", NMOS_INST_PARAMS,
    )

    # Bake PMOS: target pmos_rvt, set DEVTYPE=0
    ng_pmos_mc = RESULTS_DIR / f"ng_pmos_rvt_inv_{vin:.1f}.lib"
    bake_inst_params(
        MODELCARD_PATH, ng_pmos_mc, "pmos_rvt", PMOS_INST_PARAMS,
    )

    # For the inverter, we need BOTH models available as bsimcmg type.
    # bake_inst_params only converts the TARGET model to bsimcmg.
    # The NMOS file has nmos_rvt as bsimcmg, pmos_rvt as pmos level=72.
    # The PMOS file has pmos_rvt as bsimcmg, nmos_rvt as nmos level=72.
    # Including both files: NGSPICE sees two definitions for each model.
    # The LAST definition wins. So order matters.
    # Include NMOS first (nmos_rvt=bsimcmg), then PMOS (pmos_rvt=bsimcmg).
    # The PMOS file re-defines nmos_rvt as "nmos level=72" which NGSPICE 
    # can't parse (level 72 unknown). This causes errors.
    #
    # Solution: Create a single combined modelcard by baking BOTH models
    # into one file. We do this by:
    # 1. Bake nmos_rvt in the original file -> file A
    # 2. From file A, bake pmos_rvt -> file B (both models now bsimcmg)

    combined_mc = RESULTS_DIR / f"ng_inverter_mc_{vin:.1f}.lib"
    # Step 1: bake NMOS
    bake_inst_params(MODELCARD_PATH, combined_mc, "nmos_rvt", NMOS_INST_PARAMS)
    # Step 2: from that result, also bake PMOS
    bake_inst_params(combined_mc, combined_mc, "pmos_rvt", PMOS_INST_PARAMS)

    tag = f"inv_vin{vin:.1f}"
    net_path = RESULTS_DIR / f"ngspice_inverter_op_{tag}.cir"
    runner_path = RESULTS_DIR / f"ngspice_inverter_op_{tag}_runner.cir"
    csv_path = RESULTS_DIR / f"ngspice_inverter_op_{tag}.csv"
    log_path = RESULTS_DIR / f"ngspice_inverter_op_{tag}.log"

    net_path.write_text(
        f'* CMOS Inverter OP (Vin={vin}V)\n'
        f'.include "{combined_mc}"\n'
        f'.temp 27\n'
        f'Vdd vdd 0 {VDD}\n'
        f'Vin in 0 {vin}\n'
        f'Np out in vdd vdd pmos_rvt\n'
        f'Nn out in 0 0 nmos_rvt\n'
        f'.op\n'
        f'.end\n'
    )

    runner_path.write_text(
        f'* ngspice runner\n'
        f'.control\n'
        f'osdi {OSDI_PATH}\n'
        f'source {net_path}\n'
        f'set filetype=ascii\n'
        f'set wr_vecnames\n'
        f'run\n'
        f'wrdata {csv_path} v(out) v(in) v(vdd)\n'
        f'.endc\n'
        f'.end\n'
    )

    ng_data = run_ngspice_custom(runner_path, log_path, csv_path)
    return ng_data["v(out)"]


def test_inverter_op() -> bool:
    """Verify Inverter OP at Vin=0.0V and Vin=0.7V."""
    print("=" * 70)
    print("Task 8: CMOS Inverter Operating Point Verification")
    print("=" * 70)
    print(f"  L = {L*1e9:.0f} nm, NFIN = {NFIN}, Vdd = {VDD} V")
    print()

    all_pass = True

    for vin in [0.0, VDD]:
        label = f"Vin={vin:.1f}V"
        expected = "~0.7V" if vin == 0.0 else "~0V"
        print(f"  --- {label} (expect Vout {expected}) ---")

        # -- NGSPICE ---------------------------------------------------------
        print(f"    [1/2] Running NGSPICE...")
        ng_vout = _run_ngspice_inverter_op(vin)
        print(f"      NGSPICE: V(out) = {ng_vout:.6f} V")

        # -- PyCircuitSim ----------------------------------------------------
        print(f"    [2/2] Running PyCircuitSim...")

        inv_netlist = (
            f'* CMOS Inverter OP (Vin={vin}V)\n'
            f'Vdd 1 0 {VDD}\n'
            f'Vin 2 0 {vin}\n'
            f'\n'
            f'Mp1 3 2 1 1 pmos1 L=30n NFIN=10\n'
            f'Mn1 3 2 0 0 nmos1 L=30n NFIN=10\n'
            f'\n'
            f'.model nmos1 NMOS (LEVEL=72)\n'
            f'.model pmos1 PMOS (LEVEL=72)\n'
            f'\n'
            f'.op\n'
            f'\n'
            f'.end\n'
        )
        solution = run_pycircuitsim_op(inv_netlist)
        py_vout = solution.get("3", float("nan"))
        print(f"      PyCircuitSim: V(out) = {py_vout:.6f} V")
        print(f"      PyCircuitSim voltages: {solution}")

        # -- Compare ---------------------------------------------------------
        abs_diff = abs(py_vout - ng_vout)
        denom = max(abs(ng_vout), 0.01)
        rel_err = abs_diff / denom
        status = _pass_fail(rel_err)

        print(f"      Abs diff = {abs_diff:.6f} V, Rel err = {rel_err*100:.4f}% -> {status}")
        if status == "FAIL":
            all_pass = False
        print()

    print(f"  Overall inverter result: {'PASS' if all_pass else 'FAIL'}")
    print()

    return all_pass


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    """Run all verification tests and report summary."""
    print()
    print("*" * 70)
    print("  BSIM-CMG OP Verification: PyCircuitSim vs NGSPICE")
    print(f"  ASAP7 7nm TT, L={L*1e9:.0f}nm, NFIN={NFIN}")
    print("*" * 70)
    print()

    results: Dict[str, bool] = {}

    results["Task 6: NMOS OP"] = test_nmos_op()
    results["Task 7: PMOS OP"] = test_pmos_op()
    results["Task 8: Inverter OP"] = test_inverter_op()

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
        print("\n  All tests PASSED.")
    else:
        print(f"\n  {n_total - n_pass} test(s) FAILED.")

    print("=" * 70)
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
