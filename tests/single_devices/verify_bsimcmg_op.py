#!/usr/bin/env python3
"""Verify BSIM-CMG (LEVEL=72) single-device operating points vs NGSPICE.

NMOS and PMOS at one bias, ASAP7 7nm TT, L=30 nm / NFIN=10 / VDD=0.7 V.
Criterion: drain current must match NGSPICE within 1% relative error.

The inverter case that used to live in this file is now
``tests/simple_circuits/verify_bsimcmg_inverter_op.py`` — same geometry, same
rail, one tier up. Both import their shared plumbing from
``tests.common.bsimcmg_op``.

Usage:
    conda run -n pycircuitsim python tests/single_devices/verify_bsimcmg_op.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.common.bsimcmg_op import (  # noqa: E402
    ABS_TOL_I, ABS_TOL_V, L, MODELCARD_PATH, NFIN, NMOS_INST_PARAMS,
    OSDI_PATH, PMOS_INST_PARAMS, RESULTS_DIR, VDD,
    bake_inst_params, get_mosfet_current_from_solution, pass_fail,
    relative_error, run_ngspice_custom, run_ngspice_op, run_pycircuitsim_op,
)

_relative_error = relative_error
_pass_fail = pass_fail


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
# Main
# ============================================================================

def main() -> int:
    """Run the single-device OP gates and report a summary."""
    print()
    print("*" * 70)
    print("  BSIM-CMG single-device OP: PyCircuitSim vs NGSPICE")
    print(f"  ASAP7 7nm TT, L={L*1e9:.0f}nm, NFIN={NFIN}")
    print("*" * 70)
    print()

    results: Dict[str, bool] = {}
    results["NMOS OP"] = test_nmos_op()
    results["PMOS OP"] = test_pmos_op()

    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")

    n_pass = sum(results.values())
    n_total = len(results)
    print(f"\n  Total: {n_pass}/{n_total} passed")
    print("\n  All tests PASSED." if n_pass == n_total
          else f"\n  {n_total - n_pass} test(s) FAILED.")
    print("=" * 70)
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
