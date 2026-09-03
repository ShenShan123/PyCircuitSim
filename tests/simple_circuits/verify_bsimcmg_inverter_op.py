#!/usr/bin/env python3
"""Verify the BSIM-CMG (LEVEL=72) CMOS inverter operating point vs NGSPICE.

ASAP7 7nm TT at L=30 nm / NFIN=10 / VDD=0.7 V — the SAME geometry the
single-device OP gates use (``tests/single_devices/verify_bsimcmg_op.py``), so
that a disagreement here is attributable to the circuit rather than to the
device model. Both gates were one file until V7.5.8; they share
``tests.common.bsimcmg_op``.

The input is biased at each end of the rail (0 V, then VDD): the two points
where the inverter is fully switched and the expected output is unambiguous.
Criterion: V(out) within 1% relative error of NGSPICE.

The topology lives once in ``circuit_templates/L1_primitives/inverter.spice.tmpl``;
the gate renders PyCircuitSim and NGSPICE model adapters from that template.

Usage:
    conda run -n pycircuitsim python tests/simple_circuits/verify_bsimcmg_inverter_op.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.common.base import template_deck, render_template  # noqa: E402
from tests.common.bsimcmg_op import (  # noqa: E402
    L, MODELCARD_PATH, NFIN, NMOS_INST_PARAMS, OSDI_PATH, PMOS_INST_PARAMS,
    RESULTS_DIR, VDD, bake_inst_params, pass_fail, run_ngspice_custom,
    run_pycircuitsim_op,
)

TEMPLATE = template_deck("inverter.spice.tmpl")


def _combined_modelcard(vin: float) -> Path:
    """Bake BOTH models into ONE card.

    This has to be a single file. ``bake_inst_params`` only converts its
    TARGET model to bsimcmg type and leaves the other declared as
    ``level=72``, which NGSPICE cannot parse — so including two separately
    baked cards fails whichever model came second. Baking the second pass on
    top of the first pass's output is what produces one card with both.
    """
    combined = RESULTS_DIR / f"ng_inverter_mc_{vin:.1f}.lib"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    bake_inst_params(MODELCARD_PATH, combined, "nmos_rvt", NMOS_INST_PARAMS)
    bake_inst_params(combined, combined, "pmos_rvt", PMOS_INST_PARAMS)
    return combined


def ngspice_inverter_op(vin: float) -> float:
    """Run the NGSPICE inverter OP reference and return V(out)."""
    combined_mc = _combined_modelcard(vin)

    tag = f"inv_vin{vin:.1f}"
    net_path = RESULTS_DIR / f"ngspice_inverter_op_{tag}.cir"
    runner_path = RESULTS_DIR / f"ngspice_inverter_op_{tag}_runner.cir"
    csv_path = RESULTS_DIR / f"ngspice_inverter_op_{tag}.csv"
    log_path = RESULTS_DIR / f"ngspice_inverter_op_{tag}.log"

    net_path.write_text(render_template(TEMPLATE, {
        "MODEL_SETUP": f'.include "{combined_mc}"',
        "TEMP": "27",
        "VDD": f"{VDD}",
        "INPUT_SPEC": f"{vin}",
        "N_PREFIX": "N", "P_PREFIX": "N",
        "N_DEVICE": "nmos_rvt", "P_DEVICE": "pmos_rvt",
        "OUTPUT_LOAD": "", "INITIAL_CONDITION": "",
        "ANALYSIS": ".op",
    }))

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

    return run_ngspice_custom(runner_path, log_path, csv_path)["v(out)"]


def pycircuitsim_inverter_deck(vin: float) -> str:
    """Render the PyCircuitSim LEVEL=72 adapter at one input bias."""
    length = f"{L * 1e9:g}n"
    return render_template(TEMPLATE, {
        "MODEL_SETUP": (
            ".model nmos1 NMOS (LEVEL=72)\n"
            ".model pmos1 PMOS (LEVEL=72)"
        ),
        "TEMP": "27", "VDD": f"{VDD}", "INPUT_SPEC": f"{vin}",
        "N_PREFIX": "M", "P_PREFIX": "M",
        "N_DEVICE": f"nmos1 L={length} NFIN={NFIN}",
        "P_DEVICE": f"pmos1 L={length} NFIN={NFIN}",
        "OUTPUT_LOAD": "", "INITIAL_CONDITION": "",
        "ANALYSIS": ".op",
    })


def test_inverter_op() -> bool:
    """Verify the inverter OP at Vin=0 V and Vin=VDD."""
    print("=" * 70)
    print("CMOS Inverter Operating Point Verification")
    print("=" * 70)
    print(f"  L = {L*1e9:.0f} nm, NFIN = {NFIN}, Vdd = {VDD} V")
    print()

    all_pass = True

    for vin in [0.0, VDD]:
        label = f"Vin={vin:.1f}V"
        expected = f"~{VDD}V" if vin == 0.0 else "~0V"
        print(f"  --- {label} (expect Vout {expected}) ---")

        print("    [1/2] Running NGSPICE...")
        ng_vout = ngspice_inverter_op(vin)
        print(f"      NGSPICE: V(out) = {ng_vout:.6f} V")

        print("    [2/2] Running PyCircuitSim...")
        solution = run_pycircuitsim_op(pycircuitsim_inverter_deck(vin))
        py_vout = solution.get("out", float("nan"))
        print(f"      PyCircuitSim: V(out) = {py_vout:.6f} V")
        print(f"      PyCircuitSim voltages: {solution}")

        abs_diff = abs(py_vout - ng_vout)
        denom = max(abs(ng_vout), 0.01)
        rel_err = abs_diff / denom
        status = pass_fail(rel_err)

        print(f"      Abs diff = {abs_diff:.6f} V, "
              f"Rel err = {rel_err*100:.4f}% -> {status}")
        if status == "FAIL":
            all_pass = False
        print()

    print(f"  Overall inverter result: {'PASS' if all_pass else 'FAIL'}")
    print()
    return all_pass


def main() -> int:
    print()
    print("*" * 70)
    print("  BSIM-CMG inverter OP: PyCircuitSim vs NGSPICE")
    print(f"  ASAP7 7nm TT, L={L*1e9:.0f}nm, NFIN={NFIN}")
    print("*" * 70)
    print()

    results: Dict[str, bool] = {"Inverter OP": test_inverter_op()}

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
