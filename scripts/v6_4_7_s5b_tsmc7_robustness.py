#!/usr/bin/env python3
"""S5b adversary-required robustness probe: TSMC7 switchcap PASS must hold
across Vin in {0.55, 0.60, 0.65}*VDD (0.60 is the harness default).

The reviewer's concern: the TSMC7 droop fell 2.208 -> 0.541 mV when the held
bias moved 48 mV; if that is a lucky point on the NN error surface, a small
Vin change re-FAILs it and the S5b uic fix gets no headline credit for the
cell. Gate per point: charge err <= 5% of VDD AND |dn-ng| droop <= max(10%
of NG droop, 0.1% of VDD) — identical to verify_complex_switchcap.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "external_compact_models" / "PyCMG" / "tests",
          ROOT / "external_compact_models" / "PyCMG",
          ROOT / "external_compact_models",
          ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tests.common.complex import BENCH, render_directnet_netlist, \
    run_directnet_transient  # noqa: E402
from tests.verify_complex_switchcap import (  # noqa: E402
    TEMPLATE, SAMPLE_END, HOLD_START, HOLD_END, CHARGE_TOL, DROOP_TOL,
    DROOP_FLOOR_FRAC, _at, run_ngspice_sc)

VIN_FRACS = (0.55, 0.60, 0.65)


def main() -> int:
    bt = BENCH["TSMC7"]
    scratch = ROOT / "results" / "v6_4_7" / "s5_logs" / "s5b_robustness"
    n_pass = 0
    print("TSMC7 switchcap robustness probe (S5b adversary finding 2)")
    print(f"{'Vin':>6s} | {'NG chg':>7s} | {'DN chg':>7s} | {'chg%':>6s} | "
          f"{'droop |dn-ng| mV':>16s} | {'%alw':>6s} | verdict")
    for frac in VIN_FRACS:
        vin = round(frac * bt.vdd, 4)
        work = scratch / f"vin{round(frac * 100)}"
        work.mkdir(parents=True, exist_ok=True)

        ng = _ng_with_vin(bt, work, vin)
        netlist = render_directnet_netlist(
            TEMPLATE, bt, work / f"switchcap_TSMC7_vin{vin}.sp")
        netlist.write_text(netlist.read_text().replace(
            "Vin vin 0 0.48", f"Vin vin 0 {vin}"))
        dn, partial, err = run_directnet_transient(netlist)
        if partial:
            print(f"{vin:6.3f} | DN PARTIAL ({err}) -> FAIL")
            continue

        t, v = np.asarray(dn["time"]), np.asarray(dn["vsamp"])
        ng_chg = _at(ng["time"], ng["vsamp"], SAMPLE_END)
        dn_chg = _at(t, v, SAMPLE_END)
        ng_dr = (_at(ng["time"], ng["vsamp"], HOLD_START)
                 - _at(ng["time"], ng["vsamp"], HOLD_END))
        dn_dr = _at(t, v, HOLD_START) - _at(t, v, HOLD_END)
        chg_err = abs(dn_chg - ng_chg) / bt.vdd * 100.0
        dr_abs = abs(dn_dr - ng_dr)
        allow = max(DROOP_TOL * abs(ng_dr), DROOP_FLOOR_FRAC * bt.vdd)
        ok = chg_err <= CHARGE_TOL * 100 and dr_abs <= allow
        n_pass += int(ok)
        print(f"{vin:6.3f} | {ng_chg:7.4f} | {dn_chg:7.4f} | {chg_err:6.2f} | "
              f"{dr_abs * 1e3:16.3f} | {dr_abs / allow * 100:6.1f} | "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"\nRESULT: {n_pass}/{len(VIN_FRACS)} Vin points PASS "
          f"(adversary gate: 3/3 required for headline credit)")
    return 0 if n_pass == len(VIN_FRACS) else 1


def _ng_with_vin(bt, work: Path, vin: float):
    """run_ngspice_sc with the Vin source rewritten (no harness flag)."""
    import tests.verify_complex_switchcap as sc
    orig = sc._vin
    try:
        sc._vin = lambda b: vin  # type: ignore[assignment]
        return run_ngspice_sc(bt, work)
    finally:
        sc._vin = orig


if __name__ == "__main__":
    sys.exit(main())
