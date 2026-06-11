#!/usr/bin/env python3
"""S5 follow-up: where does the DN vsamp overshoot accrue? Print trajectory
heads (DN vs NGSPICE) for TSMC5 + TSMC16, including the t<0.4 ns segment and
the initial operating point that no S5 window covered."""
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
from tests.verify_complex_switchcap import TEMPLATE, _vin, \
    run_ngspice_sc  # noqa: E402

T_PROBE = [0.0, 0.05e-9, 0.1e-9, 0.2e-9, 0.3e-9, 0.4e-9, 0.5e-9,
           0.55e-9, 0.6e-9, 0.8e-9, 1.5e-9, 2.3e-9]


def main() -> int:
    scratch = ROOT / "results" / "v6_4_7" / "s5_logs" / "scratch_head"
    for tech in ("TSMC5", "TSMC16"):
        bt = BENCH[tech]
        work = scratch / tech
        work.mkdir(parents=True, exist_ok=True)
        netlist = render_directnet_netlist(
            TEMPLATE, bt, work / f"switchcap_{tech}.sp")
        netlist.write_text(netlist.read_text().replace(
            "Vin vin 0 0.48", f"Vin vin 0 {_vin(bt)}"))
        dn, partial, err = run_directnet_transient(netlist)
        t = np.asarray(dn["time"])
        v = np.asarray(dn["vsamp"])
        ng = run_ngspice_sc(bt, work)
        print(f"\n--- {tech} (Vin={_vin(bt)}, VDD={bt.vdd}) "
              f"partial={partial} ---")
        print(f"{'t (ns)':>8s} | {'DN vsamp':>9s} | {'NG vsamp':>9s}")
        for tq in T_PROBE:
            dnv = float(np.interp(tq, t, v))
            ngv = float(np.interp(tq, ng["time"], ng["vsamp"]))
            print(f"{tq * 1e9:8.2f} | {dnv:9.4f} | {ngv:9.4f}")
        print(f"  DN t0={t[0]:.3e} v0={v[0]:.4f}  (initial point)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
