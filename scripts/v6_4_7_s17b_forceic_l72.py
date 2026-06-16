#!/usr/bin/env python3
"""V6.4.7 S17b — native LEVEL=72 (OSDI BSIM-CMG) force_ic control.

Decisive model-vs-solver test for force_ic, mirroring S6's native-L72 RO
control. Build the SAME 6T cell as verify_complex_sram_snm but with LEVEL=72
(exact OSDI physics) instead of LEVEL=73 (NN), and run PyCircuitSim's force_ic
release. Both states.

  * native L72 RAILS (8/8)  -> force_ic is a MODEL-accuracy problem; a model
    fix (e.g. the S17 linear-region driver corridor) is the right lever.
  * native L72 does NOT rail (lands inboard/saddle, or diverges) -> PyCircuitSim's
    force_ic release cannot reach the rail even with exact physics -> a SOLVER
    fixed-point-selection problem; model levers won't help.

Run:
    OMP_NUM_THREADS=1 conda run -n pycircuitsim \
      python scripts/v6_4_7_s17b_forceic_l72.py --tech TSMC16,TSMC7,TSMC12,TSMC5
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "external_compact_models" / "PyCMG"))
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from tests.common.complex import BENCH, parse_netlist  # noqa: E402
from pycircuitsim.solver import DCSolver  # noqa: E402


def l72_6t_netlist(bt, q0, qb0, path: Path, wl_off: bool = False,
                   level: int = 72) -> Path:
    tfin_n = bt.tfin * 1e9
    nfin = bt.nfin
    wl_v = 0.0 if wl_off else bt.vdd   # wl=0 => access OFF => true HOLD/retention
    if level == 72:
        nm, pm = bt.nmos_model, bt.pmos_model
        tf = f" TFIN={tfin_n:.1f}n"
        models = (f".model {nm} NMOS (LEVEL=72)\n.model {pm} PMOS (LEVEL=72)\n")
    else:  # LEVEL=73 NN
        nm, pm = "nmos_nn", "pmos_nn"
        tf = ""
        models = (f".model {nm} NMOS (LEVEL=73 TECH={bt.nn_tech} VT={bt.vt})\n"
                  f".model {pm} PMOS (LEVEL=73 TECH={bt.nn_tech} VT={bt.vt})\n")
    path.write_text(
        f"* 6T SRAM cell — LEVEL={level} ({bt.name}) wl={'OFF' if wl_off else 'ON'}\n"
        f"Vdd vdd 0 {bt.vdd}\n"
        f"Vwl wl 0 {wl_v}\nVbl bl 0 {bt.vdd}\nVblb blb 0 {bt.vdd}\n"
        f".ic V(q)={q0} V(qb)={qb0}\n"
        f"Mpl qb q vdd vdd {pm} L=20n NFIN={nfin}{tf}\n"
        f"Mnl qb q 0   0   {nm} L=16n NFIN={nfin}{tf}\n"
        f"Mpr q qb vdd vdd {pm} L=20n NFIN={nfin}{tf}\n"
        f"Mnr q qb 0   0   {nm} L=16n NFIN={nfin}{tf}\n"
        f"Mal bl  wl q  0 {nm} L=16n NFIN={nfin}{tf}\n"
        f"Mar blb wl qb 0 {nm} L=16n NFIN={nfin}{tf}\n"
        f"{models}"
        f".op\n.end\n")
    return path


def probe(tech: str, work_dir: Path, wl_off: bool = False, level: int = 72) -> None:
    bt = BENCH[tech]
    vdd = bt.vdd
    band = 0.1 * vdd
    npass = 0
    for tag, (q0, qb0) in (("state1", (vdd, 0.0)), ("state0", (0.0, vdd))):
        netlist = l72_6t_netlist(bt, q0, qb0,
                                 work_dir / f"l{level}_{tech}_{tag}.sp", wl_off, level)
        logging.disable(logging.CRITICAL)
        q = qb = float("nan")
        resid = thr = None
        try:
            circuit = parse_netlist(netlist).circuit
            solver = DCSolver(circuit, initial_guess=circuit.initial_conditions or None,
                              use_source_stepping=True, force_ic=True)
            sol = solver.solve()
            q = sol.get("q", float("nan"))
            qb = sol.get("qb", float("nan"))
            resid = getattr(solver, "_last_dc_residual", None)
            thr = getattr(solver, "_last_dc_resid_threshold", None)
            resid_ok = resid is not None and thr is not None and resid <= thr
            rail_ok = abs(q - q0) < band and abs(qb - qb0) < band
            ok = bool(resid_ok and rail_ok)
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"  {tech} {tag}: DIVERGED/ERROR {exc!r}")
            continue
        finally:
            logging.disable(logging.NOTSET)
        npass += int(ok)
        rs = f"{resid:.2e}" if isinstance(resid, float) else "n/a"
        print(f"  {tech} {tag}: q={q:.3f} qb={qb:.3f} (seed {q0}/{qb0}) "
              f"resid={rs} rail_ok={rail_ok} -> {'PASS' if ok else 'FAIL'}")
    print(f"  >>> {tech} native-L72 force_ic: {npass}/2")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tech", default="TSMC16,TSMC7,TSMC12,TSMC5")
    ap.add_argument("--wl-off", action="store_true",
                    help="wordline=0 (access OFF) => true hold/retention test")
    ap.add_argument("--level", type=int, default=72, choices=[72, 73])
    args = ap.parse_args()
    work_dir = PROJECT_ROOT / "results" / "v6_4_7" / "s17b_scratch"
    work_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print(f"S17b — native LEVEL=72 force_ic control (exact OSDI physics, "
          f"PyCircuitSim solver) — wl={'OFF/hold' if args.wl_off else 'ON/read'}")
    print("=" * 80)
    for tech in [t.strip() for t in args.tech.split(",")]:
        probe(tech, work_dir, args.wl_off, args.level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
