#!/usr/bin/env python3
"""V6.4.7 P0 diagnostic — lifted-source NMOS Id-Vgs vs NGSPICE BSIM-CMG.

Quantifies the `_raw_voltages` NMOS source-frame bug
(``pycircuitsim/models/mosfet_nn.py:232-243``): NMOS terminal voltages are
passed ABSOLUTE into a network trained at Vs=0, so a source lifted above
ground is evaluated at phantom Vgs/Vds (inflated by +Vs) with Vbs=0 instead
of -Vs. PMOS is source-shifted and unaffected.

Sweep: NMOS Id-Vgs per tech (TSMC5/7/12/16) at tech-default L/NFIN/VT with
absolute terminal voltages Vs=vs0, Vb=0, Vd=VDD, Vg = 0..VDD (5 mV grid),
for vs0 in {0, 0.1, 0.2}*VDD. vs0=0 is the grounded control. NOTE: the drain
sits at VDD here (the 55-config gate biases Vds=VDD/2), so the control
matches the existing grounded baseline qualitatively (few-% NRMSE), not
bit-for-bit.

The NN side runs through the real LEVEL=73 netlist ``.dc`` path with four
independent DC V-sources, so `_raw_voltages` sees absolute node voltages.
Ground truth is the same NGSPICE OSDI BSIM-CMG mechanism as
``verify_nn_multi_tech_dc.py`` — never a simplified equation.

Permanent gate: NRMSE <= 10% per config (`DC_NRMSE_PASS`) — the vs input had
zero verification coverage before V6.4.7; this sweep is the canary for the
P0 frame fix and the P2 reverse-clamp work. The pre-fix capture lives at
``results/v6_4_7/lifted_source_sweep_prefix.{csv,md}`` (run with
``--label prefix --no-gate``).

Usage:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
      conda run -n pycircuitsim python tests/verify_nn_lifted_source_dc.py \\
      [--label postfix] [--no-gate]
"""
from __future__ import annotations

import argparse
import csv
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.common.nn_sweep import NN_TECHS, curve_metrics  # noqa: E402
from tests.verify_nn_dc_tran import (  # noqa: E402
    ALL_TEST_TECHS,
    NGSPICE_BIN,
    OSDI_PATH,
    TestTechConfig,
    create_baked_modelcard,
)

VS0_FRACTIONS: List[float] = [0.0, 0.1, 0.2]   # source lift, fraction of VDD
VG_STEP = 0.005                                 # same grid as the 55-cfg gate
DC_NRMSE_PASS = 10.0                            # % — same gate as the 55-cfg DC
RESULTS_DIR = PROJECT_ROOT / "tests" / "verify_nn_lifted_source_dc_results"
OUT_DIR = PROJECT_ROOT / "results" / "v6_4_7"


def run_ngspice_nmos_dc_lifted(
    tech: TestTechConfig, work_dir: Path, vs0: float,
) -> Dict[str, np.ndarray]:
    """NGSPICE OSDI BSIM-CMG Id-Vgs with the source lifted to ``vs0``."""
    baked = create_baked_modelcard(tech, work_dir)
    tag = f"{tech.name}_vs{round(vs0 * 1e3)}mV"
    netlist = work_dir / f"ngspice_nmos_lifted_{tag}.cir"
    netlist.write_text(
        f"* NMOS lifted-source Id-Vgs (NGSPICE truth, {tag})\n"
        f'.include "{baked}"\n'
        f".temp 27\n"
        f"Vd d 0 {tech.vdd}\n"
        f"Vg g 0 0.0\n"
        f"Vs s 0 {vs0}\n"
        f"Vb b 0 0.0\n"
        f"N1 d g s b {tech.nmos_model}\n"
        f".dc Vg 0 {tech.vdd} {VG_STEP}\n"
        f".end\n")
    csv_path = work_dir / f"ngspice_nmos_lifted_{tag}.csv"
    log_path = work_dir / f"ngspice_nmos_lifted_{tag}.log"
    runner = work_dir / f"ngspice_nmos_lifted_{tag}_runner.cir"
    runner.write_text(
        f"* NGSPICE DC runner ({tag})\n"
        f".control\n"
        f"osdi {OSDI_PATH}\n"
        f"source {netlist}\n"
        f"set filetype=ascii\nset wr_vecnames\nrun\n"
        f"wrdata {csv_path} i(Vd)\n"
        f".endc\n.end\n")
    res = subprocess.run(
        [NGSPICE_BIN, "-b", "-o", str(log_path), str(runner)],
        capture_output=True, text=True)
    if not csv_path.exists():
        tail = log_path.read_text()[-500:] if log_path.exists() else "(no log)"
        raise RuntimeError(
            f"NGSPICE produced no output for {tag}: RC={res.returncode}, {tail}")
    data = np.array([[float(x) for x in ln.split()]
                     for ln in csv_path.read_text().splitlines()[1:] if ln.strip()])
    if not np.all(np.isfinite(data)):
        raise RuntimeError(f"NGSPICE output contains NaN/Inf for {tag}")
    return {"sweep": data[:, 0], "id": np.abs(data[:, 1])}


def run_nn_nmos_dc_lifted(
    tech: TestTechConfig, work_dir: Path, vs0: float,
) -> Dict[str, np.ndarray]:
    """DirectNet (LEVEL=73) Id-Vgs through the real pycircuitsim .dc path."""
    from pycircuitsim.parser import Parser
    from pycircuitsim.simulation import run_dc_sweep
    from pycircuitsim.visualizer import Visualizer

    tag = f"{tech.name}_vs{round(vs0 * 1e3)}mV"
    netlist = work_dir / f"nn_nmos_lifted_{tag}.sp"
    netlist.write_text(
        f"* NN NMOS lifted-source Id-Vgs (LEVEL=73, {tag})\n"
        f"Vd d 0 {tech.vdd}\n"
        f"Vg g 0 0.0\n"
        f"Vs s 0 {vs0}\n"
        f"Vb b 0 0.0\n"
        f"Mn1 d g s b nmos_nn L={tech.l_nmos * 1e9:.0f}n NFIN={tech.nfin}\n"
        f".model nmos_nn NMOS (LEVEL=73 TECH={tech.nn_tech_key} VT={tech.nn_vt})\n"
        f".dc Vg 0 {tech.vdd} {VG_STEP}\n"
        f".end\n")
    logging.disable(logging.CRITICAL)
    try:
        parser = Parser()
        parser.parse_file(str(netlist))
        out_dir = work_dir / f"nn_dc_{tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        results = run_dc_sweep(parser.circuit, parser.analysis_params,
                               Visualizer(), out_dir, f"nn_nmos_{tag}")
    finally:
        logging.disable(logging.NOTSET)
    return {"sweep": np.array(results["g"]),
            "id": np.abs(np.array(results["i(Mn1)"]))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default="postfix",
                    help="output-file label (results/v6_4_7/"
                         "lifted_source_sweep_<label>.{csv,md})")
    ap.add_argument("--no-gate", action="store_true",
                    help="diagnostic mode: report metrics, no PASS/FAIL gate")
    args = ap.parse_args()

    print("=" * 78)
    print(f"  V6.4.7 — lifted-source NMOS Id-Vgs ({args.label})")
    print("=" * 78)
    rows: List[Dict[str, object]] = []
    curve_rows: List[List[object]] = []
    n_pass = 0
    for tk in NN_TECHS:
        tech = ALL_TEST_TECHS[tk]
        for frac in VS0_FRACTIONS:
            vs0 = round(tech.vdd * frac, 4)
            wd = RESULTS_DIR / tk / f"vs0_{round(frac * 100)}pct"
            wd.mkdir(parents=True, exist_ok=True)
            ref = run_ngspice_nmos_dc_lifted(tech, wd, vs0)
            test = run_nn_nmos_dc_lifted(tech, wd, vs0)
            m = curve_metrics(ref["sweep"], ref["id"], test["sweep"], test["id"])
            ok = m["nrmse"] <= DC_NRMSE_PASS
            n_pass += int(ok)
            verdict = "" if args.no_gate else ("  PASS" if ok else "  FAIL")
            rows.append({"tech": tk, "frac": frac, "vs0": vs0, **m,
                         "verdict": verdict.strip()})
            print(f"  {tk:<7s} vs0={vs0:6.3f}V ({frac:.1f}*VDD)  "
                  f"NRMSE={m['nrmse']:7.2f}%  MRE={m['mre']:7.2f}%  "
                  f"R2={m['r2']:8.5f}  MaxErr={m['max_err'] * 1e6:9.3f}uA"
                  f"{verdict}")
            id_nn = np.interp(ref["sweep"], test["sweep"], test["id"])
            curve_rows += [[tk, frac, vs0, f"{vg:.4f}", f"{ig:.6e}", f"{inn:.6e}"]
                           for vg, ig, inn in zip(ref["sweep"], ref["id"], id_nn)]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"lifted_source_sweep_{args.label}.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tech", "vs0_frac", "vs0_V", "vg_V",
                    "id_ngspice_A", "id_nn_A"])
        w.writerows(curve_rows)
    print(f"\n  [CSV] Raw curves saved: {csv_path}")

    md_path = OUT_DIR / f"lifted_source_sweep_{args.label}.md"
    lines = [
        f"# Lifted-source NMOS Id-Vgs sweep — {args.label} (V6.4.7 P0)",
        "",
        "DirectNet LEVEL=73 vs NGSPICE OSDI BSIM-CMG, absolute terminals"
        " Vs=vs0, Vb=0, Vd=VDD, Vg=0..VDD (5 mV grid)."
        " `_raw_voltages` NMOS source-frame canary (mosfet_nn.py:232-243).",
        "Note: Vd=VDD here; the 55-config grounded gate biases Vds=VDD/2,"
        " so vs0=0 is a qualitative (not bit-exact) control.",
        "",
        "| tech | vs0/VDD | vs0 (V) | NRMSE (%) | MRE (%) | R2 | MaxErr (uA)"
        " | verdict |",
        "|------|---------|---------|-----------|---------|----|----------"
        "|---------|",
    ]
    lines += [f"| {r['tech']} | {r['frac']:.1f} | {r['vs0']:.3f} "
              f"| {r['nrmse']:.2f} | {r['mre']:.2f} | {r['r2']:.5f} "
              f"| {r['max_err'] * 1e6:.3f} | {r['verdict'] or '-'} |"
              for r in rows]
    md_path.write_text("\n".join(lines) + "\n")
    print(f"  [MD]  Summary saved: {md_path}")

    if args.no_gate:
        return 0
    total = len(rows)
    print(f"\nRESULT: {n_pass}/{total} configs PASSED "
          f"(NRMSE <= {DC_NRMSE_PASS:.0f}%)")
    return 0 if n_pass == total else 1


if __name__ == "__main__":
    sys.exit(main())
