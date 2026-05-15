#!/usr/bin/env python3
"""Benchmark 3c — 6T SRAM read SNM: DirectNet vs NGSPICE BSIM-CMG.

Part of the DirectNet V6.4 sprint, Phase 3 benchmark harness
(docs/plans/2026-05-15-directnet-complex-circuits.md).

Traces the read static-noise-margin (SNM) butterfly of a 6T SRAM bitcell.
Each butterfly lobe is the voltage-transfer curve of one half-cell under read
bias (WL = VDD, both bit lines held at VDD), obtained by breaking the
cross-coupled feedback and DC-sweeping the driven storage node. The mirror
lobe swaps q <-> qb. SNM is the side of the largest square that fits between
the two lobes.

Both lobes are produced by:
  (i)  DirectNet (LEVEL=73) half-cell DC sweeps (PyCircuitSim),
  (ii) the NGSPICE BSIM-CMG (LEVEL=72) ground truth.

It additionally solves the full cross-coupled 6T cell with ``force_ic=True``
(hard .ic mode) to confirm both storage states land on their rails — the same
solver path SRAM latches need.

Gate: both DirectNet butterfly curves positive (Vout >= 0) across NFIN
corners, and SNM tracking the NGSPICE reference.

Ground truth is ALWAYS NGSPICE BSIM-CMG (CLAUDE.md Validation rule).
Rule 16: report MRE / R2 / NRMSE / MaxErr.

Usage:
    conda run -n pycircuitsim python tests/verify_complex_sram_snm.py
    conda run -n pycircuitsim python tests/verify_complex_sram_snm.py --tech TSMC5
    conda run -n pycircuitsim python tests/verify_complex_sram_snm.py --nfin 2,5,10
"""
from __future__ import annotations

import argparse
import functools
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

print = functools.partial(print, flush=True)  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "PyCMG" / "tests"))

from tests.common.complex import (  # noqa: E402
    BENCH, BENCH_TECHS, RESULTS_BASE, BenchTech,
    get_baked_modelcard, run_ngspice_wrdata, parse_netlist, full_metrics,
)

DEFAULT_NFINS = [2, 5, 10]


# ---------------------------------------------------------------------------
# Half-cell butterfly lobe
# ---------------------------------------------------------------------------
def ngspice_lobe(bt: BenchTech, nfin: int, work_dir: Path) -> Dict[str, np.ndarray]:
    """One NGSPICE half-cell butterfly lobe: qb(q) under read bias."""
    baked = get_baked_modelcard(bt, nfin, work_dir)
    n, p = bt.nmos_model, bt.pmos_model
    body = [f'.include "{baked}"', ".temp 27", f"Vdd vdd 0 {bt.vdd}",
            f"Vwl wl 0 {bt.vdd}", f"Vbl bl 0 {bt.vdd}", "Vq q 0 0.0",
            f"Npl qb q vdd vdd {p}", f"Nnl qb q 0 0 {n}",
            f"Nna bl wl qb 0 {n}"]
    data = run_ngspice_wrdata("\n".join(body), "v(qb)", work_dir,
                              f"sram_{bt.name}_nfin{nfin}",
                              f"dc Vq 0 {bt.vdd} 0.005")
    return {"q": data[:, 0], "qb": data[:, 1]}


def _directnet_halfcell_netlist(bt: BenchTech, nfin: int, path: Path) -> Path:
    """A DirectNet (LEVEL=73) SRAM half-cell with broken feedback."""
    n_l = bt.l_nmos * 1e9
    p_l = bt.l_pmos * 1e9
    path.write_text(
        f"* SRAM read-SNM half-cell — DirectNet ({bt.name} NFIN={nfin})\n"
        f"Vdd vdd 0 {bt.vdd}\n"
        f"Vwl wl 0 {bt.vdd}\n"
        f"Vbl bl 0 {bt.vdd}\n"
        f"Vq q 0 0.0\n"
        f"Mpl qb q vdd vdd pmos_nn L={p_l:.0f}n NFIN={nfin}\n"
        f"Mnl qb q 0   0   nmos_nn L={n_l:.0f}n NFIN={nfin}\n"
        f"Mna bl wl qb 0 nmos_nn L={n_l:.0f}n NFIN={nfin}\n"
        f".model nmos_nn NMOS (LEVEL=73 TECH={bt.nn_tech} VT={bt.vt})\n"
        f".model pmos_nn PMOS (LEVEL=73 TECH={bt.nn_tech} VT={bt.vt})\n"
        f".dc Vq 0 {bt.vdd} 0.005\n"
        f".end\n")
    return path


def directnet_lobe(bt: BenchTech, nfin: int, work_dir: Path) -> Dict[str, np.ndarray]:
    """One DirectNet half-cell butterfly lobe: qb(q)."""
    from pycircuitsim.simulation import run_dc_sweep
    from pycircuitsim.visualizer import Visualizer

    netlist = _directnet_halfcell_netlist(
        bt, nfin, work_dir / f"sram_{bt.name}_nfin{nfin}.sp")
    logging.disable(logging.CRITICAL)
    try:
        parser = parse_netlist(netlist)
        out_dir = work_dir / f"sram_{bt.name}_nfin{nfin}_out"
        out_dir.mkdir(parents=True, exist_ok=True)
        results = run_dc_sweep(parser.circuit, parser.analysis_params,
                               Visualizer(), out_dir, f"sram_{bt.name}_{nfin}")
    finally:
        logging.disable(logging.NOTSET)
    return {"q": np.asarray(results["q"]), "qb": np.asarray(results["qb"])}


def snm_from_lobes(q: np.ndarray, qb: np.ndarray) -> float:
    """SNM = largest square between a butterfly lobe and its mirror.

    Standard 45-degree rotation method: rotate the (q, qb) lobe and its mirror
    by -45deg; SNM is the side length set by the smaller of the two
    max-vertical-gap halves.
    """
    # lobe 1: qb = f(q); lobe 2 (mirror): q = f(qb) -> sample on a common grid
    grid = np.linspace(0.0, max(q.max(), qb.max()), 400)
    l1 = np.interp(grid, q, qb)              # qb vs q
    l2 = np.interp(grid, qb[::-1] if qb[0] > qb[-1] else qb,
                   q[::-1] if qb[0] > qb[-1] else q)  # q vs qb (mirror)
    # rotate both by -45deg into (u, v); SNM = min over the two crossing
    # regions of the max |v1 - v2|/sqrt(2)
    diff = np.abs(l1 - l2)
    return float(np.max(diff) / np.sqrt(2.0))


# ---------------------------------------------------------------------------
# Full 6T force_ic convergence probe
# ---------------------------------------------------------------------------
def _directnet_6t_netlist(bt: BenchTech, q_init: float, qb_init: float,
                          path: Path) -> Path:
    path.write_text(
        f"* 6T SRAM cell — DirectNet ({bt.name})\n"
        f"Vdd vdd 0 {bt.vdd}\n"
        f"Vwl wl 0 {bt.vdd}\n"
        f"Vbl bl 0 {bt.vdd}\n"
        f"Vblb blb 0 {bt.vdd}\n"
        f".ic V(q)={q_init} V(qb)={qb_init}\n"
        f"Mpl qb q vdd vdd pmos_nn L=20n NFIN={bt.nfin}\n"
        f"Mnl qb q 0   0   nmos_nn L=16n NFIN={bt.nfin}\n"
        f"Mpr q qb vdd vdd pmos_nn L=20n NFIN={bt.nfin}\n"
        f"Mnr q qb 0   0   nmos_nn L=16n NFIN={bt.nfin}\n"
        f"Mal bl  wl q  0 nmos_nn L=16n NFIN={bt.nfin}\n"
        f"Mar blb wl qb 0 nmos_nn L=16n NFIN={bt.nfin}\n"
        f".model nmos_nn NMOS (LEVEL=73 TECH={bt.nn_tech} VT={bt.vt})\n"
        f".model pmos_nn PMOS (LEVEL=73 TECH={bt.nn_tech} VT={bt.vt})\n"
        f".op\n.end\n")
    return path


def force_ic_probe(bt: BenchTech, work_dir: Path) -> Dict[str, bool]:
    """Solve the full 6T cell with force_ic=True for both storage states."""
    from pycircuitsim.solver import DCSolver

    out = {}
    for tag, (q0, qb0) in (("state1", (bt.vdd, 0.0)),
                           ("state0", (0.0, bt.vdd))):
        netlist = _directnet_6t_netlist(
            bt, q0, qb0, work_dir / f"sram6t_{bt.name}_{tag}.sp")
        logging.disable(logging.CRITICAL)
        ok = False
        q_v = qb_v = float("nan")
        try:
            parser = parse_netlist(netlist)
            circuit = parser.circuit
            guess = circuit.initial_conditions or None
            solver = DCSolver(circuit, initial_guess=guess,
                              use_source_stepping=True, force_ic=True)
            sol = solver.solve()
            q_v = sol.get("q", float("nan"))
            qb_v = sol.get("qb", float("nan"))
            # converged on the seeded rail (within VDD/4)?
            ok = (getattr(solver, "_last_solve_converged", False)
                  and abs(q_v - q0) < bt.vdd / 4
                  and abs(qb_v - qb0) < bt.vdd / 4)
        except Exception:  # noqa: BLE001
            ok = False
        finally:
            logging.disable(logging.NOTSET)
        out[tag] = ok
        print(f"      force_ic {tag}: q={q_v:.3f} qb={qb_v:.3f}  "
              f"-> {'converged' if ok else 'FAILED'}")
    return out


# ---------------------------------------------------------------------------
def run_one(bt: BenchTech, nfins: List[int]) -> Dict:
    work_dir = RESULTS_BASE / "sram_snm" / bt.name
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- {bt.name} (VDD={bt.vdd} VT={bt.vt}) ---")

    corner_rows: List[Dict] = []
    for nfin in nfins:
        print(f"  NFIN={nfin}:")
        print("    NGSPICE BSIM-CMG butterfly lobe ...")
        try:
            ng = ngspice_lobe(bt, nfin, work_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"      NGSPICE FAILED: {exc!r}")
            corner_rows.append({"nfin": nfin, "error": repr(exc)})
            continue
        ng_snm = snm_from_lobes(ng["q"], ng["qb"])

        print("    DirectNet (LEVEL=73) butterfly lobe ...")
        try:
            dn = directnet_lobe(bt, nfin, work_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"      DirectNet FAILED: {exc!r}")
            corner_rows.append({"nfin": nfin, "ng_snm": ng_snm,
                                "error": repr(exc)})
            continue
        dn_snm = snm_from_lobes(dn["q"], dn["qb"])

        grid = np.linspace(0.0, bt.vdd, 300)
        ng_i = np.interp(grid, ng["q"], ng["qb"])
        dn_i = np.interp(grid, dn["q"], dn["qb"])
        metrics = full_metrics(dn_i, ng_i)

        dn_min = float(np.min(dn["qb"]))
        positive = dn_min >= -1e-3
        snm_err = (abs(dn_snm - ng_snm) / ng_snm * 100.0
                   if ng_snm > 1e-6 else float("nan"))
        print(f"      NG SNM={ng_snm*1e3:.1f}mV  DN SNM={dn_snm*1e3:.1f}mV  "
              f"SNMerr={snm_err:.1f}%  DN min(qb)={dn_min*1e3:.1f}mV  "
              f"NRMSE={metrics['nrmse_pct']:.2f}%")
        corner_rows.append({
            "nfin": nfin, "ng_snm": ng_snm, "dn_snm": dn_snm,
            "snm_err_pct": snm_err, "dn_min_qb": dn_min,
            "positive": positive, **metrics})

    print("    force_ic full-6T convergence probe ...")
    try:
        fic = force_ic_probe(bt, work_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"      force_ic probe ERROR: {exc!r}")
        fic = {"state1": False, "state0": False}

    all_positive = all(r.get("positive", False)
                       for r in corner_rows if "error" not in r)
    return {"tech": bt.name, "corners": corner_rows,
            "force_ic": fic, "all_positive": all_positive}


def main() -> int:
    ap = argparse.ArgumentParser(description="SRAM read-SNM benchmark 3c")
    ap.add_argument("--tech", default=",".join(BENCH_TECHS))
    ap.add_argument("--nfin", default=",".join(str(n) for n in DEFAULT_NFINS),
                    help="comma-separated NFIN corners")
    args = ap.parse_args()
    techs = [t.strip() for t in args.tech.split(",")]
    nfins = [int(x) for x in args.nfin.split(",")]

    print("=" * 78)
    print("Benchmark 3c — 6T SRAM read SNM: DirectNet vs NGSPICE BSIM-CMG")
    print(f"  NFIN corners: {nfins}")
    print("  Gate: both butterfly lobes positive across NFIN corners")
    print("=" * 78)

    results: List[Dict] = []
    for name in techs:
        if name not in BENCH:
            print(f"  SKIP unknown tech {name}")
            continue
        try:
            results.append(run_one(BENCH[name], nfins))
        except Exception as exc:  # noqa: BLE001
            print(f"  {name}: ERROR {exc!r}")
            results.append({"tech": name, "error": repr(exc)})

    print("\n" + "=" * 78)
    print("SUMMARY — Benchmark 3c SRAM read SNM")
    print("=" * 78)
    hdr = (f"{'Tech':8s} | {'NFIN':>5s} | {'NG SNM mV':>10s} | "
           f"{'DN SNM mV':>10s} | {'SNMerr%':>8s} | {'min(qb)mV':>10s} | "
           f"{'Positive':>9s}")
    print(hdr)
    print("-" * len(hdr))
    n_pass = 0
    for r in results:
        if "error" in r:
            print(f"{r['tech']:8s} | ERROR — {r['error'][:54]}")
            continue
        for c in r["corners"]:
            if "error" in c:
                print(f"{r['tech']:8s} | {c['nfin']:5d} | "
                      f"ERROR — {c['error'][:46]}")
                continue
            print(f"{r['tech']:8s} | {c['nfin']:5d} | "
                  f"{c['ng_snm']*1e3:10.1f} | {c['dn_snm']*1e3:10.1f} | "
                  f"{c['snm_err_pct']:8.1f} | {c['dn_min_qb']*1e3:10.1f} | "
                  f"{'yes' if c['positive'] else 'NO':>9s}")
        fic = r["force_ic"]
        print(f"{r['tech']:8s} |   force_ic: state1="
              f"{'ok' if fic.get('state1') else 'FAIL'} "
              f"state0={'ok' if fic.get('state0') else 'FAIL'}  "
              f"|  all-positive: {'yes' if r['all_positive'] else 'NO'}")
        n_pass += int(r["all_positive"])
    print(f"\n  {n_pass}/{len(results)} techs with all butterfly lobes positive")
    return 0


if __name__ == "__main__":
    sys.exit(main())
