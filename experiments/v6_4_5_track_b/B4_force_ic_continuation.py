#!/usr/bin/env python3
"""B4 — force_ic continuation with shrinking-λ trust region (Track B, Tier 1).

Tests the env-gated soft-pin homotopy (`NN_FORCE_IC_CONTINUATION=1`, implemented
in `pycircuitsim/solver.py:_solve_force_ic_continuation`) against the default
binary hard-pin→unconstrained `force_ic` path on the 6T SRAM cell.

For each tech and each storage state we record the converged (q, qb) and the
rail residual `r = max(|q−q0|, |qb−qb0|)/VDD`. A cell "snaps" when r < 0.05.

Promotion (plan B4): ≥ 1/4 SRAM force_ic cells snap to rails under the
continuation. Hard kill: no λ schedule moves any cell off the interior
attractor → q≈0.18/0.82 is a true model attractor.

Run:
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        conda run -n pycircuitsim python experiments/v6_4_5_track_b/B4_force_ic_continuation.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT,
           PROJECT_ROOT / "external_compact_models",
           PROJECT_ROOT / "external_compact_models" / "PyCMG" / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tests.common.complex import BENCH, RESULTS_BASE, parse_netlist  # noqa: E402
from tests.verify_complex_sram_snm import _directnet_6t_netlist       # noqa: E402

TECHS = ["TSMC7", "TSMC5", "TSMC12", "TSMC16"]
SNAP_GATE = 0.05


def _probe_state(bt, q0: float, qb0: float, tag: str, work_dir: Path) -> Dict:
    from pycircuitsim.solver import DCSolver
    netlist = _directnet_6t_netlist(
        bt, q0, qb0, work_dir / f"sram6t_{bt.name}_{tag}.sp")
    logging.disable(logging.CRITICAL)
    q_v = qb_v = float("nan")
    conv = False
    try:
        parser = parse_netlist(netlist)
        circuit = parser.circuit
        guess = circuit.initial_conditions or None
        solver = DCSolver(circuit, initial_guess=guess,
                          use_source_stepping=True, force_ic=True)
        sol = solver.solve()
        q_v = float(sol.get("q", float("nan")))
        qb_v = float(sol.get("qb", float("nan")))
        conv = bool(getattr(solver, "_last_solve_converged", False))
    except Exception as exc:  # noqa: BLE001
        return {"tag": tag, "error": repr(exc), "q": q_v, "qb": qb_v}
    finally:
        logging.disable(logging.NOTSET)
    resid = max(abs(q_v - q0), abs(qb_v - qb0)) / bt.vdd
    return {"tag": tag, "q": q_v, "qb": qb_v, "resid": resid,
            "snapped": bool(resid < SNAP_GATE), "converged": conv}


def _run_all(continuation: bool) -> List[Dict]:
    os.environ["NN_FORCE_IC_CONTINUATION"] = "1" if continuation else "0"
    rows: List[Dict] = []
    for tname in TECHS:
        bt = BENCH[tname]
        work_dir = RESULTS_BASE / "sram_snm" / bt.name
        work_dir.mkdir(parents=True, exist_ok=True)
        for tag, (q0, qb0) in (("state1", (bt.vdd, 0.0)),
                               ("state0", (0.0, bt.vdd))):
            r = _probe_state(bt, q0, qb0, f"{tag}_cont{int(continuation)}", work_dir)
            r["tech"] = tname
            r["state"] = tag
            rows.append(r)
            print(f"  [{'CONT' if continuation else 'BASE'}] {tname:7s} {tag}: "
                  f"q={r.get('q', float('nan')):.3f} qb={r.get('qb', float('nan')):.3f} "
                  f"resid={r.get('resid', float('nan')):.3f} "
                  f"-> {'SNAP' if r.get('snapped') else 'no-snap'}")
    return rows


def main() -> int:
    res_dir = PROJECT_ROOT / "results" / "v6_4_5_track_b"
    res_dir.mkdir(parents=True, exist_ok=True)
    print("B4 — force_ic continuation vs baseline\n--- baseline (flag OFF) ---")
    base = _run_all(False)
    print("--- continuation (flag ON) ---")
    cont = _run_all(True)

    n_snap_cont = sum(int(r.get("snapped", False)) for r in cont)
    n_snap_base = sum(int(r.get("snapped", False)) for r in base)
    verdict = (
        f"PROMOTE — {n_snap_cont}/8 cells snap under continuation "
        f"(baseline {n_snap_base}/8)"
        if n_snap_cont > n_snap_base and n_snap_cont >= 1 else
        "KILL — continuation does not move any cell to rails; "
        "q≈0.18/0.82 is a true model attractor"
    )
    out = {"snap_gate": SNAP_GATE, "baseline": base, "continuation": cont,
           "n_snap_base": n_snap_base, "n_snap_cont": n_snap_cont,
           "verdict": verdict}
    (res_dir / "B4_force_ic_continuation.json").write_text(json.dumps(out, indent=2))
    print(f"\n  VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
