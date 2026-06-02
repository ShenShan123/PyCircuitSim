#!/usr/bin/env python3
"""V6.4.6 P0-A — force_ic railed-fixed-point EXISTENCE probe (documentation driver).

This is a *thin documenting* reproduction of the decision-critical P0-A probe.
The SOURCE OF TRUTH for the V6.4.6 P0-A numbers is the temporary-instrumentation
run inside ``pycircuitsim/solver.py`` (force_ic cleanup block, gated behind the
``P0A_RESIDUAL`` env var, reverted after measurement) driven by:

    P0A_RESIDUAL=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      conda run -n pycircuitsim python tests/verify_complex_sram_snm.py \
      --tech TSMC5,TSMC7 --nfin 2 > results/v6_4_6/phase0_logs/p0a_residual.log 2>&1

The instrumentation, inserted in ``_solve_newton`` after
``self.initial_guess = voltages.copy()`` / ``matrix_size`` recompute (around
solver.py:945-947) and after the unconstrained re-solve, was::

    import os as _p0a_os
    if _p0a_os.environ.get("P0A_RESIDUAL"):
        _railed = voltages.copy()
        _r, _s = self._dc_residual_at(_railed, node_map, nodes, num_nodes,
                                      matrix_size, self.gmin)
        _thr = max(_RESID_ABS_FLOOR, 100.0 * self.reltol * _s)
        print(f"[P0A] RAILED-seed q={_railed.get('q')} qb={_railed.get('qb')} "
              f"residual_inf={_r:.6e} rhs_scale={_s:.6e} thr={_thr:.6e} "
              f"ratio={_r/(_thr+1e-30):.4e}", flush=True)
    # ... after voltages = self._solve_newton() ...
    if _p0a_os.environ.get("P0A_RESIDUAL"):
        _r2, _s2 = self._dc_residual_at(voltages, node_map, nodes, num_nodes,
                                        matrix_size, self.gmin)
        _thr2 = max(_RESID_ABS_FLOOR, 100.0 * self.reltol * _s2)
        print(f"[P0A] STUCK-final q={voltages.get('q')} qb={voltages.get('qb')} "
              f"residual_inf={_r2:.6e} rhs_scale={_s2:.6e} thr={_thr2:.6e}",
              flush=True)

This script standalone re-derives the *railed-seed* residual without editing the
solver: it builds the 6T force_ic netlist, runs the constrained railed solve by
stamping the .ic nodes itself is NOT trivial, so instead it documents and
re-checks the unconstrained-MNA KCL residual at the LITERAL rail seed
(q=VDD, qb=0) and (q=0, qb=VDD) via the public ``_dc_residual_at`` on a fresh
solver. This is the same quantity the instrumentation printed for RAILED-seed
(the constrained solve lands on the exact rails up to ~1e-9, so the literal-rail
residual matches the instrumented RAILED-seed residual).

Run:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      conda run -n pycircuitsim python scripts/v6_4_6_p0a_force_ic_residual.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "PyCMG" / "tests"))

from tests.common.complex import BENCH, parse_netlist  # noqa: E402
from tests.verify_complex_sram_snm import _directnet_6t_netlist  # noqa: E402
from pycircuitsim.solver import DCSolver, _RESID_ABS_FLOOR  # noqa: E402


def railed_residual(tech: str, work_dir: Path) -> None:
    bt = BENCH[tech]
    print(f"\n=== {tech} (VDD={bt.vdd} NFIN={bt.nfin}) ===")
    for tag, (q0, qb0) in (("state1", (bt.vdd, 0.0)),
                           ("state0", (0.0, bt.vdd))):
        netlist = _directnet_6t_netlist(
            bt, q0, qb0, work_dir / f"p0a_{tech}_{tag}.sp")
        parser = parse_netlist(netlist)
        circuit = parser.circuit
        solver = DCSolver(circuit, initial_guess=None,
                          use_source_stepping=True, force_ic=True)
        # Build the node bookkeeping the residual probe needs.
        nodes = circuit.get_nodes()
        node_map = circuit.get_node_map()
        num_nodes = len(nodes)
        num_vs = circuit.count_voltage_sources()
        matrix_size = num_nodes + num_vs
        rail: Dict[str, float] = {n: 0.0 for n in nodes}
        rail["0"] = 0.0
        rail["vdd"] = bt.vdd
        rail["wl"] = bt.vdd
        rail["bl"] = bt.vdd
        rail["blb"] = bt.vdd
        rail["q"] = q0
        rail["qb"] = qb0
        r, s = solver._dc_residual_at(
            rail, node_map, nodes, num_nodes, matrix_size, solver.gmin)
        thr = max(_RESID_ABS_FLOOR, 100.0 * solver.reltol * s)
        print(f"  {tag}: literal-rail q={q0} qb={qb0} "
              f"residual_inf={r:.6e} rhs_scale={s:.6e} thr={thr:.6e} "
              f"ratio={r/(thr+1e-30):.4e} "
              f"-> {'FIXED-POINT EXISTS' if r <= thr else 'NO RAILED EQUILIBRIUM'}")


def main() -> int:
    work_dir = PROJECT_ROOT / "results" / "v6_4_6" / "phase0_logs" / "p0a_scratch"
    work_dir.mkdir(parents=True, exist_ok=True)
    for tech in ("TSMC5", "TSMC7"):
        railed_residual(tech, work_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
