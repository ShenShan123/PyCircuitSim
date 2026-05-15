"""Shared infrastructure for the DirectNet complex-circuit benchmark harness.

Phase 3 of the DirectNet V6.4 sprint
(``docs/plans/2026-05-15-directnet-complex-circuits.md``).

Four benchmark circuits exercise DirectNet (LEVEL=73) on larger topologies than
a single inverter:

  3a  verify_complex_ring_osc.py   — 5-stage CMOS ring oscillator (.tran)
  3b  verify_complex_opamp.py      — two-stage Miller opamp (.op + .dc)
  3c  verify_complex_sram_snm.py   — 6T SRAM read SNM butterfly (.dc, force_ic)
  3d  verify_complex_switchcap.py  — switched-cap unit cell (.tran, PULSE)

Ground truth is ALWAYS NGSPICE BSIM-CMG (LEVEL=72) via the bsimcmg OSDI binary
(CLAUDE.md Validation rule — never simplified/self-defined equations).

This module owns the shared plumbing:
  * baked-modelcard generation (NGSPICE side, L/NFIN/TFIN injected),
  * merged-modelcard generation (PyCircuitSim BSIM-CMG side — unused for the
    DirectNet runs, kept for parity / sanity checks),
  * the per-tech benchmark device geometry,
  * NGSPICE batch runner wrappers,
  * metric helpers (NRMSE / MRE / R^2 / MaxErr).

The four verify scripts own their own netlist text and orchestration; they
import from here so all four share one modelcard cache and one NGSPICE path.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from tests.common.base import (
    PROJECT_ROOT, OSDI_PATH, NGSPICE_BIN,
    ALL_TECHS, TechProfile, VtPair,
    bake_inst_params, run_ngspice_subprocess,
)
from tests.common.nn import nrmse as _nrmse_pct, mre as _mre_pct

# ---------------------------------------------------------------------------
# Benchmark techs — only the four with V6.3.1 DirectNet checkpoints.
# ASAP7 is out of scope (Rule 17); LEVEL=74 BSIMAR out of scope (Rule 18).
# ---------------------------------------------------------------------------
BENCH_TECHS: List[str] = ["TSMC5", "TSMC7", "TSMC12", "TSMC16"]

REFERENCES_DIR = PROJECT_ROOT / "tests" / "references" / "complex"
RESULTS_BASE = PROJECT_ROOT / "tests" / "verify_complex_results"


# ---------------------------------------------------------------------------
# Per-tech benchmark device geometry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BenchTech:
    """Resolved benchmark device geometry for one technology.

    DirectNet per-tech checkpoints (``tsmc{5,7,12,16}_dn_medium_{nmos,pmos}``)
    are trained for NMOS L=16nm / PMOS L=20nm; the benchmark circuits pin those
    geometries so the model interpolates rather than extrapolates.
    """
    name: str          # e.g. "TSMC5"
    nn_tech: str        # TECH= netlist parameter, e.g. "tsmc5"
    vt: str            # VT= netlist parameter, e.g. "lvt"
    vdd: float
    l_nmos: float       # [m]
    l_pmos: float       # [m]
    tfin: float         # [m]
    nfin: int           # default fin count
    nmos_model: str     # NGSPICE/BSIM-CMG model name (modelcard)
    pmos_model: str     # NGSPICE/BSIM-CMG model name (modelcard)

    @property
    def profile(self) -> TechProfile:
        return ALL_TECHS[self.name]

    @property
    def vt_pair(self) -> VtPair:
        return self.profile.get_vt_pair(self.vt)


def _resolve_bench_tech(name: str) -> BenchTech:
    """Build a BenchTech from the shared TechProfile + the V6.3.1 checkpoint VT.

    The checkpoint VT must match what the parser preempt cascade resolves; the
    verify_nn_dc_tran.py per-tech table is the source of truth:
      TSMC5 -> lvt, TSMC7 -> ulvt, TSMC12 -> svt, TSMC16 -> svt.
    """
    ckpt_vt = {"TSMC5": "lvt", "TSMC7": "ulvt",
               "TSMC12": "svt", "TSMC16": "svt"}[name]
    prof = ALL_TECHS[name]
    vp = prof.get_vt_pair(ckpt_vt)
    return BenchTech(
        name=name,
        nn_tech=name.lower(),
        vt=ckpt_vt,
        vdd=prof.vdd,
        l_nmos=16e-9,
        l_pmos=20e-9,
        tfin=prof.tfin,
        nfin=prof.default_nfin,   # 2 for all four TSMC nodes
        nmos_model=vp.nmos_model,
        pmos_model=vp.pmos_model,
    )


BENCH: Dict[str, BenchTech] = {n: _resolve_bench_tech(n) for n in BENCH_TECHS}


# ---------------------------------------------------------------------------
# Modelcard baking — NGSPICE BSIM-CMG side
# ---------------------------------------------------------------------------
_baked_cache: Dict[Tuple[str, str, int], Path] = {}


def get_baked_modelcard(bt: BenchTech, nfin: int, work_dir: Path) -> Path:
    """Merged NMOS+PMOS modelcard with L/NFIN/TFIN/DEVTYPE baked for NGSPICE.

    NGSPICE's OSDI interface rejects instance params on the device line, so all
    geometry must live inside the .model block. NFIN is baked per call so SRAM
    NFIN-corner sweeps get distinct files.
    """
    key = (bt.name, bt.vt, nfin)
    if key in _baked_cache:
        return _baked_cache[key]

    work_dir.mkdir(parents=True, exist_ok=True)
    prof = bt.profile
    vp = bt.vt_pair
    nmos_src = prof.get_nmos_modelcard(vp, bt.l_nmos)
    pmos_src = prof.get_pmos_modelcard(vp, bt.l_pmos)
    if not nmos_src.exists():
        raise FileNotFoundError(f"NMOS modelcard not found: {nmos_src}")
    if not pmos_src.exists():
        raise FileNotFoundError(f"PMOS modelcard not found: {pmos_src}")

    baked = work_dir / f"baked_{bt.name}_{bt.vt}_nfin{nfin}.lib"
    baked.write_text(nmos_src.read_text() + "\n" + pmos_src.read_text())

    bake_inst_params(baked, baked, bt.nmos_model,
                     {"L": bt.l_nmos, "NFIN": float(nfin),
                      "TFIN": bt.tfin, "DEVTYPE": 1})
    bake_inst_params(baked, baked, bt.pmos_model,
                     {"L": bt.l_pmos, "NFIN": float(nfin),
                      "TFIN": bt.tfin, "DEVTYPE": 0})

    _baked_cache[key] = baked
    return baked


# ---------------------------------------------------------------------------
# NGSPICE batch runners
# ---------------------------------------------------------------------------
def run_ngspice_wrdata(
    netlist_text: str,
    wrdata_signals: str,
    work_dir: Path,
    tag: str,
    analysis_block: str,
) -> np.ndarray:
    """Run an NGSPICE batch job and return the parsed wrdata matrix.

    Parameters
    ----------
    netlist_text : circuit body (no .control / .end) — devices, sources, .ic,
        .include of the baked modelcard.
    wrdata_signals : space-separated vector list for ``wrdata``.
    analysis_block : the analysis statement(s) inside .control, e.g.
        ``"tran 1p 20n uic"`` or ``"dc Vin 0 0.8 0.005"``.

    Returns the raw wrdata matrix; column 0 is the sweep/time axis, subsequent
    columns interleave (axis, value) per requested vector (NGSPICE wrdata
    convention).
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    cir = work_dir / f"ngspice_{tag}.cir"
    runner = work_dir / f"ngspice_{tag}_runner.cir"
    csv = work_dir / f"ngspice_{tag}.csv"
    log = work_dir / f"ngspice_{tag}.log"

    # Circuit body lives in its own deck (device lines + .include + .ic are
    # invalid inside a .control block); the runner sources it.
    cir.write_text(
        f"* complex-circuit benchmark — NGSPICE BSIM-CMG ground truth ({tag})\n"
        f"{netlist_text}\n"
        f".end\n"
    )
    runner.write_text(
        f"* NGSPICE runner ({tag})\n"
        f".control\n"
        f"osdi {OSDI_PATH}\n"
        f"source {cir}\n"
        f"set filetype=ascii\n"
        f"set wr_vecnames\n"
        f"{analysis_block}\n"
        f"wrdata {csv} {wrdata_signals}\n"
        f".endc\n"
        f".end\n"
    )

    lines = run_ngspice_subprocess(runner, log, csv)

    rows: List[List[float]] = []
    for line in lines[1:]:
        s = line.strip()
        if s:
            rows.append([float(x) for x in s.split()])
    data = np.array(rows)
    if data.size == 0:
        raise RuntimeError(f"NGSPICE produced no data rows ({tag})")
    if not np.all(np.isfinite(data)):
        raise RuntimeError(f"NGSPICE output contains NaN/Inf ({tag})")
    return data


# ---------------------------------------------------------------------------
# Metric helpers — Rule 16: always report MRE, R^2, NRMSE, MaxErr
# ---------------------------------------------------------------------------
def full_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    """Return MRE(%), R2, NRMSE(% of ptp) and MaxErr — Rule 16 quartet.

    ``pred`` / ``true`` must already be on a common grid.
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    diff = pred - true
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 1.0
    return {
        "mre_pct": _mre_pct(pred, true),
        "r2": r2,
        "nrmse_pct": _nrmse_pct(pred, true),
        "max_err": float(np.max(np.abs(diff))),
    }


def fmt_metrics(m: Dict[str, float], err_scale: float = 1e3,
                err_unit: str = "mV") -> str:
    """One-line Rule-16 metric string."""
    return (f"MRE={m['mre_pct']:.2f}%  R2={m['r2']:.4f}  "
            f"NRMSE={m['nrmse_pct']:.2f}%  "
            f"MaxErr={m['max_err']*err_scale:.2f}{err_unit}")


# ---------------------------------------------------------------------------
# PyCircuitSim DirectNet runner plumbing (shared parse/solve helpers)
# ---------------------------------------------------------------------------
def parse_netlist(netlist_path: Path):
    """Parse a PyCircuitSim netlist; return the Parser (caller reads .circuit).

    The DirectNet LEVEL=73 models self-resolve their per-tech checkpoint from
    the netlist's ``TECH=`` / ``VT=`` via the parser preempt cascade — no
    model_name_map / modelcard_path needed.
    """
    from pycircuitsim.parser import Parser
    parser = Parser()
    parser.parse_file(str(netlist_path))
    return parser


def render_directnet_netlist(template_path: Path, bt: BenchTech,
                             out_path: Path) -> Path:
    """Write a per-tech DirectNet netlist from an examples/ template.

    The examples/complex/*_directnet.sp files carry TSMC12/svt/0.80 V
    placeholders; this swaps in the benchmark tech's TECH= / VT= / VDD so the
    parser preempt cascade resolves the right ``tsmc{X}_dn_medium`` checkpoint.
    """
    text = template_path.read_text()
    text = text.replace("TECH=tsmc12", f"TECH={bt.nn_tech}")
    text = text.replace("VT=svt", f"VT={bt.vt}")
    # supply-voltage lines and .ic rails written at 0.80 in the templates
    text = text.replace("Vdd vdd 0 0.80", f"Vdd vdd 0 {bt.vdd}")
    text = text.replace("=0.80", f"={bt.vdd}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    return out_path


def run_directnet_transient(netlist_path: Path):
    """Parse + DC-OP + transient-solve a DirectNet netlist.

    Returns ``(results_dict, partial_flag, err_msg)``. On a mid-transient NR
    failure the committed waveform is recovered (partial_flag=True) so the
    harness can still report a numeric gap (Rule 12 — fail loud, not silent).
    Mirrors the retry design in verify_nn_dc_tran.py / simulation.py.
    """
    import logging
    import numpy as np
    from pycircuitsim.solver import DCSolver, TransientSolver

    logging.disable(logging.CRITICAL)
    partial = False
    err_msg = ""
    try:
        parser = parse_netlist(netlist_path)
        circuit = parser.circuit
        dt = parser.analysis_params["tstep"]
        t_stop = parser.analysis_params["tstop"]

        guess = circuit.initial_conditions or None
        op = DCSolver(circuit, initial_guess=guess, use_source_stepping=True,
                      use_gmin_stepping=False)
        try:
            op_sol = op.solve()
            if not getattr(op, "_last_solve_converged", True):
                raise RuntimeError("DC OP fast path did not converge")
        except (RuntimeError, np.linalg.LinAlgError):
            op = DCSolver(circuit, initial_guess=guess,
                          use_source_stepping=True, use_gmin_stepping=True)
            op_sol = op.solve()

        solver = TransientSolver(
            circuit, t_stop=t_stop, dt=dt, initial_guess=op_sol,
            use_gmin_stepping=True, gmin_initial=1e-9, gmin_final=1e-12,
            gmin_steps=5, use_pseudo_transient=True, pseudo_transient_steps=5,
            pseudo_transient_cap=1e-12, debug=False, nr_tolerance=1e-7,
        )
        try:
            results = solver.solve()
        except RuntimeError as exc:
            err_msg = str(exc)
            last = getattr(solver, "_last_committed_step", 0)
            if last >= 2:
                partial = True
                n = last + 1
                results = {"time": solver._partial_time[:n].copy()}
                for node, arr in solver._partial_voltages.items():
                    results[node] = arr[:n].copy()
            else:
                raise
    finally:
        logging.disable(logging.NOTSET)
    return results, partial, err_msg


def run_directnet_dc_sweep(netlist_path: Path, work_dir: Path, tag: str):
    """Parse + DC-sweep a DirectNet netlist; return the run_dc_sweep results."""
    import logging
    from pycircuitsim.simulation import run_dc_sweep
    from pycircuitsim.visualizer import Visualizer

    logging.disable(logging.CRITICAL)
    try:
        parser = parse_netlist(netlist_path)
        circuit = parser.circuit
        out_dir = work_dir / f"{tag}_dcsweep"
        out_dir.mkdir(parents=True, exist_ok=True)
        results = run_dc_sweep(circuit, parser.analysis_params,
                               Visualizer(), out_dir, tag)
    finally:
        logging.disable(logging.NOTSET)
    return results
