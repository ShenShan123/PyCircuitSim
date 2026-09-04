"""Shared infrastructure for the DirectNet circuit benchmark harness.

Phase 3 of the DirectNet V6.4 sprint.

Four benchmark circuits exercise DirectNet (LEVEL=73) on larger topologies than
a single inverter:

  3a  verify_circuit_ring_osc.py   — 5-stage CMOS ring oscillator (.tran)
  3b  verify_circuit_opamp.py      — two-stage Miller opamp (.op + .dc)
  3c  verify_circuit_sram_snm.py   — 6T SRAM read SNM butterfly (.dc, force_ic)
  3d  verify_circuit_switchcap.py  — switched-cap unit cell (.tran, PULSE)

Ground truth is ALWAYS NGSPICE BSIM-CMG (LEVEL=72) via the bsimcmg OSDI binary
(AGENTS.md Validation rule — never simplified/self-defined equations).

This module owns the shared plumbing:
  * baked-modelcard generation (NGSPICE side, L/NFIN/TFIN injected),
  * merged-modelcard generation (PyCircuitSim BSIM-CMG side — unused for the
    DirectNet runs, kept for parity / sanity checks),
  * the per-tech benchmark device geometry,
  * NGSPICE batch runner wrappers,
  * metric helpers (NRMSE / MRE / R^2 / MaxErr).

One parameterized file in ``circuit_templates`` owns each topology.
The four verify scripts own orchestration and import from here so they share
one strict renderer, modelcard cache, and NGSPICE path.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Single-thread torch BEFORE any inference (v664 P0): the opamp gain / SRAM
# trip are high-gain fixed points and multi-thread GEMM reduction perturbs the
# NR basin — an unpinned casual run must land the same basin as the CPU-pinned
# gate methodology. PYCIRCUITSIM_TORCH_THREADS overrides so the OMP fragility
# sweep can still probe multistability deliberately.
try:
    import torch

    torch.set_num_threads(int(os.environ.get("PYCIRCUITSIM_TORCH_THREADS", "1")))
except (ImportError, ValueError):  # pragma: no cover
    pass

from tests.common.base import (
    PROJECT_ROOT, OSDI_PATH,
    ALL_TECHS, TechProfile, VtPair,
    bake_inst_params, run_ngspice_subprocess,
)
from tests.common.nn import nrmse as _nrmse_pct, mre as _mre_pct

# ---------------------------------------------------------------------------
# Benchmark technologies with per-technology NN checkpoints. ASAP7 remains
# outside the NN checkpoint scope.
# ---------------------------------------------------------------------------
BENCH_TECHS: List[str] = ["TSMC5", "TSMC6", "TSMC7", "TSMC12", "TSMC16"]
_MODEL_NAMES = {
    73: "DirectNet", 74: "BSIM-AR", 75: "DirectNet-Full",
    76: "BSIM-AR-Full",
}


def _active_model_level() -> int | str:
    """Return the supported campaign force level or fail loudly."""
    raw_level = os.environ.get("PYCIRCUITSIM_NN_FORCE_LEVEL", "73")
    try:
        level = int(raw_level)
    except ValueError as exc:
        raise ValueError(
            f"PYCIRCUITSIM_NN_FORCE_LEVEL={raw_level!r} is not an integer"
        ) from exc
    if level not in _MODEL_NAMES:
        raise ValueError(
            f"unsupported NN model level {level}; supported levels are "
            "73, 74, 75, and 76"
        )
    return level


def active_model_name() -> str:
    """Return the short NN-family name selected by the force-level hook."""
    return _MODEL_NAMES.get(_active_model_level(), "NN")


def active_model_level() -> int:
    """Return the validated NN model level selected for this campaign."""
    return int(_active_model_level())


def active_model_label() -> str:
    """Return the NN family selected by the campaign force-level hook."""
    level = _active_model_level()
    return f"{_MODEL_NAMES.get(level, 'NN')} (LEVEL={level})"

# RESULTS_BASE is env-overridable so parallel sweeps (e.g. the V6.5.4 checkpoint
# bake-off) can give each worker an isolated output dir — otherwise concurrent
# runs of the same (gate, tech) clobber each other's baked modelcards / NGSPICE
# decks. Persistent artifacts belong under the repository results tree.
import os as _os  # noqa: E402
RESULTS_BASE = Path(_os.environ.get(
    "PYCIRCUITSIM_SIMPLE_RESULTS",
    str(PROJECT_ROOT / "results" / "tests" / "simple_circuits"),
))


# ---------------------------------------------------------------------------
# Per-tech benchmark device geometry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BenchTech:
    """Resolved benchmark device geometry for one technology.

    The clean per-technology NN checkpoints use NMOS L=16 nm and PMOS L=20 nm;
    the nominal benchmark pins those geometries while declared corners test
    the additional training-grid points explicitly.
    """
    name: str          # e.g. "TSMC5"
    nn_tech: str        # TECH= netlist parameter, e.g. "tsmc5"
    vt: str            # VT= netlist parameter, e.g. "lvt" (symmetric default)
    vdd: float
    l_nmos: float       # [m]
    l_pmos: float       # [m]
    tfin: float         # [m]
    nfin: int           # default fin count (NMOS fin count)
    nmos_model: str     # NGSPICE/BSIM-CMG model name (modelcard)
    pmos_model: str     # NGSPICE/BSIM-CMG model name (modelcard)
    temperature_c: float = 27.0
    # --- parametric-sweep extensions (default-empty → all existing callers
    #     resolve to the symmetric `vt` / `nfin`, so behaviour is preserved) ---
    nmos_vt: str = ""   # asymmetric NMOS VT (falls back to `vt`)
    pmos_vt: str = ""   # asymmetric PMOS VT (falls back to `vt`)
    nfin_p: int = 0     # PMOS fin count (falls back to `nfin`)

    @property
    def profile(self) -> TechProfile:
        return ALL_TECHS[self.name]

    @property
    def vt_pair(self) -> VtPair:
        return self.profile.get_vt_pair(self.vt)

    @property
    def effective_nmos_vt(self) -> str:
        return self.nmos_vt or self.vt

    @property
    def effective_pmos_vt(self) -> str:
        return self.pmos_vt or self.vt

    @property
    def effective_nfin_p(self) -> int:
        return self.nfin_p or self.nfin


def _resolve_bench_tech(name: str) -> BenchTech:
    """Build a BenchTech from the shared TechProfile + the V6.3.1 checkpoint VT.

    The checkpoint VT must match what the parser preempt cascade resolves; the
    tests.common.nn_gate per-tech table is the source of truth:
      TSMC5 -> lvt, TSMC6 -> ulvt, TSMC7 -> ulvt, TSMC12 -> svt, TSMC16 -> svt.
    """
    ckpt_vt = {"TSMC5": "lvt", "TSMC6": "ulvt", "TSMC7": "ulvt",
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
        nfin=prof.default_nfin,
        nmos_model=vp.nmos_model,
        pmos_model=vp.pmos_model,
        nmos_vt=ckpt_vt,
        pmos_vt=ckpt_vt,
    )


BENCH: Dict[str, BenchTech] = {n: _resolve_bench_tech(n) for n in BENCH_TECHS}


# ---------------------------------------------------------------------------
# Parametric-sweep helpers: usable VT vocabulary + a validated variant builder
# ---------------------------------------------------------------------------
def usable_vts(tech: str) -> Set[str]:
    """VT names usable for ``tech`` in the circuit benchmark sweep.

    The intersection of (i) the BSIM-CMG ground-truth VT pairs that survive
    ``base.py`` PDIBL2/garbage pruning and (ii) the per-tech DirectNet local
    embedding vocabulary (``neural_network.config.LOCAL_VARIANT_CODES``). A VT outside
    the DN vocab maps to LOCAL_UNKNOWN **silently** (no warning in per-tech
    scope), so enumerating from this intersection — and validating in
    ``bench_variant`` — is the only safe way to drive the VT sweep.
    """
    from neural_network.config import LOCAL_VARIANT_CODES  # lazy: heavy import
    scope = tech.lower()
    dn_vocab = LOCAL_VARIANT_CODES.get(scope, {})
    prof = ALL_TECHS[tech]
    return {vp.vt_name for vp in prof.vt_pairs
            if (scope, vp.vt_name.lower()) in dn_vocab}


def bench_variant(base: BenchTech, **overrides: Any) -> BenchTech:
    """Derive a BenchTech variant, resolving NMOS/PMOS VT **independently**.

    Recognised keyword overrides: ``nmos_vt`` / ``pmos_vt`` (each pulls its OWN
    side's modelcard from its OWN VtPair — the single most error-prone line),
    ``l_nmos`` / ``l_pmos``, ``nfin`` / ``nfin_p``, ``vdd``. A VT override is
    validated against the DirectNet local vocab (NOT just ``get_vt_pair``,
    which only knows base.py) and raises ``ValueError`` on a miss so an
    out-of-vocab VT can never silently fall through to LOCAL_UNKNOWN.
    """
    prof = ALL_TECHS[base.name]
    usable = usable_vts(base.name)   # vt_pairs ∩ DN local vocab (both limiters)
    repl: Dict[str, Any] = {}

    nmos_vt = overrides.pop("nmos_vt", None)
    if nmos_vt is not None:
        if nmos_vt.lower() not in usable:
            raise ValueError(
                f"NMOS VT {nmos_vt!r} not usable for {base.name} "
                f"(ground-truth ∩ DN vocab: {sorted(usable)})")
        vp = prof.get_vt_pair(nmos_vt)
        repl["nmos_vt"] = nmos_vt
        repl["nmos_model"] = vp.nmos_model   # NMOS side from its OWN VtPair

    pmos_vt = overrides.pop("pmos_vt", None)
    if pmos_vt is not None:
        if pmos_vt.lower() not in usable:
            raise ValueError(
                f"PMOS VT {pmos_vt!r} not usable for {base.name} "
                f"(ground-truth ∩ DN vocab: {sorted(usable)})")
        vp = prof.get_vt_pair(pmos_vt)
        repl["pmos_vt"] = pmos_vt
        repl["pmos_model"] = vp.pmos_model   # PMOS side from its OWN VtPair

    # Plain geometry / supply overrides pass through unchanged.
    for k in ("l_nmos", "l_pmos", "nfin", "nfin_p", "vdd",
              "temperature_c"):
        if k in overrides:
            repl[k] = overrides.pop(k)
    if overrides:
        raise ValueError(f"bench_variant: unknown overrides {list(overrides)}")
    return replace(base, **repl)


# ---------------------------------------------------------------------------
# Modelcard baking — NGSPICE BSIM-CMG side
# ---------------------------------------------------------------------------
# Cache key carries everything that changes the baked CONTENT: both per-device
# VTs (asymmetric N/P), both per-device L, and both per-device NFIN. The old key
# was (name, vt, nfin) — it dropped L, so two BenchTech variants sharing
# (name, vt, nfin) but differing in L silently aliased to the first one's file
# (the L cache-key bug, plan Step 1).
_baked_cache: Dict[Tuple[str, str, str, float, float, int, int], Path] = {}


def get_baked_modelcard(bt: BenchTech, nfin: int, work_dir: Path,
                        nfin_p: Optional[int] = None) -> Path:
    """Merged NMOS+PMOS modelcard with L/NFIN/TFIN/DEVTYPE baked for NGSPICE.

    NGSPICE's OSDI interface rejects instance params on the device line, so all
    geometry must live inside the .model block. The NMOS block is baked at
    ``nfin``; the PMOS block at ``nfin_p`` (defaults to ``nfin`` — symmetric).
    NMOS/PMOS modelcard SOURCES are resolved from each side's OWN effective VT
    so asymmetric N/P VT pairs bake correct, distinct cards.
    """
    nfin_n = int(nfin)
    nfin_p = nfin_n if nfin_p is None else int(nfin_p)
    key = (bt.name, bt.effective_nmos_vt, bt.effective_pmos_vt,
           bt.l_nmos, bt.l_pmos, nfin_n, nfin_p)
    if key in _baked_cache:
        return _baked_cache[key]

    work_dir.mkdir(parents=True, exist_ok=True)
    prof = bt.profile
    vp_n = prof.get_vt_pair(bt.effective_nmos_vt)
    vp_p = prof.get_vt_pair(bt.effective_pmos_vt)
    nmos_src = prof.get_nmos_modelcard(vp_n, bt.l_nmos)
    pmos_src = prof.get_pmos_modelcard(vp_p, bt.l_pmos)
    if not nmos_src.exists():
        raise FileNotFoundError(f"NMOS modelcard not found: {nmos_src}")
    if not pmos_src.exists():
        raise FileNotFoundError(f"PMOS modelcard not found: {pmos_src}")

    baked = (work_dir / f"baked_{bt.name}_{bt.effective_nmos_vt}_"
             f"{bt.effective_pmos_vt}_ln{bt.l_nmos*1e9:.0f}_"
             f"lp{bt.l_pmos*1e9:.0f}_nf{nfin_n}_nfp{nfin_p}.lib")
    baked.write_text(nmos_src.read_text() + "\n" + pmos_src.read_text())

    bake_inst_params(baked, baked, bt.nmos_model,
                     {"L": bt.l_nmos, "NFIN": float(nfin_n),
                      "TFIN": bt.tfin, "DEVTYPE": 1})
    bake_inst_params(baked, baked, bt.pmos_model,
                     {"L": bt.l_pmos, "NFIN": float(nfin_p),
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
        f"* circuit benchmark — NGSPICE BSIM-CMG ground truth ({tag})\n"
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
# Metric helpers — always report MRE, R^2, NRMSE, MaxErr
# ---------------------------------------------------------------------------
def full_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    """Return MRE(%), R2, NRMSE(% of ptp) and MaxErr.

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
    """One-line metric string."""
    return (f"MRE={m['mre_pct']:.2f}%  R2={m['r2']:.4f}  "
            f"NRMSE={m['nrmse_pct']:.2f}%  "
            f"MaxErr={m['max_err']*err_scale:.2f}{err_unit}")


# ---------------------------------------------------------------------------
# PyCircuitSim DirectNet runner plumbing (shared parse/solve helpers)
# ---------------------------------------------------------------------------
def parse_netlist(netlist_path: Path) -> Any:
    """Parse a PyCircuitSim netlist; return the Parser (caller reads .circuit).

    The DirectNet LEVEL=73 models self-resolve their per-tech checkpoint from
    the netlist's ``TECH=`` / ``VT=`` via the parser preempt cascade — no
    model_name_map / modelcard_path needed.
    """
    from pycircuitsim.parser import Parser
    parser = Parser()
    parser.parse_file(str(netlist_path))
    return parser


def run_directnet_transient(
    netlist_path: Path,
    *,
    parsed: Optional[Any] = None,
) -> Tuple[Dict[str, np.ndarray], bool, str]:
    """Parse + DC-OP + transient-solve a DirectNet netlist.

    Returns ``(results_dict, partial_flag, err_msg)``. On a mid-transient NR
    failure the committed waveform is recovered (partial_flag=True) so the
    harness can still report a numeric gap (fail loud, not silent).
    Mirrors the retry design in tests/common/nn_gate.py / simulation.py.
    """
    import logging
    import numpy as np
    from pycircuitsim.solver import DCSolver, TransientSolver

    logging.disable(logging.CRITICAL)
    partial = False
    err_msg = ""
    try:
        parser = parsed or parse_netlist(netlist_path)
        circuit = parser.circuit
        dt = parser.analysis_params["tstep"]
        t_stop = parser.analysis_params["tstop"]

        guess = circuit.initial_conditions or None

        # uic-equivalent start (V6.4.7 S5b): the NGSPICE side of every
        # simple-circuit benchmark runs `tran ... uic`, integrating from `.ic`
        # exactly. Using `.ic` only as an NR guess lets the OP converge to
        # the model's unconstrained leakage equilibrium (the S5 SC dump
        # measured vsamp(0)=0.39-0.70 V instead of the .ic 0 V), so the two
        # sides simulated different experiments. Pin the .ic nodes with
        # temporary ideal sources for the OP (constrained solution — NOT
        # force_ic, whose released re-solve drifts floating nodes back to
        # the equilibrium) and start the transient from that state.
        from pycircuitsim.models.passive import VoltageSource
        _uic_temps = []
        if circuit.initial_conditions:
            vs_constrained = set()
            for comp in circuit.components:
                if isinstance(comp, VoltageSource):
                    if comp.nodes[1] in ("0", "GND"):
                        vs_constrained.add(comp.nodes[0])
                    elif comp.nodes[0] in ("0", "GND"):
                        vs_constrained.add(comp.nodes[1])
            node_map = circuit.get_node_map()
            for node, val in circuit.initial_conditions.items():
                if (node not in ("0", "GND") and node not in vs_constrained
                        and node in node_map):
                    vs = VoltageSource(f"_V_uic_{node}", [node, "0"], val)
                    circuit.components.append(vs)
                    _uic_temps.append(vs)
            if _uic_temps:
                circuit.invalidate_topology()
        try:
            op = DCSolver(circuit, initial_guess=guess,
                          use_source_stepping=True, use_gmin_stepping=False)
            try:
                op_sol = op.solve()
                if not getattr(op, "_last_solve_converged", True):
                    raise RuntimeError("DC OP fast path did not converge")
            except (RuntimeError, np.linalg.LinAlgError):
                op = DCSolver(circuit, initial_guess=guess,
                              use_source_stepping=True,
                              use_gmin_stepping=True)
                op_sol = op.solve()
        finally:
            for vs in _uic_temps:
                circuit.components.remove(vs)
            if _uic_temps:
                circuit.invalidate_topology()

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
        n_results = len(results["time"])
        for source_name, values in getattr(solver, "source_currents", {}).items():
            current = np.asarray(values)[:n_results]
            if current.size == n_results and np.all(np.isfinite(current)):
                results[f"i({source_name})"] = current.copy()
    finally:
        logging.disable(logging.NOTSET)
    return results, partial, err_msg


def run_directnet_dc_sweep(
    netlist_path: Path,
    work_dir: Path,
    tag: str,
    *,
    require_convergence: bool = True,
    parsed: Optional[Any] = None,
) -> Dict[str, np.ndarray]:
    """Parse + DC-sweep a DirectNet netlist; return the run_dc_sweep results.

    ``require_convergence`` defaults to True: an accuracy gate must never read
    a sweep point that did not reach a physical fixed point.  Diagnostics may
    pass False to recover the numbers behind a convergence failure, and are
    responsible for keeping the resulting row out of any pass/fail aggregate.
    """
    import logging
    from pycircuitsim.simulation import run_dc_sweep
    from pycircuitsim.visualizer import Visualizer

    logging.disable(logging.CRITICAL)
    try:
        parser = parsed or parse_netlist(netlist_path)
        circuit = parser.circuit
        out_dir = work_dir / f"{tag}_dcsweep"
        out_dir.mkdir(parents=True, exist_ok=True)
        results = run_dc_sweep(
            circuit, parser.analysis_params, Visualizer(), out_dir, tag,
            require_convergence=require_convergence,
        )
    finally:
        logging.disable(logging.NOTSET)
    return results


# ===========================================================================
# Parametric sweep harness: stimulus dataclasses, shared measurement helpers,
# and authoritative example-deck adapters.
#
# Every stimulus field defaults to today's single-point value, so a bare
# ``Params()`` reproduces the ship-gate circuit. NGSPICE builders emit device
# lines with NO instance params (OSDI rejects them — all geometry lives in the
# baked .model); NN builders emit per-device ``L=..n NFIN=..`` (the
# PyCircuitSim parser accepts instance params). The two PULSE syntaxes are kept
# deliberately distinct (NGSPICE ``PULSE(...)`` vs PyCircuitSim space-separated).
# ===========================================================================
def _f(x: float) -> str:
    """Compact float formatting matching the template decks (str(float))."""
    return f"{x:g}"


def _cap(c: float) -> str:
    """Farads → 'NNf' femtofarad token (e.g. 20e-15 → '20f')."""
    return f"{c * 1e15:g}f"


def _tn(t: float) -> str:
    """Seconds → 'NNn' nanosecond token (e.g. 4e-9 → '4n')."""
    return f"{t * 1e9:g}n"


def _tp(t: float) -> str:
    """Seconds → 'NNp' picosecond token (e.g. 2e-12 → '2p')."""
    return f"{t * 1e12:g}p"


@dataclass(frozen=True)
class OpAmpParams:
    """Two-stage Miller opamp stimulus. Bias rails are fractions of VDD; the
    defaults reproduce ``verify_circuit_opamp._bias`` exactly (0.55/0.45/0.55)
    and the .dc window (vcm±0.15 @ 0.002)."""
    vcm_frac: float = 0.55
    vbn_frac: float = 0.45
    vbp_frac: float = 0.55
    cc: float = 20e-15
    cl: float = 50e-15
    span: float = 0.15
    step: float = 0.002


@dataclass(frozen=True)
class RingOscParams:
    """N-stage ring oscillator. ``tstop`` is the NGSPICE probe window; the
    transient window is grown from the *measured* period in the orchestrator
    (plan M3) so long corners (n_stages=9, cload=2 fF) stay bounded."""
    n_stages: int = 5
    cload: float = 0.5e-15
    tstep: float = 2e-12
    tstop: float = 1.2e-9
    settle: float = 0.3e-9
    n_measure_cycles: int = 12


@dataclass(frozen=True)
class SwitchCapParams:
    """Switched-cap unit cell. ``pw`` tracks the period (pw_frac·clk_per →
    1.9 ns at the 4 ns baseline) so the sample/hold windows stay valid as
    clk_per sweeps; tstop = n_periods·clk_per."""
    vin_frac: float = 0.6
    csample: float = 100e-15
    clk_per: float = 4e-9
    clk_slew: float = 0.1e-9
    td: float = 0.5e-9
    pw_frac: float = 0.475
    tstep: float = 5e-12
    n_periods: int = 3
    win_margin: float = 0.2e-9


@dataclass(frozen=True)
class SramParams:
    """6T SRAM read-SNM. ``wl_frac`` sets the butterfly read bias (Vwl =
    wl_frac·VDD); force_ic always probes the wl=OFF hold and is gated only at
    wl_frac=1.0 (plan M6)."""
    nfins: Tuple[int, ...] = (2, 5, 10)
    wl_frac: float = 1.0
    dc_step: float = 0.005


# --- shared measurement helpers (moved from the single-point verify scripts) -
def opamp_bias(bt: BenchTech, p: OpAmpParams) -> Tuple[float, float, float]:
    """(Vcm, Vbn, Vbp) — common-mode + the two bias rails, VDD-scaled."""
    return (round(bt.vdd * p.vcm_frac, 3),
            round(bt.vdd * p.vbn_frac, 3),
            round(bt.vdd * p.vbp_frac, 3))


def gain_trip(sweep: np.ndarray, vout: np.ndarray,
              vdd: float) -> Tuple[float, float, float]:
    """Peak |dVout/dVin| gain, trip point (Vout=VDD/2), worst slew step."""
    g = np.gradient(vout, sweep)
    gain = float(np.max(np.abs(g)))
    ix = int(np.argmin(np.abs(vout - vdd / 2.0)))
    trip = float(sweep[ix])
    slew = float(np.max(np.abs(np.diff(vout))))
    return gain, trip, slew


def period_from_wave(t: np.ndarray, v: np.ndarray, mid: float,
                     settle: float) -> float:
    """Oscillation period from rising-edge crossings of the midpoint level."""
    keep = t >= settle
    t, v = t[keep], v[keep]
    sign = np.sign(v - mid)
    cross = np.where((sign[:-1] < 0) & (sign[1:] >= 0))[0]
    if len(cross) < 3:
        return float("nan")
    times = []
    for i in cross:
        v0, v1 = v[i], v[i + 1]
        frac = (mid - v0) / (v1 - v0) if v1 != v0 else 0.0
        times.append(t[i] + frac * (t[i + 1] - t[i]))
    return float(np.mean(np.diff(times)))


def at(t: np.ndarray, v: np.ndarray, t0: float) -> float:
    """Linear-interpolated value of waveform ``v`` at time ``t0``."""
    return float(np.interp(t0, t, v))


def snm_from_lobes(q: np.ndarray, qb: np.ndarray) -> float:
    """SNM = largest square between a butterfly lobe and its mirror."""
    grid = np.linspace(0.0, max(q.max(), qb.max()), 400)
    l1 = np.interp(grid, q, qb)
    l2 = np.interp(grid, qb[::-1] if qb[0] > qb[-1] else qb,
                   q[::-1] if qb[0] > qb[-1] else q)
    diff = np.abs(l1 - l2)
    return float(np.max(diff) / np.sqrt(2.0))


def sc_windows(p: SwitchCapParams) -> Tuple[float, float, float, float,
                                            float, float]:
    """(sample_end, hold_start, hold_end, tstop, pw, per) from the clock.

    PULSE(0 VDD td tr tf pw per): sample phase high on [td+tr, td+tr+pw];
    hold phase on [td+tr+pw+tf, td+per]. Windows are pulled ``win_margin``
    inside each edge. Reproduces the single-point 2.3/2.6/4.3/12 ns at the
    4 ns/1.9 ns baseline.
    """
    tr = tf = p.clk_slew
    per = p.clk_per
    pw = p.pw_frac * per
    sample_high_end = p.td + tr + pw
    hold_start = p.td + tr + pw + tf
    next_rise = p.td + per
    m = p.win_margin
    sample_end = sample_high_end - m
    hold_end = next_rise - m
    tstop = p.n_periods * per
    return sample_end, hold_start, hold_end, tstop, pw, per


# --- authoritative example adapters: opamp ----------------------------------
def _catalog_pair(
    case_id: str,
    bt: BenchTech,
    baked: Path,
    *,
    analysis_card: str,
    substitutions: Optional[Dict[str, str]] = None,
    ring_n_stages: int = 5,
    ring_cload: float = 0.5e-15,
) -> Tuple[str, str]:
    """Render both adapters from one authoritative example template."""
    from tests.common.simple_circuit_catalog import get_case
    from tests.common.simple_circuit_harness import (
        CORNERS, render_case_decks,
    )

    case = get_case(case_id)
    analysis = replace(case.analyses[0], card=analysis_card)
    return render_case_decks(
        case, analysis, bt, CORNERS["nominal"], baked_lib=baked,
        substitutions=substitutions, ring_n_stages=ring_n_stages,
        ring_cload=ring_cload,
    )


def _catalog_body(deck: str) -> str:
    """Drop title and .end before embedding a rendered reference in control."""
    return "\n".join(
        line for line in deck.splitlines()
        if line.strip().lower() != ".end"
        and not line.lstrip().startswith("*")
    )


def ngspice_opamp(bt: BenchTech, p: OpAmpParams, baked: Path) -> Dict[str, str]:
    vcm, vbn, vbp = opamp_bias(bt, p)
    lo, hi = round(vcm - p.span, 3), round(vcm + p.span, 3)
    card = f"dc Vinp {_f(lo)} {_f(hi)} {_f(p.step)}"
    _, reference = _catalog_pair(
        "opamp", bt, baked, analysis_card=card,
        substitutions={
            "VCM": _f(vcm), "VBN": _f(vbn), "VBP": _f(vbp),
            "CC": _cap(p.cc), "CL": _cap(p.cl),
            "OPAMP_LO": _f(lo), "OPAMP_HI": _f(hi),
        },
    )
    return {"body": _catalog_body(reference), "signals": "v(vout)",
            "analysis": card}


def directnet_opamp(bt: BenchTech, p: OpAmpParams) -> str:
    vcm, vbn, vbp = opamp_bias(bt, p)
    lo, hi = round(vcm - p.span, 3), round(vcm + p.span, 3)
    card = f"dc Vinp {_f(lo)} {_f(hi)} {_f(p.step)}"
    candidate, _ = _catalog_pair(
        "opamp", bt, Path("<unused>"), analysis_card=card,
        substitutions={
            "VCM": _f(vcm), "VBN": _f(vbn), "VBP": _f(vbp),
            "CC": _cap(p.cc), "CL": _cap(p.cl),
            "OPAMP_LO": _f(lo), "OPAMP_HI": _f(hi),
        },
    )
    return candidate


# --- authoritative example adapters: ring oscillator ------------------------
def ngspice_ringosc(bt: BenchTech, p: RingOscParams, baked: Path,
                    tstop: float) -> Dict[str, str]:
    card = f"tran {_tp(p.tstep)} {_tn(tstop)} uic"
    _, reference = _catalog_pair(
        "ring_osc", bt, baked, analysis_card=card,
        ring_n_stages=p.n_stages,
        ring_cload=p.cload,
    )
    return {"body": _catalog_body(reference),
            "signals": f"v(n{p.n_stages})", "analysis": card}


def directnet_ringosc(bt: BenchTech, p: RingOscParams, tstop: float) -> str:
    card = f"tran {_tp(p.tstep)} {_tn(tstop)} uic"
    candidate, _ = _catalog_pair(
        "ring_osc", bt, Path("<unused>"), analysis_card=card,
        ring_n_stages=p.n_stages,
        ring_cload=p.cload,
    )
    return candidate


# --- authoritative example adapters: switched-cap unit cell -----------------
def ngspice_switchcap(bt: BenchTech, p: SwitchCapParams,
                      baked: Path) -> Dict[str, str]:
    _se, _hs, _he, tstop, pw, per = sc_windows(p)
    vin = round(bt.vdd * p.vin_frac, 3)
    card = f"tran {_tp(p.tstep)} {_tn(tstop)} uic"
    values = {
        "VIN": _f(vin), "TD": _tn(p.td), "SLEW": _tn(p.clk_slew),
        "PW": _tn(pw), "PER": _tn(per), "CSAMPLE": _cap(p.csample),
    }
    _, reference = _catalog_pair(
        "switchcap", bt, baked, analysis_card=card, substitutions=values,
    )
    return {"body": _catalog_body(reference), "signals": "v(vsamp)",
            "analysis": card}


def directnet_switchcap(bt: BenchTech, p: SwitchCapParams) -> str:
    _se, _hs, _he, tstop, pw, per = sc_windows(p)
    vin = round(bt.vdd * p.vin_frac, 3)
    card = f"tran {_tp(p.tstep)} {_tn(tstop)} uic"
    candidate, _ = _catalog_pair(
        "switchcap", bt, Path("<unused>"), analysis_card=card,
        substitutions={
            "VIN": _f(vin), "TD": _tn(p.td),
            "SLEW": _tn(p.clk_slew), "PW": _tn(pw), "PER": _tn(per),
            "CSAMPLE": _cap(p.csample),
        },
    )
    return candidate


# --- authoritative example adapters: SRAM lobe + full-cell force_ic ----------
def ngspice_sram_lobe(bt: BenchTech, p: SramParams, nfin: int,
                      baked: Path) -> Dict[str, str]:
    wl = round(bt.vdd * p.wl_frac, 3)
    render_bt = bench_variant(bt, nfin=nfin, nfin_p=nfin)
    card = f"dc Vq 0 {_f(bt.vdd)} {_f(p.dc_step)}"
    _, reference = _catalog_pair(
        "sram_snm", render_bt, baked, analysis_card=card,
        substitutions={"WL": _f(wl)},
    )
    return {"body": _catalog_body(reference), "signals": "v(qb)",
            "analysis": card}


def directnet_sram_lobe(bt: BenchTech, p: SramParams, nfin: int) -> str:
    wl = round(bt.vdd * p.wl_frac, 3)
    render_bt = bench_variant(bt, nfin=nfin, nfin_p=nfin)
    card = f"dc Vq 0 {_f(bt.vdd)} {_f(p.dc_step)}"
    candidate, _ = _catalog_pair(
        "sram_snm", render_bt, Path("<unused>"),
        analysis_card=card,
        substitutions={"WL": _f(wl)},
    )
    return candidate


def directnet_sram_6t(bt: BenchTech, q0: float, qb0: float, nfin: int) -> str:
    """Full cross-coupled 6T cell, wl=OFF/hold (the force_ic retention probe)."""
    render_bt = bench_variant(bt, nfin=nfin, nfin_p=nfin)
    candidate, _ = _catalog_pair(
        "sram6t_modes", render_bt, Path("<unused>"),
        analysis_card="op",
        substitutions={
            "WL_SPEC": "0", "BL_SPEC": _f(bt.vdd),
            "BLB_SPEC": _f(bt.vdd), "Q_IC": _f(q0),
            "QB_IC": _f(qb0),
        },
    )
    return candidate
