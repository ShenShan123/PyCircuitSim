"""Shared infrastructure for the DirectNet complex-circuit benchmark harness.

Phase 3 of the DirectNet V6.4 sprint.

Four benchmark circuits exercise DirectNet (LEVEL=73) on larger topologies than
a single inverter:

  3a  verify_complex_ring_osc.py   — 5-stage CMOS ring oscillator (.tran)
  3b  verify_complex_opamp.py      — two-stage Miller opamp (.op + .dc)
  3c  verify_complex_sram_snm.py   — 6T SRAM read SNM butterfly (.dc, force_ic)
  3d  verify_complex_switchcap.py  — switched-cap unit cell (.tran, PULSE)

Ground truth is ALWAYS NGSPICE BSIM-CMG (LEVEL=72) via the bsimcmg OSDI binary
(AGENTS.md Validation rule — never simplified/self-defined equations).

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

import os
import re
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
    PROJECT_ROOT, OSDI_PATH, NGSPICE_BIN,
    ALL_TECHS, TechProfile, VtPair,
    bake_inst_params, run_ngspice_subprocess,
)
from tests.common.nn import nrmse as _nrmse_pct, mre as _mre_pct

# ---------------------------------------------------------------------------
# Benchmark techs — only the four with V6.3.1 DirectNet checkpoints.
# ASAP7 is out of scope; LEVEL=74 BSIMAR out of scope.
# ---------------------------------------------------------------------------
BENCH_TECHS: List[str] = ["TSMC5", "TSMC6", "TSMC7", "TSMC12", "TSMC16"]
_MODEL_NAMES = {73: "DirectNet", 74: "BSIM-AR"}


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
            f"unsupported NN model level {level}; supported levels are 73 and 74"
        )
    return level


def active_model_name() -> str:
    """Return the short NN-family name selected by the force-level hook."""
    return _MODEL_NAMES.get(_active_model_level(), "NN")


def active_model_label() -> str:
    """Return the NN family selected by the campaign force-level hook."""
    level = _active_model_level()
    return f"{_MODEL_NAMES.get(level, 'NN')} (LEVEL={level})"

# RESULTS_BASE is env-overridable so parallel sweeps (e.g. the V6.5.4 checkpoint
# bake-off) can give each worker an isolated output dir — otherwise concurrent
# runs of the same (gate, tech) clobber each other's baked modelcards / NGSPICE
# decks. Default = the committed in-tree path (unchanged for normal runs).
import os as _os  # noqa: E402
RESULTS_BASE = Path(_os.environ.get(
    "PYCIRCUITSIM_COMPLEX_RESULTS",
    str(PROJECT_ROOT / "tests" / "verify_complex_results")))


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
    vt: str            # VT= netlist parameter, e.g. "lvt" (symmetric default)
    vdd: float
    l_nmos: float       # [m]
    l_pmos: float       # [m]
    tfin: float         # [m]
    nfin: int           # default fin count (NMOS fin count)
    nmos_model: str     # NGSPICE/BSIM-CMG model name (modelcard)
    pmos_model: str     # NGSPICE/BSIM-CMG model name (modelcard)
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
        nfin=prof.default_nfin,   # 2 for all four TSMC nodes
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
    """VT names usable for ``tech`` in the complex-circuit sweep.

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
    for k in ("l_nmos", "l_pmos", "nfin", "nfin_p", "vdd"):
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


def render_directnet_text(template_text: str, bt: BenchTech) -> str:
    """Per-tech rewrite of a DirectNet template's text (TECH/VT/VDD).

    Pure (string-in, string-out) so the single-point verify scripts can build
    their ship-gate deck text WITHOUT touching the filesystem, and the
    equivalence canary (verify_complex_sweep_canaries) can diff that exact text
    against the sweep builders. ``render_directnet_netlist`` is the file-writing
    wrapper around this.
    """
    text = template_text.replace("TECH=tsmc12", f"TECH={bt.nn_tech}")
    text = text.replace("VT=svt", f"VT={bt.vt}")
    # Rescale EVERY 0.80 V rail to the tech VDD. The templates are authored at
    # the TSMC12/16 rail (0.80 V); for the 0.65/0.75 V nodes every standalone
    # 0.80 token is a supply rail — the Vdd line, the PULSE clock amplitude
    # (`PULSE 0 0.80`), SRAM word/bit lines (`Vwl wl 0 0.80`), and the `.ic`
    # rails (`=0.80`). The earlier targeted `Vdd vdd 0 0.80` / `=0.80` replaces
    # MISSED the space-delimited clock/rail values, so on tsmc5/tsmc7 the
    # DirectNet clock over-drove the pass gates to 0.80 V while NGSPICE clocked
    # to VDD — different experiments (the tsmc5 switchcap "11.8% over-charge").
    # Geometry (`L=16n`, `NFIN=2`) carries no 0.80 token, so a whole-number
    # match is safe. A no-op on tsmc12/16 (0.80 -> 0.80).
    return re.sub(r"(?<![\w.])0\.80(?![\w])", f"{bt.vdd}", text)


def render_directnet_netlist(template_path: Path, bt: BenchTech,
                             out_path: Path) -> Path:
    """Write a per-tech DirectNet netlist from an examples/ template.

    The examples/simple_circuits/directnet_*.sp files carry TSMC12/svt/0.80 V
    placeholders; this swaps in the benchmark tech's TECH= / VT= / VDD so the
    parser preempt cascade resolves the right ``tsmc{X}_dn_medium`` checkpoint.
    """
    text = render_directnet_text(template_path.read_text(), bt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    return out_path


def run_directnet_transient(netlist_path: Path):
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
        parser = parse_netlist(netlist_path)
        circuit = parser.circuit
        dt = parser.analysis_params["tstep"]
        t_stop = parser.analysis_params["tstop"]

        guess = circuit.initial_conditions or None

        # uic-equivalent start (V6.4.7 S5b): the NGSPICE side of every
        # complex benchmark runs `tran ... uic`, integrating from `.ic`
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
        results = run_dc_sweep(
            circuit, parser.analysis_params, Visualizer(), out_dir, tag,
            require_convergence=True,
        )
    finally:
        logging.disable(logging.NOTSET)
    return results


# ===========================================================================
# Parametric sweep harness (plan 2026-06-20): stimulus dataclasses (Step 3),
# shared measurement helpers + programmatic builders (Step 4).
#
# Every stimulus field defaults to today's single-point value, so a bare
# ``Params()`` reproduces the ship-gate circuit. NGSPICE builders emit device
# lines with NO instance params (OSDI rejects them — all geometry lives in the
# baked .model); DirectNet builders emit per-device ``L=..n NFIN=..`` (the
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
    defaults reproduce ``verify_complex_opamp._bias`` exactly (0.55/0.45/0.55)
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
    storage_states: Tuple[str, ...] = ("state1", "state0")
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


# --- programmatic builders: opamp -------------------------------------------
def ngspice_opamp(bt: BenchTech, p: OpAmpParams, baked: Path) -> Dict[str, str]:
    vcm, vbn, vbp = opamp_bias(bt, p)
    n, pm = bt.nmos_model, bt.pmos_model
    body = [f'.include "{baked}"', ".temp 27", f"Vdd vdd 0 {_f(bt.vdd)}",
            f"Vbn vbn 0 {_f(vbn)}", f"Vbp vbp 0 {_f(vbp)}",
            f"Vinn inn 0 {_f(vcm)}", f"Vinp inp 0 {_f(vcm)}",
            f"Nn1 n1 inp vtail 0 {n}", f"Nn2 vo1i inn vtail 0 {n}",
            f"Np3 n1 n1 vdd vdd {pm}", f"Np4 vo1i n1 vdd vdd {pm}",
            f"Nn5 vtail vbn 0 0 {n}",
            f"Np6 vout vo1i vdd vdd {pm}", f"Nn7 vout vbn 0 0 {n}",
            f"Cc vo1i vout {_cap(p.cc)}", f"CL vout 0 {_cap(p.cl)}"]
    lo, hi = round(vcm - p.span, 3), round(vcm + p.span, 3)
    return {"body": "\n".join(body), "signals": "v(vout)",
            "analysis": f"dc Vinp {_f(lo)} {_f(hi)} {_f(p.step)}"}


def directnet_opamp(bt: BenchTech, p: OpAmpParams) -> str:
    vcm, vbn, vbp = opamp_bias(bt, p)
    ln, lp = bt.l_nmos * 1e9, bt.l_pmos * 1e9
    nfn, nfp = bt.nfin, bt.effective_nfin_p
    lo, hi = round(vcm - p.span, 3), round(vcm + p.span, 3)
    # Hierarchical deck (V6.12.0): the two opamp stages are .subckt
    # instances. Every pre-existing node (n1/vo1i/vtail/vout) stays a port,
    # so the flattened net names — and every harness/diagnostic probe — are
    # byte-identical to the historical flat deck.
    return (
        f"* Two-stage Miller opamp — DirectNet LEVEL=73 ({bt.name}) [sweep]\n"
        f"Vdd vdd 0 {_f(bt.vdd)}\n"
        f"Vbn vbn 0 {_f(vbn)}\n"
        f"Vbp vbp 0 {_f(vbp)}\n"
        f"Vinn inn 0 {_f(vcm)}\n"
        f"Vinp inp 0 {_f(vcm)}\n"
        f"Xs1 inp inn n1 vo1i vtail vbn vdd ota1\n"
        f"Xs2 vo1i vout vbn vdd cs2\n"
        f"Cc vo1i vout {_cap(p.cc)}\n"
        f"CL vout 0 {_cap(p.cl)}\n"
        f".subckt ota1 inp inn n1 vo1i vtail vbn vdd\n"
        f"Mn1 n1   inp vtail 0   nmos_nn L={ln:.0f}n NFIN={nfn}\n"
        f"Mn2 vo1i inn vtail 0   nmos_nn L={ln:.0f}n NFIN={nfn}\n"
        f"Mp3 n1   n1  vdd   vdd pmos_nn L={lp:.0f}n NFIN={nfp}\n"
        f"Mp4 vo1i n1  vdd   vdd pmos_nn L={lp:.0f}n NFIN={nfp}\n"
        f"Mn5 vtail vbn 0    0   nmos_nn L={ln:.0f}n NFIN={nfn}\n"
        f".ends\n"
        f".subckt cs2 g d vbn vdd\n"
        f"Mp6 d g vdd vdd pmos_nn L={lp:.0f}n NFIN={nfp}\n"
        f"Mn7 d vbn  0   0   nmos_nn L={ln:.0f}n NFIN={nfn}\n"
        f".ends\n"
        f".model nmos_nn NMOS (LEVEL=73 TECH={bt.nn_tech} VT={bt.effective_nmos_vt})\n"
        f".model pmos_nn PMOS (LEVEL=73 TECH={bt.nn_tech} VT={bt.effective_pmos_vt})\n"
        f".dc Vinp {_f(lo)} {_f(hi)} {_f(p.step)}\n"
        f".end\n")


# --- programmatic builders: ring oscillator ---------------------------------
def ngspice_ringosc(bt: BenchTech, p: RingOscParams, baked: Path,
                    tstop: float) -> Dict[str, str]:
    N = p.n_stages
    nd = [f"n{i}" for i in range(1, N + 1)]
    body = [f'.include "{baked}"', ".temp 27", f"Vdd vdd 0 {_f(bt.vdd)}"]
    for i in range(N):
        inp = nd[i - 1]            # i=0 → nd[-1]=nN (feedback wrap)
        body += [f"Np{i} {nd[i]} {inp} vdd vdd {bt.pmos_model}",
                 f"Nn{i} {nd[i]} {inp} 0 0 {bt.nmos_model}",
                 f"Cl{i} {nd[i]} 0 {_cap(p.cload)}"]
    ic = " ".join(f"v(n{i})={'0' if i % 2 == 1 else _f(bt.vdd)}"
                  for i in range(1, N + 1))
    body.append(f".ic {ic}")
    return {"body": "\n".join(body), "signals": f"v(n{N})",
            "analysis": f"tran {_tp(p.tstep)} {_tn(tstop)} uic"}


def directnet_ringosc(bt: BenchTech, p: RingOscParams, tstop: float) -> str:
    N = p.n_stages
    ln, lp = bt.l_nmos * 1e9, bt.l_pmos * 1e9
    nfn, nfp = bt.nfin, bt.effective_nfin_p
    ic = " ".join(f"V(n{i})={'0.0' if i % 2 == 1 else _f(bt.vdd)}"
                  for i in range(1, N + 1))
    # Hierarchical deck (V6.12.0): one ringinv .subckt instantiated N times;
    # stage nodes n1..nN stay top-level via the ports, so probes/period
    # extraction are unchanged.
    lines = [f"* {N}-stage ring oscillator — DirectNet LEVEL=73 ({bt.name}) [sweep]",
             f"Vdd vdd 0 {_f(bt.vdd)}", f".ic {ic}"]
    for i in range(1, N + 1):
        inp = f"n{i - 1}" if i > 1 else f"n{N}"
        lines += [f"Xinv{i} {inp} n{i} vdd ringinv"]
    lines += [f".subckt ringinv i o vdd",
              f"Mp o i vdd vdd pmos_nn L={lp:.0f}n NFIN={nfp}",
              f"Mn o i 0   0   nmos_nn L={ln:.0f}n NFIN={nfn}",
              f"Cl o 0 {_cap(p.cload)}",
              f".ends",
              f".model nmos_nn NMOS (LEVEL=73 TECH={bt.nn_tech} VT={bt.effective_nmos_vt})",
              f".model pmos_nn PMOS (LEVEL=73 TECH={bt.nn_tech} VT={bt.effective_pmos_vt})",
              # `uic` matches the ship-gate template (directnet_ring_osc_tran.sp:37);
              # the sweep transient runner pins .ic nodes regardless, but keeping the
              # token makes the deck byte-faithful so verify_complex_sweep_canaries
              # stays green (bug report B2).
              f".tran {_tp(p.tstep)} {_tn(tstop)} uic", ".end"]
    return "\n".join(lines) + "\n"


# --- programmatic builders: switched-cap unit cell --------------------------
def ngspice_switchcap(bt: BenchTech, p: SwitchCapParams,
                      baked: Path) -> Dict[str, str]:
    _se, _hs, _he, tstop, pw, per = sc_windows(p)
    n, pm = bt.nmos_model, bt.pmos_model
    vin = round(bt.vdd * p.vin_frac, 3)
    body = [f'.include "{baked}"', ".temp 27", f"Vdd vdd 0 {_f(bt.vdd)}",
            f"Vin vin 0 {_f(vin)}",
            (f"Vphi phi 0 PULSE(0 {_f(bt.vdd)} {_tn(p.td)} {_tn(p.clk_slew)} "
             f"{_tn(p.clk_slew)} {_tn(pw)} {_tn(per)})"),
            f"Npc phib phi vdd vdd {pm}", f"Nnc phib phi 0 0 {n}",
            f"Nnt vin phi vsamp 0 {n}", f"Npt vin phib vsamp vdd {pm}",
            f"Csample vsamp 0 {_cap(p.csample)}",
            f".ic v(vsamp)=0 v(phib)={_f(bt.vdd)}"]
    return {"body": "\n".join(body), "signals": "v(vsamp)",
            "analysis": f"tran {_tp(p.tstep)} {_tn(tstop)} uic"}


def directnet_switchcap(bt: BenchTech, p: SwitchCapParams) -> str:
    _se, _hs, _he, tstop, pw, per = sc_windows(p)
    ln, lp = bt.l_nmos * 1e9, bt.l_pmos * 1e9
    nfn, nfp = bt.nfin, bt.effective_nfin_p
    vin = round(bt.vdd * p.vin_frac, 3)
    return (
        f"* Switched-cap unit cell — DirectNet LEVEL=73 ({bt.name}) [sweep]\n"
        f"Vdd vdd 0 {_f(bt.vdd)}\n"
        f"Vin vin 0 {_f(vin)}\n"
        f"Vphi phi 0 PULSE 0 {_f(bt.vdd)} {_tn(p.td)} {_tn(p.clk_slew)} "
        f"{_tn(p.clk_slew)} {_tn(pw)} {_tn(per)}\n"
        f"Xck phi phib vdd ckinv\n"
        f"Xtg vin vsamp phi phib vdd tgate\n"
        f"Csample vsamp 0 {_cap(p.csample)}\n"
        f".ic V(vsamp)=0.0 V(phib)={_f(bt.vdd)}\n"
        f".subckt ckinv i o vdd\n"
        f"Mpc o i vdd vdd pmos_nn L={lp:.0f}n NFIN={nfp}\n"
        f"Mnc o i 0   0   nmos_nn L={ln:.0f}n NFIN={nfn}\n"
        f".ends\n"
        f".subckt tgate a b phi phib vdd\n"
        f"Mnt a phi  b 0   nmos_nn L={ln:.0f}n NFIN={nfn}\n"
        f"Mpt a phib b vdd pmos_nn L={lp:.0f}n NFIN={nfp}\n"
        f".ends\n"
        f".model nmos_nn NMOS (LEVEL=73 TECH={bt.nn_tech} VT={bt.effective_nmos_vt})\n"
        f".model pmos_nn PMOS (LEVEL=73 TECH={bt.nn_tech} VT={bt.effective_pmos_vt})\n"
        # `uic` matches the ship-gate template (directnet_switchcap_tran.sp:33);
        # byte-faithful with the single-point deck so the equivalence canary is
        # green (bug report B2). The sweep runner pins .ic nodes regardless.
        f".tran {_tp(p.tstep)} {_tn(tstop)} uic\n"
        f".end\n")


# --- programmatic builders: 6T SRAM (half-cell lobe + full-cell force_ic) ----
def ngspice_sram_lobe(bt: BenchTech, p: SramParams, nfin: int,
                      baked: Path) -> Dict[str, str]:
    n, pm = bt.nmos_model, bt.pmos_model
    wl = round(bt.vdd * p.wl_frac, 3)
    body = [f'.include "{baked}"', ".temp 27", f"Vdd vdd 0 {_f(bt.vdd)}",
            f"Vwl wl 0 {_f(wl)}", f"Vbl bl 0 {_f(bt.vdd)}", "Vq q 0 0.0",
            f"Npl qb q vdd vdd {pm}", f"Nnl qb q 0 0 {n}",
            f"Nna bl wl qb 0 {n}"]
    return {"body": "\n".join(body), "signals": "v(qb)",
            "analysis": f"dc Vq 0 {_f(bt.vdd)} {_f(p.dc_step)}"}


def directnet_sram_lobe(bt: BenchTech, p: SramParams, nfin: int) -> str:
    ln, lp = bt.l_nmos * 1e9, bt.l_pmos * 1e9
    wl = round(bt.vdd * p.wl_frac, 3)
    return (
        f"* SRAM read-SNM half-cell — DirectNet ({bt.name} NFIN={nfin})\n"
        f"Vdd vdd 0 {_f(bt.vdd)}\n"
        f"Vwl wl 0 {_f(wl)}\n"
        f"Vbl bl 0 {_f(bt.vdd)}\n"
        f"Vq q 0 0.0\n"
        f"Xinv q qb vdd sraminv NF={nfin}\n"
        f"Mna bl wl qb 0 nmos_nn L={ln:.0f}n NFIN={nfin}\n"
        f".subckt sraminv i o vdd NF=1\n"
        f"Mpl o i vdd vdd pmos_nn L={lp:.0f}n NFIN=NF\n"
        f"Mnl o i 0   0   nmos_nn L={ln:.0f}n NFIN=NF\n"
        f".ends\n"
        f".model nmos_nn NMOS (LEVEL=73 TECH={bt.nn_tech} VT={bt.effective_nmos_vt})\n"
        f".model pmos_nn PMOS (LEVEL=73 TECH={bt.nn_tech} VT={bt.effective_pmos_vt})\n"
        f".dc Vq 0 {_f(bt.vdd)} {_f(p.dc_step)}\n"
        f".end\n")


def directnet_sram_6t(bt: BenchTech, q0: float, qb0: float, nfin: int) -> str:
    """Full cross-coupled 6T cell, wl=OFF/hold (the force_ic retention probe)."""
    ln, lp = bt.l_nmos * 1e9, bt.l_pmos * 1e9
    return (
        f"* 6T SRAM cell — DirectNet ({bt.name}) wl=OFF/hold [sweep]\n"
        f"Vdd vdd 0 {_f(bt.vdd)}\n"
        f"Vwl wl 0 0.0\n"
        f"Vbl bl 0 {_f(bt.vdd)}\n"
        f"Vblb blb 0 {_f(bt.vdd)}\n"
        f".ic V(q)={_f(q0)} V(qb)={_f(qb0)}\n"
        f"Xl q qb vdd sraminv NF={nfin}\n"
        f"Xr qb q vdd sraminv NF={nfin}\n"
        f"Mal bl  wl q  0 nmos_nn L={ln:.0f}n NFIN={nfin}\n"
        f"Mar blb wl qb 0 nmos_nn L={ln:.0f}n NFIN={nfin}\n"
        f".subckt sraminv i o vdd NF=1\n"
        f"Mpl o i vdd vdd pmos_nn L={lp:.0f}n NFIN=NF\n"
        f"Mnl o i 0   0   nmos_nn L={ln:.0f}n NFIN=NF\n"
        f".ends\n"
        f".model nmos_nn NMOS (LEVEL=73 TECH={bt.nn_tech} VT={bt.effective_nmos_vt})\n"
        f".model pmos_nn PMOS (LEVEL=73 TECH={bt.nn_tech} VT={bt.effective_pmos_vt})\n"
        f".op\n.end\n")
