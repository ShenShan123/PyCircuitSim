#!/usr/bin/env python3
"""Verification suite for .subckt / .ends hierarchical netlist support.

Four levels:
  Level 0 (no NGSPICE, no solver): malformed hierarchies must be rejected
    loudly — unknown subckt name, instance/port count mismatch, and runaway
    recursion each raise a ValueError naming the problem.
  Level 1 (no NGSPICE): subckt-based decks must produce results identical to
    their hand-flattened equivalents (same MNA system) — RC transient, RC AC,
    nested hierarchy + parameter passing at DC OP, and `.ic`-in-body + uic.
  Level 2 (NGSPICE): BSIM-CMG (LEVEL=72) CMOS inverter wrapped in a .subckt,
    transient vs (a) the flat PyCircuitSim deck and (b) NGSPICE ground truth.
  Level 3 (NGSPICE): two-inverter buffer as nested hierarchy (inv defined
    inside buf; X-in-X instances; NFIN parameter passing; `.ic` on an
    internal hierarchical node) vs NGSPICE ground truth.

Ground truth is always NGSPICE on the identical BSIM-CMG OSDI model.

Usage:
    conda run -n pycircuitsim python tests/simple_circuits/verify_subckt.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.common.bsimcmg_tran import (  # noqa: E402
    ALL_TECHS,
    STARTUP_EXCLUSION,
    TestConfig,
    compute_metrics,
    get_baked_modelcard,
    get_merged_modelcard,
    make_baseline,
    create_pycircuitsim_netlist,
    run_ngspice_subprocess,
    OSDI_PATH,
)
from tests.common.base import (  # noqa: E402
    SIMPLE_DECKS, SUBCIRCUIT_DECKS, render_template,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "tests" / "subckt"

# Flat-vs-subckt equivalence tolerances. Component order (hence stamp
# summation order) differs between the two decks, so demand agreement far
# below any physical scale but above pure ULP noise.
LINEAR_EQUIV_TOL = 1e-9    # [V] linear circuits (direct solve)
NR_EQUIV_TOL = 1e-6        # [V] Newton-Raphson circuits (NR tol is 1e-7)
NGSPICE_NRMSE_GATE = 1.0   # [% of VDD] post-startup transient vs NGSPICE


# ---------------------------------------------------------------------------
# Generic runners
# ---------------------------------------------------------------------------
def parse_deck(text: str, path: Path, **parser_kwargs: Any) -> Any:
    from pycircuitsim.parser import Parser
    path.write_text(text)
    parser = Parser(**parser_kwargs)
    parser.parse_file(str(path))
    return parser


def run_tran(parser: Any) -> Dict[str, np.ndarray]:
    """Transient run mirroring tests/common/bsimcmg_tran.run_pycircuitsim."""
    from pycircuitsim.solver import DCSolver, TransientSolver

    circuit = parser.circuit
    logging.disable(logging.CRITICAL)
    try:
        initial_guess = circuit.initial_conditions or None
        op_solver = DCSolver(circuit, initial_guess=initial_guess,
                             use_source_stepping=True)
        op_solution = op_solver.solve()
        solver = TransientSolver(
            circuit,
            t_stop=parser.analysis_params["tstop"],
            dt=parser.analysis_params["tstep"],
            initial_guess=op_solution,
            use_gmin_stepping=True, gmin_initial=1e-9,
            gmin_final=1e-12, gmin_steps=5,
            use_pseudo_transient=True, pseudo_transient_steps=5,
            pseudo_transient_cap=1e-12,
            debug=False, nr_tolerance=1e-7,
        )
        return solver.solve()
    finally:
        logging.disable(logging.NOTSET)


def run_op(parser: Any) -> Dict[str, float]:
    from pycircuitsim.solver import DCSolver

    logging.disable(logging.CRITICAL)
    try:
        solver = DCSolver(parser.circuit,
                          initial_guess=parser.circuit.initial_conditions or None)
        return solver.solve()
    finally:
        logging.disable(logging.NOTSET)


def run_ac(parser: Any) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    from pycircuitsim.solver import ACSolver, DCSolver

    p = parser.analysis_params
    if p["sweep_type"] == "dec":
        total = int(p["num_points"] * np.log10(p["fstop"] / p["fstart"]))
        freqs = np.logspace(np.log10(p["fstart"]), np.log10(p["fstop"]), total)
    elif p["sweep_type"] == "oct":
        total = int(p["num_points"] * np.log2(p["fstop"] / p["fstart"]))
        freqs = np.logspace(np.log10(p["fstart"]), np.log10(p["fstop"]), total)
    else:
        freqs = np.linspace(p["fstart"], p["fstop"], p["num_points"])

    logging.disable(logging.CRITICAL)
    try:
        op = DCSolver(parser.circuit).solve()
        ac = ACSolver(parser.circuit, dc_solution=op)
        return freqs, ac.solve(freqs)
    finally:
        logging.disable(logging.NOTSET)


# ---------------------------------------------------------------------------
# Level 0 — loud parse errors (no NGSPICE, no solver)
# ---------------------------------------------------------------------------
# audit B5g: every other check in this file is a numeric-agreement check, so
# the three documented .subckt guard rails could be deleted from the parser
# without moving the verdict. Each entry is (label, file stem, deck, needle);
# the needle pins the diagnostic, not just the fact that something blew up.
BAD_DECKS: List[Tuple[str, str, str, str]] = [
    ("unknown subckt raises", "unknown_subckt",
     "V1 in 0 1\n"
     "X1 in out nosuch\n"
     ".end\n",
     "not found"),
    ("port-count mismatch raises", "port_count",
     ".subckt rdiv a b\n"
     "R1 a b 1k\n"
     ".ends\n"
     "V1 in 0 1\n"
     "X1 in mid out rdiv\n"
     ".end\n",
     "declares 2 ports"),
    ("recursive subckt raises", "recursion",
     ".subckt loopy a b\n"
     "R1 a b 1k\n"
     "X9 a b loopy\n"
     ".ends\n"
     "V1 in 0 1\n"
     "X1 in out loopy\n"
     ".end\n",
     "nesting deeper than"),
]


def level0() -> List[Tuple[str, bool, str]]:
    """Malformed hierarchies must raise ValueError with a diagnostic message.

    Two distinct failures are checked for, because both are silent-green
    hazards: a deck that parses anyway (the guard was dropped) and a deck that
    dies with some incidental downstream exception (TypeError from a None
    definition, RecursionError from an unguarded cycle) instead of a named
    error the user can act on.
    """
    work = RESULTS_DIR / "level0"
    work.mkdir(parents=True, exist_ok=True)
    results = []

    for label, stem, text, needle in BAD_DECKS:
        try:
            parse_deck(text, work / f"{stem}.sp")
        except ValueError as exc:
            msg = str(exc)
            ok = needle in msg
            results.append((label, ok,
                            f"ValueError: {msg[:60]}" if ok
                            else f"message lacks '{needle}': {msg[:80]}"))
        except Exception as exc:  # noqa: BLE001 — wrong type is still a miss
            results.append((label, False,
                            f"wrong exception type "
                            f"{type(exc).__name__}: {str(exc)[:60]}"))
        else:
            results.append((label, False, "NO RAISE — silent accept"))
    return results


# ---------------------------------------------------------------------------
# Level 1 — linear equivalence (no NGSPICE)
# ---------------------------------------------------------------------------
def _fixture(name: str, substitutions: Dict[str, str]) -> str:
    """Render one parameterized subcircuit fixture from ``examples/``."""
    return render_template(SUBCIRCUIT_DECKS / name, substitutions)


def level1() -> List[Tuple[str, bool, str]]:
    work = RESULTS_DIR / "level1"
    work.mkdir(parents=True, exist_ok=True)
    results = []

    # T1: RC ladder transient equivalence
    flat = run_tran(parse_deck(_fixture("rc_ladder_flat.spice.tmpl", {
        "TEMP": "27",
        "INPUT_SPEC": "PULSE 0 1.0 1n 0.1n 0.1n 5n 12n",
        "R1": "1k", "C1": "1p", "R2": "2k", "C2": "2p",
        "ANALYSIS": ".tran 50p 10n",
    }), work / "flat_tran.sp"))
    sub = run_tran(parse_deck(_fixture(
        "rc_ladder_hierarchical.spice.tmpl", {
            "TEMP": "27",
            "INPUT_SPEC": "PULSE 0 1.0 1n 0.1n 0.1n 5n 12n",
            "R1": "1k", "R_DEFAULT": "999", "C1": "1p", "C2": "2p",
            "ANALYSIS": ".tran 50p 10n",
        }), work / "sub_tran.sp"))
    delta = max(
        float(np.max(np.abs(flat["out"] - sub["out"]))),
        float(np.max(np.abs(flat["n1"] - sub["X1.m"]))),
    )
    results.append(("RC-ladder .tran subckt==flat",
                    delta < LINEAR_EQUIV_TOL, f"max|dV|={delta:.3e} V"))

    # T2: RC lowpass AC equivalence
    freqs_f, res_f = run_ac(parse_deck(render_template(
        SIMPLE_DECKS / "rc_lowpass.spice.tmpl", {
            "TEMP": "27", "INPUT_DC": "0", "INPUT_AC": "1",
            "INPUT_PHASE": "0", "RESISTANCE": "1k",
            "CAPACITANCE": "159.155n", "ANALYSIS": ".ac dec 20 10 1e6",
        }), work / "flat_ac.sp"))
    freqs_s, res_s = run_ac(parse_deck(_fixture(
        "rc_lowpass_hierarchical.spice.tmpl", {
            "TEMP": "27", "INPUT_DC": "0", "INPUT_AC": "1",
            "INPUT_PHASE": "0", "RESISTANCE": "1k",
            "CAPACITANCE": "159.155n", "ANALYSIS": ".ac dec 20 10 1e6",
        }), work / "sub_ac.sp"))
    dmag = float(np.max(np.abs(np.abs(res_f["out"]) - np.abs(res_s["out"]))))
    dph = float(np.max(np.abs(np.angle(res_f["out"]) - np.angle(res_s["out"]))))
    ok = np.array_equal(freqs_f, freqs_s) and dmag < LINEAR_EQUIV_TOL \
        and dph < LINEAR_EQUIV_TOL
    results.append(("RC-lowpass .ac subckt==flat", ok,
                    f"max|dMag|={dmag:.3e}, max|dPh|={dph:.3e} rad"))

    # T3: nested hierarchy + expression params at DC OP
    op_f = run_op(parse_deck(_fixture("resistor_tree_flat.spice.tmpl", {
        "TEMP": "27", "VTOP": "2.0", "R1": "3k", "R2": "6k",
        "CMID": "2p", "RLOAD": "1k", "ANALYSIS": ".op",
    }), work / "flat_op.sp"))
    op_s = run_op(parse_deck(_fixture(
        "resistor_tree_hierarchical.spice.tmpl", {
            "TEMP": "27", "VTOP": "2.0", "GAIN": "3",
            "RBASE": "1k", "RBASE_DOUBLE": "2k", "CMID": "2p",
            "RLOAD": "1k", "ANALYSIS": ".op",
        }), work / "sub_op.sp"))
    pairs = [("mid", "mid"), ("m1", "Xa.m"), ("top", "top")]
    delta = max(abs(op_f[a] - op_s[b]) for a, b in pairs)
    results.append(("nested subckt + {expr} params DC OP", delta < LINEAR_EQUIV_TOL,
                    f"max|dV|={delta:.3e} V "
                    f"(mid={op_s['mid']:.6f} V, Xa.m={op_s['Xa.m']:.6f} V)"))

    # T4: .ic in body (param value) + uic pinning of a high-impedance node
    parser = parse_deck(_fixture("ic_hierarchical.spice.tmpl", {
        "TEMP": "27", "VIN": "0.0", "VIC": "0.75",
        "R1": "1e6", "R2": "1e6", "CHOLD": "1p", "MID_IC": "0.2",
        "ANALYSIS": ".tran 1n 20n uic",
    }), work / "sub_ic_uic.sp")
    ics = parser.circuit.initial_conditions
    ic_ok = (abs(ics.get("hold", 0.0) - 0.75) < 1e-12
             and abs(ics.get("X1.m", 0.0) - 0.2) < 1e-12
             and parser.analysis_params.get("uic") is True)
    # uic semantics: simulate through run_transient's pinning path
    from pycircuitsim.models.passive import VoltageSource
    from pycircuitsim.solver import DCSolver
    circuit = parser.circuit
    temps = []
    for node, val in circuit.initial_conditions.items():
        vs = VoltageSource(f"_V_uic_{node}", [node, "0"], val)
        circuit.components.append(vs)
        temps.append(vs)
    try:
        op = DCSolver(circuit, initial_guess=circuit.initial_conditions).solve()
    finally:
        for vs in temps:
            circuit.components.remove(vs)
    pin_ok = abs(op["hold"] - 0.75) < 1e-9 and abs(op["X1.m"] - 0.2) < 1e-9
    results.append((".ic in body + hierarchical map + uic pin",
                    ic_ok and pin_ok,
                    f"ic={{hold: {ics.get('hold')}, X1.m: {ics.get('X1.m')}}}, "
                    f"uic-OP hold={op['hold']:.6f} V"))
    return results


# ---------------------------------------------------------------------------
# Level 2 — BSIM-CMG inverter in a .subckt vs flat vs NGSPICE
# ---------------------------------------------------------------------------
def create_flat_inverter_netlist(config: TestConfig, work_dir: Path) -> Path:
    """Render the canonical flat inverter through the shared L72 adapter."""
    return create_pycircuitsim_netlist(config, work_dir)


def create_subckt_inverter_netlist(config: TestConfig, work_dir: Path) -> Path:
    """Inverter wrapped in .subckt (same system as the flat L1 deck)."""
    tech, vt = config.tech, config.vt
    l_n_nm = config.l_nmos * 1e9
    l_p_nm = config.l_pmos * 1e9
    path = work_dir / f"pycircuitsim_subckt_{config.label}.sp"
    path.write_text(_fixture("inverter_hierarchical.spice.tmpl", {
        "MODEL_SETUP": (
            f".model {vt.nmos_model} NMOS (LEVEL=72)\n"
            f".model {vt.pmos_model} PMOS (LEVEL=72)"
        ),
        "TEMP": "27", "VDD": f"{config.vdd}",
        "INPUT_SPEC": (
            f"PULSE {config.pulse_v1} {config.pulse_v2} {config.td} "
            f"{config.tr} {config.tf} {config.pw} {config.per}"
        ),
        "NFP": str(config.nfin_p), "NFN": str(config.nfin_n),
        "P_PREFIX": "M", "N_PREFIX": "M",
        "P_DEVICE": (
            f"{vt.pmos_model} L={l_p_nm:.0f}n "
            f"TFIN={tech.tfin * 1e9:.1f}n"
        ),
        "N_DEVICE": (
            f"{vt.nmos_model} L={l_n_nm:.0f}n "
            f"TFIN={tech.tfin * 1e9:.1f}n"
        ),
        "OUTPUT_LOAD": f"Cload out 0 {config.cload}",
        "INITIAL_CONDITION": f".ic V(o)={config.vdd}",
        "ANALYSIS": f".tran {config.tstep} {config.tstop}",
    }))
    return path


def run_ngspice_buffer(config: TestConfig, work_dir: Path) -> Dict[str, np.ndarray]:
    """NGSPICE two-inverter buffer reference (flat deck, ground truth)."""
    baked_lib = get_baked_modelcard(config, work_dir)
    vt = config.vt
    netlist_path = work_dir / f"ngspice_buf_{config.label}.cir"
    netlist_path.write_text(_fixture("inverter_buffer_flat.spice.tmpl", {
        "MODEL_SETUP": f'.include "{baked_lib}"',
        "TEMP": "27", "VDD": f"{config.vdd}",
        "INPUT_SPEC": (
            f"PULSE({config.pulse_v1} {config.pulse_v2} {config.td} "
            f"{config.tr} {config.tf} {config.pw} {config.per})"
        ),
        "P_PREFIX": "N", "N_PREFIX": "N",
        "P_DEVICE": vt.pmos_model, "N_DEVICE": vt.nmos_model,
        "OUTPUT_LOAD": f"Cload out 0 {config.cload}",
        "INITIAL_CONDITION": f".ic V(mid)={config.vdd} V(out)=0",
        "ANALYSIS": f".tran {config.tstep} {config.tstop} uic",
    }))
    csv_path = work_dir / f"ngspice_buf_{config.label}.csv"
    log_path = work_dir / f"ngspice_buf_{config.label}.log"
    runner_path = work_dir / f"ngspice_buf_{config.label}_runner.cir"
    runner_path.write_text(f"""\
* NGSPICE buffer runner ({config.label})
.control
osdi {OSDI_PATH}
source {netlist_path}
set filetype=ascii
set wr_vecnames
run
wrdata {csv_path} v(out) v(mid)
.endc
.end
""")
    lines = run_ngspice_subprocess(runner_path, log_path, csv_path)
    data = np.array([[float(x) for x in ln.split()]
                     for ln in lines[1:] if ln.strip()])
    return {"time": data[:, 0], "v(out)": data[:, 1], "v(mid)": data[:, 3]}


def create_subckt_buffer_netlist(config: TestConfig, work_dir: Path) -> Path:
    """Nested hierarchy: buf contains 2 X-instances of inv (defined inside)."""
    tech, vt = config.tech, config.vt
    l_n_nm = config.l_nmos * 1e9
    l_p_nm = config.l_pmos * 1e9
    path = work_dir / f"pycircuitsim_buf_{config.label}.sp"
    path.write_text(_fixture("inverter_buffer_hierarchical.spice.tmpl", {
        "MODEL_SETUP": (
            f".model {vt.nmos_model} NMOS (LEVEL=72)\n"
            f".model {vt.pmos_model} PMOS (LEVEL=72)"
        ),
        "TEMP": "27", "VDD": f"{config.vdd}",
        "INPUT_SPEC": (
            f"PULSE {config.pulse_v1} {config.pulse_v2} {config.td} "
            f"{config.tr} {config.tf} {config.pw} {config.per}"
        ),
        "NFN": str(config.nfin_n), "NFP": str(config.nfin_p),
        "P_PREFIX": "M", "N_PREFIX": "M",
        "P_DEVICE": (
            f"{vt.pmos_model} L={l_p_nm:.0f}n "
            f"TFIN={tech.tfin * 1e9:.1f}n"
        ),
        "N_DEVICE": (
            f"{vt.nmos_model} L={l_n_nm:.0f}n "
            f"TFIN={tech.tfin * 1e9:.1f}n"
        ),
        "OUTPUT_LOAD": f"Cload out 0 {config.cload}",
        "OUT_IC": "0", "ANALYSIS": f".tran {config.tstep} {config.tstop}",
    }))
    return path


def _parse_l72(netlist_path: Path, config: TestConfig, work_dir: Path):
    from pycircuitsim.parser import Parser
    merged = get_merged_modelcard(config, work_dir)
    parser = Parser(
        modelcard_path=str(merged),
        model_name_map={"NMOS": config.vt.nmos_model,
                        "PMOS": config.vt.pmos_model},
    )
    parser.parse_file(str(netlist_path))
    return parser


def _nrmse_vs_ngspice(ng: Dict[str, np.ndarray], t_py: np.ndarray,
                      v_py: np.ndarray, ng_key: str,
                      config: TestConfig) -> Dict[str, float]:
    t_max = min(ng["time"][-1], t_py[-1])
    t_common = np.arange(max(STARTUP_EXCLUSION, ng["time"][0], t_py[0]),
                         t_max, config.tstep)
    ng_v = np.interp(t_common, ng["time"], ng[ng_key])
    py_v = np.interp(t_common, t_py, v_py)
    return compute_metrics(ng_v, py_v, config.vdd)


def level2() -> List[Tuple[str, bool, str]]:
    from tests.common.bsimcmg_tran import run_ngspice

    work = RESULTS_DIR / "level2"
    work.mkdir(parents=True, exist_ok=True)
    config = make_baseline(ALL_TECHS["ASAP7"], config_name="subckt")
    results = []

    ng = run_ngspice(config, work)
    flat = run_tran(_parse_l72(
        create_flat_inverter_netlist(config, work), config, work))
    sub = run_tran(_parse_l72(
        create_subckt_inverter_netlist(config, work), config, work))

    # (a) subckt == flat at the shared top-level output port.
    d = float(np.max(np.abs(np.interp(flat["time"], sub["time"], sub["out"])
                            - flat["out"])))
    results.append(("L72 inverter .tran subckt==flat", d < NR_EQUIV_TOL,
                    f"max|dV|={d:.3e} V"))

    # (b) subckt vs NGSPICE ground truth
    m = _nrmse_vs_ngspice(ng, sub["time"], sub["out"], "v(out)", config)
    results.append(("L72 inverter subckt vs NGSPICE",
                    m["NRMSE (% of Vdd)"] < NGSPICE_NRMSE_GATE,
                    f"post-startup NRMSE={m['NRMSE (% of Vdd)']:.3f}% of VDD, "
                    f"max|err|={m['Max |error| (mV)']:.1f} mV"))
    return results


def level3() -> List[Tuple[str, bool, str]]:
    work = RESULTS_DIR / "level3"
    work.mkdir(parents=True, exist_ok=True)
    config = make_baseline(ALL_TECHS["ASAP7"], config_name="subckt_buf")
    results = []

    ng = run_ngspice_buffer(config, work)
    parser = _parse_l72(create_subckt_buffer_netlist(config, work),
                        config, work)
    # Hierarchy sanity: nested-instance flattening + .ic on internal node
    names = sorted(c.name for c in parser.circuit.components)
    want = {"M.Xbuf.X1.Mp1", "M.Xbuf.X1.Mn1", "M.Xbuf.X2.Mp1", "M.Xbuf.X2.Mn1"}
    hier_ok = want.issubset(set(names)) and "Xbuf.m" in parser.circuit.nodes
    ic = parser.circuit.initial_conditions
    ic_ok = (abs(ic.get("Xbuf.m", -1) - config.vdd) < 1e-12
             and abs(ic.get("out", -1) - 0.0) < 1e-12)
    results.append(("nested X-in-X flattening + param NFIN + .ic map",
                    hier_ok and ic_ok,
                    f"devices={sorted(want & set(names))}, ic={ic}"))

    sub = run_tran(parser)
    m_out = _nrmse_vs_ngspice(ng, sub["time"], sub["out"], "v(out)", config)
    m_mid = _nrmse_vs_ngspice(ng, sub["time"], sub["Xbuf.m"], "v(mid)", config)
    worst = max(m_out["NRMSE (% of Vdd)"], m_mid["NRMSE (% of Vdd)"])
    results.append(("L72 buffer (nested subckt) vs NGSPICE",
                    worst < NGSPICE_NRMSE_GATE,
                    f"NRMSE out={m_out['NRMSE (% of Vdd)']:.3f}% "
                    f"mid(Xbuf.m)={m_mid['NRMSE (% of Vdd)']:.3f}% of VDD"))
    return results


# ---------------------------------------------------------------------------
def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results: List[Tuple[str, bool, str]] = []
    print("=" * 72)
    print("Subcircuit (.subckt/.ends/X) verification")
    print("=" * 72)
    for name, fn in [("Level 0 (loud parse errors)", level0),
                     ("Level 1 (linear equivalence)", level1),
                     ("Level 2 (L72 inverter vs NGSPICE)", level2),
                     ("Level 3 (L72 nested buffer vs NGSPICE)", level3)]:
        print(f"\n--- {name} ---")
        try:
            batch = fn()
        except Exception as exc:  # fail loud, keep going
            import traceback
            traceback.print_exc()
            batch = [(name, False, f"EXCEPTION: {exc}")]
        for label, ok, detail in batch:
            print(f"  [{'PASS' if ok else 'FAIL'}] {label:45s} {detail}")
        all_results.extend(batch)

    n_pass = sum(1 for _, ok, _ in all_results if ok)
    print("\n" + "=" * 72)
    print(f"RESULT: {n_pass}/{len(all_results)} PASS")
    print("=" * 72)
    return 0 if n_pass == len(all_results) else 1


if __name__ == "__main__":
    sys.exit(main())
