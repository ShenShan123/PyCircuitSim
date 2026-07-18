#!/usr/bin/env python3
"""Verification suite for .subckt / .ends hierarchical netlist support.

Three levels:
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
    conda run -n pycircuitsim python tests/verify_subckt.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.common.bsimcmg_tran import (  # noqa: E402
    ALL_TECHS,
    STARTUP_EXCLUSION,
    TestConfig,
    compute_metrics,
    get_baked_modelcard,
    get_merged_modelcard,
    make_baseline,
    run_ngspice_subprocess,
    OSDI_PATH,
)

RESULTS_DIR = PROJECT_ROOT / "tests" / "verify_subckt_results"

# Flat-vs-subckt equivalence tolerances. Component order (hence stamp
# summation order) differs between the two decks, so demand agreement far
# below any physical scale but above pure ULP noise.
LINEAR_EQUIV_TOL = 1e-9    # [V] linear circuits (direct solve)
NR_EQUIV_TOL = 1e-6        # [V] Newton-Raphson circuits (NR tol is 1e-7)
NGSPICE_NRMSE_GATE = 1.0   # [% of VDD] post-startup transient vs NGSPICE


# ---------------------------------------------------------------------------
# Generic runners
# ---------------------------------------------------------------------------
def parse_deck(text: str, path: Path, **parser_kwargs):
    from pycircuitsim.parser import Parser
    path.write_text(text)
    parser = Parser(**parser_kwargs)
    parser.parse_file(str(path))
    return parser


def run_tran(parser) -> Dict[str, np.ndarray]:
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


def run_op(parser) -> Dict[str, float]:
    from pycircuitsim.solver import DCSolver

    logging.disable(logging.CRITICAL)
    try:
        solver = DCSolver(parser.circuit,
                          initial_guess=parser.circuit.initial_conditions or None)
        return solver.solve()
    finally:
        logging.disable(logging.NOTSET)


def run_ac(parser) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
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
# Level 1 — linear equivalence (no NGSPICE)
# ---------------------------------------------------------------------------
FLAT_RC_TRAN = """* flat RC ladder transient
V1 in 0 PULSE 0 1.0 1n 0.1n 0.1n 5n 12n
R1 in n1 1k
C1 n1 0 1p
R2 n1 out 2k
C2 out 0 2p
.tran 50p 10n
.end
"""

SUB_RC_TRAN = """* subckt RC ladder transient (identical system)
V1 in 0 PULSE 0 1.0 1n 0.1n 0.1n 5n 12n
X1 in out ladder RA=1k
.subckt ladder a b RA=999
R1 a m {RA}
C1 m 0 1p
R2 m b {2*RA}
C2 b 0 2p
.ends
.tran 50p 10n
.end
"""

FLAT_RC_AC = """* flat RC lowpass AC
V1 in 0 DC=0 AC=1 0
R1 in out 1k
C1 out 0 159.155n
.ac dec 20 10 1e6
.end
"""

SUB_RC_AC = """* subckt RC lowpass AC (identical system)
V1 in 0 DC=0 AC=1 0
X1 in out lp
.subckt lp a b
R1 a b 1k
C1 b 0 159.155n
.ends
.ac dec 20 10 1e6
.end
"""

FLAT_NESTED_OP = """* flat resistor tree, DC OP
V1 top 0 2.0
R1 top m1 3k
R2 m1 mid 6k
C1 m1 0 2p
Rload mid 0 1k
.end
"""

SUB_NESTED_OP = """* nested subckt resistor tree, DC OP (identical system)
V1 top 0 2.0
Xa top mid pair GAIN=3
Rload mid 0 1k
.subckt pair x y GAIN=2
X1 x m half RV={GAIN*1k}
X2 m y half RV='GAIN*2k'
C1 m 0 2p
.subckt half p q RV=1k
R1 p q {RV}
.ends half
.ends pair
.end
"""

SUB_IC_UIC = """* .ic inside subckt body + hierarchical .ic + uic
V1 in 0 0.0
X1 in hold sample VIC=0.75
.subckt sample a b VIC=0.5
R1 a m 1e6
R2 m b 1e6
C1 b 0 1p
.ic V(b)=VIC V(m)=0.2
.ends
.tran 1n 20n uic
.end
"""


def level1() -> List[Tuple[str, bool, str]]:
    work = RESULTS_DIR / "level1"
    work.mkdir(parents=True, exist_ok=True)
    results = []

    # T1: RC ladder transient equivalence
    flat = run_tran(parse_deck(FLAT_RC_TRAN, work / "flat_tran.sp"))
    sub = run_tran(parse_deck(SUB_RC_TRAN, work / "sub_tran.sp"))
    delta = max(
        float(np.max(np.abs(flat["out"] - sub["out"]))),
        float(np.max(np.abs(flat["n1"] - sub["X1.m"]))),
    )
    results.append(("RC-ladder .tran subckt==flat",
                    delta < LINEAR_EQUIV_TOL, f"max|dV|={delta:.3e} V"))

    # T2: RC lowpass AC equivalence
    freqs_f, res_f = run_ac(parse_deck(FLAT_RC_AC, work / "flat_ac.sp"))
    freqs_s, res_s = run_ac(parse_deck(SUB_RC_AC, work / "sub_ac.sp"))
    dmag = float(np.max(np.abs(np.abs(res_f["out"]) - np.abs(res_s["out"]))))
    dph = float(np.max(np.abs(np.angle(res_f["out"]) - np.angle(res_s["out"]))))
    ok = np.array_equal(freqs_f, freqs_s) and dmag < LINEAR_EQUIV_TOL \
        and dph < LINEAR_EQUIV_TOL
    results.append(("RC-lowpass .ac subckt==flat", ok,
                    f"max|dMag|={dmag:.3e}, max|dPh|={dph:.3e} rad"))

    # T3: nested hierarchy + expression params at DC OP
    op_f = run_op(parse_deck(FLAT_NESTED_OP, work / "flat_op.sp"))
    op_s = run_op(parse_deck(SUB_NESTED_OP, work / "sub_op.sp"))
    pairs = [("mid", "mid"), ("m1", "Xa.m"), ("top", "top")]
    delta = max(abs(op_f[a] - op_s[b]) for a, b in pairs)
    results.append(("nested subckt + {expr} params DC OP", delta < LINEAR_EQUIV_TOL,
                    f"max|dV|={delta:.3e} V "
                    f"(mid={op_s['mid']:.6f} V, Xa.m={op_s['Xa.m']:.6f} V)"))

    # T4: .ic in body (param value) + uic pinning of a high-impedance node
    parser = parse_deck(SUB_IC_UIC, work / "sub_ic_uic.sp")
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
    """Historical FLAT inverter deck (pre-V6.12.0 bsimcmg_tran text).

    Kept inline here — the shared infra deck is hierarchical since V6.12.0,
    so this local copy preserves a genuinely flat reference for the
    subckt==flat equivalence gate.
    """
    tech, vt = config.tech, config.vt
    l_n_nm = config.l_nmos * 1e9
    l_p_nm = config.l_pmos * 1e9
    path = work_dir / f"pycircuitsim_flat_{config.label}.sp"
    path.write_text(f"""\
* BSIM-CMG Inverter Transient - FLAT reference ({config.label})
Vdd 1 0 {config.vdd}
Vin 2 0 PULSE {config.pulse_v1} {config.pulse_v2} {config.td} {config.tr} {config.tf} {config.pw} {config.per}
Mp1 3 2 1 1 {vt.pmos_model} L={l_p_nm:.0f}n NFIN={config.nfin_p} TFIN={tech.tfin*1e9:.1f}n
Mn1 3 2 0 0 {vt.nmos_model} L={l_n_nm:.0f}n NFIN={config.nfin_n} TFIN={tech.tfin*1e9:.1f}n
Cload 3 0 {config.cload}
.ic V(3)={config.vdd}
.model {vt.nmos_model} NMOS (LEVEL=72)
.model {vt.pmos_model} PMOS (LEVEL=72)
.tran {config.tstep} {config.tstop}
.end
""")
    return path


def create_subckt_inverter_netlist(config: TestConfig, work_dir: Path) -> Path:
    """Inverter wrapped in .subckt (same system as the flat L1 deck)."""
    tech, vt = config.tech, config.vt
    l_n_nm = config.l_nmos * 1e9
    l_p_nm = config.l_pmos * 1e9
    path = work_dir / f"pycircuitsim_subckt_{config.label}.sp"
    path.write_text(f"""\
* BSIM-CMG Inverter Transient - subckt hierarchy ({config.label})
Vdd 1 0 {config.vdd}
Vin 2 0 PULSE {config.pulse_v1} {config.pulse_v2} {config.td} {config.tr} {config.tf} {config.pw} {config.per}
Xinv 2 3 1 inv NFP={config.nfin_p} NFN={config.nfin_n}
Cload 3 0 {config.cload}
.subckt inv i o vdd NFP=1 NFN=1
Mp1 o i vdd vdd {vt.pmos_model} L={l_p_nm:.0f}n NFIN=NFP TFIN={tech.tfin*1e9:.1f}n
Mn1 o i 0 0 {vt.nmos_model} L={l_n_nm:.0f}n NFIN=NFN TFIN={tech.tfin*1e9:.1f}n
.ic V(o)={config.vdd}
.ends
.model {vt.nmos_model} NMOS (LEVEL=72)
.model {vt.pmos_model} PMOS (LEVEL=72)
.tran {config.tstep} {config.tstop}
.end
""")
    return path


def run_ngspice_buffer(config: TestConfig, work_dir: Path) -> Dict[str, np.ndarray]:
    """NGSPICE two-inverter buffer reference (flat deck, ground truth)."""
    baked_lib = get_baked_modelcard(config, work_dir)
    vt = config.vt
    netlist_path = work_dir / f"ngspice_buf_{config.label}.cir"
    netlist_path.write_text(f"""\
* BSIM-CMG buffer (2 inverters) - NGSPICE ({config.label})
.include "{baked_lib}"
.temp 27
Vdd vdd 0 {config.vdd}
Vin in 0 PULSE({config.pulse_v1} {config.pulse_v2} {config.td} {config.tr} {config.tf} {config.pw} {config.per})
Np1 mid in vdd vdd {vt.pmos_model}
Nn1 mid in 0 0 {vt.nmos_model}
Np2 out mid vdd vdd {vt.pmos_model}
Nn2 out mid 0 0 {vt.nmos_model}
Cload out 0 {config.cload}
.ic V(mid)={config.vdd} V(out)=0
.tran {config.tstep} {config.tstop} uic
.end
""")
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
    path.write_text(f"""\
* BSIM-CMG buffer - nested subckt hierarchy ({config.label})
Vdd vdd 0 {config.vdd}
Vin in 0 PULSE {config.pulse_v1} {config.pulse_v2} {config.td} {config.tr} {config.tf} {config.pw} {config.per}
Xbuf in out vdd buf NF={config.nfin_n} VDDIC={config.vdd}
Cload out 0 {config.cload}
.subckt buf a y vdd NF=1 VDDIC=1.0
X1 a m vdd inv NFP={config.nfin_p} NFN=NF
X2 m y vdd inv NFP={config.nfin_p} NFN=NF
.ic V(m)=VDDIC V(y)=0
.subckt inv i o vdd NFP=1 NFN=1
Mp1 o i vdd vdd {vt.pmos_model} L={l_p_nm:.0f}n NFIN=NFP TFIN={tech.tfin*1e9:.1f}n
Mn1 o i 0 0 {vt.nmos_model} L={l_n_nm:.0f}n NFIN=NFN TFIN={tech.tfin*1e9:.1f}n
.ends inv
.ends buf
.model {vt.nmos_model} NMOS (LEVEL=72)
.model {vt.pmos_model} PMOS (LEVEL=72)
.tran {config.tstep} {config.tstop}
.end
""")
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

    # (a) subckt == flat (node 3 in the flat deck is the port 'out' -> "3")
    d = float(np.max(np.abs(np.interp(flat["time"], sub["time"], sub["3"])
                            - flat["3"])))
    results.append(("L72 inverter .tran subckt==flat", d < NR_EQUIV_TOL,
                    f"max|dV|={d:.3e} V"))

    # (b) subckt vs NGSPICE ground truth
    m = _nrmse_vs_ngspice(ng, sub["time"], sub["3"], "v(out)", config)
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
    for name, fn in [("Level 1 (linear equivalence)", level1),
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
