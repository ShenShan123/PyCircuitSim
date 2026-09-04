#!/usr/bin/env python3
"""NGSPICE-validated AC (small-signal frequency-domain) verification.

Two levels, both gated against ground truth (never a self-defined equation):

  Level 1 -- passive RC low-pass.  PyCircuitSim vs NGSPICE `.ac` AND the
             closed-form transfer function H(jw)=1/(1+jwRC).  Pure-linear, so
             every path must agree to ~machine precision.  Isolates the
             complex-MNA / R / C / AC-source stamping from the MOSFET model.

  Level 2 -- BSIM-CMG (LEVEL=72) NMOS common-source amplifier.  PyCircuitSim
             vs NGSPICE `.ac` on the IDENTICAL OSDI model (the ground truth).
             Exercises the small-signal gm/gds/gmb + transcapacitance (Miller
             Cgd) stamping that sets the device roll-off.  Bulk is tied to
             source so the source-referenced condensed capacitance matrix is
             exact.

  Level 3 -- floating-bulk CS amplifiers (NMOS bulk behind 100k to ground,
             PMOS bulk behind 100k to VDD).  Every prior AC gate tied the
             bulk to a rail, which masked both V7.5.2 AC defects: the 3x3
             cap expansion has no bulk row/column (the same sign-flip
             hazard that broke the charge-pump transient in V7.5.1) and
             the 3-conductance stamp is blind to junction conductances.
             Gates v(out) AND the bulk node itself against NGSPICE.

Run CPU-pinned, with the repo ngspice (AGENTS.md gate methodology):

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        NGSPICE_BIN="$PWD/tools/ngspice-45.2/bin/ngspice" \\
        python tests/simple_circuits/verify_ac.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.common.base import (
    OSDI_PATH,
    control_deck,
    template_deck,
    render_template,
    run_ngspice_subprocess,
)
from tests.common.bsimcmg_tran import (
    ALL_TECHS,
    get_merged_modelcard,
    make_baseline,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "tests" / "ac"

# Levels 1 and 2 own stimulus and tolerances, not topology: both circuits are
# decks in circuit_templates/ (V7.5.9 — they had been f-strings here AND decks there,
# the drift hazard tests/__init__ names). Level 3's floating-bulk pair stays
# programmatic: it sweeps polarity, so no single deck expresses it.
RC_TEMPLATE = control_deck("rc_lowpass.spice.tmpl")
CS_TEMPLATE = template_deck("common_source.spice.tmpl")

# Acceptance thresholds.  L1 is purely linear so it must match tightly; L2
# carries the OP match (~0.05% from the DC suite) plus the cap condensation.
L1_MAG_NRMSE = 0.005      # 0.5% of peak |V|
L1_PHASE_MAXERR = 0.5     # degrees
L2_MAG_NRMSE = 0.03       # 3% of peak |V|
L2_GAIN_MAXERR_DB = 0.5   # dB
L2_PHASE_MAXERR = 5.0     # degrees


# ---------------------------------------------------------------------------
# Frequency grid (matches an NGSPICE `.ac dec N f1 f2` decade sweep)
# ---------------------------------------------------------------------------
def dec_frequencies(n_per_decade: int, fstart: float, fstop: float) -> np.ndarray:
    """Decade-spaced grid identical to NGSPICE `.ac dec N fstart fstop`."""
    n_dec = np.log10(fstop / fstart)
    npts = int(round(n_per_decade * n_dec)) + 1
    return np.logspace(np.log10(fstart), np.log10(fstop), npts)


# ---------------------------------------------------------------------------
# PyCircuitSim AC runner
# ---------------------------------------------------------------------------
def run_pycircuitsim_ac(
    deck_text: str,
    work_dir: Path,
    label: str,
    frequencies: np.ndarray,
    out_node: str,
    modelcard_path: Optional[str] = None,
    model_name_map: Optional[Dict[str, str]] = None,
) -> np.ndarray:
    """Parse a deck, solve the DC OP, run AC, return complex V(out_node)."""
    from pycircuitsim.parser import Parser
    from pycircuitsim.solver import DCSolver, ACSolver

    deck_path = work_dir / f"pycircuitsim_{label}.sp"
    deck_path.write_text(deck_text)

    logging.disable(logging.CRITICAL)
    try:
        if modelcard_path is not None:
            parser = Parser(modelcard_path=modelcard_path,
                            model_name_map=model_name_map)
        else:
            parser = Parser()
        parser.parse_file(str(deck_path))
        circuit = parser.circuit

        # DC operating point (AC sources contribute 0 to the DC solve).
        dc_solver = DCSolver(circuit, use_source_stepping=True)
        dc_solution = dc_solver.solve()

        ac_solver = ACSolver(circuit, dc_solution=dc_solution)
        ac_results = ac_solver.solve(frequencies)
    finally:
        logging.disable(logging.NOTSET)

    return np.asarray(ac_results[out_node], dtype=complex)


# ---------------------------------------------------------------------------
# NGSPICE AC runner
# ---------------------------------------------------------------------------
def run_ngspice_ac(
    deck_text: str,
    work_dir: Path,
    label: str,
    ac_card: str,
    out_node: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run NGSPICE `.ac` and return (frequency, complex V(out_node)).

    Uses `wrdata v(out)` which writes columns [frequency, real, imag] for the
    single requested vector -- unambiguous (no degrees/radians ambiguity).
    """
    deck_path = work_dir / f"ngspice_{label}.cir"
    deck_path.write_text(deck_text)

    csv_path = work_dir / f"ngspice_{label}.csv"
    log_path = work_dir / f"ngspice_{label}.log"
    runner_path = work_dir / f"ngspice_{label}_runner.cir"

    # NOTE: line 1 is consumed by ngspice as the title -- keep the comment.
    runner_content = f"""\
* NGSPICE AC runner ({label})
.control
osdi {OSDI_PATH}
source {deck_path}
set filetype=ascii
set wr_vecnames
{ac_card}
wrdata {csv_path} v({out_node})
.endc
.end
"""
    runner_path.write_text(runner_content)

    lines = run_ngspice_subprocess(runner_path, log_path, csv_path)

    rows: List[List[float]] = []
    for line in lines[1:]:  # skip the vecnames header
        s = line.strip()
        if s:
            rows.append([float(x) for x in s.split()])
    data = np.array(rows)
    freq = data[:, 0]
    vout = data[:, 1] + 1j * data[:, 2]
    return freq, vout


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _interp_to(freq_ref: np.ndarray, freq_src: np.ndarray,
               v_src: np.ndarray) -> np.ndarray:
    """Interpolate a complex response onto freq_ref via log-f mag+unwrapped phase."""
    lf_ref = np.log10(freq_ref)
    lf_src = np.log10(freq_src)
    mag = np.interp(lf_ref, lf_src, np.abs(v_src))
    ph = np.interp(lf_ref, lf_src, np.unwrap(np.angle(v_src)))
    return mag * np.exp(1j * ph)


def ac_metrics(freq: np.ndarray, v_test: np.ndarray,
               v_ref: np.ndarray) -> Dict[str, float]:
    """Magnitude/phase agreement of v_test against v_ref on a common grid."""
    mag_t, mag_r = np.abs(v_test), np.abs(v_ref)
    rmse = float(np.sqrt(np.mean((mag_t - mag_r) ** 2)))
    mag_nrmse = rmse / (float(np.max(mag_r)) + 1e-30)

    db_t = 20.0 * np.log10(mag_t + 1e-30)
    db_r = 20.0 * np.log10(mag_r + 1e-30)
    gain_maxerr_db = float(np.max(np.abs(db_t - db_r)))

    ph_t = np.rad2deg(np.unwrap(np.angle(v_test)))
    ph_r = np.rad2deg(np.unwrap(np.angle(v_ref)))
    phase_maxerr = float(np.max(np.abs(ph_t - ph_r)))

    return {
        "mag_nrmse": mag_nrmse,
        "gain_maxerr_db": gain_maxerr_db,
        "phase_maxerr_deg": phase_maxerr,
        "gain0_db": float(db_t[0]),
        "gain0_db_ref": float(db_r[0]),
    }


def minus3db_corner(freq: np.ndarray, v: np.ndarray) -> float:
    """First frequency where |V| drops 3 dB below its low-frequency value."""
    db = 20.0 * np.log10(np.abs(v) + 1e-30)
    db0 = db[0]
    below = np.where(db - db0 <= -3.0)[0]
    return float(freq[below[0]]) if below.size else float("nan")


def _deck_card(deck: str, name: str) -> List[str]:
    """The whitespace-split card named ``name``, e.g. ``R1 in out 1k``."""
    for line in deck.splitlines():
        parts = line.split()
        if parts and parts[0].lower() == name.lower():
            return parts
    raise KeyError(f"no card named {name!r} in the rendered deck")


def _deck_token(deck: str, name: str) -> str:
    """The value token of a two-terminal card: field 3, verbatim."""
    return _deck_card(deck, name)[3]


def _deck_value(deck: str, name: str) -> float:
    """That token as a number, through the simulator's OWN suffix table —
    so the analytic reference cannot disagree with the deck about what
    ``159.155n`` means."""
    from pycircuitsim.parser import Parser                   # noqa: PLC0415

    return Parser()._parse_value(_deck_token(deck, name))


def _deck_param(deck: str, name: str, key: str) -> str:
    """A ``KEY=value`` field on a card, e.g. ``Vin``'s ``DC=`` bias."""
    for tok in _deck_card(deck, name):
        if tok.upper().startswith(f"{key.upper()}="):
            return tok.split("=", 1)[1]
    raise KeyError(f"card {name!r} carries no {key}= field")


# ---------------------------------------------------------------------------
# Level 1 -- passive RC low-pass
# ---------------------------------------------------------------------------
def test_rc_lowpass(work_dir: Path) -> bool:
    print("\n" + "=" * 70)
    print("  Level 1: Passive RC low-pass  (PyCircuitSim vs NGSPICE vs analytic)")
    print("=" * 70)

    # R/C are read off the deck rather than declared here: the closed form is
    # only a valid reference if it is the transfer function of the circuit the
    # simulators were actually handed.
    rc_values = {
        "TEMP": "27", "INPUT_DC": "0", "INPUT_AC": "1", "INPUT_PHASE": "0",
        "RESISTANCE": "1k", "CAPACITANCE": "159.155n",
    }
    py_deck = render_template(
        RC_TEMPLATE, {**rc_values, "ANALYSIS": ".ac dec 20 10 1e6"})
    ng_deck = render_template(RC_TEMPLATE, {**rc_values, "ANALYSIS": ""})
    R = _deck_value(py_deck, "R1")
    C = _deck_value(py_deck, "C1")
    fc = 1.0 / (2 * np.pi * R * C)
    freqs = dec_frequencies(20, 10.0, 1e6)
    print(f"  template: {RC_TEMPLATE.name}")
    print(f"  R={R:.0f}ohm  C={C*1e9:.3f}nF  ->  fc = {fc:.2f} Hz")

    v_py = run_pycircuitsim_ac(py_deck, work_dir, "rc", freqs, "out")
    f_ng, v_ng_raw = run_ngspice_ac(ng_deck, work_dir, "rc",
                                    "ac dec 20 10 1e6", "out")
    v_ng = _interp_to(freqs, f_ng, v_ng_raw)
    v_an = 1.0 / (1.0 + 1j * 2 * np.pi * freqs * R * C)  # closed form

    m_ng = ac_metrics(freqs, v_py, v_ng)
    m_an = ac_metrics(freqs, v_py, v_an)

    print(f"  PyCircuitSim -3dB corner: {minus3db_corner(freqs, v_py):.2f} Hz "
          f"(analytic fc = {fc:.2f} Hz)")
    print(f"  vs NGSPICE  : mag NRMSE={m_ng['mag_nrmse']*100:.4f}%  "
          f"phase maxerr={m_ng['phase_maxerr_deg']:.4f} deg")
    print(f"  vs analytic : mag NRMSE={m_an['mag_nrmse']*100:.4f}%  "
          f"phase maxerr={m_an['phase_maxerr_deg']:.4f} deg")

    ok = (m_ng["mag_nrmse"] < L1_MAG_NRMSE
          and m_ng["phase_maxerr_deg"] < L1_PHASE_MAXERR
          and m_an["mag_nrmse"] < L1_MAG_NRMSE
          and m_an["phase_maxerr_deg"] < L1_PHASE_MAXERR)
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Level 2 -- BSIM-CMG common-source amplifier
# ---------------------------------------------------------------------------
def test_cs_amp(work_dir: Path) -> bool:
    print("\n" + "=" * 70)
    print("  Level 2: BSIM-CMG NMOS common-source amp  (PyCircuitSim vs NGSPICE)")
    print("=" * 70)

    tech = ALL_TECHS["ASAP7"]
    config = make_baseline(tech, nfin_n=2, nfin_p=2)  # ASAP7 RVT, L=30n, VDD=0.7
    vt = config.vt
    merged = get_merged_modelcard(config, work_dir)

    ac_card = "ac dec 20 1e3 1e12"
    freqs = dec_frequencies(20, 1e3, 1e12)

    # One template owns topology; the gate supplies the technology, geometry,
    # bias, passives, and simulator-specific model adapter.
    common = {
        "TEMP": "27", "VDD": f"{config.vdd:g}",
        "INPUT_SPEC": "DC=0.35 AC=1 0", "LOAD_NETWORK": "Rd vdd out 50k",
        "BULK_NETWORK": "", "SOURCE_NODE": "0", "BULK_NODE": "0",
        "OUTPUT_LOAD": "Cload out 0 5f",
    }
    py_deck = render_template(CS_TEMPLATE, {
        **common,
        "MODEL_SETUP": ".model nmos_rvt NMOS (LEVEL=72)",
        "DEVICE_PREFIX": "M",
        "DEVICE": (
            f"nmos_rvt L={config.l_nmos * 1e9:g}n "
            f"NFIN={config.nfin_n} TFIN={tech.tfin * 1e9:g}n"
        ),
        "ANALYSIS": ".ac dec 20 1e3 1e12",
    })
    vdd = _deck_value(py_deck, "VDD")
    vbias = _deck_param(py_deck, "Vin", "DC")
    rd = _deck_token(py_deck, "RD")
    cload = _deck_token(py_deck, "Cload")
    ng_deck = render_template(CS_TEMPLATE, {
        **common,
        "MODEL_SETUP": f'.include "{merged_baked(config, work_dir)}"',
        "DEVICE_PREFIX": "N", "DEVICE": vt.nmos_model,
        "ANALYSIS": "",
    })

    print(f"  template: {CS_TEMPLATE.name}")
    print(f"  Tech=ASAP7 VT={vt.vt_name}  L={config.l_nmos*1e9:.0f}nm "
          f"NFIN={config.nfin_n}  VDD={vdd}V Vbias={vbias}V "
          f"RD={rd} Cload={cload}")

    v_py = run_pycircuitsim_ac(
        py_deck, work_dir, "cs_amp", freqs, "out",
        modelcard_path=str(merged),
        model_name_map={"NMOS": vt.nmos_model, "PMOS": vt.pmos_model})
    f_ng, v_ng_raw = run_ngspice_ac(ng_deck, work_dir, "cs_amp", ac_card, "out")
    v_ng = _interp_to(freqs, f_ng, v_ng_raw)

    m = ac_metrics(freqs, v_py, v_ng)
    f3_py = minus3db_corner(freqs, v_py)
    f3_ng = minus3db_corner(freqs, v_ng)

    print(f"  low-f gain: PyCircuitSim={m['gain0_db']:.3f} dB  "
          f"NGSPICE={m['gain0_db_ref']:.3f} dB")
    print(f"  -3dB corner: PyCircuitSim={f3_py:.4g} Hz  NGSPICE={f3_ng:.4g} Hz")
    print(f"  mag NRMSE={m['mag_nrmse']*100:.3f}%  "
          f"gain maxerr={m['gain_maxerr_db']:.3f} dB  "
          f"phase maxerr={m['phase_maxerr_deg']:.3f} deg")

    ok = (m["mag_nrmse"] < L2_MAG_NRMSE
          and m["gain_maxerr_db"] < L2_GAIN_MAXERR_DB
          and m["phase_maxerr_deg"] < L2_PHASE_MAXERR)
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def merged_baked(config, work_dir: Path) -> Path:
    """Baked modelcard (L/NFIN/TFIN injected) for the NGSPICE `.include`."""
    from tests.common.bsimcmg_tran import get_baked_modelcard
    return get_baked_modelcard(config, work_dir)


# ---------------------------------------------------------------------------
# Level 3 -- floating-bulk CS amplifiers (bulk NOT tied to a rail)
# ---------------------------------------------------------------------------
def _floating_bulk_case(work_dir: Path, polarity: str) -> bool:
    """One floating-bulk CS amp (NMOS or PMOS) gated on v(out) and v(bulk)."""
    tech = ALL_TECHS["ASAP7"]
    config = make_baseline(tech, nfin_n=2, nfin_p=2)
    vt = config.vt
    merged = get_merged_modelcard(config, work_dir)
    baked = merged_baked(config, work_dir)

    vdd, rd, rb = 0.7, "50k", "100k"
    tfin_nm = tech.tfin * 1e9
    ac_card = "ac dec 20 1e3 1e12"
    freqs = dec_frequencies(20, 1e3, 1e12)

    if polarity == "nmos":
        model, l_nm, nfin, vbias = (
            vt.nmos_model, config.l_nmos * 1e9, config.nfin_n, 0.35)
        # Bulk floats behind Rb to ground; junction caps couple it in.
        load_network = f"Rd vdd out {rd}"
        bulk_network = f"Rb vb 0 {rb}"
        source_node = "0"
        model_card = f".model {model} NMOS (LEVEL=72)"
    else:
        model, l_nm, nfin, vbias = (
            vt.pmos_model, config.l_pmos * 1e9, config.nfin_p, 0.35)
        load_network = f"Rd out 0 {rd}"
        bulk_network = f"Rb vb vdd {rb}"
        source_node = "vdd"
        model_card = f".model {model} PMOS (LEVEL=72)"

    common = {
        "TEMP": "27", "VDD": f"{vdd:g}",
        "INPUT_SPEC": f"DC={vbias:g} AC=1 0",
        "LOAD_NETWORK": load_network, "BULK_NETWORK": bulk_network,
        "SOURCE_NODE": source_node, "BULK_NODE": "vb", "OUTPUT_LOAD": "",
    }
    py_deck = render_template(CS_TEMPLATE, {
        **common, "MODEL_SETUP": model_card, "DEVICE_PREFIX": "M",
        "DEVICE": (
            f"{model} L={l_nm:.0f}n NFIN={nfin} TFIN={tfin_nm:.1f}n"
        ),
        "ANALYSIS": f".{ac_card}",
    })
    ng_deck = render_template(CS_TEMPLATE, {
        **common, "MODEL_SETUP": f'.include "{baked}"',
        "DEVICE_PREFIX": "N", "DEVICE": model, "ANALYSIS": "",
    })

    print(f"  [{polarity.upper()}] model={model} L={l_nm:.0f}nm NFIN={nfin} "
          f"Rb={rb} to {'gnd' if polarity == 'nmos' else 'vdd'}")

    ok = True
    for node in ("out", "vb"):
        v_py = run_pycircuitsim_ac(
            py_deck, work_dir, f"fb_{polarity}", freqs, node,
            modelcard_path=str(merged),
            model_name_map={"NMOS": vt.nmos_model, "PMOS": vt.pmos_model})
        f_ng, v_ng_raw = run_ngspice_ac(
            ng_deck, work_dir, f"fb_{polarity}_{node}", ac_card, node)
        v_ng = _interp_to(freqs, f_ng, v_ng_raw)
        if node == "out":
            m = ac_metrics(freqs, v_py, v_ng)
            node_ok = (m["mag_nrmse"] < L2_MAG_NRMSE
                       and m["gain_maxerr_db"] < L2_GAIN_MAXERR_DB
                       and m["phase_maxerr_deg"] < L2_PHASE_MAXERR)
            print(f"    v({node}): mag NRMSE={m['mag_nrmse']*100:.4f}%  "
                  f"gain maxerr={m['gain_maxerr_db']:.4f} dB  "
                  f"phase maxerr={m['phase_maxerr_deg']:.4f} deg  "
                  f"-> {'PASS' if node_ok else 'FAIL'}")
        else:
            # The bulk node can be a genuinely zero response (the PMOS
            # junctions inject nothing at this OP -- NGSPICE reads 1e-83,
            # pure roundoff), where dB/phase of noise is meaningless.
            # Gate the complex residual against 3% of the reference peak
            # with a 1 pV absolute floor: catches BOTH a wrong bulk
            # response and a spurious injection into a quiet bulk (the
            # bug class this level exists for).
            peak_ng = float(np.max(np.abs(v_ng)))
            resid = float(np.max(np.abs(v_py - v_ng)))
            limit = max(L2_MAG_NRMSE * peak_ng, 1e-12)
            node_ok = resid <= limit
            print(f"    v({node}): peak|ng|={peak_ng:.3e} V  "
                  f"max|py-ng|={resid:.3e} V  (limit {limit:.3e})  "
                  f"-> {'PASS' if node_ok else 'FAIL'}")
        ok = ok and node_ok
    return ok


def test_floating_bulk(work_dir: Path) -> bool:
    print("\n" + "=" * 70)
    print("  Level 3: floating-bulk CS amps  (PyCircuitSim vs NGSPICE)")
    print("=" * 70)
    ok_n = _floating_bulk_case(work_dir, "nmos")
    ok_p = _floating_bulk_case(work_dir, "pmos")
    ok = ok_n and ok_p
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("\n" + "*" * 70)
    print("  AC Analysis Verification: PyCircuitSim vs NGSPICE (BSIM-CMG ground truth)")
    print("*" * 70)

    results = {
        "L1 RC low-pass": test_rc_lowpass(RESULTS_DIR),
        "L2 CS amplifier": test_cs_amp(RESULTS_DIR),
        "L3 floating bulk": test_floating_bulk(RESULTS_DIR),
    }

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for name, ok in results.items():
        print(f"  {name:24s}: {'PASS' if ok else 'FAIL'}")
    n_pass = sum(results.values())
    print(f"\n  Total: {n_pass}/{len(results)} passed")
    all_ok = all(results.values())
    print("  All AC tests PASSED." if all_ok else "  Some AC tests FAILED.")
    print("=" * 70)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
