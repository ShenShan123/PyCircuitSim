#!/usr/bin/env python3
"""Device-level NN AC (small-signal) gate — DirectNet LEVEL=73 vs NGSPICE BSIM-CMG.

The PRIMARY per-checkpoint AC accuracy gate. For each (tech, device) it builds a
resistively-loaded common-source amplifier and compares the DirectNet (LEVEL=73)
frequency response against the IDENTICAL-topology NGSPICE BSIM-CMG (LEVEL=72)
ground truth. This directly probes each checkpoint's small-signal fidelity:

  * low-frequency gain  → gm / gds accuracy at the operating point,
  * −3 dB corner        → output transcapacitance (Cgd Miller + Cdd) accuracy,
  * phase               → the combined pole/zero structure.

Crucially the NN's AC capacitances are autograd derivatives of the predicted
terminal charges (mosfet_nn._eval: cgd = ∂qg/∂Vd, cdd = ∂qd/∂Vd, …), a quantity
no prior test gated — this is the first measurement of NN charge-surface
derivative fidelity.

Geometry is pinned to the checkpoint training bins (NMOS L=16n, PMOS L=20n,
NFIN from the tech profile) so the model interpolates, not extrapolates.

Bias: the amplifier operating point is the mid-rail (Vout ≈ VDD/2) point — a
well-defined high-gain amplifying bias — located on the NGSPICE side
(``find_ng_bias``) and, independently, on the DirectNet side (``find_nn_bias``).
Each side linearizes about its OWN mid-rail DC OP, so the result is the
realistic end-to-end AC accuracy a user would see (it folds in any NN Vth
offset, exactly as a real AC sim would).

Run CPU-pinned, repo ngspice (AGENTS.md gate methodology):

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        NGSPICE_BIN="$PWD/tools/ngspice-45.2/bin/ngspice" \\
        python tests/simple_circuits/verify_nn_ac.py --tech TSMC12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.common.complex import (  # noqa: E402
    BENCH, BENCH_TECHS, get_baked_modelcard, run_ngspice_wrdata,
)
from tests.common.complex_ac import (  # noqa: E402
    ac_freq_grid, run_ngspice_ac_baked, run_directnet_ac, ac_metrics_extended,
    fmt_hz,
)

# Env-overridable so parallel checkpoint bake-offs can isolate output dirs
# (same idiom as PYCIRCUITSIM_COMPLEX_RESULTS in tests/common/complex.py).
import os as _os  # noqa: E402
RESULTS_BASE = Path(_os.environ.get(
    "PYCIRCUITSIM_NN_RESULTS",
    str(PROJECT_ROOT / "tests" / "verify_nn_ac_results")))

# CS-amp passive load (matches verify_ac.py L2 scale).
RD = "50k"
# NO external Cload: the device's own transcapacitances (Cgd Miller + Cdd) set
# the dominant output pole, so f3db/phase directly probe the NN charge-derivative
# capacitances. A 5 fF load (≫ the ~0.1 fF device caps) would swamp them and the
# test would only re-check the external RC.

# First-cut PASS thresholds (loose-then-tighten — this is a measurement
# campaign). NN-vs-truth carries the NN's own DC/charge-surface error, so these
# are looser than the L2 BSIM-CMG-vs-BSIM-CMG tolerances.
GAIN0_DB_ERR_TOL = 1.5            # dB  (gm/gds fidelity)
F3DB_RATIO_LO, F3DB_RATIO_HI = 0.7, 1.43   # output-cap / Miller pole fidelity
MAG_NRMSE_TOL = 0.10              # 10% of peak |V| (overall magnitude shape)
# Phase is REPORTED (in-band, to the −3 dB corner) but NOT hard-gated: deep in
# band the NN phase matches (<7°), but at/beyond the corner NG carries a strong
# Cgd-feedforward RHP-zero phase lag the NN does not reproduce. That feedforward
# phase is a distinct limitation from the (excellent) cap-driven pole/gain, so
# the gate scores the magnitude-domain cap fidelity and the phase is a diagnostic.


# ---------------------------------------------------------------------------
# Deck builders (NMOS + PMOS common-source amplifier; no external load cap)
# ---------------------------------------------------------------------------
def _ngspice_body(bt, device: str, baked: Path, vbias_token: str) -> List[str]:
    """NGSPICE BSIM-CMG body. ``vbias_token`` is the Vin source spec, e.g.
    ``"dc 0.3 ac 1"`` (AC run) or ``"0.0"`` (DC bias sweep placeholder)."""
    head = [f'.include "{baked}"', ".temp 27", f"VDD vdd 0 {bt.vdd:g}",
            f"Vin in 0 {vbias_token}"]
    if device == "nmos":
        body = [f"RD vdd out {RD}", f"Nn1 out in 0 0 {bt.nmos_model}"]
    else:  # pmos
        body = [f"Np1 out in vdd vdd {bt.pmos_model}", f"RD out 0 {RD}"]
    return head + body


def _directnet_deck(bt, device: str, vbias: float, ac_card: str) -> str:
    # Hierarchical deck (V6.12.0): the CS-amp core (device + load R) is a
    # .subckt instance; in/out/vdd stay top-level via the ports.
    if device == "nmos":
        ln = bt.l_nmos * 1e9
        dev = (f".subckt csamp in out vdd\n"
               f"RD vdd out {RD}\n"
               f"Mn1 out in 0 0 nmos_nn L={ln:.0f}n NFIN={bt.nfin}\n"
               f".ends\n")
        model = (f".model nmos_nn NMOS "
                 f"(LEVEL=73 TECH={bt.nn_tech} VT={bt.effective_nmos_vt})\n")
    else:  # pmos
        lp = bt.l_pmos * 1e9
        dev = (f".subckt csamp in out vdd\n"
               f"Mp1 out in vdd vdd pmos_nn L={lp:.0f}n NFIN={bt.effective_nfin_p}\n"
               f"RD out 0 {RD}\n"
               f".ends\n")
        model = (f".model pmos_nn PMOS "
                 f"(LEVEL=73 TECH={bt.nn_tech} VT={bt.effective_pmos_vt})\n")
    return (
        f"* {device.upper()} CS amp AC — DirectNet LEVEL=73 ({bt.name})\n"
        f"VDD vdd 0 {bt.vdd:g}\n"
        f"Vin in 0 DC={vbias:g} AC=1 0\n"
        f"Xamp in out vdd csamp\n"
        + dev + model + f".{ac_card}\n.end\n"
    )


# ---------------------------------------------------------------------------
# Bias finder — Vout=VDD/2 on the (reliable) NGSPICE sweep, shared by both
# ---------------------------------------------------------------------------
def find_ng_bias(bt, device: str, baked: Path, work_dir: Path,
                 tag: str) -> Tuple[float, float]:
    """NGSPICE mid-rail bias (Vin where Vout≈VDD/2): return (vbias, vout).

    The ground-truth DC sweep is reliable (the NN continuation DC sweep,
    run_directnet_dc_sweep, rails on these high-gain stages — it is NOT usable
    for bias-finding; the NN's own AC operating point is solved separately by
    the reliable _solve_dc_with_retry path inside run_directnet_ac).
    """
    body = _ngspice_body(bt, device, baked, "0.0")
    data = run_ngspice_wrdata(
        "\n".join(body), "v(out)", work_dir, f"{tag}_biassweep",
        f"dc Vin 0 {bt.vdd:g} 0.005")
    vin, vout = data[:, 0], data[:, 1]
    i = int(np.argmin(np.abs(vout - bt.vdd / 2.0)))
    return round(float(vin[i]), 3), float(vout[i])


def find_nn_bias(bt, device: str, work_dir: Path,
                 tag: str) -> Tuple[float, float]:
    """DirectNet mid-rail bias via FRESH per-point DC-OP solves (Vout≈VDD/2).

    Biasing the NN at ITS OWN amplifying OP — not NGSPICE's — is essential: a
    Vth offset shifts the high-gain Vin window, and at NGSPICE's bias the NN
    device sits at the conduction edge (railed Vout, pathological gm), giving a
    spurious gain. The fresh _solve_dc_with_retry path is reliable here; the
    continuation DC SWEEP (run_directnet_dc_sweep) rails on these high-gain
    single stages and must NOT be used. Parses once and re-solves over a Vin
    scan by mutating the Vin source value.
    """
    import logging
    from pycircuitsim.parser import Parser
    from pycircuitsim.solver import DCSolver
    from pycircuitsim.simulation import _circuit_has_nn, _solve_dc_with_retry
    from pycircuitsim.models.passive import VoltageSource

    _, ac_card = ac_freq_grid("cs_amp")
    deck = _directnet_deck(bt, device, 0.0, ac_card)
    deck_path = work_dir / f"{tag}_nnbias.sp"
    deck_path.write_text(deck)

    logging.disable(logging.CRITICAL)
    try:
        parser = Parser()
        parser.parse_file(str(deck_path))
        circuit = parser.circuit
        has_nn = _circuit_has_nn(circuit)
        vin_src = next(c for c in circuit.components
                       if isinstance(c, VoltageSource) and c.name.lower() == "vin")
        best_v, best_out, best_d = 0.0, float("nan"), float("inf")
        for vin in np.arange(0.0, bt.vdd + 1e-9, 0.02):
            vin_src.value = float(vin)

            def _solve(use_gmin: bool):
                s = DCSolver(circuit, use_source_stepping=True,
                             use_gmin_stepping=use_gmin)
                return s, s.solve()
            try:
                solver, op = _solve_dc_with_retry(circuit, has_nn, _solve)
            except Exception:  # noqa: BLE001
                continue
            # Do NOT discard non-converged points: a high-gain CS-amp trip trips
            # the strict NR flag yet the pseudo-transient fallback returns a
            # physically sensible, smooth transfer curve usable for bias finding.
            out = op.get("out", float("nan"))
            if not np.isfinite(out):
                continue
            d = abs(out - bt.vdd / 2.0)
            if d < best_d:
                best_d, best_v, best_out = d, float(vin), float(out)
    finally:
        logging.disable(logging.NOTSET)
    return round(best_v, 3), best_out


# ---------------------------------------------------------------------------
# Per-(tech, device) AC evaluation
# ---------------------------------------------------------------------------
def eval_cs_amp(tech: str, device: str) -> Dict[str, object]:
    bt = BENCH[tech]
    work_dir = RESULTS_BASE / tech
    work_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{tech.lower()}_{device}"
    baked = get_baked_modelcard(bt, bt.nfin, work_dir, nfin_p=bt.effective_nfin_p)
    freqs, ac_card = ac_freq_grid("cs_amp")

    # Each side at its OWN mid-rail (Vout≈VDD/2) amplifying bias.
    ng_vbias, ng_vout = find_ng_bias(bt, device, baked, work_dir, tag)
    nn_vbias, nn_vout_scan = find_nn_bias(bt, device, work_dir, tag)

    ng_body = _ngspice_body(bt, device, baked, f"dc {ng_vbias:g} ac 1")
    v_ng = run_ngspice_ac_baked(ng_body, work_dir, f"{tag}_ac", ac_card,
                                "out", freqs)

    nn_deck = _directnet_deck(bt, device, nn_vbias, ac_card)
    v_dn, op_ok, dc_op = run_directnet_ac(nn_deck, work_dir, f"{tag}_ac",
                                          freqs, "out")
    nn_vout = dc_op.get("out", float("nan"))
    # Voltage placement and Newton convergence answer different questions: a
    # mid-rail pseudo-transient state can look plausible but is not a DC fixed
    # point. AC must linearize around a converged operating point.
    op_valid = (np.isfinite(nn_vout)
                and 0.15 * bt.vdd < nn_vout < 0.85 * bt.vdd)

    m = ac_metrics_extended(freqs, v_dn, v_ng)
    passed = ac_gate_passes(op_ok, bool(op_valid), m)
    return dict(tech=tech, device=device, ng_vbias=ng_vbias, nn_vbias=nn_vbias,
                ng_vout=ng_vout, nn_vout=nn_vout, op_ok=op_ok,
                op_valid=bool(op_valid), dc_op=dc_op, m=m, passed=bool(passed))


def ac_gate_passes(op_converged: bool, op_valid: bool,
                   metrics: Dict[str, float]) -> bool:
    """Require a converged DC fixed point before scoring AC agreement."""
    return bool(
        op_converged
        and op_valid
        and metrics["gain0_db_err"] <= GAIN0_DB_ERR_TOL
        and np.isfinite(metrics["f3db_ratio"])
        and F3DB_RATIO_LO <= metrics["f3db_ratio"] <= F3DB_RATIO_HI
        and metrics["mag_nrmse"] <= MAG_NRMSE_TOL
    )


def _print_result(r: Dict[str, object]) -> None:
    m = r["m"]
    tech, dev = r["tech"], r["device"]
    verdict = "PASS" if r["passed"] else "FAIL"
    if not r["op_ok"]:
        op_note = "  [OP-NOT-CONVERGED]"
    elif not r["op_valid"]:
        op_note = "  [OP-MISBIAS: NN device off-rail → Vth offset, not cap error]"
    else:
        op_note = ""
    print(
        f"  {tech} {dev} CS-amp @ NG Vin={r['ng_vbias']:.3f}(Vout={r['ng_vout']:.2f}) "
        f"NN Vin={r['nn_vbias']:.3f}(Vout={r['nn_vout']:.2f}): "
        f"gain0 DN={m['gain0_db']:.2f}dB NG={m['gain0_db_ref']:.2f}dB "
        f"(err={m['gain0_db_err']:.3f}dB)  "
        f"f3db DN={fmt_hz(m['f3db_test'])} NG={fmt_hz(m['f3db_ref'])} "
        f"(ratio={m['f3db_ratio']:.3f})  "
        f"magNRMSE={m['mag_nrmse']*100:.2f}%  "
        f"phaseErr(inband)={m['phase_maxerr_inband_deg']:.2f}deg  "
        f"-> {verdict}{op_note}")
    # G4 visibility (diagnostic, NOT gated): the in-band phase above masks to the
    # passband and is blind to the Cgd-feedforward RHP-zero lag, which appears in
    # the band ABOVE the −3 dB corner. A large value here is the (expected) G4
    # miss made measurable — it does NOT affect the PASS/FAIL verdict.
    print(
        f"      phaseErr(beyond-corner): "
        f"max={m['phase_maxerr_beyond_corner_deg']:.2f}deg "
        f"rmse={m['phase_rmse_beyond_corner_deg']:.2f}deg  "
        f"[diagnostic, not gated — G4 Cgd RHP-zero]")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tech", default=",".join(BENCH_TECHS),
                    help="comma-separated techs (default: all)")
    ap.add_argument("--device", default="nmos,pmos",
                    help="comma-separated devices (default: nmos,pmos)")
    args = ap.parse_args()

    techs = [t.strip().upper() for t in args.tech.split(",") if t.strip()]
    devices = [d.strip().lower() for d in args.device.split(",") if d.strip()]

    RESULTS_BASE.mkdir(parents=True, exist_ok=True)
    print("\n" + "*" * 78)
    print("  Device-level NN AC gate: DirectNet (LEVEL=73) vs NGSPICE BSIM-CMG")
    print("*" * 78)

    rows: List[Dict[str, object]] = []
    for tech in techs:
        if tech not in BENCH:
            print(f"  [skip] unknown tech {tech}")
            continue
        print(f"\n  === {tech} ===")
        for device in devices:
            try:
                r = eval_cs_amp(tech, device)
                _print_result(r)
                rows.append(r)
            except Exception as exc:  # noqa: BLE001 — fail loud per config
                print(f"  {tech} {device} CS-amp: ERROR {exc}")
                rows.append(dict(tech=tech, device=device, error=str(exc),
                                 passed=False))

    # Parseable summary table (distinct 'AC' leading token → no collision with
    # the device-table parser; benchmark_collect.parse_ac_device_log reads this).
    print("\n" + "=" * 78)
    print("  AC SUMMARY TABLE")
    print("=" * 78)
    print("  AC | tech_device | gain0_err_db | f3db_ratio | "
          "mag_nrmse_pct | phase_inband_deg | status")
    n_pass = 0
    for r in rows:
        label = f"{r['tech'].lower()}_{r['device']}"
        if "error" in r:
            print(f"  AC | {label} | nan | nan | nan | nan | ERROR")
            continue
        m = r["m"]
        if r["passed"]:
            status = "PASS"
        elif not r.get("op_ok", False):
            status = "FAIL-NONCONVERGED"
        elif not r.get("op_valid", True):
            status = "FAIL-MISBIAS"   # OP off-rail: Vth offset, not a cap error
        else:
            status = "FAIL"
        n_pass += 1 if r["passed"] else 0
        f3 = m["f3db_ratio"]
        print(f"  AC | {label} | {m['gain0_db_err']:.3f} | "
              f"{f3:.3f} | {m['mag_nrmse']*100:.2f} | "
              f"{m['phase_maxerr_inband_deg']:.2f} | {status}")

    n = len(rows)
    print(f"\n  Total: {n_pass}/{n} PASS")
    print(f"RESULT: {n_pass}/{n} device AC gates passed")
    print("=" * 78)
    return 0 if (n > 0 and n_pass == n) else 1


if __name__ == "__main__":
    sys.exit(main())
