#!/usr/bin/env python3
"""Benchmark 3c — 6T SRAM read SNM: DirectNet vs NGSPICE BSIM-CMG.

Part of the DirectNet V6.4 sprint, Phase 3 benchmark harness.

Traces the read static-noise-margin (SNM) butterfly of a 6T SRAM bitcell.
Each butterfly lobe is the voltage-transfer curve of one half-cell under read
bias (WL = VDD, both bit lines held at VDD), obtained by breaking the
cross-coupled feedback and DC-sweeping the driven storage node. The mirror
lobe swaps q <-> qb. SNM is the side of the largest square that fits between
the two lobes.

Both lobes are produced by:
  (i)  DirectNet-Full (LEVEL=75) half-cell DC sweeps (PyCircuitSim),
  (ii) the NGSPICE BSIM-CMG (LEVEL=72) ground truth.

It additionally solves the full cross-coupled 6T cell with ``force_ic=True``
(hard .ic mode) to confirm both storage states land on their rails — the same
solver path SRAM latches need.

Gate: across NFIN corners, both DirectNet butterfly curves are positive
(Vout >= 0) AND track the NGSPICE BSIM-CMG reference point-by-point
(curve NRMSE <= 10%). The derived SNM *scalar* error and the full-6T
``force_ic`` convergence probe are reported as diagnostics only — they do
NOT enter the verdict (force_ic is a self-consistency probe, not an NGSPICE
comparison; the SNM scalar is too geometry-sensitive to gate on directly).

Ground truth is ALWAYS NGSPICE BSIM-CMG (AGENTS.md Validation rule).
Report MRE / R2 / NRMSE / MaxErr.

Usage:
    conda run -n pycircuitsim python tests/simple_circuits/verify_circuit_sram_snm.py
    conda run -n pycircuitsim python tests/simple_circuits/verify_circuit_sram_snm.py --tech TSMC5
    conda run -n pycircuitsim python tests/simple_circuits/verify_circuit_sram_snm.py --nfin 2,5,10
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "bsim_cmg" / "tests"))

from tests.common.circuit_benchmarks import (  # noqa: E402
    BENCH, BENCH_TECHS, RESULTS_BASE, BenchTech, active_model_label,
    active_model_name,
    get_baked_modelcard, run_ngspice_wrdata, parse_netlist, full_metrics,
    SramParams, ngspice_sram_lobe, directnet_sram_lobe,
    directnet_sram_6t,
)
from tests.common.gate_result import GateResult  # noqa: E402
from tests.common.base import parse_csv_choices  # noqa: E402
from tests.common.simple_circuit_harness import RunSpec  # noqa: E402

DEFAULT_NFINS = [2, 5, 10]
# Ground-truth tracking gate: the DirectNet butterfly lobe must match the
# NGSPICE BSIM-CMG lobe point-by-point to within this NRMSE (10%, the suite-wide
# NN tolerance). This is the actual "SNM tracking the NGSPICE reference" gate
# the docstring promises — it is robust where the derived SNM *scalar* is not
# (a single corner can swing the geometric SNM 60%+ while the curve NRMSE stays
# ~3%). The SNM scalar (snm_err) and the force_ic convergence probe are reported
# as diagnostics but are NOT part of the verdict (bug report B5).
SRAM_NRMSE_TOL = 0.10



# ---------------------------------------------------------------------------
# Half-cell butterfly lobe
# ---------------------------------------------------------------------------
def ngspice_sram_lobe_body(bt: BenchTech, nfin: int, baked: Path) -> Dict[str, str]:
    """Single-point NGSPICE SRAM half-cell ground-truth deck body.

    The topology is NOT here — it is ``circuit_templates/
    sram_snm_half_cell.spice.tmpl``, rendered per tech. This function owns the read
    bias and the sweep, nothing else. ``nfin`` reaches the devices through the
    baked modelcard, which is why it appears in the signature but not in the
    substitutions.

    Pure (returns text) so verify_circuit_sweep_canaries can diff it against the
    parametric ``tests.common.circuit_benchmarks.ngspice_sram_lobe`` builder (B7/B8)."""
    return ngspice_sram_lobe(bt, SramParams(), nfin, baked)


def ngspice_lobe(bt: BenchTech, nfin: int, work_dir: Path) -> Dict[str, np.ndarray]:
    """One NGSPICE half-cell butterfly lobe: qb(q) under read bias."""
    baked = get_baked_modelcard(bt, nfin, work_dir)
    spec = ngspice_sram_lobe_body(bt, nfin, baked)
    data = run_ngspice_wrdata(spec["body"], spec["signals"], work_dir,
                              f"sram_{bt.name}_nfin{nfin}", spec["analysis"])
    return {"q": data[:, 0], "qb": data[:, 1]}


def directnet_sram_lobe_deck(bt: BenchTech, nfin: int) -> str:
    """Single-point DirectNet SRAM half-cell ship-gate deck text (broken
    feedback).

    The topology is NOT here — it is ``circuit_templates/
    sram_snm_half_cell.spice.tmpl``, the same arrangement the other three
    simple-circuit gates already use. Until V7.5.8 this gate was the one that
    carried its own
    copy, which is exactly why its ``circuit_templates/`` deck had been free to drift
    (and had: the 6T deck still documented a read bias the gate stopped using
    in V6.4.7).

    Pure text so the canary diffs the REAL deck against the sweep builder
    (B7/B8)."""
    return directnet_sram_lobe(bt, SramParams(), nfin)


def _directnet_halfcell_netlist(bt: BenchTech, nfin: int, path: Path) -> Path:
    """A DirectNet-Full SRAM half-cell with broken feedback."""
    path.write_text(directnet_sram_lobe_deck(bt, nfin))
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
        results = run_dc_sweep(
            parser.circuit, parser.analysis_params, Visualizer(), out_dir,
            f"sram_{bt.name}_{nfin}", require_convergence=True,
        )
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
                          path: Path, wl_on: bool = False) -> Path:
    # force_ic / retention test => wordline OFF (wl=0): the access transistors
    # are OFF and the cross-coupled latch holds its `.ic` state in isolation.
    # This is the physically-correct HOLD condition.
    #
    # V6.4.7 S17b/S17c (2026-06-16) GROUND-TRUTH-PROVEN HARNESS FIX. The old test
    # pinned wl=VDD (access ON) with BOTH bitlines forced to VDD by ideal
    # sources — a non-physical harsh READ-DISTURB, not a hold. Under it the "0"
    # node settles at the read-SNM level (~0.18*VDD), which exceeds the 0.1*VDD
    # rail band ⇒ a guaranteed 0/8. The native LEVEL=72 OSDI control (exact
    # BSIM-CMG physics) fails wl=ON 0/8 and passes wl=OFF 8/8 IDENTICALLY to the
    # NN ⇒ the wl=ON gate rejected ground-truth physics (mis-specified); wl=OFF
    # is the correction (ground truth passes it). Read-stability is separately
    # covered by the butterfly SNM gate and the paired simple-v2 read mode;
    # callers cannot silently re-enable the retired wl=ON probe. See
    # results/v6_4_7/S17c_forceic_harness_fix.md.
    # Topology + hold bias come from the canonical SRAM-mode template.
    # deck. Only the two storage-state initial conditions vary per call.
    if wl_on:
        raise ValueError(
            "wl_on read-disturb moved to the paired sram6t_modes diagnostic")
    text = directnet_sram_6t(bt, q_init, qb_init, bt.nfin)
    path.write_text(text)
    return path


def force_ic_probe(bt: BenchTech, work_dir: Path) -> Dict[str, bool]:
    """Solve the full 6T cell with force_ic=True for both storage states.

    V6.4.6 Phase 1 — hardened acceptance. The old accept was
    ``_last_solve_converged AND |q-q0|<VDD/4 AND |qb-qb0|<VDD/4``: a
    convergence flag (stale on the old early-return force_ic path) plus
    rail-proximity, with NO KCL-residual check — a pinned-node artifact
    could false-PASS. The solver now exposes the *released* (fully
    unconstrained) solution's KCL residual + per-solve threshold
    (``_last_dc_residual`` / ``_last_dc_resid_threshold``); acceptance is

        residual <= threshold        (a TRUE unconstrained fixed point)
        AND |q-q0|  < 0.1*VDD         (the "1" node held near its rail)
        AND |qb-qb0| < 0.1*VDD        (the "0" node held near its rail)

    Both must hold. The residual gate rejects a non-physical iterate; the
    rail-proximity gate rejects the inboard attractor (also a real fixed
    point). The band is **0.1*VDD, NOT VDD/4** — V6.4.6 review showed the
    VDD/4 (=25 % VDD) band false-PASSES the *documented failure* attractor
    (q≈0.87 / qb≈0.20, i.e. the storage-"0" node parked at 24–30 % VDD ≈ 1×
    the true SNM above ground): for the 0.80 V techs qb≈0.19 < 0.20 sneaks
    inside VDD/4 purely by VDD-scaling, while TSMC5 rails *closer* in
    absolute volts (qb=0.163) yet fell outside its smaller band. A 10 %-VDD
    band sits well below the 24–30 % attractor on every tech, so it reports
    the honest result while still accepting a genuinely railed solution.

    This probe is a DIAGNOSTIC, not a gate (bug report B5): its result is
    printed but does NOT enter the tech verdict. With the V6.5.x production
    checkpoints it rails on TSMC7/TSMC12 and lands in the inboard basin on
    TSMC5/TSMC16 (i.e. it currently passes 2/4, not the historical wl=ON
    "0/8") — the SRAM read-stability gate is the butterfly positivity +
    NGSPICE-NRMSE tracking, not this convergence probe. See
    results/v6_4_6/phase1_force_ic_recovery.md.
    """
    from pycircuitsim.solver import DCSolver

    out = {}
    for tag, (q0, qb0) in (("state1", (bt.vdd, 0.0)),
                           ("state0", (0.0, bt.vdd))):
        netlist = _directnet_6t_netlist(
            bt, q0, qb0, work_dir / f"sram6t_{bt.name}_{tag}.sp")
        logging.disable(logging.CRITICAL)
        ok = False
        q_v = qb_v = float("nan")
        resid = thr = float("nan")
        resid_ok = rail_ok = False
        try:
            parser = parse_netlist(netlist)
            circuit = parser.circuit
            guess = circuit.initial_conditions or None
            solver = DCSolver(circuit, initial_guess=guess,
                              use_source_stepping=True, force_ic=True)
            sol = solver.solve()
            q_v = sol.get("q", float("nan"))
            qb_v = sol.get("qb", float("nan"))
            # Released-solution KCL residual gate (TRUE unconstrained fixed
            # point), set by the solver's force_ic release path.
            resid = getattr(solver, "_last_dc_residual", None)
            thr = getattr(solver, "_last_dc_resid_threshold", None)
            resid_ok = (resid is not None and thr is not None
                        and resid <= thr)
            # Landed on the seeded rail? Band is 0.1*VDD (NOT VDD/4): VDD/4
            # false-PASSES the documented inboard attractor (qb at 24-30% VDD)
            # for the higher-VDD techs. See the docstring.
            rail_band = 0.1 * bt.vdd
            rail_ok = (abs(q_v - q0) < rail_band
                       and abs(qb_v - qb0) < rail_band)
            ok = bool(resid_ok and rail_ok)
        except Exception:  # noqa: BLE001
            ok = False
        finally:
            logging.disable(logging.NOTSET)
        out[tag] = ok
        resid_s = f"{resid:.3e}" if isinstance(resid, float) else "n/a"
        thr_s = f"{thr:.3e}" if isinstance(thr, float) else "n/a"
        print(f"      force_ic {tag}: q={q_v:.3f} qb={qb_v:.3f}  "
              f"resid={resid_s} thr={thr_s} "
              f"(resid_ok={resid_ok} rail_ok={rail_ok})  "
              f"-> {'PASS' if ok else 'FAIL'}")
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

        model_label = active_model_label()
        print(f"    {model_label} butterfly lobe ...")
        try:
            dn = directnet_lobe(bt, nfin, work_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"      {model_label} FAILED: {exc!r}")
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
        nrmse_ok = metrics["nrmse_pct"] <= SRAM_NRMSE_TOL * 100
        snm_err = (abs(dn_snm - ng_snm) / ng_snm * 100.0
                   if ng_snm > 1e-6 else float("nan"))
        model_name = active_model_name()
        print(f"      NG SNM={ng_snm*1e3:.1f}mV  "
              f"{model_name} SNM={dn_snm*1e3:.1f}mV  "
              f"SNMerr={snm_err:.1f}%  "
              f"{model_name} min(qb)={dn_min*1e3:.1f}mV  "
              f"NRMSE={metrics['nrmse_pct']:.2f}% "
              f"({'ok' if nrmse_ok else 'HIGH'})")
        corner_rows.append({
            "nfin": nfin, "ng_snm": ng_snm, "dn_snm": dn_snm,
            "snm_err_pct": snm_err, "dn_min_qb": dn_min,
            "positive": positive, "nrmse_ok": nrmse_ok, **metrics})

    print("    force_ic full-6T convergence probe ...")
    try:
        fic = force_ic_probe(bt, work_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"      force_ic probe ERROR: {exc!r}")
        fic = {"state1": False, "state0": False}

    # Verdict (bug report B3 + B5): a tech PASSES only when there is at least
    # one comparable corner AND every comparable corner is both positive and
    # NGSPICE-NRMSE-tracking. ``bool(corner_rows)`` guards the empty-generator
    # false-PASS (``all([]) == True``) when every corner errored, and the
    # ground-truth ``nrmse_ok`` term closes the "wildly inaccurate but
    # non-negative lobe scores PASS" hole. ``force_ic`` stays a printed
    # diagnostic (a convergence probe, not an NGSPICE comparison).
    # an errored corner is a gate FAIL, not a silent skip: a corner that
    # cannot be characterized must not leave the remaining corners to carry
    # the verdict
    comparable = [r for r in corner_rows if "error" not in r]
    all_pass = (bool(comparable)
                and len(comparable) == len(corner_rows)
                and all(r["positive"] and r["nrmse_ok"] for r in comparable))
    return {"tech": bt.name, "corners": corner_rows,
            "force_ic": fic, "all_positive": all_pass}


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SRAM read-SNM benchmark 3c")
    ap.add_argument("--tech", default=",".join(BENCH_TECHS))
    ap.add_argument("--nfin", default=",".join(str(n) for n in DEFAULT_NFINS),
                    help="comma-separated NFIN corners")
    args = ap.parse_args(argv)
    techs = parse_csv_choices(
        ap, args.tech, flag="--tech", choices=BENCH_TECHS,
        normalize=str.upper,
    )
    nfin_fields = args.nfin.split(",")
    if not nfin_fields or any(not value.strip() for value in nfin_fields):
        ap.error(f"--nfin contains an empty value: {args.nfin!r}")
    raw_nfins = [value.strip() for value in nfin_fields]
    try:
        nfins = [int(value) for value in raw_nfins]
    except ValueError:
        ap.error(f"--nfin values must be integers: {raw_nfins}")
    if any(value <= 0 for value in nfins):
        ap.error(f"--nfin values must be positive: {nfins}")
    if len(nfins) != len(set(nfins)):
        ap.error(f"--nfin contains duplicates: {nfins}")

    print("=" * 78)
    print(f"Benchmark 3c — 6T SRAM read SNM: {active_model_label()} "
          "vs NGSPICE BSIM-CMG")
    print(f"  NFIN corners: {nfins}")
    print("  Gate: butterfly lobes positive AND NGSPICE-NRMSE-tracking "
          f"(<= {SRAM_NRMSE_TOL*100:.0f}%) across NFIN corners")
    print("  (SNMerr% and force_ic are diagnostics, not gated)")
    print("=" * 78)

    results: List[Dict] = []
    for name in techs:
        try:
            results.append(run_one(BENCH[name], nfins))
        except Exception as exc:  # noqa: BLE001
            print(f"  {name}: ERROR {exc!r}")
            results.append({"tech": name, "error": repr(exc)})

    print("\n" + "=" * 78)
    print("SUMMARY — Benchmark 3c SRAM read SNM")
    print("=" * 78)
    model_snm = f"{active_model_name()} SNM mV"
    hdr = (f"{'Tech':8s} | {'NFIN':>5s} | {'NG SNM mV':>10s} | "
           f"{model_snm:>10s} | {'SNMerr%':>8s} | {'min(qb)mV':>10s} | "
           f"{'NRMSE%':>7s} | {'Gate':>5s}")
    print(hdr)
    print("-" * len(hdr))
    n_pass = 0
    for r in results:
        if "error" in r:
            print(f"{r['tech']:8s} | ERROR — {r['error'][:54]}")
            print(GateResult(
                case_id="sram_snm", tech=r["tech"], corner="nfin_sweep",
                analysis="read_snm", role="qualification", status="error",
                error=r["error"], reference_converged=False,
                candidate_converged=False,
                **RunSpec.from_environment().result_fields(),
            ).marker())
            continue
        for c in r["corners"]:
            if "error" in c:
                print(f"{r['tech']:8s} | {c['nfin']:5d} | "
                      f"ERROR — {c['error'][:46]}")
                continue
            corner_ok = c["positive"] and c["nrmse_ok"]
            print(f"{r['tech']:8s} | {c['nfin']:5d} | "
                  f"{c['ng_snm']*1e3:10.1f} | {c['dn_snm']*1e3:10.1f} | "
                  f"{c['snm_err_pct']:8.1f} | {c['dn_min_qb']*1e3:10.1f} | "
                  f"{c['nrmse_pct']:7.2f} | "
                  f"{'PASS' if corner_ok else 'FAIL':>5s}")
        fic = r["force_ic"]
        # Keep the legacy-readable diagnostic tokens beside the structured
        # GateResult marker. ``all-positive`` encodes the full gate (positive
        # plus NGSPICE-NRMSE tracking), while ``force_ic`` is diagnostic only.
        print(f"{r['tech']:8s} |   force_ic: state1="
              f"{'ok' if fic.get('state1') else 'FAIL'} "
              f"state0={'ok' if fic.get('state0') else 'FAIL'} (diag)  "
              f"|  GATE: {'PASS' if r['all_positive'] else 'FAIL'}  "
              f"(all-positive: {'yes' if r['all_positive'] else 'NO'})")
        n_pass += int(r["all_positive"])
        comparable = [c for c in r["corners"] if "error" not in c]
        corner_errors = [c for c in r["corners"] if "error" in c]
        marker_status = (
            "error" if corner_errors else
            ("pass" if r["all_positive"] else "fail")
        )
        print(GateResult(
            case_id="sram_snm", tech=r["tech"], corner="nfin_sweep",
            analysis="read_snm", role="qualification", status=marker_status,
            metrics={
                "metric": max(
                    (c["nrmse_pct"] for c in comparable),
                    default=float("nan"),
                ),
                "worst_nrmse_pct": max(
                    (c["nrmse_pct"] for c in comparable),
                    default=float("nan"),
                ),
                "worst_mre_pct": max(
                    (c["mre_pct"] for c in comparable),
                    default=float("nan"),
                ),
                "min_r2": min(
                    (c["r2"] for c in comparable),
                    default=float("nan"),
                ),
                "max_err": max(
                    (c["max_err"] for c in comparable),
                    default=float("nan"),
                ),
            },
            domain={
                "nfins": [c["nfin"] for c in r["corners"]],
                "force_ic": r["force_ic"],
            },
            error="; ".join(c["error"] for c in corner_errors),
            reference_converged=not corner_errors,
            candidate_converged=not corner_errors,
            **RunSpec.from_environment().result_fields(),
        ).marker())
    n_total = len(results)
    print(f"\n  {n_pass}/{n_total} techs pass (positive + NGSPICE-NRMSE-tracking "
          "across NFIN corners)")
    # B10: reflect the verdict in the exit code (consumers also parse stdout).
    # empty results (all techs skipped) must not exit green
    return 0 if (n_total > 0 and n_pass == n_total) else 1


if __name__ == "__main__":
    sys.exit(main())
